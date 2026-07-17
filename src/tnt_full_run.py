"""
TNT Full Run — increase_complexity-03
=======================================
FULL confirmatory run (decisive experiment for the paper):
  - softmax-inner vs linear-inner (ReLU-kernel, Performer-style)
  - n_inner=4, d_inner=24 — NOTE: n_inner < d_inner → linear is 6× MORE expensive;
    study is purely expressivity (no efficiency/FLOP claim)
  - BOTH CIFAR-10 and CIFAR-100
  - Seeds {0, 1, 2} (3 per arm per dataset)
  - 100 epochs with cosine annealing to 0 (78× more than probe)
  - AMP (float16) for speed

Outputs per-arm mean±std, delta with 95% CI, Welch's t-test.
"""

import sys, os, math, time, json, warnings
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, "/workspace/CV-Backbones/tnt_pytorch")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from timm.models.layers import DropPath, trunc_normal_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry check (documented before any training run)
# ─────────────────────────────────────────────────────────────────────────────
N_INNER = 4
D_INNER = 24
SOFTMAX_COST = N_INNER ** 2           # 16  (∝ n²)
LINEAR_COST  = N_INNER * D_INNER      # 96  (∝ n·d)
COST_RATIO   = LINEAR_COST / SOFTMAX_COST  # 6.0

print("\n" + "=" * 70)
print("GEOMETRY TABLE (pre-training gate, from study spec §H5)")
print("=" * 70)
print(f"  Image size       : 32×32 px")
print(f"  Outer patch size : 8×8 px → 16 outer tokens")
print(f"  Sub-patch size   : 4×4 px (inner_stride=4)")
print(f"  n_inner          : {N_INNER} sub-patch tokens per outer patch")
print(f"  d_inner          : {D_INNER} (inner attention head dim = 24/4 heads = 6)")
print(f"  Softmax attn cost (∝ n²)  : {SOFTMAX_COST}")
print(f"  Linear-attn cost (∝ n·d) : {LINEAR_COST}")
print(f"  Linear/Softmax ratio      : {COST_RATIO:.1f}× MORE EXPENSIVE")
print(f"  → Efficiency claim REFUTED: linear-inner is not cheaper at this scale")
print(f"  → Study is EXPRESSIVITY-ONLY — does inner-block attention type matter?")
print("=" * 70 + "\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture modules (identical to prior rounds)
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_f, hidden_f=None, out_f=None, drop=0.):
        super().__init__()
        out_f    = out_f or in_f
        hidden_f = hidden_f or in_f
        self.fc1  = nn.Linear(in_f, hidden_f)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_f, out_f)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SoftmaxAttention(nn.Module):
    """Standard softmax self-attention (used in outer block always; inner when arm='softmax')."""
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qk  = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v   = nn.Linear(dim, dim,     bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class LinearAttention(nn.Module):
    """ReLU-kernel linear attention (Performer-style) — inner block only.

    φ(x) = ReLU(x) + ε
    Attention(Q,K,V) = D⁻¹ · (φ(Q) @ (φ(K)ᵀ @ V))
    Cost: O(n·d²) vs O(n²·d) for softmax; at n=4 < d=24, LINEAR IS 6× MORE EXPENSIVE.
    """
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.eps       = eps
        self.qk  = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v   = nn.Linear(dim, dim,     bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # B, H, N, head_dim
        v    = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = torch.nn.functional.relu(q) + self.eps
        k = torch.nn.functional.relu(k) + self.eps

        kv    = torch.einsum("bhnd,bhnm->bhdm", k, v)    # B, H, d, d
        qkv   = torch.einsum("bhnd,bhdm->bhnm", q, kv)   # B, H, N, d

        k_sum = k.sum(dim=-2, keepdim=True)               # B, H, 1, d
        denom = (q * k_sum).sum(dim=-1, keepdim=True)     # B, H, N, 1
        denom = denom.clamp(min=self.eps)

        out = (qkv / denom).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TNTBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads,
                 num_words, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., inner_attn_type="softmax"):
        super().__init__()
        self.inner_norm1 = nn.LayerNorm(inner_dim)
        if inner_attn_type == "linear":
            self.inner_attn = LinearAttention(inner_dim, inner_num_heads,
                                               qkv_bias=qkv_bias, attn_drop=attn_drop,
                                               proj_drop=drop)
        else:
            self.inner_attn = SoftmaxAttention(inner_dim, inner_num_heads,
                                                qkv_bias=qkv_bias, attn_drop=attn_drop,
                                                proj_drop=drop)
        self.inner_norm2 = nn.LayerNorm(inner_dim)
        self.inner_mlp   = Mlp(inner_dim, int(inner_dim * mlp_ratio), drop=drop)

        self.proj_norm1  = nn.LayerNorm(num_words * inner_dim)
        self.proj        = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2  = nn.LayerNorm(outer_dim)

        self.outer_norm1 = nn.LayerNorm(outer_dim)
        self.outer_attn  = SoftmaxAttention(outer_dim, outer_num_heads, qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop)
        self.drop_path   = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = nn.LayerNorm(outer_dim)
        self.outer_mlp   = Mlp(outer_dim, int(outer_dim * mlp_ratio), drop=drop)

    def forward(self, inner_tokens, outer_tokens):
        inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))
        inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))
        B, N, C = outer_tokens.shape
        outer_tokens[:, 1:] = (outer_tokens[:, 1:]
            + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1)))))
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
        outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, inner_dim=24, inner_stride=4):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_words   = (math.ceil(patch_size / inner_stride)) ** 2
        self.inner_dim   = inner_dim
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj   = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, self.patch_size, self.patch_size)
        x = self.proj(x)
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)
        return x


class TNTSmall(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, num_classes=10,
                 outer_dim=192, inner_dim=24, depth=6,
                 outer_num_heads=3, inner_num_heads=4,
                 mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 inner_stride=4, inner_attn_type="softmax"):
        super().__init__()
        self.num_classes = num_classes
        self.outer_dim   = outer_dim

        self.patch_embed  = PatchEmbed(img_size, patch_size, in_chans, inner_dim, inner_stride)
        num_patches = self.patch_embed.num_patches
        num_words   = self.patch_embed.num_words

        self.proj_norm1 = nn.LayerNorm(num_words * inner_dim)
        self.proj       = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = nn.LayerNorm(outer_dim)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_pos  = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        self.inner_pos  = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop   = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TNTBlock(outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i],
                     inner_attn_type=inner_attn_type)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(outer_dim)
        self.head = nn.Linear(outer_dim, num_classes)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(
            inner_tokens.reshape(B, self.patch_embed.num_patches, -1))))
        outer_tokens = torch.cat([self.cls_token.expand(B, -1, -1), outer_tokens], dim=1)
        outer_tokens = self.pos_drop(outer_tokens + self.outer_pos)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        x = self.norm(outer_tokens)[:, 0]
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HFDataset(torch.utils.data.Dataset):
    """Thin wrapper over a HuggingFace split with a torchvision transform."""
    def __init__(self, hf_split, label_key: str, transform=None):
        self.ds        = hf_split
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        img   = item["img"]
        label = item[self.label_key]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(dataset_name: str, batch_size: int = 128):
    """Returns (train_loader, test_loader) for CIFAR-10 or CIFAR-100."""
    from datasets import load_dataset

    if dataset_name == "cifar10":
        mean, std   = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        label_key   = "label"
        hf_name     = "uoft-cs/cifar10"
        num_classes = 10
    else:  # cifar100
        mean, std   = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        label_key   = "fine_label"
        hf_name     = "uoft-cs/cifar100"
        num_classes = 100

    norm = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    print(f"Loading {dataset_name} from HuggingFace...", flush=True)
    hf_ds = load_dataset(hf_name)

    train_ds = HFDataset(hf_ds["train"], label_key=label_key, transform=train_tf)
    test_ds  = HFDataset(hf_ds["test"],  label_key=label_key, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────

def make_model(seed: int, inner_attn_type: str, num_classes: int):
    set_seed(seed)
    return TNTSmall(
        img_size=32, patch_size=8, num_classes=num_classes,
        outer_dim=192, inner_dim=24, depth=6,
        outer_num_heads=3, inner_num_heads=4,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        inner_stride=4,
        inner_attn_type=inner_attn_type,
    ).to(DEVICE)


def get_cosine_lr(optimizer, epoch: int, max_epochs: int, warmup_epochs: int, base_lr: float):
    """Linear warmup + cosine annealing to 0."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100. * correct / total


def train_full(
    train_loader, test_loader,
    inner_attn_type: str,
    num_classes: int,
    seed: int,
    max_epochs: int = 100,
    base_lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    run_label: str = "",
):
    """Train TNT for max_epochs with cosine LR schedule. Return (best_acc, final_acc)."""
    tag = f"[{run_label}]"
    print(f"\n{tag} Starting training (inner={inner_attn_type}, "
          f"classes={num_classes}, seed={seed}, epochs={max_epochs})", flush=True)

    model   = make_model(seed, inner_attn_type, num_classes)
    opt     = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    crit    = nn.CrossEntropyLoss()
    scaler  = GradScaler()  # AMP

    best_acc  = 0.
    run_start = time.time()

    for epoch in range(max_epochs):
        lr = get_cosine_lr(opt, epoch, max_epochs, warmup_epochs, base_lr)
        model.train()
        epoch_loss = 0.
        n_batches  = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate every 10 epochs and at last epoch
        if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
            acc = evaluate(model, test_loader)
            best_acc = max(best_acc, acc)
            elapsed  = time.time() - run_start
            print(f"{tag} Epoch {epoch+1:3d}/{max_epochs} | lr={lr:.2e} | "
                  f"loss={avg_loss:.4f} | test_acc={acc:.2f}% | "
                  f"best={best_acc:.2f}% | elapsed={elapsed:.0f}s", flush=True)

    final_acc = evaluate(model, test_loader)
    print(f"{tag} DONE. final_acc={final_acc:.2f}% best_acc={best_acc:.2f}%", flush=True)
    return best_acc, final_acc


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistics
# ─────────────────────────────────────────────────────────────────────────────

def welch_ttest_95ci(a: list, b: list):
    """Two-sided Welch's t-test and 95% CI for difference (b - a)."""
    a, b   = np.array(a), np.array(b)
    t_stat, p_val = stats.ttest_ind(b, a, equal_var=False)

    # Confidence interval for the difference in means
    na, nb   = len(a), len(b)
    mean_diff = b.mean() - a.mean()
    se_a = a.std(ddof=1) / math.sqrt(na)
    se_b = b.std(ddof=1) / math.sqrt(nb)
    se_diff = math.sqrt(se_a**2 + se_b**2)

    # Welch–Satterthwaite degrees of freedom
    df = (se_a**2 + se_b**2)**2 / (
        (se_a**2)**2 / (na - 1) + (se_b**2)**2 / (nb - 1)
    )
    t_crit = stats.t.ppf(0.975, df=df)
    ci_lo  = mean_diff - t_crit * se_diff
    ci_hi  = mean_diff + t_crit * se_diff

    # Cohen's d (pooled)
    pooled_std = math.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    cohen_d    = mean_diff / pooled_std if pooled_std > 0 else float("nan")

    # Minimum detectable effect at 80% power (two-sided α=0.05)
    # MDE = t_crit_alpha * se_diff (approximate)
    mde = t_crit * se_diff * math.sqrt(2)  # rough estimate

    return {
        "mean_a":    float(np.round(a.mean(), 4)),
        "std_a":     float(np.round(a.std(ddof=1), 4)),
        "mean_b":    float(np.round(b.mean(), 4)),
        "std_b":     float(np.round(b.std(ddof=1), 4)),
        "delta_b_minus_a": float(np.round(mean_diff, 4)),
        "ci_95_lo":  float(np.round(ci_lo, 4)),
        "ci_95_hi":  float(np.round(ci_hi, 4)),
        "t_stat":    float(np.round(t_stat, 4)),
        "p_value":   float(np.round(p_val, 6)),
        "df":        float(np.round(df, 2)),
        "cohen_d":   float(np.round(cohen_d, 4)),
        "mde_80pct_approx": float(np.round(mde, 4)),
        "significant_at_alpha_0p05": bool(p_val < 0.05),
        "null_indistinguishable": bool(p_val >= 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("TNT Full Run — increase_complexity-03")
    print("softmax-inner vs linear-inner | CIFAR-10 + CIFAR-100 | 3 seeds")
    print("=" * 70, flush=True)

    SEEDS       = [0, 1, 2]
    MAX_EPOCHS  = 100
    BATCH_SIZE  = 128
    BASE_LR     = 1e-3
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 10
    DATASETS    = ["cifar10", "cifar100"]
    ARMS        = ["softmax", "linear"]

    # Storage: accs[dataset][arm] = list of per-seed final accuracies
    accs = {ds: {arm: [] for arm in ARMS} for ds in DATASETS}

    run_start_total = time.time()
    run_count = 0
    total_runs = len(DATASETS) * len(ARMS) * len(SEEDS)

    for ds_name in DATASETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f"DATASET: {ds_name.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)

        train_loader, test_loader, num_classes = get_loaders(ds_name, BATCH_SIZE)

        for arm in ARMS:
            for seed in SEEDS:
                run_count += 1
                label = f"{ds_name}/{arm}/seed{seed} ({run_count}/{total_runs})"
                best_acc, final_acc = train_full(
                    train_loader, test_loader,
                    inner_attn_type=arm,
                    num_classes=num_classes,
                    seed=seed,
                    max_epochs=MAX_EPOCHS,
                    base_lr=BASE_LR,
                    weight_decay=WEIGHT_DECAY,
                    warmup_epochs=WARMUP_EPOCHS,
                    run_label=label,
                )
                accs[ds_name][arm].append(final_acc)
                print(f"\n>>> Completed {label}: final={final_acc:.2f}%  best={best_acc:.2f}%",
                      flush=True)

    total_elapsed = time.time() - run_start_total
    print(f"\nAll {total_runs} runs done in {total_elapsed/3600:.2f}h", flush=True)

    # ── Statistical analysis ─────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70, flush=True)

    stats_per_dataset = {}
    for ds_name in DATASETS:
        softmax_accs = accs[ds_name]["softmax"]
        linear_accs  = accs[ds_name]["linear"]

        stat = welch_ttest_95ci(softmax_accs, linear_accs)  # delta = linear - softmax
        stats_per_dataset[ds_name] = stat

        print(f"\n── {ds_name.upper()} ──────────────────────────────────────────────────")
        print(f"  softmax-inner : {stat['mean_a']:.2f}% ± {stat['std_a']:.2f}pp "
              f"  (seeds: {[round(v,2) for v in softmax_accs]})")
        print(f"  linear-inner  : {stat['mean_b']:.2f}% ± {stat['std_b']:.2f}pp "
              f"  (seeds: {[round(v,2) for v in linear_accs]})")
        print(f"  delta (linear−softmax): {stat['delta_b_minus_a']:+.4f}pp")
        print(f"  95% CI: [{stat['ci_95_lo']:+.4f}, {stat['ci_95_hi']:+.4f}]pp")
        print(f"  t={stat['t_stat']:.4f}, df={stat['df']:.1f}, p={stat['p_value']:.6f}")
        print(f"  Cohen's d = {stat['cohen_d']:.4f}")
        print(f"  MDE@80% power ≈ {stat['mde_80pct_approx']:.4f}pp")
        null_str = ("INDISTINGUISHABLE FROM 0 (null not rejected)"
                    if stat["null_indistinguishable"]
                    else "SIGNIFICANTLY DIFFERENT FROM 0 (null rejected)")
        print(f"  → Delta is {null_str}", flush=True)

    # ── CIFAR-100 vs CIFAR-10 gap comparison ─────────────────────────────────
    delta_c10  = stats_per_dataset["cifar10"]["delta_b_minus_a"]
    delta_c100 = stats_per_dataset["cifar100"]["delta_b_minus_a"]
    delta_change = delta_c100 - delta_c10  # >0 = linear does MORE relative better on CIFAR-100

    print("\n── Cross-dataset delta comparison ──────────────────────────────────────")
    print(f"  CIFAR-10  delta (linear−softmax) : {delta_c10:+.4f}pp")
    print(f"  CIFAR-100 delta (linear−softmax) : {delta_c100:+.4f}pp")
    print(f"  Change (C100−C10)                : {delta_change:+.4f}pp")
    if delta_change < -0.5:
        prediction_str = "CONFIRMED — linear hurts MORE on CIFAR-100 (> 0.5pp additional degradation)"
    elif delta_change < 0:
        prediction_str = "BORDERLINE — linear slightly worse on CIFAR-100 but < 0.5pp additional"
    else:
        prediction_str = ("NOT CONFIRMED — CIFAR-100 delta not more negative than CIFAR-10 delta; "
                          "publishable negative: inner-block attention type is dataset-invariant "
                          "at n_inner=4")
    print(f"  Pre-registered prediction: {prediction_str}", flush=True)

    # ── Geometry restated ──────────────────────────────────────────────────────
    print("\n── Geometry (6× per-head cost, efficiency claim REFUTED) ──────────────")
    print(f"  n_inner={N_INNER}, d_inner={D_INNER}")
    print(f"  Softmax cost ∝ n² = {SOFTMAX_COST}")
    print(f"  Linear-attn cost ∝ n·d = {LINEAR_COST}")
    print(f"  Ratio: {COST_RATIO:.1f}× — linear-inner is MORE expensive, not cheaper")
    print(f"  → No efficiency claim. Study is purely about expressivity.", flush=True)

    # ── Build RESULTS.json ────────────────────────────────────────────────────
    results = {
        "status": "SUCCESS",
        "scale": "full",
        "metrics": {
            "cifar10": {
                "softmax_inner": {
                    "mean_acc_pct":  stats_per_dataset["cifar10"]["mean_a"],
                    "std_acc_pct":   stats_per_dataset["cifar10"]["std_a"],
                    "per_seed":      [round(v, 2) for v in accs["cifar10"]["softmax"]],
                },
                "linear_inner": {
                    "mean_acc_pct":  stats_per_dataset["cifar10"]["mean_b"],
                    "std_acc_pct":   stats_per_dataset["cifar10"]["std_b"],
                    "per_seed":      [round(v, 2) for v in accs["cifar10"]["linear"]],
                },
                "delta_linear_minus_softmax_pp": delta_c10,
                "ci_95": [stats_per_dataset["cifar10"]["ci_95_lo"],
                          stats_per_dataset["cifar10"]["ci_95_hi"]],
                "p_value": stats_per_dataset["cifar10"]["p_value"],
                "null_indistinguishable": stats_per_dataset["cifar10"]["null_indistinguishable"],
                "cohen_d": stats_per_dataset["cifar10"]["cohen_d"],
                "mde_80pct_pp": stats_per_dataset["cifar10"]["mde_80pct_approx"],
            },
            "cifar100": {
                "softmax_inner": {
                    "mean_acc_pct":  stats_per_dataset["cifar100"]["mean_a"],
                    "std_acc_pct":   stats_per_dataset["cifar100"]["std_a"],
                    "per_seed":      [round(v, 2) for v in accs["cifar100"]["softmax"]],
                },
                "linear_inner": {
                    "mean_acc_pct":  stats_per_dataset["cifar100"]["mean_b"],
                    "std_acc_pct":   stats_per_dataset["cifar100"]["std_b"],
                    "per_seed":      [round(v, 2) for v in accs["cifar100"]["linear"]],
                },
                "delta_linear_minus_softmax_pp": delta_c100,
                "ci_95": [stats_per_dataset["cifar100"]["ci_95_lo"],
                          stats_per_dataset["cifar100"]["ci_95_hi"]],
                "p_value": stats_per_dataset["cifar100"]["p_value"],
                "null_indistinguishable": stats_per_dataset["cifar100"]["null_indistinguishable"],
                "cohen_d": stats_per_dataset["cifar100"]["cohen_d"],
                "mde_80pct_pp": stats_per_dataset["cifar100"]["mde_80pct_approx"],
            },
            "cross_dataset": {
                "delta_c10_pp":  round(delta_c10, 4),
                "delta_c100_pp": round(delta_c100, 4),
                "delta_change_c100_minus_c10_pp": round(delta_change, 4),
                "prediction_result": prediction_str,
            },
            "geometry": {
                "n_inner": N_INNER,
                "d_inner": D_INNER,
                "softmax_cost_prop_n_sq":     SOFTMAX_COST,
                "linear_cost_prop_n_times_d": LINEAR_COST,
                "linear_vs_softmax_ratio":    COST_RATIO,
                "note": (f"n_inner={N_INNER} < d_inner={D_INNER}: linear-attn is {COST_RATIO:.0f}× "
                         "MORE expensive per head than softmax at this scale. "
                         "Study is expressivity-only; no efficiency claim."),
            },
            "training_config": {
                "max_epochs":     MAX_EPOCHS,
                "batch_size":     BATCH_SIZE,
                "base_lr":        BASE_LR,
                "weight_decay":   WEIGHT_DECAY,
                "warmup_epochs":  WARMUP_EPOCHS,
                "lr_schedule":    "linear warmup + cosine annealing to 0",
                "amp":            True,
                "seeds":          SEEDS,
                "probe_steps":    500,
                "full_steps_per_epoch": 391,
                "full_total_steps_approx": MAX_EPOCHS * 391,
            },
            "total_wall_clock_hours": round(total_elapsed / 3600, 2),
        },
        "subject_executed": (
            "TNT (Transformer-in-Transformer) on CIFAR-10 and CIFAR-100. "
            "Two arms: softmax inner attention (baseline) vs ReLU-kernel linear inner attention. "
            "Outer block: softmax in both arms (unchanged). "
            "Config: img_size=32, patch_size=8, inner_stride=4, n_inner=4, d_inner=24, "
            "outer_dim=192, depth=6, outer_num_heads=3, inner_num_heads=4. "
            f"Training: {MAX_EPOCHS} epochs, AdamW (lr={BASE_LR}, wd={WEIGHT_DECAY}), "
            f"cosine annealing, AMP. Seeds: {SEEDS}. "
            "Statistical test: two-sided Welch's t-test (α=0.05)."
        ),
        "notes": (
            f"Geometry: n_inner=4 < d_inner=24 → linear-attn {COST_RATIO:.0f}× MORE expensive. "
            "Study is purely expressivity. "
            f"CIFAR-10 delta: {delta_c10:+.4f}pp; "
            f"CIFAR-100 delta: {delta_c100:+.4f}pp; "
            f"Cross-dataset change: {delta_change:+.4f}pp. "
            f"Prediction: {prediction_str}. "
            f"Total runtime: {total_elapsed/3600:.2f}h."
        ),
    }

    out_path = "/workspace/results/increase_complexity-03/RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))
