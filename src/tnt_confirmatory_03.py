"""
TNT Confirmatory Run — increase_complexity-03
==============================================
EXPERIMENT.md spec (full scale):
  - Reuse committed TNT code (architecture from tnt_full_run.py)
  - softmax-inner vs linear-inner (n_inner=4, d_inner=24)
  - BOTH CIFAR-10 and CIFAR-100
  - Seeds {0, 1, 2}  →  3 per arm per dataset
  - SAME ~500-step budget as the probe  (NOT to convergence)
  - Goal: SEED VARIANCE for confidence intervals, not high accuracy
  - Total: 2 datasets × 2 arms × 3 seeds = 12 runs, each ~2-3 min
  - MUST finish < 40 min total

Outputs:
  - Per (dataset, arm): mean±std top-1 over 3 seeds
  - Delta (linear − softmax) with 95% CI (Welch's t-test)
  - delta_within_ci_of_zero per dataset
  - 6× per-head cost restated (expressivity-only study)
  - results/increase_complexity-03/RESULTS.json
"""

import sys, os, math, time, json, warnings
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.insert(0, "/workspace/CV-Backbones/tnt_pytorch")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from timm.models.layers import DropPath, trunc_normal_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry table — must be printed BEFORE any training
# ─────────────────────────────────────────────────────────────────────────────
N_INNER      = 4
D_INNER      = 24
SOFTMAX_COST = N_INNER ** 2          # 16  (∝ n²)
LINEAR_COST  = N_INNER * D_INNER     # 96  (∝ n·d)
COST_RATIO   = LINEAR_COST / SOFTMAX_COST  # 6.0

print("\n" + "=" * 70)
print("GEOMETRY TABLE (pre-training gate)")
print("=" * 70)
print(f"  Image size       : 32×32 px")
print(f"  Outer patch size : 8×8 px → 16 outer tokens")
print(f"  Sub-patch size   : 4×4 px (inner_stride=4)")
print(f"  n_inner          : {N_INNER} sub-patch tokens per outer patch")
print(f"  d_inner          : {D_INNER}")
print(f"  Softmax cost (∝ n²)      : {SOFTMAX_COST}")
print(f"  Linear-attn cost (∝ n·d): {LINEAR_COST}")
print(f"  Linear/Softmax ratio     : {COST_RATIO:.1f}× MORE EXPENSIVE")
print(f"  n_inner ({N_INNER}) < d_inner ({D_INNER})  →  linear kernel has NO FLOP advantage")
print(f"  Study is EXPRESSIVITY-ONLY (no efficiency/FLOP claim)")
print("=" * 70 + "\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Architecture — identical to tnt_full_run.py (committed code reused)
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
        v    = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class LinearAttention(nn.Module):
    """ReLU-kernel linear attention (Performer-style) — inner block only.
    φ(x) = ReLU(x) + ε
    Cost ∝ n·d² vs softmax ∝ n²·d; at n=4 < d=24, linear is 6× MORE expensive.
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
        q, k = qk[0], qk[1]
        v    = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = torch.nn.functional.relu(q) + self.eps
        k = torch.nn.functional.relu(k) + self.eps
        kv    = torch.einsum("bhnd,bhnm->bhdm", k, v)
        qkv   = torch.einsum("bhnd,bhdm->bhnm", q, kv)
        k_sum = k.sum(dim=-2, keepdim=True)
        denom = (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=self.eps)
        out   = (qkv / denom).transpose(1, 2).reshape(B, N, C)
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
# 3. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class HFDataset(torch.utils.data.Dataset):
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
    from datasets import load_dataset

    if dataset_name == "cifar10":
        mean, std   = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        label_key   = "label"
        hf_name     = "uoft-cs/cifar10"
        num_classes = 10
    else:
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


def train_500steps(
    train_loader, test_loader,
    inner_attn_type: str,
    num_classes: int,
    seed: int,
    n_steps: int = 500,
    base_lr: float = 1e-3,
    weight_decay: float = 0.05,
    run_label: str = "",
):
    """Train TNT for exactly n_steps gradient steps (same budget as probe).
    Returns final test accuracy.
    """
    tag = f"[{run_label}]"
    print(f"\n{tag} Training {n_steps} steps "
          f"(inner={inner_attn_type}, classes={num_classes}, seed={seed})", flush=True)

    model = make_model(seed, inner_attn_type, num_classes)
    opt   = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    crit  = nn.CrossEntropyLoss()

    t0 = time.time()
    step = 0
    train_iter = iter(train_loader)

    model.train()
    while step < n_steps:
        try:
            imgs, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, labels = next(train_iter)

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(imgs), labels)
        loss.backward()
        opt.step()
        step += 1

        if step % 100 == 0:
            print(f"  {tag} step {step:4d}/{n_steps}  loss={loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    acc = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"  {tag} DONE  test_acc={acc:.2f}%  elapsed={elapsed:.1f}s", flush=True)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistics
# ─────────────────────────────────────────────────────────────────────────────

def welch_ci(a, b):
    """Two-sided Welch's t-test and 95% CI for difference (b − a)."""
    a, b  = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    mean_diff = float(b.mean() - a.mean())
    se_a  = float(a.std(ddof=1)) / math.sqrt(na)
    se_b  = float(b.std(ddof=1)) / math.sqrt(nb)
    se_diff = math.sqrt(se_a**2 + se_b**2)
    df = (se_a**2 + se_b**2)**2 / (
         (se_a**2)**2 / max(na - 1, 1) + (se_b**2)**2 / max(nb - 1, 1))
    t_crit = float(stats.t.ppf(0.975, df=df)) if se_diff > 0 else 0.
    t_stat, p_val = stats.ttest_ind(b, a, equal_var=False)
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff
    # delta_within_ci_of_zero = CI contains 0
    ci_contains_zero = ci_lo <= 0 <= ci_hi
    return {
        "mean_softmax":  round(float(a.mean()), 4),
        "std_softmax":   round(float(a.std(ddof=1)), 4),
        "mean_linear":   round(float(b.mean()), 4),
        "std_linear":    round(float(b.std(ddof=1)), 4),
        "delta_linear_minus_softmax_pp": round(mean_diff, 4),
        "ci_95_lo":  round(ci_lo, 4),
        "ci_95_hi":  round(ci_hi, 4),
        "t_stat":    round(float(t_stat), 4),
        "p_value":   round(float(p_val), 6),
        "df":        round(df, 2),
        "delta_within_ci_of_zero": bool(ci_contains_zero),
        "significant_p05": bool(float(p_val) < 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_STEPS   = 500          # same budget as probe — goal is SEED VARIANCE for CIs
    SEEDS     = [0, 1, 2]
    BATCH_SIZE = 128
    BASE_LR   = 1e-3
    WEIGHT_DECAY = 0.05
    DATASETS  = ["cifar10", "cifar100"]
    ARMS      = ["softmax", "linear"]

    print("=" * 70)
    print("TNT Confirmatory Run — increase_complexity-03")
    print(f"softmax-inner vs linear-inner | CIFAR-10 + CIFAR-100 | 3 seeds × {N_STEPS} steps")
    print("=" * 70, flush=True)

    accs = {ds: {arm: [] for arm in ARMS} for ds in DATASETS}
    run_count = 0
    total_runs = len(DATASETS) * len(ARMS) * len(SEEDS)
    wall_start = time.time()

    for ds_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'=' * 70}", flush=True)

        train_loader, test_loader, num_classes = get_loaders(ds_name, BATCH_SIZE)

        for arm in ARMS:
            for seed in SEEDS:
                run_count += 1
                label = f"{ds_name}/{arm}/seed{seed} ({run_count}/{total_runs})"
                acc = train_500steps(
                    train_loader, test_loader,
                    inner_attn_type=arm,
                    num_classes=num_classes,
                    seed=seed,
                    n_steps=N_STEPS,
                    base_lr=BASE_LR,
                    weight_decay=WEIGHT_DECAY,
                    run_label=label,
                )
                accs[ds_name][arm].append(acc)
                print(f">>> {label}: test_acc={acc:.2f}%", flush=True)

    total_elapsed = time.time() - wall_start
    print(f"\nAll {total_runs} runs done in {total_elapsed/60:.1f} min", flush=True)

    # ── Statistical analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70, flush=True)

    stat_results = {}
    for ds_name in DATASETS:
        softmax_accs = accs[ds_name]["softmax"]
        linear_accs  = accs[ds_name]["linear"]
        stat = welch_ci(softmax_accs, linear_accs)
        stat_results[ds_name] = stat
        print(f"\n── {ds_name.upper()} ──")
        print(f"  softmax-inner: {stat['mean_softmax']:.2f}% ± {stat['std_softmax']:.2f}pp "
              f"  seeds={[round(v,2) for v in softmax_accs]}")
        print(f"  linear-inner : {stat['mean_linear']:.2f}% ± {stat['std_linear']:.2f}pp "
              f"  seeds={[round(v,2) for v in linear_accs]}")
        print(f"  delta (linear−softmax): {stat['delta_linear_minus_softmax_pp']:+.4f}pp")
        print(f"  95% CI: [{stat['ci_95_lo']:+.4f}, {stat['ci_95_hi']:+.4f}]pp")
        print(f"  Welch t={stat['t_stat']:.4f}, df={stat['df']:.1f}, p={stat['p_value']:.4f}")
        z_str = "YES (CI contains 0 → null not rejected)" if stat["delta_within_ci_of_zero"] \
                else "NO (CI excludes 0 → statistically significant)"
        print(f"  delta_within_ci_of_zero: {z_str}", flush=True)

    # Pre-registered directional prediction
    delta_c10  = stat_results["cifar10"]["delta_linear_minus_softmax_pp"]
    delta_c100 = stat_results["cifar100"]["delta_linear_minus_softmax_pp"]
    delta_change = delta_c100 - delta_c10

    print("\n── Cross-dataset comparison ─────────────────────────────────────────")
    print(f"  CIFAR-10  delta (linear−softmax): {delta_c10:+.4f}pp")
    print(f"  CIFAR-100 delta (linear−softmax): {delta_c100:+.4f}pp")
    print(f"  Change (C100−C10):                {delta_change:+.4f}pp")

    if delta_change < -0.5:
        pred_result = ("CONFIRMED — linear inner hurts more on CIFAR-100 "
                       "(>0.5pp additional degradation)")
    elif delta_change < 0:
        pred_result = ("BORDERLINE — linear slightly worse on CIFAR-100, "
                       "but delta_change < 0.5pp threshold")
    else:
        pred_result = ("NOT CONFIRMED — CIFAR-100 delta not more negative than CIFAR-10; "
                       "publishable negative: inner-block attention type dataset-invariant "
                       "at n_inner=4")
    print(f"  Pre-registered prediction: {pred_result}", flush=True)

    print("\n── Geometry restated ────────────────────────────────────────────────")
    print(f"  n_inner={N_INNER}, d_inner={D_INNER}")
    print(f"  Softmax cost ∝ n²    = {SOFTMAX_COST}")
    print(f"  Linear cost  ∝ n·d   = {LINEAR_COST}")
    print(f"  Ratio: {COST_RATIO:.0f}× — linear inner MORE expensive, NOT cheaper")
    print(f"  No efficiency claim; study is purely expressivity.", flush=True)

    # ── RESULTS.json ──────────────────────────────────────────────────────────
    results = {
        "status": "SUCCESS",
        "scale": "full",
        "metrics": {
            "cifar10": {
                "softmax_inner": {
                    "mean_acc_pct": stat_results["cifar10"]["mean_softmax"],
                    "std_acc_pct":  stat_results["cifar10"]["std_softmax"],
                    "per_seed_acc": [round(v, 2) for v in accs["cifar10"]["softmax"]],
                },
                "linear_inner": {
                    "mean_acc_pct": stat_results["cifar10"]["mean_linear"],
                    "std_acc_pct":  stat_results["cifar10"]["std_linear"],
                    "per_seed_acc": [round(v, 2) for v in accs["cifar10"]["linear"]],
                },
                "delta_linear_minus_softmax_pp": stat_results["cifar10"]["delta_linear_minus_softmax_pp"],
                "ci_95": [stat_results["cifar10"]["ci_95_lo"], stat_results["cifar10"]["ci_95_hi"]],
                "p_value": stat_results["cifar10"]["p_value"],
                "delta_within_ci_of_zero": stat_results["cifar10"]["delta_within_ci_of_zero"],
                "welch_t": stat_results["cifar10"]["t_stat"],
                "welch_df": stat_results["cifar10"]["df"],
            },
            "cifar100": {
                "softmax_inner": {
                    "mean_acc_pct": stat_results["cifar100"]["mean_softmax"],
                    "std_acc_pct":  stat_results["cifar100"]["std_softmax"],
                    "per_seed_acc": [round(v, 2) for v in accs["cifar100"]["softmax"]],
                },
                "linear_inner": {
                    "mean_acc_pct": stat_results["cifar100"]["mean_linear"],
                    "std_acc_pct":  stat_results["cifar100"]["std_linear"],
                    "per_seed_acc": [round(v, 2) for v in accs["cifar100"]["linear"]],
                },
                "delta_linear_minus_softmax_pp": stat_results["cifar100"]["delta_linear_minus_softmax_pp"],
                "ci_95": [stat_results["cifar100"]["ci_95_lo"], stat_results["cifar100"]["ci_95_hi"]],
                "p_value": stat_results["cifar100"]["p_value"],
                "delta_within_ci_of_zero": stat_results["cifar100"]["delta_within_ci_of_zero"],
                "welch_t": stat_results["cifar100"]["t_stat"],
                "welch_df": stat_results["cifar100"]["df"],
            },
            "cross_dataset": {
                "delta_c10_pp":  round(delta_c10, 4),
                "delta_c100_pp": round(delta_c100, 4),
                "delta_change_c100_minus_c10_pp": round(delta_change, 4),
                "prediction_result": pred_result,
            },
            "geometry": {
                "n_inner": N_INNER,
                "d_inner": D_INNER,
                "softmax_cost_n_sq":     SOFTMAX_COST,
                "linear_cost_n_times_d": LINEAR_COST,
                "linear_vs_softmax_ratio": COST_RATIO,
                "note": (f"n_inner={N_INNER} < d_inner={D_INNER}: "
                         f"linear-attn {COST_RATIO:.0f}× MORE expensive per head than softmax. "
                         "Study is expressivity-only; no efficiency claim."),
            },
            "training_config": {
                "n_steps_per_run":   N_STEPS,
                "batch_size":        BATCH_SIZE,
                "base_lr":           BASE_LR,
                "weight_decay":      WEIGHT_DECAY,
                "optimizer":         "AdamW",
                "seeds":             SEEDS,
                "total_runs":        total_runs,
                "note": "500 steps = same probe budget (goal: seed variance for CIs, not convergence)",
            },
            "total_wall_clock_min": round(total_elapsed / 60, 1),
        },
        "subject_executed": (
            "TNT (Transformer-in-Transformer) on CIFAR-10 and CIFAR-100. "
            "Two arms: softmax inner attention (baseline) vs ReLU-kernel linear attention (inner only). "
            "Outer block: softmax in both arms. "
            "Config: img_size=32, patch_size=8, inner_stride=4, n_inner=4, d_inner=24, "
            "outer_dim=192, depth=6, outer_num_heads=3, inner_num_heads=4. "
            f"Training: {N_STEPS} gradient steps per run (same 500-step probe budget). "
            f"Seeds: {SEEDS}. Total runs: {total_runs}. "
            "Statistics: two-sided Welch's t-test, 95% CI per dataset."
        ),
        "notes": (
            f"Geometry: n_inner={N_INNER} < d_inner={D_INNER} → "
            f"linear-attn {COST_RATIO:.0f}× MORE expensive per head. Study is expressivity-only. "
            f"500-step budget used (same as probe): goal is seed variance for CIs. "
            f"CIFAR-10 delta: {delta_c10:+.4f}pp CI=[{stat_results['cifar10']['ci_95_lo']:+.4f},{stat_results['cifar10']['ci_95_hi']:+.4f}]. "
            f"CIFAR-100 delta: {delta_c100:+.4f}pp CI=[{stat_results['cifar100']['ci_95_lo']:+.4f},{stat_results['cifar100']['ci_95_hi']:+.4f}]. "
            f"Prediction: {pred_result}. "
            f"Total runtime: {total_elapsed/60:.1f} min."
        ),
    }

    out_path = "/workspace/results/increase_complexity-03/RESULTS.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))
