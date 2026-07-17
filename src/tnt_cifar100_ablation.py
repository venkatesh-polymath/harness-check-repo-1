"""
TNT CIFAR-100 Probe — ablation-02
==================================
Experiment: Same comparison as increase_complexity-01 (softmax-inner vs
ReLU-kernel linear-inner, n_inner=4, d_inner=24) but on CIFAR-100 (100 classes).

Question: Is the accuracy delta (linear - softmax) WORSE on CIFAR-100 than
the +0.15pp seen on CIFAR-10?

Prior numbers (increase_complexity-01):
  CIFAR-10 softmax-inner : 37.53%
  CIFAR-10 linear-inner  : 37.68%
  CIFAR-10 delta         : +0.15pp  (linear slightly better)

Geometry note:
  n_inner=4 < d_inner=24  → linear-attn kernel is 6x MORE expensive than softmax
  → study is purely expressivity; no efficiency claim.

Seed = 42, 500 AdamW steps (lr=1e-3, wd=0.05), batch_size=128.
"""

import sys
import os
import math
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

sys.path.insert(0, "/workspace/CV-Backbones/tnt_pytorch")
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Attention Modules  (identical to increase_complexity-01)
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
    """Standard softmax self-attention (used in outer block)."""
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v  = nn.Linear(dim, dim,     bias=qkv_bias)
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
    """ReLU-kernel linear attention (Performer-style) for inner block.

    phi(x) = ReLU(x) + eps
    Attention(Q,K,V) = D^{-1} * (phi(Q) @ (phi(K)^T @ V))
    """
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.eps       = eps
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v  = nn.Linear(dim, dim,     bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # B, H, N, head_dim
        v    = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ReLU feature map
        q = torch.nn.functional.relu(q) + self.eps
        k = torch.nn.functional.relu(k) + self.eps

        # KV aggregation: O(n*d)
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)     # B, H, d, d
        qkv = torch.einsum("bhnd,bhdm->bhnm", q, kv)   # B, H, N, d

        # Normalisation
        k_sum = k.sum(dim=-2, keepdim=True)             # B, H, 1, d
        denom = (q * k_sum).sum(dim=-1, keepdim=True)   # B, H, N, 1
        denom = denom.clamp(min=self.eps)

        out = (qkv / denom).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TNT Block and full model
# ─────────────────────────────────────────────────────────────────────────────

class TNTBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads,
                 num_words, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., inner_attn_type="linear"):
        super().__init__()
        self.inner_norm1 = nn.LayerNorm(inner_dim)
        if inner_attn_type == "linear":
            self.inner_attn = LinearAttention(
                inner_dim, inner_num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)
        else:
            self.inner_attn = SoftmaxAttention(
                inner_dim, inner_num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop)
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
        self.num_words   = (math.ceil(patch_size / inner_stride)) ** 2  # 4
        self.inner_dim   = inner_dim
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, C,
                                       self.patch_size, self.patch_size)
        x = self.proj(x)
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)
        return x


class TNTSmall(nn.Module):
    """TNT for CIFAR probe — parameterised by num_classes."""
    def __init__(self, img_size=32, patch_size=8, in_chans=3, num_classes=100,
                 outer_dim=192, inner_dim=24, depth=6,
                 outer_num_heads=3, inner_num_heads=4,
                 mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 inner_stride=4, inner_attn_type="linear"):
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
# 3.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_model(seed=42, inner_attn_type="linear", num_classes=100):
    set_seed(seed)
    return TNTSmall(
        img_size=32, patch_size=8, num_classes=num_classes,
        outer_dim=192, inner_dim=24, depth=6,
        outer_num_heads=3, inner_num_heads=4,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        inner_stride=4,
        inner_attn_type=inner_attn_type,
    ).to(DEVICE)


class HFCifar100Dataset(torch.utils.data.Dataset):
    """Wraps the HuggingFace CIFAR-100 dataset; uses fine_label (100 classes)."""
    def __init__(self, hf_split, transform=None):
        self.ds        = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        img   = item["img"]
        label = item["fine_label"]   # 100-class label
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar100_loaders(batch_size=128):
    from datasets import load_dataset
    # CIFAR-100 channel stats
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    print("Loading CIFAR-100 from HuggingFace cache...", flush=True)
    hf_ds = load_dataset("uoft-cs/cifar100")   # no trust_remote_code needed
    train_ds = HFCifar100Dataset(hf_ds["train"], transform=train_tf)
    test_ds  = HFCifar100Dataset(hf_ds["test"],  transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


def check_init_loss(train_loader, inner_attn_type, num_classes=100):
    """Sanity: initial CE ≈ ln(num_classes)."""
    print(f"\n── Init loss check (inner={inner_attn_type}, {num_classes} classes) ───", flush=True)
    model = make_model(seed=42, inner_attn_type=inner_attn_type, num_classes=num_classes)
    model.eval()
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        imgs, labels = next(iter(train_loader))
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = crit(logits, labels).item()
    ln_c   = math.log(num_classes)
    passed = abs(loss - ln_c) < 0.30   # slightly looser tolerance for 100 classes
    print(f"  init loss={loss:.4f}  ln({num_classes})={ln_c:.4f}  "
          f"|diff|={abs(loss-ln_c):.4f}  PASS={passed}", flush=True)
    return loss, passed


def train_and_eval(train_loader, test_loader, inner_attn_type,
                   num_classes=100, n_steps=500, seed=42):
    """Train TNT for n_steps, return test accuracy."""
    print(f"\n── Training (inner={inner_attn_type}, {num_classes} classes, "
          f"seed={seed}, steps={n_steps}) ──", flush=True)
    model = make_model(seed=seed, inner_attn_type=inner_attn_type, num_classes=num_classes)
    model.train()
    opt  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()

    step = 0
    t0   = time.time()
    train_iter = iter(train_loader)
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
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds    = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100. * correct / total
    print(f"  Test accuracy = {acc:.2f}%", flush=True)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TNT CIFAR-100 Probe — ablation-02")
    print("Inner block: ReLU-kernel LINEAR vs SOFTMAX attention")
    print("Outer block: softmax (unchanged in both arms)")
    print("=" * 60, flush=True)

    # Prior CIFAR-10 numbers from increase_complexity-01
    CIFAR10_SOFTMAX_ACC = 37.53   # from results/increase_complexity-01/RESULTS.json
    CIFAR10_LINEAR_ACC  = 37.68
    CIFAR10_DELTA       = 0.15    # linear - softmax = +0.15pp on CIFAR-10

    NUM_CLASSES = 100
    N_STEPS     = 500             # same probe length as prior rounds
    SEED        = 42

    # Geometry (same as increase_complexity-01)
    n_inner, d_inner = 4, 24
    print(f"\nGeometry: n_inner={n_inner}, d_inner={d_inner}")
    print(f"  softmax cost ∝ n²={n_inner**2}")
    print(f"  linear-kernel cost ∝ n*d={n_inner*d_inner}")
    print(f"  → linear attn is {(n_inner*d_inner)/(n_inner**2):.1f}x MORE expensive per head")
    print(f"  → study is purely expressivity (no efficiency claim)", flush=True)

    # Prior CIFAR-10 reference
    print(f"\nPrior CIFAR-10 numbers (increase_complexity-01):")
    print(f"  softmax-inner : {CIFAR10_SOFTMAX_ACC:.2f}%")
    print(f"  linear-inner  : {CIFAR10_LINEAR_ACC:.2f}%")
    print(f"  delta         : {CIFAR10_DELTA:+.2f}pp", flush=True)

    # Load CIFAR-100
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)

    # 1. Init loss sanity check (should be ≈ ln(100) ≈ 4.605)
    init_loss, init_pass = check_init_loss(train_loader, inner_attn_type="linear",
                                           num_classes=NUM_CLASSES)

    # 2. Train softmax-inner baseline on CIFAR-100
    softmax_c100_acc = train_and_eval(train_loader, test_loader,
                                      inner_attn_type="softmax",
                                      num_classes=NUM_CLASSES,
                                      n_steps=N_STEPS, seed=SEED)

    # 3. Train linear-inner on CIFAR-100
    linear_c100_acc = train_and_eval(train_loader, test_loader,
                                     inner_attn_type="linear",
                                     num_classes=NUM_CLASSES,
                                     n_steps=N_STEPS, seed=SEED)

    # ── Derived metrics ──────────────────────────────────────────────────────
    delta_c100   = linear_c100_acc - softmax_c100_acc   # positive = linear better
    delta_change = delta_c100 - CIFAR10_DELTA            # negative = worse on CIFAR-100

    print("\n" + "=" * 60)
    print("SUMMARY — ablation-02 (CIFAR-100 vs CIFAR-10 comparison)")
    print("=" * 60)
    print(f"  CIFAR-10  softmax-inner : {CIFAR10_SOFTMAX_ACC:.2f}%")
    print(f"  CIFAR-10  linear-inner  : {CIFAR10_LINEAR_ACC:.2f}%")
    print(f"  CIFAR-10  delta         : {CIFAR10_DELTA:+.2f}pp")
    print(f"  CIFAR-100 softmax-inner : {softmax_c100_acc:.2f}%")
    print(f"  CIFAR-100 linear-inner  : {linear_c100_acc:.2f}%")
    print(f"  CIFAR-100 delta         : {delta_c100:+.2f}pp")
    print(f"  delta_change (C100−C10) : {delta_change:+.2f}pp  "
          f"({'worse on CIFAR-100' if delta_change < 0 else 'NOT worse on CIFAR-100'})")
    print(f"  init CE pass            : {init_pass}", flush=True)

    # Prediction assessment
    prediction_text = (
        "CONFIRMED: delta is more negative on CIFAR-100 than on CIFAR-10 "
        "(linear attention hurts more on the harder dataset)."
        if delta_change < -0.5
        else ("BORDERLINE: delta worsened slightly on CIFAR-100 but <0.5pp."
              if delta_change < 0
              else "NOT CONFIRMED: CIFAR-100 delta is not worse than CIFAR-10 delta.")
    )
    print(f"\nPrediction assessment: {prediction_text}", flush=True)

    results = {
        "status": "SUCCESS",
        "scale": "probe",
        "metrics": {
            "cifar10_softmax_inner_acc_pct": CIFAR10_SOFTMAX_ACC,
            "cifar10_linear_inner_acc_pct":  CIFAR10_LINEAR_ACC,
            "cifar10_delta_pp":              round(CIFAR10_DELTA, 2),
            "cifar100_softmax_inner_acc_pct": round(softmax_c100_acc, 2),
            "cifar100_linear_inner_acc_pct":  round(linear_c100_acc,  2),
            "cifar100_delta_pp":              round(delta_c100, 2),
            "delta_change_c100_minus_c10_pp": round(delta_change, 2),
            "prediction_confirmed":           delta_change < -0.5,
            "init_ce_cifar100":               round(init_loss, 4),
            "init_ce_pass":                   init_pass,
            "n_steps": N_STEPS,
            "seed":    SEED,
            "geometry": {
                "n_inner": n_inner,
                "d_inner": d_inner,
                "softmax_cost_prop":      n_inner ** 2,
                "linear_kernel_cost_prop": n_inner * d_inner,
                "linear_vs_softmax_cost_ratio": round((n_inner * d_inner) / (n_inner ** 2), 1),
                "note": (
                    f"n_inner={n_inner} < d_inner={d_inner}: linear-attn kernel is "
                    f"{(n_inner*d_inner)//(n_inner**2)}x MORE expensive per head than softmax; "
                    "study is expressivity-only, no FLOP claim."
                ),
            },
            "inner_attn_method": "ReLU-kernel linear attention (Performer-style phi(x)=ReLU(x)+eps)",
            "outer_attn_method": "softmax (unchanged in both arms)",
        },
        "subject_executed": (
            "TNT (Transformer-in-Transformer) on CIFAR-100. "
            "Arm A: softmax inner attention; Arm B: ReLU-kernel linear inner attention. "
            "Outer block: softmax in both arms. "
            "Config: img_size=32, patch_size=8, inner_stride=4, n_inner=4, d_inner=24, "
            "outer_dim=192, depth=6, outer_num_heads=3, inner_num_heads=4, "
            "num_classes=100, seed=42. "
            "Probe: 500 AdamW steps (lr=1e-3, wd=0.05). "
            "Direct contrast with increase_complexity-01 (same protocol on CIFAR-10)."
        ),
        "notes": (
            f"CIFAR-10 delta (linear−softmax): {CIFAR10_DELTA:+.2f}pp (from increase_complexity-01). "
            f"CIFAR-100 delta (linear−softmax): {delta_c100:+.2f}pp. "
            f"delta_change (C100−C10): {delta_change:+.2f}pp. "
            f"Prediction ({prediction_text}). "
            "Geometry: n_inner=4 < d_inner=24 → linear attn 6x MORE expensive per head; "
            "study is purely expressivity."
        ),
    }

    out_path = "/workspace/results/ablation-02/RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))
