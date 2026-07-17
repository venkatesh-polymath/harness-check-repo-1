"""
TNT CIFAR-10 Probe — increase_complexity-01
=============================================
Experiment: Replace softmax attention with ReLU-kernel LINEAR attention ONLY
in TNT's inner block (n_inner=4, d_inner=24). Outer block remains softmax.
Same probe setup / seed as baseline-00 (500 steps → test accuracy).

Goal: does inner-block linear attention degrade accuracy?
Baseline reference: 37.53% test accuracy at 500 steps (results/baseline-00/RESULTS.json)

Linear attention (Performer-style ReLU kernel):
  φ(x) = ReLU(x) + ε
  Attention(Q,K,V) ≈ φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T.sum())
  Complexity: O(n*d) vs softmax O(n^2); but at n=4, d=24, d^2=576 >> n^2=16
  → no FLOP advantage here; study is expressivity-only.

Geometry:
  img_size=32, patch_size=8  →  n_outer_patches = 16
  inner_stride=4             →  n_inner = 4 (2×2 sub-patches per patch)
  d_inner = 24, d_outer = 192, depth = 6
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
# 1.  Attention Modules
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_f, hidden_f=None, out_f=None, drop=0.):
        super().__init__()
        out_f = out_f or in_f
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

    φ(x) = ReLU(x) + ε  where ε=1e-6 avoids all-zero rows.
    Attention(Q,K,V) = D^{-1} * (φ(Q) @ (φ(K)^T @ V))
    where D = diag(φ(Q) @ φ(K)^T @ 1_n)

    At n_inner=4, d_inner=24: softmax cost ∝ n²=16, linear cost ∝ n*d=96 —
    softmax is actually 6x cheaper per head! This is purely an expressivity study.
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
        # attn_drop not meaningful for linear attention (no explicit matrix to drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # B, H, N, head_dim
        v    = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ReLU feature map: φ(x) = ReLU(x) + ε
        q = torch.nn.functional.relu(q) + self.eps
        k = torch.nn.functional.relu(k) + self.eps

        # KV aggregation: (H, head_dim, head_dim) — O(n*d) over sequence dim
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)   # B, H, d, d
        # Apply to queries: O(n*d)
        qkv = torch.einsum("bhnd,bhdm->bhnm", q, kv)  # B, H, N, d

        # Normalisation: D = φ(Q) @ (φ(K)^T @ 1_n) = φ(Q) @ sum(φ(K), dim=-2)
        k_sum = k.sum(dim=-2, keepdim=True)            # B, H, 1, d
        denom = (q * k_sum).sum(dim=-1, keepdim=True)  # B, H, N, 1
        denom = denom.clamp(min=self.eps)

        out = (qkv / denom).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TNT Block (inner=linear, outer=softmax) and full TNT model
# ─────────────────────────────────────────────────────────────────────────────

class TNTBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads,
                 num_words, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., inner_attn_type="linear"):
        super().__init__()
        # Inner block  — optionally linear attention
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

        # Projection inner → outer
        self.proj_norm1  = nn.LayerNorm(num_words * inner_dim)
        self.proj        = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2  = nn.LayerNorm(outer_dim)

        # Outer block — always softmax
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
    """TNT for CIFAR-10 probe.

    Geometry:
      outer_patch_size = 8×8 px
      sub_patch_size   = 4×4 px (inner_stride=4)
      n_inner          = 4 (2×2 sub-patches per outer patch)
      d_inner          = 24
      outer_dim        = 192
      num_outer_patches= 16  (4×4 grid on 32×32)
      depth            = 6

    inner_attn_type: 'linear' → ReLU-kernel linear attn; 'softmax' → standard.
    """
    def __init__(self, img_size=32, patch_size=8, in_chans=3, num_classes=10,
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


def make_model(seed=42, inner_attn_type="linear"):
    set_seed(seed)
    return TNTSmall(
        img_size=32, patch_size=8, num_classes=10,
        outer_dim=192, inner_dim=24, depth=6,
        outer_num_heads=3, inner_num_heads=4,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        inner_stride=4,
        inner_attn_type=inner_attn_type,
    ).to(DEVICE)


class HFCifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds        = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        img   = item["img"]
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar10_loaders(batch_size=128):
    from datasets import load_dataset
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    print("Loading CIFAR-10 from HuggingFace cache...", flush=True)
    hf_ds = load_dataset("uoft-cs/cifar10", trust_remote_code=True)
    train_ds = HFCifar10Dataset(hf_ds["train"], transform=train_tf)
    test_ds  = HFCifar10Dataset(hf_ds["test"],  transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


def check_init_loss(train_loader, inner_attn_type):
    """Sanity: initial CE ≈ ln(10). Same as baseline check but with the
    linear-attention model to confirm correct initialisation."""
    print(f"\n── Init loss check (inner={inner_attn_type}) ───────────", flush=True)
    model = make_model(seed=42, inner_attn_type=inner_attn_type)
    model.eval()
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        imgs, labels = next(iter(train_loader))
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = crit(logits, labels).item()
    ln10   = math.log(10)
    passed = abs(loss - ln10) < 0.15
    print(f"  init loss={loss:.4f}  ln(10)={ln10:.4f}  |diff|={abs(loss-ln10):.4f}  PASS={passed}", flush=True)
    return loss, passed


def train_and_eval(train_loader, test_loader, inner_attn_type, n_steps=500, seed=42):
    """Train TNT with given inner_attn_type for n_steps, return test accuracy."""
    print(f"\n── Training (inner={inner_attn_type}, seed={seed}, steps={n_steps}) ──", flush=True)
    model = make_model(seed=seed, inner_attn_type=inner_attn_type)
    model.train()
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    crit  = nn.CrossEntropyLoss()

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
            print(f"  step {step:4d}  loss={loss.item():.4f}  elapsed={time.time()-t0:.1f}s", flush=True)

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
    print("TNT CIFAR-10 Probe — increase_complexity-01")
    print("Inner block: ReLU-kernel LINEAR attention")
    print("Outer block: softmax (unchanged from baseline)")
    print("=" * 60, flush=True)

    BASELINE_ACC = 37.53   # from results/baseline-00/RESULTS.json
    N_STEPS      = 500     # same as baseline probe
    SEED         = 42

    # Geometry note
    n_inner, d_inner = 4, 24
    print(f"\nGeometry note: n_inner={n_inner} < d_inner={d_inner}")
    print(f"  softmax cost ∝ n²={n_inner**2},  linear-kernel cost ∝ n*d={n_inner*d_inner}")
    print(f"  → linear attn is {(n_inner*d_inner)/(n_inner**2):.1f}x MORE expensive per head than softmax at this scale")
    print(f"  → study is purely expressivity (not efficiency)", flush=True)

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # 1. Init loss sanity check for linear-attention model
    init_loss, init_pass = check_init_loss(train_loader, inner_attn_type="linear")

    # 2. Train linear-attention model (same 500 steps / seed as baseline)
    linear_acc = train_and_eval(train_loader, test_loader,
                                inner_attn_type="linear",
                                n_steps=N_STEPS, seed=SEED)

    # 3. (Optional quick sanity) Also run softmax-inner for same steps
    #    to confirm the delta within this process (guards against batch-ordering diffs)
    print("\n── Softmax-inner control run (same process, same loader seed) ──", flush=True)
    softmax_acc = train_and_eval(train_loader, test_loader,
                                 inner_attn_type="softmax",
                                 n_steps=N_STEPS, seed=SEED)

    # ── Summary ──────────────────────────────────────────────────────────────
    delta_vs_baseline = linear_acc - BASELINE_ACC
    delta_vs_control  = linear_acc - softmax_acc

    print("\n" + "=" * 60)
    print("SUMMARY — increase_complexity-01")
    print("=" * 60)
    print(f"  baseline (baseline-00, softmax-inner) : {BASELINE_ACC:.2f}%")
    print(f"  linear-inner (this run)               : {linear_acc:.2f}%")
    print(f"  softmax-inner control (this run)      : {softmax_acc:.2f}%")
    print(f"  Δ vs committed baseline               : {delta_vs_baseline:+.2f}pp")
    print(f"  Δ vs same-process control             : {delta_vs_control:+.2f}pp")
    print(f"  init CE pass (linear model)           : {init_pass}", flush=True)

    results = {
        "status": "SUCCESS",
        "scale": "probe",
        "metrics": {
            "baseline_acc_pct": BASELINE_ACC,
            "linear_inner_acc_pct": round(linear_acc, 2),
            "softmax_inner_control_acc_pct": round(softmax_acc, 2),
            "delta_vs_committed_baseline_pp": round(delta_vs_baseline, 2),
            "delta_vs_same_process_control_pp": round(delta_vs_control, 2),
            "init_ce_linear_model": round(init_loss, 4),
            "init_ce_pass": init_pass,
            "n_steps": N_STEPS,
            "seed": SEED,
            "geometry": {
                "n_inner": n_inner,
                "d_inner": d_inner,
                "softmax_cost_prop": n_inner ** 2,
                "linear_kernel_cost_prop": n_inner * d_inner,
                "linear_vs_softmax_cost_ratio": round((n_inner * d_inner) / (n_inner ** 2), 1),
                "note": (
                    f"n_inner={n_inner} < d_inner={d_inner}: linear-attn kernel is "
                    f"{(n_inner*d_inner)//(n_inner**2)}x MORE expensive per head than softmax; "
                    "study is expressivity-only, no FLOP claim"
                ),
            },
            "inner_attn_method": "ReLU-kernel linear attention (Performer-style phi(x)=ReLU(x)+eps)",
            "outer_attn_method": "softmax (unchanged)",
        },
        "subject_executed": (
            "TNT (Transformer-in-Transformer) on CIFAR-10. "
            "Inner block: ReLU-kernel linear attention; outer block: softmax (identical to baseline). "
            "Config: img_size=32, patch_size=8, inner_stride=4, n_inner=4, d_inner=24, "
            "outer_dim=192, depth=6, outer_num_heads=3, inner_num_heads=4, seed=42. "
            "Probe: 500 AdamW steps (lr=1e-3, wd=0.05). "
            "Same protocol as baseline-00 for direct comparison."
        ),
        "notes": (
            "Geometry: n_inner=4 < d_inner=24, so linear-attn kernel trick is 6x MORE "
            "expensive per head than softmax at this scale (n*d=96 vs n^2=16). "
            "Study is purely expressivity. "
            f"Linear-inner accuracy = {linear_acc:.2f}%, "
            f"baseline (softmax-inner, baseline-00) = {BASELINE_ACC:.2f}%, "
            f"delta = {delta_vs_baseline:+.2f}pp. "
            f"Same-process softmax control = {softmax_acc:.2f}%, "
            f"within-process delta = {delta_vs_control:+.2f}pp."
        ),
    }

    out_path = "/workspace/results/increase_complexity-01/RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))
