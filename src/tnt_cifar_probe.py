"""
TNT CIFAR-10 Sanity Probe — baseline-00
========================================
Geometry:
  img_size=32, patch_size=8  ->  num_outer_patches = (32/8)^2 = 16
  inner_stride=4             ->  num_words (n_inner) = ceil(8/4)^2 = 4
  inner_dim (d_inner) = 24

Sanity checks:
  (a) Initial CE loss ≈ ln(10) = 2.303
  (b) Test accuracy > 10 % after a short training run
  (c) Overfit a frozen 32-image batch to < 0.01 CE within 300 gradient steps
  (d) Reproducibility: two identical-seed runs give |loss_run1[i] - loss_run2[i]| < 1e-6
"""

import sys
import os
import math
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ── make the cloned repo importable ──────────────────────────────────────────
sys.path.insert(0, "/workspace/CV-Backbones/tnt_pytorch")

# Import layers that are still compatible with current timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Standalone TNT implementation (adapted from CV-Backbones/tnt_pytorch/tnt.py)
#     Only the geometry changes: img_size=32, patch_size=8, inner_stride=4, inner_dim=24
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


class Attention(nn.Module):
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


class TNTBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads,
                 num_words, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0.):
        super().__init__()
        # Inner block
        self.inner_norm1 = nn.LayerNorm(inner_dim)
        self.inner_attn  = Attention(inner_dim, inner_num_heads, qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.inner_norm2 = nn.LayerNorm(inner_dim)
        self.inner_mlp   = Mlp(inner_dim, int(inner_dim * mlp_ratio), drop=drop)
        # Projection inner → outer
        self.proj_norm1  = nn.LayerNorm(num_words * inner_dim)
        self.proj        = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
        self.proj_norm2  = nn.LayerNorm(outer_dim)
        # Outer block
        self.outer_norm1 = nn.LayerNorm(outer_dim)
        self.outer_attn  = Attention(outer_dim, outer_num_heads, qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.drop_path   = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = nn.LayerNorm(outer_dim)
        self.outer_mlp   = Mlp(outer_dim, int(outer_dim * mlp_ratio), drop=drop)

    def forward(self, inner_tokens, outer_tokens):
        # Inner self-attention
        inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))
        inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))
        # Project and add to outer (skip cls token at index 0)
        B, N, C = outer_tokens.shape
        outer_tokens[:, 1:] = (outer_tokens[:, 1:]
            + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1)))))
        # Outer self-attention
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
        outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class PatchEmbed(nn.Module):
    """Image → inner tokens.
       patch_size=8, inner_stride=4 → num_words=4 (2×2) per patch."""
    def __init__(self, img_size=32, patch_size=8, in_chans=3, inner_dim=24, inner_stride=4):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_words   = (math.ceil(patch_size / inner_stride)) ** 2  # = 4
        self.inner_dim   = inner_dim
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # Conv over each patch: 3 → inner_dim; kernel=7 pad=3 stride=4 → (8→2)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.unfold(x)                                            # B, C*p*p, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C,
                                       self.patch_size, self.patch_size)  # B*N, C, 8, 8
        x = self.proj(x)                                              # B*N, inner_dim, 2, 2
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)  # B*N, 4, inner_dim
        return x


class TNTSmall(nn.Module):
    """Tiny TNT for CIFAR-10 probe.

    Geometry log:
      outer_patch_size  = 8×8 px
      sub_patch_size    = 4×4 px  (inner_stride=4)
      n_inner (words)   = 4       (2×2 sub-patches per outer patch)
      d_inner           = 24
      outer_dim         = 192
      num_outer_patches = 16      (4×4 grid on 32×32 image)
      depth             = 6
    Note: n_inner (4) < d_inner (24) → softmax attention cheaper than
    linear-kernel (d² feature-map overhead); efficiency claim does NOT hold.
    """
    def __init__(self, img_size=32, patch_size=8, in_chans=3, num_classes=10,
                 outer_dim=192, inner_dim=24, depth=6,
                 outer_num_heads=3, inner_num_heads=4,
                 mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 inner_stride=4):
        super().__init__()
        self.num_classes = num_classes
        self.outer_dim   = outer_dim

        self.patch_embed  = PatchEmbed(img_size, patch_size, in_chans, inner_dim, inner_stride)
        num_patches = self.patch_embed.num_patches   # 16
        num_words   = self.patch_embed.num_words      # 4

        # Initial linear projection inner → outer token space
        self.proj_norm1 = nn.LayerNorm(num_words * inner_dim)
        self.proj       = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = nn.LayerNorm(outer_dim)

        # Positional embeddings
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, outer_dim))
        self.outer_pos  = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        self.inner_pos  = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop   = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TNTBlock(outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i])
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
        inner_tokens = self.patch_embed(x) + self.inner_pos      # B*N, 4, 24

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(
            inner_tokens.reshape(B, self.patch_embed.num_patches, -1))))  # B, 16, 192
        outer_tokens = torch.cat([self.cls_token.expand(B, -1, -1), outer_tokens], dim=1)
        outer_tokens = self.pos_drop(outer_tokens + self.outer_pos)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        x = self.norm(outer_tokens)[:, 0]
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_model(seed=42):
    set_seed(seed)
    return TNTSmall(
        img_size=32, patch_size=8, num_classes=10,
        outer_dim=192, inner_dim=24, depth=6,
        outer_num_heads=3, inner_num_heads=4,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        inner_stride=4
    ).to(DEVICE)


class HFCifar10Dataset(torch.utils.data.Dataset):
    """Wraps the HuggingFace CIFAR-10 dataset (uoft-cs/cifar10) in a PyTorch Dataset."""
    def __init__(self, hf_split, transform=None):
        self.ds        = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        img   = item["img"]                  # PIL Image
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar10_loaders(batch_size=128, data_root="/tmp/cifar10"):
    from datasets import load_dataset
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    print("Loading CIFAR-10 from HuggingFace cache (uoft-cs/cifar10)...", flush=True)
    hf_ds = load_dataset("uoft-cs/cifar10", trust_remote_code=True)
    train_ds = HFCifar10Dataset(hf_ds["train"], transform=train_tf)
    test_ds  = HFCifar10Dataset(hf_ds["test"],  transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def check_init_loss(train_loader):
    """(a) Initial CE ≈ ln(10) = 2.303 (pass if |loss - ln10| < 0.15)."""
    print("\n── Sanity (a): Initial cross-entropy ──────────────────", flush=True)
    model = make_model(seed=42)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        imgs, labels = next(iter(train_loader))
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels).item()
    ln10 = math.log(10)
    passed = abs(loss - ln10) < 0.15
    print(f"  init loss = {loss:.4f}, ln(10) = {ln10:.4f}, |diff| = {abs(loss-ln10):.4f}")
    print(f"  PASS: {passed}", flush=True)
    return loss, passed


def check_short_training_acc(train_loader, test_loader, n_steps=500):
    """(b) Test accuracy > 10 % after n_steps gradient steps."""
    print(f"\n── Sanity (b): Test acc >10% after {n_steps} steps ────", flush=True)
    model = make_model(seed=42)
    model.train()
    opt  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()

    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
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

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds   = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100. * correct / total
    passed = acc > 10.0
    print(f"  test acc = {acc:.2f}%  (pass if > 10%)  PASS: {passed}", flush=True)
    return acc, passed


def check_overfit_batch(train_loader, max_steps=300):
    """(c) Overfit a frozen 32-image batch to <0.01 CE within 300 steps.

    Strategy: lr=1e-2 + gradient clipping (max_norm=1.0) prevents explosion/collapse.
    If the 300-step hard gate is not met, we extend to 1500 steps and record both
    the 300-step min and the extended min so the reviewer can see the full picture.
    The PASS flag is evaluated strictly at ≤300 steps per the experiment spec.
    """
    print("\n── Sanity (c): Overfit 32-image fixed batch ───────────", flush=True)
    model = make_model(seed=42)
    # SGD + Nesterov momentum with cosine LR schedule from 0.5 → 0.
    # Nesterov SGD typically converges faster than Adam on small batch overfit tasks
    # because its look-ahead step exploits smooth loss surfaces more aggressively.
    opt   = optim.SGD(model.parameters(), lr=0.5, momentum=0.9,
                      nesterov=True, weight_decay=0.)
    crit  = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-4)

    # Grab exactly 32 images once and freeze them (no further augmentation)
    imgs, labels = next(iter(train_loader))
    imgs   = imgs[:32].to(DEVICE)
    labels = labels[:32].to(DEVICE)

    min_loss_300   = float("inf")
    min_loss_all   = float("inf")
    step_reached   = None
    HARD_LIMIT     = 300
    EXTENDED_LIMIT = max_steps  # caller sets this to 1500

    for step in range(1, EXTENDED_LIMIT + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(imgs), labels)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2
                        for p in model.parameters() if p.grad is not None) ** 0.5
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step <= 300:
            sched.step()
        val = loss.item()
        if val < min_loss_all:
            min_loss_all = val
        if step <= HARD_LIMIT and val < min_loss_300:
            min_loss_300 = val
        if step % 100 == 0 or (val < 0.01 and step_reached is None):
            print(f"  step {step:4d}  loss={val:.6f}  min={min_loss_all:.6f}  gnorm={grad_norm:.3f}", flush=True)
        if min_loss_all < 0.01 and step_reached is None:
            step_reached = step
            if step <= HARD_LIMIT:
                break  # strict gate met within 300 steps

    passed_300  = min_loss_300  < 0.01
    passed_ext  = min_loss_all  < 0.01
    print(f"  min_loss@300steps = {min_loss_300:.6f}  PASS(≤300): {passed_300}")
    print(f"  min_loss@{EXTENDED_LIMIT}steps  = {min_loss_all:.6f}  PASS(extended): {passed_ext}")
    if step_reached:
        print(f"  ✓ reached <0.01 at step {step_reached}", flush=True)
    return min_loss_300, passed_300, min_loss_all, step_reached


def check_reproducibility(train_loader):
    """(d) Two identical-seed runs give identical first-5-batch losses."""
    print("\n── Sanity (d): Reproducibility (two runs, seed=42) ────", flush=True)
    crit = nn.CrossEntropyLoss()

    def collect_losses(seed):
        model = make_model(seed=seed)
        model.train()
        set_seed(seed)  # reset RNG for dataloader order too
        losses = []
        loader_it = iter(train_loader)
        for _ in range(5):
            imgs, labels = next(loader_it)
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                losses.append(crit(model(imgs), labels).item())
        return losses

    # Run twice with the same seed (dataloader is already deterministic here)
    losses_a = collect_losses(42)
    losses_b = collect_losses(42)
    diffs    = [abs(a - b) for a, b in zip(losses_a, losses_b)]
    max_diff = max(diffs)
    passed   = max_diff < 1e-6
    print(f"  losses_run1 = {[f'{v:.6f}' for v in losses_a]}")
    print(f"  losses_run2 = {[f'{v:.6f}' for v in losses_b]}")
    print(f"  max |diff| = {max_diff:.2e}  PASS: {passed}", flush=True)
    return max_diff, passed


def geometry_table():
    """Print and return the geometry / FLOP table (Hypothesis-chain gate 5 & 6)."""
    print("\n── Geometry & FLOP table ───────────────────────────────", flush=True)
    img      = 32
    patch    = 8
    inner_s  = 4
    n_outer  = (img // patch) ** 2                     # 16
    n_inner  = (math.ceil(patch / inner_s)) ** 2       # 4
    d_inner  = 24
    d_outer  = 192
    depth    = 6

    # Rough FLOP counts (multiply-adds × 2 = FLOPs, but we compare ratios)
    # Inner attention per layer: 2 * n_inner^2 * d_inner (QK + AV) * n_outer
    inner_attn_flops_per_layer = 2 * (n_inner ** 2) * d_inner * n_outer
    # Outer attention per layer: 2 * (n_outer+1)^2 * d_outer
    outer_attn_flops_per_layer = 2 * ((n_outer + 1) ** 2) * d_outer

    total_inner = inner_attn_flops_per_layer * depth
    total_outer = outer_attn_flops_per_layer * depth
    total_all   = total_inner + total_outer

    print(f"  outer_patch_size : {patch}×{patch} px")
    print(f"  sub_patch_size   : {inner_s}×{inner_s} px (inner_stride)")
    print(f"  n_outer_patches  : {n_outer}")
    print(f"  n_inner (words)  : {n_inner}   ← per outer patch")
    print(f"  d_inner          : {d_inner}")
    print(f"  d_outer          : {d_outer}")
    print(f"  depth            : {depth}")
    print(f"  n_inner ≤ d_inner: {n_inner} ≤ {d_inner} → {n_inner <= d_inner}  (linear attn has NO FLOP advantage)")
    print(f"  inner_attn FLOPs (all layers) : {total_inner:,}")
    print(f"  outer_attn FLOPs (all layers) : {total_outer:,}")
    print(f"  inner_attn fraction           : {total_inner/total_all:.3f}")
    print(f"  max total-model cut (rm inner): {total_inner/total_all*100:.1f}%  (<10%? {total_inner/total_all < 0.10})")
    print("  → '12-18% total FLOP reduction' claim is REFUTED; study reframed as expressivity-only", flush=True)

    return {
        "outer_patch_size"   : patch,
        "sub_patch_size"     : inner_s,
        "n_outer_patches"    : n_outer,
        "n_inner"            : n_inner,
        "d_inner"            : d_inner,
        "d_outer"            : d_outer,
        "depth"              : depth,
        "inner_attn_fraction": round(total_inner / total_all, 4),
        "n_inner_le_d_inner" : bool(n_inner <= d_inner),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/tmp/cifar10")
    args = parser.parse_args()

    print("=" * 60)
    print("TNT CIFAR-10 Probe — baseline-00")
    print(f"Commit (CV-Backbones): f90e129b645c3b1684fe07cd361cd557d0ad71f7")
    print("=" * 60, flush=True)

    # --- Geometry table (gates 5 & 6) ---
    geo = geometry_table()

    # --- Data ---
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, data_root=args.data_root)

    # (a) init loss
    init_loss, pass_a = check_init_loss(train_loader)

    # (b) short training acc
    test_acc, pass_b = check_short_training_acc(train_loader, test_loader, n_steps=500)

    # (c) overfit batch (300-step hard gate + 1500-step extended run)
    min_loss_overfit, pass_c, min_loss_ext, step_reached = check_overfit_batch(
        train_loader, max_steps=1500)

    # (d) reproducibility
    max_repro_diff, pass_d = check_reproducibility(train_loader)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all([pass_a, pass_b, pass_c, pass_d])
    print(f"  (a) init_loss={init_loss:.4f}  target≈{math.log(10):.4f}  PASS={pass_a}")
    print(f"  (b) test_acc={test_acc:.2f}%   target>10%            PASS={pass_b}")
    print(f"  (c) min_overfit_loss@300={min_loss_overfit:.6f}  target<0.01   PASS(strict)={pass_c}")
    print(f"      min_overfit_loss@1500={min_loss_ext:.6f}  step_reached={step_reached}")
    print(f"  (d) max_repro_diff={max_repro_diff:.2e}  target<1e-6  PASS={pass_d}")
    print(f"  ALL_PASS: {all_pass}", flush=True)

    # all_pass uses the strict 300-step gate for (c)
    all_pass = all([pass_a, pass_b, pass_c, pass_d])

    results = {
        "status"            : "SUCCESS" if all_pass else "PARTIAL",
        "scale"             : "probe",
        "metrics"           : {
            "init_cross_entropy"         : round(init_loss, 4),
            "ln10_target"                : round(math.log(10), 4),
            "init_ce_pass"               : pass_a,
            "test_acc_pct_500steps"      : round(test_acc, 2),
            "test_acc_gt10_pass"         : pass_b,
            "min_overfit_loss_300steps"  : round(min_loss_overfit, 6),
            "overfit_lt_0p01_pass_strict": pass_c,
            "min_overfit_loss_1500steps" : round(min_loss_ext, 6),
            "overfit_step_reached"       : step_reached,
            "repro_max_diff"             : float(f"{max_repro_diff:.2e}"),
            "repro_pass"                 : pass_d,
            "all_sanity_pass"            : all_pass,
            "geometry"                   : geo,
        },
        "subject_executed"  : (
            "TNT (Transformer-in-Transformer) on CIFAR-10. "
            "Source: huawei-noah/CV-Backbones commit f90e129. "
            "Config: img_size=32, patch_size=8, inner_stride=4, "
            "n_inner=4, d_inner=24, outer_dim=192, depth=6, seed=42. "
            "Ran 4 sanity checks: (a) init CE, (b) 500-step test acc, "
            "(c) 300-step overfit batch, (d) reproducibility."
        ),
        "notes"             : (
            "Geometry: n_inner=4 < d_inner=24, so linear-attention has NO FLOP "
            "advantage (d^2 feature-map overhead dominates). Inner-block attention "
            f"accounts for {geo['inner_attn_fraction']*100:.1f}% of total attention "
            "FLOPs → '12-18% total FLOP reduction' claim is refuted; study reframed "
            "as expressivity-only. All 4 sanity checks reported above."
        ),
    }

    out_path = "/workspace/results/baseline-00/RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(results, indent=2))
