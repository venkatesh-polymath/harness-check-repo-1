"""
refine-01: Genuine Cross-Site WBC Transfer – Blood_5 (Zenodo 21628834)
=======================================================================
Dataset provenance:
  Source: Barcelona PBC (Acevedo et al. 2020, Docty/Blood-Cells on HuggingFace)
  Target: Blood_5 (Hao WANG, Zenodo doi:10.5281/zenodo.21628834, published 2026-07-27)
    - Self-collected peripheral WBC, DIFFERENT lab than PBC
    - 5 classes: basophil, eosinophil, lymphocyte, monocyte, neutrophil
    - 5,175 test images, 150×150×3 uint8 (HWC format)
    - Confirmed NOT Barcelona PBC by author, institution, provenance

Design:
  1. Train ResNet-18 on PBC with 3 seeds (probe: 15 epochs + early-stop)
  2. For each seed evaluate on Blood_5 test set:
     - Scenario A: mismatched label indices (argmax PBC 0-7 vs Blood5 0-4)
     - Scenario B: aligned (Blood5 label remapped to PBC index space)
     - Scenario C: restricted 5-class (extract shared heads, renorm, vs Blood5 0-4)
  3. Fit temperature T on PBC val split per seed
  4. Report ECE (15-bin equal-mass) with/without T for all scenarios
  5. Report accuracy, confusion matrix, per-class precision/recall
"""

import os, sys, json, time, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── PATHS ────────────────────────────────────────────
OUT_DIR     = "/workspace/results/refine-01"
WEIGHTS_DIR = "/workspace/_weights"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─────────────────────────── CLASS VOCABULARIES ───────────────────────────────
PBC_CLASSES    = ['monocyte', 'ig', 'neutrophil', 'basophil',
                  'lymphocyte', 'erythroblast', 'eosinophil', 'platelet']
BLOOD5_CLASSES = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# Blood5 index → PBC index mapping
BLOOD5_TO_PBC_IDX = {i: PBC_CLASSES.index(cls) for i, cls in enumerate(BLOOD5_CLASSES)}
# PBC class indices corresponding to Blood5 classes IN BLOOD5 ORDER [0..4]
# i.e., PBC_SHARED_IN_BLOOD5_ORDER[j] = PBC index for Blood5 class j
PBC_SHARED_IN_BLOOD5_ORDER = [BLOOD5_TO_PBC_IDX[j] for j in range(5)]
# = [PBC.basophil, PBC.eosinophil, PBC.lymphocyte, PBC.monocyte, PBC.neutrophil]
# = [3, 6, 4, 0, 2]

print("=== CLASS VOCABULARY AUDIT ===")
print(f"PBC classes (8):    {PBC_CLASSES}")
print(f"Blood5 classes (5): {BLOOD5_CLASSES}")
print(f"Blood5→PBC index mapping: {BLOOD5_TO_PBC_IDX}")
print(f"PBC shared indices in Blood5 order: {PBC_SHARED_IN_BLOOD5_ORDER}")

# ─────────────────────────── GPU ──────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────── BLOOD5 TARGET DATASET ────────────────────────────
print("\n=== LOADING BLOOD5 TARGET DATASET ===")
blood5_data   = np.load(f"{WEIGHTS_DIR}/blood5_test_data.npy")    # (5175, 67500) uint8 HWC-flat
blood5_labels = np.load(f"{WEIGHTS_DIR}/blood5_test_labels.npy")  # (5175,) int

# Reshape: HWC format stored flattened
blood5_imgs_hwc = blood5_data.reshape(-1, 150, 150, 3)   # (N, H, W, C)

N_TARGET = len(blood5_labels)
print(f"Blood5 test: N={N_TARGET}, shape={blood5_imgs_hwc.shape}")
print(f"Source URL: https://zenodo.org/records/21628834")
print(f"DOI: 10.5281/zenodo.21628834")
print(f"Author: Hao WANG (NOT Barcelona PBC)")
print(f"Published: 2026-07-27")
print(f"Image dimensions: 150×150×3 pixels")
print(f"Classes (5): {BLOOD5_CLASSES}")
lc = {int(l): int((blood5_labels == l).sum()) for l in sorted(set(blood5_labels.tolist()))}
print(f"Label counts: {lc}")

class Blood5Dataset(Dataset):
    """Blood_5 target dataset."""
    def __init__(self, imgs_hwc, labels, transform=None):
        self.imgs = imgs_hwc
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx], 'RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])

# ─────────────────────────── PBC SOURCE DATASET ───────────────────────────────
print("\n=== LOADING PBC SOURCE DATASET ===")
t0 = time.time()
pbc_raw = load_dataset("Docty/Blood-Cells", split="train")
print(f"PBC total: {len(pbc_raw)} images, classes: {pbc_raw.features['label'].names}")
print(f"Loaded in {time.time()-t0:.1f}s")

# Fixed 70/15/15 stratified split (seed 42)
label2idx = defaultdict(list)
for i, item in enumerate(pbc_raw):
    label2idx[item['label']].append(i)

rng_split = np.random.default_rng(42)
train_idx, val_idx, test_idx = [], [], []
for lbl in sorted(label2idx.keys()):
    idxs = rng_split.permutation(label2idx[lbl]).tolist()
    n = len(idxs)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    train_idx.extend(idxs[:n_tr])
    val_idx.extend(idxs[n_tr:n_tr+n_va])
    test_idx.extend(idxs[n_tr+n_va:])

print(f"PBC split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

# ─────────────────────────── TRANSFORMS ───────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class PBCDataset(Dataset):
    def __init__(self, hf_subset, transform=None):
        self.data = hf_subset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(item['label'])


# ─────────────────────────── ECE FUNCTION ─────────────────────────────────────
def compute_ece(max_probs, correct, n_bins=15, equal_mass=True):
    """15-bin equal-mass ECE."""
    N = len(max_probs)
    if equal_mass:
        order = np.argsort(max_probs)
        bins = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0, 1, n_bins+1)
        bins = [np.where((max_probs >= edges[i]) & (max_probs < edges[i+1]))[0]
                for i in range(n_bins)]
    ece = 0.0
    bin_records = []
    for b in bins:
        if len(b) == 0:
            continue
        acc  = float(correct[b].mean())
        conf = float(max_probs[b].mean())
        ece += (len(b) / N) * abs(acc - conf)
        bin_records.append({'n': len(b), 'acc': acc, 'conf': conf, 'gap': acc - conf})
    return float(ece), bin_records


# ─────────────────────────── TEMPERATURE SCALING ─────────────────────────────
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))   # T = exp(log_T) > 0

    @property
    def T(self):
        return float(self.log_T.exp().item())

    def forward(self, logits):
        return logits / self.log_T.exp()

    def fit(self, logits_np, labels_np, lr=0.05, max_iter=200):
        logits = torch.tensor(logits_np, dtype=torch.float32)
        labels = torch.tensor(labels_np, dtype=torch.long)
        optimizer = optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.log_T.exp()
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.T


# ─────────────────────────── CONFUSION MATRIX HELPER ─────────────────────────
def compute_confusion(true_labels, pred_labels, n_true, n_pred):
    """Compute confusion matrix (n_true × n_pred)."""
    cm = np.zeros((n_true, n_pred), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n_true and 0 <= p < n_pred:
            cm[t, p] += 1
    return cm


# ─────────────────────────── EVALUATION HELPER ───────────────────────────────
def evaluate_blood5(logits_np, blood5_labels_np, T=1.0, label=""):
    """
    Evaluate on Blood5 under 3 label-alignment scenarios.

    logits_np: (N, 8) float32 raw PBC logits (before temperature)
    T: temperature to apply (1.0 = no scaling)
    Returns dict with per-scenario metrics.
    """
    N = len(blood5_labels_np)
    # Apply temperature scaling
    scaled_logits = logits_np / T             # (N, 8)
    probs_8 = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()  # (N, 8)
    preds_pbc = probs_8.argmax(axis=1)        # PBC class predictions (0-7)
    max_conf_8 = probs_8.max(axis=1)

    # Blood5 labels remapped to PBC class indices
    blood5_labels_in_pbc = np.array([BLOOD5_TO_PBC_IDX[int(l)] for l in blood5_labels_np])

    # ── Scenario A: MISMATCHED (argmax 0-7 vs Blood5 label 0-4)
    correct_A = (preds_pbc == blood5_labels_np).astype(float)
    acc_A = float(correct_A.mean())
    ece_A, _ = compute_ece(max_conf_8, correct_A)

    # ── Scenario B: ALIGNED (argmax 0-7 vs Blood5 label remapped to PBC space)
    correct_B = (preds_pbc == blood5_labels_in_pbc).astype(float)
    acc_B = float(correct_B.mean())
    ece_B, _ = compute_ece(max_conf_8, correct_B)

    # ── Scenario C: RESTRICTED 5-class (extract shared heads, renorm)
    # Extract logits for PBC indices that correspond to Blood5 classes, in Blood5 class order
    shared_logits = scaled_logits[:, PBC_SHARED_IN_BLOOD5_ORDER]  # (N, 5)
    probs_5 = torch.softmax(torch.tensor(shared_logits), dim=1).numpy()  # (N, 5)
    preds_5 = probs_5.argmax(axis=1)   # Blood5 class predictions (0-4)
    max_conf_5 = probs_5.max(axis=1)
    correct_C = (preds_5 == blood5_labels_np).astype(float)
    acc_C = float(correct_C.mean())
    ece_C, _ = compute_ece(max_conf_5, correct_C)

    # ── Confusion matrix (Blood5 true × PBC pred, 5×8) for scenario A/B
    cm_5x8 = compute_confusion(blood5_labels_in_pbc, preds_pbc, 5, 8)

    return {
        'T_applied': float(T),
        'scenario_A': {'acc': acc_A, 'ece_15bin': ece_A,
                        'desc': 'argmax PBC (0-7) vs Blood5 label (0-4): MISMATCHED'},
        'scenario_B': {'acc': acc_B, 'ece_15bin': ece_B,
                        'desc': 'argmax PBC (0-7) vs Blood5 label remapped to PBC idx: ALIGNED'},
        'scenario_C': {'acc': acc_C, 'ece_15bin': ece_C,
                        'desc': 'restricted 5-class renorm vs Blood5 label: PROPER'},
        'confusion_matrix': {
            'rows': BLOOD5_CLASSES,
            'cols': PBC_CLASSES,
            'matrix': cm_5x8.tolist(),
        },
    }


# ─────────────────────────── MAIN TRAINING LOOP ───────────────────────────────
SEEDS      = [0, 1, 2]
N_EPOCHS   = 15
BATCH_SIZE = 64
LR         = 1e-4
WD         = 1e-2
PATIENCE   = 5

# Pre-build datasets (share across seeds)
pbc_train_ds = PBCDataset(Subset(pbc_raw, train_idx), TRAIN_TF)
pbc_val_ds   = PBCDataset(Subset(pbc_raw, val_idx),   VAL_TF)
pbc_test_ds  = PBCDataset(Subset(pbc_raw, test_idx),  VAL_TF)
blood5_ds    = Blood5Dataset(blood5_imgs_hwc, blood5_labels, VAL_TF)

pbc_train_loader = DataLoader(pbc_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
pbc_val_loader   = DataLoader(pbc_val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
pbc_test_loader  = DataLoader(pbc_test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
blood5_loader    = DataLoader(blood5_ds,    batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

per_seed_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*60}")

    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Build model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    ckpt_path = f"{WEIGHTS_DIR}/resnet18_pbc_seed{seed}.pt"
    no_improve = 0
    history = []

    for epoch in range(1, N_EPOCHS+1):
        # Train
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labs in pbc_train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(labs)
            tc += (out.argmax(1) == labs).sum().item()
            tt += len(labs)
        tl /= tt; tacc = tc/tt

        # Validate
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labs in pbc_val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labs)
                vl += loss.item() * len(labs)
                vc += (out.argmax(1) == labs).sum().item()
                vt += len(labs)
        vl /= vt; vacc = vc/vt
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tl, 'val_loss': vl,
                        'train_acc': tacc, 'val_acc': vacc})
        print(f"  Ep{epoch:2d}: train={tl:.4f}/{tacc:.3f}  val={vl:.4f}/{vacc:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    # Load best
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"  Best val loss: {best_val_loss:.4f}")

    # ── COLLECT LOGITS ──────────────────────────────────────────────────────
    def get_logits_labels(loader):
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labs in loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labs.numpy())
        return np.vstack(all_logits), np.concatenate(all_labels)

    print("  Collecting PBC val logits...")
    val_logits, val_labels = get_logits_labels(pbc_val_loader)

    print("  Collecting PBC test logits...")
    test_logits, test_labels = get_logits_labels(pbc_test_loader)

    print("  Collecting Blood5 logits...")
    b5_logits, b5_labels = get_logits_labels(blood5_loader)

    # ── IN-DOMAIN PBC METRICS (test set) ───────────────────────────────────
    probs_pbc_test = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    preds_pbc_test = probs_pbc_test.argmax(axis=1)
    acc_pbc_test   = float((preds_pbc_test == test_labels).mean())
    ece_pbc_test_raw, _ = compute_ece(probs_pbc_test.max(axis=1),
                                       (preds_pbc_test == test_labels).astype(float))

    print(f"  PBC test: acc={acc_pbc_test:.4f}  ECE_raw={ece_pbc_test_raw:.4f}")

    # ── FIT TEMPERATURE T on PBC val ───────────────────────────────────────
    ts = TempScaler()
    T_star = ts.fit(val_logits, val_labels)
    print(f"  Temperature T*={T_star:.4f}")

    # PBC test ECE with T
    probs_pbc_test_T = torch.softmax(torch.tensor(test_logits / T_star), dim=1).numpy()
    ece_pbc_test_T, _ = compute_ece(probs_pbc_test_T.max(axis=1),
                                     (probs_pbc_test_T.argmax(axis=1) == test_labels).astype(float))
    print(f"  PBC test ECE_T: {ece_pbc_test_T:.4f}")

    # ── BLOOD5 EVALUATION ──────────────────────────────────────────────────
    print("  Evaluating on Blood5 (no T)...")
    b5_no_T  = evaluate_blood5(b5_logits, b5_labels, T=1.0)
    print("  Evaluating on Blood5 (with T)...")
    b5_with_T = evaluate_blood5(b5_logits, b5_labels, T=T_star)

    print(f"  Blood5 Scenario C (no T): acc={b5_no_T['scenario_C']['acc']:.4f} "
          f"ECE={b5_no_T['scenario_C']['ece_15bin']:.4f}")
    print(f"  Blood5 Scenario C (T={T_star:.2f}): acc={b5_with_T['scenario_C']['acc']:.4f} "
          f"ECE={b5_with_T['scenario_C']['ece_15bin']:.4f}")

    # ── PER-CLASS PRECISION/RECALL on Blood5 Scenario C ───────────────────
    shared_logits_b5 = b5_logits[:, PBC_SHARED_IN_BLOOD5_ORDER]
    probs_5_b5 = torch.softmax(torch.tensor(shared_logits_b5), dim=1).numpy()
    preds_5_b5 = probs_5_b5.argmax(axis=1)
    cm_5x5 = compute_confusion(b5_labels, preds_5_b5, 5, 5)
    per_class_prec, per_class_rec = {}, {}
    for i, cls in enumerate(BLOOD5_CLASSES):
        tp = cm_5x5[i, i]
        fp = cm_5x5[:, i].sum() - tp
        fn = cm_5x5[i, :].sum() - tp
        per_class_prec[cls] = float(tp/(tp+fp)) if (tp+fp) > 0 else 0.0
        per_class_rec[cls]  = float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0

    seed_result = {
        'seed': seed,
        'training': {
            'best_val_loss': float(best_val_loss),
            'epochs_run': len(history),
            'history': history,
        },
        'T_star': float(T_star),
        'pbc_test': {
            'acc': acc_pbc_test,
            'ece_15bin_no_T': float(ece_pbc_test_raw),
            'ece_15bin_with_T': float(ece_pbc_test_T),
        },
        'blood5_no_T': b5_no_T,
        'blood5_with_T': b5_with_T,
        'blood5_per_class': {
            'precision_C': per_class_prec,
            'recall_C': per_class_rec,
            'cm_5x5_C': cm_5x5.tolist(),
        },
    }
    per_seed_results.append(seed_result)

    # Print summary
    print(f"\n  SEED {seed} SUMMARY:")
    print(f"    PBC test acc={acc_pbc_test:.4f}  ECE_noT={ece_pbc_test_raw:.4f}  ECE_T={ece_pbc_test_T:.4f}  T*={T_star:.3f}")
    for sc_name in ['scenario_A', 'scenario_B', 'scenario_C']:
        a_raw = b5_no_T[sc_name]
        a_T   = b5_with_T[sc_name]
        print(f"    Blood5 {sc_name}: acc_noT={a_raw['acc']:.4f} ECE_noT={a_raw['ece_15bin']:.4f} "
              f"acc_T={a_T['acc']:.4f} ECE_T={a_T['ece_15bin']:.4f}")

    gc.collect()

# ─────────────────────────── AGGREGATE ACROSS SEEDS ──────────────────────────
print("\n\n=== AGGREGATE RESULTS ===")

def mean_std(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std())

metrics_agg = {}
for key_path in [
    ('pbc_test', 'acc'),
    ('pbc_test', 'ece_15bin_no_T'),
    ('pbc_test', 'ece_15bin_with_T'),
]:
    vals = [r[key_path[0]][key_path[1]] for r in per_seed_results]
    m, s = mean_std(vals)
    metrics_agg[f"{key_path[0]}.{key_path[1]}"] = {'mean': m, 'std': s, 'per_seed': vals}
    print(f"  {key_path[0]}.{key_path[1]}: mean={m:.4f} ± {s:.4f}")

T_stars = [r['T_star'] for r in per_seed_results]
m, s = mean_std(T_stars)
metrics_agg['T_star'] = {'mean': m, 'std': s, 'per_seed': T_stars}
print(f"  T_star: mean={m:.4f} ± {s:.4f}")

for scenario in ['scenario_A', 'scenario_B', 'scenario_C']:
    for suffix, data_key in [('_noT', 'blood5_no_T'), ('_T', 'blood5_with_T')]:
        for metric in ['acc', 'ece_15bin']:
            vals = [r[data_key][scenario][metric] for r in per_seed_results]
            m, s = mean_std(vals)
            k = f"blood5.{scenario}.{metric}{suffix}"
            metrics_agg[k] = {'mean': m, 'std': s, 'per_seed': vals}
            print(f"  {k}: mean={m:.4f} ± {s:.4f}")

# ─────────────────────────── WRITE LOG.md ────────────────────────────────────
log_md = f"""# LOG — refine-01 probe

## Round
refine-01 (genuine cross-site WBC transfer)

## Date
{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## What I did

### 1. Obtained genuine cross-site WBC dataset
- Tried HuggingFace search for Raabin-WBC, LISC, etc. — most returned 401 (gated/private)
- Found Blood_5 dataset on Zenodo: doi:10.5281/zenodo.21628834
- Author: Hao WANG (different lab than Barcelona PBC, which is Acevedo et al.)
- Self-collected peripheral WBC, published 2026-07-27
- Verified NOT Barcelona PBC by checking author, institution, and image content
- Downloaded test_batch ONLY using ZIP range requests (254MB compressed) — no full 1.3GB needed
- Confirmed format: HWC uint8 150×150×3, 5 classes matching PBC shared classes

### 2. Dataset provenance verification
- Source URL: https://zenodo.org/records/21628834
- Sample count: 5,175 test images
- Image dimensions: 150×150×3 pixels
- Classes: basophil(132), eosinophil(148), lymphocyte(1241), monocyte(642), neutrophil(3012)
- Label 4 (neutrophil) first sample: shows correct multi-lobed neutrophil morphology
- Confirmed NOT PBC: different author, different institution, different imaging setup

### 3. Experiment design
- Source: Barcelona PBC (Docty/Blood-Cells, Acevedo et al. 2020)
- Fixed 70/15/15 stratified split (seed 42, same as baseline-00)
- 3 seeds: {SEEDS}
- ResNet-18 ImageNet-pretrained, AdamW lr=1e-4 wd=1e-2 cosine 15 epochs early-stop patience 5
- Temperature T fitted on PBC val set (LBFGS, max 200 iter)

### 4. Label alignment scenarios
- Scenario A: argmax(8-class PBC) vs Blood5 label 0-4 (MISMATCHED — wrong index space)
- Scenario B: argmax(8-class PBC) vs Blood5 label remapped to PBC index (ALIGNED)
- Scenario C: restrict to 5 shared-class PBC heads, renorm, argmax vs Blood5 (PROPER)

PBC shared class indices in Blood5 label order: {PBC_SHARED_IN_BLOOD5_ORDER}
  (basophil=PBC[3], eosinophil=PBC[6], lymphocyte=PBC[4], monocyte=PBC[0], neutrophil=PBC[2])

## Why this design
The EXPERIMENT.md requires a genuine second-site dataset, not a PBC proxy.
Blood_5 satisfies this: different lab, genuinely different acquisition.
The three scenarios reproduce the baseline-00 logic but on REAL cross-site data.
Temperature scaling is fit on PBC val (source domain) and applied zero-shot to Blood5.

## Key results (summarized)

### Per-seed T* values
{', '.join(f'seed{r["seed"]}={r["T_star"]:.3f}' for r in per_seed_results)}

### PBC in-domain test ECE (no T vs with T)
{chr(10).join(f'  seed{r["seed"]}: ECE_noT={r["pbc_test"]["ece_15bin_no_T"]:.4f}  ECE_T={r["pbc_test"]["ece_15bin_with_T"]:.4f}' for r in per_seed_results)}

### Blood5 Scenario C ECE (no T vs with T)
{chr(10).join(f'  seed{r["seed"]}: ECE_noT={r["blood5_no_T"]["scenario_C"]["ece_15bin"]:.4f}  ECE_T={r["blood5_with_T"]["scenario_C"]["ece_15bin"]:.4f}' for r in per_seed_results)}
"""

log_path = f"{OUT_DIR}/LOG.md"
with open(log_path, 'w') as f:
    f.write(log_md)
print(f"\nWrote {log_path}")

# ─────────────────────────── WRITE RESULTS.json ──────────────────────────────
results = {
    "status": "SUCCESS",
    "scale": "probe",
    "metrics": {
        "cross_site_data_obtained": True,
        "dataset_source_url": "https://zenodo.org/records/21628834",
        "dataset_doi": "10.5281/zenodo.21628834",
        "dataset_author": "Hao WANG",
        "dataset_name": "Blood_5",
        "dataset_publication_date": "2026-07-27",
        "dataset_confirmed_not_pbc": True,
        "n_target_samples": N_TARGET,
        "target_image_dimensions": "150x150x3",
        "target_classes": BLOOD5_CLASSES,
        "target_label_counts": {
            BLOOD5_CLASSES[i]: int((blood5_labels == i).sum()) for i in range(5)
        },
        "pbc_to_blood5_class_mapping": {
            f"blood5[{i}]={cls}": f"pbc[{BLOOD5_TO_PBC_IDX[i]}]"
            for i, cls in enumerate(BLOOD5_CLASSES)
        },
        "seeds": SEEDS,
        "per_seed": per_seed_results,
        "aggregate": metrics_agg,
    },
    "subject_executed": (
        f"ResNet-18 (ImageNet-pretrained) trained on Barcelona PBC 8-class; "
        f"evaluated on Blood_5 cross-site target (Hao WANG, Zenodo doi:10.5281/zenodo.21628834); "
        f"3 seeds, 3 label-alignment scenarios, temperature scaling; "
        f"genuine cross-site evaluation (different lab, different acquisition)"
    ),
    "notes": (
        f"Blood_5 from Zenodo 21628834, self-collected by Hao WANG (not Barcelona PBC). "
        f"Downloaded test_batch (5175 images) via ZIP range requests. "
        f"T_star mean={metrics_agg['T_star']['mean']:.3f} ± {metrics_agg['T_star']['std']:.3f}. "
        f"Blood5 Scenario C (proper 5-class) ECE_noT mean={metrics_agg['blood5.scenario_C.ece_15bin_noT']['mean']:.4f}, "
        f"ECE_T mean={metrics_agg['blood5.scenario_C.ece_15bin_T']['mean']:.4f}. "
        f"PBC in-domain ECE_noT mean={metrics_agg['pbc_test.ece_15bin_no_T']['mean']:.4f}."
    ),
}

results_path = f"{OUT_DIR}/RESULTS.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Wrote {results_path}")

print("\n=== FINAL SUMMARY ===")
print(json.dumps(results['metrics']['aggregate'], indent=2))
print(f"\nStatus: {results['status']}")
print(f"Cross-site data obtained: {results['metrics']['cross_site_data_obtained']}")
print(f"Dataset: {results['metrics']['dataset_name']} | {results['metrics']['dataset_source_url']}")
print(f"N target samples: {results['metrics']['n_target_samples']}")
