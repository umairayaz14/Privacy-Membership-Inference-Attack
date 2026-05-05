import os
import sys
import torch
import pandas as pd
import numpy as np
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from scipy.stats import norm as scipy_norm
from copy import deepcopy

# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "1f7a1953a1d6aae8e315f6ed9201dadc"
TASK_ID = "01-mia"  #DONOT CHANGE

# Shadow model training config
N_SHADOWS  = 8
N_EPOCHS   = 20
BATCH_SIZE = 128
LR         = 0.05

# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
ckpt = torch.load(MODEL_PATH, map_location="cpu")
NUM_CLASSES = ckpt["fc.weight"].shape[0]
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# # create random submission (remove this later or it will rewrite your actual submission)
# print("Creating random submission...")
# ids = [str(i) for i in priv_ds.ids]

# df = pd.DataFrame({
#     "id": ids,
#     "score": [random.random() for _ in ids]
# })

# df.to_csv(OUTPUT_CSV, index=False)
# print("Saved:", OUTPUT_CSV)


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)


# # 8 shadow models try later with more models and epochs
# ==========================================
# HELPER: build a fresh shadow model
# ==========================================
def make_model():
    m = resnet18(weights=None)
    m.conv1  = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, NUM_CLASSES)
    return m
 
 
# ==========================================
# HELPER: get per-sample logit-scaled loss φ
# φ = log(p_correct / (1 - p_correct))
# members → high φ, non-members → low φ
# ==========================================
def get_phi(model, loader):
    model.eval()
    phi_list  = []
    ids_list  = []
    labs_list = []
 
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                batch_ids, imgs, labels, _ = batch
            else:
                batch_ids, imgs, labels = batch
 
            imgs   = imgs.to(device)
            labels = labels.to(device)
 
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            conf   = probs[torch.arange(len(labels)), labels].clamp(1e-7, 1 - 1e-7)
            phi    = torch.log(conf / (1 - conf))
 
            phi_list.extend(phi.cpu().numpy())
            labs_list.extend(labels.cpu().numpy())
 
            if torch.is_tensor(batch_ids):
                ids_list.extend([str(i.item()) for i in batch_ids])
            else:
                ids_list.extend([str(i) for i in batch_ids])
 
    return np.array(phi_list), ids_list, np.array(labs_list, dtype=int)
 
 
# ==========================================
# HELPER: train one shadow model
# ==========================================
def train_shadow(train_indices, epochs=N_EPOCHS):
    shadow_ds = deepcopy(pub_ds)
    shadow_ds.transform = train_transform
 
    subset = Subset(shadow_ds, train_indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
 
    m   = make_model().to(device)
    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
 
    m.train()
    for epoch in range(epochs):
        for batch in loader:
            if len(batch) == 4:
                _, imgs, labels, _ = batch
            else:
                _, imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            F.cross_entropy(m(imgs), labels).backward()
            opt.step()
        sched.step()
 
    m.eval()
    return m
 
 
# ==========================================
# STEP 1 — Train shadow models, collect φ
# ==========================================
print(f"\n[1/4] Training {N_SHADOWS} shadow models...")
 
N_pub   = len(pub_ds)
phi_in  = [[] for _ in range(N_pub)]   # φ when sample WAS in training
phi_out = [[] for _ in range(N_pub)]   # φ when sample was NOT in training
 
eval_pub_ds = deepcopy(pub_ds)
eval_pub_ds.transform = transform
 
for s in range(N_SHADOWS):
    print(f"  Shadow {s+1}/{N_SHADOWS} ...", end="", flush=True)
 
    perm    = np.random.permutation(N_pub)
    in_idx  = set(perm[:N_pub // 2].tolist())
    in_list = sorted(in_idx)
 
    shadow = train_shadow(in_list)
 
    full_loader = DataLoader(eval_pub_ds, batch_size=BATCH_SIZE, shuffle=False)
    phi_all, _, _ = get_phi(shadow, full_loader)
 
    for i, phi in enumerate(phi_all):
        if i in in_idx:
            phi_in[i].append(phi)
        else:
            phi_out[i].append(phi)
 
    print(" done")
    del shadow
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
 
# ==========================================
# STEP 2 — Score pub.pt with LiRA likelihood ratio
# ==========================================
print("\n[2/4] Scoring pub.pt for local evaluation...")
 
full_loader = DataLoader(eval_pub_ds, batch_size=BATCH_SIZE, shuffle=False)
phi_target_pub, pub_ids, pub_labels = get_phi(model, full_loader)
y_pub = np.array(pub_ds.membership)
 
pub_scores = np.zeros(N_pub)
for i in range(N_pub):
    phi      = phi_target_pub[i]
    in_vals  = np.array(phi_in[i])
    out_vals = np.array(phi_out[i])
 
    if len(in_vals) < 2 or len(out_vals) < 2:
        pub_scores[i] = phi   # fallback
        continue
 
    log_p_in  = scipy_norm.logpdf(phi, in_vals.mean(),  in_vals.std()  + 1e-8)
    log_p_out = scipy_norm.logpdf(phi, out_vals.mean(), out_vals.std() + 1e-8)
    pub_scores[i] = log_p_in - log_p_out
 
fpr, tpr, _ = roc_curve(y_pub, pub_scores)
valid = tpr[fpr <= 0.05]
local_tpr = valid[-1] if len(valid) > 0 else 0.0
print(f"  Local TPR@5%FPR (LiRA): {local_tpr:.4f}")
 
 
# ==========================================
# STEP 3 — Score priv.pt
# ==========================================
print("\n[3/4] Scoring priv.pt...")
 
# Build global OUT distribution from shadow runs (non-member reference)
all_out_phi    = [phi for out_vals in phi_out for phi in out_vals]
all_out_phi    = np.array(all_out_phi)
mu_out_global  = all_out_phi.mean()
std_out_global = all_out_phi.std() + 1e-8
 
# Per-class OUT distribution for class-calibrated scoring
class_mu_out  = np.zeros(NUM_CLASSES)
class_std_out = np.ones(NUM_CLASSES)
for c in range(NUM_CLASSES):
    c_phi = [phi for i, out_vals in enumerate(phi_out)
             for phi in out_vals if pub_labels[i] == c]
    if len(c_phi) > 1:
        class_mu_out[c]  = np.mean(c_phi)
        class_std_out[c] = np.std(c_phi) + 1e-8
 
priv_ds.membership = [0] * len(priv_ds)
eval_priv_ds = deepcopy(priv_ds)
eval_priv_ds.transform = transform
 
priv_loader = DataLoader(eval_priv_ds, batch_size=BATCH_SIZE, shuffle=False)
phi_target_priv, priv_ids, priv_labels = get_phi(model, priv_loader)
 
# Global calibration
global_scores = (phi_target_priv - mu_out_global) / std_out_global
 
# Per-class calibration
class_scores = np.array([
    (phi_target_priv[i] - class_mu_out[priv_labels[i]]) / class_std_out[priv_labels[i]]
    for i in range(len(priv_ids))
])
 
# Ensemble both calibrations
raw_scores  = (global_scores + class_scores) / 2.0
lo, hi      = raw_scores.min(), raw_scores.max()
final_scores = (raw_scores - lo) / (hi - lo + 1e-9)
 
 
# ==========================================
# STEP 4 — Save submission
# ==========================================
print("\n[4/4] Saving submission...")
df = pd.DataFrame({"id": priv_ids, "score": final_scores.tolist()})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}  ({len(df)} rows)")

# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)