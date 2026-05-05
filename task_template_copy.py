import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse
import csv
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

# config
#BASE = Path(__file__).parent
PUB_PATH = "C:/Users/NARESHSARATHY S/Downloads/TML Solution/pub.pt"
PRIV_PATH = "C:/Users/NARESHSARATHY S/Downloads/TML Solution/priv.pt"
MODEL_PATH = "C:/Users/NARESHSARATHY S/Downloads/TML Solution/model.pt"
OUTPUT_CSV = Path("C:/Users/NARESHSARATHY S/Downloads/TML Solution/submission.csv")

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "YOUR API KEY"
TASK_ID = "01-mia"  #DONOT CHANGE



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

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

#From here LiRA is executed
#LiRA ATTACK
print("Fitting per-class LiRA distributions...")
member_scores = {i: [] for i in range(9)}
nonmember_scores = {i: [] for i in range(9)}
with torch.no_grad():
    for id_, img, label, membership in pub_ds:
        img = img.unsqueeze(0)
        logits = model(img)
        # margin score
        top2 = torch.topk(logits, 2).values
        score = (top2[0, 0] - top2[0, 1]).item()
        if membership == 1:
            member_scores[label].append(score)
        else:
            nonmember_scores[label].append(score)

#Gaussian
params = {}
for c in range(9):
    mem = np.array(member_scores[c])
    nonmem = np.array(nonmember_scores[c])
    mu_in, std_in = mem.mean(), mem.std() + 1e-6
    mu_out, std_out = nonmem.mean(), nonmem.std() + 1e-6
    params[c] = (mu_in, std_in, mu_out, std_out)
    print(f"Class {c}: in({mu_in:.4f},{std_in:.4f}) out({mu_out:.4f},{std_out:.4f})")
print("Generating LiRA...")
ids = []
scores = []
with torch.no_grad():
    for id_, img, label, _ in priv_ds:
        img = img.unsqueeze(0)
        logits = model(img)
        top2 = torch.topk(logits, 2).values
        score = (top2[0, 0] - top2[0, 1]).item()
        mu_in, std_in, mu_out, std_out = params[label]
        p_in = norm.pdf(score, mu_in, std_in)
        p_out = norm.pdf(score, mu_out, std_out)
        lira_score = np.log(p_in + 1e-12) - np.log(p_out + 1e-12)
        ids.append(str(id_))
        scores.append(lira_score)
scores_np = np.array(scores)
low, high = np.percentile(scores_np, [1, 99])
scores_np = np.clip(scores_np, low, high)
scores_np = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-12)

df = pd.DataFrame({
    "id": ids,
    "score": scores_np
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)

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
