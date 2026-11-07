#!/usr/bin/env python3
"""Train the interim head on trainable rows (holdout.npz mask), freeze it.

Roster per approved plan: {ds-flash, qwen3.7-plus} workhorse +
{claude-opus-4.8, gpt-5.5} escalation. Downweights contaminated 0-cells
(routed-vote cells weight 0.5, exploration/paid cells 1.0). Saves
recon/interim_head.npz — routes are FROZEN before any label buying (1.5b).
"""
import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from common import load_matrix, load_hidden, route_costs
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import grouped_split

ROSTER = ["deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
          "anthropic/claude-opus-4.8", "openai/gpt-5.5"]

m = load_matrix()
ho = np.load(os.path.join(HERE, "holdout.npz"), allow_pickle=False)
trainable = ho["trainable_mask"]
idx_all = np.where(trainable)[0]

models_all = m["models"]
cols = np.array([models_all.index(x) for x in ROSTER])
Y = m["Ybin"][:, cols]                      # (N, 4), NaN missing
NR = m["counts"][:, cols]                   # routed-vote counts
NE = m["expl_counts"][:, cols]

# per-cell sample weight: exploration 1.0, routed-only 0.5 (contaminated bias)
Wgt = np.where(np.isfinite(Y), np.where(NE > 0, 1.0, 0.5), 0.0)

hidden = load_hidden()
qs = [m["questions"][i] for i in idx_all]
tr_g, va_g, _ = grouped_split(qs, 42)      # grouped on question text
tr = idx_all[tr_g]; va = idx_all[va_g]

H = torch.tensor(hidden, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)
Wt = torch.tensor(Wgt, dtype=torch.float32)
torch.manual_seed(42)
Wh = torch.zeros((len(ROSTER), 1024), requires_grad=True)
bh = torch.zeros(len(ROSTER), requires_grad=True)
opt = torch.optim.AdamW([Wh, bh], lr=0.01, weight_decay=1e-4)

def wbce(rows):
    logits = H[rows] @ Wh.T + bh
    mask = torch.isfinite(Yt[rows]) & (Wt[rows] > 0)
    if not mask.any():
        raise ValueError("no labels")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], Yt[rows][mask], reduction="none")
    return (loss * Wt[rows][mask]).sum() / Wt[rows][mask].sum()

best, ck = 1e9, None
for ep in range(300):
    l = wbce(tr)
    opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        v = wbce(va).item()
    if v < best:
        best, ck = v, (Wh.detach().numpy().copy(), bh.detach().numpy().copy(), ep)
W, b, ep = ck
print(f"interim head: epoch {ep}, val wbce {best:.4f}")

manifest = json.load(open(os.path.join(REPO, "data", "benchmark_tokens_v1.json")))
prov = json.dumps({"src": "interim v9 plan", "roster": ROSTER, "seed": 42})
zh = contract.make_head(W.astype(np.float32), b.astype(np.float32), ROSTER, manifest, prov)
np.savez(os.path.join(HERE, "interim_head.npz"), **zh)
print("wrote interim_head.npz (FROZEN before label buying)")
