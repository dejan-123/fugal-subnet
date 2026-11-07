#!/usr/bin/env python3
"""Fill the 73% missing label cells via low-rank factorization (+biases),
then distill the richer predictions into the linear head's W,b.

Y[q,m] ≈ mu + a_q + c_m + u_q . v_m   (rank k, fit on observed cells only)
The imputed matrix predicts how EVERY model does on EVERY question — the
signal routed-only labels cannot give.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fugal_subnet.api import load_prices
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import grouped_split
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from score_sim import build_epoch_frames_and_frontiers
from optimize_head import evaluate, optimize_bias, build_fallback

K = 6
EPOCHS = 400
LR = 0.05

z = np.load(os.path.join(HERE, "matrix_v4.npz"), allow_pickle=False)
Ym = z["matrix"].astype(float)
Y = np.where(np.isfinite(Ym), (Ym >= 0.5).astype(float), np.nan)
models_all = [str(m) for m in z["models"]]
questions = [str(q) for q in z["questions"]]
qids = [str(q) for q in z["qids"]]
bench = z["bench"]
hidden = np.load(os.path.join(HERE, "hidden_v4.npz"), allow_pickle=False)["H"].astype(np.float64)
prices = load_prices()
manifest = json.load(open(os.path.join(REPO, "data/benchmark_tokens_v1.json")))
frontiers = build_epoch_frames_and_frontiers()
frontier = frontiers[sorted(frontiers)[-1]]
Y_fb = build_fallback(Y, bench, models_all)
train_idx, val_idx, test_idx = grouped_split(questions, 42)

# ---- factorize (fit on ALL observed cells, train rows only for honesty) ----
mask = np.isfinite(Y)
sub = Y.copy()
obs_rows = np.unique(np.concatenate([train_idx, val_idx]))  # don't fit on test
mask_fit = mask.copy(); mask_fit[test_idx] = False

mu = np.nanmean(Y[obs_rows])
a = torch.zeros(len(Y)); c = torch.zeros(len(models_all))
U = torch.randn(len(Y), K) * 0.1
V = torch.randn(len(models_all), K) * 0.1
a.requires_grad_(); c.requires_grad_(); U.requires_grad_(); V.requires_grad_()
opt = torch.optim.Adam([a, c, U, V], lr=LR)
Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
Mt = torch.tensor(mask_fit, dtype=torch.float32)
for e in range(EPOCHS):
    pred = mu + a[:, None] + c[None, :] + U @ V.T
    loss = (((pred - Yt) * Mt) ** 2).sum() / Mt.sum()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    Yfill = (mu + a[:, None] + c[None, :] + U @ V.T).numpy()
resid = ((np.nan_to_num(Yfill - np.nan_to_num(Y)))[mask_fit] ** 2).mean() ** 0.5
print(f"factorization RMSE on observed cells: {resid:.3f}")

# blend: observed cells keep real labels, unobserved use factorization
Yhat = np.where(mask, Y, Yfill)

# ---- for each roster, train a head whose DECISIONS imitate argmax(Yhat - kc)
def fit_soft(hidden, target_p, train, val, epochs=300, seed=42):
    """BCE against soft probability targets (all cells, since Yhat is dense)."""
    torch.manual_seed(seed)
    h = torch.tensor(hidden, dtype=torch.float32)
    t = torch.tensor(np.clip(target_p, 0.02, 0.98), dtype=torch.float32)
    W = torch.zeros((target_p.shape[1], 1024), requires_grad=True)
    b = torch.zeros(target_p.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=0.01, weight_decay=1e-4)
    best, ck = 1e9, None
    for e in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            h[train] @ W.T + b, t[train])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = torch.nn.functional.binary_cross_entropy_with_logits(
                h[val] @ W.T + b, t[val]).item()
        if vl < best: best, ck = vl, (W.detach().numpy().copy(), b.detach().numpy().copy())
    return ck

ROSTERS = {
    "cheap-pos {flash,llama,nano,minimax,opus}": [
        "deepseek/deepseek-v4-flash", "meta-llama/llama-4-maverick",
        "openai/gpt-5.4-nano", "minimax/minimax-m3", "anthropic/claude-opus-4.8"],
    "incumbent {flash,qplus,llama,opus,5.5}": [
        "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
        "meta-llama/llama-4-maverick", "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
}
for name, roster in ROSTERS.items():
    cols = np.array([models_all.index(m) for m in roster])
    W, b = fit_soft(hidden, Yhat[:, cols], train_idx, val_idx)
    bias, _ = optimize_bias(W, b, roster, hidden[val_idx], Y[val_idx],
                            Y_fb[val_idx], models_all, prices, manifest, frontier)
    acc, cost, _, acco = evaluate(W, b + bias, roster, hidden[test_idx],
                                Y[test_idx], Y_fb[test_idx], models_all, prices, manifest)
    print(f"{name:<38} acc={acc:.3f} (obs {acco:.3f}) cost={cost:.5f} "
          f"hr={acc - frontier.accuracy_at(cost):+.4f}")
