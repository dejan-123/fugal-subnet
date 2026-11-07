#!/usr/bin/env python3
"""Beat-the-top-miner run: ensemble seeds, oracle-imitation targets, and
score on REAL epoch slices through the actual evidence simulator."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fugal_subnet.api import load_prices
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import fit, grouped_split
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from score_sim import build_epoch_frames_and_frontiers, simulate
from optimize_head import evaluate, optimize_bias, build_fallback

ROSTER = ["deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
          "meta-llama/llama-4-maverick", "anthropic/claude-opus-4.8",
          "openai/gpt-5.5"]

z = np.load(os.path.join(HERE, "matrix_v4.npz"), allow_pickle=False)
Ym = z["matrix"].astype(float)
Y = np.where(np.isfinite(Ym), (Ym >= 0.5).astype(float), np.nan)
models_all = [str(m) for m in z["models"]]
qids = [str(q) for q in z["qids"]]
questions = [str(q) for q in z["questions"]]
bench = z["bench"]
hidden = np.load(os.path.join(HERE, "hidden_v4.npz"), allow_pickle=False)["H"].astype(np.float64)
prices = load_prices()
manifest = json.load(open(os.path.join(REPO, "data/benchmark_tokens_v1.json")))
frontiers = build_epoch_frames_and_frontiers()
latest_frontier = frontiers[sorted(frontiers)[-1]]
Y_fb = build_fallback(Y, bench, models_all)
cols = np.array([models_all.index(m) for m in ROSTER])
y_sub = Y[:, cols]
train_idx, val_idx, test_idx = grouped_split(questions, 42)

# ---- A: ensemble of 5 seeds ----
Ws, bs = [], []
for seed in (1, 7, 13, 42, 99):
    W, b, _, _ = fit(hidden, y_sub, train_idx, val_idx, epochs=250, seed=seed)
    Ws.append(W); bs.append(b)
W_e, b_e = np.mean(Ws, 0), np.mean(bs, 0)
bias_e, _ = optimize_bias(W_e, b_e, ROSTER, hidden[val_idx], Y[val_idx],
                          Y_fb[val_idx], models_all, prices, manifest, latest_frontier)
acc, cost, _, acco = evaluate(W_e, b_e + bias_e, ROSTER, hidden[test_idx],
                            Y[test_idx], Y_fb[test_idx], models_all, prices, manifest)
print(f"ENSEMBLE  acc={acc:.3f} (obs {acco:.3f}) cost={cost:.5f} "
      f"hr={acc-latest_frontier.accuracy_at(cost):+.4f}")

# ---- B: oracle-imitation targets ----
# target y2[q,m] = 1 if m is the cheapest observed-correct model for q else 0.
costs_est = contract.estimated_cost(
    {"models": np.array(ROSTER),
     "mean_in_tokens": np.array([{r["id"]: r for r in manifest["models"]}[m]["mean_in_tokens"] for m in ROSTER]),
     "mean_out_tokens": np.array([{r["id"]: r for r in manifest["models"]}[m]["mean_out_tokens"] for m in ROSTER])},
    prices)
y2 = np.zeros_like(y_sub)
for i in range(len(y_sub)):
    ok = [j for j in range(len(ROSTER)) if np.isfinite(y_sub[i, j]) and y_sub[i, j] == 1]
    if ok:
        m = min(ok, key=lambda j: costs_est[j])
        y2[i, m] = 1.0
    else:
        y2[i, :] = np.nan
W2, b2, _, _ = fit(hidden, y2, train_idx, val_idx, epochs=250, seed=42)
bias2, _ = optimize_bias(W2, b2, ROSTER, hidden[val_idx], Y[val_idx],
                         Y_fb[val_idx], models_all, prices, manifest, latest_frontier)
acc2, cost2, _, acco2 = evaluate(W2, b2 + bias2, ROSTER, hidden[test_idx],
                               Y[test_idx], Y_fb[test_idx], models_all, prices, manifest)
print(f"ORACLE    acc={acc2:.3f} (obs {acco2:.3f}) cost={cost2:.5f} "
      f"hr={acc2-latest_frontier.accuracy_at(cost2):+.4f}")

# ---- C: per-epoch slice simulation for the best candidate ----
# replay each epoch: our decisions on ITS 300 questions -> per-epoch acc/cost
def per_epoch_stream(W, b):
    q2i = {q: i for i, q in enumerate(qids)}
    stream = []
    for path in sorted(glob.glob(os.path.join(REPO, "results/reveals-70/*/reveal.json"))):
        rev = json.load(open(path))
        rows = np.array([q2i[q["question_id"]] for q in rev["questions"]
                         if q["question_id"] in q2i])
        if len(rows) < 50:
            continue
        p = contract.predictions(W, b, hidden[rows])
        dec = contract.rank(p, costs_est, BENCHMARK_LAMBDA)[:, 0]
        chosen = Y[rows[:, None], cols[dec]].diagonal() if False else Y[rows, cols[dec]]
        fb = Y_fb[rows, cols[dec]]
        has = np.isfinite(chosen)
        acc_e = (chosen[has].sum() + fb[~has].sum()) / len(rows)
        cost_e = costs_est[dec].mean()
        stream.append((acc_e * len(rows), len(rows), cost_e * len(rows)))
    return stream

for name, (W, b) in {"ensemble": (W_e, b_e + bias_e),
                     "oracle": (W2, b2 + bias2)}.items():
    stream = per_epoch_stream(W, b)
    h = simulate(stream, latest_frontier)
    sc = [x["score"] for x in h]
    print(f"{name:>9}: sustained score over 31 epochs: "
          f"last={sc[-1]:.4f} mean(last10)={np.mean(sc[-10:]):.4f} "
          f"(incumbent ~0.033)")
