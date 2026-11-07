#!/usr/bin/env python3
"""Train the winning roster, optimize its bias for headroom, package a
deployable head.npz via contract.make_head, and dedup-check it against the
latest epoch's published routing decisions (cached embeddings — no recompute)."""
import json
import os
import sys
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fugal_subnet.api import load_prices
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import fit, grouped_split
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from score_sim import build_epoch_frames_and_frontiers
from optimize_head import evaluate, optimize_bias, build_fallback

ROSTER = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.7-plus",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
]

z = np.load(os.path.join(HERE, "matrix_v4.npz"), allow_pickle=False)
Y = np.where(np.isfinite(z["matrix"].astype(float)),
             (z["matrix"] >= 0.5).astype(float), np.nan)
models_all = [str(m) for m in z["models"]]
qids = [str(q) for q in z["qids"]]
questions = [str(q) for q in z["questions"]]
bench = z["bench"]
hidden = np.load(os.path.join(HERE, "hidden_v4.npz"), allow_pickle=False)["H"].astype(np.float64)
prices = load_prices()
manifest = json.load(open(os.path.join(REPO, "data/benchmark_tokens_v1.json")))
frontiers = build_epoch_frames_and_frontiers()
frontier = frontiers[sorted(frontiers)[-1]]
Y_fb = build_fallback(Y, bench, models_all)

train_idx, val_idx, test_idx = grouped_split(questions, 42)
cols = np.array([models_all.index(m) for m in ROSTER])
y = Y[:, cols]
W, b, ep, hist = fit(hidden, y, train_idx, val_idx, epochs=300, seed=42)
bias, hr_val = optimize_bias(W, b, ROSTER, hidden[val_idx], Y[val_idx],
                             Y_fb[val_idx], models_all, prices, manifest, frontier)
b_final = b + bias
acc_t, cost_t, dec_t, obs_t = evaluate(W, b_final, ROSTER, hidden[test_idx],
                                       Y[test_idx], Y_fb[test_idx], models_all,
                                       prices, manifest)
print(f"test: acc={acc_t:.3f} (obs-bound {obs_t:.3f}) cost={cost_t:.5f} "
      f"headroom={acc_t - frontier.accuracy_at(cost_t):+.4f}")

# package the head
prov = json.dumps({"source": "recon reveals 189-219", "seed": 42,
                   "bias_tune": np.round(bias, 3).tolist()})
zhead = contract.make_head(W, b_final, ROSTER, manifest, prov)
out = os.path.join(HERE, "head_v1.npz")
np.savez(out, **zhead)
print("wrote", out, os.path.getsize(out), "bytes",
      "sha256", hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])

# dedup check on latest slice using cached hidden states
import glob
reveal = json.load(open(sorted(glob.glob(os.path.join(
    REPO, "results/reveals-70/*/reveal.json")))[-1]))
slice_qids = [q["question_id"] for q in reveal["questions"]]
q2i = {q: i for i, q in enumerate(qids)}
rows = np.array([q2i[q] for q in slice_qids if q in q2i])
print(f"slice {reveal['epoch_id']}: {len(rows)}/300 questions in matrix")
h_slice = hidden[rows]
costs = contract.estimated_cost(zhead, prices)
p = contract.predictions(W, b_final, h_slice)
ours = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
ours_models = np.array(ROSTER)[ours]
rev_models = np.array(reveal["models"])
ranked = sorted(reveal["scores"].items(), key=lambda kv: -(kv[1].get("score") or 0))
print(f"{'uid':>4} {'score':>7} {'agree':>6}")
for uid, s in ranked[:12]:
    dec = np.array(reveal["routing_decisions"][uid])[:len(rows)]
    theirs = rev_models[dec]
    print(f"{uid:>4} {s.get('score',0):7.4f} {(theirs == ours_models).mean():6.1%}")
