#!/usr/bin/env python3
"""Export the cheap-position head (head_v2): train, tune bias, package via
contract.make_head, dedup-check + sustained-score sim vs incumbents."""
import glob
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
from score_sim import build_epoch_frames_and_frontiers, simulate
from optimize_head import evaluate, optimize_bias, build_fallback

ROSTER = ["deepseek/deepseek-v4-flash", "meta-llama/llama-4-maverick",
          "openai/gpt-5.4-nano", "minimax/minimax-m3", "deepseek/deepseek-v4-pro"]

z = np.load(os.path.join(HERE, "matrix_v4.npz"), allow_pickle=False)
Y = np.where(np.isfinite(z["matrix"].astype(float)),
             (z["matrix"] >= 0.5).astype(float), np.nan)
models_all = [str(m) for m in z["models"]]
qids = [str(q) for q in z["qids"]]
questions = [str(q) for q in z["questions"]]
hidden = np.load(os.path.join(HERE, "hidden_v4.npz"), allow_pickle=False)["H"].astype(np.float64)
prices = load_prices()
manifest = json.load(open(os.path.join(REPO, "data/benchmark_tokens_v1.json")))
frontier = build_epoch_frames_and_frontiers()[sorted(
    build_epoch_frames_and_frontiers())[-1]]
Y_fb = build_fallback(Y, z["bench"], models_all)
tr, va, te = grouped_split(questions, 42)
cols = np.array([models_all.index(m) for m in ROSTER])
W, b, _, _ = fit(hidden, Y[:, cols], tr, va, epochs=300, seed=42)
bias, _ = optimize_bias(W, b, ROSTER, hidden[va], Y[va], Y_fb[va],
                        models_all, prices, manifest, frontier)
b_f = b + bias
acc, cost, dec, acco = evaluate(W, b_f, ROSTER, hidden[te], Y[te], Y_fb[te],
                              models_all, prices, manifest)
print(f"test acc={acc:.3f} obs={acco:.3f} cost={cost:.6f} hr={acc-frontier.accuracy_at(cost):+.4f}")

prov = json.dumps({"src": "reveals e26189-e26219", "seed": 42, "pos": "cheap"})
zh = contract.make_head(W, b_f, ROSTER, manifest, prov)
np.savez(os.path.join(HERE, "head_v2_cheap.npz"), **zh)
print("wrote head_v2_cheap.npz sha256",
      hashlib.sha256(open(os.path.join(HERE, "head_v2_cheap.npz"), 'rb').read()).hexdigest()[:16])

# sustained score across the 31 real epochs
costs = contract.estimated_cost(zh, prices)
q2i = {q: i for i, q in enumerate(qids)}
stream = []
for path in sorted(glob.glob(os.path.join(REPO, "results/reveals-70/*/reveal.json"))):
    rev = json.load(open(path))
    rows = np.array([q2i[q["question_id"]] for q in rev["questions"] if q["question_id"] in q2i])
    if len(rows) < 50:
        continue
    p = contract.predictions(W, b_f, hidden[rows])
    dec = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
    ch = Y[rows, cols[dec]]
    fb = Y_fb[rows, cols[dec]]
    has = np.isfinite(ch)
    acc_e = (ch[has].sum() + fb[~has].sum()) / len(rows)
    stream.append((acc_e * len(rows), len(rows), costs[dec].mean() * len(rows)))
h = simulate(stream, frontier)
print(f"sustained: last={h[-1]['score']:.4f} mean10={np.mean([x['score'] for x in h[-10:]]):.4f} (incumbent ~0.033)")

# dedup vs latest slice
rev = json.load(open(sorted(glob.glob(os.path.join(REPO, "results/reveals-70/*/reveal.json")))[-1]))
rows = np.array([q2i[q["question_id"]] for q in rev["questions"] if q["question_id"] in q2i])
p = contract.predictions(W, b_f, hidden[rows])
ours = np.array(ROSTER)[contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]]
rm = np.array(rev["models"])
ranked = sorted(rev["scores"].items(), key=lambda kv: -(kv[1].get("score") or 0))
for uid, s in ranked[:8]:
    theirs = rm[np.array(rev["routing_decisions"][uid])[:len(rows)]]
    print(f"dedup uid {uid:>4} score {s.get('score',0):.4f} agree {(theirs == ours).mean():.1%}")
