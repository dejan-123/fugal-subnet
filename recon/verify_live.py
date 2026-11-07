#!/usr/bin/env python3
"""Verify head_v2 on REAL OpenRouter calls — one epoch's actual slice.

Reads the key from ~/.fugal/openrouter.key (0600 file), routes all 300
questions of the latest reveal's slice with the real contract math, calls the
chosen models concurrently, grades with the pinned graders, and reports
realized accuracy/cost/headroom vs the simulated estimate.

[PAID ~$0.10] — the whole point is real calls on real models.
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fugal_subnet.api import call_model, load_prices
from fugal_subnet.api import SpendTracker
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task
from fugal_subnet.pricing import question_input_tokens
from fugal_subnet.frontier import build_frontier
from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
from fugal_subnet.reference_frame import load_bootstrap, accumulate_exploration
from score_sim import epoch_samples

KEYFILE = os.path.expanduser("~/.fugal/openrouter.key")
ENVFILE = os.path.join(REPO, ".env")
BUDGET = 1.00  # hard cap — aborts runaway spend


def main():
    key = ""
    if os.path.exists(KEYFILE) and os.path.getsize(KEYFILE) > 0:
        key = open(KEYFILE).read().strip()
    elif os.path.exists(ENVFILE):
        for line in open(ENVFILE):
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or key == "sk-or-paste-your-key-here":
        sys.exit("no key: paste it into ~/.fugal/openrouter.key or .env")
    os.environ["OPENROUTER_API_KEY"] = key
    prices = load_prices()
    manifest = json.load(open(os.path.join(REPO, "data/benchmark_tokens_v1.json")))

    rev = json.load(open(sorted(glob.glob(os.path.join(
        REPO, "results/reveals-70/*/reveal.json")))[-1]))
    questions = rev["questions"]
    print(f"epoch {rev['epoch_id']}: {len(questions)} questions")

    zh = np.load(sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE, "head_v2_cheap.npz"), allow_pickle=False)
    roster = [str(m) for m in zh["models"]]
    W, b = zh["W"], zh["b"]
    if "mean_in_tokens" in zh.files:
        costs = contract.estimated_cost(zh, prices)
    else:
        mrow = {r["id"]: r for r in manifest["models"]}
        costs = np.array([prices[m][0]*mrow[m]["mean_in_tokens"] + prices[m][1]*mrow[m]["mean_out_tokens"] for m in roster])

    qz = np.load(os.path.join(HERE, "matrix_v4.npz"), allow_pickle=False)
    q2i = {str(q): i for i, q in enumerate(qz["qids"])}
    hidden = np.load(os.path.join(HERE, "hidden_v4.npz"), allow_pickle=False)["H"].astype(np.float64)
    rows = np.array([q2i.get(q["question_id"], -1) for q in questions])
    assert (rows >= 0).all()

    p = contract.predictions(W, b, hidden[rows])
    dec = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
    picks = [roster[i] for i in dec]
    est_cost = float(costs[dec].mean())
    from collections import Counter
    print("routing:", dict(Counter(picks)), f"est_cost=${est_cost:.6f}/q")

    tracker = SpendTracker(budget_cap_usd=BUDGET)
    results = [None] * len(questions)

    def work(i):
        m, q = picks[i], questions[i]
        try:
            text, pt, ct = call_model(m, q["prompt"], tracker=tracker,
                                      prices=prices, live=True, max_tokens=2048)
            return i, m, text, pt, ct, False
        except Exception as e:
            return i, m, "", 0, 0, str(e)[:60]

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, m, text, pt, ct, err in ex.map(work, range(len(questions))):
            results[i] = (m, text, pt, ct, err)

    spent = tracker.committed_cost_usd if hasattr(tracker, "committed_cost_usd") else float("nan")
    n_ok = n_fail = 0
    real_cost = 0.0
    per_model = {}
    for i, (m, text, pt, ct, err) in enumerate(results):
        cost_i = prices[m][0] * pt + prices[m][1] * ct
        real_cost += cost_i
        if err:
            n_fail += 1
            ok = 0
        else:
            task = build_grader_task(questions[i])
            ok = grade(task, text)
            n_ok += ok
        a, c = per_model.get(m, [0, 0])
        per_model[m] = [a + ok, c + 1]

    acc = n_ok / len(questions)
    cost_q = real_cost / len(questions)

    # real frontier for this slice
    frame = load_bootstrap()
    for path in sorted(glob.glob(os.path.join(REPO, "results/reveals-70/*/reveal.json"))):
        frame = accumulate_exploration(frame, epoch_samples(json.load(open(path))))
    slice_tok = sum(question_input_tokens(q["prompt"]) for q in questions)
    frontier = build_frontier(frame, prices, slice_tok, len(questions),
                            FRAME_DEFAULT_COMPLETION_TOKENS)
    ref = frontier.accuracy_at(cost_q)

    print(f"\n=== REALIZED (paid) ===")
    print(f"accuracy   = {acc:.3f}  ({n_ok}/{len(questions)} correct, {n_fail} failed calls)")
    print(f"cost/q     = ${cost_q:.6f}  total ${real_cost:.4f}")
    print(f"frontier   = {ref:.3f}")
    print(f"headroom   = {acc - ref:+.4f}   (estimated +0.044, incumbent ~+0.033)")
    # persist real labels for retraining
    out = np.zeros(len(questions), dtype=np.float32)
    for i,(m,text,pt,ct,err) in enumerate(results):
        out[i] = -1 if err else float(grade(build_grader_task(questions[i]), text))
    np.savez(os.path.join(HERE,'live_labels.npz'),
             qids=np.array([q['question_id'] for q in questions]),
             picks=np.array(picks), outcome=out)
    print("saved live_labels.npz")
    print("per-model:")
    for m, (c_, n_) in sorted(per_model.items()):
        print(f"  {m:<40} {c_:3d}/{n_:3d} = {c_/n_:.3f}")


if __name__ == "__main__":
    main()
