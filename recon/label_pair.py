#!/usr/bin/env python3
"""Self-label (qid, model) cells with our own paid calls.

Same call path as the TEE harness: call_model(live=True, max_tokens=2048)
+ graders.grade(build_grader_task(q), text). Failed calls are recorded as
failed=True with outcome NaN (never 0) — matching reference_frame semantics.

Checkpointed: re-runs skip (qid, model) pairs already labeled. Hard budget cap.

Usage:
  python label_pair.py jobs.npz out_labels.npz [--budget 5.0]
  jobs.npz: qids (U), models (U), aligned same-length arrays.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from common import pool_lookup
from fugal_subnet.api import call_model, load_prices, SpendTracker, BudgetExceeded
from fugal_subnet.graders import grade
from fugal_subnet.grading_task import build_grader_task

KEYFILE = os.path.expanduser("~/.fugal/openrouter.key")


def get_key():
    if os.path.exists(KEYFILE) and os.path.getsize(KEYFILE) > 0:
        return open(KEYFILE).read().strip()
    envf = os.path.join(REPO, ".env")
    if os.path.exists(envf):
        for line in open(envf):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = get_key()
    if not key or key == "sk-or-paste-your-key-here":
        sys.exit("no OpenRouter key: ~/.fugal/openrouter.key or .env")
    os.environ["OPENROUTER_API_KEY"] = key
    prices = load_prices()
    pool = pool_lookup()

    jobs = np.load(args.jobs, allow_pickle=False)
    jq = [str(q) for q in jobs["qids"]]
    jm = [str(m) for m in jobs["models"]]

    done = {}
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=False)
        for q, m, o, f, pt, ct in zip(z["qids"], z["models"], z["outcome"],
                                      z["failed"], z["ptok"], z["ctok"]):
            done[(str(q), str(m))] = (float(o), bool(f), int(pt), int(ct))
        print(f"resuming: {len(done)} already labeled")

    todo = [(q, m) for q, m in zip(jq, jm) if (q, m) not in done]
    print(f"jobs: {len(todo)} to label, budget cap ${args.budget}")
    tracker = SpendTracker(budget_cap_usd=args.budget)

    def work(qm):
        qid, m = qm
        q = pool.get(qid)
        if q is None:
            return qid, m, np.nan, True, 0, 0
        try:
            text, pt, ct = call_model(m, q["prompt"], tracker=tracker,
                                      prices=prices, live=True, max_tokens=2048)
            ok = bool(grade(build_grader_task(q), text))
            return qid, m, float(ok), False, pt, ct
        except BudgetExceeded:
            raise
        except Exception as e:
            print(f"fail {m} {qid}: {type(e).__name__}", flush=True)
            return qid, m, np.nan, True, 0, 0

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for qid, m, o, f, pt, ct in ex.map(work, todo):
            done[(qid, m)] = (o, f, pt, ct)
            n += 1
            if n % 25 == 0 or n == len(todo):
                keys = sorted(done)
                np.savez(args.out,
                         qids=np.array([k[0] for k in keys]),
                         models=np.array([k[1] for k in keys]),
                         outcome=np.array([done[k][0] for k in keys]),
                         failed=np.array([done[k][1] for k in keys]),
                         ptok=np.array([done[k][2] for k in keys]),
                         ctok=np.array([done[k][3] for k in keys]))
                spent = tracker.committed_cost_usd
                print(f"{n}/{len(todo)} spent=${spent:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done. spent=${tracker.committed_cost_usd:.3f}")


if __name__ == "__main__":
    main()
