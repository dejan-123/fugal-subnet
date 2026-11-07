#!/usr/bin/env python3
"""Assemble the training label matrix from all published reveals — v2.

Rows: every distinct question seen ANYWHERE (scored slice + exploration
assignments), prompts pulled from the pinned pool for exploration-only ids.
Cols: the 17 pool models.
Cell: mean correctness over all observed answers.

Label types tracked separately in `counts`/`expl_counts` so training can
weight unbiased (exploration) vs head-selected (routed) evidence.
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

REVEALS = sorted(glob.glob(os.path.join(REPO, "results/reveals-70/*/reveal.json")))
POOL = os.path.join(REPO, "data/benchmarks/pool.json.gz")
OUT = os.path.join(HERE, "matrix_v4.npz")


def main():
    pool = {q["question_id"]: q for q in json.load(gzip.open(POOL))}
    print(f"pool: {len(pool)} questions")

    qid2prompt = {}
    qid2bench = {}
    votes = defaultdict(lambda: [0.0, 0])          # routed/scored calls
    evotes = defaultdict(lambda: [0.0, 0])         # exploration samples
    models_seen = []

    for path in REVEALS:
        d = json.load(open(path))
        for m in d["models"]:
            if m not in models_seen:
                models_seen.append(m)
        M = np.array(d["matrix"], dtype=float)
        for r, q in enumerate(d["questions"]):
            qid = q["question_id"]
            qid2prompt[qid] = q.get("prompt", "")
            qid2bench[qid] = q.get("benchmark", "?")
            for j, m in enumerate(d["models"]):
                if M[r, j] >= 0:
                    votes[(qid, m)][0] += M[r, j]
                    votes[(qid, m)][1] += 1
        for e in d.get("exploration", []):
            qid, m = e.get("question_id"), e.get("model")
            if not (qid and m):
                continue
            evotes[(qid, m)][0] += float(bool(e.get("correct")))
            evotes[(qid, m)][1] += 1
            if qid not in qid2prompt and qid in pool:
                qid2prompt[qid] = pool[qid].get("prompt", "")
                qid2bench[qid] = pool[qid].get("benchmark", "?")

    # questions with at least one label of any kind, and a prompt
    all_qids = sorted({q for q, _ in list(votes) + list(evotes)} | set(qid2prompt))
    qids = [q for q in all_qids if qid2prompt.get(q)]
    price_models = [r["id"] for r in json.load(open(os.path.join(REPO, "data/models.json")))]
    models = [m for m in price_models if m in set(models_seen)]
    midx = {m: i for i, m in enumerate(models)}
    qidx = {q: i for i, q in enumerate(qids)}

    Y = np.full((len(qids), len(models)), np.nan, dtype=np.float32)
    NR = np.zeros_like(Y, dtype=np.int32)   # routed votes
    NE = np.zeros_like(Y, dtype=np.int32)   # exploration votes
    # exploration votes are unbiased: merge into a separate mean first, then
    # combined matrix = weighted blend, exploration dominating on ties.
    for (qid, m), (s, n) in evotes.items():
        if qid in qidx and m in midx:
            i, j = qidx[qid], midx[m]
            NE[i, j] = n
            Y[i, j] = s / n if not np.isfinite(Y[i, j]) else Y[i, j]
    for (qid, m), (s, n) in votes.items():
        if qid in qidx and m in midx:
            i, j = qidx[qid], midx[m]
            NR[i, j] = n
            if np.isfinite(Y[i, j]):
                # blend: existing (exploration or routed) with routed votes
                prev_n = NE[i, j]
                Y[i, j] = (Y[i, j] * prev_n + s) / (prev_n + n)
            else:
                Y[i, j] = s / n

    np.savez(OUT,
             questions=np.array([qid2prompt[q] for q in qids]),
             qids=np.array(qids),
             bench=np.array([qid2bench[q] for q in qids]),
             models=np.array(models), matrix=Y,
             counts=NR, expl_counts=NE)

    obs = np.isfinite(Y)
    print(f"questions={len(qids)} models={len(models)} cells={obs.sum()} ({obs.mean():.1%})")
    print(f"routed cells={int((NR>0).sum())} exploration cells={int((NE>0).sum())}")
    for j, m in enumerate(models):
        col = Y[:, j]; o = np.isfinite(col)
        print(f"  {m:<40} n={int(o.sum()):5d} acc={np.nanmean(col):.3f}")


if __name__ == "__main__":
    main()
