#!/usr/bin/env python3
"""Paired comparison: candidate head vs an incumbent uid on identical questions.

For every holdout epoch, aligns the incumbent's published routing_decisions
by SORTED question_id (the order _extract_routing_decisions writes them),
routes the candidate on the same questions, and compares accuracy/cost/
headroom on cells where the two policies' outcomes are known.

Labels come from matrix_v4 (contaminated by last-writer-wins) unless a
self-labeled npz overrides them — see label_pair_cells.py.
"""
import sys
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from common import (load_matrix, load_hidden, load_reveals, load_manifest,
                    route_costs, score_costs, epoch_frontiers)
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA
from fugal_subnet.pricing import question_input_tokens
from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
from score_sim import epoch_samples
from fugal_subnet.reference_frame import accumulate_exploration, load_bootstrap


def holdout_data(uid="126", head_hash=None):
    """[(epoch, qids_sorted, inc_decisions(models), reveal)] for holdout epochs."""
    revs = load_reveals()
    if head_hash is None:
        for ep, r in reversed(revs):
            hh = r["head_hashes"].get(uid)
            if hh:
                head_hash = hh
                break
    out = []
    for ep, r in revs:
        if r["head_hashes"].get(uid) != head_hash or uid not in r["routing_decisions"]:
            continue
        dec = np.array(r["routing_decisions"][uid])
        qids_sorted = sorted(q["question_id"] for q in r["questions"])
        models = np.array(r["models"])
        out.append((ep, qids_sorted, models[dec], r))
    return head_hash, out


def candidate_decisions(W, b, roster, hidden_by_qid, qids_sorted):
    costs = route_costs(roster)
    H = np.stack([hidden_by_qid[q] for q in qids_sorted])
    p = contract.predictions(W, b, H)
    dec = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
    return np.array(roster)[dec]


def paired_compare(W, b, roster, uid="126", head_hash=None, self_labels=None):
    """Returns per-question paired rows + summary.

    self_labels: {qid: {model: outcome}} from our own calls, overriding matrix.
    """
    m = load_matrix()
    q2i = {q: i for i, q in enumerate(m["qids"])}
    hidden = load_hidden()
    hidden_by_qid = {q: hidden[i] for q, i in q2i.items()}
    frontiers = epoch_frontiers()
    models_all = m["models"]
    midx = {mo: j for j, mo in enumerate(models_all)}

    head_hash, epochs = holdout_data(uid, head_hash)
    rows = []  # (epoch, qid, bench, inc_model, cand_model, y_inc, y_cand, c_inc, c_cand)
    for ep, qids_sorted, inc_models, r in epochs:
        M = np.array(r["matrix"], float)
        r_q2i = {q["question_id"]: i for i, q in enumerate(r["questions"])}
        rev_models = np.array(r["models"])
        rev_midx = {mo: j for j, mo in enumerate(rev_models)}
        cand_models = candidate_decisions(W, b, roster, hidden_by_qid, qids_sorted)
        for i, qid in enumerate(qids_sorted):
            inc_m, cand_m = str(inc_models[i]), str(cand_models[i])
            rrow = r_q2i[qid]
            y_inc = M[rrow, rev_midx[inc_m]] if inc_m in rev_midx else np.nan
            y_cand = M[rrow, rev_midx[cand_m]] if cand_m in rev_midx else np.nan
            if y_inc < 0: y_inc = np.nan
            if y_cand < 0: y_cand = np.nan
            if self_labels:
                sl = self_labels.get(qid, {})
                y_inc = sl.get(inc_m, y_inc)
                y_cand = sl.get(cand_m, y_cand)
            rows.append((ep, qid, "", inc_m, cand_m, y_inc, y_cand))
    return head_hash, rows


def summarize(rows, frontier, m_by_qid=None):
    """Paired diff on cells where BOTH outcomes are labeled."""
    m = load_matrix()
    q2i = {q: i for i, q in enumerate(m["qids"])}
    bench = {q: m["bench"][i] for q, i in q2i.items()}
    prompts = {q: m["questions"][i] for q, i in q2i.items()}
    frame = build_frame()

    d_acc, d_cost, qs = [], [], []
    per_bench = {}
    for ep, qid, _b, inc_m, cand_m, y_inc, y_cand in rows:
        if not (np.isfinite(y_inc) and np.isfinite(y_cand)):
            continue
        pr = prompts.get(qid, "")
        sc = score_costs([pr], [inc_m, cand_m], frame)[0]
        d_acc.append(y_cand - y_inc)
        d_cost.append(sc[1] - sc[0])
        b = bench.get(qid, "?")
        per_bench.setdefault(b, []).append(y_cand - y_inc)
        qs.append(qid)
    d_acc, d_cost = np.array(d_acc), np.array(d_cost)
    n = len(d_acc)
    slope = 6.18  # frontier acc per $ between llama and gpt-5.5 vertices
    d_hr = d_acc - slope * d_cost
    se = d_hr.std(ddof=1) / np.sqrt(max(n, 1))
    lb80 = d_hr.mean() - 1.282 * se
    print(f"paired n={n} ({len(set(qs))} unique qids)")
    print(f"  acc diff   = {d_acc.mean():+.4f}")
    print(f"  cost diff  = {d_cost.mean():+.6f} $/q")
    print(f"  headroom   = {d_hr.mean():+.4f}  se={se:.4f}  LB80={lb80:+.4f}")
    print("  by benchmark:")
    for bname, v in sorted(per_bench.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        print(f"    {bname:<10} n={len(v):5d}  acc-diff={v.mean():+.4f}")
    return d_hr.mean(), lb80


def build_frame():
    from common import build_frame as _bf
    return _bf()


if __name__ == "__main__":
    z = np.load(sys.argv[1], allow_pickle=False)
    roster = [str(x) for x in z["models"]]
    hh, rows = paired_compare(z["W"], z["b"], roster)
    print("incumbent head:", hh[:16], "epochs:", len({r[0] for r in rows}))
    summarize(rows, None)
