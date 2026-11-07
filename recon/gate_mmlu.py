#!/usr/bin/env python3
"""mmlu cheap-gate discriminator for head_v10.

Trains per-model success predictors (contract.fit masked BCE on 1024-dim
embeddings) over {flash, llama, qplus} on all labeled mmlu cells — dense
self-labels plus the fleet's attested qplus outcomes from the reveal matrices.

Evaluates the realized gate on held-out mmlu questions where the FLEET
actually picked qplus: for a sweep of thresholds, route to the best cheap
model when its predicted success clears the bar, else keep the fleet pick.
Reports realized acc delta, cost delta, and net headroom delta at the
mid-frontier slope — the quantity that decides whether the gate ships.

Usage: python gate_mmlu.py [--epochs 400] [--seed 42]
"""
import argparse
import glob
import json
import os
import sys
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from paired_compare import holdout_data
from common import (load_matrix, load_hidden, score_costs, build_frame)
from fugal_subnet.success_training import fit, grouped_split
from fugal_subnet.vendor import success_contract as contract

FLASH = "deepseek/deepseek-v4-flash"
LLAMA = "meta-llama/llama-4-maverick"
QPLUS = "qwen/qwen3.7-plus"
ROSTER3 = [FLASH, LLAMA, QPLUS]
MID_SLOPE = 4.7  # frontier acc/$ on the llama->gpt-5.5 segment


def labels_for(models, bench_name="mmlu"):
    """{(qid, model): outcome} from every self-label file + reveal matrices."""
    out = {}
    for f in glob.glob(os.path.join(HERE, "labels_*.npz")):
        z = np.load(f, allow_pickle=False)
        for q, mm, o, fl in zip(z["qids"], z["models"], z["outcome"], z["failed"]):
            if not bool(fl):
                out[(str(q), str(mm))] = float(o)
    # attested reveal outcomes for the same cells (fleet's own calls)
    _, epochs = holdout_data(uid="126")
    m = load_matrix()
    bench = {q: m["bench"][i] for i, q in enumerate(m["qids"])}
    for ep, qids_sorted, _, r in epochs:
        M = np.array(r["matrix"], float)
        r_q2i = {q["question_id"]: i for i, q in enumerate(r["questions"])}
        rm = {mo: j for j, mo in enumerate(r["models"])}
        for q in qids_sorted:
            if bench.get(q) != bench_name:
                continue
            for mo in models:
                if mo in rm:
                    v = M[r_q2i[q], rm[mo]]
                    if np.isfinite(v) and v >= 0:
                        out.setdefault((q, mo), float(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-penalty", type=float, default=0.02,
                    help="route-side failure penalty subtracted from cheap prob")
    args = ap.parse_args()

    m = load_matrix()
    q2i = {q: i for i, q in enumerate(m["qids"])}
    bench = {q: m["bench"][i] for i, q in enumerate(m["qids"])}
    questions = {q: m["questions"][i] for i, q in enumerate(m["qids"])}
    H = load_hidden().astype(np.float64)

    lbl = labels_for(ROSTER3)
    mmlu_qs = [q for q in m["qids"] if bench.get(q) == "mmlu"]
    # rows: every mmlu qid with >=1 label among the 3 models
    qs = [q for q in mmlu_qs if any((q, mo) in lbl for mo in ROSTER3)]
    X = np.stack([H[q2i[q]] for q in qs])
    Y = np.full((len(qs), 3), np.nan)
    for i, q in enumerate(qs):
        for j, mo in enumerate(ROSTER3):
            if (q, mo) in lbl:
                Y[i, j] = lbl[(q, mo)]
    print(f"training rows: {len(qs)} mmlu qids | labels per model: "
          f"{[int(np.isfinite(Y[:, j]).sum()) for j in range(3)]}")

    tr, va, te = grouped_split(qs, args.seed)
    W, b, ep, _ = fit(X, Y, tr, va, epochs=args.epochs, seed=args.seed)
    p_te = contract.predictions(W, b, X[te])
    print(f"fit done at epoch {ep}; test-set per-model Brier:")
    for j, mo in enumerate(ROSTER3):
        sel = np.isfinite(Y[te, j])
        if sel.sum():
            br = np.mean((p_te[sel, j] - Y[te, j][sel]) ** 2)
            print(f"  {mo.split('/')[-1]:<22} n={sel.sum():4d}  brier={br:.4f}  "
                  f"acc={Y[te, j][sel].mean():.3f}  pred_mean={p_te[sel, j].mean():.3f}")

    # --- realized gate on fleet mmlu-qplus cells in held-out qids ---
    _, epochs = holdout_data(uid="126")
    frame = build_frame()
    val_q = {qs[i] for i in te}
    rows = []  # (qid, y_qplus, y_llama, y_flash)
    for ep2, qids_sorted, inc_models, r in epochs:
        models_r = np.array(r["models"])
        dec = r["routing_decisions"]["126"]
        M = np.array(r["matrix"], float)
        r_q2i = {q["question_id"]: i for i, q in enumerate(r["questions"])}
        rm = {mo: j for j, mo in enumerate(models_r)}
        for q, mo in zip(qids_sorted, [str(models_r[i]) for i in dec]):
            if q not in val_q or mo != QPLUS or bench.get(q) != "mmlu":
                continue
            yq = M[r_q2i[q], rm[QPLUS]] if QPLUS in rm else np.nan
            rows.append((q, yq,
                         lbl.get((q, LLAMA)), lbl.get((q, FLASH))))
    print(f"\nheld-out fleet mmlu-qplus cells: {len(rows)}")

    # predictions for those qids
    pidx = {q: i for i, q in enumerate(qs)}
    pr = [questions[q] for q, *_ in rows]
    sc = score_costs(pr, ROSTER3, frame)
    P = {q: contract.predictions(W, b, H[q2i[q]][None, :])[0] for q, *_ in rows}

    def realized(tau):
        acc_d, cost_d, n_sw = [], [], 0
        for i, (q, yq, yl, yf) in enumerate(rows):
            if not (np.isfinite(yq) and yl is not None):
                continue
            p_cheap = max(P[q][0], P[q][1]) - args.fail_penalty
            cheap = FLASH if P[q][0] >= P[q][1] else LLAMA
            use_cheap = p_cheap >= tau and (cheap == LLAMA and yl is not None
                                            or cheap == FLASH and yf is not None)
            if use_cheap:
                n_sw += 1
                y_use = yl if cheap == LLAMA else yf
                acc_d.append(y_use - yq)
                cost_d.append(sc[i, ROSTER3.index(cheap)] - sc[i, 2])
            else:
                acc_d.append(0.0)
                cost_d.append(0.0)
        acc_d, cost_d = np.array(acc_d), np.array(cost_d)
        hr_d = (acc_d - MID_SLOPE * cost_d).mean() if len(acc_d) else 0
        return len(acc_d), n_sw, acc_d.mean(), cost_d.mean(), hr_d

    print(f"\n{'tau':>5} {'n':>5} {'switched':>9} {'acc_d':>8} {'cost_d':>10} {'net_hr_d':>9}")
    for tau in [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        n, ns, ad, cd, hd = realized(tau)
        print(f"{tau:>5.2f} {n:>5} {ns:>9} {ad:>+8.4f} {cd:>+10.6f} {hd:>+9.4f}")

    # baselines
    ll = [r for r in rows if np.isfinite(r[1]) and r[2] is not None]
    print(f"\nbaselines on held-out cells (n={len(ll)}):")
    print(f"  keep qplus:           acc={np.mean([r[1] for r in ll]):.3f}")
    print(f"  blanket llama:        acc={np.mean([r[2] for r in ll]):.3f} "
          f"delta={np.mean([r[2]-r[1] for r in ll]):+.4f}")
    ff = [r for r in rows if np.isfinite(r[1]) and r[3] is not None]
    print(f"  blanket flash:        acc={np.mean([r[3] for r in ff]):.3f} "
          f"delta={np.mean([r[3]-r[1] for r in ff]):+.4f} (n={len(ff)})")
    orc = [r for r in ll if r[2] >= r[1]]
    print(f"  oracle (llama>=qplus): {len(orc)}/{len(ll)} cells")

    np.savez(os.path.join(HERE, "gate_mmlu_W.npz"),
             W=np.asarray(W), b=np.asarray(b), roster=np.array(ROSTER3))
    print("\nsaved gate_mmlu_W.npz")


if __name__ == "__main__":
    main()
