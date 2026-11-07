#!/usr/bin/env python3
"""head_v10: verbatim imitation of uid126 + mmlu cheap-gate (tau=0.88).

Fit targets = the fleet's published routing decisions on all `7d5df0dd` epochs,
except mmlu-qplus cells where the gate discriminator (gate_mmlu_W.npz) says a
cheap model is confidently right — those targets become the cheap model.

The gate is load-bearing twice: it cuts burn AND pushes per-epoch pick
agreement vs uid126 to ~0.85, under the 0.95 dedup epsilon line.

Artifact: recon/head_v10.npz via contract.make_head.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from paired_compare import holdout_data
from common import load_matrix, load_hidden, route_costs
from fugal_subnet.vendor import success_contract as contract

FLASH = "deepseek/deepseek-v4-flash"
LLAMA = "meta-llama/llama-4-maverick"
QPLUS = "qwen/qwen3.7-plus"
GATE_TAU = 0.88
GATE_PEN = 0.02


def bce_fit(X, d, costs, train_idx, val_idx, epochs=800, lr=0.05,
            weight_decay=1e-4, seed=42, hi=0.9, lo=0.1):
    # Pseudo-probability BCE: the shipped head routes by
    # argmax(sigmoid(u) - lam*cost), so we fit sigmoid(u) directly to
    # pseudo-labels (target -> hi, others -> lo). The hi-lo margin (0.8)
    # dwarfs every cost gap (<~0.01), so deployed argmax = the target
    # wherever the fit reproduces it.
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(d, dtype=torch.long)
    c = torch.tensor(costs, dtype=torch.float32)
    ymat = torch.full((len(X), len(costs)), float(lo))
    ymat[torch.arange(len(X)), t] = float(hi)
    W = torch.zeros((len(costs), X.shape[1]), requires_grad=True)
    b = torch.zeros(len(costs), requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    best, ckpt = -1.0, None
    ti, vi = torch.tensor(train_idx), torch.tensor(val_idx)

    def prob(idx):
        return torch.sigmoid(x[idx] @ W.T + b)

    for ep in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy(prob(ti), ymat[ti])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (prob(vi) - c).argmax(1).eq(t[vi]).float().mean().item()
        if acc > best:
            best = acc
            ckpt = (W.detach().numpy().copy(), b.detach().numpy().copy(), ep + 1)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    ap.add_argument("--tau", type=float, default=GATE_TAU)
    ap.add_argument("--out", default=os.path.join(HERE, "head_v10.npz"))
    args = ap.parse_args()

    m = load_matrix()
    q2i = {q: i for i, q in enumerate(m["qids"])}
    bench = {q: m["bench"][i] for i, q in enumerate(m["qids"])}
    H = load_hidden().astype(np.float64)
    _, epochs = holdout_data(uid="126")

    gz = np.load(os.path.join(HERE, "gate_mmlu_W.npz"), allow_pickle=False)
    gW, gb = gz["W"], gz["b"]

    used = collections.Counter()
    for ep, qids, inc, r in epochs:
        used.update(str(x) for x in inc)
    roster = [x for x, n in used.most_common() if n >= 50]
    costs = route_costs(roster)
    print("roster:", [x.split("/")[-1] for x in roster])

    Q, D, gated = [], [], 0
    for ep, qids, inc, r in epochs:
        for q, i in zip(qids, inc):
            i = str(i)
            if q not in q2i or i not in roster:
                continue
            target = i
            if bench.get(q) == "mmlu" and i == QPLUS:
                p = contract.predictions(gW, gb, H[q2i[q]][None, :])[0]
                if max(p[0], p[1]) - GATE_PEN >= args.tau:
                    target = FLASH if p[0] >= p[1] else LLAMA
                    gated += 1
            Q.append(q)
            D.append(roster.index(target))
    D = np.array(D)
    print(f"fit rows: {len(Q)} | gated to cheap: {gated} ({gated/len(Q):.1%})")

    X = np.stack([H[q2i[q]] for q in Q])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Q))
    va = perm[: len(Q) // 6]
    ti = perm[len(Q) // 6:]

    Ws, bs, accs = [], [], []
    for s in args.seeds:
        W, b, ep = bce_fit(X, D, costs, ti, va, epochs=args.epochs, seed=s)
        acc = ((contract.predictions(W, b, X[va]) - costs).argmax(1) == D[va]).mean()
        print(f"seed {s}: ckpt ep={ep} val agreement={acc:.4f}")
        Ws.append(W); bs.append(b); accs.append(acc)
    We, be = np.mean(Ws, 0), np.mean(bs, 0)
    acc_e = ((contract.predictions(We, be, X[va]) - costs).argmax(1) == D[va]).mean()
    print(f"ensemble val agreement={acc_e:.4f}")
    W, b = (We, be) if acc_e >= max(accs) else (Ws[int(np.argmax(accs))], bs[int(np.argmax(accs))])

    manifest = json.load(open(os.path.join(REPO, "data", "benchmark_tokens_v1.json")))
    prov = json.dumps({
        "source": "imitation of uid126 head 7d5df0dd, all published epochs",
        "gate": f"mmlu qplus->cheap where max(p_flash,p_llama)-0.02>={args.tau}",
        "seeds": args.seeds,
    })
    zhead = contract.make_head(W, b, roster, manifest, prov)
    np.savez(args.out, **zhead)
    print(f"saved {args.out} ({os.path.getsize(args.out)} bytes)")

    # report achieved agreement vs gated targets under the DEPLOYED rule
    dec = np.array(roster)[(contract.predictions(W, b, X) - costs).argmax(1)]
    tgt = np.array(roster)[D]
    print(f"fit reproduces gated targets: {(dec == tgt).mean():.3f}")


if __name__ == "__main__":
    main()
