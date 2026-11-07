#!/usr/bin/env python3
"""Train a decision-imitation head on uid126's published routing decisions.

Multiclass softmax-CE over (X @ W.T + b - manifest_cost), target = incumbent
pick. Sigmoid is monotone, so contract.rank(sigmoid(u) - c) ranks u - c the
same — the multiclass logits are stored as the head's success logits.

Split by epoch inside the current head-hash run: train early, measure
agreement on the last N epochs, then refit on all epochs for the artifact.
"""
import sys, numpy as np, collections, argparse, torch

sys.path.insert(0, ".")
sys.path.insert(0, "/root/fugal-subnet")

from paired_compare import holdout_data
from common import load_matrix, load_hidden, route_costs
from fugal_subnet.vendor import success_contract as contract


def softmax_fit(X, d, costs, train_idx, val_idx, epochs=1500, lr=0.05,
                weight_decay=1e-4, seed=42, verbose=False):
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(d, dtype=torch.long)
    c = torch.tensor(costs, dtype=torch.float32)
    W = torch.zeros((len(costs), X.shape[1]), requires_grad=True)
    b = torch.zeros(len(costs), requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    best, ckpt = -1.0, None
    ti, vi = torch.tensor(train_idx), torch.tensor(val_idx)
    for ep in range(epochs):
        loss = torch.nn.functional.cross_entropy(x[ti] @ W.T + b - c, y[ti])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (x[vi] @ W.T + b - c).argmax(1).eq(y[vi]).float().mean().item()
        if acc > best:
            best = acc
            ckpt = (W.detach().numpy().copy(), b.detach().numpy().copy(), ep + 1)
        if verbose and ep % 200 == 0:
            print(f"  ep{ep} val_acc={acc:.4f}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--out", default="imit_v2")
    args = ap.parse_args()

    m = load_matrix()
    q2i = {q: i for i, q in enumerate(m["qids"])}
    H = load_hidden().astype(np.float64)
    _, epochs = holdout_data(uid="126", head_hash=None)
    train_ep, eval_ep = epochs[:-args.eval_epochs], epochs[-args.eval_epochs:]

    used = collections.Counter()
    for ep, qids, inc, r in epochs:
        used.update(str(x) for x in inc)
    roster = [x for x, n in used.most_common() if n >= 50]
    costs = route_costs(roster)
    print("roster:", [x.split("/")[-1] for x in roster], "costs:", np.round(costs, 5))

    def rows_of(eps):
        Q, D = [], []
        for ep, qids, inc, r in eps:
            keep = [(q, i) for q, i in zip(qids, inc)
                    if q in q2i and str(i) in roster]
            Q += [q for q, _ in keep]
            D += [roster.index(str(i)) for _, i in keep]
        return Q, np.array(D)

    tr_q, tr_d = rows_of(train_ep)
    ev_q, ev_d = rows_of(eval_ep)
    Xtr = np.stack([H[q2i[q]] for q in tr_q])
    Xev = np.stack([H[q2i[q]] for q in ev_q])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr_q))
    va = perm[: len(tr_q) // 6]
    ti = perm[len(tr_q) // 6:]

    Ws, bs, accs = [], [], []
    for s in args.seeds:
        W, b, ep = softmax_fit(Xtr, tr_d, costs, ti, va,
                               epochs=args.epochs, seed=s)
        Ws.append(W); bs.append(b)
        acc = ((Xev @ W.T + b - costs).argmax(1) == ev_d).mean()
        accs.append(acc)
        print(f"seed {s}: val-pick-acc={ep and None or ''} held-out agreement={acc:.4f}")
    if len(Ws) > 1:
        We, be = np.mean(Ws, 0), np.mean(bs, 0)
        acc = ((Xev @ We.T + be - costs).argmax(1) == ev_d).mean()
        print(f"ensemble held-out agreement={acc:.4f}")
        W, b = (We, be) if acc >= max(accs) else (Ws[int(np.argmax(accs))],
                                              bs[int(np.argmax(accs))])
    dec_ev = (Xev @ W.T + b - costs).argmax(1)
    dis = collections.Counter()
    for q, i, c in zip(ev_q, ev_d, dec_ev):
        if i != c:
            dis[(roster[i].split("/")[-1], roster[c].split("/")[-1],
                 str(m["bench"][q2i[q]]))] += 1
    for k, n in dis.most_common(10):
        print("  inc->cand:", k, n)

    np.save(f"{args.out}_W.npy", W)
    np.save(f"{args.out}_b.npy", b)
    np.savez(f"{args.out}_roster.npz", models=np.array(roster))
    print(f"saved {args.out}_*")


if __name__ == "__main__":
    main()
