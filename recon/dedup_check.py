#!/usr/bin/env python3
"""Dedup self-check: route the latest epoch's slice with a candidate head and
compare decisions against every incumbent miner's published routing_decisions.

Dedup rule (fugal_subnet/dedup.py): behavioural agreement >= 99% on the slice
-> copy. >= 95% agreement + not beating senior's score by 0.5% -> disqualified.
We want comfortable margin: < 90% agreement with everyone.
"""
import glob
import os
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from fugal_subnet.api import load_prices
from fugal_subnet.backbone import compute_hidden_states
from fugal_subnet.head_eval import load_head_from_npz
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA


def main(head_path, epoch=None):
    paths = sorted(glob.glob("REPO/results/reveals-70/*/reveal.json"))
    path = paths[-1] if epoch is None else [p for p in paths if epoch in p][0]
    reveal = json.load(open(path))
    questions = reveal["questions"]
    models = reveal["models"]
    print(f"epoch {reveal['epoch_id']} slice: {len(questions)} questions")

    head = load_head_from_npz(open(head_path, "rb").read())
    prices = load_prices()
    z = np.load(head_path, allow_pickle=False)
    costs = contract.estimated_cost(z, prices)

    prompts = [q["prompt"] for q in questions]
    hidden = compute_hidden_states(prompts, batch_size=16)
    p = contract.predictions(head.W, head.b, hidden)
    ours = contract.rank(p, costs, BENCHMARK_LAMBDA)[:, 0]
    ours_models = np.array(head.models)[ours]

    ranked = sorted(reveal["scores"].items(), key=lambda kv: -(kv[1].get("score") or 0))
    print(f"\n{'uid':>4} {'score':>7} {'agree%':>7}  note")
    for uid, s in ranked[:25]:
        dec = np.array(reveal["routing_decisions"][uid])
        their_models = np.array(models)[dec]
        agree = (their_models == ours_models).mean()
        flag = "  <-- DEDUP RISK" if agree >= 0.95 else ""
        print(f"{uid:>4} {s.get('score', 0):7.4f} {agree:7.1%}{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
