#!/usr/bin/env python3
"""Score trajectory simulator: replay a candidate head's per-epoch
(accuracy, cost/question) through the real evidence accumulation + composite
scoring, against the reconstructed frontier.

Usage: edit the SCENARIOS list or import simulate().
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np

from fugal_subnet.api import load_prices
from fugal_subnet.config import (
    BURN_IN_QUESTIONS,
    EVIDENCE_HALF_LIFE,
    FRAME_DEFAULT_COMPLETION_TOKENS,
    SCORE_QUALITY_FLOOR,
    SLICE_SIZE,
)
from fugal_subnet.evidence import accumulate_epoch, apply_miss
from fugal_subnet.frontier import build_frontier
from fugal_subnet.pricing import question_input_tokens
from fugal_subnet.reference_frame import (
    Observation,
    accumulate_exploration,
    aggregate_observations,
    load_bootstrap,
)
from fugal_subnet.scoring import burn_in_factor, composite


def epoch_samples(reveal):
    return aggregate_observations([
        Observation(
            question_id=str(e.get("question_id", "")),
            model=str(e.get("model", "")),
            correct=bool(e.get("correct", False)),
            prompt_tokens=int(e.get("prompt_tokens", 0)),
            completion_tokens=int(e.get("completion_tokens", 0)),
            failed=bool(e.get("failed", False)),
            coldkey=str(e.get("coldkey") or f"entry-{i}"),
        )
        for i, e in enumerate(reveal.get("exploration", []))
        if e.get("model")
    ])


def build_epoch_frames_and_frontiers(reveals_dir="results/reveals-70"):
    """Returns {epoch_id: (frontier, slice_prompt_tokens)} replaying frame state
    exactly as the validator does (fold exploration in, then build frontier)."""
    prices = load_prices()
    frame = load_bootstrap()
    out = {}
    for path in sorted(glob.glob(os.path.join(reveals_dir, "*", "reveal.json"))):
        reveal = json.load(open(path))
        frame = accumulate_exploration(frame, epoch_samples(reveal))
        slice_tok = sum(
            question_input_tokens(q.get("prompt", "")) for q in reveal["questions"]
        )
        out[reveal["epoch_id"]] = build_frontier(
            frame, prices, slice_tok, len(reveal["questions"]),
            FRAME_DEFAULT_COMPLETION_TOKENS,
        )
    return out


def simulate(per_epoch, frontier, head_hash="sim", n_expected=SLICE_SIZE,
             pool_size=21553.0):
    """per_epoch: list of (n_correct, n_scored, total_cost) or None for a miss.
    Returns list of dicts with the evidence/score state after each epoch."""
    ev = None
    hist = []
    for ep in per_epoch:
        if ep is None:
            if ev is not None:
                ev = apply_miss(ev, n_expected, EVIDENCE_HALF_LIFE)
        else:
            n_correct, n_total, cost = ep
            ev = accumulate_epoch(ev, head_hash, n_correct, n_total, cost,
                                  0.0, EVIDENCE_HALF_LIFE, pool_size=pool_size)
        if ev is None:
            hist.append({"lcb": 0.0, "cost": 0.0, "score": 0.0})
            continue
        hist.append({
            "lcb": ev.wilson_lcb,
            "acc": ev.accuracy,
            "cost": ev.cost_per_question,
            "ref": frontier.accuracy_at(ev.cost_per_question),
            "headroom": ev.wilson_lcb - frontier.accuracy_at(ev.cost_per_question),
            "burn_in": burn_in_factor(ev.n_total),
            "score": composite(ev, frontier),
        })
    return hist


def demo():
    frontiers = build_epoch_frames_and_frontiers()
    frontier = frontiers[sorted(frontiers)[-1]]
    print("latest frontier hull:", [(round(c, 6), round(a, 3)) for c, a in frontier.points])
    print(f"max_acc={frontier.max_accuracy:.3f} confidence={frontier.confidence:.2f}")

    N = 300
    scenarios = {
        # name: (accuracy, cost_per_question)
        "top-miner-like   acc .92 @ $0.0017": (0.92, 0.0017),
        "better-acc       acc .94 @ $0.0017": (0.94, 0.0017),
        "cheap+accurate   acc .90 @ $0.0008": (0.90, 0.0008),
        "same cost higher acc .93 @ $0.0020": (0.93, 0.0020),
        "expensive        acc .94 @ $0.0050": (0.94, 0.0050),
        "all-gpt5.5       acc .935 @ $0.0084": (0.935, 0.008448),
        "all-cheap        acc .85 @ $0.0004": (0.85, 0.0004),
    }
    print(f"\n{'scenario':<42} {'score@e5':>9} {'@e10':>9} {'@e20':>9} {'@e30':>9}")
    for name, (acc, cost) in scenarios.items():
        per = [(acc * N, N, cost * N)] * 30
        h = simulate(per, frontier)
        def s(i): return h[min(i, len(h) - 1)]["score"]
        print(f"{name:<42} {s(4):9.4f} {s(9):9.4f} {s(19):9.4f} {s(29):9.4f}")

    # top miner comparison: their published headroom ~0.033
    print("\ntop miner published score latest epoch: ~0.033")
    print("floor: lcb must exceed", f"{SCORE_QUALITY_FLOOR * frontier.max_accuracy:.3f}",
          "(0.8 x frontier max)")


if __name__ == "__main__":
    demo()
