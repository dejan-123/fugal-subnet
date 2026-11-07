#!/usr/bin/env python3
"""Reconstruct the exact validator frontier for each published epoch and
profile the top miner. Replays bootstrap frame + live exploration in epoch
order, rebuilds the frontier from the reveal's own questions, and verifies
against the published reference_accuracy values."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)

from fugal_subnet.api import load_prices
from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
from fugal_subnet.frontier import build_frontier
from fugal_subnet.pricing import question_input_tokens
from fugal_subnet.reference_frame import (
    Observation,
    accumulate_exploration,
    aggregate_observations,
    load_bootstrap,
)

REVEALS = sorted(glob.glob("results/reveals-70/*/reveal.json"))


def epoch_samples(reveal):
    obs = [
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
    ]
    return aggregate_observations(obs)


def main():
    prices = load_prices("data/models.json") if os.path.exists("data/models.json") else load_prices()
    frame = load_bootstrap(require=False)
    rows = []
    for path in REVEALS:
        reveal = json.load(open(path))
        epoch = reveal["epoch_id"]
        # Validator folds this epoch's exploration in BEFORE building the frontier
        frame = accumulate_exploration(frame, epoch_samples(reveal))
        questions = reveal["questions"]
        slice_tokens = sum(
            question_input_tokens(q.get("prompt", "")) for q in questions
        )
        frontier = build_frontier(
            frame, prices, slice_tokens, len(questions),
            FRAME_DEFAULT_COMPLETION_TOKENS,
        )
        rows.append((epoch, frontier, reveal))

    epoch, frontier, reveal = rows[-1]
    print(f"=== Epoch {epoch} frontier ===")
    print(f"confidence={frontier.confidence:.4f} least_trials={frontier.least_trials:.0f} max_acc={frontier.max_accuracy:.3f}")
    print("hull points (cost/question -> dumb accuracy):")
    for c, a in frontier.points:
        print(f"  ${c:.6f} -> {a:.3f}")
    print("hull models:", list(frontier.models))

    # Verify reconstruction against published reference_accuracy
    err = 0.0
    n = 0
    for uid, s in reveal["scores"].items():
        if not s:
            continue
        pred = frontier.accuracy_at(s["cost_per_question"])
        err = max(err, abs(pred - s["reference_accuracy"]))
        n += 1
    print(f"max |frontier(c) - published ref_acc| over {n} miners = {err:.6f}")

    # Field summary: who is where
    models = reveal["models"]
    ranked = sorted(reveal["scores"].items(), key=lambda kv: -(kv[1].get("score") or 0))
    print(f"\n{'uid':>4} {'acc':>5} {'cost/q':>9} {'ref':>5} {'headroom':>9} {'score':>8} {'weight':>7}")
    for uid, s in ranked[:20]:
        print(f"{uid:>4} {s['accuracy']:.3f} {s['cost_per_question']:9.6f} "
              f"{s['reference_accuracy']:.3f} {s['headroom']:+.4f} {s['score']:8.4f} "
              f"{reveal['weights'].get(uid, 0):7.4f}")

    # Per-model realized accuracy THIS epoch from the matrix (all miners' calls)
    import numpy as np
    M = np.array(reveal["matrix"], dtype=float)
    obs = np.isfinite(M) & (M >= 0)
    acc = np.where(obs.sum(0) > 0, np.nan_to_num(np.where(M >= 0, M, np.nan)).sum(0) / obs.sum(0), np.nan)
    print("\nthis-epoch matrix accuracy (all routed calls, not just exploration):")
    for i, m in enumerate(models):
        if obs[:, i].sum() > 3:
            print(f"  {m:<40} n={int(obs[:,i].sum()):4d} acc={acc[i]:.3f}")

    return rows


if __name__ == "__main__":
    main()
