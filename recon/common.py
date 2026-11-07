#!/usr/bin/env python3
"""Canonical recon data: matrix_v4 + hidden_v4, and the two cost models.

TWO COST MODELS, keep them separate:
  route_costs(models) -> what the head's routing rule sees: manifest
      mean_in/out_tokens x pinned prices (contract.estimated_cost, locked
      by check_manifest into the head artifact).
  score_costs(prompts, models, frame) -> what the validator's scoreboard
      charges: pinned in-price x question_input_tokens(prompt) +
      pinned out-price x frame.avg_completion_tokens(model)
      (validator.py policy_cost_total). Per-question, not per-model —
      prompt length varies.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from fugal_subnet.api import load_prices
from fugal_subnet.config import FRAME_DEFAULT_COMPLETION_TOKENS
from fugal_subnet.pricing import question_input_tokens
from fugal_subnet.vendor import success_contract as contract

MATRIX = os.path.join(HERE, "matrix_v4.npz")
HIDDEN = os.path.join(HERE, "hidden_v4.npz")
MANIFEST = os.path.join(REPO, "data", "benchmark_tokens_v1.json")
REVEALS_GLOB = os.path.join(REPO, "results", "reveals-70", "*", "reveal.json")

CACHE = {}


def load_matrix():
    """Canonical label matrix: 12,083 questions x 17 models, NaN=unobserved."""
    if "matrix" not in CACHE:
        z = np.load(MATRIX, allow_pickle=False)
        CACHE["matrix"] = dict(
            Y=z["matrix"].astype(float),
            Ybin=np.where(np.isfinite(z["matrix"].astype(float)),
                          (z["matrix"] >= 0.5).astype(float), np.nan),
            models=[str(m) for m in z["models"]],
            qids=[str(q) for q in z["qids"]],
            questions=[str(q) for q in z["questions"]],
            bench=np.array([str(b) for b in z["bench"]]),
            counts=z["counts"],
            expl_counts=z["expl_counts"],
        )
    return CACHE["matrix"]


def load_hidden():
    if "hidden" not in CACHE:
        CACHE["hidden"] = np.load(HIDDEN, allow_pickle=False)["H"].astype(np.float64)
    return CACHE["hidden"]


def load_manifest():
    if "manifest" not in CACHE:
        CACHE["manifest"] = json.load(open(MANIFEST))
    return CACHE["manifest"]


def route_costs(models):
    """Per-model cost the ROUTING rule subtracts: manifest estimate."""
    rows = {r["id"]: r for r in load_manifest()["models"]}
    prices = load_prices()
    return np.array([
        rows[m]["mean_in_tokens"] * prices[m][0]
        + rows[m]["mean_out_tokens"] * prices[m][1]
        for m in models
    ])


def score_costs(prompts, models, frame):
    """Per-(question, model) cost the SCOREBOARD charges (frame verbosity)."""
    prices = load_prices()
    ctok = {m: frame.avg_completion_tokens(m, FRAME_DEFAULT_COMPLETION_TOKENS)
            for m in models}
    q_tok = np.array([question_input_tokens(p) for p in prompts], dtype=float)
    in_rates = np.array([prices[m][0] for m in models])
    out_rates = np.array([prices[m][1] for m in models])
    comp = np.array([ctok[m] for m in models])
    # (N,1) * (M,) + (M,) broadcast -> (N, M) dollars
    return q_tok[:, None] * in_rates[None, :] + comp[None, :] * out_rates[None, :]


def latest_frame_and_frontier():
    """(frame, frontier) replayed over every reveal on disk, epoch order."""
    from score_sim import build_epoch_frames_and_frontiers
    frontiers = build_epoch_frames_and_frontiers()
    return None, frontiers[sorted(frontiers)[-1]]


def build_frame():
    """ReferenceFrame after folding every reveal's exploration (bootstrap + live)."""
    from score_sim import epoch_samples
    from fugal_subnet.reference_frame import accumulate_exploration, load_bootstrap
    frame = load_bootstrap()
    for path in sorted(glob.glob(REVEALS_GLOB)):
        frame = accumulate_exploration(frame, epoch_samples(json.load(open(path))))
    return frame


def epoch_frontiers():
    """{epoch_id: Frontier} rebuilt exactly as the validator does."""
    from score_sim import build_epoch_frames_and_frontiers
    return build_epoch_frames_and_frontiers()


def load_reveals():
    """[(epoch_id, reveal_dict)] sorted by epoch."""
    if "reveals" not in CACHE:
        out = []
        for path in sorted(glob.glob(REVEALS_GLOB)):
            r = json.load(open(path))
            out.append((r["epoch_id"], r))
        CACHE["reveals"] = out
    return CACHE["reveals"]


def incumbent_history(uid="126"):
    """[(epoch_id, head_hash, score_dict)] for every epoch the uid appears in."""
    rows = []
    for ep, r in load_reveals():
        h = r["head_hashes"].get(uid)
        s = r["scores"].get(uid) or {}
        rows.append((ep, h, s))
    return rows


def current_head_run(uid="126"):
    """Epochs where uid runs its LATEST head hash, contiguous from the newest."""
    hist = incumbent_history(uid)
    latest_hash = None
    for _ep, h, _s in reversed(hist):
        if h:
            latest_hash = h
            break
    run = [(ep, h, s) for ep, h, s in hist if h == latest_hash]
    return latest_hash, run


def pool_lookup():
    if "pool" not in CACHE:
        pool = json.load(gzip.open(os.path.join(REPO, "data", "benchmarks", "pool.json.gz")))
        CACHE["pool"] = {q["question_id"]: q for q in pool}
    return CACHE["pool"]
