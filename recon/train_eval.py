#!/usr/bin/env python3
"""Train head candidates on the reveals matrix, evaluate with the validator's
own math (evaluate_success_head + reconstructed frontier), and report
simulated sustained score vs the incumbent (~0.033).

Roster and cost-shading are the two levers:
  - roster: which models the head may pick
  - shade: subtract kappa * (manifest_cost - min_cost) from each model's logit
    post-training — pushes routing toward cheaper models without touching the
    learned boundary (equivalent to a bias vector; the contract does not care
    how b was produced).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from fugal_subnet.api import load_prices
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import fit, grouped_split
from score_sim import build_epoch_frames_and_frontiers, simulate

sys.path.insert(0, HERE)

MATRIX = os.path.join(HERE, "matrix_v4.npz")
HIDDEN = os.path.join(HERE, "hidden_v4.npz")
MANIFEST = REPO + "/data/benchmark_tokens_v1.json"


def per_q_costs(models, prices, manifest):
    """Manifest-estimated cost per question — what the head's routing uses."""
    rows = {r["id"]: r for r in manifest["models"]}
    return np.array([
        rows[m]["mean_in_tokens"] * prices[m][0] + rows[m]["mean_out_tokens"] * prices[m][1]
        for m in models
    ])


def eval_head(W, b, models, hidden, Y, models_all, prices, manifest, frontier):
    """Route every matrix question; score on observed cells only.
    Returns accuracy on observed cells, avg actual cost/q, decisions."""
    costs_est = per_q_costs(models, prices, manifest)
    p = contract.predictions(W, b, hidden)
    decisions = contract.rank(p, costs_est, 1.0)[:, 0]
    cols = np.array([models_all.index(m) for m in models])
    labels = Y[:, cols]
    chosen = labels[np.arange(len(labels)), decisions]
    observed = np.isfinite(chosen) & (chosen >= 0)
    acc = float((chosen[observed] == 1).mean()) if observed.any() else 0.0
    return acc, decisions, observed


def main():
    z = np.load(MATRIX, allow_pickle=False)
    Y = z["matrix"].astype(float)
    # Trainer requires binary labels: majority vote, NaN stays missing.
    Y = np.where(np.isfinite(Y), (Y >= 0.5).astype(float), np.nan)
    models_all = [str(m) for m in z["models"]]
    questions = [str(q) for q in z["questions"]]
    hidden = np.load(HIDDEN, allow_pickle=False)["H"].astype(np.float64)
    prices = load_prices()
    manifest = json.load(open(MANIFEST))

    frontiers = build_epoch_frames_and_frontiers()
    frontier = frontiers[sorted(frontiers)[-1]]

    train_idx, val_idx, test_idx = grouped_split(questions, 42)
    h64 = hidden  # fit expects float; keep float64

    ROSTERS = {
        "flash+mini+opus+5.5": [
            "deepseek/deepseek-v4-flash", "openai/gpt-5.4-mini",
            "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "flash+llama+mini+opus+5.5": [
            "deepseek/deepseek-v4-flash", "meta-llama/llama-4-maverick",
            "openai/gpt-5.4-mini", "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "flash+mini+qmax+opus+5.5": [
            "deepseek/deepseek-v4-flash", "openai/gpt-5.4-mini",
            "qwen/qwen3.7-max", "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "wide8": [
            "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
            "meta-llama/llama-4-maverick", "openai/gpt-5.4-mini",
            "google/gemini-3.5-flash", "qwen/qwen3.7-max",
            "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
    }

    for name, roster in ROSTERS.items():
        cols = np.array([models_all.index(m) for m in roster])
        y = Y[:, cols]
        if (np.isfinite(y)).sum() < 1000:
            print(f"{name}: insufficient labels, skip")
            continue
        W, b, ep, hist = fit(h64, y, train_idx, val_idx, epochs=150, seed=42)

        for shade in (0.0, 30.0, 60.0, 100.0):
            costs_est = per_q_costs(roster, prices, manifest)
            b_s = b - shade * (costs_est - costs_est.min())
            acc, dec, obs = eval_head(W, b_s, roster, h64[test_idx], Y[test_idx],
                                      models_all, prices, manifest, frontier)
            # cost/q actually paid on observed cells (manifest estimates)
            paid = costs_est[dec[obs]].mean() if obs.any() else 0.0
            ref = frontier.accuracy_at(paid)
            sim = simulate([(acc * 300, 300, paid * 300)] * 25, frontier)
            print(f"{name:<28} shade={shade:5.0f} acc={acc:.3f} "
                  f"cost={paid:.6f} ref={ref:.3f} headroom={acc-ref:+.4f} "
                  f"score~{sim[-1]['score']:.4f} (n_obs={obs.sum()})")


if __name__ == "__main__":
    main()
