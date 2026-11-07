#!/usr/bin/env python3
"""Optimize the head's bias vector directly for score.

After training W,b on masked BCE, the bias is only ~4-8 free parameters.
Grid+coordinate search over bias offsets maximizes what actually pays:
    simulated headroom = acc(decisions) - frontier(cost(decisions))
on validation data, verified on test.

Eval caveat handled here: cells with no observed label are imputed by a
smoothed per-(model,benchmark) accuracy estimated from the whole matrix —
an honest-ish expectation rather than silently dropping the head's weaker
picks (live scoring counts every routed question).
"""
import json
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from fugal_subnet.api import load_prices
from fugal_subnet.vendor import success_contract as contract
from fugal_subnet.success_training import fit, grouped_split
from score_sim import build_epoch_frames_and_frontiers, simulate

MATRIX = os.path.join(HERE, "matrix_v4.npz")
HIDDEN = os.path.join(HERE, "hidden_v4.npz")
MANIFEST = os.path.join(REPO, "data", "benchmark_tokens_v1.json")


def build_fallback(Y_bin, bench, models_all):
    """P(correct) fallback for unobserved (q,m) cells: smoothed
    per-(model,benchmark) mean shrunk toward the model's global mean."""
    B = sorted(set(bench.tolist()))
    fb = np.zeros_like(Y_bin)
    for j in range(Y_bin.shape[1]):
        col = Y_bin[:, j]
        g = np.nanmean(col) if np.isfinite(col).any() else 0.5
        for b in B:
            mask = (bench == b)
            obs = np.isfinite(col[mask])
            n = obs.sum()
            mean = np.nanmean(col[mask]) if n else g
            k = 20.0  # shrinkage
            fb[mask, j] = (n * mean + k * g) / (n + k)
    return fb


def evaluate(W, b, roster, hidden, Y_obs, Y_fb, models_all, prices, manifest):
    """Route all rows; accuracy = observed where present else fallback.
    Returns (acc, cost_per_q_est, decisions, acc_observed_only)."""
    costs_est = np.array([
        prices[m][0] * {r["id"]: r for r in manifest["models"]}[m]["mean_in_tokens"]
        + prices[m][1] * {r["id"]: r for r in manifest["models"]}[m]["mean_out_tokens"]
        for m in roster])
    p = contract.predictions(W, b, hidden)
    dec = contract.rank(p, costs_est, 1.0)[:, 0]
    cols = np.array([models_all.index(m) for m in roster])
    chosen_obs = Y_obs[np.arange(len(Y_obs)), cols[dec]]
    chosen_fb = Y_fb[np.arange(len(Y_fb)), cols[dec]]
    has = np.isfinite(chosen_obs) & (chosen_obs >= 0)
    acc = (chosen_obs[has].sum() + chosen_fb[~has].sum()) / len(chosen_obs)
    acc_obs = float(chosen_obs[has].mean()) if has.any() else float('nan')
    cost = costs_est[dec].mean()
    return float(acc), float(cost), dec, acc_obs


def optimize_bias(W, b0, roster, hidden, Y_obs, Y_fb, models_all, prices,
                  manifest, frontier):
    """Coordinate ascent on bias offsets maximizing (acc - frontier(cost))."""
    def objective(bias):
        acc, cost, _, _o = evaluate(W, b0 + bias, roster, hidden, Y_obs, Y_fb,
                                models_all, prices, manifest)
        return acc - frontier.accuracy_at(cost), acc, cost
    bias = np.zeros(len(roster))
    best_hr, _, _ = objective(bias)
    steps = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for j in range(len(roster)):
                for d in (+step, -step):
                    cand = bias.copy(); cand[j] += d
                    hr, _, _ = objective(cand)
                    if hr > best_hr + 1e-5:
                        bias, best_hr = cand, hr
                        improved = True
    return bias, best_hr


def main():
    z = np.load(MATRIX, allow_pickle=False)
    Y = np.where(np.isfinite(z["matrix"].astype(float)),
                 (z["matrix"] >= 0.5).astype(float), np.nan)
    models_all = [str(m) for m in z["models"]]
    bench = z["bench"]
    hidden = np.load(HIDDEN, allow_pickle=False)["H"].astype(np.float64)
    prices = load_prices()
    manifest = json.load(open(MANIFEST))
    frontiers = build_epoch_frames_and_frontiers()
    frontier = frontiers[sorted(frontiers)[-1]]
    Y_fb = build_fallback(Y, bench, models_all)

    train_idx, val_idx, test_idx = grouped_split([str(q) for q in z["questions"]], 42)

    ROSTERS = {
        "incumbent-like {flash,qplus,opus,5.5}": [
            "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
            "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "incumbent+mini {flash,qplus,mini,opus,5.5}": [
            "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
            "openai/gpt-5.4-mini", "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "incumbent+llama {flash,qplus,llama,opus,5.5}": [
            "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
            "meta-llama/llama-4-maverick", "anthropic/claude-opus-4.8",
            "openai/gpt-5.5"],
        "incumbent6 {flash,qplus,llama,mini,opus,5.5}": [
            "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
            "meta-llama/llama-4-maverick", "openai/gpt-5.4-mini",
            "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        "incumbent7 +qmax": [
            "deepseek/deepseek-v4-flash", "qwen/qwen3.7-plus",
            "qwen/qwen3.7-max", "openai/gpt-5.4-mini",
            "anthropic/claude-opus-4.8", "openai/gpt-5.5"],
    }

    for name, roster in ROSTERS.items():
        cols = np.array([models_all.index(m) for m in roster])
        y = Y[:, cols]
        try:
            W, b, ep, hist = fit(hidden, y, train_idx, val_idx, epochs=300, seed=42)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        # optimize bias on validation
        bias, hr_val = optimize_bias(
            W, b, roster, hidden[val_idx], Y[val_idx], Y_fb[val_idx],
            models_all, prices, manifest, frontier)
        # report on test
        acc_t, cost_t, _, obs_t = evaluate(
            W, b + bias, roster, hidden[test_idx], Y[test_idx], Y_fb[test_idx],
            models_all, prices, manifest)
        ref = frontier.accuracy_at(cost_t)
        sim = simulate([(acc_t * 300, 300, cost_t * 300)] * 25, frontier)
        acc_b, cost_b, _, obs_b = evaluate(
            W, b, roster, hidden[test_idx], Y[test_idx], Y_fb[test_idx],
            models_all, prices, manifest)
        print(f"{name:<28} base: acc={acc_b:.3f}@{cost_b:.5f} hr={acc_b-frontier.accuracy_at(cost_b):+.4f} | "
              f"tuned: acc={acc_t:.3f}@{cost_t:.5f} hr={acc_t-ref:+.4f} "
              f"score~{sim[-1]['score']:.4f} acc_obs={obs_t:.3f} bias={np.round(bias,2)}")


if __name__ == "__main__":
    main()
