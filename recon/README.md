# recon/ — miner strategy toolkit

Offline analysis + head-training pipeline for beating the top miner on netuid 70.
Everything here runs locally and free (no API spend, no chain needed).

## Contents

| File | What it does |
|---|---|
| `matrix.npz` | Label matrix: 7,438 questions × 17 models, 0/1/NaN (built from 31 epoch reveals) |
| `build_matrix.py` | Rebuilds `matrix.npz` from `results/reveals-70/` |
| `recon_frontier.py` | Reconstructs the validator frontier per epoch; profiles the field |
| `score_sim.py` | Replays a candidate (acc, cost/q) through evidence accumulation + composite scoring |
| `embed_matrix.py` | Embeds matrix questions (checkpointed; `FUGAL_BACKBONE_THREADS=N` to set cores) |
| `train_eval.py` | Trains head candidates on the matrix; evals with validator math + simulated score |
| `dedup_check.py` | Compares a head's routing decisions vs incumbents' published ones |

## Data

`results/reveals-70/<epoch>/reveal.json` — 31 mainnet epochs downloaded from the
public `gs://fugal-reveals/70/` bucket. Refresh with the fetch loop used to
populate it. `matrix.npz` is ~134 MB — keep it out of git.

## Key numbers (epoch e00026219)

- Frontier hull: ds-flash ($0.0001/q, .814) → llama (.0004/q, .885) → gpt-5.5 ($0.0084/q, .935)
- Top miner sustained headroom ≈ **+0.033** (score 0.033 at ~92–93% acc, ~$0.0016/q)
- Target: pooled acc ≥ .94 at ≤ ~$0.0017/q → score ~.04
- Oracle ceiling on the slice: ~+0.11 headroom
- Traps: `mistral-large-2512` (36% exploration failures), `llama` bad-epoch risk

## Usage

```bash
cd recon
FUGAL_BACKBONE_THREADS=11 ../.venv/bin/python embed_matrix.py   # one-time, checkpointed
../.venv/bin/python train_eval.py                               # rosters x shade sweep
../.venv/bin/python dedup_check.py <head.npz>                   # vs latest slice
```
