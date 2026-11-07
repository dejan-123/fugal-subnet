#!/usr/bin/env python3
"""Embed matrix questions under the pinned success profile, checkpointed.

Saves every CHUNK questions to hidden_partial.npz (this dir) so a crash/OOM
resumes instead of restarting. On completion writes the contract cache
hidden.npz (this dir).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from fugal_subnet.backbone import compute_hidden_states
from fugal_subnet.vendor import success_contract as contract

CHUNK = 500
PARTIAL = os.path.join(HERE, "hidden_partial.npz")
FINAL = os.path.join(HERE, "hidden.npz")
BATCH = 8  # smaller batches = lower peak RSS

z = np.load(os.path.join(HERE, "matrix.npz"), allow_pickle=False)
questions = [str(q) for q in z["questions"]]
n = len(questions)

done = 0
hidden = np.zeros((n, 1024), dtype=np.float32)
if os.path.exists(PARTIAL):
    p = np.load(PARTIAL, allow_pickle=False)
    done = int(p["done"])
    hidden[:done] = p["H"][:done]
    print(f"resuming at {done}/{n}", flush=True)

while done < n:
    end = min(done + CHUNK, n)
    t0 = time.time()
    hidden[done:end] = compute_hidden_states(questions[done:end], batch_size=BATCH)
    done = end
    np.savez(PARTIAL, H=hidden, done=done)
    print(f"{done}/{n} (+{time.time()-t0:.0f}s)", flush=True)

contract.save_cache(FINAL, questions, hidden)
print("saved", FINAL)
os.remove(PARTIAL)
