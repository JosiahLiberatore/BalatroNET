"""Benchmark: scoreHand unjitted vs jitted vs jit(vmap) batch."""

import time

import jax
import jax.numpy as jnp
from jax import jit, vmap, random

from main import scoreHand          # the @jit-wrapped version
raw_scoreHand = scoreHand.__wrapped__   # the original python function underneath

r = jnp.array([11, 11, 5, 5, 5])    # full house test hand
s = jnp.array([2, 3, 1, 0, 3])
v = jnp.array([1, 1, 1, 1, 1])

N = 1000

# --- warm-up: first jitted call pays compilation, exclude it from timing
t0 = time.perf_counter()
scoreHand(r, s, v).block_until_ready()
compile_time = time.perf_counter() - t0

# --- unjitted: every op dispatches to the device one at a time, eagerly
t0 = time.perf_counter()
for _ in range(N):
    raw_scoreHand(r, s, v).block_until_ready()
unjit_per_call = (time.perf_counter() - t0) / N

# --- jitted: one fused compiled kernel per call
t0 = time.perf_counter()
for _ in range(N):
    scoreHand(r, s, v).block_until_ready()
jit_per_call = (time.perf_counter() - t0) / N

# --- jit(vmap): score a whole batch of hands in ONE call
BATCH = 100_000
key = random.key(0)
k1, k2 = random.split(key)
batch_ranks = random.randint(k1, (BATCH, 5), 0, 13)   # (100000, 5) random hands
batch_suits = random.randint(k2, (BATCH, 5), 0, 4)
batch_valid = jnp.ones((BATCH, 5))

score_batch = jit(vmap(scoreHand))
score_batch(batch_ranks, batch_suits, batch_valid).block_until_ready()   # warm-up compile

t0 = time.perf_counter()
REPS = 20
for _ in range(REPS):
    score_batch(batch_ranks, batch_suits, batch_valid).block_until_ready()
batch_per_hand = (time.perf_counter() - t0) / (REPS * BATCH)

print(f"compile time (one-off):   {compile_time * 1e3:10.2f} ms")
print(f"unjitted, per hand:       {unjit_per_call * 1e6:10.2f} us")
print(f"jitted,   per hand:       {jit_per_call * 1e6:10.2f} us   ({unjit_per_call / jit_per_call:6.1f}x faster)")
print(f"jit+vmap, per hand:       {batch_per_hand * 1e6:10.2f} us   ({unjit_per_call / batch_per_hand:6.1f}x faster)")
print(f"jit+vmap throughput:      {1 / batch_per_hand:,.0f} hands/sec")
