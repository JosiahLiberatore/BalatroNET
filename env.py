"""The Balatro environment — EnvState, reset, step.

Pure functional: the state pytree is the only memory. reset births the full
state; env_step takes a full state and returns the next full state. Capacity
is SHAPE (MAX_* constants), current count is DATA (scalars/masks in state).
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple

MAX_HAND_SIZE = 32
MAX_DECK_SIZE = 128


class balatroEnvState(NamedTuple):
    hand: jnp.ndarray
    hand_size : jnp.int32
    hand_valid : jnp.array
    deck : jnp.ndarray
    deck_size : jnp.int32
    deck_ptr : jnp.int32
    hands: jnp.int32
    discards: jnp.int32
    round: jnp.int32
    ante: jnp.int32
    chips_scored : jnp.int32
    blind_target : jnp.int32
    key : jax.random.key


@jit
def env_step(state: balatroEnvState) -> tuple[balatroEnvState, float, bool]:
    #deal hand

    # play
        # score
        # deal
    # discard
        # deal

    next_state = balatroEnvState(playerHand=jnp.zeros((MAX_HAND_SIZE)))

    return next_state, reward, done

@jit
def reset(key, deckSize, handSize):
    key, subkey = jax.random.split(key)

    deck = jax.random.permutation(subkey, MAX_DECK_SIZE)

    dead = deck >= deckSize
    deck = deck[jnp.argsort(dead)]

    hand_valid = jnp.arange(MAX_HAND_SIZE) < handSize
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)

    return balatroEnvState(
        hand=hand,
        hand_size=jnp.int32(handSize),
        hand_valid=hand_valid,
        deck=deck,
        deck_size=jnp.int32(deckSize),
        deck_ptr=jnp.int32(handSize),
        hands=jnp.int32(4),
        discards=jnp.int32(3),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(0),
        blind_target=jnp.int32(300),
        key=key,
    )


def show(state: balatroEnvState, label=""):
    """Pretty-print a state. Plain Python — call it on concrete states only
    (outside jit, or on the result of a jitted function), never inside one."""
    n = int(state.hand_size)
    cards = [f"{int(c):>3d}" if v else "  ." for c, v in
             zip(state.hand[:n], state.hand_valid[:n])]
    ptr, size = int(state.deck_ptr), int(state.deck_size)
    upcoming = " ".join(str(int(c)) for c in state.deck[ptr:min(ptr + 8, size)])
    print(f"--- {label or 'state'} ---")
    print(f"  hand  [{' '.join(cards)}]   ({int(state.hand_valid.sum())}/{n} full)")
    print(f"  deck  ptr {ptr}/{size}, next: {upcoming}{' ...' if ptr + 8 < size else ''}")
    print(f"  hands {int(state.hands)}  discards {int(state.discards)}"
          f"  |  ante {int(state.ante)} round {int(state.round)}"
          f"  |  chips {int(state.chips_scored)}/{int(state.blind_target)}")

@jit
def deal(state : balatroEnvState):
    deal_count = state.hand_size - state.hand_valid.sum()
    empty = ~state.hand_valid
    queue = jnp.cumsum(empty)
    src = state.deck_ptr + queue - 1
    fill = empty & (queue - 1 < deal_count)

    new_hand = jnp.where(fill, state.deck[src], state.hand)
    new_hand_valid = (jnp.arange(MAX_HAND_SIZE) < state.hand_size) | fill

    return state._replace(
        hand=new_hand,
        hand_valid=new_hand_valid,
        deck_ptr=state.deck_ptr+deal_count
    )


if __name__ == "__main__":
    # dev scratch: `python env.py` runs deal on the test scenario from tests.py
    # (deck[i] = i+13 mod 52 — NOT identity, so src vs deck[src] mixups are
    # visible; opening 8 dealt = cards 13..20, lanes 2/5/6 discarded).
    # NOTE: comment out the @jit on deal while developing — plain print() then
    # shows real arrays instead of tracers.
    hand = jnp.full(MAX_HAND_SIZE, -1, dtype=jnp.int32).at[:8].set(
        jnp.array([13, 14, -1, 16, 17, -1, -1, 20], dtype=jnp.int32))
    hand_valid = jnp.zeros(MAX_HAND_SIZE, dtype=bool).at[:8].set(
        jnp.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=bool))
    pre = balatroEnvState(
        hand=hand,
        hand_size=jnp.int32(8),
        hand_valid=hand_valid,
        deck=jnp.concatenate([(jnp.arange(52, dtype=jnp.int32) + 13) % 52,
                              jnp.arange(52, MAX_DECK_SIZE, dtype=jnp.int32)]),
        deck_size=jnp.int32(52),
        deck_ptr=jnp.int32(8),
        hands=jnp.int32(3),
        discards=jnp.int32(2),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(60),
        blind_target=jnp.int32(300),
        key=jax.random.key(0),
    )

    got = deal(pre)
    if got is None:
        print("--- deal returned None (no return yet) ---")
    else:
        show(got, "deal returned")
    print("want hand[:8]:   [13 14 21 16 17 22 23 20], valid all 1, ptr 11")
