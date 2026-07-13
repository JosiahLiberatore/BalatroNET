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


@jit
def deal(state : balatroEnvState):
    deal_count = state.hand_size - state.hand_valid.sum()
    empty = ~state.hand_valid

    queue = jnp.cumsum(empty)

    print(empty)



    # new_hand = hand_valid.aray

    # return state._replace(hand=new_hand)


    pass


if __name__ == "__main__":
    # dev scratch: `python env.py` runs deal on the test scenario from tests.py
    # (deck = identity order, opening 8 dealt, lanes 2/5/6 discarded).
    # NOTE: comment out the @jit on deal while developing — plain print() then
    # shows real arrays instead of tracers.
    hand = jnp.full(MAX_HAND_SIZE, -1, dtype=jnp.int32).at[:8].set(
        jnp.array([0, 1, -1, 3, 4, -1, -1, 7], dtype=jnp.int32))
    hand_valid = jnp.zeros(MAX_HAND_SIZE, dtype=bool).at[:8].set(
        jnp.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=bool))
    pre = balatroEnvState(
        hand=hand,
        hand_size=jnp.int32(8),
        hand_valid=hand_valid,
        deck=jnp.arange(MAX_DECK_SIZE, dtype=jnp.int32),
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
    print("--- deal returned ---")
    if got is None:
        print("None (deal has no return yet)")
    else:
        print("hand[:12]:      ", got.hand[:12])
        print("hand_valid[:12]:", got.hand_valid[:12].astype(int))
        print("deck_ptr:       ", got.deck_ptr)
        print("want hand[:8]:   [0 1 8 3 4 9 10 7], valid all 1, ptr 11")
