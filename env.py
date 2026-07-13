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
    hand_valid : jnp.array
    deck: jnp.ndarray
    deck_ptr : jnp.int32
    hands: jnp.int32
    discards: jnp.int32
    round: jnp.int32
    ante: jnp.int32
    chips_scored : jnp.int32
    blind_target : jnp.int32
    key: jax.random.key


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


def reset(key, deckSize):
    key, subkey = jax.random.split(key)

    deck = jax.random.permutation(subkey, MAX_DECK_SIZE)

    dead = deck >= deckSize
    deck = deck[jnp.argsort(dead)]

    hand_valid = jnp.arange(MAX_HAND_SIZE) < 8
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)

    result = balatroEnvState(deck = deck, deck_ptr=8, hand = hand, hand_valid=hand_valid,
                             hands=4, discards=3, ante=1, chips_scored=0, blind_target=300, round=1, key=key)

    return result


@jit
def deal(state):

    pass
