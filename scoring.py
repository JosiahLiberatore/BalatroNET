"""Poker hand scoring — detection, scoring mask, chips x mult.

Encoding conventions (everything in the project honors these):
- rank: 0=two, 1=three, ... 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- a played hand is fixed shape (5,) + a valid mask (1-5 real cards)
"""

import jax
import jax.numpy as jnp
from jax import jit

# which candidate mask each hand type uses: 0=all, 1=groups, 2=high
MASK_CHOICE = jnp.array([0, 1, 0, 0, 0, 1, 1, 1, 2])

# index-aligned with HAND_NAMES (and the argmax flag array, and MASK_CHOICE)
#                       SF  4oak  full  flush strt  3oak  2pair pair  high
BASE_CHIPS = jnp.array([100,  60,   40,   35,   30,   30,   20,   10,   5])
BASE_MULT  = jnp.array([  8,   7,    4,    4,    4,    3,    2,    2,   1])

BASE_CARD_CHIPS = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

# index i = position i in the argmax flag array above — keep the two in sync
HAND_NAMES = ["straight_flush", "four_kind", "full_house", "flush", "straight",
              "three_kind", "two_pair", "pair", "high_card"]


@jit
def scoreHand(rank, suit, valid):
    rank_counts = (jax.nn.one_hot(rank, 13) * valid[:, None]).sum(axis=0)
    suit_counts = (jax.nn.one_hot(suit, 4) * valid[:, None]).sum(axis=0)

    top_ranks = jnp.sort(rank_counts)[::-1]
    top_suits = jnp.sort(suit_counts)[::-1]

    is_4oak = top_ranks[0] >= 4
    is_full = (top_ranks[0] == 3) & (top_ranks[1] == 2)

    present = rank_counts > 0
    # add the ace as a low bit just for straight detection
    ace_low = jnp.concatenate([present[12:], present])
    is_straight = jnp.convolve(ace_low, jnp.ones(5)).max() == 5

    is_flush = top_suits[0] == 5
    is_3oak = top_ranks[0] >= 3
    is_twopair = (top_ranks[0] >= 2) & (top_ranks[1] >= 2)
    is_pair = top_ranks[0] >= 2
    is_straight_flush = is_flush & is_straight

    handtype = jnp.argmax(jnp.array([is_straight_flush, is_4oak, is_full, is_flush, is_straight, is_3oak, is_twopair, is_pair, True]))

    # set the scoring mask (played cards that score)
    mask_all = valid.astype(bool)
    mask_group = (rank_counts[rank] >= 2) & mask_all
    highest_card = jnp.argmax(jnp.where(mask_all, rank, -1))
    mask_high = jax.nn.one_hot(highest_card, 5).astype(bool) & mask_all

    masks = jnp.stack([mask_all, mask_group, mask_high])
    scoring_mask = masks[MASK_CHOICE[handtype]]

    chips = (BASE_CARD_CHIPS[rank] * scoring_mask).sum() + BASE_CHIPS[handtype]
    mult = BASE_MULT[handtype]

    return handtype, scoring_mask, chips, mult
