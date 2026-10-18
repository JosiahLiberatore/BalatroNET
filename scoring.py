"""Poker hand scoring — detection, scoring mask, chips x mult.

Encoding conventions (everything in the project honors these):
- rank: 0=two, 1=three, ... 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- a played hand is fixed shape (5,) + a valid mask (1-5 real cards)
"""

import jax
import jax.numpy as jnp
from jax import jit

from jokers import (Joker, C_NONE, C_SUIT, C_RANK_IN, C_IS_FACE, C_HAND_HAS, C_COUNT_CMP,
                    T_PER_SCORED, T_ONCE_MAIN, E_ADD_MULT, E_ADD_CHIPS, E_XMULT)

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

# CONTAINS[played, wanted]: does a played hand of type `played` contain a
# `wanted`? C_HAND_HAS reads this, not ==, so a Full House fires Jolly Joker
# (Pair), Zany Joker (Three of a Kind) and Mad Joker (Two Pair).
# Four of a Kind contains a Pair and a Three of a Kind, but not a Two Pair.
#                        SF 4oak full flush strt 3oak 2pair pair high
CONTAINS = jnp.array([[1,  0,   0,   1,   1,   0,   0,   0,   1],    # straight_flush
                      [0,  1,   0,   0,   0,   1,   0,   1,   1],    # four_kind
                      [0,  0,   1,   0,   0,   1,   1,   1,   1],    # full_house
                      [0,  0,   0,   1,   0,   0,   0,   0,   1],    # flush
                      [0,  0,   0,   0,   1,   0,   0,   0,   1],    # straight
                      [0,  0,   0,   0,   0,   1,   0,   1,   1],    # three_kind
                      [0,  0,   0,   0,   0,   0,   1,   1,   1],    # two_pair
                      [0,  0,   0,   0,   0,   0,   0,   1,   1],    # pair
                      [0,  0,   0,   0,   0,   0,   0,   0,   1]],   # high_card
                     bool)


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


# ---------------------------------------------------------------------------
# JOKER PASS — turn resolved joker values into chips and mult
# ---------------------------------------------------------------------------
# scoreHand returns the BASE (chips, mult) of the hand. This pass adds what the
# held jokers contribute. The split is why scoreHand stopped returning a single
# product: a joker adds mult AFTER the base chips are known, so the multiply
# must happen last.
#
# One generic reader, no per-joker branches. A slot contributes
#     v * fires
# where v comes from jokers.resolve() and fires is how many times the row
# triggers this hand.
#
# Order: every PER_SCORED add first (stage 4), then the ONCE_MAIN rows in SLOT
# order (stage 6). In stage 6 an add and an XMULT do not commute —
# (4+4)*3 != (4*3)+4 — so slot j's add is multiplied by every XMULT in a
# later slot:
#     mult = m0 * prod(x) + sum_j add_j * prod_{k>j} x_k
# x_k is 1 for every slot that is not a firing XMULT row.
#
# NOT SUPPORTED YET, by design:
#   T_PER_HELD          needs the held cards, which this function does not receive.
#   PER_SCORED E_XMULT  needs a (card x joker) scan in trigger order.
# Rows using either contribute nothing instead of contributing something wrong.

def joker_score(chips, mult, handtype, rank, suit, valid, scoring_mask, rows: Joker, active, v):
    """Add the held jokers' contribution. -> (chips, mult) as float32.

    chips, mult  : scalars              -- the base values from scoreHand
    handtype     : scalar int32         -- index into HAND_NAMES
    rank, suit   : (5,) int32           -- the played cards
    valid        : (5,) bool            -- which lanes hold a played card
    scoring_mask : (5,) bool            -- which played cards score
    rows         : Joker, (MAX_JOKERS,) -- jokers.rows_for(ids), slot order
    active       : (MAX_JOKERS,) bool   -- ids != J_EMPTY
    v            : (MAX_JOKERS,) float32 -- jokers.resolve() output
    """
    n_slots = rows.CONDITION.shape[0]
    shape = (n_slots, rank.shape[0])

    cond  = rows.CONDITION[:, None]     # (J, 1)
    param = rows.COND_PARAM[:, None]    # (J, 1)
    r, s  = rank[None, :], suit[None, :]  # (1, 5)
    bc = lambda x: jnp.broadcast_to(x, shape)

    # PER-CARD conditions: does slot j fire on card c?
    match = jnp.select(
        [bc(cond == C_NONE), bc(cond == C_SUIT),
         bc(cond == C_RANK_IN), bc(cond == C_IS_FACE)],
        [bc(True), bc(s == param),
         bc(((param >> r) & 1) != 0), bc((r >= 9) & (r <= 11))],
        default=bc(False))

    # GATE conditions: is slot j on at all for this hand?
    n_played = valid.astype(jnp.int32).sum()
    gate = jnp.select(
        [rows.CONDITION == C_NONE, rows.CONDITION == C_HAND_HAS,
         rows.CONDITION == C_COUNT_CMP],
        [jnp.ones(n_slots, bool),
         CONTAINS[handtype, jnp.clip(rows.COND_PARAM, 0, len(HAND_NAMES) - 1)],
         n_played <= rows.COND_PARAM],
        default=jnp.zeros(n_slots, bool))

    per_scored = rows.TRIGGER == T_PER_SCORED
    once_main = rows.TRIGGER == T_ONCE_MAIN
    fires = jnp.select(
        [per_scored, once_main],
        [(match & scoring_mask[None, :]).sum(axis=1), gate.astype(jnp.int32)],
        default=jnp.zeros(n_slots, jnp.int32))
    fires = jnp.where(active, fires, 0).astype(jnp.float32)

    contrib = v * fires
    add_chips = jnp.where(rows.EFFECT == E_ADD_CHIPS, contrib, 0.0)
    add_mult = jnp.where(rows.EFFECT == E_ADD_MULT, contrib, 0.0)

    # stage 4: the PER_SCORED adds
    chips = chips + jnp.where(per_scored, add_chips, 0.0).sum()
    mult = mult + jnp.where(per_scored, add_mult, 0.0).sum()

    # stage 6: the ONCE_MAIN rows in slot order. Chips only add.
    chips = chips + jnp.where(once_main, add_chips, 0.0).sum()
    x = jnp.where(once_main & (rows.EFFECT == E_XMULT) & (fires > 0), v, 1.0)
    after = jnp.concatenate([jnp.cumprod(x[::-1])[::-1][1:], jnp.ones(1)])   # prod_{k>j} x_k
    mult = mult * jnp.prod(x) + (jnp.where(once_main, add_mult, 0.0) * after).sum()
    return chips, mult
