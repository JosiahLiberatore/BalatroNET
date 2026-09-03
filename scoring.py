"""Poker hand scoring — detection, scoring mask, chips x mult.

Encoding conventions (everything in the project honors these):
- rank: 0=two, 1=three, ... 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- a played hand is fixed shape (5,) + a valid mask (1-5 real cards)
"""

import jax
import jax.numpy as jnp
from jax import jit

from jokers import (Joker, C_NONE, C_SUIT, C_RANK_IN, C_IS_FACE, C_HAND_HAS,
                    T_PER_SCORED, T_ONCE_MAIN, E_ADD_MULT, E_ADD_CHIPS)

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
# NOT SUPPORTED YET, by design:
#   T_PER_HELD  needs the held cards, which this function does not receive.
#   E_XMULT     needs the ordered scan — (4+4)*3 != (4*3)+4 — so it cannot be
#               a masked sum. The scan arrives with the first XMULT joker.
# Rows using either contribute nothing instead of contributing something wrong.

def joker_score(chips, mult, handtype, rank, suit, scoring_mask, rows: Joker, active, v):
    """Add the held jokers' contribution. -> (chips, mult) as float32.

    chips, mult  : scalars              -- the base values from scoreHand
    handtype     : scalar int32         -- index into HAND_NAMES
    rank, suit   : (5,) int32           -- the played cards
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
    gate = jnp.select(
        [rows.CONDITION == C_NONE, rows.CONDITION == C_HAND_HAS],
        [jnp.ones(n_slots, bool), handtype == rows.COND_PARAM],
        default=jnp.zeros(n_slots, bool))

    fires = jnp.select(
        [rows.TRIGGER == T_PER_SCORED, rows.TRIGGER == T_ONCE_MAIN],
        [(match & scoring_mask[None, :]).sum(axis=1), gate.astype(jnp.int32)],
        default=jnp.zeros(n_slots, jnp.int32))
    fires = jnp.where(active, fires, 0).astype(jnp.float32)

    contrib = v * fires
    chips = chips + jnp.where(rows.EFFECT == E_ADD_CHIPS, contrib, 0.0).sum()
    mult  = mult  + jnp.where(rows.EFFECT == E_ADD_MULT,  contrib, 0.0).sum()
    return chips, mult
