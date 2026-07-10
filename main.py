"""BalatroNET — a JAX-native RL agent that plays and learns Balatro."""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax

# deck = {
#     'rank' : jnp.zeros((52, 13), dtype=int),
#     'suit' : jnp.zeros((52, 4), dtype=int),
#     # 'enhacement' : jnp.array((52, 5), dtype=int)
# }

# seed = 42
# key = random.key(seed)
# order = random.permutation(key, 52)
# ranks = order % 13
# suits = order // 13

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
    scoring_mask = valid

    top_ranks = jnp.sort(rank_counts)[::-1]
    top_suits = jnp.sort(suit_counts)[::-1]

    is_foak = top_ranks[0] >= 4
    is_full = (top_ranks[0] == 3) & (top_ranks[1] == 2)

    present = rank_counts > 0
    # add the ace as a low bit just for straight detection
    ace_low = jnp.concatenate([present[12:], present])
    is_straight = jnp.convolve(ace_low, jnp.ones(5)).max() == 5

    is_flush = top_suits[0] == 5
    is_toak = top_ranks[0] >= 3
    is_twopair = (top_ranks[0] >= 2) & (top_ranks[1] >= 2)
    is_pair = top_ranks[0] >= 2
    is_straight_flush = is_flush & is_straight

    handtype = jnp.argmax(jnp.array([is_straight_flush, is_foak, is_full, is_flush, is_straight, is_toak, is_twopair, is_pair, True]))

    # set the scoring mask (played cards that score)
    valid_mask = valid.astype(bool)
    mask_all = valid_mask
    mask_group = (rank_counts[rank] >= 2) & valid_mask
    highest_card = jnp.argmax(jnp.where(valid_mask, rank, -1))
    mask_high = jax.nn.one_hot(highest_card, 5).astype(bool) & valid_mask

    masks = jnp.stack([mask_all, mask_group, mask_high])
    scoring_mask = masks[MASK_CHOICE[handtype]]

    chips = (BASE_CARD_CHIPS[rank] * scoring_mask).sum() + BASE_CHIPS[handtype]
    mult = BASE_MULT[handtype]

    return handtype, scoring_mask, chips*mult



TEST_HANDS = [
    # (name, ranks, suits, valid, expected type, expected scoring mask, expected score)
    # score = (BASE_CHIPS[type] + sum of BASE_CARD_CHIPS[rank] over mask) * BASE_MULT[type]
    ("high card",      [12, 11, 7, 3, 1],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card",   [1, 0, 0, 0, 0], 16),
    ("pair",           [11, 11, 5, 2, 0],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",        [1, 1, 0, 0, 0], 60),
    ("two pair",       [11, 11, 5, 5, 0],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "two_pair",    [1, 1, 1, 1, 0], 108),
    ("three of a kind",[5, 5, 5, 11, 0],    [1, 0, 3, 2, 3], [1, 1, 1, 1, 1], "three_kind",  [1, 1, 1, 0, 0], 153),
    ("straight",       [3, 4, 5, 6, 7],     [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight",    [1, 1, 1, 1, 1], 260),
    ("ace-low straight",[12, 0, 1, 2, 3],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight",    [1, 1, 1, 1, 1], 220),
    ("flush",          [1, 4, 7, 9, 11],    [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], "flush",       [1, 1, 1, 1, 1], 292),
    ("full house",     [11, 11, 5, 5, 5],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "full_house",  [1, 1, 1, 1, 1], 324),
    ("four of a kind", [5, 5, 5, 5, 11],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "four_kind",   [1, 1, 1, 1, 0], 616),
    ("straight flush", [3, 4, 5, 6, 7],     [3, 3, 3, 3, 3], [1, 1, 1, 1, 1], "straight_flush", [1, 1, 1, 1, 1], 1080),
    # traps:
    ("2-card pair",    [11, 11, 0, 0, 0],   [2, 3, 0, 0, 0], [1, 1, 0, 0, 0], "pair",        [1, 1, 0, 0, 0], 60),
    ("4-card no-straight",[3, 4, 5, 6, 0],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 0], "high_card",   [0, 0, 0, 1, 0], 13),
    ("pair-in-run",    [3, 3, 4, 5, 6],     [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "pair",        [1, 1, 0, 0, 0], 40),
    # mask traps: padding slots share ranks with real cards — the & valid seatbelt
    ("deuce pair",     [0, 0, 0, 0, 0],     [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], "pair",        [1, 1, 0, 0, 0], 28),
    # mask trap: padding holds an ace — the where(valid, rank, -1) sentinel
    ("high card + junk ace", [7, 3, 12, 12, 12], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], "high_card", [1, 0, 0, 0, 0], 14),
    ("single card",    [9, 0, 0, 0, 0],     [2, 0, 0, 0, 0], [1, 0, 0, 0, 0], "high_card",   [1, 0, 0, 0, 0], 15),
    # position traps: groups scattered across slots — mask must follow the cards, not the slot order
    ("mid pair",       [7, 11, 3, 3, 1],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "pair",        [0, 0, 1, 1, 0], 40),
    ("split two pair", [5, 11, 5, 2, 11],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "two_pair",    [1, 1, 1, 0, 1], 108),
    ("split trips",    [11, 5, 11, 3, 11],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "three_kind",  [1, 0, 1, 0, 1], 180),
    ("mid kicker quads",[5, 5, 11, 5, 5],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "four_kind",   [1, 1, 0, 1, 1], 616),
    ("high card last", [1, 3, 7, 2, 12],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card",   [0, 0, 0, 0, 1], 16),
]

if __name__ == "__main__":
    for name, r, s, v, expected, expected_mask, expected_score in TEST_HANDS:
        out = scoreHand(jnp.array(r), jnp.array(s), jnp.array(v))
        got = HAND_NAMES[out[0]]
        got_mask = out[1]
        got_score = int(out[2])
        type_ok = got == expected
        mask_ok = bool(jnp.array_equal(got_mask, jnp.array(expected_mask, dtype=bool)))
        score_ok = got_score == expected_score
        status = "PASS" if (type_ok and mask_ok and score_ok) else "FAIL"
        got_m = "".join(str(int(m)) for m in got_mask)
        want_m = "".join(str(m) for m in expected_mask)
        print(f"{status}  {name:22s} type: {got:15s} (want {expected:15s}) mask: {got_m} (want {want_m}) score: {got_score:5d} (want {expected_score:5d})")
