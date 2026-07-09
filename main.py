"""BalatroNET — a JAX-native RL agent that plays and learns Balatro."""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax

# deck = {
#     'rank' : jnp.zeros((52, 13), dtype=int),
#     'suit' : jnp.zeros((52, 4), dtype=int),
#     # 'enhacement' : jnp.array((52, 5), dtype=int)
# }

seed = 42
key = random.key(seed)
order = random.permutation(key, 52)
ranks = order % 13
suits = order // 13


@jit
def scoreHand(rank, suit, valid):
    rank_counts = (jax.nn.one_hot(rank, 13) * valid[:, None]).sum(axis=0)
    suit_counts = (jax.nn.one_hot(suit, 4) * valid[:, None]).sum(axis=0)

    top_ranks = jnp.sort(rank_counts)[::-1]
    top_suits = jnp.sort(suit_counts)[::-1]

    is_foak = top_ranks[0] >= 4
    is_full = (top_ranks[0] == 3) & (top_ranks[1] == 2)

    # add the ace as a low bit just for straight detection
    present = rank_counts > 0
    ace_low = jnp.concatenate([present[12:], present])
    is_straight = jnp.convolve(ace_low, jnp.ones(5)).max() == 5

    is_flush = top_suits[0] == 5
    is_toak = top_ranks[0] >= 3
    is_twopair = (top_ranks[0] >= 2) & (top_ranks[1] >= 2)
    is_pair = top_ranks[0] >= 2
    is_straight_flush = is_flush & is_straight

    handtype = jnp.argmax(jnp.array([is_straight_flush, is_foak, is_full, is_flush, is_straight, is_toak, is_twopair, is_pair, True]))
    return handtype


# index i = position i in the argmax flag array above — keep the two in sync
HAND_NAMES = ["straight_flush", "four_kind", "full_house", "flush", "straight",
              "three_kind", "two_pair", "pair", "high_card"]


TEST_HANDS = [
    ("high card",      [12, 11, 7, 3, 1],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card"),
    ("pair",           [11, 11, 5, 2, 0],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair"),
    ("two pair",       [11, 11, 5, 5, 0],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "two_pair"),
    ("three of a kind",[5, 5, 5, 11, 0],    [1, 0, 3, 2, 3], [1, 1, 1, 1, 1], "three_kind"),
    ("straight",       [3, 4, 5, 6, 7],     [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight"),
    ("ace-low straight",[12, 0, 1, 2, 3],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight"),
    ("flush",          [1, 4, 7, 9, 11],    [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], "flush"),
    ("full house",     [11, 11, 5, 5, 5],   [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "full_house"),
    ("four of a kind", [5, 5, 5, 5, 11],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "four_kind"),
    ("straight flush", [3, 4, 5, 6, 7],     [3, 3, 3, 3, 3], [1, 1, 1, 1, 1], "straight_flush"),
    # traps:
    ("2-card pair",    [11, 11, 0, 0, 0],   [2, 3, 0, 0, 0], [1, 1, 0, 0, 0], "pair"),
    ("4-card no-straight",[3, 4, 5, 6, 0],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 0], "high_card"),
    ("pair-in-run",    [3, 3, 4, 5, 6],     [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "pair"),
]

if __name__ == "__main__":
    for name, r, s, v, expected in TEST_HANDS:
        got = HAND_NAMES[int(scoreHand(jnp.array(r), jnp.array(s), jnp.array(v)))]
        status = "PASS" if got == expected else "FAIL"
        print(f"{status}  {name:20s} got: {got:15s} expected: {expected}")
