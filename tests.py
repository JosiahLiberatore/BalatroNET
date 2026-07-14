"""Test suites — scoreHand table today, env suites as the env grows."""

import jax
import jax.numpy as jnp

from scoring import scoreHand, HAND_NAMES
from env import balatroEnvState, deal, MAX_HAND_SIZE, MAX_DECK_SIZE

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
    ("split trips",    [11, 5, 11, 3, 11],  [0, 1, 2, 3, 1], [1, 1, 1, 1, 1], "three_kind",  [1, 0, 1, 0, 1], 180),
    ("mid kicker quads",[5, 5, 11, 5, 5],   [0, 1, 2, 3, 2], [1, 1, 1, 1, 1], "four_kind",   [1, 1, 0, 1, 1], 616),
    ("high card last", [1, 3, 7, 2, 12],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card",   [0, 0, 0, 0, 1], 16),
]


def run_tests():
    failures = []
    for name, r, s, v, expected, expected_mask, expected_score in TEST_HANDS:
        out = scoreHand(jnp.array(r), jnp.array(s), jnp.array(v))
        got = HAND_NAMES[out[0]]
        got_mask = out[1]
        got_score = int(out[2])
        type_ok = got == expected
        mask_ok = bool(jnp.array_equal(got_mask, jnp.array(expected_mask, dtype=bool)))
        score_ok = got_score == expected_score
        if not (type_ok and mask_ok and score_ok):
            got_m = "".join(str(int(m)) for m in got_mask)
            want_m = "".join(str(m) for m in expected_mask)
            failures.append(f"FAIL  {name:22s} type: {got:15s} (want {expected:15s}) "
                            f"mask: {got_m} (want {want_m}) score: {got_score:5d} (want {expected_score:5d})")
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)}/{len(TEST_HANDS)} scoreHand tests FAILED")
    else:
        print(f"scoreHand: all {len(TEST_HANDS)} tests passed")


def _hand_lanes(vals):
    """32-wide hand array from the first few lanes; -1 = empty lane."""
    a = jnp.full(MAX_HAND_SIZE, -1, dtype=jnp.int32)
    return a.at[:len(vals)].set(jnp.array(vals, dtype=jnp.int32))


def _valid_lanes(bits):
    """32-wide bool mask from the first few lanes; rest False."""
    m = jnp.zeros(MAX_HAND_SIZE, dtype=bool)
    return m.at[:len(bits)].set(jnp.array(bits, dtype=bool))


def run_deal_test():
    # Scenario: deck[i] = i + 13 (mod 52) — deliberately NOT the identity, so
    # confusing a deck POSITION (src) with the CARD at that position shows up
    # as visibly wrong values. Opening 8 cards were deck[0..7] = 13..20 and
    # deck_ptr sits at 8. The player then discarded lanes 2, 5, 6 — three
    # holes. deal must fill them from the top of the deck IN ORDER
    # (lane 2 <- deck[8]=21, lane 5 <- deck[9]=22, lane 6 <- deck[10]=23),
    # advance deck_ptr to 11, and touch NOTHING else.
    key = jax.random.key(0)
    pre = balatroEnvState(
        hand=_hand_lanes([13, 14, -1, 16, 17, -1, -1, 20]),
        hand_size=jnp.int32(8),
        hand_valid=_valid_lanes([1, 1, 0, 1, 1, 0, 0, 1]),
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
        key=key,
    )
    expected = pre._replace(
        hand=_hand_lanes([13, 14, 21, 16, 17, 22, 23, 20]),
        hand_valid=_valid_lanes([1, 1, 1, 1, 1, 1, 1, 1]),
        deck_ptr=jnp.int32(11),
    )

    got = deal(pre)
    failures = []
    for field in balatroEnvState._fields:
        g, e = getattr(got, field), getattr(expected, field)
        if field == "key":
            same = bool(jnp.array_equal(jax.random.key_data(g), jax.random.key_data(e)))
        else:
            same = bool(jnp.array_equal(g, e))
        if not same:
            failures.append(f"FAIL  deal: field '{field}'\n      got:  {g}\n      want: {e}")
    if failures:
        print("\n".join(failures))
        print(f"deal: {len(failures)} field(s) wrong")
    else:
        print("deal: refill test passed")


if __name__ == "__main__":
    run_tests()
    run_deal_test()
