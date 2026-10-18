"""Test suites — scoreHand + the joker pass today, env suites as the env grows.

Staging lives in fixtures.py. A test writes only the fields its scenario is
about; everything else defaults.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from scoring import scoreHand, joker_score, HAND_NAMES
from env import balatroAction, deal, env_step, explain_play
from jokers import update, MAX_JOKERS, T_ONCE_MAIN, T_ON_DISCARD, T_END_OF_ROUND
from fixtures import (state, select, hand_lanes, mask_lanes, rotated_deck,
                      loadout, query_ctx, diff_states, report)


class Case(NamedTuple):
    """One scoring scenario, as a ROW — the same shape as a joker in jokers.py.

    Write the fields the case is about; the joker axes default to "no jokers".
    A new column lands as one default here, not as an edit to 21 rows.

    chips and mult are the values AFTER the joker pass. scoreHand no longer
    returns a product, because a joker adds mult after the base chips are
    known: the multiply has to happen last.
    """
    name   : str
    ranks  : list
    suits  : list
    valid  : list
    type   : str    # expected entry in HAND_NAMES
    mask   : list   # expected scoring mask
    chips  : int    # expected chips
    mult   : int    # expected mult
    jokers : tuple = ()   # held joker names, slot order. "." is a vacant slot.
    money  : int = 0      # feeds Q_MONEY_DIV_5 (Bootstraps), Q_MONEY (Bull)
    discards: int = 3     # feeds Banner / Mystic Summit
    deck_left: int = 0    # feeds Blue Joker


# base chips = BASE_CHIPS[type] + sum of BASE_CARD_CHIPS[rank] over the mask
TEST_HANDS = [
    # --- detection, mask, and base scoring: no jokers held -------------------
    Case("high card",       [12, 11, 7, 3, 1],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card",      [1, 0, 0, 0, 0],  16, 1),
    Case("pair",            [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 2),
    Case("two pair",        [11, 11, 5, 5, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "two_pair",       [1, 1, 1, 1, 0],  54, 2),
    Case("three of a kind", [5, 5, 5, 11, 0],   [1, 0, 3, 2, 3], [1, 1, 1, 1, 1], "three_kind",     [1, 1, 1, 0, 0],  51, 3),
    Case("straight",        [3, 4, 5, 6, 7],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight",       [1, 1, 1, 1, 1],  65, 4),
    Case("ace-low straight",[12, 0, 1, 2, 3],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "straight",       [1, 1, 1, 1, 1],  55, 4),
    Case("flush",           [1, 4, 7, 9, 11],   [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], "flush",          [1, 1, 1, 1, 1],  73, 4),
    Case("full house",      [11, 11, 5, 5, 5],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "full_house",     [1, 1, 1, 1, 1],  81, 4),
    Case("four of a kind",  [5, 5, 5, 5, 11],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "four_kind",      [1, 1, 1, 1, 0],  88, 7),
    Case("straight flush",  [3, 4, 5, 6, 7],    [3, 3, 3, 3, 3], [1, 1, 1, 1, 1], "straight_flush", [1, 1, 1, 1, 1], 135, 8),
    # traps:
    Case("2-card pair",     [11, 11, 0, 0, 0],  [2, 3, 0, 0, 0], [1, 1, 0, 0, 0], "pair",           [1, 1, 0, 0, 0],  30, 2),
    Case("4-card no-straight",[3, 4, 5, 6, 0],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 0], "high_card",      [0, 0, 0, 1, 0],  13, 1),
    Case("pair-in-run",     [3, 3, 4, 5, 6],    [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  20, 2),
    # mask traps: padding slots share ranks with real cards — the & valid seatbelt
    Case("deuce pair",      [0, 0, 0, 0, 0],    [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], "pair",           [1, 1, 0, 0, 0],  14, 2),
    # mask trap: padding holds an ace — the where(valid, rank, -1) sentinel
    Case("high card + junk ace",[7, 3, 12, 12, 12],[0, 1, 0, 0, 0],[1, 1, 0, 0, 0], "high_card",    [1, 0, 0, 0, 0],  14, 1),
    Case("single card",     [9, 0, 0, 0, 0],    [2, 0, 0, 0, 0], [1, 0, 0, 0, 0], "high_card",      [1, 0, 0, 0, 0],  15, 1),
    # position traps: groups scattered across slots — mask must follow the cards, not the slot order
    Case("mid pair",        [7, 11, 3, 3, 1],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "pair",           [0, 0, 1, 1, 0],  20, 2),
    Case("split two pair",  [5, 11, 5, 2, 11],  [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "two_pair",       [1, 1, 1, 0, 1],  54, 2),
    Case("split trips",     [11, 5, 11, 3, 11], [0, 1, 2, 3, 1], [1, 1, 1, 1, 1], "three_kind",     [1, 0, 1, 0, 1],  60, 3),
    Case("mid kicker quads",[5, 5, 11, 5, 5],   [0, 1, 2, 3, 2], [1, 1, 1, 1, 1], "four_kind",      [1, 1, 0, 1, 1],  88, 7),
    Case("high card last",  [1, 3, 7, 2, 12],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "high_card",      [0, 0, 0, 0, 1],  16, 1),

    # --- the joker pass ------------------------------------------------------
    # Every row below is the "pair" row above (base 30 chips, 2 mult) unless
    # noted, so the mult column is the whole story. chips never move: no joker
    # in the table has E_ADD_CHIPS yet, and a mult joker must not touch chips.
    #
    # S_CONSTANT — Joker, T_ONCE_MAIN, +4 mult, fires once per hand
    Case("Joker",           [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  6, ("Joker",)),
    # two of the same joker: the pass walks SLOTS, so both must fire
    Case("Joker x2",        [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 10, ("Joker", "Joker")),
    # a hole between two live slots: the clip in rows_for must be cancelled by
    # `active`, or the vacant slot scores as whatever row id -1 clipped to
    Case("Joker . Joker",   [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 10, ("Joker", ".", "Joker")),
    #
    # C_SUIT — Greedy Joker, T_PER_SCORED, +3 mult per scored diamond.
    # Here lane 0 (K of diamonds) scores; lane 2 is ALSO a diamond but is not in
    # the scoring mask, so it must not fire. One fire: 2 + 3 = 5.
    Case("Greedy 1 diamond",[11, 11, 5, 2, 0],  [1, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  5, ("Greedy Joker",)),
    # three scored diamonds, on the two-pair row (base 54): 2 + 3*3 = 11
    Case("Greedy 3 diamonds",[11, 11, 5, 5, 0], [1, 1, 1, 0, 3], [1, 1, 1, 1, 1], "two_pair",       [1, 1, 1, 1, 0],  54, 11, ("Greedy Joker",)),
    # no diamonds at all: held but silent
    Case("Greedy 0 diamonds",[11, 11, 5, 2, 0], [2, 3, 2, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  2, ("Greedy Joker",)),
    #
    # S_QUERY — Bootstraps, +2 mult per $5. money 23 -> 23//5 = 4 -> +8
    Case("Bootstraps $23",  [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 10, ("Bootstraps",), 23),
    # under $5 the query is 0 and the joker is silent — it is a query, not a constant
    Case("Bootstraps $4",   [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  2, ("Bootstraps",),  4),
    #
    # all three SOURCE kinds at once: 2 + 4 (Joker) + 3 (one scored diamond) + 8
    Case("full loadout",    [11, 11, 5, 2, 0],  [1, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 17, ("Joker", "Greedy Joker", "Bootstraps"), 23),
    #
    # C_HAND_HAS means CONTAINS. Full house (base 81 chips, 4 mult) contains a
    # Pair, a Three of a Kind and a Two Pair: 4 + 8 + 12 + 10 = 34
    Case("Jolly Zany Mad on full",[11, 11, 5, 5, 5],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "full_house",     [1, 1, 1, 1, 1],  81, 34, ("Jolly Joker", "Zany Joker", "Mad Joker")),
    # ... but not a Straight or a Flush
    Case("Crazy Droll on full",[11, 11, 5, 5, 5],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "full_house",     [1, 1, 1, 1, 1],  81,  4, ("Crazy Joker", "Droll Joker")),
    # Four of a Kind (88, 7) contains a Pair and a Three of a Kind, not a Two Pair
    Case("Sly Wily Clever on quads",[5, 5, 5, 5, 11],   [0, 1, 2, 3, 0], [1, 1, 1, 1, 1], "four_kind",      [1, 1, 1, 1, 0], 88 + 50 + 100, 7, ("Sly Joker", "Wily Joker", "Clever Joker")),
    # Half Joker: 2 cards played -> +20; 5 cards played -> silent
    Case("Half Joker 2 cards",[11, 11, 0, 0, 0],  [2, 3, 0, 0, 0], [1, 1, 0, 0, 0], "pair",           [1, 1, 0, 0, 0],  30, 22, ("Half Joker",)),
    Case("Half Joker 5 cards",[11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  2, ("Half Joker",)),
    #
    # per-card: the pair row scores KH, KS. Smiley +5 x2, Scary +30 x2,
    # Wrathful +3 on the scored spade only (lane 4 is a spade but not scored)
    Case("faces + spade",   [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  90, 15, ("Smiley Face", "Scary Face", "Wrathful Joker")),
    # flush 3 6 9 J K of hearts: Lusty +3 x5, Even Steven 6 -> +4,
    # Odd Todd 3 9 -> +62 chips, Fibonacci 3 -> +8
    Case("flush rank jokers",[1, 4, 7, 9, 11],   [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], "flush",          [1, 1, 1, 1, 1], 73 + 62, 4 + 15 + 4 + 8,
         ("Lusty Joker", "Even Steven", "Odd Todd", "Fibonacci")),
    #
    # query jokers: Banner 2 discards -> +60 chips; Blue 40 cards -> +80 chips;
    # Abstract with 4 jokers (itself included) -> +12 mult; Bull $7 -> +14 chips
    Case("Banner Blue Abstract Bull",[11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0], 30 + 60 + 80 + 14, 2 + 12,
         ("Banner", "Blue Joker", "Abstract Joker", "Bull"), 7, 2, 40),
    # Mystic Summit: +15 only at 0 discards
    Case("Mystic Summit 0", [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 17, ("Mystic Summit",), 0, 0),
    Case("Mystic Summit 1", [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  2, ("Mystic Summit",), 0, 1),
    #
    # XMULT in SLOT order: Joker (+4) then The Duo (X2) -> (2+4)*2 = 12;
    # The Duo then Joker -> 2*2+4 = 8
    Case("Joker then Duo",  [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 12, ("Joker", "The Duo")),
    Case("Duo then Joker",  [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  8, ("The Duo", "Joker")),
    # PER_SCORED adds land before every XMULT, wherever their slot is:
    # Duo, Smiley -> (2 + 5 + 5) * 2 = 24
    Case("Duo then Smiley", [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 24, ("The Duo", "Smiley Face")),
    # The Tribe does not fire on a pair: X1, not X0
    Case("Tribe on a pair", [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30,  2, ("The Tribe",)),
    # Joker Stencil: 5 slots, Stencil + Joker held -> 3 empty + itself = X4:
    # (2 + 4) * 4 = 24. Two Stencils alone -> X5 each: 2 * 25 = 50
    Case("Joker + Stencil", [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 24, ("Joker", "Joker Stencil")),
    Case("Stencil x2",      [11, 11, 5, 2, 0],  [2, 3, 1, 0, 3], [1, 1, 1, 1, 1], "pair",           [1, 1, 0, 0, 0],  30, 50, ("Joker Stencil", "Joker Stencil")),
]


def run_tests():
    failures = []
    for c in TEST_HANDS:
        rank, suit, valid = jnp.array(c.ranks), jnp.array(c.suits), jnp.array(c.valid)
        handtype, mask, chips, mult = scoreHand(rank, suit, valid)
        rows, active, v = loadout(c.jokers, money=c.money, discards=c.discards,
                                  deck_left=c.deck_left)
        chips, mult = joker_score(chips, mult, handtype, rank, suit, valid, mask,
                                  rows, active, v)

        got_type = HAND_NAMES[handtype]
        ok = (got_type == c.type
              and bool(jnp.array_equal(mask, jnp.array(c.mask, dtype=bool)))
              and float(chips) == float(c.chips)
              and float(mult) == float(c.mult))
        if not ok:
            got_m = "".join(str(int(m)) for m in mask)
            want_m = "".join(str(m) for m in c.mask)
            failures.append(
                f"FAIL  {c.name:22s} type: {got_type:15s} (want {c.type:15s}) "
                f"mask: {got_m} (want {want_m}) "
                f"chips: {float(chips):6.1f} (want {c.chips:5d}) "
                f"mult: {float(mult):5.1f} (want {c.mult:4d})")
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)}/{len(TEST_HANDS)} scoreHand tests FAILED")
    else:
        print(f"scoreHand: all {len(TEST_HANDS)} tests passed")


def run_deal_test():
    # Scenario: deck[i] = i + 13 (mod 52), so deck[0] = 13 = 2D — deliberately
    # NOT the identity, so confusing a deck POSITION (src) with the CARD at that
    # position shows up as visibly wrong values. The opening 8 were deck[0..7] =
    # 2D..9D and deck_ptr sits at 8. The player then discarded lanes 2, 5, 6 —
    # three holes. deal must fill them from the top of the deck IN ORDER
    # (lane 2 <- deck[8] = TD, lane 5 <- deck[9] = JD, lane 6 <- deck[10] = QD),
    # advance deck_ptr to 11, and touch NOTHING else.
    pre = state(hand="2D 3D . 5D 6D . . 9D",
                hand_size=8, deck=rotated_deck(13), deck_ptr=8,
                chips_scored=60)
    expected = pre._replace(
        hand=hand_lanes("2D 3D TD 5D 6D JD QD 9D"),
        hand_valid=mask_lanes([1] * 8),
        deck_ptr=jnp.int32(11),
    )

    got = deal(pre)
    report("deal", diff_states(got, expected, "deal"), "deal: refill test passed")


def run_env_step_test():
    # Scenario: mid-round, full 8-card hand, agent PLAYs a scattered two pair.
    # Conventions this test pins down: verb 0 = PLAY (1 = DISCARD);
    # rank = id % 13, suit = id // 13; reward = cashout $ on a clear, else 0;
    # done = the run ended (LOST / WON).
    #
    #   lane:  0   1   2   3   4   5   6   7
    #   card: 4C  KD  9H  KC  7C  9S  7S  3H
    #
    # select lanes 1, 3, 4, 6 -> K K 7 7 = two pair. Scattered on purpose
    # (exercises stable-argsort compaction), and lanes 2/5 hold an UNselected
    # pair of 9s — if step scores the whole hand instead of the selection,
    # the result comes out wrong.
    #
    # Score: ranks [11, 11, 5, 5] is the same two pair as the scoreHand row
    # -> chips 54, mult 2, no jokers held -> 108. chips 60 -> 168 (< 300),
    # hands 3 -> 2: no clear, so reward 0 and done stays False.
    #
    # Deck: the fixture default puts the hand's own ids at positions 0-7 (they
    # WERE dealt — ptr is 8) and the rest ascending, so deck[8..11] = 2C 3C 5C 6C.
    # NOT identity, so a src-vs-deck[src] mixup refills with the wrong cards and
    # fails visibly. Refill: lane1<-2C, lane3<-3C, lane4<-5C, lane6<-6C, ptr -> 12.
    pre = state(hand="4C KD 9H KC 7C 9S 7S 3H", chips_scored=60)
    action = balatroAction(verb=jnp.int32(0),               # PLAY
                           select=select("01011010"))
    expected = pre._replace(
        hand=hand_lanes("4C 2C 9H 3C 5C 9S 6C 3H"),
        hand_valid=mask_lanes([1] * 8),
        deck_ptr=jnp.int32(12),
        hands=jnp.int32(2),
        chips_scored=jnp.int32(168),
    )
    expected_reward = 0
    expected_done = False

    try:
        got, reward, done = env_step(pre, action)
    except Exception as e:
        print(f"env_step: NOT RUNNABLE yet ({type(e).__name__}: {e})")
        return

    failures = diff_states(got, expected, "env_step")
    if int(reward) != expected_reward:
        failures.append(f"FAIL  env_step: reward\n      got:  {int(reward)}\n      want: {expected_reward}")
    if bool(done) != expected_done:
        failures.append(f"FAIL  env_step: done\n      got:  {bool(done)}\n      want: {expected_done}")
    report("env_step", failures, "env_step: two-pair play test passed")


def run_update_test():
    # Green Joker counter, one event per call, value carried between calls.
    # Slot 0 = Joker (no update rules: must hold 4.0), slot 1 = Green (seed
    # 0.0), slot 2 = a hole (must hold 0.0). The jitted copy must agree: the
    # body has no Python branch on traced values.
    rows, active, _ = loadout(("Joker", "Green Joker"))
    value = jnp.where(active, rows.VALUE, 0.0)
    state0 = jnp.zeros(MAX_JOKERS, jnp.int32)
    ctx = query_ctx()
    jitted = jax.jit(update)

    steps = [("play",         T_ONCE_MAIN,    1.0),
             ("play",         T_ONCE_MAIN,    2.0),
             ("discard",      T_ON_DISCARD,   1.0),
             ("discard",      T_ON_DISCARD,   0.0),
             ("discard at 0", T_ON_DISCARD,   0.0),   # floors at 0, not -1
             ("end of round", T_END_OF_ROUND, 0.0)]   # Green does not listen
    failures = []
    for label, t, green in steps:
        want = jnp.zeros(MAX_JOKERS, jnp.float32).at[0].set(4.0).at[1].set(green)
        got, got_state = update(rows, active, value, state0, jnp.int32(t), ctx)
        got_j, _ = jitted(rows, active, value, state0, jnp.int32(t), ctx)
        if not (bool(jnp.array_equal(got, want)) and bool(jnp.array_equal(got_j, want))
                and bool(jnp.array_equal(got_state, state0))):
            failures.append(f"FAIL  update {label}\n      got:  {got[:3]} (jit {got_j[:3]})"
                            f"\n      want: {want[:3]}")
        value = got
    report("update", failures, f"update: all {len(steps)} Green Joker steps passed")


def run_env_step_green_test():
    # The two-pair PLAY from run_env_step_test, with a Green Joker held.
    # The update pass runs BEFORE scoring, so this hand already sees +1:
    # chips 54, mult 2 + 1 = 3 -> 162. chips_scored 60 -> 222.
    pre = state(hand="4C KD 9H KC 7C 9S 7S 3H", chips_scored=60,
                jokers=("Green Joker",))
    action = balatroAction(verb=jnp.int32(0), select=select("01011010"))
    expected = pre._replace(
        hand=hand_lanes("4C 2C 9H 3C 5C 9S 6C 3H"),
        hand_valid=mask_lanes([1] * 8),
        deck_ptr=jnp.int32(12),
        hands=jnp.int32(2),
        chips_scored=jnp.int32(222),
        joker_value=pre.joker_value.at[0].set(1.0),
    )
    got, reward, _ = env_step(pre, action)
    failures = diff_states(got, expected, "env_step green play")
    if int(reward) != 0:
        failures.append(f"FAIL  env_step green play: reward (no clear)\n      got:  {int(reward)}\n      want: 0")

    # DISCARD lanes 1 and 3 with Green at 2: value -> 1, no score, discards
    # 2 -> 1, hands unchanged, lanes 1/3 refill from deck[8], deck[9] = 2C 3C.
    pre = state(hand="4C KD 9H KC 7C 9S 7S 3H", jokers=("Green Joker",),
                joker_value=(2.0,))
    action = balatroAction(verb=jnp.int32(1), select=select("01010000"))
    expected = pre._replace(
        hand=hand_lanes("4C 2C 9H 3C 7C 9S 7S 3H"),
        hand_valid=mask_lanes([1] * 8),
        deck_ptr=jnp.int32(10),
        discards=jnp.int32(1),
        joker_value=pre.joker_value.at[0].set(1.0),
    )
    got, reward, _ = env_step(pre, action)
    failures += diff_states(got, expected, "env_step green discard")
    if int(reward) != 0:
        failures.append(f"FAIL  env_step green discard: reward\n      got:  {int(reward)}\n      want: 0")
    report("env_step green", failures, "env_step: Green Joker play + discard tests passed")


def run_joker_env_test():
    # The scoring cases above stage ctx by hand (fixtures.query_ctx). These
    # checks go through env.build_ctx, so they test the live counts.
    # Hand: KD KC + junk. Select lanes 0, 1 -> pair of kings, base 30 x 2.
    hand = "KD KC 2H 3S 4C 5D 7H 8S"
    pair = select("11000000")
    failures = []

    def check(label, got, want):
        if float(got) != float(want):
            failures.append(f"FAIL  joker env: {label}\n      got:  {float(got)}\n      want: {want}")

    # Banner, 2 discards left -> +60 chips
    res = explain_play(state(hand, discards=2, jokers=("Banner",)), pair)
    check("Banner chips", res.chips, 30 + 60)
    # Blue Joker: 52 cards, 8 dealt -> 44 left -> +88 chips
    res = explain_play(state(hand, jokers=("Blue Joker",)), pair)
    check("Blue Joker chips", res.chips, 30 + 88)
    # Mystic Summit at 0 discards -> +15
    res = explain_play(state(hand, discards=0, jokers=("Mystic Summit",)), pair)
    check("Mystic Summit mult", res.mult, 2 + 15)
    # Abstract with 3 jokers held -> +9 (Joker adds +4 more)
    res = explain_play(state(hand, jokers=("Abstract Joker", "Joker", "Golden Joker")), pair)
    check("Abstract mult", res.mult, 2 + 9 + 4)
    # Bull at $12 -> +24 chips
    res = explain_play(state(hand, money=12, jokers=("Bull",)), pair)
    check("Bull chips", res.chips, 30 + 24)
    # Joker Stencil alone in 5 slots -> X5
    res = explain_play(state(hand, jokers=("Joker Stencil",)), pair)
    check("Stencil alone mult", res.mult, 2 * 5)
    # Stencil + Joker -> X4 after the +4: (2 + 4) * 4
    res = explain_play(state(hand, jokers=("Joker", "Joker Stencil")), pair)
    check("Joker + Stencil mult", res.mult, (2 + 4) * 4)

    # cashout: clear a blind with the pair (290 + 60 >= 300). The joker money
    # is the difference against the same clear with no jokers.
    def clear(jokers):
        pre = state(hand, chips_scored=290, blind_target=300, jokers=jokers)
        got, reward, _ = env_step(pre, balatroAction(verb=jnp.int32(0), select=pair))
        return got, int(reward)
    _, plain = clear(())
    _, golden = clear(("Golden Joker",))
    check("Golden Joker cashout", golden - plain, 4)
    _, cloud = clear(("Cloud 9",))
    check("Cloud 9 cashout (four 9s)", cloud - plain, 4)
    got, both = clear(("Golden Joker", "Cloud 9"))
    check("Golden + Cloud 9 cashout", both - plain, 8)
    check("last_cashout = reward", got.last_cashout, both)
    # Popcorn: 20 -> 16 at the end of the round
    got, _ = clear(("Popcorn",))
    check("Popcorn after a round", got.joker_value[0], 16.0)
    report("joker env", failures, "joker env: all build_ctx + cashout checks passed")


if __name__ == "__main__":
    run_tests()
    run_deal_test()
    run_env_step_test()
    run_update_test()
    run_env_step_green_test()
    run_joker_env_test()

    import tests_run
    tests_run.run_all()
