"""Joker test suite — proves every joker against an independent reference.

REF below is a second implementation of the jokers, written from the in-game
text in plain Python. It does not read the jokers.py columns or scoring.py:
hand detection, the scoring order, the counters and the cashout money are all
re-derived here. Each layer compares the env against it:

  table  every TABLE joker has a REF entry and a spec case. A new joker fails
         here until it has both.
  spec   hand-computed cases per joker (explain_play), fire and no-fire. They
         also check REF itself, so the layers below can trust it.
  order  every slot order of a 5-joker add / XMULT mix, and MOVE_JOKER -> PLAY.
  fuzz   random hands x random loadouts x random live state, vmapped.
  life   scripted env_step sequences: counters, live queries mid-blind, the
         cashout money, counters that follow MOVE / SELL / BUY.
  soak   vmapped random runs. Every step of every env goes through the
         reference transition model (score, counters, money, stage, reward).

Run: python tests_jokers.py   (tests.py also runs it)
"""

import itertools
import math
import time
from collections import Counter
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from env import (balatroAction, env_step, explain_play, reset, invariant_errors,
                 MAX_HAND_SIZE, MAX_SHOP_SLOTS, V_PLAY, V_DISCARD, V_BUY, V_SELL, V_REROLL,
                 V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, STAGE_BLIND, STAGE_SHOP, STAGE_LOST,
                 STAGE_WON, BLIND_BOSS)
from jokers import TABLE, JOKER_NAMES, JOKER_ID, JOKERS, JOKER_COST, MAX_JOKERS, N_JOKERS
from scoring import HAND_NAMES
from fixtures import state, select, cards as card_ids
from tests_run import Checks, act


# ===========================================================================
# THE REFERENCE — Balatro rules, plain Python
# ===========================================================================
TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(13)
CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
CARD_CHIPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
# best first; (base chips, base mult)
HAND_BASE = {"straight_flush": (100, 8), "four_kind": (60, 7), "full_house": (40, 4),
             "flush": (35, 4), "straight": (30, 4), "three_kind": (30, 3),
             "two_pair": (20, 2), "pair": (10, 2), "high_card": (5, 1)}
BLIND_PAY = [3, 4, 5]          # small / big / boss
WIN_ANTE = 8


def ref_hand(cards):
    """Played card ids (lane order) -> (hand name, contained hand names, scored ids)."""
    ranks, suits = [c % 13 for c in cards], [c // 13 for c in cards]
    cnt = Counter(ranks)
    top = sorted(cnt.values(), reverse=True) + [0, 0]
    flush = len(cards) == 5 and len(set(suits)) == 1
    straight = len(cards) == 5 and len(cnt) == 5 and (
        max(ranks) - min(ranks) == 4 or set(ranks) == {ACE, TWO, THREE, FOUR, FIVE})
    has = {"high_card"}
    if top[0] >= 2: has.add("pair")
    if top[0] >= 3: has.add("three_kind")
    if top[0] >= 4: has.add("four_kind")
    if sum(v >= 2 for v in cnt.values()) >= 2: has.add("two_pair")   # two DIFFERENT ranks
    if top[0] == 3 and top[1] == 2: has.add("full_house")
    if flush: has.add("flush")
    if straight: has.add("straight")
    if flush and straight: has.add("straight_flush")
    hand = next(h for h in HAND_BASE if h in has)
    if hand in ("straight_flush", "full_house", "flush", "straight"):
        scored = list(cards)
    elif hand == "high_card":
        scored = [max(cards, key=lambda c: c % 13)] if cards else []
    else:
        scored = [c for c in cards if cnt[c % 13] >= 2]
    return hand, has, scored


class Ref(NamedTuple):
    """One joker, as the reference sees it. Every field is optional."""
    card: Optional[Callable] = None    # (rank, suit) -> (chips, mult), per scored card
    main: Optional[Callable] = None    # h -> [(op, x)], op in chips / mult / xmult; slot order
    money: Optional[Callable] = None   # ctx -> $ at the end of the round
    seed: Optional[float] = None       # a counter: its value on buy
    on_play: float = 0.0               # counter deltas, floored at 0
    on_discard: float = 0.0
    on_round: float = 0.0


def per_card(test, chips=0, mult=0):
    return lambda r, s: (chips, mult) if test(r, s) else (0, 0)


def when(test, op, x):
    return lambda h: [(op, x)] if test(h) else []


def has(name):
    return lambda h: name in h["has"]


suit_is = lambda want: (lambda r, s: s == want)
rank_in = lambda *want: (lambda r, s: r in want)
is_face = lambda r, s: r in (JACK, QUEEN, KING)

REF = {
    "Joker":            Ref(main=lambda h: [("mult", 4)]),
    "Greedy Joker":     Ref(card=per_card(suit_is(DIAMONDS), mult=3)),
    "Bootstraps":       Ref(main=lambda h: [("mult", 2 * (h["ctx"]["money"] // 5))]),
    "Green Joker":      Ref(main=lambda h: [("mult", h["v"])], seed=0.0, on_play=+1, on_discard=-1),
    "Jolly Joker":      Ref(main=when(has("pair"), "mult", 8)),
    "Zany Joker":       Ref(main=when(has("three_kind"), "mult", 12)),
    "Mad Joker":        Ref(main=when(has("two_pair"), "mult", 10)),
    "Crazy Joker":      Ref(main=when(has("straight"), "mult", 12)),
    "Droll Joker":      Ref(main=when(has("flush"), "mult", 10)),
    "Sly Joker":        Ref(main=when(has("pair"), "chips", 50)),
    "Wily Joker":       Ref(main=when(has("three_kind"), "chips", 100)),
    "Clever Joker":     Ref(main=when(has("two_pair"), "chips", 80)),
    "Devious Joker":    Ref(main=when(has("straight"), "chips", 100)),
    "Crafty Joker":     Ref(main=when(has("flush"), "chips", 80)),
    "Half Joker":       Ref(main=when(lambda h: h["n_played"] <= 3, "mult", 20)),
    "Lusty Joker":      Ref(card=per_card(suit_is(HEARTS), mult=3)),
    "Wrathful Joker":   Ref(card=per_card(suit_is(SPADES), mult=3)),
    "Gluttonous Joker": Ref(card=per_card(suit_is(CLUBS), mult=3)),
    "Even Steven":      Ref(card=per_card(rank_in(TEN, EIGHT, SIX, FOUR, TWO), mult=4)),
    "Odd Todd":         Ref(card=per_card(rank_in(ACE, NINE, SEVEN, FIVE, THREE), chips=31)),
    "Scary Face":       Ref(card=per_card(is_face, chips=30)),
    "Smiley Face":      Ref(card=per_card(is_face, mult=5)),
    "Fibonacci":        Ref(card=per_card(rank_in(ACE, TWO, THREE, FIVE, EIGHT), mult=8)),
    "Arrowhead":        Ref(card=per_card(suit_is(SPADES), chips=50)),
    "Onyx Agate":       Ref(card=per_card(suit_is(CLUBS), mult=7)),
    "Banner":           Ref(main=lambda h: [("chips", 30 * h["ctx"]["discards"])]),
    "Mystic Summit":    Ref(main=when(lambda h: h["ctx"]["discards"] == 0, "mult", 15)),
    "Blue Joker":       Ref(main=lambda h: [("chips", 2 * h["ctx"]["deck_left"])]),
    "Abstract Joker":   Ref(main=lambda h: [("mult", 3 * h["ctx"]["n_jokers"])]),
    "Bull":             Ref(main=lambda h: [("chips", 2 * h["ctx"]["money"])]),
    "Joker Stencil":    Ref(main=lambda h: [("xmult", h["ctx"]["stencil_x"])]),
    "Popcorn":          Ref(main=lambda h: [("mult", h["v"])], seed=20.0, on_round=-4),
    "Golden Joker":     Ref(money=lambda ctx: 4),
    "Cloud 9":          Ref(money=lambda ctx: ctx["nines"]),
    "The Duo":          Ref(main=when(has("pair"), "xmult", 2)),
    "The Trio":         Ref(main=when(has("three_kind"), "xmult", 3)),
    "The Family":       Ref(main=when(has("four_kind"), "xmult", 4)),
    "The Order":        Ref(main=when(has("straight"), "xmult", 3)),
    "The Tribe":        Ref(main=when(has("flush"), "xmult", 2)),
}
NAME = JOKER_NAMES          # id -> name; REF is keyed by name, never by id


def ref_ctx(s):
    """The live counts the query jokers read, from a host state."""
    held = [int(j) for j in s.jokers if j >= 0]
    size = int(s.deck_size)
    live = [int(c) for c in s.deck[:size] if 0 <= c < 52]
    stencils = sum(NAME[j] == "Joker Stencil" for j in held)
    return dict(money=int(s.money), discards=max(int(s.discards), 0),
                deck_left=max(size - int(s.deck_ptr), 0), n_jokers=len(held),
                stencil_x=int(s.joker_slots) - len(held) + stencils,   # a Stencil counts as empty
                nines=sum(c % 13 == NINE for c in live))


def ref_counter(jokers, values, event):
    """Counter values after an event (on_play / on_discard / on_round). Floored at 0."""
    out = []
    for j, v in zip(jokers, values):
        d = getattr(REF[NAME[j]], event)
        out.append(max(v + d, 0.0) if d else v)
    return out


def ref_score(cards, jokers, values, ctx):
    """-> (chips, mult, score, slots that changed the score). Balatro order:
    each scored card in play order (its chips, then every joker's per-card
    effect in slot order), then every joker's main effect in slot order."""
    hand, contains, scored = ref_hand(cards)
    chips, mult = HAND_BASE[hand]
    acted = set()
    for c in scored:
        chips += CARD_CHIPS[c % 13]
        for i, j in enumerate(jokers):
            f = REF[NAME[j]].card
            if f:
                dc, dm = f(c % 13, c // 13)
                chips, mult = chips + dc, mult + dm
                if dc or dm:
                    acted.add(i)
    h = dict(hand=hand, has=contains, n_played=len(cards), ctx=ctx)
    for i, (j, v) in enumerate(zip(jokers, values)):
        f = REF[NAME[j]].main
        for op, x in (f({**h, "v": v}) if f else []):
            if op == "chips":
                chips += x
            elif op == "mult":
                mult += x
            else:
                mult *= x
            if x != (1 if op == "xmult" else 0):
                acted.add(i)
    return chips, mult, math.floor(chips * mult), acted


def played_lanes(s, sel):
    """The env's repair: the first 5 selected lanes that hold a card, lane order."""
    return [l for l in range(MAX_HAND_SIZE) if sel[l] and s.hand_valid[l]][:5]


def ref_play(s, sel):
    """Score a PLAY from a host state. -> (chips, mult, score, hand name, new values)."""
    jokers = [int(j) for j in s.jokers if j >= 0]
    values = ref_counter(jokers, [float(v) for v in s.joker_value[:len(jokers)]], "on_play")
    cards = [int(s.hand[l]) for l in played_lanes(s, sel)]
    chips, mult, score, _ = ref_score(cards, jokers, values, ref_ctx(s))
    return chips, mult, score, ref_hand(cards)[0], values


def exact(got, want):
    """float32 is exact below 2**24; above it, allow its rounding."""
    return got == want if abs(want) < 2 ** 24 else abs(got - want) <= abs(want) * 2 ** -21


def host(s):
    """Device state -> numpy state (the PRNG key as raw words)."""
    return jax.device_get(s._replace(key=jax.random.key_data(s.key)))


# ===========================================================================
# THE REFERENCE TRANSITION — what one env_step must do, field by field
# ===========================================================================
LEGAL_NP = np.array([
    #  PLAY  DISC   BUY    SELL   REROLL NEXT   MOVE_J MOVE_C
    [True,  True,  False, True,  False, False, True,  True ],   # BLIND
    [False, False, True,  True,  True,  True,  True,  False],   # SHOP
    [False] * 8, [False] * 8])                                    # LOST, WON


def npeq(c, name, got, want):
    """Checks.eq without the device round trip (the soak calls it ~10**5 times)."""
    c.check(name, np.array_equal(np.asarray(got), np.asarray(want)), got, want)


def check_step(c, tag, p, a, q, reward):
    """Compare post-state q against the reference model of (p, a).
    p, q are host states; a is a host balatroAction. -> the effective verb."""
    verb, t = int(a.verb), int(a.target)
    sel = np.asarray(a.select).astype(bool)
    stage = int(p.game_stage)
    if verb == V_DISCARD and int(p.discards) <= 0:
        verb = V_PLAY                                       # DISCARD at 0 left is a PLAY
    legal = 0 <= verb < 8 and LEGAL_NP[min(max(stage, 0), 3), verb]

    errs = invariant_errors(q)
    c.check(f"{tag}: invariants", not errs, errs)
    if not legal:
        same = all(np.array_equal(getattr(p, f), getattr(q, f)) for f in p._fields)
        c.check(f"{tag}: illegal verb {int(a.verb)} in stage {stage} is a no-op", same)
        npeq(c, f"{tag}: no-op reward", reward, 0)
        return None

    held = [int(j) for j in p.jokers if j >= 0]
    n = len(held)
    J, V, S = list(p.jokers), [float(v) for v in p.joker_value], list(p.joker_sell_value)
    want = dict(money=int(p.money), chips_scored=int(p.chips_scored), hands=int(p.hands),
                discards=int(p.discards), game_stage=stage, last_cashout=int(p.last_cashout))
    want_reward = 0
    hand = None                                             # set when the hand lanes are modeled

    if verb == V_PLAY:
        chips, mult, score, name, vals = ref_play(p, sel)
        V[:n] = vals
        got = int(q.chips_scored) - int(p.chips_scored)
        ok = exact(got, score)
        c.check(f"{tag}: PLAY {name} scores {score} ({chips} x {mult})", ok, got, score)
        total = int(p.chips_scored) + (got if ok else score)   # float32 rounding above 2**24
        want.update(chips_scored=total, hands=int(p.hands) - 1)
        if total >= int(p.blind_target):
            V[:n] = ref_counter(held, V[:n], "on_round")
            ctx = ref_ctx(q)                                # the full deck: same as before the deal
            money_j = sum(REF[NAME[j]].money(ctx) for j in held if REF[NAME[j]].money)
            cashout = (BLIND_PAY[int(p.blind_index)] + int(p.hands) - 1
                       + min(int(p.money) // 5, 5) + money_j)
            won = int(p.blind_index) == BLIND_BOSS and int(p.ante) >= WIN_ANTE
            want.update(money=int(p.money) + cashout, last_cashout=cashout,
                        game_stage=STAGE_WON if won else STAGE_SHOP)
            want_reward = cashout
        elif int(p.hands) - 1 <= 0:
            want["game_stage"] = STAGE_LOST
    elif verb == V_DISCARD:
        V[:n] = ref_counter(held, V[:n], "on_discard")
        want["discards"] = int(p.discards) - 1
    elif verb == V_BUY:
        jid = int(p.shop_slots[t]) if 0 <= t < MAX_SHOP_SLOTS else -1
        cost = int(JOKER_COST[jid]) if jid >= 0 else 0
        if 0 <= t < int(p.shop_size) and jid >= 0 and int(p.money) >= cost and n < int(p.joker_slots):
            J[n], S[n] = jid, max(1, cost // 2)
            seed = REF[NAME[jid]].seed
            V[n] = float(JOKERS.VALUE[jid]) if seed is None else seed
            want["money"] = int(p.money) - cost
            npeq(c, f"{tag}: BUY empties the shop slot", q.shop_slots[t], -1)
    elif verb == V_SELL:
        if 0 <= t < MAX_JOKERS and int(p.jokers[t]) >= 0:
            want["money"] = int(p.money) + int(p.joker_sell_value[t])
            for arr, empty in ((J, -1), (V, 0.0), (S, 0)):
                del arr[t]
                arr.append(empty)
    elif verb == V_REROLL:
        if int(p.money) >= int(p.reroll_cost):
            want["money"] = int(p.money) - int(p.reroll_cost)
    elif verb == V_NEXT:
        want.update(chips_scored=0, hands=int(p.handcount), discards=int(p.discardcount),
                    game_stage=STAGE_BLIND)
    elif verb == V_MOVE_JOKER:
        if 0 <= t < MAX_JOKERS - 1 and int(p.jokers[t]) >= 0 and int(p.jokers[t + 1]) >= 0:
            for arr in (J, V, S):
                arr[t], arr[t + 1] = arr[t + 1], arr[t]
    elif verb == V_MOVE_CARD:
        hand, valid = list(p.hand), list(p.hand_valid)
        if 0 <= t and t + 1 < int(p.hand_size) and (valid[t] or valid[t + 1]):
            for arr in (hand, valid):
                arr[t], arr[t + 1] = arr[t + 1], arr[t]
        npeq(c, f"{tag}: MOVE_CARD hand", q.hand, np.array(hand))
        npeq(c, f"{tag}: MOVE_CARD valid", q.hand_valid, np.array(valid))

    npeq(c, f"{tag}: jokers", q.jokers, np.array(J, np.int32))
    c.check(f"{tag}: joker values", np.allclose(q.joker_value, V, rtol=0, atol=1e-6), q.joker_value[:8], V[:8])
    npeq(c, f"{tag}: sell values", q.joker_sell_value, np.array(S, np.int32))
    for f, w in want.items():
        npeq(c, f"{tag}: {f}", getattr(q, f), w)
    npeq(c, f"{tag}: reward", reward, want_reward)
    return verb


def drive(c, tag, s, actions):
    """Step a scripted action list; check every step. -> the final state."""
    for i, a in enumerate(actions):
        post, r, _ = env_step(s, a)
        check_step(c, f"{tag} #{i}", host(s), jax.device_get(a), host(post), int(r))
        s = post
    return s


# ===========================================================================
# TABLE + SPEC — hand-computed cases, one block per joker
# ===========================================================================
# Each hand: 8 lanes, then the selected lanes. Base chips x mult with no joker:
HANDS = {
    "pair":     ("KD KC 2H 3S 4C 6D 7H 8S", "11"),       # 30 x 2   KD KC score
    "pair3":    ("KD KC 2H 4S 6C 8D TH QS", "111"),      # 30 x 2   3 played, 2H does not score
    "pair4":    ("KD KC 2H 4S 6C 8D TH QS", "1111"),     # 30 x 2   4 played
    "two_pair": ("AS AH 8C 8D 3S 5H 6C TD", "11111"),    # 58 x 2   3S kicker does not score
    "three":    ("QS QH QD 2C 4D 7S 8H 9C", "11111"),    # 60 x 3   2C 4D kickers
    "straight": ("5C 6D 7H 8S 9C 2H 3D JS", "11111"),    # 65 x 4
    "flush":    ("2H 5H 7H 9H JH 3C 4D 6S", "11111"),    # 68 x 4
    "full":     ("TS TH TD 4C 4D 2S 6H 8C", "11111"),    # 78 x 4
    "four":     ("7S 7H 7D 7C 2S 4H 5D 9C", "11111"),    # 88 x 7   2S kicker
    "sf":       ("5S 6S 7S 8S 9S 2H 3D JC", "11111"),    # 135 x 8
    "high":     ("AS 9D 6C 4H 2S 3C 5D 8H", "11111"),    # 16 x 1   only the Ace scores
}

# (jokers in slot order, hand, state overrides, want chips, want mult)
# The state defaults: money 0, discards 2, 44 cards left in the deck, 5 slots.
SPEC = [
    (("Joker",), "pair", {}, 30, 6),
    (("Joker",), "high", {}, 16, 5),
    (("Greedy Joker",), "pair", {}, 30, 5),
    (("Greedy Joker",), "three", {}, 60, 6),        # QD scores, the 4D kicker does not
    (("Greedy Joker",), "sf", {}, 135, 8),
    (("Bootstraps",), "pair", dict(money=23), 30, 10),
    (("Bootstraps",), "pair", dict(money=4), 30, 2),
    (("Green Joker",), "pair", {}, 30, 3),          # seed 0, +1 before it scores
    (("Green Joker",), "pair", dict(joker_value=(5.0,)), 30, 8),
    (("Jolly Joker",), "pair", {}, 30, 10),
    (("Jolly Joker",), "two_pair", {}, 58, 10),
    (("Jolly Joker",), "full", {}, 78, 12),
    (("Jolly Joker",), "four", {}, 88, 15),
    (("Jolly Joker",), "straight", {}, 65, 4),
    (("Zany Joker",), "three", {}, 60, 15),
    (("Zany Joker",), "full", {}, 78, 16),
    (("Zany Joker",), "four", {}, 88, 19),
    (("Zany Joker",), "pair", {}, 30, 2),
    (("Mad Joker",), "two_pair", {}, 58, 12),
    (("Mad Joker",), "full", {}, 78, 14),
    (("Mad Joker",), "four", {}, 88, 7),            # one rank: not a Two Pair
    (("Mad Joker",), "pair", {}, 30, 2),
    (("Crazy Joker",), "straight", {}, 65, 16),
    (("Crazy Joker",), "sf", {}, 135, 20),
    (("Crazy Joker",), "flush", {}, 68, 4),
    (("Droll Joker",), "flush", {}, 68, 14),
    (("Droll Joker",), "sf", {}, 135, 18),
    (("Droll Joker",), "straight", {}, 65, 4),
    (("Sly Joker",), "pair", {}, 80, 2),
    (("Sly Joker",), "full", {}, 128, 4),
    (("Sly Joker",), "high", {}, 16, 1),
    (("Wily Joker",), "three", {}, 160, 3),
    (("Wily Joker",), "full", {}, 178, 4),
    (("Wily Joker",), "two_pair", {}, 58, 2),
    (("Clever Joker",), "two_pair", {}, 138, 2),
    (("Clever Joker",), "full", {}, 158, 4),
    (("Clever Joker",), "three", {}, 60, 3),
    (("Devious Joker",), "straight", {}, 165, 4),
    (("Devious Joker",), "sf", {}, 235, 8),
    (("Devious Joker",), "high", {}, 16, 1),
    (("Crafty Joker",), "flush", {}, 148, 4),
    (("Crafty Joker",), "sf", {}, 215, 8),
    (("Crafty Joker",), "straight", {}, 65, 4),
    (("Half Joker",), "pair", {}, 30, 22),          # 2 played
    (("Half Joker",), "pair3", {}, 30, 22),         # 3 played (one does not score)
    (("Half Joker",), "pair4", {}, 30, 2),          # 4 played
    (("Half Joker",), "three", {}, 60, 3),
    (("Lusty Joker",), "flush", {}, 68, 19),
    (("Lusty Joker",), "high", {}, 16, 1),          # 4H is played, not scored
    (("Lusty Joker",), "sf", {}, 135, 8),
    (("Wrathful Joker",), "sf", {}, 135, 23),
    (("Wrathful Joker",), "high", {}, 16, 4),       # AS scores, 2S does not
    (("Wrathful Joker",), "flush", {}, 68, 4),
    (("Gluttonous Joker",), "four", {}, 88, 10),
    (("Gluttonous Joker",), "high", {}, 16, 1),     # 6C does not score
    (("Gluttonous Joker",), "flush", {}, 68, 4),
    (("Even Steven",), "full", {}, 78, 24),
    (("Even Steven",), "straight", {}, 65, 12),     # 6 and 8
    (("Even Steven",), "high", {}, 16, 1),
    (("Odd Todd",), "straight", {}, 158, 4),        # 5, 7, 9
    (("Odd Todd",), "high", {}, 47, 1),
    (("Odd Todd",), "pair", {}, 30, 2),
    (("Scary Face",), "pair", {}, 90, 2),
    (("Scary Face",), "three", {}, 150, 3),
    (("Scary Face",), "flush", {}, 98, 4),          # JH
    (("Scary Face",), "straight", {}, 65, 4),
    (("Smiley Face",), "pair", {}, 30, 12),
    (("Smiley Face",), "three", {}, 60, 18),
    (("Smiley Face",), "two_pair", {}, 58, 2),
    (("Fibonacci",), "two_pair", {}, 58, 34),       # A A 8 8
    (("Fibonacci",), "straight", {}, 65, 20),       # 5, 8
    (("Fibonacci",), "high", {}, 16, 9),
    (("Fibonacci",), "full", {}, 78, 4),
    (("Arrowhead",), "sf", {}, 385, 8),
    (("Arrowhead",), "two_pair", {}, 108, 2),       # AS scores, the 3S kicker does not
    (("Arrowhead",), "pair", {}, 30, 2),
    (("Onyx Agate",), "four", {}, 88, 14),
    (("Onyx Agate",), "two_pair", {}, 58, 9),       # 8C
    (("Onyx Agate",), "flush", {}, 68, 4),
    (("Banner",), "pair", dict(discards=2), 90, 2),
    (("Banner",), "pair", dict(discards=3), 120, 2),
    (("Banner",), "pair", dict(discards=0), 30, 2),
    (("Mystic Summit",), "pair", dict(discards=0), 30, 17),
    (("Mystic Summit",), "pair", dict(discards=1), 30, 2),
    (("Blue Joker",), "pair", {}, 118, 2),          # 52 - 8 = 44 left
    (("Blue Joker",), "pair", dict(deck_ptr=50), 34, 2),
    (("Blue Joker",), "pair", dict(deck_ptr=52), 30, 2),
    (("Abstract Joker",), "pair", {}, 30, 5),
    (("Joker", "Abstract Joker"), "pair", {}, 30, 12),
    (("Abstract Joker",) * 3, "pair", {}, 30, 29),  # each sees 3 jokers: 3 x 9
    (("Bull",), "pair", dict(money=12), 54, 2),
    (("Bull",), "pair", dict(money=0), 30, 2),
    (("Joker Stencil",), "pair", {}, 30, 10),       # 5 slots, itself counts as empty
    (("Joker Stencil",), "pair", dict(joker_slots=7), 30, 14),
    (("Joker", "Joker Stencil"), "pair", {}, 30, 24),
    (("Joker Stencil", "Joker"), "pair", {}, 30, 12),
    (("Joker Stencil", "Joker Stencil"), "pair", {}, 30, 50),
    (("Joker Stencil", "Joker", "Joker", "Joker", "Joker"), "pair", {}, 30, 18),
    (("Popcorn",), "pair", {}, 30, 22),
    (("Popcorn",), "pair", dict(joker_value=(8.0,)), 30, 10),
    (("Popcorn",), "pair", dict(joker_value=(0.0,)), 30, 2),
    (("Golden Joker",), "pair", {}, 30, 2),         # money only, no score
    (("Cloud 9",), "pair", {}, 30, 2),
    (("The Duo",), "pair", {}, 30, 4),
    (("The Duo",), "full", {}, 78, 8),
    (("The Duo",), "four", {}, 88, 14),
    (("The Duo",), "straight", {}, 65, 4),
    (("The Trio",), "three", {}, 60, 9),
    (("The Trio",), "full", {}, 78, 12),
    (("The Trio",), "four", {}, 88, 21),
    (("The Trio",), "two_pair", {}, 58, 2),
    (("The Family",), "four", {}, 88, 28),
    (("The Family",), "full", {}, 78, 4),
    (("The Order",), "straight", {}, 65, 12),
    (("The Order",), "sf", {}, 135, 24),
    (("The Order",), "flush", {}, 68, 4),
    (("The Tribe",), "flush", {}, 68, 8),
    (("The Tribe",), "sf", {}, 135, 16),
    (("The Tribe",), "straight", {}, 65, 4),
    # mixes: the order matters only among the main effects
    (("Joker", "The Duo"), "pair", {}, 30, 12),     # (2 + 4) x 2
    (("The Duo", "Joker"), "pair", {}, 30, 8),      # 2 x 2 + 4
    (("Greedy Joker", "The Duo"), "pair", {}, 30, 10),
    (("The Duo", "Greedy Joker"), "pair", {}, 30, 10),   # per-card adds come first in any slot
    (("Sly Joker", "The Duo", "Jolly Joker"), "pair", {}, 80, 12),
    (("Jolly Joker", "The Duo", "Zany Joker"), "full", {}, 78, 36),
    (("The Trio", "The Duo", "Joker"), "full", {}, 78, 28),
    (("Fibonacci", "Arrowhead", "Wrathful Joker", "Odd Todd"), "two_pair", {}, 58 + 50 + 62, 2 + 32 + 3),
]


def run_table_test():
    c = Checks("joker table")
    names = set(JOKER_NAMES)
    c.check("every TABLE joker has a REF entry", names <= set(REF), sorted(names - set(REF)))
    c.check("every REF entry is a TABLE joker", set(REF) <= names, sorted(set(REF) - names))
    in_spec = {n for jokers, *_ in SPEC for n in jokers}
    c.check("every TABLE joker has a spec case", names <= in_spec, sorted(names - in_spec))
    for d in TABLE:
        r = REF.get(d.name)
        if r is None:
            continue
        c.check(f"{d.name}: a counter in TABLE is a counter in REF",
                (r.seed is not None) == (len(d.updates) > 0))
        if r.seed is not None:
            c.eq(f"{d.name}: counter seed", float(d.value), r.seed)
    return c.done()


def run_spec_test():
    c = Checks("joker spec")
    for jokers, hand, kw, want_chips, want_mult in SPEC:
        lanes, sel = HANDS[hand]
        s = state(lanes, jokers=jokers, **kw)
        sel = select(sel)
        res = explain_play(s, sel)
        tag = f"{'+'.join(jokers)} on {hand} {kw or ''}"
        want = math.floor(want_chips * want_mult)
        c.eq(f"{tag}: env chips", float(res.chips), float(want_chips))
        c.eq(f"{tag}: env mult", float(res.mult), float(want_mult))
        c.eq(f"{tag}: env score", int(res.score), want)
        chips, mult, score, name, _ = ref_play(host(s), np.asarray(sel))
        c.check(f"{tag}: REF agrees", (chips, mult, score) == (want_chips, want_mult, want),
                (chips, mult, score), (want_chips, want_mult, want))
        c.check(f"{tag}: hand type", HAND_NAMES[int(res.handtype)] == name, HAND_NAMES[int(res.handtype)], name)
    return c.done()


# ===========================================================================
# BATCHED SCORING — one jit over many (loadout, hand, live state) cases
# ===========================================================================
_BASE = state("2C 3C 4C 5C 6C 7C 8C 9C")


@jax.jit
def _batch_explain(jokers, joker_value, hand, hand_valid, money, discards, deck_ptr, joker_slots, sel):
    def one(j, jv, h, hv, m, d, dp, js, s):
        st = _BASE._replace(jokers=j, joker_value=jv, hand=h, hand_valid=hv, money=m,
                            discards=d, deck_ptr=dp, joker_slots=js)
        r = explain_play(st, s)
        return r.chips, r.mult, r.score, r.handtype
    return jax.vmap(one)(jokers, joker_value, hand, hand_valid, money, discards, deck_ptr,
                         joker_slots, sel)


def _run_batch(cases):
    """cases: list of dicts (jokers, values, hand, valid, money, discards, deck_ptr, slots, sel)."""
    def col(key, width, fill, dt):
        out = np.full((len(cases), width), fill, dt)
        for i, k in enumerate(cases):
            out[i, :len(k[key])] = k[key]
        return out
    scal = lambda key: np.array([k[key] for k in cases], np.int32)
    out = _batch_explain(col("jokers", MAX_JOKERS, -1, np.int32),
                         col("values", MAX_JOKERS, 0.0, np.float32),
                         col("hand", MAX_HAND_SIZE, -1, np.int32),
                         col("valid", MAX_HAND_SIZE, False, bool),
                         scal("money"), scal("discards"), scal("deck_ptr"), scal("slots"),
                         col("sel", MAX_HAND_SIZE, False, bool))
    return [np.asarray(x) for x in jax.device_get(out)]


class _Host(NamedTuple):
    """The fields ref_play reads, for a batched case."""
    jokers: np.ndarray
    joker_value: np.ndarray
    hand: np.ndarray
    hand_valid: np.ndarray
    money: int
    discards: int
    deck: np.ndarray
    deck_size: int
    deck_ptr: int
    joker_slots: int


def _ref_case(k):
    jokers = np.full(MAX_JOKERS, -1, np.int32); jokers[:len(k["jokers"])] = k["jokers"]
    values = np.zeros(MAX_JOKERS, np.float32); values[:len(k["values"])] = k["values"]
    hand = np.full(MAX_HAND_SIZE, -1, np.int32); hand[:len(k["hand"])] = k["hand"]
    valid = np.zeros(MAX_HAND_SIZE, bool); valid[:len(k["valid"])] = k["valid"]
    sel = np.zeros(MAX_HAND_SIZE, bool); sel[:len(k["sel"])] = k["sel"]
    s = _Host(jokers, values, hand, valid, k["money"], k["discards"], np.asarray(_BASE.deck),
              52, k["deck_ptr"], k["slots"])
    return s, sel


def _compare(c, tag, cases, out):
    """Env batch output vs REF, case by case. -> per-name count of cases where the joker acted."""
    chips, mult, score, htype = out
    acted = Counter()
    for i, k in enumerate(cases):
        s, sel = _ref_case(k)
        rc, rm, rs, name, vals = ref_play(s, sel)
        jk = [int(j) for j in k["jokers"]]
        cards = [int(s.hand[l]) for l in played_lanes(s, sel)]
        for slot in ref_score(cards, jk, vals, ref_ctx(s))[3]:
            acted[NAME[jk[slot]]] += 1
        ok = (exact(float(chips[i]), rc) and exact(float(mult[i]), rm) and exact(int(score[i]), rs)
              and HAND_NAMES[int(htype[i])] == name)
        c.check(f"{tag} case {i}: {[NAME[j] for j in jk]} {[_cn(x) for x in cards]} "
                f"money {k['money']} disc {k['discards']} ptr {k['deck_ptr']} slots {k['slots']}",
                ok, (HAND_NAMES[int(htype[i])], float(chips[i]), float(mult[i]), int(score[i])),
                (name, rc, rm, rs))
    return acted


def _cn(cid):
    return "23456789TJQKA"[cid % 13] + "CDHS"[cid // 13]


def _case(jokers, hand, valid, sel, *, values=None, money=0, discards=2, deck_ptr=8, slots=5):
    if values is None:
        values = [REF[NAME[j]].seed if REF[NAME[j]].seed is not None else float(JOKERS.VALUE[j])
                  for j in jokers]
    return dict(jokers=list(jokers), values=list(values), hand=list(hand), valid=list(valid),
                sel=list(sel), money=money, discards=discards, deck_ptr=deck_ptr, slots=slots)


# ===========================================================================
# ORDER — every slot order of a mix, and MOVE_JOKER then PLAY
# ===========================================================================
def run_order_test():
    c = Checks("joker order")
    mix = [JOKER_ID[n] for n in ("Joker", "The Duo", "Jolly Joker", "Joker Stencil", "Greedy Joker")]
    for hname in ("pair", "full", "high"):
        lanes, sel = HANDS[hname]
        ids = card_ids(lanes)
        sel = [int(b) for b in sel] + [0] * (8 - len(sel))
        cases = [_case(p, ids, [True] * 8, sel) for p in itertools.permutations(mix)]
        out = _run_batch(cases)
        _compare(c, f"order {hname}", cases, out)
        if hname == "pair":
            c.check("pair: the slot order changes the score", len(set(out[2].tolist())) > 1,
                    sorted(set(out[2].tolist())))

    # MOVE_JOKER changes the next PLAY: [Duo, Joker] 2*2+4 = 8 -> [Joker, Duo] (2+4)*2 = 12
    lanes, sel = HANDS["pair"]
    s = state(lanes, jokers=("The Duo", "Joker"), blind_target=10 ** 6)
    a = drive(c, "no move", s, [act(V_PLAY, sel)])
    b = drive(c, "move", s, [act(V_MOVE_JOKER, target=0), act(V_PLAY, sel)])
    c.eq("no move: 30 x 8", a.chips_scored, 240)
    c.eq("after the move: 30 x 12", b.chips_scored, 360)
    return c.done()


# ===========================================================================
# FUZZ — random hands x random loadouts x random live state
# ===========================================================================
SHAPES = ["high", "pair", "two_pair", "three", "four", "full", "flush", "straight", "sf", "random"]


def _played(rng):
    """Random played cards for a random hand shape. Ids are distinct."""
    shape = SHAPES[rng.integers(len(SHAPES))]
    if shape == "random":
        return list(rng.choice(52, rng.integers(1, 6), replace=False))
    if shape in ("flush", "straight", "sf"):
        if shape == "flush":
            ranks = list(rng.choice(13, 5, replace=False))
        else:
            lo = rng.integers(-1, 9)                     # -1 = the A-2-3-4-5 wheel
            ranks = [ACE if r < 0 else r for r in range(lo, lo + 5)]
        suit = rng.integers(4)
        suits = [suit] * 5 if shape != "straight" else list(rng.integers(0, 4, 5))
        cards = [s * 13 + r for r, s in zip(ranks, suits)]
    else:
        groups = {"high": [1] * rng.integers(1, 6), "pair": [2] + [1] * rng.integers(0, 4),
                  "two_pair": [2, 2] + [1] * rng.integers(0, 2), "three": [3] + [1] * rng.integers(0, 3),
                  "four": [4] + [1] * rng.integers(0, 2), "full": [3, 2]}[shape]
        ranks = rng.choice(13, len(groups), replace=False)
        cards = [int(s) * 13 + int(r) for r, g in zip(ranks, groups)
                 for s in rng.choice(4, g, replace=False)]
    rng.shuffle(cards)
    return [int(x) for x in cards]


def _fuzz_cases(rng, n):
    cases = []
    for _ in range(n):
        played = _played(rng)
        rest = [int(x) for x in rng.permutation([x for x in range(52) if x not in played])]
        lanes = rng.permutation(8)[:len(played)]
        hand, valid, sel = [-1] * 8, [True] * 8, [False] * 8
        for card, l in zip(played, lanes):
            hand[l], sel[l] = card, True
        for l in range(8):
            if hand[l] < 0:
                hand[l] = rest.pop()
                if rng.random() < 0.1:                       # a hole (an empty deck leaves them)
                    hand[l], valid[l] = -1, False
                    sel[l] = rng.random() < 0.5              # selecting a hole does nothing
                elif len(played) == 5 and rng.random() < 0.1:
                    sel[l] = True                            # 6 selected: the first 5 valid play
        slots = 5 if rng.random() < 0.8 else int(rng.integers(1, 9))
        jokers = [int(j) for j in rng.integers(0, N_JOKERS, rng.integers(0, slots + 1))]
        values = []
        for j in jokers:
            r = REF[NAME[j]]
            if NAME[j] == "Popcorn":
                values.append(float(rng.choice([0, 4, 8, 12, 16, 20])))
            elif r.seed is not None:
                values.append(float(rng.integers(0, 13)))
            else:
                values.append(float(JOKERS.VALUE[j]))
        cases.append(_case(jokers, hand, valid, sel, values=values, money=int(rng.integers(0, 120)),
                           discards=int(rng.integers(0, 4)), deck_ptr=int(rng.integers(8, 53)),
                           slots=slots))
    return cases


def run_fuzz_test(n=6000, seed=0, min_acted=25):
    c = Checks("joker fuzz")
    rng = np.random.default_rng(seed)
    cases = _fuzz_cases(rng, n)
    acted = _compare(c, "fuzz", cases, _run_batch(cases))
    money_only = {n for n, r in REF.items() if r.card is None and r.main is None}
    for name in JOKER_NAMES:
        if name in money_only:
            c.check(f"{name} never changes a score", acted[name] == 0, acted[name], 0)
        else:
            c.check(f"{name} changed the score in at least {min_acted} cases",
                    acted[name] >= min_acted, acted[name], min_acted)
    low = sorted(((acted[n], n) for n in JOKER_NAMES if n not in money_only))[:3]
    print(f"  joker fuzz: {n} cases; fewest firings {low}")
    return c.done()


# ===========================================================================
# LIFE — scripted env_step sequences, every step checked against the model
# ===========================================================================
BIG = 10 ** 7          # a blind target the scripted plays never reach


def run_life_test():
    c = Checks("joker life")
    hand = "KD KC 2H 3S 4C 6D 7H 8S"

    # Green: +1 per play before it scores, -1 per discard, floored at 0.
    # A DISCARD with 0 discards left is a PLAY, so it counts +1.
    s = state(hand, jokers=("Green Joker",), hands=4, discards=3, blind_target=BIG)
    seq = [act(V_PLAY, "11"), act(V_DISCARD, "1"), act(V_DISCARD, "1"), act(V_DISCARD, "1"),
           act(V_DISCARD, "11"), act(V_PLAY, "1")]
    seen = []
    for i, a in enumerate(seq):
        s = drive(c, f"green {i}", s, [a])
        seen.append(float(s.joker_value[0]))
    c.eq("green values", seen, [1.0, 0.0, 0.0, 0.0, 1.0, 2.0])

    # Popcorn -4 and Golden / Cloud 9 money over 7 cleared blinds; Green +1 a play.
    s = reset(jax.random.key(3), 52, 8)
    for n in ("Popcorn", "Green Joker", "Golden Joker", "Cloud 9"):
        s = s._replace(jokers=s.jokers.at[int((s.jokers >= 0).sum())].set(JOKER_ID[n]))
    held = int((s.jokers >= 0).sum())
    s = s._replace(joker_value=s.joker_value.at[:held].set(
        jnp.array([20.0, 0.0, float(JOKERS.VALUE[JOKER_ID["Golden Joker"]]),
                   float(JOKERS.VALUE[JOKER_ID["Cloud 9"]])])),
                   joker_sell_value=s.joker_sell_value.at[:held].set(jnp.array([2, 2, 3, 3])))
    pops, greens = [], []
    for r in range(7):
        s = s._replace(blind_target=jnp.int32(1))
        pre = host(s)
        s = drive(c, f"round {r}", s, [act(V_PLAY, "1")])
        c.eq(f"round {r}: cashout has $8 of joker money",
             s.last_cashout, BLIND_PAY[int(pre.blind_index)] + int(pre.hands) - 1
             + min(int(pre.money) // 5, 5) + 8)
        pops.append(float(s.joker_value[0]))
        greens.append(float(s.joker_value[1]))
        s = drive(c, f"round {r} next", s, [act(V_NEXT)])
    c.eq("popcorn values", pops, [16.0, 12.0, 8.0, 4.0, 0.0, 0.0, 0.0])
    c.eq("green values over rounds", greens, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Sell Golden mid-blind: the clear pays Cloud 9 only.
    pair = "11"
    for jokers, extra in ((("Golden Joker", "Cloud 9"), 8), (("Cloud 9",), 4), ((), 0)):
        pre = state(hand, jokers=jokers, chips_scored=290, blind_target=300, money=0)
        got = drive(c, f"cashout {jokers}", pre, [act(V_PLAY, pair)])
        c.eq(f"cashout {jokers}", got.last_cashout, 3 + 2 + extra)
    pre = state(hand, jokers=("Golden Joker", "Cloud 9"), chips_scored=290, blind_target=300)
    got = drive(c, "sold golden", pre, [act(V_SELL, target=0), act(V_PLAY, pair)])
    c.eq("sold Golden: Cloud 9 pays alone", got.last_cashout, 3 + 2 + 4)

    # Live queries mid-blind: a SELL changes money, the joker count and the
    # empty slots before the next PLAY; a DISCARD changes Banner and Mystic Summit.
    s = state(hand, jokers=("Abstract Joker", "Joker", "Bull", "Bootstraps", "Joker Stencil"),
              money=9, hands=4, discards=3, blind_target=BIG)
    drive(c, "queries", s, [act(V_PLAY, "11"), act(V_SELL, target=1), act(V_PLAY, "11"),
                            act(V_MOVE_JOKER, target=2), act(V_DISCARD, "1"), act(V_PLAY, "111")])
    s = state(hand, jokers=("Banner", "Blue Joker", "Mystic Summit"), hands=4, discards=1,
              blind_target=BIG)
    drive(c, "discard queries", s, [act(V_PLAY, "11"), act(V_DISCARD, "11111"), act(V_PLAY, "11"),
                                    act(V_PLAY, "11111")])

    # Counters follow MOVE / SELL; a bought counter starts at its seed.
    s = state(hand, jokers=("Joker", "Green Joker", "Popcorn"), joker_value=(4.0, 3.0, 12.0),
              game_stage=STAGE_SHOP, money=20, shop_slots=("Green Joker", "Joker"))
    s = drive(c, "follow", s, [act(V_MOVE_JOKER, target=0), act(V_MOVE_JOKER, target=1),
                               act(V_SELL, target=0), act(V_BUY, target=0)])
    c.eq("follow: jokers", s.jokers[:3],
         jnp.array([JOKER_ID["Popcorn"], JOKER_ID["Joker"], JOKER_ID["Green Joker"]]))
    c.eq("follow: values", s.joker_value[:3], jnp.array([12.0, 4.0, 0.0]))
    s = drive(c, "follow next", s, [act(V_NEXT)])._replace(blind_target=jnp.int32(BIG))
    s = drive(c, "follow play", s, [act(V_PLAY, "1"), act(V_DISCARD, "1")])
    c.eq("follow: after a play and a discard", s.joker_value[:3], jnp.array([12.0, 4.0, 0.0]))
    return c.done()


# ===========================================================================
# SOAK — vmapped random runs; every step of every env through the model
# ===========================================================================
def _policy(p, rng):
    """A random action with mostly-legal targets. -> (verb, select, target)."""
    stage = int(p.game_stage)
    held = int((p.jokers >= 0).sum())
    sel = np.zeros(MAX_HAND_SIZE, bool)
    valid = np.flatnonzero(p.hand_valid)
    u = rng.random()
    if u < 0.02:                                             # any verb, any target
        return int(rng.integers(-1, 10)), rng.random(MAX_HAND_SIZE) < 0.3, int(rng.integers(-2, 34))
    if stage == STAGE_BLIND and len(valid):
        k = int(rng.integers(1, min(5, len(valid)) + 1))
        if rng.random() < 0.05:
            k = len(valid)                                   # more than 5: the first 5 play
        sel[rng.choice(valid, k, replace=False)] = True
        if u < 0.55:
            return V_PLAY, sel, 0
        if u < 0.75:
            return V_DISCARD, sel, 0
        if u < 0.82:
            return V_SELL, sel, int(rng.integers(-1, held + 1))
        if u < 0.91:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, max(held, 1)))
        return V_MOVE_CARD, sel, int(rng.integers(-1, int(p.hand_size)))
    if stage == STAGE_SHOP:
        if u < 0.4:
            return V_BUY, sel, int(rng.integers(-1, MAX_SHOP_SLOTS))
        if u < 0.5:
            return V_SELL, sel, int(rng.integers(-1, held + 1))
        if u < 0.6:
            return V_REROLL, sel, 0
        if u < 0.7:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, max(held, 1)))
        return V_NEXT, sel, 0
    return int(rng.integers(0, 8)), sel, 0                  # LOST / WON: all no-ops


_batch_reset = jax.jit(jax.vmap(lambda k: reset(k, 52, 8)))


def _soak_batch(rng, key, n_envs):
    """Fresh runs: 0-3 random starting jokers, some money, a mix of curriculum scales."""
    batch = _batch_reset(jax.random.split(key, n_envs))
    J = np.full((n_envs, MAX_JOKERS), -1, np.int32)
    V = np.zeros((n_envs, MAX_JOKERS), np.float32)
    S = np.zeros((n_envs, MAX_JOKERS), np.int32)
    for i in range(n_envs):
        for slot, j in enumerate(rng.integers(0, N_JOKERS, rng.integers(0, 4))):
            seed_v = REF[NAME[j]].seed
            J[i, slot], S[i, slot] = j, max(1, int(JOKER_COST[j]) // 2)
            V[i, slot] = float(JOKERS.VALUE[j]) if seed_v is None else seed_v
    scale = rng.choice([0.01, 0.03, 0.1, 0.3], n_envs).astype(np.float32)
    return batch._replace(
        jokers=jnp.asarray(J), joker_value=jnp.asarray(V), joker_sell_value=jnp.asarray(S),
        money=jnp.asarray(rng.integers(0, 30, n_envs), jnp.int32), target_scale=jnp.asarray(scale),
        blind_target=jnp.asarray(np.maximum(np.floor(100 * scale), 1), jnp.int32))


def _swap_in(done, fresh, batch):
    """Per env: the fresh run where done, else the batch run. Keys go through their raw words."""
    kd = jax.random.key_data
    pick = lambda f, b: jnp.where(done.reshape((-1,) + (1,) * (b.ndim - 1)), f, b)
    out = jax.tree.map(pick, fresh._replace(key=None), batch._replace(key=None))
    return out._replace(key=jax.random.wrap_key_data(pick(kd(fresh.key), kd(batch.key))))


def run_soak_test(n_envs=64, n_steps=400, seed=0):
    """A finished run (LOST / WON) is swapped for a fresh one, so every step is live."""
    c = Checks("joker soak")
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    key, sub = jax.random.split(key)
    batch = _soak_batch(rng, sub, n_envs)
    step = jax.jit(jax.vmap(env_step))

    stats = Counter()
    held_at_play = Counter()
    t0 = time.time()
    for t in range(n_steps):
        pre = host(batch)
        acts = [_policy(jax.tree.map(lambda x: x[i], pre), rng) for i in range(n_envs)]
        action = balatroAction(verb=jnp.asarray([a[0] for a in acts], jnp.int32),
                               select=jnp.asarray(np.stack([a[1] for a in acts])),
                               target=jnp.asarray([a[2] for a in acts], jnp.int32))
        batch, r, _ = step(batch, action)
        post, r, a_host = host(batch), np.asarray(r), jax.device_get(action)
        for i in range(n_envs):
            one = lambda tree: jax.tree.map(lambda x: x[i], tree)
            p, q = one(pre), one(post)
            verb = check_step(c, f"step {t} env {i}", p, one(a_host), q, int(r[i]))
            stats[{None: "noop", V_PLAY: "play", V_DISCARD: "discard", V_BUY: "buy", V_SELL: "sell",
                   V_REROLL: "reroll", V_NEXT: "next", V_MOVE_JOKER: "move_joker",
                   V_MOVE_CARD: "move_card"}[verb]] += 1
            if verb == V_PLAY:
                for j in p.jokers:
                    if j >= 0:
                        held_at_play[NAME[j]] += 1
                if int(q.game_stage) in (STAGE_SHOP, STAGE_WON):
                    stats["clears"] += 1
            if int(q.game_stage) == STAGE_LOST and int(p.game_stage) != STAGE_LOST:
                stats["lost"] += 1
            if int(q.game_stage) == STAGE_WON and int(p.game_stage) != STAGE_WON:
                stats["won"] += 1
        done = np.isin(np.asarray(post.game_stage), (STAGE_LOST, STAGE_WON))
        if done.any():
            stats["max ante"] = max(stats["max ante"], int(np.asarray(post.ante)[done].max()))
            key, sub = jax.random.split(key)
            batch = _swap_in(jnp.asarray(done), _soak_batch(rng, sub, n_envs), batch)
    stats["max ante"] = max(stats["max ante"], int(np.asarray(post.ante).max()))
    print(f"  joker soak: {n_envs} envs x {n_steps} steps in {time.time() - t0:.0f} s; {dict(stats)}")
    never = [n for n in JOKER_NAMES if held_at_play[n] == 0]
    print(f"  joker soak: jokers never held at a PLAY: {never or 'none'}")
    c.check("the soak cleared blinds", stats["clears"] > 0, stats["clears"])
    c.check("the soak bought jokers", stats["buy"] > 0, stats["buy"])
    return c.done()


ALL = [run_table_test, run_spec_test, run_order_test, run_fuzz_test, run_life_test, run_soak_test]


def run_all():
    return all([f() for f in ALL])


if __name__ == "__main__":
    run_all()
