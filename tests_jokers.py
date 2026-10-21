"""Joker test suite — proves every joker against an independent reference.

ref_balatro.py is a second implementation of the jokers, ported from the
game's Lua in plain Python. It does not read the jokers.py columns or
scoring.py: hand detection, the scoring order, the copies, the counters, the
discard and end-of-round effects and the cashout money are all re-derived
there. The env's 1-in-N coin flips and Misprint's draw are the one shared
input (PlayOut.rolls / hrolls / rnd). Each layer compares the env against it:

  table  every TABLE joker has a spec case. A new joker fails here until it has one.
  spec   hand-computed cases per joker (explain_play), fire and no-fire. They
         also check the reference itself, so the layers below can trust it.
  order  every slot order of a 5-joker add / XMULT / copy mix, and MOVE_JOKER -> PLAY.
  fuzz   random hands x random loadouts x random live state, vmapped.
  life   scripted env_step sequences with hand-computed values: counters,
         discards, the end of the round, budget jokers, the shop jokers.
  soak   vmapped random runs. Every step of every env goes through the
         reference transition model (score, counters, money, stage, budget, reward).

Run: python tests_jokers.py   (tests.py also runs it)
"""

import itertools
import math
import time
from collections import Counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import ref_balatro as R
from env import (balatroAction, env_step, explain_play, reset, invariant_errors, score_play,
                 apply_budget, roll_shop, acquire, MAX_HAND_SIZE, MAX_SHOP_SLOTS, V_PLAY, V_DISCARD, V_BUY,
                 V_SELL, V_REROLL, V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, STAGE_BLIND, STAGE_SHOP,
                 STAGE_LOST, STAGE_WON, BLIND_BOSS)
from jokers import TABLE, JOKER_NAMES, JOKER_ID, JOKERS, JOKER_COST, MAX_JOKERS, N_JOKERS
from scoring import HAND_NAMES
from fixtures import state, select, cards as card_ids
from tests_run import Checks, act

NAME = JOKER_NAMES          # id -> name; the reference is keyed by name, never by id
assert HAND_NAMES == R.HAND_ORDER


def host(s):
    """Device state -> numpy state (the PRNG key as raw words)."""
    return jax.device_get(s._replace(key=jax.random.key_data(s.key)))


_explain = jax.jit(score_play)
_explain_batch = jax.jit(jax.vmap(score_play))


def exact(got, want):
    """float32 is exact below 2**24; above it, allow its rounding."""
    return got == want if abs(want) < 2 ** 24 else abs(got - want) <= abs(want) * 2 ** -20


def near(got, want):
    return abs(got - want) <= 1e-4 * max(1.0, abs(want))


# ===========================================================================
# HOST STATE -> REFERENCE INPUTS
# ===========================================================================
def records(p):
    """The held jokers as reference records, slot order."""
    return [dict(name=NAME[int(j)], v=float(p.joker_value[i]), st=int(p.joker_state[i]),
                 sell=int(p.joker_sell_value[i])) for i, j in enumerate(p.jokers) if j >= 0]


def played_lanes(p, sel):
    """The env's repair: the first 5 selected lanes that hold a card, lane order."""
    return [l for l in range(MAX_HAND_SIZE) if sel[l] and p.hand_valid[l]][:5]


def counts(arr):
    return Counter({HAND_NAMES[k]: int(arr[k]) for k in range(9) if int(arr[k])})


def ref_g(p):
    size = int(p.deck_size)
    live = [int(c) for c in p.deck[:size] if 0 <= c < 52]
    return dict(money=int(p.money), hands_left=int(p.hands) - 1, discards=max(int(p.discards), 0),
                deck_left=max(size - int(p.deck_ptr), 0), slots=int(p.joker_slots),
                nines=sum(c % 13 == R.NINE for c in live), deck_size=size,
                played=counts(p.hand_played), round=counts(p.hand_round),
                targets=dict(idol=int(p.idol_card), mail=int(p.mail_rank),
                             ancient=int(p.ancient_suit), castle=int(p.castle_suit)),
                is_boss=int(p.blind_index) == BLIND_BOSS, discards_used=int(p.discards_used))


def ref_play_state(p, sel, out, mods=None):
    """Score a PLAY from a host state, with the env's draws. -> (result, records after)."""
    J = records(p)
    lanes = played_lanes(p, sel)
    cards = [int(p.hand[l]) for l in lanes]
    held = [(l, int(p.hand[l])) for l in range(MAX_HAND_SIZE) if p.hand_valid[l] and l not in lanes]
    res = R.ref_play(J, cards, held, ref_g(p),
                     rolls=lambda k, r, j: bool(out.rolls[k, min(r, out.rolls.shape[1] - 1), j]),
                     hrolls=lambda l, r, j: bool(out.hrolls[l, min(r, out.hrolls.shape[1] - 1), j]),
                     rnd=lambda j: float(out.rnd[j]), mods=mods)
    return res, J, cards


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


def _drop(J, gone):
    return [x for i, x in enumerate(J) if i not in gone]


def _deaths(c, tag, J, q):
    """Gros Michel / Cavendish die at random: read which ones did from q.
    -> (records left, a Gros Michel died). Anything else missing is a failure."""
    got = [NAME[int(j)] for j in q.jokers if j >= 0]
    left, gros, k = [], False, 0
    for x in J:
        if k < len(got) and got[k] == x["name"]:
            left.append(x)
            k += 1
        elif x["name"] in ("Gros Michel", "Cavendish"):
            gros |= x["name"] == "Gros Michel"
        else:
            c.check(f"{tag}: {x['name']} left the loadout", False, got, [y["name"] for y in J])
            left.append(x)
    return left, gros


def check_step(c, tag, p, a, q, reward, out=None):
    """Compare post-state q against the reference model of (p, a).
    p, q are host states; a is a host balatroAction; out the host PlayOut of
    (p, a.select) for a PLAY. -> the effective verb."""
    verb, t, d = int(a.verb), int(a.target), int(a.dest)
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

    J = records(p)
    n = len(J)
    names = [x["name"] for x in J]
    m = R.mods_of(names)
    want = dict(money=int(p.money), chips_scored=int(p.chips_scored), hands=int(p.hands),
                discards=int(p.discards), game_stage=stage, last_cashout=int(p.last_cashout),
                discards_used=int(p.discards_used), free_rerolls=int(p.free_rerolls),
                reroll_cost=int(p.reroll_cost), gros_extinct=bool(p.gros_extinct))
    played, in_round = np.array(p.hand_played), np.array(p.hand_round)
    want_reward = 0
    observe_state = set()                                   # To Do slots whose new hand is random

    if verb == V_PLAY:
        res, J, cards = ref_play_state(p, sel, out)
        got = int(q.chips_scored) - int(p.chips_scored)
        ok = exact(got, res["score"])
        c.check(f"{tag}: PLAY {res['hand']} scores {res['score']} ({res['chips']} x {res['mult']})",
                ok, got, res["score"])
        total = int(p.chips_scored) + (got if ok else res["score"])
        played = np.array([res["played"][h] for h in HAND_NAMES])
        in_round = np.array([res["round"][h] for h in HAND_NAMES])
        J = _drop(J, res["gone"])
        want.update(chips_scored=total, hands=int(p.hands) - 1, money=int(p.money) + int(res["money"]))
        if total >= int(p.blind_target):
            g = ref_g(p)
            g.update(discards=max(int(p.discards), 0), is_boss=int(p.blind_index) == BLIND_BOSS)
            J = _drop(J, R.ref_end_of_round(J, g))
            J, gros = _deaths(c, tag, J, q)
            want["gros_extinct"] = bool(p.gros_extinct) or gros
            observe_state = {i for i, x in enumerate(J) if x["name"] == "To Do List"}
            money = want["money"]
            rate = 1 + sum(x["name"] == "To the Moon" for x in J)
            interest = rate * min(money // 5, 5) if money >= 5 else 0
            cashout = R.BLIND_PAY[int(p.blind_index)] + int(p.hands) - 1 + interest \
                + int(R.ref_dollar_bonus(J, g))
            won = int(p.blind_index) == BLIND_BOSS and int(p.ante) >= R.WIN_ANTE
            want.update(money=money + cashout, last_cashout=cashout, reroll_cost=5,
                        game_stage=STAGE_WON if won else STAGE_SHOP)
            want_reward = cashout
            c.check(f"{tag}: the round targets are cards / suits",
                    0 <= int(q.idol_card) < 52 and 0 <= int(q.mail_rank) < 13
                    and 0 <= int(q.castle_suit) < 4 and 0 <= int(q.ancient_suit) < 4)
            c.check(f"{tag}: Ancient Joker gets a new suit", int(q.ancient_suit) != int(p.ancient_suit),
                    int(q.ancient_suit), int(p.ancient_suit))
        elif int(p.hands) - 1 <= 0:
            want["game_stage"] = STAGE_LOST
    elif verb == V_DISCARD:
        lanes = played_lanes(p, sel)
        money, gone = R.ref_discard(J, [int(p.hand[l]) for l in lanes], ref_g(p))
        J = _drop(J, gone)
        want.update(discards=int(p.discards) - 1, discards_used=int(p.discards_used) + 1,
                    money=int(p.money) + money)
    elif verb == V_BUY:
        jid = int(p.shop_slots[t]) if 0 <= t < MAX_SHOP_SLOTS else -1
        cost = int(JOKER_COST[jid]) if jid >= 0 else 0
        floor = -20 * names.count("Credit Card")
        if 0 <= t < int(p.shop_size) and jid >= 0 and int(p.money) - cost >= floor and n < int(p.joker_slots):
            nm = NAME[jid]
            st = int(sum(p.hand_played)) if nm == "Loyalty Card" else 0
            J.append(dict(name=nm, v=float(R.SEED.get(nm, JOKERS.VALUE[jid])), st=st,
                          sell=max(1, cost // 2)))
            if nm == "To Do List":
                observe_state = {len(J) - 1}
            want.update(money=int(p.money) - cost, free_rerolls=int(p.free_rerolls) + (nm == "Chaos the Clown"),
                        discards=int(p.discards) + R.D_SIZE.get(nm, 0))
            npeq(c, f"{tag}: BUY empties the shop slot", q.shop_slots[t], -1)
    elif verb == V_SELL:
        if 0 <= t < MAX_JOKERS and int(p.jokers[t]) >= 0:
            nm = NAME[int(p.jokers[t])]
            for i, x in enumerate(J):
                if i != t and x["name"] == "Campfire":
                    x["v"] += 0.25
            want.update(money=int(p.money) + int(p.joker_sell_value[t]),
                        discards=max(int(p.discards) - R.D_SIZE.get(nm, 0), 0),
                        free_rerolls=max(int(p.free_rerolls) - (nm == "Chaos the Clown"), 0))
            J = _drop(J, {t})
    elif verb == V_REROLL:
        free = int(p.free_rerolls) > 0
        cost = 0 if free else int(p.reroll_cost)
        if free or int(p.money) - cost >= -20 * names.count("Credit Card"):
            want.update(money=int(p.money) - cost, reroll_cost=int(p.reroll_cost) + (not free),
                        free_rerolls=int(p.free_rerolls) - free)
            for x in J:
                if x["name"] == "Flash Card":
                    x["v"] += 2
    elif verb == V_NEXT:
        burgle = sum(1 for i in range(n) if (k := R.copy_target(names, i)) is not None
                     and names[k] == "Burglar")
        want.update(chips_scored=0, hands=int(p.handcount) + 3 * burgle,
                    discards=0 if burgle else int(p.discardcount), discards_used=0,
                    free_rerolls=names.count("Chaos the Clown"), game_stage=STAGE_BLIND)
        in_round = np.zeros(9, np.int64)
    elif verb == V_MOVE_JOKER:                              # INSERT slot t at slot d
        if 0 <= t < n and 0 <= d < n and t != d:
            J.insert(d, J.pop(t))
    elif verb == V_MOVE_CARD:                               # INSERT lane t at lane d
        hand, valid = list(p.hand), list(p.hand_valid)
        size = int(p.hand_size)
        if 0 <= t < size and 0 <= d < size and t != d and valid[t]:
            for arr in (hand, valid):
                arr.insert(d, arr.pop(t))
        npeq(c, f"{tag}: MOVE_CARD hand", q.hand, np.array(hand))
        npeq(c, f"{tag}: MOVE_CARD valid", q.hand_valid, np.array(valid))

    # To Do List: a new hand is random; it must be a hand, and never the old one at a round end
    qs = [int(x) for x in q.joker_state]
    for i in observe_state:
        if i < len(J):
            c.check(f"{tag}: To Do List hand in range", 0 <= qs[i] < 9, qs[i])
            if verb == V_PLAY:
                c.check(f"{tag}: To Do List hand changes", qs[i] != J[i]["st"], qs[i], J[i]["st"])
            J[i]["st"] = qs[i]
    k = len(J)
    npeq(c, f"{tag}: jokers", q.jokers, np.array([JOKER_ID[x["name"]] for x in J] + [-1] * (MAX_JOKERS - k), np.int32))
    vals = np.array([x["v"] for x in J] + [0.0] * (MAX_JOKERS - k))
    c.check(f"{tag}: joker values", np.allclose(q.joker_value, vals, rtol=1e-5, atol=1e-4),
            q.joker_value[:8], vals[:8])
    npeq(c, f"{tag}: sell values", q.joker_sell_value, np.array([x["sell"] for x in J] + [0] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: joker states", q.joker_state, np.array([x["st"] for x in J] + [0] * (MAX_JOKERS - k)))
    hs, hc, dc = R.ref_budget(J, int(p.base_hand_size), int(p.base_hands), int(p.base_discards))
    want.update(hand_size=hs, handcount=hc, discardcount=dc)
    for f, w in want.items():
        npeq(c, f"{tag}: {f}", getattr(q, f), w)
    npeq(c, f"{tag}: hand_played", q.hand_played, played)
    npeq(c, f"{tag}: hand_round", q.hand_round, in_round)
    npeq(c, f"{tag}: reward", reward, want_reward)
    return verb


def drive(c, tag, s, actions):
    """Step a scripted action list; check every step. -> the final state."""
    for i, a in enumerate(actions):
        out = jax.device_get(_explain(s, a.select)[1])
        post, r, _ = env_step(s, a)
        check_step(c, f"{tag} #{i}", host(s), jax.device_get(a), host(post), int(r), out)
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
    "pair_blk": ("KD KC 2S 3S 4C 6C 7S 8S", "11"),       # 30 x 2   every held card is black
    "pair_q":   ("KD KC QS QH 4C 6D 7H 8S", "11"),       # 30 x 2   two Queens held
    "pair_k":   ("QD QC KS 3S 4C 6D 7H 8S", "11"),       # 30 x 2   a King held
    "flush4":   ("2H 5H 7H 9H 3C 4D 6S 8S", "11111"),    # 14 x 1   high card 9H (4 hearts)
    "gappy":    ("2C 4D 6H 8S TC 3H 5S 7D", "11111"),    # 15 x 1   high card TC
    "redflush": ("2H 5D 7H 9D JH 3C 4C 6S", "11111"),    # 15 x 1   high card JH
}

# (jokers in slot order, hand, state overrides, want chips, want mult)
# The state defaults: money 0, hands 3, discards 2, 44 cards left in the deck,
# 5 slots, Idol = Ace of Spades, Mail-In = Ace, Ancient = Spades, Castle = Spades.
PAIR_PLAYED = dict(hand_played=(0, 0, 0, 0, 0, 0, 0, 4, 0))
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

    # ======================= GROUP A =======================
    (("Scholar",), "two_pair", {}, 98, 10),         # AS AH: +20 chips +4 mult each
    (("Scholar",), "high", {}, 36, 5),
    (("Scholar",), "pair", {}, 30, 2),
    (("Walkie Talkie",), "full", {}, 128, 24),      # three 10s and two 4s
    (("Walkie Talkie",), "pair", {}, 30, 2),
    (("Ride the Bus",), "pair", dict(joker_value=(3.0,)), 30, 2),   # scored faces: back to 0
    (("Ride the Bus",), "straight", {}, 65, 5),     # no face: +1 first
    (("Ride the Bus",), "high", dict(joker_value=(3.0,)), 16, 5),
    (("Spare Trousers",), "two_pair", {}, 58, 4),
    (("Spare Trousers",), "full", {}, 78, 6),
    (("Spare Trousers",), "pair", {}, 30, 2),
    (("Square Joker",), "pair4", {}, 34, 2),
    (("Square Joker",), "pair", {}, 30, 2),
    (("Runner",), "straight", {}, 80, 4),
    (("Runner",), "sf", {}, 150, 8),
    (("Runner",), "pair", {}, 30, 2),
    (("Ice Cream",), "pair", {}, 130, 2),
    (("Ice Cream",), "pair", dict(joker_value=(5.0,)), 35, 2),
    (("Wee Joker",), "flush", {}, 76, 4),           # 2H scores: +8 before the main effect
    (("Wee Joker",), "flush", dict(joker_value=(16.0,)), 92, 4),
    (("Wee Joker",), "high", {}, 16, 1),            # 2S does not score
    (("Castle",), "pair", dict(joker_value=(9.0,)), 39, 2),
    (("Castle",), "pair", {}, 30, 2),
    (("Flash Card",), "pair", dict(joker_value=(6.0,)), 30, 8),
    (("Swashbuckler",), "pair", {}, 30, 2),
    (("Swashbuckler", "Joker"), "pair", {}, 30, 7),  # Joker sells for $1, then +4
    (("Supernova",), "pair", {}, 30, 3),
    (("Supernova",), "pair", PAIR_PLAYED, 30, 7),
    (("Card Sharp",), "pair", dict(hand_round=(0, 0, 0, 0, 0, 0, 0, 1, 0)), 30, 6),
    (("Card Sharp",), "pair", {}, 30, 2),
    (("Obelisk",), "pair", dict(joker_value=(2.0,)), 30, 2),          # pair is now most played: reset
    (("Obelisk",), "pair", dict(joker_value=(2.0,), hand_played=(0, 0, 0, 0, 0, 0, 3, 0, 0)), 30, 4.4),
    (("Hit the Road",), "pair", dict(joker_value=(2.5,)), 30, 5),
    (("Hit the Road",), "pair", {}, 30, 2),
    (("Ramen",), "pair", {}, 30, 4),
    (("Campfire",), "pair", dict(joker_value=(1.5,)), 30, 3),
    (("Campfire",), "pair", {}, 30, 2),
    (("Acrobat",), "pair", dict(hands=1), 30, 6),
    (("Acrobat",), "pair", {}, 30, 2),
    (("Loyalty Card",), "pair", dict(hand_played=(0, 0, 0, 0, 0, 0, 0, 5, 0)), 30, 8),   # the 6th hand
    (("Loyalty Card",), "pair", {}, 30, 2),
    (("Seeing Double",), "two_pair", {}, 58, 4),    # 8C and a non-Club
    (("Seeing Double",), "flush", {}, 68, 4),
    (("Flower Pot",), "two_pair", {}, 58, 6),       # S H C D
    (("Flower Pot",), "pair", {}, 30, 2),
    (("Stuntman",), "pair", {}, 280, 2),
    (("Erosion",), "pair", {}, 30, 2),
    (("Gros Michel",), "pair", {}, 30, 17),
    (("Cavendish",), "pair", {}, 30, 6),
    (("Baseball Card",), "pair", {}, 30, 2),
    (("Baseball Card", "Bull"), "pair", dict(money=10), 50, 3),   # Bull is Uncommon
    (("Baseball Card", "Joker"), "pair", {}, 30, 6),               # Joker is Common
    (("Bull", "Baseball Card", "Joker"), "pair", dict(money=10), 50, 7),  # 2 x 1.5 + 4
    (("Blackboard",), "pair_blk", {}, 30, 6),
    (("Blackboard",), "pair", {}, 30, 2),
    (("To Do List",), "pair", dict(joker_state=(7,)), 30, 2),
    (("To Do List", "Bull"), "pair", dict(joker_state=(7, 0)), 38, 2),   # its $4 comes first
    (("To Do List", "Bull"), "pair", dict(joker_state=(6, 0)), 30, 2),
    (("Mail-In Rebate",), "pair", {}, 30, 2),
    (("Faceless Joker",), "pair", {}, 30, 2),
    (("Delayed Gratification",), "pair", {}, 30, 2),
    (("Rocket",), "pair", {}, 30, 2),
    (("Egg",), "pair", {}, 30, 2),
    (("Gift Card",), "pair", {}, 30, 2),
    (("Credit Card",), "pair", {}, 30, 2),
    (("To the Moon",), "pair", {}, 30, 2),
    (("Chaos the Clown",), "pair", {}, 30, 2),
    (("Juggler",), "pair", {}, 30, 2),
    (("Drunkard",), "pair", {}, 30, 2),
    (("Merry Andy",), "pair", {}, 30, 2),
    (("Troubadour",), "pair", {}, 30, 2),
    (("Turtle Bean",), "pair", {}, 30, 2),
    (("Burglar",), "pair", {}, 30, 2),

    # ======================= GROUP B =======================
    (("Hanging Chad",), "pair", {}, 50, 2),         # KD scores 3 times
    (("Hanging Chad", "Scary Face"), "pair", {}, 170, 2),
    (("Dusk",), "pair", dict(hands=1), 50, 2),
    (("Dusk",), "pair", {}, 30, 2),
    (("Hack",), "straight", {}, 70, 4),             # the 5 again
    (("Hack",), "two_pair", {}, 58, 2),
    (("Sock and Buskin",), "pair", {}, 50, 2),
    (("Sock and Buskin",), "three", {}, 90, 3),
    (("Seltzer",), "pair", {}, 50, 2),
    (("Seltzer", "Smiley Face"), "three", {}, 90, 33),
    (("Mime",), "pair", {}, 30, 2),
    (("Mime", "Baron"), "pair_k", {}, 30, 4.5),     # the King again
    (("Photograph",), "pair", {}, 30, 4),
    (("Photograph",), "straight", {}, 65, 4),
    (("Photograph", "Hanging Chad"), "pair", {}, 50, 16),    # KD x3: X2 each time
    (("The Idol",), "two_pair", {}, 58, 4),         # AS
    (("The Idol",), "pair", {}, 30, 2),
    (("Ancient Joker",), "two_pair", {}, 58, 3),    # AS is a Spade
    (("Ancient Joker",), "pair", {}, 30, 2),
    (("Oops! All 6s", "Bloodstone"), "flush", {}, 68, 30.375),   # 2 in 2: every Heart
    (("Bloodstone",), "pair", {}, 30, 2),
    (("Rough Gem",), "flush", {}, 68, 4),
    (("Oops! All 6s", "Business Card"), "pair", {}, 30, 2),
    (("Triboulet",), "pair", {}, 30, 8),
    (("Baron",), "pair_k", {}, 30, 3),
    (("Baron",), "pair", {}, 30, 2),
    (("Shoot the Moon",), "pair_q", {}, 30, 28),
    (("Shoot the Moon",), "pair", {}, 30, 2),
    (("Raised Fist",), "pair", {}, 30, 6),          # lowest held: 2H
    (("Raised Fist",), "pair_q", {}, 30, 10),       # lowest held: 4C
    (("Oops! All 6s", "Reserved Parking"), "pair_q", {}, 30, 2),

    # ======================= GROUP C =======================
    (("Four Fingers",), "flush4", {}, 58, 4),
    (("Four Fingers",), "pair", {}, 30, 2),
    (("Shortcut",), "gappy", {}, 60, 4),
    (("Shortcut",), "pair", {}, 30, 2),
    (("Smeared Joker",), "redflush", {}, 68, 4),
    (("Smeared Joker", "Greedy Joker"), "flush", {}, 68, 19),   # Hearts count as Diamonds
    (("Splash",), "pair3", {}, 32, 2),
    (("Pareidolia", "Smiley Face"), "straight", {}, 65, 29),
    (("Pareidolia",), "straight", {}, 65, 4),
    (("Showman",), "pair", {}, 30, 2),
    (("Oops! All 6s",), "pair", {}, 30, 2),
    (("Blueprint", "Joker"), "pair", {}, 30, 10),
    (("Joker", "Blueprint"), "pair", {}, 30, 6),
    (("Blueprint", "Blueprint", "The Duo"), "pair", {}, 30, 16),
    (("Joker", "Brainstorm"), "pair", {}, 30, 10),
    (("Brainstorm", "Joker"), "pair", {}, 30, 6),
    (("Blueprint", "Brainstorm", "Joker"), "pair", {}, 30, 6),      # a cycle: nothing
]
# no spec numbers: Misprint's draw is random (checked by run_misprint_test)
NO_SPEC = {"Misprint"}


def run_table_test():
    c = Checks("joker table")
    names = set(JOKER_NAMES)
    in_spec = {n for jokers, *_ in SPEC for n in jokers} | NO_SPEC
    c.check("every TABLE joker has a spec case", names <= in_spec, sorted(names - in_spec))
    for d in TABLE:
        if d.name in R.SEED:
            c.eq(f"{d.name}: counter seed", float(d.value), float(R.SEED[d.name]))
        c.eq(f"{d.name}: hand size", d.h_size, R.H_SIZE.get(d.name, 0))
        c.eq(f"{d.name}: discards", d.d_size, R.D_SIZE.get(d.name, 0))
        c.check(f"{d.name}: Uncommon in both tables", (d.rarity == 2) == (d.name in R.UNCOMMON))
    return c.done()


def run_spec_test():
    c = Checks("joker spec")
    for jokers, hand, kw, want_chips, want_mult in SPEC:
        lanes, sel = HANDS[hand]
        s = state(lanes, jokers=jokers, **kw)
        sel = select(sel)
        res, out = jax.device_get(_explain(s, sel))
        tag = f"{'+'.join(jokers)} on {hand} {kw or ''}"
        want = math.floor(want_chips * want_mult)
        c.check(f"{tag}: env chips", near(float(res.chips), want_chips), float(res.chips), want_chips)
        c.check(f"{tag}: env mult", near(float(res.mult), want_mult), float(res.mult), want_mult)
        c.check(f"{tag}: env score", abs(int(res.score) - want) <= 1, int(res.score), want)
        r, _, _ = ref_play_state(host(s), np.asarray(sel), out)
        c.check(f"{tag}: REF agrees", near(r["chips"], want_chips) and near(r["mult"], want_mult),
                (r["chips"], r["mult"]), (want_chips, want_mult))
        c.check(f"{tag}: hand type", HAND_NAMES[int(res.handtype)] == r["hand"], HAND_NAMES[int(res.handtype)], r["hand"])
    return c.done()


def run_misprint_test(n=400):
    """Misprint adds a whole number 0..23, every value can come up, and it is the
    draw the reference reads."""
    c = Checks("misprint")
    lanes, sel = HANDS["pair"]
    seen = set()
    for k in range(n):
        s = state(lanes, jokers=("Misprint",), key=k)
        res, out = jax.device_get(_explain(s, select(sel)))
        add = float(res.mult) - 2
        seen.add(add)
        c.check(f"key {k}: a whole number 0..23", add == int(add) and 0 <= add <= 23, add)
        c.check(f"key {k}: the draw", add == float(out.rnd[0]), add, float(out.rnd[0]))
    c.check("every value 0..23 came up", seen == set(range(24)), sorted(seen))
    return c.done()


def run_odds_test(n=4000):
    """1 in 2 draws hit about half the time; Oops! All 6s makes them certain."""
    c = Checks("odds")
    lanes, sel = HANDS["flush"]
    hits = 0
    for k in range(n // 100):
        s = state(lanes, jokers=("Bloodstone",), key=k)
        out = jax.device_get(_explain(s, select(sel))[1])
        hits += int(out.rolls[:, 0, 0].sum())
    rate = hits / (n // 100 * 5)
    c.check("Bloodstone hits 1 in 2", 0.4 < rate < 0.6, rate)
    s = state(lanes, jokers=("Bloodstone", "Oops! All 6s"))
    out = jax.device_get(_explain(s, select(sel))[1])
    c.check("Oops! All 6s: 2 in 2", bool(out.rolls[:, :, 0].all()))
    return c.done()


# ===========================================================================
# BATCHED SCORING — one jit over many (loadout, hand, live state) cases
# ===========================================================================
_BASE = host(state("2C 3C 4C 5C 6C 7C 8C 9C"))


def _batch_states(cases):
    """cases: dicts of field overrides -> one stacked device state."""
    n = len(cases)
    out = {}
    for f in _BASE._fields:
        base = np.asarray(getattr(_BASE, f))
        arr = np.repeat(base[None], n, axis=0).copy()
        for i, k in enumerate(cases):
            if f in k:
                v = np.asarray(k[f], arr.dtype)
                if arr.ndim == 1:
                    arr[i] = v
                else:
                    arr[i] = np.asarray(arr[i])
                    arr[i, :len(v)] = v
                    if f in ("jokers",):
                        arr[i, len(v):] = -1
                    elif f in ("joker_value", "joker_state", "joker_sell_value", "hand_valid"):
                        arr[i, len(v):] = 0
                    elif f == "hand":
                        arr[i, len(v):] = -1
        out[f] = arr
    s = _BASE._replace(**{f: jnp.asarray(v) for f, v in out.items() if f != "key"})
    return s._replace(key=jax.random.wrap_key_data(jnp.asarray(out["key"], jnp.uint32)))


def _host_case(k):
    """The host view of one case (for the reference)."""
    fields = {}
    for f in _BASE._fields:
        base = np.asarray(getattr(_BASE, f)).copy()
        if f in k:
            v = np.asarray(k[f], base.dtype)
            if base.ndim == 0:
                base = v
            else:
                base[:len(v)] = v
                base[len(v):] = -1 if f in ("jokers", "hand") else 0
        fields[f] = base
    return SimpleNamespace(**fields)


def _compare(c, tag, cases):
    """Env batch vs the reference, case by case. -> per-name count of cases where the joker acted."""
    res, out = jax.device_get(_explain_batch(_batch_states(cases), jnp.asarray(np.stack([k["sel"] for k in cases]))))
    acted = Counter()
    for i, k in enumerate(cases):
        p = _host_case(k)
        o = jax.tree.map(lambda x: x[i], out)
        r, _, cards = ref_play_state(p, np.asarray(k["sel"]), o)
        names = [NAME[int(j)] for j in k["jokers"]]
        for slot in r["acted"]:
            acted[names[slot]] += 1
        m = R.mods_of(names)
        for mod, key, off in (("Four Fingers", "ff", False), ("Shortcut", "sc", False),
                              ("Smeared Joker", "sm", False), ("Splash", "sp", False),
                              ("Pareidolia", "pa", False), ("Oops! All 6s", "odds", 1)):
            if mod in names and m[key] != off:
                alt = ref_play_state(p, np.asarray(k["sel"]), o, mods={**m, key: off})[0]
                if (alt["score"], alt["money"]) != (r["score"], r["money"]) or key == "odds" and alt["acted"] != r["acted"]:
                    acted[mod] += 1
        ok = (near(float(res.chips[i]), r["chips"]) and near(float(res.mult[i]), r["mult"])
              and abs(int(res.score[i]) - r["score"]) <= max(1, abs(r["score"]) * 2 ** -18)
              and HAND_NAMES[int(res.handtype[i])] == r["hand"]
              and near(float(res.money[i]), r["money"]))
        c.check(f"{tag} case {i}: {names} {[_cn(x) for x in cards]} money {k['money']}",
                ok, (HAND_NAMES[int(res.handtype[i])], float(res.chips[i]), float(res.mult[i]),
                     int(res.score[i]), float(res.money[i])),
                (r["hand"], r["chips"], r["mult"], r["score"], r["money"]))
    return acted


def _cn(cid):
    return "23456789TJQKA"[cid % 13] + "CDHS"[cid // 13]


def _seed(j):
    return float(R.SEED.get(NAME[j], JOKERS.VALUE[j]))


def _case(jokers, hand, valid, sel, **kw):
    values = kw.pop("values", None)
    if values is None:
        values = [_seed(j) for j in jokers]
    sell = [max(1, int(JOKER_COST[j]) // 2) for j in jokers]
    d = dict(jokers=list(jokers), joker_value=list(values), joker_sell_value=sell,
             hand=list(hand), hand_valid=list(valid), sel=np.array(list(sel) + [False] * (MAX_HAND_SIZE - len(sel))),
             money=0)
    d.update(kw)
    return d


# ===========================================================================
# ORDER — every slot order of a mix, and MOVE_JOKER then PLAY
# ===========================================================================
def run_order_test():
    c = Checks("joker order")
    for mix in (("Joker", "The Duo", "Jolly Joker", "Joker Stencil", "Greedy Joker"),
                ("Blueprint", "Photograph", "Joker", "Baseball Card", "Bull"),
                ("Brainstorm", "Hanging Chad", "Scary Face", "The Duo", "Mime")):
        ids = [JOKER_ID[n] for n in mix]
        for hname in ("pair", "full", "high", "pair_k"):
            lanes, sel = HANDS[hname]
            hand = card_ids(lanes)
            sel = [bool(int(b)) for b in sel] + [False] * (8 - len(sel))
            cases = [_case(p, hand, [True] * 8, sel, money=10) for p in itertools.permutations(ids)]
            _compare(c, f"order {mix[0]} {hname}", cases)

    # MOVE_JOKER changes the next PLAY: [Duo, Joker] 2*2+4 = 8 -> [Joker, Duo] (2+4)*2 = 12
    lanes, sel = HANDS["pair"]
    s = state(lanes, jokers=("The Duo", "Joker"), blind_target=10 ** 6)
    a = drive(c, "no move", s, [act(V_PLAY, sel)])
    b = drive(c, "move", s, [act(V_MOVE_JOKER, target=0, dest=1), act(V_PLAY, sel)])
    c.eq("no move: 30 x 8", a.chips_scored, 240)
    c.eq("after the move: 30 x 12", b.chips_scored, 360)
    return c.done()


# ===========================================================================
# FUZZ — random hands x random loadouts x random live state
# ===========================================================================
SHAPES = ["high", "pair", "two_pair", "three", "four", "full", "flush", "straight", "sf",
          "flush4", "straight4", "gappy", "random"]
RULES = [JOKER_ID[n] for n in ("Four Fingers", "Shortcut", "Smeared Joker", "Splash", "Pareidolia",
                               "Oops! All 6s", "Blueprint", "Brainstorm")]
# loadouts that random draws rarely make: a joker and the joker it needs
THEMES = [("Mime", "Baron"), ("Mime", "Shoot the Moon"), ("Mime", "Raised Fist"),
          ("Mime", "Reserved Parking", "Oops! All 6s"), ("The Idol",), ("Pareidolia", "Smiley Face"),
          ("Pareidolia", "Scary Face", "Hanging Chad"), ("Pareidolia", "Photograph"),
          ("Smeared Joker", "Greedy Joker"), ("Smeared Joker", "Droll Joker"),
          ("Four Fingers", "Droll Joker", "Crazy Joker"), ("Shortcut", "Crazy Joker"),
          ("Splash", "Hack"), ("Blueprint", "The Duo"), ("Brainstorm",), ("The Family",),
          ("The Tribe",), ("Crafty Joker",), ("Droll Joker",), ("Blackboard",), ("Baron",),
          ("Oops! All 6s", "Bloodstone"), ("Oops! All 6s", "Business Card", "Bull"),
          ("Mad Joker",), ("To Do List", "Bull"), ("Hit the Road",), ("Card Sharp",)]


def _played(rng):
    """Random played cards for a random hand shape. Ids are distinct."""
    shape = SHAPES[rng.integers(len(SHAPES))]
    if shape == "random":
        return list(rng.choice(52, rng.integers(1, 6), replace=False))
    if shape in ("flush", "straight", "sf", "flush4", "straight4", "gappy"):
        if shape in ("flush", "flush4"):
            ranks = list(rng.choice(13, 5, replace=False))
        elif shape == "gappy":
            lo = int(rng.integers(0, 4))
            ranks = sorted(rng.choice(range(lo, lo + 9), 5, replace=False))
        else:
            lo = rng.integers(-1, 9)                     # -1 = the A-2-3-4-5 wheel
            ranks = [ACE if r < 0 else r for r in range(lo, lo + 5)]
        s0 = rng.integers(4)
        suits = [s0] * 5 if shape in ("flush", "sf", "flush4") else list(rng.integers(0, 4, 5))
        if shape in ("flush4", "straight4"):
            suits[4] = (s0 + 1 + rng.integers(3)) % 4
            if shape == "straight4":
                ranks[4] = int(rng.integers(13))
        cards = list(dict.fromkeys(int(s) * 13 + int(r) for r, s in zip(ranks, suits)))
    else:
        groups = {"high": [1] * rng.integers(1, 6), "pair": [2] + [1] * rng.integers(0, 4),
                  "two_pair": [2, 2] + [1] * rng.integers(0, 2), "three": [3] + [1] * rng.integers(0, 3),
                  "four": [4] + [1] * rng.integers(0, 2), "full": [3, 2]}[shape]
        ranks = rng.choice(13, len(groups), replace=False)
        cards = [int(s) * 13 + int(r) for r, g in zip(ranks, groups)
                 for s in rng.choice(4, g, replace=False)]
    rng.shuffle(cards)
    return [int(x) for x in cards]


ACE = R.ACE


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
        if jokers and rng.random() < 0.4:                    # rule and copy jokers, often
            jokers[int(rng.integers(len(jokers)))] = int(rng.choice(RULES))
        if rng.random() < 0.3:
            theme = [JOKER_ID[nm] for nm in THEMES[rng.integers(len(THEMES))]]
            jokers = (theme + jokers)[:max(slots, len(theme))]
            slots = max(slots, len(jokers))
        values, states = [], []
        for j in jokers:
            nm = NAME[j]
            if nm in ("Obelisk", "Hit the Road", "Campfire"):
                values.append(1.0 + float(rng.integers(0, 10)) * 0.25)
            elif nm == "Ramen":
                values.append(1.0 + float(rng.integers(1, 101)) * 0.01)
            elif nm in R.SEED:
                values.append(float(rng.integers(0, 13)) + (nm in ("Ice Cream", "Seltzer", "Turtle Bean", "Popcorn")))
            else:
                values.append(float(JOKERS.VALUE[j]))
            states.append(int(rng.integers(0, 9)) if nm == "To Do List" else int(rng.integers(0, 12)) if nm == "Loyalty Card" else 0)
        hp = [int(x) for x in rng.integers(0, 4, 9)]
        hr = [min(int(x), h) for x, h in zip(rng.integers(0, 2, 9), hp)]
        cases.append(_case(jokers, hand, valid, sel, values=values, joker_state=states,
                           joker_sell_value=[int(rng.integers(1, 9)) for _ in jokers],
                           money=int(rng.integers(-20, 120)), discards=int(rng.integers(0, 4)),
                           hands=int(rng.integers(1, 5)), deck_ptr=int(rng.integers(8, 53)),
                           joker_slots=slots, hand_played=hp, hand_round=hr,
                           idol_card=int(rng.choice(played)) if rng.random() < 0.5 else int(rng.integers(52)),
                           mail_rank=int(rng.integers(13)),
                           ancient_suit=int(rng.integers(4)), castle_suit=int(rng.integers(4)),
                           key=[int(x) for x in rng.integers(0, 2 ** 31, 2)]))
    return cases


MONEY_ONLY = {"Golden Joker", "Cloud 9", "Mail-In Rebate", "Faceless Joker", "Delayed Gratification",
              "Rocket", "Egg", "Gift Card", "Credit Card", "To the Moon", "Chaos the Clown", "Juggler",
              "Drunkard", "Merry Andy", "Troubadour", "Turtle Bean", "Burglar", "Showman",
              "Oops! All 6s", "Blueprint", "Brainstorm", "Erosion", "Ice Cream", "Seltzer"}


def run_fuzz_test(n=10000, seed=0, min_acted=10):
    c = Checks("joker fuzz")
    rng = np.random.default_rng(seed)
    acted = Counter()
    for chunk in range(0, n, 2000):
        cases = _fuzz_cases(rng, min(2000, n - chunk))
        acted += _compare(c, "fuzz", cases)
    for name in JOKER_NAMES:
        if name in MONEY_ONLY:
            continue
        c.check(f"{name} acted in at least {min_acted} cases", acted[name] >= min_acted, acted[name], min_acted)
    low = sorted(((acted[n], n) for n in JOKER_NAMES if n not in MONEY_ONLY))[:4]
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
    s = state(hand, jokers=("Green Joker",), hands=4, discards=3, blind_target=BIG)
    seq = [act(V_PLAY, "11"), act(V_DISCARD, "1"), act(V_DISCARD, "1"), act(V_DISCARD, "1"),
           act(V_DISCARD, "11"), act(V_PLAY, "1")]
    seen = []
    for i, a in enumerate(seq):
        s = drive(c, f"green {i}", s, [a])
        seen.append(float(s.joker_value[0]))
    c.eq("green values", seen, [1.0, 0.0, 0.0, 0.0, 1.0, 2.0])

    # Popcorn -4 a round and eaten at 0; Golden / Cloud 9 money over 6 cleared blinds.
    s = reset(jax.random.key(3), 52, 8)
    for nm in ("Popcorn", "Green Joker", "Golden Joker", "Cloud 9"):
        s, _ = jax.jit(acquire)(s, JOKER_ID[nm])
    pops = []
    for r in range(6):
        s = s._replace(blind_target=jnp.int32(1))
        pre = host(s)
        s = drive(c, f"round {r}", s, [act(V_PLAY, "1")])
        c.eq(f"round {r}: cashout has $8 of joker money",
             s.last_cashout, R.BLIND_PAY[int(pre.blind_index)] + int(pre.hands) - 1
             + min(int(pre.money) // 5, 5) + 8)
        pops.append(float(s.joker_value[0]) if int(s.jokers[0]) == JOKER_ID["Popcorn"] else None)
        s = drive(c, f"round {r} next", s, [act(V_NEXT)])
    want_pops = [16.0, 12.0, 8.0, 4.0, None, None]
    c.check("popcorn values, then eaten", pops == want_pops, pops, want_pops)

    # Discards: Castle (Spades), Mail-In (Aces), Hit the Road (Jacks), Ramen, Faceless, Green
    s = state("AS JS JD KH QC 3S 4C 6D", jokers=("Castle", "Mail-In Rebate", "Hit the Road", "Ramen",
                                                 "Faceless Joker"),
              hands=4, discards=3, blind_target=BIG, mail_rank=12, castle_suit=3)
    s = drive(c, "discard", s, [act(V_DISCARD, "11111")])
    c.eq("discard: Castle +3 x 2 Spades", s.joker_value[0], 6.0)
    c.eq("discard: Hit the Road +0.5 x 2 Jacks", s.joker_value[2], 2.0)
    c.check("discard: Ramen -0.01 x 5", near(float(s.joker_value[3]), 1.95), float(s.joker_value[3]))
    c.eq("discard: $5 (an Ace) + $5 (4 faces)", s.money, 10)

    # Delayed Gratification pays $2 per discard left when none was used; Rocket; To the Moon
    for used, want_extra in ((0, 2 * 2), (1, 0)):
        s = state(hand, jokers=("Delayed Gratification",), chips_scored=290, blind_target=300,
                  discards=2, discards_used=used, money=0)
        s = drive(c, f"delayed {used}", s, [act(V_PLAY, "11")])
        c.eq(f"delayed {used}: cashout", s.last_cashout, 3 + 2 + want_extra)
    s = state(hand, jokers=("Rocket", "To the Moon"), chips_scored=290, blind_target=300, money=25,
              blind_index=BLIND_BOSS, ante=2)
    s = drive(c, "rocket boss", s, [act(V_PLAY, "11")])
    c.eq("rocket boss: 5 + 2 hands + $10 interest + $3 Rocket", s.last_cashout, 5 + 2 + 10 + 3)
    c.eq("rocket boss: +2 payout", s.joker_value[0], 3.0)

    # Ice Cream -5 a hand, eaten at 0; Seltzer 10 hands, then gone
    s = state(hand, jokers=("Ice Cream", "Seltzer"), joker_value=(10.0, 2.0), hands=4, blind_target=BIG)
    s = drive(c, "decay", s, [act(V_PLAY, "11"), act(V_PLAY, "11")])
    c.eq("decay: both eaten", s.jokers[:2], jnp.array([-1, -1]))

    # Budget jokers: bought in the shop, the next blind has them
    s = state(hand, game_stage=STAGE_SHOP, money=100,
              shop_slots=("Juggler", "Merry Andy", "Troubadour", "Drunkard"), shop_size=4)
    s = drive(c, "budget", s, [act(V_BUY, target=k) for k in range(4)] + [act(V_NEXT)])
    c.eq("budget: hand size 8 + 1 - 1 + 2", s.hand_size, 10)
    c.eq("budget: hands 4 - 1", s.hands, 3)
    c.eq("budget: discards 3 + 3 + 1", s.discards, 7)
    c.eq("budget: 10 cards dealt", s.hand_valid.sum(), 10)
    s = drive(c, "budget sell", s, [act(V_SELL, target=2), act(V_PLAY, "1")])
    c.eq("budget: Troubadour sold, hand size 8", s.hand_size, 8)
    c.check("budget: the extra cards stay", int(s.hand_valid.sum()) >= 8, int(s.hand_valid.sum()))

    # Burglar: +3 hands, no discards
    s = state(hand, jokers=("Burglar", "Blueprint"), game_stage=STAGE_SHOP)
    s = drive(c, "burglar", s, [act(V_MOVE_JOKER, target=1, dest=0), act(V_NEXT)])
    c.eq("burglar + its copy: 4 + 6 hands", s.hands, 10)
    c.eq("burglar: 0 discards", s.discards, 0)

    # Chaos the Clown: one free reroll per shop; Flash Card +2 a reroll; Credit Card debt
    s = state(hand, jokers=("Chaos the Clown", "Flash Card", "Credit Card"), game_stage=STAGE_SHOP,
              money=3, free_rerolls=1, shop_slots=("Bull", "Joker"))
    s = drive(c, "shop jokers", s, [act(V_REROLL), act(V_REROLL), act(V_REROLL), act(V_BUY, target=0)])
    c.eq("shop jokers: Flash Card +2 x 3", s.joker_value[1], 6.0)
    c.check("shop jokers: into debt", int(s.money) < 0, int(s.money))

    # Loyalty Card: X4 on the 6th hand after the buy, then every 6th
    mults = [float(explain_play(state(hand, jokers=("Loyalty Card",), joker_state=(1,),
                                      hand_played=(0, 0, 0, 0, 0, 0, 0, k, 0)), select("11")).mult)
             for k in range(1, 14)]
    c.eq("loyalty: X4 on the 6th and 12th hand", mults, [8.0 if k in (6, 12) else 2.0 for k in range(1, 14)])

    # Gros Michel extinct -> Cavendish in the shop, Gros Michel never
    keys = jax.random.split(jax.random.key(0), 3000)
    held = jnp.full(MAX_JOKERS, -1, jnp.int32)
    ext = jax.vmap(lambda k: roll_shop(k, held, 4, False, True))(keys)
    alive = jax.vmap(lambda k: roll_shop(k, held, 4, False, False))(keys)
    c.check("extinct: Cavendish appears", bool((ext == JOKER_ID["Cavendish"]).any()))
    c.check("extinct: Gros Michel never", not bool((ext == JOKER_ID["Gros Michel"]).any()))
    c.check("alive: Cavendish never", not bool((alive == JOKER_ID["Cavendish"]).any()))
    # Showman: held jokers and repeats in the shop
    held = held.at[0].set(JOKER_ID["Joker"])
    show = jax.vmap(lambda k: roll_shop(k, held, 4, True, False))(keys)
    c.check("showman: a held joker appears", bool((show == JOKER_ID["Joker"]).any()))
    rep = np.asarray(show)
    c.check("showman: a joker twice in one shop", any(len(set(r)) < 4 for r in rep))
    return c.done()


# ===========================================================================
# SOAK — vmapped random runs; every step of every env through the model
# ===========================================================================
def _policy(p, rng):
    """A random action with mostly-legal targets. -> (verb, select, target, dest)."""
    stage = int(p.game_stage)
    held = int((p.jokers >= 0).sum())
    sel = np.zeros(MAX_HAND_SIZE, bool)
    valid = np.flatnonzero(p.hand_valid)
    u = rng.random()
    if u < 0.02:                                             # any verb, any target
        return (int(rng.integers(-1, 10)), rng.random(MAX_HAND_SIZE) < 0.3, int(rng.integers(-2, 34)),
                int(rng.integers(-2, 34)))
    if stage == STAGE_BLIND and len(valid):
        k = int(rng.integers(1, min(5, len(valid)) + 1))
        if rng.random() < 0.05:
            k = len(valid)                                   # more than 5: the first 5 play
        sel[rng.choice(valid, k, replace=False)] = True
        if u < 0.55:
            return V_PLAY, sel, 0, 0
        if u < 0.75:
            return V_DISCARD, sel, 0, 0
        if u < 0.82:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.91:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        size = int(p.hand_size)
        return V_MOVE_CARD, sel, int(rng.integers(-1, size + 1)), int(rng.integers(-1, size + 1))
    if stage == STAGE_SHOP:
        if u < 0.4:
            return V_BUY, sel, int(rng.integers(-1, MAX_SHOP_SLOTS)), 0
        if u < 0.5:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.6:
            return V_REROLL, sel, 0, 0
        if u < 0.7:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        return V_NEXT, sel, 0, 0
    return int(rng.integers(0, 8)), sel, 0, 0              # LOST / WON: all no-ops


_batch_reset = jax.jit(jax.vmap(lambda k: reset(k, 52, 8)))
_batch_budget = jax.jit(jax.vmap(apply_budget))


def _soak_batch(rng, key, n_envs):
    """Fresh runs: 0-4 random starting jokers, some money, a mix of curriculum scales."""
    batch = _batch_reset(jax.random.split(key, n_envs))
    J = np.full((n_envs, MAX_JOKERS), -1, np.int32)
    V = np.zeros((n_envs, MAX_JOKERS), np.float32)
    S = np.zeros((n_envs, MAX_JOKERS), np.int32)
    T = np.zeros((n_envs, MAX_JOKERS), np.int32)
    for i in range(n_envs):
        for slot, j in enumerate(rng.integers(0, N_JOKERS, rng.integers(0, 5))):
            J[i, slot], S[i, slot] = j, max(1, int(JOKER_COST[j]) // 2)
            V[i, slot] = _seed(j)
            T[i, slot] = int(rng.integers(9)) if NAME[j] == "To Do List" else 0
    scale = rng.choice([0.01, 0.03, 0.1, 0.3], n_envs).astype(np.float32)
    batch = batch._replace(
        jokers=jnp.asarray(J), joker_value=jnp.asarray(V), joker_sell_value=jnp.asarray(S),
        joker_state=jnp.asarray(T),
        money=jnp.asarray(rng.integers(0, 30, n_envs), jnp.int32), target_scale=jnp.asarray(scale),
        blind_target=jnp.asarray(np.maximum(np.floor(100 * scale), 1), jnp.int32))
    batch = _batch_budget(batch)
    # the first blind starts with the budget the jokers give
    return batch._replace(hands=batch.handcount, discards=batch.discardcount)


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
                               target=jnp.asarray([a[2] for a in acts], jnp.int32),
                               dest=jnp.asarray([a[3] for a in acts], jnp.int32))
        outs = jax.device_get(_explain_batch(batch, action.select)[1])
        batch, r, _ = step(batch, action)
        post, r, a_host = host(batch), np.asarray(r), jax.device_get(action)
        for i in range(n_envs):
            one = lambda tree: jax.tree.map(lambda x: x[i], tree)
            p, q = one(pre), one(post)
            verb = check_step(c, f"step {t} env {i}", p, one(a_host), q, int(r[i]), one(outs))
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


ALL = [run_table_test, run_spec_test, run_misprint_test, run_odds_test, run_order_test,
       run_fuzz_test, run_life_test, run_soak_test]


def run_all():
    return all([f() for f in ALL])


if __name__ == "__main__":
    run_all()
