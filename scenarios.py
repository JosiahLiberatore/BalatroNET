"""Scenario tests — one or more for every joker, Tarot, Planet, Spectral card, pack,
enhancement, edition, seal, Boss Blind, tag and voucher.

A scenario is a start state (fixtures.state), a few env_step actions, and the values
the real game gives, written out by hand next to the rule in words. tests.py runs every
scenario, and the reference transition model (tests_jokers.check_step) on every step.
gui.py shows them one step at a time (T), so a person who knows the game can read each
rule and each number and see the cards.

The scoring SPEC rows of tests_jokers (one PLAY each) join the list as the "Scoring"
group, so the GUI shows those too.

    python scenarios.py         # run every scenario, print the failures
"""

import ast
import io
import math
import tokenize
from types import SimpleNamespace
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from env import (balatroAction, env_step, explain_play, roll_shop, roll_pack_cards, _clear_blind, MAX_HAND_SIZE,
                 STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT, STAGE_NAMES, VERB_NAMES, V_PLAY, V_DISCARD,
                 V_BUY, V_SELL, V_REROLL, V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, V_USE, V_SELL_CONS,
                 V_OPEN, V_PICK, V_SKIP, V_REDEEM, _hooked, _hook_discard, score_play, fix_select)
from items import (CONS_ID, CONS_NAMES, CONS_KIND, CK_TAROT, CK_PLANET, PACK_NAMES, PACK_KIND,
                   PACK_CHOOSE, PK_ARCANA, PK_CELESTIAL, PK_STANDARD, PK_BUFFOON, MAX_PACK, ENH_NAMES,
                   ED_NAMES, SEAL_NAMES, HAND_NAMES, N_HAND_TYPES, IK_JOKER, IK_PLANET, IK_TAROT, IK_CARD,
                   CK_SPECTRAL, PK_SPECTRAL, BOSS_NAMES, TAG_NAMES, VOUCHER_NAMES, BOSS_ID, TAG_ID, VOUCHER_ID,
                   PACK_SIZE, hand_chips_mult)
import ref_balatro as R
from jokers import JOKER_NAMES, JOKER_ID, JOKER_RARITY
from fixtures import state, card_id, lanes as pad

H8 = "KD KC 2H 3S 4C 6D 7H 8S"      # a Pair of Kings to play; 2H 3S 4C 6D 7H 8S held
BIG = 10 ** 7                        # a blind target no scripted hand reaches
PACK = {n: i for i, n in enumerate(PACK_NAMES)}
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]


# ---------------------------------------------------------------------------
# THE PIECES — a step, a check, a scenario
# ---------------------------------------------------------------------------
class Step(NamedTuple):
    verb: int
    cards: object = ""      # "KD KC": card names (instances 0-51 of the standard deck), or a lane tuple
    target: int = 0         # the shop / joker / consumable / pack slot
    dest: int = 0


class Check(NamedTuple):
    label: str
    got: Callable           # fn(run) -> the value the env gave
    want: object            # the value the game gives (or Between / OneOf, or fn(run) -> it)
    tol: float = 1e-4       # relative, for numbers


class Scenario(NamedTuple):
    group: str              # Jokers, Tarots, Planets, Spectrals, Packs, Cards, Bosses, Tags, Vouchers, Rules, Scoring
    subject: str            # what the test is about; "A + B" covers both
    title: str
    expect: str             # the rule in words, and how the numbers come out
    build: object           # fixtures.state keywords, or fn() -> state
    steps: tuple = ()
    checks: tuple = ()


class Between(NamedTuple):
    lo: float
    hi: float

    def ok(self, x):
        return self.lo <= x <= self.hi

    def __str__(self):
        return f"{self.lo:g}..{self.hi:g}"


class OneOf(NamedTuple):
    vals: tuple

    def ok(self, x):
        return x in self.vals

    def __str__(self):
        return " or ".join(fmt(v) for v in self.vals)


def play(cards):
    return Step(V_PLAY, cards)


def discard(cards):
    return Step(V_DISCARD, cards)


def use(slot=0, cards=""):
    return Step(V_USE, cards, slot)


def sell(slot):
    return Step(V_SELL, "", slot)


def sell_cons(slot=0):
    return Step(V_SELL_CONS, "", slot)


def buy(slot=0):
    return Step(V_BUY, "", slot)


def reroll():
    return Step(V_REROLL)


def next_blind():
    """NEXT in the shop: to the blind select."""
    return Step(V_NEXT)


def play_blind():
    """NEXT at the blind select: play the blind on deck."""
    return Step(V_NEXT)


def skip_blind():
    """SKIP at the blind select: skip the Small / Big Blind, take its tag."""
    return Step(V_SKIP)


def reroll_boss():
    """REROLL at the blind select: a new boss (Director's Cut, Retcon)."""
    return Step(V_REROLL)


def redeem(slot=0):
    return Step(V_REDEEM, "", slot)


def open_pack(slot=0):
    return Step(V_OPEN, "", slot)


def pick(i=0, cards=""):
    return Step(V_PICK, cards, i)


def skip():
    return Step(V_SKIP)


C = Check


def S(group, subject, title, expect, build, steps=(), checks=()):
    return Scenario(group, subject, title, " ".join(expect.split()), build, tuple(steps), tuple(checks))


def blind(hand=H8, clear=False, **kw):
    """A blind in progress (the Small Blind of ante 1, 3 hands and 2 discards left).
    clear: the next PLAY beats it (299 of 300 chips scored)."""
    base = dict(hand=hand, blind_target=300 if clear else BIG, chips_scored=299 if clear else 0)
    base.update(kw)
    return base


def _counts(**kw):
    """Per hand type values (HAND_NAMES order) from names: _counts(pair=2)."""
    return [kw.get(h, 0) for h in HAND_NAMES]


def shop(**kw):
    """The shop: no hand (the deck is back together), $10."""
    base = dict(hand="", hand_size=8, deck_ptr=0, game_stage=STAGE_SHOP, money=10)
    base.update(kw)
    return base


def pack_state(pid, items, hand=H8, left=None, eds=(), cards=(), **kw):
    """An open pack. items: Tarot / Planet / Spectral / joker names; cards: (name, enh, ed, seal)
    for a Standard pack. An Arcana or Spectral pack has a hand dealt to use its cards on."""
    kind = int(PACK_KIND[pid])
    dealt = kind in (PK_ARCANA, PK_SPECTRAL)
    s = state(hand if dealt else "", hand_size=8, game_stage=STAGE_PACK, deck_ptr=None if dealt else 0, **kw)
    if kind == PK_STANDARD:
        ids = [0] * len(cards)
        rank = [card_id(c[0]) % 13 for c in cards]
        suit = [card_id(c[0]) // 13 for c in cards]
        enh, ed, seal = ([c[k] for c in cards] for k in (1, 2, 3))
    else:
        ids = [JOKER_ID[n] if kind == PK_BUFFOON else CONS_ID[n] for n in items]
        rank = suit = enh = seal = []
        ed = list(eds)
    i32 = lambda v, fill=0: pad(list(v), MAX_PACK, fill)
    return s._replace(pack_id=jnp.int32(pid),
                      pack_left=jnp.int32(int(PACK_CHOOSE[pid]) if left is None else left),
                      pack_items=i32(ids, -1), pack_ed=i32(ed), pack_rank=i32(rank),
                      pack_suit=i32(suit), pack_enh=i32(enh), pack_seal=i32(seal))


# ---------------------------------------------------------------------------
# READING A STATE — r[k] in a check is the View of state k (0 = the start, -1 = the end)
# ---------------------------------------------------------------------------
def host(s):
    return jax.device_get(s._replace(key=jax.random.key_data(s.key)))


def _int(field):
    return property(lambda self: int(getattr(self.h, field)))


class View:
    """One host state, read by name."""
    money, hands, discards, hand_size, deck_size, joker_slots, last_cashout, reroll_cost, \
        free_rerolls, ante, blind_index, blind_target, tarots_used, pack_left, skips, consumable_slots, \
        base_hand_size, shop_size, round, unused_discards, chips_scored, ecto_minus, base_hands,         base_discards = map(_int, (
            "money", "hands", "discards", "hand_size", "deck_size", "joker_slots", "last_cashout",
            "reroll_cost", "free_rerolls", "ante", "blind_index", "blind_target", "tarots_used",
            "pack_left", "skips", "consumable_slots", "base_hand_size", "shop_size", "round",
            "unused_discards", "chips_scored", "ecto_minus", "base_hands", "base_discards"))

    def __init__(self, h):
        self.h = h

    @property
    def stage(self):
        return STAGE_NAMES[int(self.h.game_stage)]

    @property
    def jokers(self):
        return [JOKER_NAMES[int(j)] for j in self.h.jokers if j >= 0]

    def jv(self, i):
        return round(float(self.h.joker_value[i]), 5)

    def jsell(self, i):
        return int(self.h.joker_sell_value[i])

    def jed(self, i):
        return ED_NAMES[int(self.h.joker_ed[i])] or "none"

    def jrarity(self, i):
        j = int(self.h.jokers[i])
        return RARITY[int(JOKER_RARITY[j])] if j >= 0 else "none"

    @property
    def cons(self):
        return [CONS_NAMES[int(c)] for c in self.h.consumables if c >= 0]

    def csell(self, i):
        return int(self.h.cons_sell[i])

    def n_cons(self, kind):
        return sum(int(CONS_KIND[int(c)]) == kind for c in self.h.consumables if c >= 0)

    def card(self, c):
        """Card instance c (a name like 'KD' = its standard instance, or an index)."""
        i = card_id(c) if isinstance(c, str) else int(c)
        h = self.h
        rank, suit = int(h.card_rank[i]), int(h.card_suit[i])
        return SimpleNamespace(rank=rank, suit=suit, enh=ENH_NAMES[int(h.card_enh[i])] or "none",
                               ed=ED_NAMES[int(h.card_ed[i])] or "none",
                               seal=SEAL_NAMES[int(h.card_seal[i])] or "none",
                               bonus=float(h.card_bonus[i]), live=bool(h.card_live[i]),
                               name="23456789TJQKA"[rank] + "CDHS"[suit])

    def level(self, hand):
        return int(self.h.hand_level[HAND_NAMES.index(hand)])

    def n_cards(self, rank=None, suit=None, enh=None, seal=None):
        """Live cards in the full deck with these fields (names or ints)."""
        h = self.h
        m = np.asarray(h.card_live, bool)
        for arr, v, names in ((h.card_rank, rank, None), (h.card_suit, suit, None),
                              (h.card_enh, enh, ENH_NAMES), (h.card_seal, seal, SEAL_NAMES)):
            if v is not None:
                m = m & (np.asarray(arr) == (names.index(v) if isinstance(v, str) else v))
        return int(m.sum())

    @property
    def in_hand(self):
        return int(np.asarray(self.h.hand_valid).sum())

    @property
    def hand(self):
        return [self.card(int(self.h.hand[l])).name for l in range(MAX_HAND_SIZE) if self.h.hand_valid[l]]

    @property
    def packs(self):
        return [PACK_NAMES[int(p)] if p >= 0 else "-" for p in self.h.pack_slots]

    @property
    def shop(self):
        return [_item_name(self.h, i) for i in range(len(self.h.shop_slots)) if self.h.shop_slots[i] >= 0]

    def shop_price(self, i):
        """$ shop slot i costs now (vouchers, coupons)."""
        from env import shop_item_cost
        return int(shop_item_cost(self.h, i))

    def shop_ed(self, i):
        return ED_NAMES[int(self.h.shop_ed[i])] or "none"

    def shop_rarity(self, i):
        """The rarity of the joker in shop slot i."""
        if int(self.h.shop_kind[i]) != IK_JOKER or int(self.h.shop_slots[i]) < 0:
            return "not a joker"
        return RARITY[int(JOKER_RARITY[int(self.h.shop_slots[i])])]

    def pack_price(self, i):
        from env import pack_cost
        return int(pack_cost(self.h, int(self.h.pack_slots[i]), bool(self.h.pack_free[i])))

    @property
    def voucher_price(self):
        from env import voucher_cost
        return int(voucher_cost(self.h))

    @property
    def pack_name(self):
        p = int(self.h.pack_id)
        return PACK_NAMES[p] if p >= 0 else "none"

    @property
    def played_ante(self):
        """Cards marked as played this ante (The Pillar)."""
        return int((np.asarray(self.h.card_played_ante) & np.asarray(self.h.card_live)).sum())

    @property
    def boss(self):
        b = int(self.h.boss)
        return BOSS_NAMES[b] if b >= 0 else "none"

    @property
    def boss_fits(self):
        """The boss may come up in this ante (min ante; the showdown bosses at ante 8)."""
        return self.boss != "none" and R.boss_eligible(self.boss, int(self.h.ante))

    @property
    def boss_disabled(self):
        return bool(self.h.boss_disabled)

    @property
    def tags(self):
        return [TAG_NAMES[int(t)] for t in self.h.tags if t >= 0]

    @property
    def blind_tags(self):
        return [TAG_NAMES[int(t)] if t >= 0 else "none" for t in self.h.blind_tags]

    @property
    def vouchers(self):
        return [VOUCHER_NAMES[i] for i in range(len(VOUCHER_NAMES)) if self.h.vouchers[i]]

    @property
    def shop_vouchers(self):
        return [VOUCHER_NAMES[int(v)] for v in self.h.shop_vouchers if v >= 0]

    @property
    def down(self):
        """The cards in hand drawn face down."""
        return [self.card(int(self.h.hand[l])).name for l in range(MAX_HAND_SIZE)
                if self.h.hand_valid[l] and self.h.hand_down[l]]

    @property
    def forced(self):
        f = int(self.h.forced_card)
        return self.card(f).name if f >= 0 else "none"

    def jdeb(self, i):
        return bool(self.h.joker_debuff[i])

    def jdown(self, i):
        return bool(self.h.joker_down[i])

    def ced(self, i):
        return ED_NAMES[int(self.h.cons_ed[i])] or "none"

    def debuffed(self, c):
        """Is card c (a name) debuffed by the boss now?"""
        from env import card_debuffed
        return bool(card_debuffed(self.h, card_id(c)))


# ---------------------------------------------------------------------------
# RUNNING ONE
# ---------------------------------------------------------------------------
def fmt(x):
    if isinstance(x, (Between, OneOf)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4g}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in x) + "]" if x else "none"
    return str(x)


def _same(got, want, tol):
    if isinstance(want, (Between, OneOf)):
        return bool(want.ok(got))
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want)
    if isinstance(want, (int, float)) and isinstance(got, (int, float, np.number)):
        return abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
    return got == want


def _lanes(h, cards):
    if isinstance(cards, str):
        out = []
        for n in cards.split():
            hit = [l for l in range(MAX_HAND_SIZE) if h.hand_valid[l] and int(h.hand[l]) == card_id(n)]
            if not hit:
                raise ValueError(f"{n} is not in the hand")
            out.append(hit[0])
        return out
    return list(cards)


def _item_name(h, i):
    it = int(h.shop_slots[i])
    if it < 0:
        return "an empty slot"
    kind = int(h.shop_kind[i])
    if kind == IK_CARD:
        name = "23456789TJQKA"[int(h.shop_rank[i])] + "CDHS"[int(h.shop_suit[i])]
        mods = [m for m in (ENH_NAMES[int(h.shop_enh[i])], ED_NAMES[int(h.shop_ed[i])]) if m]
        return name + (f" ({', '.join(mods)})" if mods else "")
    return JOKER_NAMES[it] if kind == IK_JOKER else CONS_NAMES[it]


def _pack_item(h, i):
    it, pid = int(h.pack_items[i]), int(h.pack_id)
    if it < 0 or pid < 0:
        return "nothing"
    kind = int(PACK_KIND[pid])
    if kind == PK_BUFFOON:
        return JOKER_NAMES[it]
    if kind == PK_STANDARD:
        name = "23456789TJQKA"[int(h.pack_rank[i])] + "CDHS"[int(h.pack_suit[i])]
        mods = [m for m in (ENH_NAMES[int(h.pack_enh[i])], ED_NAMES[int(h.pack_ed[i])],
                            SEAL_NAMES[int(h.pack_seal[i])] and SEAL_NAMES[int(h.pack_seal[i])] + " Seal") if m]
        return name + (f" ({', '.join(mods)})" if mods else "")
    return CONS_NAMES[it]


def describe(step, h):
    """The step in words, read against the state it acts on."""
    v, t = step.verb, step.target
    try:
        cards = " ".join(View(h).card(int(h.hand[l])).name for l in _lanes(h, step.cards))
    except ValueError:
        cards = str(step.cards)
    on = f" on {cards}" if cards else ""
    cons = [CONS_NAMES[int(c)] for c in h.consumables if c >= 0]
    jok = [JOKER_NAMES[int(j)] for j in h.jokers if j >= 0]
    if v in (V_PLAY, V_DISCARD):
        return f"{VERB_NAMES[v]} {cards}"
    if v == V_USE:
        return f"USE {cons[t] if t < len(cons) else 'slot ' + str(t + 1)}{on}"
    if v == V_SELL_CONS:
        return f"SELL {cons[t] if t < len(cons) else 'slot ' + str(t + 1)}"
    if v == V_SELL:
        return f"SELL {jok[t] if t < len(jok) else 'slot ' + str(t + 1)}"
    if v == V_BUY:
        return f"BUY {_item_name(h, t)}"
    if v == V_OPEN:
        p = int(h.pack_slots[t])
        return f"OPEN {PACK_NAMES[p] if p >= 0 else 'an empty pack slot'}"
    if v == V_PICK:
        return f"TAKE {_pack_item(h, t)}{on}"
    blinds = ["Small Blind", "Big Blind", "Boss Blind"]
    on_deck = blinds[min(max(int(h.blind_index), 0), 2)]
    if v == V_SKIP:
        return "SKIP the pack" if int(h.game_stage) == STAGE_PACK else f"SKIP the {on_deck}"
    if v == V_NEXT:
        return "NEXT: to the blind select" if int(h.game_stage) == STAGE_SHOP else f"NEXT: play the {on_deck}"
    if v == V_REROLL:
        return "REROLL the shop" if int(h.game_stage) == STAGE_SHOP else "REROLL the boss"
    if v == V_REDEEM:
        vv = int(h.shop_vouchers[t]) if 0 <= t < len(h.shop_vouchers) else -1
        return f"REDEEM {VOUCHER_NAMES[vv] if vv >= 0 else 'an empty voucher slot'}"
    if v == V_MOVE_JOKER:
        return f"MOVE joker {t + 1} to slot {step.dest + 1}"
    if v == V_MOVE_CARD:
        return f"MOVE the card in lane {t + 1} to lane {step.dest + 1}"
    return VERB_NAMES[v]


class Run:
    """A scenario, played. raw[k] / r[k]: the state after k steps; plays[k]: the PlayResult
    of step k (a PLAY), else None; results: (label, got, want, ok) per check."""

    def __init__(self, sc):
        self.sc = sc
        self.raw, self.plays, self.labels, self.actions, self.rewards = [], [], [], [], []
        self.error = None
        try:
            s = sc.build() if callable(sc.build) else state(**sc.build)
            self.raw.append(s)
            h = host(s)
            for st in sc.steps:
                self.labels.append(describe(st, h))
                sel = np.zeros(MAX_HAND_SIZE, bool)
                sel[_lanes(h, st.cards)] = True
                a = balatroAction(verb=jnp.int32(st.verb), select=jnp.asarray(sel),
                                  target=jnp.int32(st.target), dest=jnp.int32(st.dest))
                scores = ((st.verb == V_PLAY or (st.verb == V_DISCARD and int(h.discards) <= 0))
                          and bool(fix_select(s, a.select, 5).any()))     # no card: a no-op
                self.plays.append(jax.device_get(explain_play(s, a.select)) if scores else None)
                s, r, _ = env_step(s, a)
                self.actions.append(a)
                self.rewards.append(int(r))
                self.raw.append(s)
                h = host(s)
        except Exception as e:              # an authoring error: the scenario fails, with the reason
            self.error = f"{type(e).__name__}: {e}"
        self.views = [View(host(x)) for x in self.raw]
        self.results = []
        if self.error is None:
            for ck in sc.checks:
                want = ck.want
                try:
                    got = ck.got(self)
                    if isinstance(got, np.generic):
                        got = got.item()
                    if callable(want):                  # a value read from another state of the run
                        want = want(self)
                    ok = _same(got, want, ck.tol)
                except Exception as e:
                    got, ok = f"error: {type(e).__name__}: {e}", False
                self.results.append((ck.label, got, want, ok))
        self.ok = self.error is None and all(x[3] for x in self.results)

    def __getitem__(self, k):
        return self.views[k]

    def play(self, k=0):
        """The PlayResult of step k."""
        return self.plays[k]


# ---------------------------------------------------------------------------
# SHOP / PACK ROLLS — for the rules that only show over many shops
# ---------------------------------------------------------------------------
# the state is an argument, so all scenarios share one compile
_shop_rolls = jax.jit(jax.vmap(roll_shop, in_axes=(0, None)))
_pack_rolls = jax.jit(jax.vmap(lambda k, s, pid: roll_pack_cards(k, s, pid)[0], in_axes=(0, None, None)))


def shop_rolls(s, n=2000, seed=1):
    """(n, 4) shop items of n fresh shops of s: joker ids, -1 for a Tarot / Planet / locked slot."""
    keys = jax.random.split(jax.random.key(seed), n)
    roll = _shop_rolls(keys, s)
    return np.where(np.asarray(roll.kinds) == IK_JOKER, np.asarray(roll.ids), -1)


def in_shops(s, name, n=2000):
    """Is joker name in any of n fresh shops of s?"""
    return bool((shop_rolls(s, n) == JOKER_ID[name]).any())


# the jokers after the end of the round (_clear_blind, without the hand) of s with key k
_round_end = jax.jit(jax.vmap(lambda s, k: _clear_blind(s._replace(key=k)).jokers, in_axes=(None, 0)))


def death_rate(s, name, n, seed=1):
    """The share of n end-of-round rolls of s (each with a new key) that destroy joker name."""
    jid = JOKER_ID[name]
    before = int((np.asarray(s.jokers) == jid).sum())
    after = (np.asarray(_round_end(s, jax.random.split(jax.random.key(seed), n))) == jid).sum(1)
    return float((before - after).mean() / before)


def pack_rolls(s, pid, n=2000, seed=1):
    """(n, 5) the items of n packs pid opened in state s."""
    keys = jax.random.split(jax.random.key(seed), n)
    return np.asarray(_pack_rolls(keys, s, jnp.int32(pid)))


def play_of(k=0):
    return lambda r: r.play(k)


# ===========================================================================
# THE SCENARIOS
# ===========================================================================
def _lvl2(hand):
    c, m = hand_chips_mult(jnp.asarray([2 if n == hand else 1 for n in HAND_NAMES]))
    k = HAND_NAMES.index(hand)
    return int(c[k]), int(m[k])


JOKERS = [
    # --- counters that grow or shrink -------------------------------------------------
    S("Jokers", "Green Joker", "+1 Mult per hand, -1 per discard",
      """Green Joker gains +1 Mult before each hand scores and loses 1 Mult per discard, never
      below 0. Play KD KC: Green goes 0 -> 1, so the Pair scores 30 x (2 + 1) = 90. Discard 2H:
      1 -> 0. Discard 3S: it stays at 0.""",
      blind(jokers=("Green Joker",)),
      [play("KD KC"), discard("2H"), discard("3S")],
      [C("the Pair scores 30 x 3 = 90", lambda r: int(r.play(0).score), 90),
       C("Green after the play", lambda r: r[1].jv(0), 1.0),
       C("Green after one discard", lambda r: r[2].jv(0), 0.0),
       C("Green after a second discard (never below 0)", lambda r: r[3].jv(0), 0.0)]),
    S("Jokers", "Ride the Bus", "+1 Mult per hand with no scoring face card",
      """Ride the Bus gains +1 Mult before each hand that scores no face card, and goes back to 0
      when a face card scores. Play 2C (High Card, 5 + 2 = 7 chips): Bus 0 -> 1, 7 x 2 = 14.
      Play 3D (8 chips): Bus 1 -> 2, 8 x 3 = 24. Play KD (15 chips): a face card scores, Bus -> 0,
      15 x 1 = 15.""",
      blind("2C 3D 5H 7S 9C KD KC 4H", jokers=("Ride the Bus",), hands=4),
      [play("2C"), play("3D"), play("KD")],
      [C("2C scores 7 x 2 = 14", lambda r: int(r.play(0).score), 14),
       C("Bus after 2C", lambda r: r[1].jv(0), 1.0),
       C("3D scores 8 x 3 = 24", lambda r: int(r.play(1).score), 24),
       C("Bus after 3D", lambda r: r[2].jv(0), 2.0),
       C("KD scores 15 x 1 = 15", lambda r: int(r.play(2).score), 15),
       C("Bus after KD (a face card scored)", lambda r: r[3].jv(0), 0.0)]),
    S("Jokers", "Popcorn", "-4 Mult at the end of each round",
      """Popcorn gives +20 Mult and loses 4 Mult at the end of each round. The Pair scores
      30 x (2 + 20) = 660 and beats the blind. At the end of the round Popcorn goes 20 -> 16.""",
      blind(clear=True, jokers=("Popcorn",)),
      [play("KD KC")],
      [C("the Pair: mult 2 + 20", lambda r: float(r.play(0).mult), 22.0),
       C("Popcorn after the round", lambda r: r[1].jv(0), 16.0)]),
    S("Jokers", "Popcorn", "eaten when it reaches 0",
      """At +4 Mult, the end of the round takes Popcorn to 0 and it is eaten (destroyed). The
      Pair still gets the +4: 30 x 6 = 180.""",
      blind(clear=True, jokers=("Popcorn",), joker_value=(4.0,)),
      [play("KD KC")],
      [C("the Pair: mult 2 + 4", lambda r: float(r.play(0).mult), 6.0),
       C("jokers after the round", lambda r: r[1].jokers, [])]),
    S("Jokers", "Ice Cream", "-5 Chips per hand played",
      """Ice Cream gives +100 Chips and loses 5 Chips after each hand. Play KD KC: 30 + 100 = 130
      chips, then 100 -> 95. Play QD QC: 30 + 95 = 125 chips, then 95 -> 90.""",
      blind("KD KC QD QC 2H 3S 4C 6D", jokers=("Ice Cream",), hands=4),
      [play("KD KC"), play("QD QC")],
      [C("KD KC chips", lambda r: float(r.play(0).chips), 130.0),
       C("Ice Cream after one hand", lambda r: r[1].jv(0), 95.0),
       C("QD QC chips", lambda r: float(r.play(1).chips), 125.0),
       C("Ice Cream after two hands", lambda r: r[2].jv(0), 90.0)]),
    S("Jokers", "Ice Cream", "melts at 0",
      """At +5 Chips, the hand still gets +5 (35 chips), then Ice Cream reaches 0 and melts.""",
      blind(jokers=("Ice Cream",), joker_value=(5.0,)),
      [play("KD KC")],
      [C("chips 30 + 5", lambda r: float(r.play(0).chips), 35.0),
       C("jokers after the hand", lambda r: r[1].jokers, [])]),
    S("Jokers", "Seltzer", "retriggers for 10 hands, then gone",
      """Seltzer retriggers every played card for the next 10 hands. With 1 hand left on it,
      KD and KC each score twice: 10 + 4 x 10 = 50 chips. Then Seltzer is used up.""",
      blind(jokers=("Seltzer",), joker_value=(1.0,)),
      [play("KD KC")],
      [C("chips 10 + 10 + 10 + 10 + 10", lambda r: float(r.play(0).chips), 50.0),
       C("jokers after the hand", lambda r: r[1].jokers, [])]),
    S("Jokers", "Castle", "+3 Chips per discarded card of the round's suit",
      """Castle gains +3 Chips per discarded card of the round's suit (Spades here). Discard
      AS JS 3S: 3 Spades, +9 Chips. Then play JD: High Card 5 + 10 + Castle 9 = 24 chips x 1.""",
      blind("AS JS JD KH QC 3S 4C 6D", jokers=("Castle",), castle_suit=3),
      [discard("AS JS 3S"), play("JD")],
      [C("Castle after the discard", lambda r: r[1].jv(0), 9.0),
       C("JD chips 5 + 10 + 9", lambda r: float(r.play(1).chips), 24.0)]),
    S("Jokers", "Flash Card", "+2 Mult per reroll",
      """Flash Card gains +2 Mult per shop reroll. Two rerolls cost $5 then $6: Flash Card
      0 -> 2 -> 4, money $20 -> $9, and the next reroll costs $7.""",
      shop(jokers=("Flash Card",), money=20),
      [reroll(), reroll()],
      [C("Flash Card after 1 reroll", lambda r: r[1].jv(0), 2.0),
       C("Flash Card after 2 rerolls", lambda r: r[2].jv(0), 4.0),
       C("money", lambda r: r[2].money, 9),
       C("next reroll price", lambda r: r[2].reroll_cost, 7)]),
    S("Jokers", "Hit the Road", "X0.5 per Jack discarded, back to X1 at the end of the round",
      """Hit the Road gains X0.5 Mult per Jack discarded this round. Discard JS JD: X1 -> X2.
      Play 2C: High Card 7 chips x 1 x 2 = 14, which beats the blind. At the end of the round
      Hit the Road goes back to X1.""",
      blind("JS JD 2C 3D 4H 5S 6C 7D", clear=True, jokers=("Hit the Road",)),
      [discard("JS JD"), play("2C")],
      [C("Hit the Road after the discard", lambda r: r[1].jv(0), 2.0),
       C("2C mult 1 x 2", lambda r: float(r.play(1).mult), 2.0),
       C("Hit the Road after the round", lambda r: r[2].jv(0), 1.0)]),
    S("Jokers", "Ramen", "-X0.01 per card discarded",
      """Ramen gives X2 Mult and loses X0.01 per discarded card. Discard 5 cards: X2 -> X1.95.""",
      blind(jokers=("Ramen",)),
      [discard("2H 3S 4C 6D 7H")],
      [C("Ramen after 5 discarded cards", lambda r: r[1].jv(0), 1.95)]),
    S("Jokers", "Ramen", "eaten at X1",
      """Ramen is eaten when a discard takes it to X1 or below. At X1.03, the 3rd of 5 discarded
      cards takes it to X1.00.""",
      blind(jokers=("Ramen",), joker_value=(1.03,)),
      [discard("2H 3S 4C 6D 7H")],
      [C("jokers after the discard", lambda r: r[1].jokers, [])]),
    S("Jokers", "Campfire", "X0.25 per card sold",
      """Campfire gains X0.25 Mult for each card sold: a joker, a Tarot or a Planet. Sell Joker:
      X1 -> X1.25. Sell The Hermit: X1.25 -> X1.5.""",
      shop(jokers=("Campfire", "Joker"), consumables=("The Hermit",)),
      [sell(1), sell_cons(0)],
      [C("Campfire after selling a joker", lambda r: r[1].jv(0), 1.25),
       C("Campfire after selling a Tarot", lambda r: r[2].jv(0), 1.5)]),
    S("Jokers", "Campfire", "back to X1 after a Boss Blind",
      """Campfire resets to X1 when a Boss Blind is defeated. At X2, the Pair scores
      30 x 2 x 2 = 120 and beats the Boss Blind; then Campfire is X1.""",
      blind(clear=True, jokers=("Campfire",), joker_value=(2.0,), blind_index=2),
      [play("KD KC")],
      [C("the Pair: mult 2 x 2", lambda r: float(r.play(0).mult), 4.0),
       C("Campfire after the Boss Blind", lambda r: r[1].jv(0), 1.0)]),
    S("Jokers", "Campfire", "keeps its value after a Small Blind",
      """Only a Boss Blind resets Campfire. After the Small Blind it stays at X2.""",
      blind(clear=True, jokers=("Campfire",), joker_value=(2.0,)),
      [play("KD KC")],
      [C("Campfire after the Small Blind", lambda r: r[1].jv(0), 2.0)]),
    S("Jokers", "Erosion", "+4 Mult per card below 52",
      """Erosion gives +4 Mult per card below the starting 52 in the full deck. This deck has 50
      cards (KS and AS are gone): +8 Mult. Pair: 30 x (2 + 8) = 300.""",
      blind(jokers=("Erosion",), deck_size=50),
      [play("KD KC")],
      [C("the Pair: mult 2 + 8", lambda r: float(r.play(0).mult), 10.0)]),
    S("Jokers", "Misprint", "+0 to +23 Mult, at random",
      """Misprint adds a random whole number from 0 to 23 to the Mult.""",
      blind(jokers=("Misprint",), key=7),
      [play("KD KC")],
      [C("Misprint's add", lambda r: float(r.play(0).mult) - 2, Between(0, 23)),
       C("a whole number", lambda r: float(r.play(0).mult) % 1, 0.0)]),
    S("Jokers", "Gros Michel + Oops! All 6s + Cavendish", "destroyed at the end of the round; then Cavendish in the shop",
      """Gros Michel gives +15 Mult and has a 1 in 6 chance to be destroyed at the end of the
      round. Three Oops! All 6s make it 8 in 6, so it always happens. The Pair scores
      30 x (2 + 15) = 510, then Gros Michel is destroyed and marked extinct: from now on a shop
      can have Cavendish and never Gros Michel. With this seed, the shop after the round has a
      Cavendish; buy it for $4.""",
      blind(clear=True, jokers=("Gros Michel",) + ("Oops! All 6s",) * 3, key=22),
      [play("KD KC"), buy(0)],
      [C("the Pair: mult 2 + 15", lambda r: float(r.play(0).mult), 17.0),
       C("jokers after the round", lambda r: r[1].jokers, ["Oops! All 6s"] * 3),
       C("Gros Michel is extinct", lambda r: bool(r[1].h.gros_extinct), True),
       C("the shop after the round", lambda r: r[1].shop, ["Cavendish", "Splash"]),
       C("2000 shops after the round: a Cavendish", lambda r: in_shops(r.raw[1], "Cavendish"), True),
       C("2000 shops after the round: a Gros Michel", lambda r: in_shops(r.raw[1], "Gros Michel"), False),
       C("jokers after the buy", lambda r: r[2].jokers, ["Oops! All 6s"] * 3 + ["Cavendish"])]),
    S("Jokers", "Gros Michel", "survives the roll 5 times in 6",
      """Without Oops! All 6s, Gros Michel is destroyed 1 time in 6. With this seed it survives
      ("Safe!"): it stays, it is not extinct, and no shop has a Cavendish. Over 6000 end-of-round
      rolls, it is destroyed in about 1 in 6 (0.167).""",
      blind(clear=True, jokers=("Gros Michel",), key=1),
      [play("KD KC")],
      [C("the Pair: mult 2 + 15", lambda r: float(r.play(0).mult), 17.0),
       C("jokers after the round", lambda r: r[1].jokers, ["Gros Michel"]),
       C("Gros Michel is extinct", lambda r: bool(r[1].h.gros_extinct), False),
       C("2000 shops after the round: a Cavendish", lambda r: in_shops(r.raw[1], "Cavendish"), False),
       C("destroyed in 6000 rounds", lambda r: death_rate(r.raw[0], "Gros Michel", 6000),
         Between(0.15, 0.183))]),
    S("Jokers", "Gros Michel", "selling it does not make it extinct",
      """Only the end-of-round roll makes Gros Michel extinct. Sell it in the shop: it is not
      extinct, so a later shop can have Gros Michel again, and never a Cavendish.""",
      shop(jokers=("Gros Michel",)),
      [sell(0)],
      [C("jokers after the sale", lambda r: r[1].jokers, []),
       C("Gros Michel is extinct", lambda r: bool(r[1].h.gros_extinct), False),
       C("2000 shops: a Gros Michel", lambda r: in_shops(r.raw[1], "Gros Michel"), True),
       C("2000 shops: a Cavendish", lambda r: in_shops(r.raw[1], "Cavendish"), False)]),
    S("Jokers", "Cavendish", "X3 Mult; destroyed 1 time in 1000 at the end of the round",
      """Cavendish gives X3 Mult and has a 1 in 1000 chance to be destroyed at the end of the
      round. The Pair scores 30 x (2 x 3) = 180. This seed is one of the 1 in 1000 rounds:
      Cavendish is destroyed. Gros Michel stays extinct (only Gros Michel changes the flag).
      Over 50000 end-of-round rolls, it is destroyed in about 1 in 1000 (0.001).""",
      blind(clear=True, jokers=("Cavendish",), gros_extinct=True, key=113),
      [play("KD KC")],
      [C("the Pair: mult 2 x 3", lambda r: float(r.play(0).mult), 6.0),
       C("jokers after the round", lambda r: r[1].jokers, []),
       C("Gros Michel is extinct", lambda r: bool(r[1].h.gros_extinct), True),
       C("destroyed in 50000 rounds", lambda r: death_rate(r.raw[0], "Cavendish", 50000),
         Between(0.0005, 0.0015))]),

    # --- money -------------------------------------------------------------------------
    S("Jokers", "Golden Joker", "$4 at the end of the round",
      """Golden Joker earns $4 at the end of the round. Cash out: $3 (Small Blind) + $2 (2 hands
      left) + $0 interest + $4 = $9.""",
      blind(clear=True, jokers=("Golden Joker",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 9), C("money", lambda r: r[1].money, 9)]),
    S("Jokers", "Cloud 9", "$1 per 9 in the full deck",
      """Cloud 9 earns $1 per 9 in the full deck at the end of the round. A standard deck has
      four 9s: cash out $3 + $2 + $4 = $9.""",
      blind(clear=True, jokers=("Cloud 9",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 9)]),
    S("Jokers", "Cloud 9", "a Stone card is not a 9",
      """A Stone card has no rank. With 9C made a Stone card the deck has three 9s: cash out
      $3 + $2 + $3 = $8.""",
      blind(clear=True, jokers=("Cloud 9",), cards_mod={"9C": dict(enh=6)}),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 8)]),
    S("Jokers", "To Do List", "$4 for the To Do hand",
      """To Do List earns $4 when the poker hand is its hand (Pair here). Play KD KC: +$4.""",
      blind(jokers=("To Do List",), joker_state=(10,)),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 4)]),
    S("Jokers", "To Do List", "a new hand at the end of the round",
      """At the end of the round To Do List picks a new poker hand, never the same one and never
      a secret hand (Flush Five, Flush House, Five of a Kind) that was not played this run.""",
      blind(clear=True, jokers=("To Do List",), joker_state=(10,)),
      [play("KD KC")],
      [C("the new hand", lambda r: HAND_NAMES[int(r[1].h.joker_state[0])],
         OneOf(tuple(n for n in HAND_NAMES[3:] if n != "pair"))),
       C("share of Flush Five picks (4000 rolls)", lambda r: _todo_share(r.raw[1].hand_played, 10, "flush_five"), 0.0)]),
    S("Jokers", "To Do List", "a secret hand once it was played",
      """A secret hand becomes visible when it is played (evaluate_play), and To Do List can then
      pick it. Flush Five was played once: the new hand is one of the 9 plain hands or Flush Five,
      not Pair (the old one), so Flush Five comes up 1 time in 9. Flush House and Five of a Kind
      were not played and never come up. (4000 rolls.)""",
      blind(clear=True, jokers=("To Do List",), joker_state=(10,), hand_played=_counts(flush_five=1)),
      [play("KD KC")],
      [C("the new hand", lambda r: HAND_NAMES[int(r[1].h.joker_state[0])],
         OneOf(tuple(n for n in HAND_NAMES if n not in ("pair", "flush_house", "five_kind")))),
       C("share of Flush Five picks", lambda r: _todo_share(r.raw[1].hand_played, 10, "flush_five"),
         Between(0.09, 0.133)),
       C("share of Flush House picks", lambda r: _todo_share(r.raw[1].hand_played, 10, "flush_house"), 0.0),
       C("share of Five of a Kind picks", lambda r: _todo_share(r.raw[1].hand_played, 10, "five_kind"), 0.0)]),
    S("Jokers", "To Do List", "a bought To Do List can pick a played secret hand",
      """A new To Do List picks its hand from every visible hand (no old hand to skip). Five of a
      Kind was played: 10 hands, so Five of a Kind comes up 1 time in 10. (4000 rolls.)""",
      shop(shop_slots=("To Do List",), hand_played=_counts(five_kind=1)),
      [buy(0)],
      [C("its hand", lambda r: HAND_NAMES[int(r[1].h.joker_state[0])],
         OneOf(tuple(n for n in HAND_NAMES if n not in ("flush_five", "flush_house")))),
       C("share of Five of a Kind picks", lambda r: _todo_share(r.raw[0].hand_played, -1, "five_kind"),
         Between(0.08, 0.12)),
       C("share of Flush Five picks", lambda r: _todo_share(r.raw[0].hand_played, -1, "flush_five"), 0.0)]),
    S("Jokers", "Mail-In Rebate", "$5 per discarded card of the round's rank",
      """Mail-In Rebate earns $5 per discarded card of the round's rank (Ace here). Discard
      AS AH 2C: two Aces, +$10.""",
      blind("AS AH 2C 3D 4H 5S 6C 7D", jokers=("Mail-In Rebate",), mail_rank=12),
      [discard("AS AH 2C")],
      [C("money", lambda r: r[1].money, 10)]),
    S("Jokers", "Faceless Joker", "$5 for 3 or more face cards in one discard",
      """Faceless Joker earns $5 when a discard has 3 or more face cards. Discard JS QC (2 faces):
      $0. Discard KH JD QD (3 faces): +$5.""",
      blind("JS QC KH JD QD KS 2C 3D", jokers=("Faceless Joker",)),
      [discard("JS QC"), discard("KH JD QD")],
      [C("money after 2 faces", lambda r: r[1].money, 0), C("money after 3 faces", lambda r: r[2].money, 5)]),
    S("Jokers", "Delayed Gratification", "$2 per discard left, if none was used",
      """Delayed Gratification earns $2 per discard left at the end of the round when no discard
      was used. 2 discards left: $4. Cash out $3 + $2 + $4 = $9.""",
      blind(clear=True, jokers=("Delayed Gratification",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 9)]),
    S("Jokers", "Delayed Gratification", "nothing after a discard",
      """One discard was used this round, so Delayed Gratification pays $0. Cash out $3 + $2 = $5.""",
      blind(clear=True, jokers=("Delayed Gratification",), discards_used=1),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 5)]),
    S("Jokers", "Rocket", "$1 at the end of the round",
      """Rocket earns $1 at the end of the round. Cash out $3 + $2 + $1 = $6.""",
      blind(clear=True, jokers=("Rocket",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 6), C("Rocket's payout after", lambda r: r[1].jv(0), 1.0)]),
    S("Jokers", "Rocket", "+$2 payout after a Boss Blind",
      """Rocket's payout goes up by $2 when a Boss Blind is defeated, and pays the new value at
      once. Cash out $5 (Boss Blind) + $2 + $3 = $10.""",
      blind(clear=True, jokers=("Rocket",), blind_index=2),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 10), C("Rocket's payout after", lambda r: r[1].jv(0), 3.0)]),
    S("Jokers", "Egg", "+$3 sell value per round",
      """Egg costs $4 and sells for $2. At the end of each round it gains $3 of sell value: $5.""",
      blind(clear=True, jokers=("Egg",)),
      [play("KD KC")],
      [C("Egg sell value before", lambda r: r[0].jsell(0), 2), C("after", lambda r: r[1].jsell(0), 5)]),
    S("Jokers", "Gift Card", "+$1 sell value on every joker and consumable per round",
      """Gift Card adds $1 of sell value to every joker and every Tarot / Planet held at the end of
      the round. Gift Card $3 -> $4, Joker $1 -> $2, The Hermit $1 -> $2. Cash out $3 + $2 = $5;
      then sell The Hermit for $2: money $7.""",
      blind(clear=True, jokers=("Gift Card", "Joker"), consumables=("The Hermit",)),
      [play("KD KC"), sell_cons(0)],
      [C("Gift Card sell value", lambda r: r[1].jsell(0), 4), C("Joker sell value", lambda r: r[1].jsell(1), 2),
       C("The Hermit sell value", lambda r: r[1].csell(0), 2), C("money after selling it", lambda r: r[2].money, 7)]),
    S("Jokers", "Gift Card + Blue Seal", "a Planet made at the end of the round misses that round",
      """The jokers' end-of-round effects come before the held cards. A Blue Seal's Planet is made
      after Gift Card acted, so it sells for $1; The Hermit, held before, sells for $2.""",
      blind(clear=True, jokers=("Gift Card",), consumables=("The Hermit",), cards_mod={"2H": dict(seal=3)}),
      [play("KD KC")],
      [C("consumables", lambda r: r[1].cons, ["The Hermit", "Mercury"]),
       C("sell values", lambda r: [r[1].csell(0), r[1].csell(1)], [2, 1])]),
    S("Jokers", "Gift Card", "the sell value grows every round",
      """Two rounds with Gift Card: a Tarot held through both sells for $1 + $1 + $1 = $3.""",
      blind(clear=True, jokers=("Gift Card",), consumables=("The Hermit",), cons_sell=(2,)),
      [play("KD KC")],
      [C("The Hermit sell value", lambda r: r[1].csell(0), 3)]),
    S("Jokers", "Credit Card", "buy down to -$20",
      """Credit Card lets money go down to -$20. With $0, buy Bull ($6): money -$6.""",
      shop(jokers=("Credit Card",), money=0, shop_slots=("Bull",)),
      [buy(0)],
      [C("money", lambda r: r[1].money, -6), C("jokers", lambda r: r[1].jokers, ["Credit Card", "Bull"])]),
    S("Jokers", "To the Moon", "+$1 interest per $5",
      """To the Moon earns an extra $1 of interest for every $5 held. With $25: interest
      2 x 5 = $10. Cash out $3 + $2 + $10 = $15.""",
      blind(clear=True, jokers=("To the Moon",), money=25),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 15), C("money", lambda r: r[1].money, 40)]),
    S("Jokers", "Chaos the Clown", "1 free reroll per shop",
      """Chaos the Clown gives 1 free reroll in each shop. With $0, the first reroll is free and
      the price stays $5. The second reroll needs $5, so nothing happens.""",
      shop(jokers=("Chaos the Clown",), money=0, free_rerolls=1),
      [reroll(), reroll()],
      [C("money after the free reroll", lambda r: r[1].money, 0),
       C("free rerolls left", lambda r: r[1].free_rerolls, 0),
       C("reroll price", lambda r: r[1].reroll_cost, 5),
       C("the second reroll changes nothing", lambda r: bool(np.array_equal(r[2].h.shop_slots, r[1].h.shop_slots)),
         True)]),
    S("Jokers", "Rough Gem", "$1 per scored Diamond",
      """Rough Gem earns $1 per scored Diamond card. A Flush of 5 Diamonds: +$5.""",
      blind("KD QD 2D 5D 9D 3S 4C 6H", jokers=("Rough Gem",)),
      [play("KD QD 2D 5D 9D")],
      [C("money", lambda r: r[1].money, 5)]),
    S("Jokers", "Business Card + Oops! All 6s", "$2 per scored face card, 1 in 2",
      """Business Card: each scored face card has a 1 in 2 chance to give $2. Oops! All 6s makes
      it 2 in 2. KD and KC: +$4.""",
      blind(jokers=("Business Card", "Oops! All 6s")),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 4)]),
    S("Jokers", "Reserved Parking + Oops! All 6s", "$1 per held face card, 1 in 2",
      """Reserved Parking: each face card held in hand has a 1 in 2 chance to give $1. Oops! All
      6s makes it 2 in 2. Play 2C 3D with QS QH JD held: +$3.""",
      blind("2C 3D QS QH JD 5S 6C 7D", jokers=("Reserved Parking", "Oops! All 6s")),
      [play("2C 3D")],
      [C("money", lambda r: r[1].money, 3)]),
    S("Jokers", "Golden Ticket", "$4 per scored Gold card",
      """Golden Ticket earns $4 for each scored Gold card. KD is a Gold card: +$4.""",
      blind(jokers=("Golden Ticket",), cards_mod={"KD": dict(enh=7)}),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 4)]),
    S("Jokers", "Satellite", "$1 per unique Planet used",
      """Satellite earns $1 per unique Planet card used this run. Mercury and Jupiter were used:
      cash out $3 + $2 + $2 = $7.""",
      blind(clear=True, jokers=("Satellite",),
            planets_used=[n in ("pair", "flush") for n in HAND_NAMES]),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 7)]),
    S("Jokers", "Trading Card", "the first discard of 1 card: destroy it, +$3",
      """Trading Card: if the first discard of the round is a single card, that card is destroyed
      and you earn $3. Discard 2H: +$3, the deck has 51 cards. A second 1-card discard does
      nothing.""",
      blind(jokers=("Trading Card",)),
      [discard("2H"), discard("3S")],
      [C("money", lambda r: r[1].money, 3), C("deck size", lambda r: r[1].deck_size, 51),
       C("2H destroyed", lambda r: r[1].card("2H").live, False),
       C("money after the 2nd discard", lambda r: r[2].money, 3),
       C("3S still in the deck", lambda r: r[2].card("3S").live, True)]),

    # --- hand size, hands, discards, slots --------------------------------------------------
    S("Jokers", "Juggler", "+1 hand size",
      """Juggler gives +1 hand size. Buy it ($4), start the blind: 9 cards are dealt.""",
      shop(shop_slots=("Juggler",)),
      [buy(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 9), C("cards dealt", lambda r: r[3].in_hand, 9)]),
    S("Jokers", "Drunkard", "+1 discard each round",
      """Drunkard gives +1 discard each round: 3 + 1 = 4 at the next blind.""",
      shop(shop_slots=("Drunkard",)),
      [buy(0), next_blind(), play_blind()],
      [C("discards", lambda r: r[3].discards, 4)]),
    S("Jokers", "Merry Andy", "+3 discards, -1 hand size",
      """Merry Andy gives +3 discards each round and -1 hand size: 6 discards, 7 cards.""",
      shop(shop_slots=("Merry Andy",)),
      [buy(0), next_blind(), play_blind()],
      [C("discards", lambda r: r[3].discards, 6), C("hand size", lambda r: r[3].hand_size, 7)]),
    S("Jokers", "Troubadour", "+2 hand size, -1 hand",
      """Troubadour gives +2 hand size and -1 hand each round: 10 cards, 3 hands.""",
      shop(shop_slots=("Troubadour",)),
      [buy(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 10), C("hands", lambda r: r[3].hands, 3)]),
    S("Jokers", "Turtle Bean", "+5 hand size",
      """Turtle Bean gives +5 hand size: 13 cards at the next blind.""",
      shop(shop_slots=("Turtle Bean",)),
      [buy(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 13), C("cards dealt", lambda r: r[3].in_hand, 13)]),
    S("Jokers", "Turtle Bean", "-1 hand size each round",
      """Turtle Bean loses 1 hand size at the end of each round: +5 -> +4, hand size 13 -> 12.""",
      blind(clear=True, jokers=("Turtle Bean",), hand_size=13, base_hand_size=8),
      [play("KD KC")],
      [C("Turtle Bean after the round", lambda r: r[1].jv(0), 4.0),
       C("hand size", lambda r: r[1].hand_size, 12)]),
    S("Jokers", "Stuntman", "-2 hand size",
      """Stuntman gives +250 Chips and -2 hand size: 6 cards at the next blind.""",
      shop(shop_slots=("Stuntman",)),
      [buy(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 6)]),
    S("Jokers", "Burglar", "+3 hands, no discards",
      """When the blind starts, Burglar gives +3 hands and removes all discards: 7 hands, 0
      discards.""",
      shop(jokers=("Burglar",)),
      [next_blind(), play_blind()],
      [C("hands", lambda r: r[2].hands, 7), C("discards", lambda r: r[2].discards, 0)]),

    # --- cards, Tarots, Planets, packs -----------------------------------------------------
    S("Jokers", "Marble Joker", "a Stone card into the deck at each blind",
      """When the blind starts, Marble Joker adds one Stone card to the deck: 53 cards, 1 Stone.""",
      shop(jokers=("Marble Joker",)),
      [next_blind(), play_blind()],
      [C("deck size", lambda r: r[2].deck_size, 53), C("Stone cards", lambda r: r[2].n_cards(enh="Stone"), 1)]),
    S("Jokers", "Certificate", "a card with a seal into the hand",
      """When the round starts, Certificate adds a random playing card with a random seal to the
      hand: 8 + 1 = 9 cards in hand, 53 in the deck, 1 card with a seal.""",
      shop(jokers=("Certificate",)),
      [next_blind(), play_blind()],
      [C("cards in hand", lambda r: r[2].in_hand, 9), C("deck size", lambda r: r[2].deck_size, 53),
       C("cards with a seal", lambda r: 53 - r[2].n_cards(seal="") , 1)]),
    S("Jokers", "Cartomancer", "a Tarot at each blind",
      """When the blind starts, Cartomancer creates a Tarot card (if there is room).""",
      shop(jokers=("Cartomancer",)),
      [next_blind(), play_blind()],
      [C("Tarots held", lambda r: r[2].n_cons(CK_TAROT), 1)]),
    S("Jokers", "DNA", "the first hand of 1 card is copied",
      """If the first hand of the round is a single card, DNA adds a permanent copy of it to the
      deck and draws it to the hand. Play KD alone: the deck has 53 cards, two King of Diamonds,
      and the copy is in hand.""",
      blind(jokers=("DNA",)),
      [play("KD")],
      [C("deck size", lambda r: r[1].deck_size, 53),
       C("Kings of Diamonds in the deck", lambda r: r[1].n_cards(rank=11, suit=1), 2),
       C("a King of Diamonds in hand", lambda r: r[1].hand.count("KD"), 1)]),
    S("Jokers", "DNA", "not on the second hand",
      """DNA copies only on the first hand of the round. Play KD KC first, then 2C alone on the
      2nd hand: no copy, the deck stays at 52.""",
      blind(jokers=("DNA",)),
      [play("KD KC"), play("2C")],
      [C("deck size", lambda r: r[2].deck_size, 52)]),
    S("Jokers", "Hiker", "+5 permanent Chips on each scored card",
      """Every played card that scores gains +5 Chips for good. This hand scores 30 chips (the
      bonus counts from the next time); then KD and KC each have +5.""",
      blind(jokers=("Hiker",)),
      [play("KD KC")],
      [C("chips", lambda r: float(r.play(0).chips), 30.0),
       C("KD bonus", lambda r: r[1].card("KD").bonus, 5.0), C("KC bonus", lambda r: r[1].card("KC").bonus, 5.0),
       C("2H (held) bonus", lambda r: r[1].card("2H").bonus, 0.0)]),
    S("Jokers", "Midas Mask", "played face cards become Gold",
      """All played face cards that score become Gold cards. KD and KC are Gold after the hand.""",
      blind(jokers=("Midas Mask",)),
      [play("KD KC")],
      [C("KD", lambda r: r[1].card("KD").enh, "Gold"), C("KC", lambda r: r[1].card("KC").enh, "Gold")]),
    S("Jokers", "Vampire", "X0.1 per scored Enhanced card, takes the enhancement",
      """Vampire gains X0.1 Mult per scoring Enhanced card and removes the enhancement first.
      KD (Bonus) and KC (Mult): Vampire X1 -> X1.2, and the cards score as plain cards:
      30 x 2 x 1.2 = 72.""",
      blind(jokers=("Vampire",), cards_mod={"KD": dict(enh=1), "KC": dict(enh=2)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 72), C("Vampire after", lambda r: r[1].jv(0), 1.2),
       C("KD", lambda r: r[1].card("KD").enh, "none"), C("KC", lambda r: r[1].card("KC").enh, "none")]),
    S("Jokers", "Vampire + Flower Pot", "Flower Pot reads the cards after Vampire",
      """A Wild card counts for every suit, but Vampire removes the enhancement before Flower Pot
      looks. 2C 5C AH 4C 3D with 4C Wild: after Vampire there is no Spade, so no X3. Straight
      30 + 2 + 5 + 11 + 4 + 3 = 55 chips x 4 x 1.1 (Vampire) = 242.""",
      blind("2C 5C AH 4C 3D 9S TS JS", jokers=("Flower Pot", "Vampire"), cards_mod={"4C": dict(enh=3)}),
      [play("2C 5C AH 4C 3D")],
      [C("mult 4 x 1.1", lambda r: float(r.play(0).mult), 4.4), C("score", lambda r: int(r.play(0).score), 242)]),
    S("Jokers", "Vampire + Midas Mask", "Vampire first: Vampire scales, then the cards turn Gold",
      """Midas Mask and Vampire act one after the other, in slot order (left to right). KD is a
      Bonus card. Vampire (slot 1) takes KD's Bonus: X1 -> X1.1. Then Midas Mask (slot 2) makes
      KD and KC Gold. The Pair scores 30 x (2 x 1.1) = 66 (no +30: the Bonus is gone), and both
      Kings stay Gold.""",
      blind(jokers=("Vampire", "Midas Mask"), cards_mod={"KD": dict(enh=1)}),
      [play("KD KC")],
      [C("Vampire after", lambda r: r[1].jv(0), 1.1), C("score", lambda r: int(r.play(0).score), 66),
       C("KD", lambda r: r[1].card("KD").enh, "Gold"), C("KC", lambda r: r[1].card("KC").enh, "Gold")]),
    S("Jokers", "Midas Mask + Vampire", "Midas Mask first: the cards turn Gold, then Vampire takes the Gold",
      """In the other order, Midas Mask (slot 1) makes the plain KD and KC Gold, and then Vampire
      (slot 2) takes both Gold enhancements: X1 -> X1.2. The Pair scores 30 x (2 x 1.2) = 72, and
      both Kings end plain.""",
      blind(jokers=("Midas Mask", "Vampire")),
      [play("KD KC")],
      [C("Vampire after", lambda r: r[1].jv(1), 1.2), C("score", lambda r: int(r.play(0).score), 72),
       C("KD", lambda r: r[1].card("KD").enh, "none"), C("KC", lambda r: r[1].card("KC").enh, "none")]),
    S("Jokers", "Vampire + Midas Mask", "a second Vampire skips the cards the first one took",
      """A card a Vampire takes stays marked until the hand ends, and a later Vampire skips it.
      (In a run, two Vampires need Showman.) KD is a Bonus card. Vampire 1 takes KD's Bonus:
      X1.1, KD marked. Midas Mask makes KD and KC Gold. Vampire 2 skips KD and takes KC's Gold:
      X1.1. The Pair scores 30 x (2 x 1.1 x 1.1) = 72.6, so 72. KD stays Gold, KC ends plain.""",
      blind(jokers=("Vampire", "Midas Mask", "Vampire"), cards_mod={"KD": dict(enh=1)}),
      [play("KD KC")],
      [C("Vampire 1 after", lambda r: r[1].jv(0), 1.1), C("Vampire 2 after", lambda r: r[1].jv(2), 1.1),
       C("score", lambda r: int(r.play(0).score), 72),
       C("KD", lambda r: r[1].card("KD").enh, "Gold"), C("KC", lambda r: r[1].card("KC").enh, "none")]),
    S("Jokers", "Midas Mask + Vampire", "a second Midas Mask does not clear the mark",
      """Midas Mask 1 makes KD and KC Gold. Vampire 1 takes both: X1.2, both marked. Midas Mask 2
      makes them Gold again, but Vampire 2 skips both marked cards and stays X1. The Pair scores
      30 x (2 x 1.2) = 72, and both Kings end Gold. (In a run, two Vampires need Showman.)""",
      blind(jokers=("Midas Mask", "Vampire", "Midas Mask", "Vampire")),
      [play("KD KC")],
      [C("Vampire 1 after", lambda r: r[1].jv(1), 1.2), C("Vampire 2 after", lambda r: r[1].jv(3), 1.0),
       C("score", lambda r: int(r.play(0).score), 72),
       C("KD", lambda r: r[1].card("KD").enh, "Gold"), C("KC", lambda r: r[1].card("KC").enh, "Gold")]),
    S("Jokers", "Vampire + Ride the Bus", "Ride the Bus sees the cards after a Vampire to its left",
      """The BEFORE step goes one joker at a time, left to right. KD is a Stone card: no rank, so
      not a face card. Vampire (slot 1) takes the Stone, and KD is a plain King again: a face card.
      Ride the Bus (slot 2, +3 Mult) then sees a scoring face card and resets to 0. High Card:
      5 + 2 (2H) + 10 (KD, plain now) = 17 chips x (1 + 0) x 1.1 = 18.7, so 18.""",
      blind(jokers=("Vampire", "Ride the Bus"), joker_value=(1.0, 3.0), cards_mod={"KD": dict(enh=6)}),
      [play("KD 2H")],
      [C("Ride the Bus after", lambda r: r[1].jv(1), 0.0), C("Vampire after", lambda r: r[1].jv(0), 1.1),
       C("score", lambda r: int(r.play(0).score), 18), C("KD", lambda r: r[1].card("KD").enh, "none")]),
    S("Jokers", "Ride the Bus + Vampire", "Ride the Bus acts before a Vampire to its right",
      """The other order: Ride the Bus (slot 1, +3 Mult) acts first and sees KD still Stone (not a
      face card), so it gains +1: 4. Then Vampire (slot 2) takes the Stone: X1.1. High Card:
      17 chips x (1 + 4) x 1.1 = 93.5, so 93.""",
      blind(jokers=("Ride the Bus", "Vampire"), joker_value=(3.0, 1.0), cards_mod={"KD": dict(enh=6)}),
      [play("KD 2H")],
      [C("Ride the Bus after", lambda r: r[1].jv(0), 4.0), C("Vampire after", lambda r: r[1].jv(1), 1.1),
       C("score", lambda r: int(r.play(0).score), 93)]),
    S("Jokers", "DNA + Midas Mask", "DNA copies the card before a Midas Mask to its right",
      """DNA copies the card as it is when DNA acts. DNA (slot 1) copies the plain KD; then Midas
      Mask (slot 2) makes the played KD Gold. The deck has 53 cards: a Gold KD (the played one)
      and a plain KD (the copy, now in hand).""",
      blind(jokers=("DNA", "Midas Mask")),
      [play("KD")],
      [C("deck size", lambda r: r[1].deck_size, 53),
       C("Gold Kings of Diamonds", lambda r: r[1].n_cards(rank=11, suit=1, enh="Gold"), 1),
       C("plain Kings of Diamonds", lambda r: r[1].n_cards(rank=11, suit=1, enh=""), 1),
       C("a King of Diamonds in hand", lambda r: r[1].hand.count("KD"), 1)]),
    S("Jokers", "Midas Mask + DNA", "DNA copies the Gold card after a Midas Mask to its left",
      """Midas Mask (slot 1) makes KD Gold first; DNA (slot 2) then copies the Gold KD. Both Kings
      of Diamonds are Gold.""",
      blind(jokers=("Midas Mask", "DNA")),
      [play("KD")],
      [C("deck size", lambda r: r[1].deck_size, 53),
       C("Gold Kings of Diamonds", lambda r: r[1].n_cards(rank=11, suit=1, enh="Gold"), 2)]),
    S("Jokers", "DNA + Vampire", "DNA copies the Bonus card before a Vampire takes it",
      """KD is a Bonus card. DNA (slot 1) copies it with the Bonus; then Vampire (slot 2) takes the
      played KD's Bonus: X1.1. The copy keeps the Bonus, the played KD ends plain.""",
      blind(jokers=("DNA", "Vampire"), cards_mod={"KD": dict(enh=1)}),
      [play("KD")],
      [C("Vampire after", lambda r: r[1].jv(1), 1.1),
       C("Bonus Kings of Diamonds", lambda r: r[1].n_cards(rank=11, suit=1, enh="Bonus"), 1),
       C("plain Kings of Diamonds", lambda r: r[1].n_cards(rank=11, suit=1, enh=""), 1)]),
    S("Jokers", "Superposition", "a Tarot for an Ace in a Straight",
      """Superposition creates a Tarot card if the poker hand has a scoring Ace and a Straight.
      Play A 2 3 4 5: one Tarot.""",
      blind("AS 2D 3H 4C 5S 9D TC JH", jokers=("Superposition",)),
      [play("AS 2D 3H 4C 5S")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 1)]),
    S("Jokers", "8 Ball + Oops! All 6s", "a Tarot per scored 8, 1 in 4",
      """8 Ball: each scored 8 has a 1 in 4 chance to create a Tarot. Two Oops! All 6s make it
      4 in 4. A Pair of 8s: two Tarots (2 slots).""",
      blind("8S 8D 2H 3S 4C 6D 7H 9S", jokers=("8 Ball", "Oops! All 6s", "Oops! All 6s")),
      [play("8S 8D")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 2)]),
    S("Jokers", "Vagabond", "a Tarot for a hand played with $4 or less",
      """Vagabond creates a Tarot card if the hand is played with $4 or less. $4: one Tarot.""",
      blind(jokers=("Vagabond",), money=4),
      [play("KD KC")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 1)]),
    S("Jokers", "Vagabond", "nothing at $5",
      """With $5, Vagabond creates nothing.""",
      blind(jokers=("Vagabond",), money=5),
      [play("KD KC")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 0)]),
    S("Jokers", "Vagabond + Gold Seal", "the money before the hand counts",
      """Vagabond reads the money from before the hand: the $3 of a Gold Seal in the same hand is
      paid later. $4 + $3 = $7 after the hand, but Vagabond saw $4: one Tarot.""",
      blind(jokers=("Vagabond",), money=4, cards_mod={"KD": dict(seal=1)}),
      [play("KD KC")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 1), C("money", lambda r: r[1].money, 7)]),
    S("Jokers", "Space Joker + Oops! All 6s", "levels up the played hand, 1 in 4",
      """Space Joker: 1 in 4 to upgrade the played poker hand before it scores. Two Oops! All 6s
      make it 4 in 4. Pair level 2 is 25 chips x 3 mult: 25 + 20 = 45 x 3 = 135.""",
      blind(jokers=("Space Joker", "Oops! All 6s", "Oops! All 6s")),
      [play("KD KC")],
      [C("Pair level", lambda r: r[1].level("pair"), 2), C("score", lambda r: int(r.play(0).score), 135)]),
    S("Jokers", "Constellation", "X0.1 per Planet used",
      """Constellation gains X0.1 Mult each time a Planet card is used. Use Mercury: X1.1 and Pair
      level 2. Then KD KC: (25 + 20) x 3 x 1.1 = 148.5, floor 148.""",
      blind(jokers=("Constellation",), consumables=("Mercury",)),
      [use(0), play("KD KC")],
      [C("Constellation after Mercury", lambda r: r[1].jv(0), 1.1),
       C("score", lambda r: int(r.play(1).score), 148)]),
    S("Jokers", "Fortune Teller", "+1 Mult per Tarot used",
      """Fortune Teller gives +1 Mult per Tarot card used this run. Use The Hermit (at $0 it
      doubles nothing), then the Pair scores 30 x (2 + 1) = 90.""",
      blind(jokers=("Fortune Teller",), consumables=("The Hermit",)),
      [use(0), play("KD KC")],
      [C("Tarots used", lambda r: r[1].tarots_used, 1), C("score", lambda r: int(r.play(1).score), 90)]),
    S("Jokers", "Red Card", "+3 Mult per pack skipped",
      """Red Card gains +3 Mult when any Booster Pack is skipped. Open an Arcana Pack, skip it:
      Red Card 0 -> 3.""",
      shop(jokers=("Red Card",), pack_slots=(PACK["Arcana Pack"], -1)),
      [open_pack(0), skip()],
      [C("Red Card after the skip", lambda r: r[2].jv(0), 3.0), C("stage", lambda r: r[2].stage, "SHOP")]),
    S("Jokers", "Hologram", "X0.25 per card added to the deck",
      """Hologram gains X0.25 Mult every time a playing card is added to the deck. Take a card
      from a Standard Pack: X1 -> X1.25, 53 cards.""",
      shop(jokers=("Hologram",), pack_slots=(PACK["Standard Pack"], -1)),
      [open_pack(0), pick(0)],
      [C("Hologram after", lambda r: r[2].jv(0), 1.25), C("deck size", lambda r: r[2].deck_size, 53)]),
    S("Jokers", "Hallucination + Oops! All 6s", "a Tarot when a pack is opened, 1 in 2",
      """Hallucination: 1 in 2 to create a Tarot when any Booster Pack is opened. Oops! All 6s
      makes it 2 in 2. Open a Standard Pack: one Tarot.""",
      shop(jokers=("Hallucination", "Oops! All 6s"), pack_slots=(PACK["Standard Pack"], -1)),
      [open_pack(0)],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 1)]),
    S("Jokers", "Lucky Cat + Oops! All 6s", "X0.25 per Lucky card trigger",
      """Lucky Cat gains X0.25 each time a Lucky card triggers. Four Oops! All 6s make both Lucky
      chances certain (1 in 5 for +20 Mult, 1 in 15 for $20). KD is Lucky: +20 Mult and $20, Lucky
      Cat X1 -> X1.25 in time for this hand: 30 x (2 + 20) x 1.25 = 825.""",
      blind(jokers=("Lucky Cat",) + ("Oops! All 6s",) * 4, cards_mod={"KD": dict(enh=8)}),
      [play("KD KC")],
      [C("Lucky Cat after", lambda r: r[1].jv(0), 1.25), C("score", lambda r: int(r.play(0).score), 825),
       C("money", lambda r: r[1].money, 20)]),
    S("Jokers", "Glass Joker + The Hanged Man", "X0.75 per Glass card destroyed",
      """Glass Joker gains X0.75 Mult for every Glass card destroyed. The Hanged Man destroys KD
      (Glass): X1 -> X1.75.""",
      blind(jokers=("Glass Joker",), consumables=("The Hanged Man",), cards_mod={"KD": dict(enh=4)}),
      [use(0, "KD")],
      [C("Glass Joker after", lambda r: r[1].jv(0), 1.75), C("deck size", lambda r: r[1].deck_size, 51)]),
    S("Jokers", "Glass Joker + Oops! All 6s", "a Glass card that breaks counts after the hand",
      """A Glass card has a 1 in 4 chance to break after it scores; two Oops! All 6s make it
      certain. KD (Glass, X2) scores 30 x 2 x 2 = 120; Glass Joker is still X1 for this hand. Then
      KD breaks and Glass Joker is X1.75.""",
      blind(jokers=("Glass Joker", "Oops! All 6s", "Oops! All 6s"), cards_mod={"KD": dict(enh=4)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 120), C("Glass Joker after", lambda r: r[1].jv(0), 1.75),
       C("KD destroyed", lambda r: r[1].card("KD").live, False)]),
    S("Jokers", "Burnt Joker", "levels up the first discarded hand",
      """Burnt Joker upgrades the level of the first discarded poker hand each round. Discard
      KD KC (a Pair): Pair level 2. Discard 2H 2C (another Pair): no change.""",
      blind(jokers=("Burnt Joker",)),
      [discard("KD KC"), discard("2H 2C")],
      [C("Pair level after the 1st discard", lambda r: r[1].level("pair"), 2),
       C("Pair level after the 2nd discard", lambda r: r[2].level("pair"), 2)]),
    S("Jokers", "Astronomer", "Planets and Celestial Packs are free",
      """Astronomer makes every Planet card and Celestial Pack in the shop free. With $0: buy
      Mercury for $0, open a Celestial Pack for $0.""",
      shop(jokers=("Astronomer",), money=0, shop_slots=(CONS_ID["Mercury"],), shop_kind=(IK_PLANET,),
           pack_slots=(PACK["Celestial Pack"], -1)),
      [buy(0), open_pack(0)],
      [C("consumables", lambda r: r[1].cons, ["Mercury"]), C("money", lambda r: r[2].money, 0),
       C("stage", lambda r: r[2].stage, "PACK")]),
    S("Jokers", "Showman", "jokers may appear more than once",
      """Without Showman a held joker never appears in the shop and no joker appears twice in one
      shop. Showman allows both. (2000 shops of 4 slots each way, Joker held.)""",
      shop(jokers=("Showman", "Joker"), shop_size=4),
      [reroll()],
      [C("Showman: a held joker appears", lambda r: bool((shop_rolls(r.raw[0]) == JOKER_ID["Joker"]).any()), True),
       C("Showman: a joker twice in one shop", lambda r: any(
           len(set(x for x in row if x >= 0)) < (row >= 0).sum() for row in shop_rolls(r.raw[0])), True),
       C("no Showman: a held joker appears", lambda r: bool((shop_rolls(state(**shop(
           jokers=("Joker",), shop_size=4))) == JOKER_ID["Joker"]).any()), False)]),
]

TAROTS = [
    S("Tarots", "The Fool", "creates the last Tarot or Planet used",
      """The Fool creates a copy of the last Tarot or Planet card used this run (not The Fool).
      The last card used was The Hermit: The Fool becomes The Hermit.""",
      blind(consumables=("The Fool",), last_cons=CONS_ID["The Hermit"]),
      [use(0)],
      [C("consumables", lambda r: r[1].cons, ["The Hermit"])]),
    S("Tarots", "The Fool", "nothing to copy yet",
      """With no Tarot or Planet used yet, The Fool cannot be used.""",
      blind(consumables=("The Fool",)),
      [use(0)],
      [C("consumables", lambda r: r[1].cons, ["The Fool"])]),
    S("Tarots", "The Fool", "a second Fool cannot copy",
      """After The Fool is used, The Fool is the last card used, so a second Fool cannot be used
      until another Tarot or Planet is. The first Fool makes The Hermit; the second stays.""",
      blind(consumables=("The Fool", "The Fool"), last_cons=CONS_ID["The Hermit"]),
      [use(0), use(0)],
      [C("after the 1st Fool", lambda r: r[1].cons, ["The Fool", "The Hermit"]),
       C("after the 2nd Fool", lambda r: r[2].cons, ["The Fool", "The Hermit"])]),
    S("Tarots", "The Magician", "2 cards become Lucky",
      """The Magician enhances up to 2 selected cards into Lucky cards.""",
      blind(consumables=("The Magician",)),
      [use(0, "KD KC")],
      [C("KD", lambda r: r[1].card("KD").enh, "Lucky"), C("KC", lambda r: r[1].card("KC").enh, "Lucky"),
       C("Tarots used", lambda r: r[1].tarots_used, 1)]),
    S("Tarots", "The High Priestess", "2 random Planets",
      """The High Priestess creates up to 2 random Planet cards (if there is room). Its own slot is
      free once it is used: 2 Planets.""",
      blind(consumables=("The High Priestess",)),
      [use(0)],
      [C("Planets held", lambda r: r[1].n_cons(CK_PLANET), 2)]),
    S("Tarots", "The Empress", "2 cards become Mult cards",
      """The Empress enhances up to 2 selected cards into Mult cards (+4 Mult).""",
      blind(consumables=("The Empress",)),
      [use(0, "KD KC")],
      [C("KD", lambda r: r[1].card("KD").enh, "Mult"), C("KC", lambda r: r[1].card("KC").enh, "Mult")]),
    S("Tarots", "The Emperor", "2 random Tarots",
      """The Emperor creates up to 2 random Tarot cards (if there is room).""",
      blind(consumables=("The Emperor",)),
      [use(0)],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 2)]),
    S("Tarots", "The Hierophant", "2 cards become Bonus cards",
      """The Hierophant enhances up to 2 selected cards into Bonus cards (+30 Chips).""",
      blind(consumables=("The Hierophant",)),
      [use(0, "KD KC")],
      [C("KD", lambda r: r[1].card("KD").enh, "Bonus"), C("KC", lambda r: r[1].card("KC").enh, "Bonus")]),
    S("Tarots", "The Lovers", "1 card becomes Wild",
      """The Lovers enhances 1 selected card into a Wild card.""",
      blind(consumables=("The Lovers",)),
      [use(0, "KD")],
      [C("KD", lambda r: r[1].card("KD").enh, "Wild")]),
    S("Tarots", "The Chariot", "1 card becomes Steel",
      """The Chariot enhances 1 selected card into a Steel card.""",
      blind(consumables=("The Chariot",)),
      [use(0, "KD")],
      [C("KD", lambda r: r[1].card("KD").enh, "Steel")]),
    S("Tarots", "Justice", "1 card becomes Glass",
      """Justice enhances 1 selected card into a Glass card.""",
      blind(consumables=("Justice",)),
      [use(0, "KD")],
      [C("KD", lambda r: r[1].card("KD").enh, "Glass")]),
    S("Tarots", "The Hermit", "doubles money",
      """The Hermit doubles your money, at most +$20. $12 -> $24.""",
      blind(consumables=("The Hermit",), money=12),
      [use(0)],
      [C("money", lambda r: r[1].money, 24)]),
    S("Tarots", "The Hermit", "at most +$20",
      """At $30, The Hermit gives the maximum $20: $50.""",
      blind(consumables=("The Hermit",), money=30),
      [use(0)],
      [C("money", lambda r: r[1].money, 50)]),
    S("Tarots", "The Wheel of Fortune + Oops! All 6s", "an edition on a random joker, 1 in 4",
      """The Wheel of Fortune: 1 in 4 to add Foil, Holographic or Polychrome to a random joker that
      has no edition. Two Oops! All 6s make it 4 in 4: exactly one joker gets an edition, and its
      sell value goes up with it.""",
      blind(jokers=("Joker", "Oops! All 6s", "Oops! All 6s"), consumables=("The Wheel of Fortune",)),
      [use(0)],
      [C("jokers with an edition", lambda r: sum(r[1].jed(i) != "none" for i in range(3)), 1),
       C("the edition", lambda r: next((r[1].jed(i) for i in range(3) if r[1].jed(i) != "none"), "none"),
         OneOf(("Foil", "Holographic", "Polychrome")))]),
    S("Tarots", "Strength", "+1 rank on up to 2 cards",
      """Strength raises the rank of up to 2 selected cards by 1. KD -> AD; AS -> 2S (an Ace goes
      round to 2).""",
      blind("KD KC AS 3S 4C 6D 7H 8S", consumables=("Strength",)),
      [use(0, "KD AS")],
      [C("KD is now", lambda r: r[1].card("KD").name, "AD"), C("AS is now", lambda r: r[1].card("AS").name, "2S")]),
    S("Tarots", "The Hanged Man", "destroys up to 2 cards",
      """The Hanged Man destroys up to 2 selected cards. KD and KC are gone: 50 cards.""",
      blind(consumables=("The Hanged Man",)),
      [use(0, "KD KC")],
      [C("deck size", lambda r: r[1].deck_size, 50), C("KD destroyed", lambda r: r[1].card("KD").live, False),
       C("cards in hand", lambda r: r[1].in_hand, 6)]),
    S("Tarots", "Death", "the left card becomes a copy of the right card",
      """Death: select 2 cards; the left one becomes a copy of the right one (rank, suit,
      enhancement, edition, seal). KD (left) becomes 2H, a Glass card with a Red Seal like it.""",
      blind(consumables=("Death",), cards_mod={"2H": dict(enh=4, seal=2)}),
      [use(0, "KD 2H")],
      [C("KD is now", lambda r: r[1].card("KD").name, "2H"), C("its enhancement", lambda r: r[1].card("KD").enh, "Glass"),
       C("its seal", lambda r: r[1].card("KD").seal, "Red"), C("2H is unchanged", lambda r: r[1].card("2H").name, "2H")]),
    S("Tarots", "Temperance", "the jokers' total sell value",
      """Temperance gives the total sell value of all held jokers (at most $50). Joker $1 + Bull $3
      = +$4.""",
      blind(jokers=("Joker", "Bull"), consumables=("Temperance",)),
      [use(0)],
      [C("money", lambda r: r[1].money, 4)]),
    S("Tarots", "The Devil", "1 card becomes Gold",
      """The Devil enhances 1 selected card into a Gold card.""",
      blind(consumables=("The Devil",)),
      [use(0, "KD")],
      [C("KD", lambda r: r[1].card("KD").enh, "Gold")]),
    S("Tarots", "The Tower", "1 card becomes Stone",
      """The Tower enhances 1 selected card into a Stone card.""",
      blind(consumables=("The Tower",)),
      [use(0, "KD")],
      [C("KD", lambda r: r[1].card("KD").enh, "Stone")]),
    S("Tarots", "The Star", "up to 3 cards become Diamonds",
      """The Star converts up to 3 selected cards to Diamonds.""",
      blind(consumables=("The Star",)),
      [use(0, "KC 2H 3S")],
      [C("suits", lambda r: [SUITS[r[1].card(c).suit] for c in ("KC", "2H", "3S")], ["Diamonds"] * 3)]),
    S("Tarots", "The Moon", "up to 3 cards become Clubs",
      """The Moon converts up to 3 selected cards to Clubs.""",
      blind(consumables=("The Moon",)),
      [use(0, "KD 2H 3S")],
      [C("suits", lambda r: [SUITS[r[1].card(c).suit] for c in ("KD", "2H", "3S")], ["Clubs"] * 3)]),
    S("Tarots", "The Sun", "up to 3 cards become Hearts",
      """The Sun converts up to 3 selected cards to Hearts.""",
      blind(consumables=("The Sun",)),
      [use(0, "KD KC 3S")],
      [C("suits", lambda r: [SUITS[r[1].card(c).suit] for c in ("KD", "KC", "3S")], ["Hearts"] * 3)]),
    S("Tarots", "Judgement", "a random joker",
      """Judgement creates a random Joker card (if there is a free joker slot).""",
      blind(consumables=("Judgement",)),
      [use(0)],
      [C("jokers held", lambda r: len(r[1].jokers), 1)]),
    S("Tarots", "The World", "up to 3 cards become Spades",
      """The World converts up to 3 selected cards to Spades.""",
      blind(consumables=("The World",)),
      [use(0, "KD 2H 4C")],
      [C("suits", lambda r: [SUITS[r[1].card(c).suit] for c in ("KD", "2H", "4C")], ["Spades"] * 3)]),
    S("Tarots", "The Magician", "needs the right number of cards",
      """A Tarot that acts on cards needs a selection: The Magician with no card selected does
      nothing and stays in its slot.""",
      blind(consumables=("The Magician",)),
      [use(0)],
      [C("consumables", lambda r: r[1].cons, ["The Magician"])]),
]

_PLANET_HAND = {"Mercury": "pair", "Venus": "three_kind", "Earth": "full_house", "Mars": "four_kind",
                "Jupiter": "flush", "Saturn": "straight", "Uranus": "two_pair", "Neptune": "straight_flush",
                "Pluto": "high_card", "Planet X": "five_kind", "Ceres": "flush_house", "Eris": "flush_five"}
_HAND_WORD = {"pair": "Pair", "three_kind": "Three of a Kind", "full_house": "Full House",
              "four_kind": "Four of a Kind", "flush": "Flush", "straight": "Straight", "two_pair": "Two Pair",
              "straight_flush": "Straight Flush", "high_card": "High Card", "five_kind": "Five of a Kind",
              "flush_house": "Flush House", "flush_five": "Flush Five"}
# the game's G.GAME.hands at level 2: (chips, mult) = base + one level
_LEVEL2 = {"flush_five": (210, 19), "flush_house": (180, 18), "five_kind": (155, 15),
           "straight_flush": (140, 12), "four_kind": (90, 10), "full_house": (65, 6), "flush": (50, 6),
           "straight": (60, 7), "three_kind": (50, 5), "two_pair": (40, 3), "pair": (25, 3),
           "high_card": (15, 2)}

PLANETS = [
    S("Planets", p, f"levels up {_HAND_WORD[h]}",
      f"""{p} levels up {_HAND_WORD[h]}: level 1 -> 2, {_LEVEL2[h][0]} chips x {_LEVEL2[h][1]} mult.""",
      blind(consumables=(p,)),
      [use(0)],
      [C(f"{_HAND_WORD[h]} level", lambda r, h=h: r[1].level(h), 2),
       C("Planet used (Satellite)", lambda r, h=h: bool(r[1].h.planets_used[HAND_NAMES.index(h)]), True),
       C(f"{_HAND_WORD[h]} at level 2", lambda r, h=h: _lvl2(h), _LEVEL2[h])])
    for p, h in _PLANET_HAND.items()
] + [
    S("Planets", "Mercury", "a Pair at level 2 scores more",
      """Pair level 2 is 25 chips x 3 mult. Use Mercury, then play KD KC: (25 + 20) x 3 = 135.""",
      blind(consumables=("Mercury",)),
      [use(0), play("KD KC")],
      [C("score", lambda r: int(r.play(1).score), 135)]),
    S("Planets", "Pluto", "a level counts for every use",
      """Planets stack: two Plutos take High Card to level 3: 5 + 10 + 10 = 25 chips x 3. Play AS
      alone: (25 + 11) x 3 = 108.""",
      blind("AS 9D 6C 4H 2S 3C 5D 8H", consumables=("Pluto", "Pluto")),
      [use(0), use(0), play("AS")],
      [C("High Card level", lambda r: r[2].level("high_card"), 3), C("score", lambda r: int(r.play(2).score), 108)]),
    S("Planets", "Planet X + Ceres + Eris", "secret Planets only after the hand was played",
      """Planet X, Ceres and Eris (Five of a Kind, Flush House, Flush Five) do not appear in
      Celestial Packs until that hand has been played this run. (2000 packs each way.)""",
      shop(),
      [],
      [C("before: a secret Planet in a pack", lambda r: bool(np.isin(pack_rolls(r.raw[0], PACK["Celestial Pack"]),
         [CONS_ID["Planet X"], CONS_ID["Ceres"], CONS_ID["Eris"]]).any()), False),
       C("after a Flush Five: Eris in a pack", lambda r: bool((pack_rolls(r.raw[0]._replace(
           hand_played=r.raw[0].hand_played.at[0].set(1)), PACK["Celestial Pack"]) == CONS_ID["Eris"]).any()), True)]),
]


def _open(name, n_items, kind_ok, cost, hand=False):
    pid = PACK[name]
    return S("Packs", name, f"${cost}, {n_items} cards, take {int(PACK_CHOOSE[pid])}",
             f"""{name}: costs ${cost}, shows {n_items} cards, take {int(PACK_CHOOSE[pid])}.
             {'An Arcana Pack also draws a hand from the deck to use its Tarots on.' if hand else ''}""",
             shop(pack_slots=(pid, -1)),
             [open_pack(0)],
             [C("money", lambda r: r[1].money, 10 - cost), C("stage", lambda r: r[1].stage, "PACK"),
              C("cards shown", lambda r: int((np.asarray(r[1].h.pack_items) >= 0).sum()), n_items),
              C("cards to take", lambda r: r[1].pack_left, int(PACK_CHOOSE[pid])),
              C("the right kind of card", lambda r: all(kind_ok(r[1].h, i) for i in range(n_items)), True),
              C("cards in hand", lambda r: r[1].in_hand, 8 if hand else 0)])


_tarot = lambda h, i: int(CONS_KIND[int(h.pack_items[i])]) == CK_TAROT
_planet = lambda h, i: int(CONS_KIND[int(h.pack_items[i])]) == CK_PLANET
_card = lambda h, i: int(h.pack_items[i]) == 0 and 0 <= int(h.pack_rank[i]) < 13
_joker = lambda h, i: 0 <= int(h.pack_items[i]) < len(JOKER_NAMES)

PACKS = [
    _open("Arcana Pack", 3, _tarot, 4, hand=True),
    _open("Jumbo Arcana Pack", 5, _tarot, 6, hand=True),
    _open("Mega Arcana Pack", 5, _tarot, 8, hand=True),
    _open("Celestial Pack", 3, _planet, 4),
    _open("Jumbo Celestial Pack", 5, _planet, 6),
    _open("Mega Celestial Pack", 5, _planet, 8),
    _open("Standard Pack", 3, _card, 4),
    _open("Jumbo Standard Pack", 5, _card, 6),
    _open("Mega Standard Pack", 5, _card, 8),
    _open("Buffoon Pack", 2, _joker, 4),
    _open("Jumbo Buffoon Pack", 4, _joker, 6),
    _open("Mega Buffoon Pack", 4, _joker, 8),
    S("Packs", "Arcana Pack", "take a Tarot and use it at once",
      """A Tarot taken from an Arcana Pack is used at once on the selected hand cards. Take The
      Empress with KD KC selected: both become Mult cards, the pack closes and the hand goes back to
      the deck.""",
      lambda: pack_state(PACK["Arcana Pack"], ("The Empress", "The Hermit", "The Star")),
      [pick(0, "KD KC")],
      [C("KD", lambda r: r[1].card("KD").enh, "Mult"), C("KC", lambda r: r[1].card("KC").enh, "Mult"),
       C("stage", lambda r: r[1].stage, "SHOP"), C("cards in hand", lambda r: r[1].in_hand, 0),
       C("consumables (none kept)", lambda r: r[1].cons, [])]),
    S("Packs", "Mega Arcana Pack", "take 2",
      """A Mega pack lets you take 2 cards. The Hermit ($10 -> $20), then Temperance (no jokers:
      +$0); the pack closes after the second.""",
      lambda: pack_state(PACK["Mega Arcana Pack"], ("The Hermit", "Temperance", "The Star", "The Sun", "The Moon"),
                         money=10),
      [pick(0), pick(1)],
      [C("money after The Hermit", lambda r: r[1].money, 20), C("stage after 1", lambda r: r[1].stage, "PACK"),
       C("cards left to take", lambda r: r[1].pack_left, 1), C("stage after 2", lambda r: r[2].stage, "SHOP")]),
    S("Packs", "Celestial Pack", "take a Planet and use it at once",
      """A Planet taken from a Celestial Pack is used at once. Take Mercury: Pair level 2.""",
      lambda: pack_state(PACK["Celestial Pack"], ("Mercury", "Venus", "Earth")),
      [pick(0)],
      [C("Pair level", lambda r: r[1].level("pair"), 2), C("stage", lambda r: r[1].stage, "SHOP"),
       C("consumables (none kept)", lambda r: r[1].cons, [])]),
    S("Packs", "Standard Pack", "take a playing card into the deck",
      """A card taken from a Standard Pack goes into the deck, with its enhancement, edition and
      seal. Take a Glass Ace of Hearts with a Red Seal: 53 cards.""",
      lambda: pack_state(PACK["Standard Pack"], (), cards=(("AH", 4, 0, 2), ("5C", 0, 1, 0), ("9S", 0, 0, 0))),
      [pick(0)],
      [C("deck size", lambda r: r[1].deck_size, 53),
       C("Glass Aces of Hearts with a Red Seal", lambda r: r[1].n_cards(rank=12, suit=2, enh="Glass", seal="Red"), 1),
       C("stage", lambda r: r[1].stage, "SHOP")]),
    S("Packs", "Buffoon Pack", "take a joker",
      """A joker taken from a Buffoon Pack goes into a free joker slot, with its edition. Take a
      Foil Bull.""",
      lambda: pack_state(PACK["Buffoon Pack"], ("Bull", "Joker"), eds=(1, 0)),
      [pick(0)],
      [C("jokers", lambda r: r[1].jokers, ["Bull"]), C("Bull's edition", lambda r: r[1].jed(0), "Foil"),
       C("Bull's sell value ($6 + $2 Foil) / 2", lambda r: r[1].jsell(0), 4), C("stage", lambda r: r[1].stage, "SHOP")]),
    S("Packs", "Buffoon Pack", "no room: the joker cannot be taken",
      """With every joker slot full, a joker cannot be taken from a Buffoon Pack (skip it instead).""",
      lambda: pack_state(PACK["Buffoon Pack"], ("Bull", "Joker"), jokers=("Joker",) * 5),
      [pick(0)],
      [C("jokers held", lambda r: len(r[1].jokers), 5), C("stage", lambda r: r[1].stage, "PACK")]),
    S("Packs", "Arcana Pack", "skip",
      """Skip closes the pack without taking a card; the hand goes back to the deck.""",
      lambda: pack_state(PACK["Arcana Pack"], ("The Empress", "The Hermit", "The Star")),
      [skip()],
      [C("stage", lambda r: r[1].stage, "SHOP"), C("cards in hand", lambda r: r[1].in_hand, 0),
       C("consumables", lambda r: r[1].cons, [])]),
    S("Packs", "Buffoon Pack", "the first shop always has one",
      """The first shop of a run always offers a Buffoon Pack in its first pack slot.""",
      blind(clear=True, buffoon_given=False),
      [play("KD KC")],
      [C("pack slot 1", lambda r: r[1].packs[0], "Buffoon Pack")]),
]

CARDS = [
    S("Cards", "Gold", "$3 per Gold card held at the end of the round",
      """A Gold card held in hand at the end of the round earns $3 (before interest). 2H is Gold
      and held: $3; then cash out $3 + $2 = $5. Money $8.""",
      blind(clear=True, cards_mod={"2H": dict(enh=7)}),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 8), C("cash out", lambda r: r[1].last_cashout, 5)]),
    S("Cards", "Gold", "a played Gold card earns nothing",
      """A Gold card pays only when held at the end of the round, not when played.""",
      blind(cards_mod={"KD": dict(enh=7)}),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 0), C("score", lambda r: int(r.play(0).score), 60)]),
    S("Cards", "Glass + Oops! All 6s", "a Glass card breaks, 1 in 4",
      """A Glass card gives X2 Mult and has a 1 in 4 chance to break after scoring. Two Oops! All
      6s make it certain: 30 x 2 x 2 = 120, then KD is destroyed (51 cards).""",
      blind(jokers=("Oops! All 6s", "Oops! All 6s"), cards_mod={"KD": dict(enh=4)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 120), C("KD destroyed", lambda r: r[1].card("KD").live, False),
       C("deck size", lambda r: r[1].deck_size, 51)]),
    S("Cards", "Lucky + Oops! All 6s", "+20 Mult (1 in 5) and $20 (1 in 15)",
      """A Lucky card has a 1 in 5 chance for +20 Mult and a 1 in 15 chance for $20. Four Oops!
      All 6s make both certain: 30 x (2 + 20) = 660 and +$20.""",
      blind(jokers=("Oops! All 6s",) * 4, cards_mod={"KD": dict(enh=8)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 660), C("money", lambda r: r[1].money, 20)]),
    S("Cards", "Steel", "X1.5 Mult while held",
      """A Steel card gives X1.5 Mult while it stays in hand. 2H is Steel and held:
      30 x 2 x 1.5 = 90.""",
      blind(cards_mod={"2H": dict(enh=5)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 90)]),
    S("Cards", "Stone", "+50 Chips, no rank or suit, always scores",
      """A Stone card gives +50 Chips, has no rank or suit, and always scores. KD KC + a Stone card
      (QS): Pair 30 + 50 = 80 chips x 2 = 160.""",
      blind("KD KC QS 3S 4C 6D 7H 8S", cards_mod={"QS": dict(enh=6)}),
      [play("KD KC QS")],
      [C("chips", lambda r: float(r.play(0).chips), 80.0), C("hand", lambda r: HAND_NAMES[int(r.play(0).handtype)], "pair")]),
    S("Cards", "Polychrome", "X1.5 Mult on a card",
      """A Polychrome card gives X1.5 Mult when scored: 30 x 2 x 1.5 = 90.""",
      blind(cards_mod={"KD": dict(ed=3)}),
      [play("KD KC")],
      [C("score", lambda r: int(r.play(0).score), 90)]),
    S("Cards", "Negative", "a Negative joker adds a joker slot",
      """A Negative joker gives +1 joker slot. With 5 jokers (one Negative) there are 6 slots, so a
      6th joker can be bought.""",
      shop(jokers=("Joker",) * 5, joker_ed=(4,), shop_slots=("Bull",)),
      [buy(0)],
      [C("joker slots", lambda r: r[0].joker_slots, 6), C("jokers held", lambda r: len(r[1].jokers), 6)]),
    S("Cards", "Negative", "Negative jokers add slots up to 16",
      """Each Negative joker adds a joker slot. With 15 jokers (11 of them Negative) there are
      5 + 11 = 16 slots, so a 16th joker can be bought.""",
      shop(jokers=("Joker",) * 15, joker_ed=(4,) * 11, shop_slots=("Bull",)),
      [buy(0)],
      [C("joker slots", lambda r: r[0].joker_slots, 16), C("jokers held", lambda r: len(r[1].jokers), 16)]),
    S("Cards", "Negative", "no slot past 16 (divergence)",
      """The env stops at 16 joker slots. With 16 jokers (11 Negative) in 16 slots, a Negative
      Bull cannot be bought: the money stays $20. (The game gives a 17th slot: a deliberate
      divergence, the env holds at most 16 jokers.)""",
      shop(jokers=("Joker",) * 16, joker_ed=(4,) * 11, shop_slots=("Bull",), shop_ed=(4,), money=20),
      [buy(0)],
      [C("joker slots", lambda r: r[1].joker_slots, 16), C("jokers held", lambda r: len(r[1].jokers), 16),
       C("money", lambda r: r[1].money, 20)]),
    S("Cards", "Foil", "an edition costs more",
      """A joker's edition adds to its price: Foil +$2, Holographic +$3, Polychrome +$5, Negative
      +$5. A Foil Joker costs $2 + $2 = $4 and sells for $2.""",
      shop(shop_slots=("Joker",), shop_ed=(1,)),
      [buy(0)],
      [C("money", lambda r: r[1].money, 6), C("sell value", lambda r: r[1].jsell(0), 2)]),
    S("Cards", "Gold Seal", "$3 when the card scores",
      """A card with a Gold Seal earns $3 when it scores.""",
      blind(cards_mod={"KD": dict(seal=1)}),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 3)]),
    S("Cards", "Red Seal", "the card triggers twice",
      """A card with a Red Seal is retriggered once: KD scores twice, 10 + 10 + 10 + 10 = 40 chips.""",
      blind(cards_mod={"KD": dict(seal=2)}),
      [play("KD KC")],
      [C("chips", lambda r: float(r.play(0).chips), 40.0)]),
    S("Cards", "Blue Seal", "the Planet of the last hand, if held at the end of the round",
      """A card with a Blue Seal held in hand at the end of the round creates the Planet card of
      the last poker hand played. 2H has a Blue Seal; the last hand is a Pair: Mercury.""",
      blind(clear=True, cards_mod={"2H": dict(seal=3)}),
      [play("KD KC")],
      [C("consumables", lambda r: r[1].cons, ["Mercury"])]),
    S("Cards", "Purple Seal", "a Tarot when discarded",
      """A card with a Purple Seal creates a Tarot card when it is discarded.""",
      blind(cards_mod={"2H": dict(seal=4)}),
      [discard("2H")],
      [C("Tarots held", lambda r: r[1].n_cons(CK_TAROT), 1)]),
]

def _changed(r):
    """The state fields that differ between the start and the end (none after a no-op)."""
    a, b = r[0].h, r[-1].h
    return tuple(f for f in a._fields if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))))


def _todo_share(hand_played, old, hand, n=4000):
    """The share of n To Do List picks (env.todo_pick, old = its hand now) that are hand."""
    from env import todo_pick
    keys = jax.random.split(jax.random.key(3), n)
    got = jax.vmap(lambda k: todo_pick(k, jnp.asarray(hand_played, jnp.int32), jnp.int32(old)))(keys)
    return float((np.asarray(got) == HAND_NAMES.index(hand)).mean())


RULES = [
    S("Rules", "Play", "Play needs a card",
      """The game turns the Play Hand button off when no card is selected. In the env a PLAY with
      no card is a no-op: no hand is used, nothing scores, the state stays the same.""",
      blind(jokers=("Joker",)),
      [play("")],
      [C("hands left", lambda r: r[1].hands, 3), C("chips scored", lambda r: r[1].chips_scored, 0),
       C("changed fields", _changed, ())]),
    S("Rules", "Discard", "Discard needs a card",
      """The Discard button is off when no card is selected. A DISCARD with no card is a no-op: no
      discard is used and Green Joker keeps its Mult.""",
      blind(jokers=("Green Joker",), joker_value=(3.0,)),
      [discard("")],
      [C("discards left", lambda r: r[1].discards, 2), C("Green Joker", lambda r: r[1].jv(0), 3.0),
       C("changed fields", _changed, ())]),
    S("Rules", "Discard", "Discard with none left needs a card too",
      """A DISCARD with 0 discards left acts as a PLAY, and a PLAY needs a card: with no card it is
      a no-op.""",
      blind(discards=0),
      [discard("")],
      [C("hands left", lambda r: r[1].hands, 3), C("changed fields", _changed, ())]),
    S("Rules", "Cash out", "blind reward + $1 per hand left + interest",
      """Beating the Small Blind pays $3, +$1 per hand left, +$1 interest per $5 held (at most
      $5). With $12 and 2 hands left: $3 + $2 + $2 = $7, money $19.""",
      blind(clear=True, money=12),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 7), C("money", lambda r: r[1].money, 19),
       C("stage", lambda r: r[1].stage, "SHOP")]),
    S("Rules", "Cash out", "interest stops at $5",
      """Interest is at most $5 (at $25 or more). With $40: $3 + $2 + $5 = $10.""",
      blind(clear=True, money=40),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 10)]),
    S("Rules", "Cash out", "Big Blind $4, Boss Blind $5",
      """The Big Blind pays $4 and the Boss Blind $5. Big Blind: $4 + $2 = $6.""",
      blind(clear=True, blind_index=1),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 6)]),
    S("Rules", "Reroll", "each reroll costs $1 more",
      """A reroll costs $5, then $6, $7 ... in the same shop. Three rerolls: $20 -> $2, the next
      one costs $8.""",
      shop(money=20),
      [reroll(), reroll(), reroll()],
      [C("money", lambda r: r[3].money, 2), C("next reroll price", lambda r: r[3].reroll_cost, 8)]),
    S("Rules", "Reroll", "back to $5 in the next shop",
      """The reroll price goes back to $5 in each new shop.""",
      blind(clear=True, reroll_cost=8),
      [play("KD KC")],
      [C("reroll price", lambda r: r[1].reroll_cost, 5)]),
    S("Rules", "Buy", "needs the money",
      """With $5, Bull ($6) cannot be bought.""",
      shop(money=5, shop_slots=("Bull",)),
      [buy(0)],
      [C("money", lambda r: r[1].money, 5), C("jokers", lambda r: r[1].jokers, [])]),
    S("Rules", "Buy", "needs a free joker slot",
      """With 5 jokers in 5 slots, no joker can be bought.""",
      shop(jokers=("Joker",) * 5, shop_slots=("Bull",)),
      [buy(0)],
      [C("jokers held", lambda r: len(r[1].jokers), 5), C("money", lambda r: r[1].money, 10)]),
    S("Rules", "Buy", "a Tarot or Planet needs a free consumable slot",
      """With both consumable slots full, Mercury cannot be bought.""",
      shop(consumables=("The Fool", "The Hermit"), shop_slots=(CONS_ID["Mercury"],), shop_kind=(IK_PLANET,)),
      [buy(0)],
      [C("consumables", lambda r: r[1].cons, ["The Fool", "The Hermit"]), C("money", lambda r: r[1].money, 10)]),
    S("Rules", "Buy", "a Tarot or Planet costs $3",
      """Tarot and Planet cards cost $3 in the shop and sell for $1.""",
      shop(shop_slots=(CONS_ID["The Hermit"],), shop_kind=(IK_TAROT,)),
      [buy(0), sell_cons(0)],
      [C("money after buying", lambda r: r[1].money, 7), C("money after selling", lambda r: r[2].money, 8)]),
    S("Rules", "Sell", "half the price, rounded down",
      """A joker sells for half its price, rounded down, at least $1. Bull ($6) sells for $3.""",
      shop(jokers=("Bull",), money=0),
      [sell(0)],
      [C("money", lambda r: r[1].money, 3)]),
    S("Rules", "Next blind", "Small, Big, then Boss",
      """Beating the Small Blind puts the Big Blind on deck. NEXT leaves the shop for the blind
      select, NEXT there plays it: 300 x 1.5 = 450 chips to beat, 4 hands, 3 discards.""",
      blind(clear=True, blind_index=0),
      [play("KD KC"), next_blind(), play_blind()],
      [C("on deck after the Small Blind", lambda r: r[1].blind_index, 1),
       C("the blind select", lambda r: r[2].stage, "SELECT"),
       C("blind", lambda r: r[3].blind_index, 1), C("target", lambda r: r[3].blind_target, 450),
       C("hands", lambda r: r[3].hands, 4), C("discards", lambda r: r[3].discards, 3)]),
    S("Rules", "Next blind", "after the Boss Blind, the next ante",
      """Beating the Boss Blind ends the ante (in the cash out): ante 2, the Small Blind on deck,
      a new boss for ante 2. Its target: 800 chips.""",
      blind(clear=True, blind_index=2, ante=1),
      [play("KD KC"), next_blind(), play_blind()],
      [C("ante", lambda r: r[1].ante, 2), C("blind", lambda r: r[1].blind_index, 0),
       C("a boss for ante 2", lambda r: r[1].boss_fits, True),
       C("target", lambda r: r[3].blind_target, 800)]),
    S("Rules", "Hands", "out of hands: the run is lost",
      """With 1 hand left, a hand that does not beat the blind loses the run.""",
      blind(hands=1),
      [play("KD KC")],
      [C("stage", lambda r: r[1].stage, "LOST")]),
    S("Rules", "Hand levels", "the level table",
      """Each hand type has base chips and mult, and each level adds a fixed amount (the game's
      G.GAME.hands). Level 2 of every hand:
      Flush Five 210 x 19, Flush House 180 x 18, Five of a Kind 155 x 15, Straight Flush 140 x 12,
      Four of a Kind 90 x 10, Full House 65 x 6, Flush 50 x 6, Straight 60 x 7, Three of a Kind
      50 x 5, Two Pair 40 x 3, Pair 25 x 3, High Card 15 x 2.""",
      blind(),
      [],
      [C(f"{_HAND_WORD[h]} at level 2", lambda r, h=h: _lvl2(h), _LEVEL2[h]) for h in HAND_NAMES]),
    S("Rules", "Hand levels", "Five of a Kind, Flush House and Flush Five need copied cards",
      """The three secret hands need duplicate or Wild cards. KD KC KH KS + a second KD (QD made a
      King): Five of a Kind, 120 + 50 = 170 chips x 12.""",
      blind("KD KC KH KS QD 2C 3C 4C", cards_mod={"QD": dict(rank=11)}),
      [play("KD KC KH KS QD")],
      [C("hand", lambda r: HAND_NAMES[int(r.play(0).handtype)], "five_kind"),
       C("score", lambda r: int(r.play(0).score), 2040)]),
]

# ---------------------------------------------------------------------------
# THE BLIND SELECT, BOSSES, TAGS, VOUCHERS, SPECTRAL CARDS
# ---------------------------------------------------------------------------
def select(**kw):
    """The blind select: no hand (the deck is together), $10. NEXT plays the blind on deck."""
    base = dict(hand="", hand_size=8, deck_ptr=0, game_stage=STAGE_SELECT, money=10)
    base.update(kw)
    return base


def boss(name, hand=H8, **kw):
    """The Boss Blind name in progress, at its first ante (the Small Blind's 3 hands and
    2 discards left, as blind() has)."""
    kw.setdefault("ante", R.BOSSES[name][0])
    return blind(hand, boss=name, blind_index=2, **kw)


def boss_select(name, **kw):
    """The blind select with the Boss Blind name on deck, at its first ante."""
    kw.setdefault("ante", R.BOSSES[name][0])
    return select(boss=name, blind_index=2, **kw)


def _levels(**kw):
    return [kw.get(h, 1) for h in HAND_NAMES]


def _used(*names):
    """bosses_used with each named boss used once."""
    return [int(n in names) for n in BOSS_NAMES]


RARITY = {1: "Common", 2: "Uncommon", 3: "Rare", 4: "Legendary"}
FACES = "JQK"
_start = lambda r: r[0].hand


def _score(k=0):
    return lambda r: int(r.play(k).score)


def _new_in_hand(r, k=1):
    """The cards in hand at state k that were not in hand at the start."""
    old = list(r[0].hand)
    out = []
    for c in r[k].hand:
        if c in old:
            old.remove(c)
        else:
            out.append(c)
    return out


def _enhanced_in_hand(v, ranks):
    """Cards in hand with one of ranks (a string like 'JQK') and an enhancement, not Stone."""
    return sum(1 for l in range(MAX_HAND_SIZE) if v.h.hand_valid[l]
               and v.card(int(v.h.hand[l])).name[0] in ranks
               and v.card(int(v.h.hand[l])).enh not in ("none", "Stone"))


BOSSES = [
    S("Bosses", "The Hook", "2 random held cards are discarded after each hand",
      """The Hook: after each hand is played, 2 random cards held in hand are discarded (no discard
      is used). Play KD KC with 6 cards held: 4 of them stay, and the hand fills up to 8 again.""",
      boss("The Hook"),
      [play("KD KC")],
      [C("held cards that stay (of 2H 3S 4C 6D 7H 8S)",
         lambda r: len(set(r[1].hand) & {"2H", "3S", "4C", "6D", "7H", "8S"}), 4),
       C("cards in hand", lambda r: r[1].in_hand, 8), C("discards left", lambda r: r[1].discards, 2),
       C("score: the Pair as normal", _score(), 60)]),
    S("Bosses", "The Ox", "the most played hand sets money to $0",
      """The Ox: playing the run's most played poker hand (as it was when the ante began) sets
      money to $0; the hand still scores. The Pair is the most played; play a Pair with $20.""",
      boss("The Ox", ox_hand="pair", money=20),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 0), C("score", _score(), 60)]),
    S("Bosses", "The Ox", "another hand keeps the money",
      """The Ox only takes the money for its hand. Flush is the most played hand; a Pair keeps $20.""",
      boss("The Ox", ox_hand="flush", money=20),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 20)]),
    S("Bosses", "The Ox", "a tie: the last hand in the game's table order",
      """When a Boss Blind is beaten, the game sets The Ox's hand to the most played hand of the run.
      Its loop (state_events.lua end_round) never updates _order, so on a tie the hand it reads
      last in pairs() order wins, not the better hand. That order (LuaJIT 2.0.5, a new run):
      Flush House, Full House, Flush, Pair, High Card, Straight Flush, Straight, Two Pair, Flush
      Five, Five of a Kind, Three of a Kind, Four of a Kind. Beat a boss with a Pair: Pair 2,
      High Card 2. High Card comes later, so it is The Ox's hand.""",
      boss("The Hook", clear=True, hand_played=_counts(pair=1, high_card=2)),
      [play("KD KC")],
      [C("The Ox's hand", lambda r: HAND_NAMES[int(r[1].h.ox_hand)], "high_card")]),
    S("Bosses", "The Ox", "a four-way tie",
      """Flush, Straight, Full House and Pair all played 2 times after the Pair: in pairs() order
      (Full House, Flush, Pair, ..., Straight) Straight comes last, so it is The Ox's hand.""",
      boss("The Hook", clear=True, hand_played=_counts(flush=2, straight=2, full_house=2, pair=1)),
      [play("KD KC")],
      [C("The Ox's hand", lambda r: HAND_NAMES[int(r[1].h.ox_hand)], "straight")]),
    S("Bosses", "The House", "the first hand is drawn face down",
      """The House: every card of the first hand drawn this round is face down. A card face down
      plays as normal; only the player cannot see it. After a hand, the new cards are face up:
      play 2 cards, the 6 held stay face down, the 2 new ones are face up.""",
      boss_select("The House"),
      [play_blind(), play((0, 1))],
      [C("face down after the draw", lambda r: len(r[1].down), 8),
       C("face down after a hand", lambda r: len(r[2].down), 6)]),
    S("Bosses", "The Wall", "an extra large blind: x4",
      """The Wall: the blind needs 4 x the ante's base chips (other bosses 2 x). Ante 2: 800 x 4
      = 3200 chips.""",
      boss_select("The Wall"),
      [play_blind()],
      [C("target", lambda r: r[1].blind_target, 3200)]),
    S("Bosses", "The Wheel + Oops! All 6s", "1 in 7 cards are drawn face down",
      """The Wheel: each card drawn has a 1 in 7 chance to be face down. Three Oops! All 6s make
      it 8 in 7: every card is face down.""",
      boss_select("The Wheel", jokers=("Oops! All 6s",) * 3),
      [play_blind()],
      [C("face down", lambda r: len(r[1].down), 8)]),
    S("Bosses", "The Arm", "the hand played goes down a level",
      """The Arm: the poker hand played loses 1 level (never below 1) before it scores. Pair at
      level 2 drops to level 1: 10 + 10 + 10 = 30 x 2 = 60 (at level 2 it would be 45 x 3).""",
      boss("The Arm", hand_level=_levels(pair=2)),
      [play("KD KC")],
      [C("Pair level", lambda r: r[1].level("pair"), 1), C("score", _score(), 60)]),
    S("Bosses", "The Arm", "a level 1 hand stays at level 1",
      """The Arm never takes a hand below level 1: the Pair at level 1 scores 60 and stays at 1.""",
      boss("The Arm"),
      [play("KD KC")],
      [C("Pair level", lambda r: r[1].level("pair"), 1), C("score", _score(), 60)]),
    S("Bosses", "The Club", "Club cards are debuffed",
      """The Club: every Club card is debuffed: it gives no chips and none of its effects (it still
      counts for the poker hand). KD KC is a Pair, but KC adds nothing: 10 + 10 = 20 x 2 = 40.""",
      boss("The Club"),
      [play("KD KC")],
      [C("KC debuffed", lambda r: r[0].debuffed("KC"), True), C("KD debuffed", lambda r: r[0].debuffed("KD"), False),
       C("score", _score(), 40)]),
    S("Bosses", "The Fish", "the cards drawn after a hand are face down",
      """The Fish: the cards drawn after each hand played are face down. Play KD KC: the 2 new
      cards are face down; the 6 held stay face up.""",
      boss("The Fish"),
      [play("KD KC")],
      [C("face down", lambda r: len(r[1].down), 2),
       C("the held cards stay face up", lambda r: set(r[1].down) & {"2H", "3S", "4C", "6D", "7H", "8S"}, set())]),
    S("Bosses", "The Fish", "a discard draws face up",
      """The Fish only turns the cards drawn after a hand played. After a discard they are face up.""",
      boss("The Fish"),
      [discard("2H 3S")],
      [C("face down", lambda r: len(r[1].down), 0)]),
    S("Bosses", "The Psychic", "a hand of fewer than 5 cards scores nothing",
      """The Psychic: every hand must have 5 cards. KD KC (2 cards) is debuffed: it scores 0, and
      the hand is used.""",
      boss("The Psychic"),
      [play("KD KC")],
      [C("score", _score(), 0), C("chips scored", lambda r: r[1].chips_scored, 0), C("hands left", lambda r: r[1].hands, 2)]),
    S("Bosses", "The Psychic", "5 cards score",
      """The Psychic lets a 5-card hand score: KD KC 2H 3S 4C is a Pair, 60.""",
      boss("The Psychic"),
      [play("KD KC 2H 3S 4C")],
      [C("score", _score(), 60)]),
    S("Bosses", "The Goad", "Spade cards are debuffed",
      """The Goad: every Spade card is debuffed. KS KD: 10 + 10 (KD) = 20 x 2 = 40.""",
      boss("The Goad", hand="KS KD 2H 3C 4C 6D 7H 8D"),
      [play("KS KD")],
      [C("score", _score(), 40)]),
    S("Bosses", "The Water", "no discards",
      """The Water: the blind starts with 0 discards.""",
      boss_select("The Water"),
      [play_blind()],
      [C("discards", lambda r: r[1].discards, 0), C("hands", lambda r: r[1].hands, 4)]),
    S("Bosses", "The Window", "Diamond cards are debuffed",
      """The Window: every Diamond card is debuffed. KD KC: 10 + 10 (KC) = 20 x 2 = 40.""",
      boss("The Window"),
      [play("KD KC")],
      [C("score", _score(), 40)]),
    S("Bosses", "The Manacle", "-1 hand size",
      """The Manacle: -1 hand size for this blind: 7 cards are drawn.""",
      boss_select("The Manacle"),
      [play_blind()],
      [C("hand size", lambda r: r[1].hand_size, 7), C("cards in hand", lambda r: r[1].in_hand, 7)]),
    S("Bosses", "The Eye", "a hand type played this round scores nothing",
      """The Eye: no poker hand type may be played twice in a round. A Pair was played this round,
      so KD KC is debuffed: 0.""",
      boss("The Eye", hand_round=_counts(pair=1)),
      [play("KD KC")],
      [C("score", _score(), 0)]),
    S("Bosses", "The Eye", "a new hand type scores",
      """The Eye: a poker hand type not played this round scores as normal: the first Pair, 60.""",
      boss("The Eye"),
      [play("KD KC")],
      [C("score", _score(), 60)]),
    S("Bosses", "The Mouth", "only the first hand type of the round scores",
      """The Mouth: only one poker hand type may be played this round, the first one. High Card
      was played first, so the Pair KD KC is debuffed: 0.""",
      boss("The Mouth", boss_hand=HAND_NAMES.index("high_card")),
      [play("KD KC")],
      [C("score", _score(), 0)]),
    S("Bosses", "The Mouth", "the first hand sets the type",
      """The Mouth: the first hand of the round sets the type. Play a Pair first (60), then High
      Card 8S: debuffed, 0.""",
      boss("The Mouth"),
      [play("KD KC"), play("8S")],
      [C("first hand", _score(0), 60), C("second hand", _score(1), 0)]),
    S("Bosses", "The Plant", "face cards are debuffed",
      """The Plant: every face card (J, Q, K) is debuffed. KD KC: 10 x 2 = 20.""",
      boss("The Plant"),
      [play("KD KC")],
      [C("score", _score(), 20)]),
    S("Bosses", "The Serpent", "always draw 3 cards",
      """The Serpent: after each hand or discard, exactly 3 cards are drawn, not up to the hand
      size. Discard 2H: 7 held + 3 = 10 cards.""",
      boss("The Serpent"),
      [discard("2H")],
      [C("cards in hand", lambda r: r[1].in_hand, 10)]),
    S("Bosses", "The Pillar", "cards played earlier this ante are debuffed",
      """The Pillar: a card played in an earlier blind of this ante is debuffed. KD was played in
      the Small Blind: KD KC = 10 + 10 = 20 x 2 = 40.""",
      boss("The Pillar", played_ante=("KD",)),
      [play("KD KC")],
      [C("score", _score(), 40)]),
    S("Bosses", "The Needle", "only 1 hand, x1 blind",
      """The Needle: 1 hand only, and the blind is small: 1 x the base. Ante 2: 800 chips.""",
      boss_select("The Needle"),
      [play_blind()],
      [C("hands", lambda r: r[1].hands, 1), C("target", lambda r: r[1].blind_target, 800)]),
    S("Bosses", "The Head", "Heart cards are debuffed",
      """The Head: every Heart card is debuffed. KH KD: 10 + 10 (KD) = 20 x 2 = 40.""",
      boss("The Head", hand="KH KD 2C 3S 4C 6D 7C 8S"),
      [play("KH KD")],
      [C("score", _score(), 40)]),
    S("Bosses", "The Tooth", "$1 per card played",
      """The Tooth: lose $1 for each card played. KD KC with $10: $8.""",
      boss("The Tooth", money=10),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 8)]),
    S("Bosses", "The Flint", "the base chips and mult are halved",
      """The Flint: the poker hand's base chips and mult are halved, rounded (mult at least 1).
      Pair 10 x 2 -> 5 x 1; + 10 + 10 = 25 x 1 = 25.""",
      boss("The Flint"),
      [play("KD KC")],
      [C("chips", lambda r: float(r.play(0).chips), 25.0), C("mult", lambda r: float(r.play(0).mult), 1.0),
       C("score", _score(), 25)]),
    S("Bosses", "The Mark", "face cards are drawn face down",
      """The Mark: every face card is drawn face down; the other cards face up.""",
      boss_select("The Mark", key=3),
      [play_blind()],
      [C("face cards in hand", lambda r: sum(c[0] in FACES for c in r[1].hand) > 0, True),
       C("face down = the face cards", lambda r: sorted(r[1].down), lambda r: sorted(c for c in r[1].hand if c[0] in FACES))]),
    S("Bosses", "Amber Acorn", "the jokers are flipped and shuffled",
      """Amber Acorn: at the start of the blind every joker is turned face down and the jokers are
      shuffled.""",
      boss_select("Amber Acorn", jokers=("Joker", "Bull", "Banner")),
      [play_blind()],
      [C("face down", lambda r: [r[1].jdown(i) for i in range(3)], [True] * 3),
       C("the same jokers", lambda r: sorted(r[1].jokers), sorted(["Joker", "Bull", "Banner"]))]),
    S("Bosses", "Verdant Leaf", "every card is debuffed",
      """Verdant Leaf: every playing card is debuffed until a joker is sold. KD KC: 10 x 2 = 20.""",
      boss("Verdant Leaf"),
      [play("KD KC")],
      [C("score", _score(), 20)]),
    S("Bosses", "Verdant Leaf", "sell a joker to end it",
      """Verdant Leaf ends when a joker is sold: sell Bull, then KD KC scores 60.""",
      boss("Verdant Leaf", jokers=("Bull",)),
      [sell(0), play("KD KC")],
      [C("boss disabled", lambda r: r[1].boss_disabled, True), C("score", _score(1), 60)]),
    S("Bosses", "Violet Vessel", "a very large blind: x6",
      """Violet Vessel: the blind needs 6 x the ante's base. Ante 8: 50000 x 6 = 300000 chips.""",
      boss_select("Violet Vessel"),
      [play_blind()],
      [C("target", lambda r: r[1].blind_target, 300000)]),
    S("Bosses", "Crimson Heart", "one random joker is disabled each hand",
      """Crimson Heart: after each draw for a new hand, one random joker is disabled, and it is a
      different one than before (with 2 or more jokers).""",
      boss_select("Crimson Heart", jokers=("Joker", "Bull")),
      [play_blind(), play((0,))],
      [C("disabled after the first draw", lambda r: r[1].jdeb(0) + r[1].jdeb(1), 1),
       C("the other one after a hand", lambda r: [r[2].jdeb(0), r[2].jdeb(1)],
         lambda r: [not r[1].jdeb(0), not r[1].jdeb(1)])]),
    S("Bosses", "Crimson Heart", "the disabled joker does nothing",
      """A joker Crimson Heart disabled gives nothing: the Joker's +4 Mult is gone. KD KC = 30 x 2
      = 60. With 1 joker it is disabled again after each hand.""",
      boss("Crimson Heart", jokers=("Joker",), joker_debuff=(1,)),
      [play("KD KC")],
      [C("score", _score(), 60), C("disabled again", lambda r: r[1].jdeb(0), True)]),
    S("Bosses", "Cerulean Bell", "one card is always selected",
      """Cerulean Bell: one card in hand is forced into every hand played. 2H is forced: play KD
      KC and 2H is played too (a kicker; the Pair scores 60). A new forced card is picked after the
      draw.""",
      boss("Cerulean Bell", forced_card="2H"),
      [play("KD KC")],
      [C("2H was played", lambda r: "2H" in r[1].hand, False), C("score", _score(), 60),
       C("a new forced card in hand", lambda r: r[1].forced in r[1].hand, True)]),
    S("Bosses", "Chicot", "the boss has no ability",
      """Chicot disables every Boss Blind at its start. The Wall at ante 2 then needs 800 x 2 =
      1600 chips, not 3200.""",
      boss_select("The Wall", jokers=("Chicot",)),
      [play_blind()],
      [C("boss disabled", lambda r: r[1].boss_disabled, True), C("target", lambda r: r[1].blind_target, 1600)]),
    S("Bosses", "Luchador", "sell it to disable the boss",
      """Selling Luchador during a Boss Blind disables the boss. Then KC is not debuffed by The
      Club: KD KC = 60.""",
      boss("The Club", jokers=("Luchador",)),
      [sell(0), play("KD KC")],
      [C("boss disabled", lambda r: r[1].boss_disabled, True), C("score", _score(1), 60)]),
    S("Bosses", "Matador", "$8 when the hand triggers the boss",
      """Matador earns $8 when the played hand triggers the Boss Blind's ability. The Club
      debuffs KC in KD KC: +$8.""",
      boss("The Club", jokers=("Matador",)),
      [play("KD KC")],
      [C("money", lambda r: r[1].money, 8)]),
    S("Bosses", "Matador", "no trigger, no money",
      """Matador pays only if the boss acts on the hand: KD KH has no Club, so The Club does not
      trigger. $0.""",
      boss("The Club", hand="KD KH 2H 3S 4C 6D 7H 8S", jokers=("Matador",)),
      [play("KD KH")],
      [C("money", lambda r: r[1].money, 0)]),
]


def _shop_arrays(s, n=8000, seed=1):
    """n fresh shops of s: (ids, kinds, eds, enh) per unlocked slot."""
    keys = jax.random.split(jax.random.key(seed), n)
    roll = _shop_rolls(keys, s)
    k = int(s.shop_size)
    return [np.asarray(x)[:, :k] for x in (roll.ids, roll.kinds, roll.eds, roll.enh)]


def kind_share(s, kind):
    """The share of shop cards of kind (IK_*) in fresh shops of s."""
    _, kinds, _, _ = _shop_arrays(s)
    return float((kinds == kind).mean())


def edition_share(s):
    """The share of shop jokers with an edition."""
    _, kinds, eds, _ = _shop_arrays(s)
    j = kinds == IK_JOKER
    return float((eds[j] > 0).mean())


def card_shares(s):
    """(enhanced, with an edition) shares of the playing cards in fresh shops of s."""
    _, kinds, eds, enh = _shop_arrays(s)
    c = kinds == IK_CARD
    return round(float((enh[c] > 0).mean()), 3), round(float((eds[c] > 0).mean()), 3)


def pack_share(s, pid, test, n=4000):
    """The share of the cards in n packs pid (in state s) for which test(cons id) holds."""
    items = pack_rolls(s, pid, n)
    k = int(PACK_SIZE[pid])
    items = items[:, :k].ravel()
    return float(np.mean([test(int(x)) for x in items]))


def _skip_tag(tag, **kw):
    """The blind select with tag on the Small Blind (the Big Blind's is a D6 Tag)."""
    kw.setdefault("ante", 2)
    return select(blind_tags=(tag, "D6 Tag"), **kw)


def _free_shop_joker(tag, ed):
    return S("Tags", tag, f"the next shop's first joker turns {ed} and is free",
             f"""{tag}: the next joker in the shop without an edition becomes {ed}, and it is free. An
             Uncommon Tag makes the first shop card a joker.""",
             blind(clear=True, tags=("Uncommon Tag", tag)),
             [play("KD KC")],
             [C("shop slot 1's edition", lambda r: r[1].shop_ed(0), ed), C("its price", lambda r: r[1].shop_price(0), 0),
              C("tags left", lambda r: r[1].tags, [])])


TAGS = [
    S("Tags", "Uncommon Tag", "a free Uncommon joker in the next shop",
      """Uncommon Tag: the next shop's first card is an Uncommon joker, and it is free.""",
      blind(clear=True, tags=("Uncommon Tag",)),
      [play("KD KC")],
      [C("shop slot 1", lambda r: r[1].shop_rarity(0), "Uncommon"), C("its price", lambda r: r[1].shop_price(0), 0),
       C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Rare Tag", "a free Rare joker in the next shop",
      """Rare Tag: the next shop's first card is a Rare joker, and it is free.""",
      blind(clear=True, tags=("Rare Tag",)),
      [play("KD KC")],
      [C("shop slot 1", lambda r: r[1].shop_rarity(0), "Rare"), C("its price", lambda r: r[1].shop_price(0), 0)]),
    _free_shop_joker("Negative Tag", "Negative"),
    _free_shop_joker("Foil Tag", "Foil"),
    _free_shop_joker("Holographic Tag", "Holographic"),
    _free_shop_joker("Polychrome Tag", "Polychrome"),
    S("Tags", "Investment Tag", "$25 after the Boss Blind",
      """Investment Tag: after the Boss Blind is beaten, +$25 in the cash out. $5 (the boss) + $2
      (hands left) + $25 = $32.""",
      blind(clear=True, blind_index=2, tags=("Investment Tag",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 32), C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Investment Tag", "kept through a Small Blind",
      """The Investment Tag waits for a Boss Blind: the Small Blind pays $3 + $2 = $5, and the tag
      stays.""",
      blind(clear=True, tags=("Investment Tag",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 5), C("tags left", lambda r: r[1].tags, ["Investment Tag"])]),
    S("Tags", "Voucher Tag", "one more voucher in the next shop",
      """Voucher Tag: the next shop has one more voucher than the ante's voucher: 2.""",
      blind(clear=True, tags=("Voucher Tag",), ante_voucher="Grabber"),
      [play("KD KC")],
      [C("vouchers in the shop", lambda r: len(r[1].shop_vouchers), 2),
       C("the ante's voucher first", lambda r: r[1].shop_vouchers[0], "Grabber"),
       C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Boss Tag", "a new boss",
      """Boss Tag: the Boss Blind of this ante is rerolled at once (free). The Wall was used; the
      new boss is another one that fits ante 2.""",
      _skip_tag("Boss Tag", boss="The Wall", bosses_used=_used("The Wall")),
      [skip_blind()],
      [C("new boss", lambda r: r[1].boss != "The Wall", True), C("it fits ante 2", lambda r: r[1].boss_fits, True),
       C("money", lambda r: r[1].money, 10), C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Standard Tag", "a free Mega Standard Pack",
      """Standard Tag: a free Mega Standard Pack opens at once (take 2 of 5 playing cards).""",
      _skip_tag("Standard Tag"),
      [skip_blind()],
      [C("stage", lambda r: r[1].stage, "PACK"), C("pack", lambda r: r[1].pack_name, "Mega Standard Pack"),
       C("money", lambda r: r[1].money, 10), C("cards to take", lambda r: r[1].pack_left, 2)]),
    S("Tags", "Charm Tag", "a free Mega Arcana Pack",
      """Charm Tag: a free Mega Arcana Pack opens at once, with a hand dealt to use its Tarots on.""",
      _skip_tag("Charm Tag"),
      [skip_blind()],
      [C("pack", lambda r: r[1].pack_name, "Mega Arcana Pack"), C("money", lambda r: r[1].money, 10),
       C("cards in hand", lambda r: r[1].in_hand, 8)]),
    S("Tags", "Meteor Tag", "a free Mega Celestial Pack",
      """Meteor Tag: a free Mega Celestial Pack opens at once.""",
      _skip_tag("Meteor Tag"),
      [skip_blind()],
      [C("pack", lambda r: r[1].pack_name, "Mega Celestial Pack"), C("money", lambda r: r[1].money, 10)]),
    S("Tags", "Buffoon Tag", "a free Mega Buffoon Pack",
      """Buffoon Tag: a free Mega Buffoon Pack opens at once.""",
      _skip_tag("Buffoon Tag"),
      [skip_blind()],
      [C("pack", lambda r: r[1].pack_name, "Mega Buffoon Pack"), C("money", lambda r: r[1].money, 10)]),
    S("Tags", "Ethereal Tag", "a free Spectral Pack",
      """Ethereal Tag: a free Spectral Pack opens at once, with a hand dealt.""",
      _skip_tag("Ethereal Tag"),
      [skip_blind()],
      [C("pack", lambda r: r[1].pack_name, "Spectral Pack"), C("money", lambda r: r[1].money, 10),
       C("cards in hand", lambda r: r[1].in_hand, 8)]),
    S("Tags", "Charm Tag", "back to the blind select after the pack",
      """A pack from a tag closes back to the blind select, with the Big Blind on deck.""",
      _skip_tag("Charm Tag"),
      [skip_blind(), skip()],
      [C("stage", lambda r: r[2].stage, "SELECT"), C("on deck", lambda r: r[2].blind_index, 1),
       C("cards in hand", lambda r: r[2].in_hand, 0)]),
    S("Tags", "Handy Tag", "$1 per hand played this run",
      """Handy Tag: +$1 for each hand played this run. 7 hands played: $10 + $7 = $17.""",
      _skip_tag("Handy Tag", hand_played=_counts(pair=4, high_card=3)),
      [skip_blind()],
      [C("money", lambda r: r[1].money, 17), C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Garbage Tag", "$1 per unused discard this run",
      """Garbage Tag: +$1 for each discard not used this run. 5 unused: $10 + $5 = $15.""",
      _skip_tag("Garbage Tag", unused_discards=5),
      [skip_blind()],
      [C("money", lambda r: r[1].money, 15)]),
    S("Tags", "Coupon Tag", "the next shop's cards and packs are free",
      """Coupon Tag: the cards and packs in the next shop are free (a reroll's cards are not).""",
      blind(clear=True, tags=("Coupon Tag",)),
      [play("KD KC")],
      [C("card prices", lambda r: [r[1].shop_price(i) for i in range(2)], [0, 0]),
       C("pack prices", lambda r: [r[1].pack_price(i) for i in range(2)], [0, 0]),
       C("tags left", lambda r: r[1].tags, [])]),
    S("Tags", "Double Tag", "a copy of the next tag",
      """Double Tag: the next tag taken comes twice (a Double Tag is not copied). Skip for a
      Coupon Tag with a Double Tag held: two Coupon Tags.""",
      _skip_tag("Coupon Tag", tags=("Double Tag",)),
      [skip_blind()],
      [C("tags", lambda r: r[1].tags, ["Coupon Tag", "Coupon Tag"])]),
    S("Tags", "Juggle Tag", "+3 hand size next round",
      """Juggle Tag: +3 hand size for the next round only. Skip the Small Blind, play the Big
      Blind: 11 cards.""",
      _skip_tag("Juggle Tag"),
      [skip_blind(), play_blind()],
      [C("hand size", lambda r: r[2].hand_size, 11), C("cards in hand", lambda r: r[2].in_hand, 11),
       C("tags left", lambda r: r[2].tags, [])]),
    S("Tags", "D6 Tag", "rerolls in the next shop start at $0",
      """D6 Tag: in the next shop the rerolls start at $0 ($0, $1, $2 ...).""",
      blind(clear=True, tags=("D6 Tag",), money=10),
      [play("KD KC"), reroll()],
      [C("reroll price", lambda r: r[1].reroll_cost, 0), C("money after a reroll", lambda r: r[2].money, lambda r: r[1].money),
       C("the next reroll", lambda r: r[2].reroll_cost, 1)]),
    S("Tags", "Top-up Tag", "2 Common jokers",
      """Top-up Tag: up to 2 Common jokers, if there is room.""",
      _skip_tag("Top-up Tag"),
      [skip_blind()],
      [C("jokers", lambda r: len(r[1].jokers), 2),
       C("rarities", lambda r: [r[1].jrarity(i) for i in range(2)], ["Common", "Common"])]),
    S("Tags", "Top-up Tag", "only as many as there is room for",
      """Top-up Tag with 4 jokers in 5 slots: 1 Common joker.""",
      _skip_tag("Top-up Tag", jokers=("Joker",) * 4),
      [skip_blind()],
      [C("jokers", lambda r: len(r[1].jokers), 5)]),
    S("Tags", "Skip Tag", "$5 per blind skipped",
      """Skip Tag: +$5 for each blind skipped this run, this one too. 2 skipped before: 3 x $5 =
      $15.""",
      _skip_tag("Skip Tag", skips=2),
      [skip_blind()],
      [C("skips", lambda r: r[1].skips, 3), C("money", lambda r: r[1].money, 25)]),
    S("Tags", "Orbital Tag", "+3 levels to a poker hand",
      """Orbital Tag: +3 levels to its poker hand (Flush here): level 4.""",
      _skip_tag("Orbital Tag", orbital_hands=("flush", "pair")),
      [skip_blind()],
      [C("Flush level", lambda r: r[1].level("flush"), 4)]),
    S("Tags", "Economy Tag", "doubles money, at most +$40",
      """Economy Tag: money is doubled, by at most +$40. $30 -> $60.""",
      _skip_tag("Economy Tag", money=30),
      [skip_blind()],
      [C("money", lambda r: r[1].money, 60)]),
    S("Tags", "Economy Tag", "+$40 at most",
      """Economy Tag with $50: +$40, $90.""",
      _skip_tag("Economy Tag", money=50),
      [skip_blind()],
      [C("money", lambda r: r[1].money, 90)]),
]


def _redeem_shop(name, **kw):
    kw.setdefault("money", 10)
    return shop(shop_vouchers=(name,), **kw)


VOUCHERS = [
    S("Vouchers", "Overstock", "+1 card slot in the shop",
      """Overstock: +1 card slot in the shop; the new slot gets a card at once. $10.""",
      _redeem_shop("Overstock", shop_slots=("Joker", "Bull")),
      [redeem(0)],
      [C("money", lambda r: r[1].money, 0), C("shop size", lambda r: r[1].shop_size, 3),
       C("cards in the shop", lambda r: len(r[1].shop), 3), C("vouchers", lambda r: r[1].vouchers, ["Overstock"])]),
    S("Vouchers", "Overstock Plus", "+1 more card slot",
      """Overstock Plus (needs Overstock): +1 more card slot: 4.""",
      _redeem_shop("Overstock Plus", vouchers=("Overstock",), shop_size=3, shop_slots=("Joker", "Bull", "Banner")),
      [redeem(0)],
      [C("shop size", lambda r: r[1].shop_size, 4), C("cards in the shop", lambda r: len(r[1].shop), 4)]),
    S("Vouchers", "Clearance Sale", "everything 25% off",
      """Clearance Sale: every card, pack and voucher in the shop costs 25% less (price x 0.75
      after +$0.50, rounded down). Bull $6 -> $4.""",
      _redeem_shop("Clearance Sale", shop_slots=("Bull",), money=20),
      [redeem(0), buy(0)],
      [C("Bull's price", lambda r: r[1].shop_price(0), 4), C("money", lambda r: r[2].money, 6),
       C("a voucher's price", lambda r: r[1].voucher_price, 7)]),
    S("Vouchers", "Liquidation", "everything 50% off",
      """Liquidation (needs Clearance Sale): 50% off. Its own price with Clearance Sale: $7. Bull
      $6 -> $3.""",
      _redeem_shop("Liquidation", vouchers=("Clearance Sale",), shop_slots=("Bull",), money=20),
      [redeem(0)],
      [C("money", lambda r: r[1].money, 13), C("Bull's price", lambda r: r[1].shop_price(0), 3)]),
    S("Vouchers", "Clearance Sale", "sell values follow the new prices",
      """With Clearance Sale, a held joker's sell value is half its new price: Bull $4 -> $2.""",
      _redeem_shop("Clearance Sale", jokers=("Bull",)),
      [redeem(0)],
      [C("Bull sells for", lambda r: r[1].jsell(0), 2)]),
    S("Vouchers", "Hone", "editions 2X more often",
      """Hone: Foil, Holographic and Polychrome jokers appear 2 x as often in the shop: about 8% of
      the shop's jokers have an edition (4% without).""",
      lambda: state(vouchers=("Hone",)),
      [],
      [C("jokers with an edition", lambda r: edition_share(r.raw[0]), Between(0.065, 0.095))]),
    S("Vouchers", "Glow Up", "editions 4X more often",
      """Glow Up (needs Hone): editions 4 x as often: about 16% of the shop's jokers.""",
      lambda: state(vouchers=("Hone", "Glow Up")),
      [],
      [C("jokers with an edition", lambda r: edition_share(r.raw[0]), Between(0.14, 0.18))]),
    S("Vouchers", "Reroll Surplus", "rerolls cost $2 less",
      """Reroll Surplus: rerolls cost $2 less, at once and in every shop: $5 -> $3.""",
      _redeem_shop("Reroll Surplus"),
      [redeem(0)],
      [C("reroll price", lambda r: r[1].reroll_cost, 3)]),
    S("Vouchers", "Reroll Surplus", "the next shop's rerolls start at $3",
      """With Reroll Surplus each new shop's rerolls start at $3.""",
      blind(clear=True, vouchers=("Reroll Surplus",)),
      [play("KD KC")],
      [C("reroll price", lambda r: r[1].reroll_cost, 3)]),
    S("Vouchers", "Reroll Glut", "rerolls cost $2 less again",
      """Reroll Glut (needs Reroll Surplus): $2 less again: $3 -> $1.""",
      _redeem_shop("Reroll Glut", vouchers=("Reroll Surplus",), reroll_cost=3),
      [redeem(0)],
      [C("reroll price", lambda r: r[1].reroll_cost, 1)]),
    S("Vouchers", "Crystal Ball", "+1 consumable slot",
      """Crystal Ball: +1 consumable slot: 3.""",
      _redeem_shop("Crystal Ball"),
      [redeem(0)],
      [C("consumable slots", lambda r: r[1].consumable_slots, 3)]),
    S("Vouchers", "Omen Globe", "Spectral cards in Arcana Packs",
      """Omen Globe (needs Crystal Ball): each card of an Arcana Pack has a 20% chance to be a
      Spectral card.""",
      lambda: state(vouchers=("Crystal Ball", "Omen Globe")),
      [],
      [C("Spectral cards in Arcana Packs", lambda r: pack_share(r.raw[0], PACK["Arcana Pack"],
                                                                lambda x: int(CONS_KIND[x]) == CK_SPECTRAL),
         Between(0.17, 0.23))]),
    S("Vouchers", "Telescope", "the most played hand's Planet in every Celestial Pack",
      """Telescope: every Celestial Pack has the Planet card of the most played poker hand (Flush:
      Jupiter) as its first card.""",
      shop(vouchers=("Telescope",), pack_slots=(PACK["Celestial Pack"], -1), hand_played=_counts(flush=3, pair=1)),
      [open_pack(0)],
      [C("the first card", lambda r: _pack_item(r[1].h, 0), "Jupiter")]),
    S("Vouchers", "Observatory", "a held Planet gives X1.5 Mult for its hand",
      """Observatory (needs Telescope): each Planet card held gives X1.5 Mult when its poker hand
      is played. Mercury held, play a Pair: 30 x 2 x 1.5 = 90.""",
      blind(vouchers=("Telescope", "Observatory"), consumables=("Mercury",)),
      [play("KD KC")],
      [C("score", _score(), 90)]),
    S("Vouchers", "Observatory", "another hand's Planet does nothing",
      """Observatory: Venus (Three of a Kind) does nothing for a Pair: 60.""",
      blind(vouchers=("Telescope", "Observatory"), consumables=("Venus",)),
      [play("KD KC")],
      [C("score", _score(), 60)]),
    S("Vouchers", "Grabber", "+1 hand every round",
      """Grabber: +1 hand in every round: 5.""",
      _redeem_shop("Grabber"),
      [redeem(0), next_blind(), play_blind()],
      [C("hands", lambda r: r[3].hands, 5)]),
    S("Vouchers", "Nacho Tong", "+1 more hand",
      """Nacho Tong (needs Grabber): +1 more hand: 6.""",
      _redeem_shop("Nacho Tong", vouchers=("Grabber",), handcount=5),
      [redeem(0), next_blind(), play_blind()],
      [C("hands", lambda r: r[3].hands, 6)]),
    S("Vouchers", "Wasteful", "+1 discard every round",
      """Wasteful: +1 discard in every round: 4.""",
      _redeem_shop("Wasteful"),
      [redeem(0), next_blind(), play_blind()],
      [C("discards", lambda r: r[3].discards, 4)]),
    S("Vouchers", "Recyclomancy", "+1 more discard",
      """Recyclomancy (needs Wasteful): +1 more discard: 5.""",
      _redeem_shop("Recyclomancy", vouchers=("Wasteful",), discardcount=4),
      [redeem(0), next_blind(), play_blind()],
      [C("discards", lambda r: r[3].discards, 5)]),
    S("Vouchers", "Tarot Merchant", "Tarot cards 2.4X more often in the shop",
      """Tarot Merchant: the Tarot rate goes from 4 to 9.6 (jokers 20, Planets 4): 9.6 / 33.6 =
      29% of the shop's cards.""",
      lambda: state(vouchers=("Tarot Merchant",)),
      [],
      [C("Tarot share", lambda r: kind_share(r.raw[0], IK_TAROT), Between(0.265, 0.305))]),
    S("Vouchers", "Tarot Tycoon", "Tarot cards 8X more often",
      """Tarot Tycoon (needs Tarot Merchant): the Tarot rate is 32: 32 / 56 = 57%.""",
      lambda: state(vouchers=("Tarot Merchant", "Tarot Tycoon")),
      [],
      [C("Tarot share", lambda r: kind_share(r.raw[0], IK_TAROT), Between(0.55, 0.595))]),
    S("Vouchers", "Planet Merchant", "Planet cards 2.4X more often",
      """Planet Merchant: the Planet rate goes from 4 to 9.6: 29% of the shop's cards.""",
      lambda: state(vouchers=("Planet Merchant",)),
      [],
      [C("Planet share", lambda r: kind_share(r.raw[0], IK_PLANET), Between(0.265, 0.305))]),
    S("Vouchers", "Planet Tycoon", "Planet cards 8X more often",
      """Planet Tycoon (needs Planet Merchant): the Planet rate is 32: 57%.""",
      lambda: state(vouchers=("Planet Merchant", "Planet Tycoon")),
      [],
      [C("Planet share", lambda r: kind_share(r.raw[0], IK_PLANET), Between(0.55, 0.595))]),
    S("Vouchers", "Seed Money", "interest up to $10",
      """Seed Money: interest stops at $10 (at $50), not $5. $60: $3 + $2 + $10 = $15.""",
      blind(clear=True, money=60, vouchers=("Seed Money",)),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 15)]),
    S("Vouchers", "Money Tree", "interest up to $20",
      """Money Tree (needs Seed Money): interest stops at $20 (at $100). $120: $3 + $2 + $20 = $25.""",
      blind(clear=True, money=120, vouchers=("Seed Money", "Money Tree")),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 25)]),
    S("Vouchers", "Blank", "does nothing",
      """Blank does nothing (it opens Antimatter).""",
      _redeem_shop("Blank"),
      [redeem(0)],
      [C("vouchers", lambda r: r[1].vouchers, ["Blank"]), C("money", lambda r: r[1].money, 0),
       C("joker slots", lambda r: r[1].joker_slots, 5)]),
    S("Vouchers", "Antimatter", "+1 joker slot",
      """Antimatter (needs Blank): +1 joker slot: 6.""",
      _redeem_shop("Antimatter", vouchers=("Blank",)),
      [redeem(0)],
      [C("joker slots", lambda r: r[1].joker_slots, 6)]),
    S("Vouchers", "Magic Trick", "playing cards in the shop",
      """Magic Trick: playing cards can come up in the shop (rate 4 of 32): 12.5%.""",
      lambda: state(vouchers=("Magic Trick",)),
      [],
      [C("playing card share", lambda r: kind_share(r.raw[0], IK_CARD), Between(0.11, 0.14))]),
    S("Vouchers", "Illusion", "the shop's playing cards may be enhanced or have an edition",
      """Illusion (needs Magic Trick): a playing card in the shop is enhanced 40% of the time and
      has an edition 20% of the time.""",
      lambda: state(vouchers=("Magic Trick", "Illusion")),
      [],
      [C("enhanced", lambda r: card_shares(r.raw[0])[0], Between(0.36, 0.44)),
       C("with an edition", lambda r: card_shares(r.raw[0])[1], Between(0.16, 0.24))]),
    S("Vouchers", "Magic Trick", "buy a playing card",
      """A playing card bought in the shop goes into the deck: $1 (+ its edition).""",
      shop(shop_slots=(0,), shop_kind=(IK_CARD,), shop_rank=(12,), shop_suit=(2,), vouchers=("Magic Trick",)),
      [buy(0)],
      [C("deck size", lambda r: r[1].deck_size, 53), C("money", lambda r: r[1].money, 9),
       C("Aces of Hearts", lambda r: r[1].n_cards(rank=12, suit=2), 2)]),
    S("Vouchers", "Hieroglyph", "-1 ante, -1 hand every round",
      """Hieroglyph: the ante goes back by 1, and every round has 1 hand less: 3.""",
      _redeem_shop("Hieroglyph", ante=2),
      [redeem(0), next_blind(), play_blind()],
      [C("ante", lambda r: r[1].ante, 1), C("hands", lambda r: r[3].hands, 3),
       C("target (ante 1)", lambda r: r[3].blind_target, 300)]),
    S("Vouchers", "Petroglyph", "-1 ante, -1 discard every round",
      """Petroglyph (needs Hieroglyph): the ante goes back by 1 again, and every round has 1
      discard less: 2.""",
      _redeem_shop("Petroglyph", ante=3, vouchers=("Hieroglyph",), handcount=3),
      [redeem(0), next_blind(), play_blind()],
      [C("ante", lambda r: r[1].ante, 2), C("discards", lambda r: r[3].discards, 2)]),
    S("Vouchers", "Director's Cut", "reroll the boss once per ante, $10",
      """Director's Cut: the Boss Blind can be rerolled once per ante for $10. The second reroll
      does nothing.""",
      select(vouchers=("Director's Cut",), money=25, boss="The Wall", ante=2, bosses_used=_used("The Wall")),
      [reroll_boss(), reroll_boss()],
      [C("money after 1", lambda r: r[1].money, 15), C("a new boss", lambda r: r[1].boss != "The Wall", True),
       C("it fits ante 2", lambda r: r[1].boss_fits, True), C("money after 2", lambda r: r[2].money, 15),
       C("the same boss after 2", lambda r: r[2].boss, lambda r: r[1].boss)]),
    S("Vouchers", "Retcon", "reroll the boss any number of times",
      """Retcon (needs Director's Cut): the boss can be rerolled again and again, $10 each.""",
      select(vouchers=("Director's Cut", "Retcon"), money=25, boss="The Wall", ante=2, bosses_used=_used("The Wall")),
      [reroll_boss(), reroll_boss()],
      [C("money", lambda r: r[2].money, 5), C("it fits ante 2", lambda r: r[2].boss_fits, True)]),
    S("Vouchers", "Paint Brush", "+1 hand size",
      """Paint Brush: +1 hand size: 9 cards.""",
      _redeem_shop("Paint Brush"),
      [redeem(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 9), C("cards in hand", lambda r: r[3].in_hand, 9)]),
    S("Vouchers", "Palette", "+1 more hand size",
      """Palette (needs Paint Brush): +1 more hand size: 10.""",
      _redeem_shop("Palette", vouchers=("Paint Brush",), hand_size=9),
      [redeem(0), next_blind(), play_blind()],
      [C("hand size", lambda r: r[3].hand_size, 10)]),
    S("Vouchers", "Overstock Plus", "needs Overstock",
      """The second voucher of a pair only comes after the first: with no voucher redeemed, a new
      ante's voucher is never a second one.""",
      blind(clear=True, blind_index=2, ante_voucher=-1),
      [play("KD KC")],
      [C("a first voucher", lambda r: VOUCHER_NAMES.index(r[1].shop_vouchers[0]) % 2 == 0, True)]),
]


def _seal(name, seal):
    return S("Spectrals", name, f"a {seal} Seal on 1 card",
             f"""{name}: adds a {seal} Seal to 1 selected card.""",
             blind(consumables=(name,)),
             [use(0, "KD")],
             [C("KD's seal", lambda r: r[1].card("KD").seal, seal), C("consumables", lambda r: r[1].cons, [])])


SPECTRALS = [
    S("Spectrals", "Familiar", "destroy 1 card, add 3 enhanced face cards",
      """Familiar: destroys 1 random card in hand and adds 3 random face cards (J, Q, K), each with a
      random enhancement (not Stone), to the hand. 52 - 1 + 3 = 54 cards.""",
      blind(consumables=("Familiar",)),
      [use(0)],
      [C("deck size", lambda r: r[1].deck_size, 54), C("cards in hand", lambda r: r[1].in_hand, 10),
       C("enhanced face cards in hand", lambda r: _enhanced_in_hand(r[1], "JQK"), 3),
       C("cards of the old hand left", lambda r: 10 - len(_new_in_hand(r)), 7)]),
    S("Spectrals", "Grim", "destroy 1 card, add 2 enhanced Aces",
      """Grim: destroys 1 random card in hand, adds 2 Aces with random enhancements. 53 cards.""",
      blind(consumables=("Grim",)),
      [use(0)],
      [C("deck size", lambda r: r[1].deck_size, 53), C("cards in hand", lambda r: r[1].in_hand, 9),
       C("enhanced Aces in hand", lambda r: _enhanced_in_hand(r[1], "A"), 2)]),
    S("Spectrals", "Incantation", "destroy 1 card, add 4 enhanced number cards",
      """Incantation: destroys 1 random card in hand, adds 4 number cards (2-10) with random
      enhancements. 55 cards.""",
      blind(consumables=("Incantation",)),
      [use(0)],
      [C("deck size", lambda r: r[1].deck_size, 55), C("cards in hand", lambda r: r[1].in_hand, 11),
       C("enhanced number cards in hand", lambda r: _enhanced_in_hand(r[1], "23456789T"), 4)]),
    _seal("Talisman", "Gold"),
    S("Spectrals", "Aura", "Foil, Holographic or Polychrome on 1 card",
      """Aura: gives 1 selected card a random edition: Foil, Holographic or Polychrome.""",
      blind(consumables=("Aura",)),
      [use(0, "KD")],
      [C("KD's edition", lambda r: r[1].card("KD").ed, OneOf(("Foil", "Holographic", "Polychrome")))]),
    S("Spectrals", "Wraith", "a Rare joker, money to $0",
      """Wraith: creates a random Rare joker and sets money to $0.""",
      blind(consumables=("Wraith",), money=15),
      [use(0)],
      [C("jokers", lambda r: len(r[1].jokers), 1), C("rarity", lambda r: r[1].jrarity(0), "Rare"),
       C("money", lambda r: r[1].money, 0)]),
    S("Spectrals", "Wraith", "needs a free joker slot",
      """Wraith cannot be used with every joker slot full.""",
      blind(consumables=("Wraith",), money=15, jokers=("Joker",) * 5),
      [use(0)],
      [C("money", lambda r: r[1].money, 15), C("consumables", lambda r: r[1].cons, ["Wraith"])]),
    S("Spectrals", "Sigil", "every card in hand to one random suit",
      """Sigil: every card in hand becomes the same random suit.""",
      blind(consumables=("Sigil",)),
      [use(0)],
      [C("suits in hand", lambda r: len({c[1] for c in r[1].hand}), 1), C("cards in hand", lambda r: r[1].in_hand, 8)]),
    S("Spectrals", "Ouija", "every card in hand to one random rank, -1 hand size",
      """Ouija: every card in hand becomes the same random rank; -1 hand size for the run.""",
      blind(consumables=("Ouija",)),
      [use(0)],
      [C("ranks in hand", lambda r: len({c[0] for c in r[1].hand}), 1), C("hand size", lambda r: r[1].hand_size, 7)]),
    S("Spectrals", "Ectoplasm", "a random joker turns Negative, -1 hand size",
      """Ectoplasm: a random joker without an edition becomes Negative (+1 joker slot); -1 hand
      size. Each Ectoplasm after takes 1 more hand size than the one before.""",
      blind(consumables=("Ectoplasm",), jokers=("Joker",)),
      [use(0)],
      [C("Joker's edition", lambda r: r[1].jed(0), "Negative"), C("hand size", lambda r: r[1].hand_size, 7),
       C("joker slots", lambda r: r[1].joker_slots, 6), C("the next one takes", lambda r: r[1].ecto_minus, 2)]),
    S("Spectrals", "Ectoplasm", "the second one takes 2 hand size",
      """The second Ectoplasm of a run takes 2 hand size: 8 -> 6.""",
      blind(consumables=("Ectoplasm",), jokers=("Joker",), ecto_minus=2),
      [use(0)],
      [C("hand size", lambda r: r[1].hand_size, 6)]),
    S("Spectrals", "Immolate", "destroy 5 random cards in hand, +$20",
      """Immolate: destroys 5 random cards in hand and gives $20.""",
      blind(consumables=("Immolate",)),
      [use(0)],
      [C("money", lambda r: r[1].money, 20), C("cards in hand", lambda r: r[1].in_hand, 3),
       C("deck size", lambda r: r[1].deck_size, 47)]),
    S("Spectrals", "Ankh", "copy a random joker, destroy the others",
      """Ankh: copies a random joker and destroys every other joker: 2 of the same joker.""",
      blind(consumables=("Ankh",), jokers=("Joker", "Bull", "Banner")),
      [use(0)],
      [C("jokers", lambda r: len(r[1].jokers), 2), C("the same joker", lambda r: len(set(r[1].jokers)), 1)]),
    _seal("Deja Vu", "Red"),
    S("Spectrals", "Hex", "a random joker turns Polychrome, the others are destroyed",
      """Hex: a random joker without an edition becomes Polychrome; every other joker is destroyed.""",
      blind(consumables=("Hex",), jokers=("Joker", "Bull")),
      [use(0)],
      [C("jokers", lambda r: len(r[1].jokers), 1), C("edition", lambda r: r[1].jed(0), "Polychrome")]),
    _seal("Trance", "Blue"),
    _seal("Medium", "Purple"),
    S("Spectrals", "Cryptid", "2 copies of 1 card",
      """Cryptid: 2 exact copies of 1 selected card go into the hand. 54 cards, 3 King of Diamonds.""",
      blind(consumables=("Cryptid",), cards_mod={"KD": dict(enh=2, seal=1)}),
      [use(0, "KD")],
      [C("deck size", lambda r: r[1].deck_size, 54), C("cards in hand", lambda r: r[1].in_hand, 10),
       C("Mult Kings of Diamonds with a Gold Seal", lambda r: r[1].n_cards(rank=11, suit=1, enh="Mult", seal="Gold"), 3)]),
    S("Spectrals", "The Soul", "a Legendary joker",
      """The Soul: creates a random Legendary joker (Canio, Triboulet, Yorick, Chicot, Perkeo).""",
      blind(consumables=("The Soul",)),
      [use(0)],
      [C("rarity", lambda r: r[1].jrarity(0), "Legendary")]),
    S("Spectrals", "Black Hole", "+1 level to every poker hand",
      """Black Hole: every poker hand goes up 1 level.""",
      blind(consumables=("Black Hole",)),
      [use(0)],
      [C("levels", lambda r: [r[1].level(h) for h in HAND_NAMES], [2] * len(HAND_NAMES))]),
    S("Spectrals", "The Soul", "1 in 333 Tarot cards of an Arcana Pack",
      """Each card of an Arcana Pack has a 0.3% chance to be The Soul.""",
      lambda: state(),
      [],
      [C("The Soul share", lambda r: pack_share(r.raw[0], PACK["Arcana Pack"], lambda x: x == CONS_ID["The Soul"]),
         Between(0.0015, 0.0045))]),
    S("Spectrals", "Black Hole", "1 in 333 Planet cards of a Celestial Pack",
      """Each card of a Celestial Pack has a 0.3% chance to be Black Hole.""",
      lambda: state(),
      [],
      [C("Black Hole share", lambda r: pack_share(r.raw[0], PACK["Celestial Pack"], lambda x: x == CONS_ID["Black Hole"]),
         Between(0.0015, 0.0045))]),
    S("Spectrals", "Familiar", "needs a card in hand",
      """Familiar, Grim and Incantation need cards in hand: in the shop they cannot be used.""",
      shop(consumables=("Familiar",)),
      [use(0)],
      [C("consumables", lambda r: r[1].cons, ["Familiar"]), C("deck size", lambda r: r[1].deck_size, 52)]),
]

_spectral = lambda h, i: int(CONS_KIND[int(h.pack_items[i])]) == CK_SPECTRAL

SPECTRAL_PACKS = [
    _open("Spectral Pack", 2, _spectral, 4, hand=True),
    _open("Jumbo Spectral Pack", 4, _spectral, 6, hand=True),
    _open("Mega Spectral Pack", 4, _spectral, 8, hand=True),
    S("Packs", "Spectral Pack", "take a Spectral card and use it at once",
      """A Spectral card taken from a Spectral Pack is used at once on the selected hand cards. Take
      Talisman with KD selected: KD gets a Gold Seal.""",
      lambda: pack_state(PACK["Spectral Pack"], ("Talisman", "Grim")),
      [pick(0, "KD")],
      [C("KD's seal", lambda r: r[1].card("KD").seal, "Gold"), C("stage", lambda r: r[1].stage, "SHOP"),
       C("consumables (none kept)", lambda r: r[1].cons, [])]),
]

NEW_JOKERS = [
    S("Jokers", "Ceremonial Dagger", "destroys the joker to its right, +2 x its sell value",
      """Ceremonial Dagger: when a blind is selected, it destroys the joker to its right and gains
      Mult = 2 x that joker's sell value. Bull sells for $3: +6 Mult.""",
      select(jokers=("Ceremonial Dagger", "Bull")),
      [play_blind()],
      [C("jokers", lambda r: r[1].jokers, ["Ceremonial Dagger"]), C("Mult", lambda r: r[1].jv(0), 6.0)]),
    S("Jokers", "Ceremonial Dagger", "nothing to its right",
      """Ceremonial Dagger in the last slot destroys nothing and gains nothing.""",
      select(jokers=("Bull", "Ceremonial Dagger"), joker_value=(0.0, 4.0)),
      [play_blind()],
      [C("jokers", lambda r: r[1].jokers, ["Bull", "Ceremonial Dagger"]), C("Mult", lambda r: r[1].jv(1), 4.0)]),
    S("Jokers", "Sixth Sense", "a single 6 as the first hand: destroyed, a Spectral card",
      """Sixth Sense: if the first hand of the round is a single 6, the 6 is destroyed and a
      Spectral card is created (if there is room).""",
      blind("6D KC 2H 3S 4C 7D 7H 8S", jokers=("Sixth Sense",)),
      [play("6D")],
      [C("6D destroyed", lambda r: r[1].card("6D").live, False), C("deck size", lambda r: r[1].deck_size, 51),
       C("Spectral cards held", lambda r: r[1].n_cons(CK_SPECTRAL), 1)]),
    S("Jokers", "Sixth Sense", "not after the first hand",
      """Sixth Sense acts only on the first hand of the round.""",
      blind("6D KC 2H 3S 4C 7D 7H 8S", jokers=("Sixth Sense",), hand_round=_counts(pair=1)),
      [play("6D")],
      [C("6D destroyed", lambda r: r[1].card("6D").live, True), C("Spectral cards held", lambda r: r[1].n_cons(CK_SPECTRAL), 0)]),
    S("Jokers", "Madness", "X0.5 more and destroys a random joker",
      """Madness: when a Small or Big Blind is selected, it gains X0.5 Mult and destroys a random
      other joker. X1 -> X1.5.""",
      select(jokers=("Madness", "Bull")),
      [play_blind()],
      [C("jokers", lambda r: r[1].jokers, ["Madness"]), C("X Mult", lambda r: r[1].jv(0), 1.5)]),
    S("Jokers", "Madness", "not at a Boss Blind",
      """Madness does nothing when a Boss Blind is selected.""",
      select(jokers=("Madness", "Bull"), blind_index=2),
      [play_blind()],
      [C("jokers", lambda r: r[1].jokers, ["Madness", "Bull"]), C("X Mult", lambda r: r[1].jv(0), 1.0)]),
    S("Jokers", "Seance", "a Straight Flush makes a Spectral card",
      """Seance: if the poker hand is a Straight Flush, it creates a random Spectral card.""",
      blind("5S 6S 7S 8S 9S 2H 3D JC", jokers=("Seance",)),
      [play("5S 6S 7S 8S 9S")],
      [C("Spectral cards held", lambda r: r[1].n_cons(CK_SPECTRAL), 1)]),
    S("Jokers", "Riff-raff", "2 Common jokers when a blind is selected",
      """Riff-raff: when a blind is selected, it creates 2 Common jokers (if there is room).""",
      select(jokers=("Riff-raff",)),
      [play_blind()],
      [C("jokers", lambda r: len(r[1].jokers), 3),
       C("the new ones", lambda r: [r[1].jrarity(i) for i in (1, 2)], ["Common", "Common"])]),
    S("Jokers", "Riff-raff", "only into free slots",
      """Riff-raff with 4 jokers in 5 slots: 1 Common joker.""",
      select(jokers=("Riff-raff", "Joker", "Joker", "Joker")),
      [play_blind()],
      [C("jokers", lambda r: len(r[1].jokers), 5)]),
    S("Jokers", "Diet Cola", "sell it for a free Double Tag",
      """Diet Cola: sell it to create a free Double Tag.""",
      shop(jokers=("Diet Cola",)),
      [sell(0)],
      [C("tags", lambda r: r[1].tags, ["Double Tag"]), C("money", lambda r: r[1].money, 13)]),
    S("Jokers", "Mr. Bones", "saves the run at 25% of the blind",
      """Mr. Bones: if the last hand ends with at least 25% of the blind's chips, the run goes on
      and Mr. Bones is destroyed. The blind pays no $3. 20 + 60 = 80 of 300 (>= 75).""",
      blind(hands=1, blind_target=300, chips_scored=20, jokers=("Mr. Bones",)),
      [play("KD KC")],
      [C("stage", lambda r: r[1].stage, "SHOP"), C("jokers", lambda r: r[1].jokers, []),
       C("cash out", lambda r: r[1].last_cashout, 0)]),
    S("Jokers", "Mr. Bones", "below 25%: the run is lost",
      """Mr. Bones does not save a hand below 25% of the blind: 60 of 300 (< 75). Lost.""",
      blind(hands=1, blind_target=300, jokers=("Mr. Bones",)),
      [play("KD KC")],
      [C("stage", lambda r: r[1].stage, "LOST")]),
    S("Jokers", "Throwback", "X0.25 Mult per blind skipped",
      """Throwback: X0.25 Mult for each blind skipped this run. 2 skipped: X1.5. 30 x 2 x 1.5 = 90.""",
      blind(jokers=("Throwback",), skips=2),
      [play("KD KC")],
      [C("score", _score(), 90)]),
    S("Jokers", "Throwback", "a skip counts at once",
      """Skipping a blind adds to Throwback at once: 1 skip, X1.25.""",
      select(jokers=("Throwback",), blind_tags=("D6 Tag", "D6 Tag")),
      [skip_blind()],
      [C("skips", lambda r: r[1].skips, 1)]),
    S("Jokers", "Invisible Joker", "after 2 rounds, sell it to copy a joker",
      """Invisible Joker: after 2 rounds held, selling it duplicates a random other joker.""",
      shop(jokers=("Invisible Joker", "Bull"), joker_value=(2.0, 0.0)),
      [sell(0)],
      [C("jokers", lambda r: r[1].jokers, ["Bull", "Bull"])]),
    S("Jokers", "Invisible Joker", "too early: no copy",
      """Sold after 1 round, Invisible Joker copies nothing.""",
      shop(jokers=("Invisible Joker", "Bull"), joker_value=(1.0, 0.0)),
      [sell(0)],
      [C("jokers", lambda r: r[1].jokers, ["Bull"])]),
    S("Jokers", "Invisible Joker", "counts the rounds",
      """Invisible Joker counts each round it is held.""",
      blind(clear=True, jokers=("Invisible Joker",), joker_value=(1.0,)),
      [play("KD KC")],
      [C("rounds", lambda r: r[1].jv(0), 2.0)]),
    S("Jokers", "Canio", "+X1 Mult for each face card destroyed",
      """Canio: gains X1 Mult each time a face card is destroyed. The Hanged Man destroys KD KC:
      X1 -> X3.""",
      blind(jokers=("Canio",), consumables=("The Hanged Man",)),
      [use(0, "KD KC")],
      [C("X Mult", lambda r: r[1].jv(0), 3.0)]),
    S("Jokers", "Yorick", "+X1 Mult every 23 cards discarded",
      """Yorick: gains X1 Mult for every 23 cards discarded. 20 counted, discard 3: X2.""",
      blind(jokers=("Yorick",), joker_state=(20,)),
      [discard("2H 3S 4C")],
      [C("X Mult", lambda r: r[1].jv(0), 2.0)]),
    S("Jokers", "Chicot", "a boss with no ability",
      """Chicot: every Boss Blind's ability is disabled when the blind starts. The Club debuffs no
      Club card.""",
      boss_select("The Club", jokers=("Chicot",)),
      [play_blind()],
      [C("boss disabled", lambda r: r[1].boss_disabled, True),
       C("debuffed cards in hand", lambda r: [c for c in r[1].hand if r[1].debuffed(c)], [])]),
    S("Jokers", "Perkeo", "a Negative copy of a consumable when leaving the shop",
      """Perkeo: leaving the shop, it creates a Negative copy of 1 random consumable held. A
      Negative consumable takes no slot.""",
      shop(jokers=("Perkeo",), consumables=("The Fool",)),
      [next_blind()],
      [C("consumables", lambda r: r[1].cons, ["The Fool", "The Fool"]), C("the copy", lambda r: r[1].ced(1), "Negative")]),
]

SELECT_RULES = [
    S("Rules", "Skip", "skip the Small Blind for its tag",
      """At the blind select, the Small or Big Blind can be skipped for its tag. Skip the Small
      Blind: the Big Blind is on deck, 1 skip, a Coupon Tag held.""",
      select(blind_tags=("Coupon Tag", "D6 Tag")),
      [skip_blind()],
      [C("on deck", lambda r: r[1].blind_index, 1), C("skips", lambda r: r[1].skips, 1),
       C("tags", lambda r: r[1].tags, ["Coupon Tag"]), C("stage", lambda r: r[1].stage, "SELECT")]),
    S("Rules", "Skip", "the Boss Blind cannot be skipped",
      """The Boss Blind cannot be skipped.""",
      select(blind_index=2),
      [skip_blind()],
      [C("on deck", lambda r: r[1].blind_index, 2), C("skips", lambda r: r[1].skips, 0)]),
    S("Rules", "Boss reroll", "needs Director's Cut or Retcon",
      """Without Director's Cut or Retcon the boss cannot be rerolled.""",
      select(money=20, boss="The Wall", ante=2),
      [reroll_boss()],
      [C("money", lambda r: r[1].money, 20), C("boss", lambda r: r[1].boss, "The Wall")]),
    S("Rules", "Win", "beat the ante 8 Boss Blind",
      """Beating the Boss Blind of ante 8 wins the run.""",
      blind(clear=True, blind_index=2, ante=8),
      [play("KD KC")],
      [C("stage", lambda r: r[1].stage, "WON")]),
    S("Rules", "Next blind", "ante 8 has a showdown boss",
      """The boss of ante 8 is a showdown boss: Amber Acorn, Verdant Leaf, Violet Vessel, Crimson
      Heart or Cerulean Bell.""",
      blind(clear=True, blind_index=2, ante=7),
      [play("KD KC")],
      [C("ante", lambda r: r[1].ante, 8),
       C("boss", lambda r: r[1].boss, OneOf(("Amber Acorn", "Verdant Leaf", "Violet Vessel", "Crimson Heart",
                                              "Cerulean Bell")))]),
    S("Rules", "Next blind", "the least used boss",
      """A new boss is one of the bosses used least this run. Every boss but The Fish was used
      once: the boss of ante 2 is The Fish.""",
      blind(clear=True, blind_index=2, ante=1, bosses_used=[int(n != "The Fish") for n in BOSS_NAMES]),
      [play("KD KC")],
      [C("boss", lambda r: r[1].boss, "The Fish")]),
    S("Rules", "Next blind", "the boss stays for the whole ante",
      """The boss is chosen once per ante: beating the Big Blind keeps it.""",
      blind(clear=True, blind_index=1, boss="The Club"),
      [play("KD KC")],
      [C("boss", lambda r: r[1].boss, "The Club")]),
    S("Rules", "Vouchers", "the ante's voucher in each shop until it is redeemed",
      """Each shop of an ante offers the ante's voucher until it is redeemed.""",
      blind(clear=True, ante_voucher="Grabber"),
      [play("KD KC")],
      [C("vouchers in the shop", lambda r: r[1].shop_vouchers, ["Grabber"])]),
    S("Rules", "Vouchers", "redeemed: no voucher for the rest of the ante",
      """Once the ante's voucher is redeemed, the next shops of that ante have no voucher (a
      Voucher Tag still adds one).""",
      blind(clear=True, ante_voucher=-1),
      [play("KD KC")],
      [C("vouchers in the shop", lambda r: r[1].shop_vouchers, [])]),
    S("Rules", "Vouchers", "a new voucher with each ante",
      """Beating the Boss Blind brings a new voucher for the next ante (one not redeemed yet).""",
      blind(clear=True, blind_index=2, ante_voucher=-1, vouchers=("Grabber",)),
      [play("KD KC")],
      [C("vouchers in the shop", lambda r: len(r[1].shop_vouchers), 1),
       C("not one redeemed", lambda r: r[1].shop_vouchers[0] != "Grabber", True)]),
    S("Rules", "Hieroglyph", "ante 0 needs 100 chips",
      """Below ante 1 (Hieroglyph at ante 1), a Small Blind needs 100 chips.""",
      _redeem_shop("Hieroglyph", ante=1),
      [redeem(0), next_blind(), play_blind()],
      [C("ante", lambda r: r[3].ante, 0), C("target", lambda r: r[3].blind_target, 100)]),
    S("Rules", "The Pillar", "a new ante forgets the cards played",
      """The cards played in an ante are forgotten when the ante ends.""",
      blind(clear=True, blind_index=2, played_ante=("QS", "2C")),
      [play("KD KC")],
      [C("cards marked as played this ante", lambda r: r[1].played_ante, 0)]),
    S("Rules", "Cash out", "Mr. Bones: no blind money",
      """When Mr. Bones saves the run, the blind pays $0, but the hands left and interest still pay.
      $25 held: $0 + $0 hands + $5 interest.""",
      blind(hands=1, blind_target=300, chips_scored=20, jokers=("Mr. Bones",), money=25),
      [play("KD KC")],
      [C("cash out", lambda r: r[1].last_cashout, 5)]),
]


SCENARIOS = (JOKERS + NEW_JOKERS + TAROTS + PLANETS + SPECTRALS + PACKS + SPECTRAL_PACKS + CARDS + BOSSES + TAGS
             + VOUCHERS + RULES + SELECT_RULES)


# ---------------------------------------------------------------------------
# THE SCORING SPEC ROWS (tests_jokers.SPEC) as scenarios, with their source comments
# ---------------------------------------------------------------------------
def _spec_comments():
    """SPEC row index -> its trailing comment in tests_jokers.py."""
    import tests_jokers
    src = open(tests_jokers.__file__, encoding="utf-8").read()
    rows = next(n.value.elts for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SPEC")
    notes = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            notes[tok.start[0]] = tok.string.lstrip("# ").strip()
    out = []
    for e in rows:
        out.append(" ".join(notes[l] for l in range(e.lineno, e.end_lineno + 1) if l in notes
                            and not notes[l].startswith("=")))
    return out


def _mod_words(kw):
    words = []
    for name, m in (kw.get("cards_mod") or {}).items():
        if m.get("enh"):
            words.append(ENH_NAMES[m["enh"]])
        if m.get("ed"):
            words.append(ED_NAMES[m["ed"]])
        if m.get("seal"):
            words.append(SEAL_NAMES[m["seal"]] + " Seal")
        if "rank" in m:
            words.append("Rank change")
        if m.get("bonus"):
            words.append("Perma chips")
    for e in kw.get("joker_ed", ()):
        if e:
            words.append(ED_NAMES[e] + " joker")
    if "hand_level" in kw:
        words.append("Hand levels")
    return list(dict.fromkeys(words))


def spec_scenarios():
    """tests_jokers.SPEC: one PLAY each, chips and mult computed by hand."""
    from tests_jokers import SPEC, HANDS
    notes = _spec_comments()
    out = []
    for (jokers, hand, kw, chips, mult), note in zip(SPEC, notes):
        lanes, bits = HANDS[hand]
        sel = tuple(i for i, b in enumerate(bits) if b == "1")
        played = " ".join(lanes.split()[i] for i in sel)
        held = " ".join(c for i, c in enumerate(lanes.split()) if i not in sel)
        want = math.floor(chips * mult)
        subject = " + ".join(list(jokers) + _mod_words(kw)) or "no joker"
        live = ", ".join(f"{k} {v}" for k, v in kw.items() if k != "cards_mod")
        mods = ", ".join(f"{c} {m}" for c, m in (kw.get("cards_mod") or {}).items())
        expect = (f"{note + '. ' if note else ''}Play {played}" + (f", hold {held}" if held else "")
                  + f". Want {chips:g} chips x {mult:g} mult = {want}."
                  + (f" State: {live}." if live else "") + (f" Cards: {mods}." if mods else ""))
        build = dict(hand=lanes, jokers=jokers, blind_target=BIG, **kw)
        out.append(Scenario("Scoring", subject, f"{hand}: {chips:g} x {mult:g}", expect, build,
                            (play(sel),),
                            (C("chips", lambda r: float(r.play(0).chips), float(chips)),
                             C("mult", lambda r: float(r.play(0).mult), float(mult)),
                             C("score", lambda r, w=want: int(r.play(0).score), OneOf((want, want + 1, want - 1))))))
    return out


def all_scenarios():
    return SCENARIOS + spec_scenarios()


# ---------------------------------------------------------------------------
# COVERAGE — every joker, consumable, pack, card modifier, boss, tag and voucher has a test
# ---------------------------------------------------------------------------
def subjects(scenarios):
    return {part.strip() for sc in scenarios for part in sc.subject.split("+")}


def missing():
    """The items no scenario is about. A SPEC row counts for its jokers only when it scores
    differently from the same hand with no joker (a row that scores like no joker shows nothing)."""
    from tests_jokers import SPEC, HANDS
    have = subjects(SCENARIOS)
    for jokers, hand, kw, chips, mult in SPEC:
        have |= set(_mod_words(kw))
        if jokers:
            lanes, bits = HANDS[hand]
            sel = jnp.asarray([b == "1" for b in bits] + [False] * (MAX_HAND_SIZE - len(bits)))
            base = explain_play(state(lanes, **kw), sel)
            if abs(float(base.chips) - chips) > 1e-3 or abs(float(base.mult) - mult) > 1e-3:
                have |= set(jokers)
    want = (set(JOKER_NAMES) | set(CONS_NAMES) | set(PACK_NAMES) | set(ENH_NAMES[1:]) | set(ED_NAMES[1:])
            | {n + " Seal" for n in SEAL_NAMES[1:]} | set(BOSS_NAMES) | set(TAG_NAMES) | set(VOUCHER_NAMES))
    return sorted(want - have)


def run_scenario_test(model=True):
    """Every scenario's checks, the reference model on every step, and the coverage."""
    from tests_run import Checks
    import tests_jokers as T
    c = Checks("scenarios")
    for sc in SCENARIOS:
        r = Run(sc)
        tag = f"{sc.subject}: {sc.title}"
        c.check(f"{tag}: runs", r.error is None, r.error)
        for label, got, want, ok in r.results:
            c.check(f"{tag}: {label}", ok, fmt(got), fmt(want))
        if model and r.error is None:
            for k, a in enumerate(r.actions):
                pre, post = r.raw[k], r.raw[k + 1]
                out = jax.device_get(T._explain(pre, a.select)[1])
                T.check_step(c, f"{tag} step {k + 1}", T.host(pre), jax.device_get(a), T.host(post),
                             r.rewards[k], out)
    gaps = missing()
    c.check("every joker / consumable / pack / modifier / boss / tag / voucher has a scenario", not gaps, gaps)
    return c.done()


if __name__ == "__main__":
    run_scenario_test()
