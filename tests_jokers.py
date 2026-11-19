"""Joker test suite — proves every joker against an independent reference.

ref_balatro.py is a second implementation of the jokers, ported from the
game's Lua in plain Python. It does not read the jokers.py columns or
scoring.py: hand detection, the card modifiers, the hand levels, the scoring
order, the copies, the counters, the discard and end-of-round effects and the
cashout money are all re-derived there. The env's random draws are the one
shared input (PlayOut.rolls / hrolls / lucky_rolls / glass_roll / space_roll /
rnd). Each layer compares the env against it:

  table  every TABLE joker has a spec case. A new joker fails here until it has one.
  spec   hand-computed cases per joker (explain_play), fire and no-fire. They
         also check the reference itself, so the layers below can trust it.
  order  every slot order of a 5-joker add / XMULT / copy mix, and MOVE_JOKER -> PLAY.
  fuzz   random hands x random card modifiers x random loadouts x random live state, vmapped.
  life   scripted env_step sequences with hand-computed values: counters,
         discards, the end of the round, budget jokers, the shop jokers.
  soak   vmapped random runs. Every step of every env goes through the
         reference transition model (score, counters, money, stage, budget,
         reward, the card instances, the consumables, the packs).

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
                 apply_budget, roll_shop, roll_joker, acquire, MAX_HAND_SIZE, MAX_SHOP_SLOTS,
                 MAX_DECK_SIZE, MAX_CONSUMABLES, V_PLAY, V_DISCARD, V_BUY, V_SELL, V_REROLL, V_NEXT,
                 V_MOVE_JOKER, V_MOVE_CARD, V_USE, V_SELL_CONS, V_OPEN, V_PICK, V_SKIP, V_REDEEM, STAGE_BLIND,
                 STAGE_SHOP, STAGE_LOST, STAGE_WON, STAGE_PACK, STAGE_SELECT, STAGE_END, STAGE_CASHOUT,
                 BLIND_BOSS, N_VERBS, K_HOOK, _hooked, _hook_discard)
from env import CONS_SELL
from items import (CONS_NAMES, CONS_ID, CONS_KIND, CONS_USE, CONS_PARAM, CONS_LO, CONS_HI, CONS_NEED,
                   CONS_RANKS, CK_TAROT, CK_PLANET, CK_SPECTRAL, U_ENHANCE, U_SUIT, U_STRENGTH, U_HANGED,
                   U_DEATH, U_FOOL, U_HERMIT, U_TEMPERANCE, U_SPAWN_TAROT, U_SPAWN_PLANET, U_WHEEL,
                   U_JUDGEMENT, U_PLANET, U_CREATE_CARDS, U_SEAL, U_AURA, U_WRAITH, U_SIGIL, U_OUIJA,
                   U_ECTOPLASM, U_IMMOLATE, U_ANKH, U_HEX, U_CRYPTID, U_SOUL, U_BLACK_HOLE,
                   PACK_NAMES, PACK_KIND, PACK_SIZE, PACK_CHOOSE, PACK_COST, PK_ARCANA, PK_CELESTIAL,
                   PK_STANDARD, PK_BUFFOON, PK_SPECTRAL, PLANET_OF_HAND, ED_COST, IK_JOKER, IK_PLANET,
                   IK_TAROT, IK_CARD, IK_SPECTRAL, B_GREEN, B_GHOST, B_ANAGLYPH, B_PLASMA, N_BACKS,
                   BACK_JOKER_SLOT, BACK_ANTE_SCALING, N_HAND_TYPES, BOSS_NAMES, TAG_NAMES, VOUCHER_NAMES,
                   STK_ETERNAL, STK_PERISH, STK_RENTAL)
from jokers import TABLE, JOKER_NAMES, JOKER_ID, JOKERS, JOKER_COST, JOKER_RARITY, MAX_JOKERS, N_JOKERS
from scoring import HAND_NAMES
from fixtures import state, select, cards as card_ids
from tests_run import Checks, act

NAME = JOKER_NAMES          # id -> name; the reference is keyed by name, never by id
assert HAND_NAMES == R.HAND_ORDER
CARD_FIELDS = ("card_rank", "card_suit", "card_enh", "card_ed", "card_seal", "card_bonus", "card_live")
SCORE_CAP = 2147483520                 # scoring.SCORE_CAP: a hand's score stops here (float32 < 2^31)
INT32_MAX = 2147483647                 # env: chips_scored stops here
_KIND = [int(k) for k in CONS_KIND]
_USE = [int(u) for u in CONS_USE]
_PARAM = [int(x) for x in CONS_PARAM]
_LO = [int(x) for x in CONS_LO]
_HI = [int(x) for x in CONS_HI]
_NEED = [int(x) for x in CONS_NEED]
_RANKS = [int(x) for x in CONS_RANKS]
_PKIND = [int(k) for k in PACK_KIND]


def host(s):
    """Device state -> numpy state (the PRNG key as raw words)."""
    return jax.device_get(s._replace(key=jax.random.key_data(s.key)))


def _score_hooked(s, sel):
    """score_play as env_step scores a PLAY: after The Hook's discard. -> (PlayResult, PlayOut)"""
    return score_play(_hook_discard(s, _hooked(s, sel)), sel)[:2]


_explain = jax.jit(_score_hooked)
_explain_batch = jax.jit(jax.vmap(_score_hooked))


def exact(got, want, prod=None):
    """float32 is exact below 2**24; above it, allow its rounding. prod = the
    reference's chips x mult: when it sits on a whole number (12.6 x 15), float
    error may land on either side, so both are fine."""
    if prod is not None and abs(prod - round(prod)) <= 1e-6 * max(1.0, abs(prod)):
        return got in (math.floor(prod), round(prod)) or exact(got, want)
    return got == want if abs(want) < 2 ** 24 else abs(got - want) <= abs(want) * 2 ** -20


def near(got, want):
    return abs(got - want) <= 1e-4 * max(1.0, abs(want))


# ===========================================================================
# HOST STATE -> REFERENCE INPUTS
# ===========================================================================
# records, card_of and played_lanes: see THE REFERENCE TRANSITION below (they know the boss)


def counts(arr):
    return Counter({HAND_NAMES[k]: int(arr[k]) for k in range(N_HAND_TYPES) if int(arr[k])})


def ref_g(p):
    live = np.asarray(p.card_live, bool)
    enh = np.asarray(p.card_enh)
    size = int(live.sum())
    return dict(money=int(p.money), hands_left=int(p.hands) - 1, discards=max(int(p.discards), 0),
                deck_left=max(int(p.deck_size) - int(p.deck_ptr), 0), slots=int(p.joker_slots),
                nines=int((live & (enh != R.STONE) & (np.asarray(p.card_rank) == R.NINE)).sum()),
                deck_size=size, deck_start=int(p.deck_start), played=counts(p.hand_played), round=counts(p.hand_round),
                targets=dict(idol=int(p.idol_card), mail=int(p.mail_rank),
                             ancient=int(p.ancient_suit), castle=int(p.castle_suit)),
                is_boss=int(p.blind_index) == BLIND_BOSS, discards_used=int(p.discards_used),
                levels={HAND_NAMES[k]: int(p.hand_level[k]) for k in range(N_HAND_TYPES)},
                steel=int((live & (enh == R.STEEL)).sum()), stone=int((live & (enh == R.STONE)).sum()),
                enhanced=int((live & (enh != 0)).sum()), tarots_used=int(p.tarots_used),
                unique_planets=int(np.asarray(p.planets_used).sum()), skips=int(p.skips),
                plasma=int(p.back) == B_PLASMA)


def ref_play_state(p, sel, out, mods=None):
    """Score a PLAY from a host state, with the env's draws. -> (result, records after, cards)."""
    J = records(p)
    lanes = played_lanes(p, sel)
    cards = [card_of(p, p.hand[l]) for l in lanes]
    held = [(l, card_of(p, p.hand[l])) for l in range(MAX_HAND_SIZE) if p.hand_valid[l] and l not in lanes]
    R_ = out.rolls.shape[1]
    res = R.ref_play(J, cards, held, ref_g(p),
                     rolls=lambda k, r, j: bool(out.rolls[k, min(r, R_ - 1), j]),
                     hrolls=lambda l, r, j: bool(out.hrolls[l, min(r, out.hrolls.shape[1] - 1), j]),
                     rnd=lambda j: float(out.rnd[j]), mods=mods,
                     lucky=lambda k, r, w: bool(out.lucky_rolls[k, min(r, R_ - 1), w]),
                     glass=lambda k: bool(out.glass_roll[k]),
                     space=lambda j: bool(out.space_roll[j]))
    return res, J, [c.i for c in cards]


# ===========================================================================
# THE REFERENCE TRANSITION — what one env_step must do, field by field
# ===========================================================================
_T, _F = True, False
LEGAL_NP = np.array([
    #  PLAY DISC BUY  SELL REROL NEXT MOVEJ MOVEC USE  SELLC OPEN PICK SKIP REDEEM
    [_T,  _T,  _F,  _T,  _F,   _F,  _T,   _T,   _T,  _T,   _F,  _F,  _F,  _F],   # BLIND
    [_F,  _F,  _T,  _T,  _T,   _T,  _T,   _F,   _T,  _T,   _T,  _F,  _F,  _T],   # SHOP
    [_F] * 14, [_F] * 14,                                                          # LOST, WON
    [_F,  _F,  _F,  _T,  _F,   _F,  _T,   _T,   _T,  _T,   _F,  _T,  _T,  _F],   # PACK
    [_F,  _F,  _F,  _T,  _T,   _T,  _T,   _F,   _T,  _T,   _F,  _F,  _T,  _F],   # SELECT
    [_F,  _F,  _F,  _F,  _F,   _T,  _T,   _F,   _F,  _F,   _F,  _F,  _F,  _F],   # END: drag a joker, NEXT
    [_F,  _F,  _F,  _T,  _F,   _T,  _T,   _F,   _T,  _T,   _F,  _F,  _F,  _F],   # CASH OUT
])
STAGES = 8


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
        elif x["name"] in ("Gros Michel", "Cavendish") and (not x.get("deb") or x.get("fresh")):
            gros |= x["name"] == "Gros Michel"
        else:
            c.check(f"{tag}: {x['name']} left the loadout", False, got, [y["name"] for y in J])
            left.append(x)
    return left, gros


def _eff_count(J, name):
    """Slots that act as `name` (the joker itself and its copies; a debuffed one does not)."""
    names = [x["name"] for x in J]
    return sum(1 for t in R.effective(J) if t is not None and names[t] == name)


def _bump(J, name, dv, copies=False):
    """+dv to every acting joker called name (its counter; copies never count)."""
    for x in J:
        if x["name"] == name and not x.get("deb"):
            x["v"] += dv


def boss_on(p):
    """The boss whose ability is on in this state (its round, STAGE_END too; not disabled), or None."""
    if int(p.game_stage) not in (STAGE_BLIND, STAGE_END) or int(p.blind_index) != BLIND_BOSS \
            or bool(p.boss_disabled) or int(p.boss) < 0:
        return None
    return BOSS_NAMES[int(p.boss)]


def vouchers_of(p):
    return {VOUCHER_NAMES[i] for i in range(len(VOUCHER_NAMES)) if bool(p.vouchers[i])}


def tags_of(p):
    return [(TAG_NAMES[int(t)], int(h)) for t, h in zip(p.tags, p.tag_hand) if int(t) >= 0]


def _sell_of(cost):
    return max(1, cost // 2)


def joker_cost_ref(name, ed, disc, stk=0):
    """set_cost: the joker's cost + its edition after the discount; a Rental joker $1."""
    return 1 if stk & STK_RENTAL else R.cost(int(JOKER_COST[JOKER_ID[name]]) + int(ED_COST[ed]), disc)


def joker_sell(name, ed, disc, stk=0):
    return _sell_of(joker_cost_ref(name, ed, disc, stk))


def _reprice_cons(w, astro_old, astro_new):
    """Astronomer in or out: every held consumable's sell value follows its new cost (a
    Negative Planet $1 <-> $4); the extra (Gift Card) stays."""
    if astro_old != astro_new:
        w.cons_sell = [s + _sell_of(cons_cost_ref(cid, e, w.disc, astro_new))
                       - _sell_of(cons_cost_ref(cid, e, w.disc, astro_old))
                       for cid, e, s in zip(w.cons, w.cons_ed, w.cons_sell)]


def cons_cost_ref(cid, ed, disc, astro):
    if _KIND[cid] == CK_PLANET and astro:
        return 0
    return R.cost((4 if _KIND[cid] == CK_SPECTRAL else 3) + int(ED_COST[ed]), disc)


def _hand_u(p, purpose):
    """The env's uniform draw (32,) for a fold_in purpose of p's key (a shared random input)."""
    return np.asarray(_fold_uniform(np.asarray(p.key, np.uint32), purpose))


_fold_uniform = jax.jit(lambda kd, purpose: jax.random.uniform(
    jax.random.fold_in(jax.random.wrap_key_data(kd), purpose), (MAX_HAND_SIZE,)), static_argnums=1)


class Want:
    """The expected post-state, built up by the reference model of one step."""

    def __init__(self, p):
        F = ("money", "chips_scored", "hands", "discards", "game_stage", "last_cashout", "discards_used",
             "free_rerolls", "reroll_cost", "last_hand", "last_cons", "tarots_used", "skips", "ante",
             "blind_index", "boss_hand", "unused_discards", "round_hand_bonus", "ecto_minus",
             "consumable_slots", "base_hand_size", "base_hands", "base_discards", "shop_size", "boss", "round",
             "ox_hand", "boss_sub", "pack_return", "blind_target", "pack_id", "pack_left", "reroll_inc", "stake",
             "back")
        self.f = {f: int(getattr(p, f)) for f in F}
        self.f.update(gros_extinct=bool(p.gros_extinct), boss_disabled=bool(p.boss_disabled),
                      boss_rerolled=bool(p.boss_rerolled), reroll_d6=bool(p.reroll_d6))
        self.extra_room = 0                # a Negative card being used keeps its slot until it dissolves
        self.level = np.array(p.hand_level)
        self.planets_used = np.array(p.planets_used)
        held = [k for k in range(MAX_CONSUMABLES) if int(p.consumables[k]) >= 0]
        self.cons = [int(p.consumables[k]) for k in held]
        self.cons_ed = [int(p.cons_ed[k]) for k in held]
        self.cons_sell = [int(p.cons_sell[k]) for k in held]
        self.new_cons = []                 # (kind, count, exact, sell, ed): random consumables made by the step
        self.cards = {f: np.array(getattr(p, f)) for f in CARD_FIELDS + ("card_played_ante",)}
        self.new_cards = []                # (to_hand, expected attrs or None): cards the step made
        self.tags = tags_of(p)
        self.vouchers = vouchers_of(p)
        self.used = np.array(p.bosses_used)

    @property
    def disc(self):
        return R.discount_of(self.vouchers)

    def capacity(self):
        return self.f["consumable_slots"] + sum(e == 4 for e in self.cons_ed) \
            + sum(n for _, n, _, _, e in self.new_cons if e == 4) + self.extra_room

    def room(self):
        return self.capacity() - len(self.cons) - sum(n for _, n, _, _, _ in self.new_cons)

    def spawn(self, kind, n, exact=True, astro=False):
        """n random cards of kind (CK_*), each needs room."""
        n = max(0, min(n, self.room()))
        if n:
            base = 4 if kind == CK_SPECTRAL else 3
            sell = 1 if (kind == CK_PLANET and astro) else _sell_of(R.cost(base, self.disc))
            self.new_cons.append((kind, n, exact, sell, 0))

    def spawn_seq(self, kinds, astro=False):
        for kd in kinds:
            self.spawn(kd, 1, astro=astro)

    def give(self, cid, astro=False, ed=0):
        if self.room() > 0 or ed == 4:
            self.cons.append(cid)
            self.cons_ed.append(ed)
            self.cons_sell.append(_sell_of(cons_cost_ref(cid, ed, self.disc, astro)))

    def drop_cons(self, t):
        """Consumable slot t leaves (used or sold); the rest shift left."""
        for arr in (self.cons, self.cons_ed, self.cons_sell):
            del arr[t]


def _check_cons(c, tag, w, q):
    """The consumables: the known ones first, then the random new ones (their kind)."""
    got = [int(x) for x in q.consumables if x >= 0]
    k = len(w.cons)
    npeq(c, f"{tag}: consumables (known)", got[:k], w.cons)
    rest = got[k:]
    n_max = sum(n for _, n, _, _, _ in w.new_cons)
    exact = all(e for _, _, e, _, _ in w.new_cons)
    if exact:
        c.eq(f"{tag}: new consumables", len(rest), n_max)
    else:
        c.check(f"{tag}: new consumables at most {n_max}", len(rest) <= n_max, rest, w.new_cons)
    kinds = [kd for kd, n, _, _, _ in w.new_cons for _ in range(n)]
    for j, x in enumerate(rest):
        want = kinds[j] if exact and j < len(kinds) else set(kinds)
        ok = _KIND[x] == want if exact and j < len(kinds) else _KIND[x] in want
        c.check(f"{tag}: new consumable kind", ok, CONS_NAMES[x], want)
    c.check(f"{tag}: consumables compact", all(int(x) < 0 for x in q.consumables[len(got):]))
    sells = [int(v) for x, v in zip(q.consumables, q.cons_sell) if x >= 0]
    npeq(c, f"{tag}: consumable sell values (known)", sells[:k], w.cons_sell)
    eds = [int(v) for x, v in zip(q.consumables, q.cons_ed) if x >= 0]
    npeq(c, f"{tag}: consumable editions (known)", eds[:k], w.cons_ed)
    new_sells = [v for _, n, _, v, _ in w.new_cons for _ in range(n)]
    if exact:
        npeq(c, f"{tag}: new consumable sell values", sells[k:], new_sells)
    else:
        c.check(f"{tag}: new consumable sell values", all(v in new_sells for v in sells[k:]), sells[k:], new_sells)
    c.check(f"{tag}: a vacant consumable slot sells for 0",
            all(int(v) == 0 for x, v in zip(q.consumables, q.cons_sell) if x < 0))
    return len(rest)


def _check_cards(c, tag, w, p, q):
    """The card instances: every old slot as predicted; new slots as predicted."""
    new = [i for i in range(MAX_DECK_SIZE) if q.card_live[i] and not w.cards["card_live"][i]]
    keep = np.array([i not in new for i in range(MAX_DECK_SIZE)])
    for f in CARD_FIELDS + ("card_played_ante",):
        g, e = np.asarray(getattr(q, f))[keep], w.cards[f][keep]
        if f == "card_played_ante":
            g, e = g & np.asarray(q.card_live)[keep], e & np.asarray(q.card_live)[keep]
        ok = np.allclose(g, e) if f == "card_bonus" else np.array_equal(g, e)
        c.check(f"{tag}: {f}", ok, np.flatnonzero(~np.isclose(g, e))[:6])
    c.eq(f"{tag}: cards added", len(new), len(w.new_cards))
    for i, (to_hand, attrs) in zip(new, w.new_cards):
        if attrs:
            for f, v in attrs.items():
                if isinstance(v, (set, frozenset)):
                    c.check(f"{tag}: new card {f}", int(getattr(q, f)[i]) in v, int(getattr(q, f)[i]), v)
                else:
                    c.check(f"{tag}: new card {f}", np.isclose(float(getattr(q, f)[i]), v),
                            float(getattr(q, f)[i]), v)
        if to_hand and int(q.game_stage) == STAGE_BLIND:      # a cleared blind sends the hand back
            c.check(f"{tag}: new card in hand", i in [int(h) for h, v in zip(q.hand, q.hand_valid) if v])
    return new


def _forced_lanes(p):
    if boss_on(p) != "Cerulean Bell" or int(p.forced_card) < 0:
        return []
    return [l for l in range(MAX_HAND_SIZE) if p.hand_valid[l] and int(p.hand[l]) == int(p.forced_card)]


def played_lanes(p, sel, n=5):
    """The env's repair: the first n selected lanes that hold a card, lane order; Cerulean Bell's
    forced card is always one of them."""
    forced = _forced_lanes(p)
    others = [l for l in range(MAX_HAND_SIZE) if sel[l] and p.hand_valid[l] and l not in forced][:n - len(forced)]
    return sorted(others + forced)


def _sel(p, sel, cid):
    """The lanes a consumable acts on (first CONS_HI valid selected, Bell's card in) and whether the count fits."""
    lanes = played_lanes(p, sel, _HI[cid])
    return lanes, _LO[cid] <= len(lanes) <= _HI[cid]


def _perished(x):
    """A Perishable joker after its last round (debuffed for good)."""
    return bool(x.get("stk", 0) & STK_PERISH) and x.get("left", 0) <= 0


def _eternal(x):
    return bool(x.get("stk", 0) & STK_ETERNAL)


def records(p):
    """The held jokers as reference records, slot order. deb: it does not act (Crimson Heart
    disabled it, or it perished); stk: its stickers (STK_*); left: a Perishable joker's rounds left."""
    out = []
    for i, j in enumerate(p.jokers):
        if j >= 0:
            x = dict(name=NAME[int(j)], v=float(p.joker_value[i]), st=int(p.joker_state[i]),
                     sell=int(p.joker_sell_value[i]), ed=int(p.joker_ed[i]), stk=int(p.joker_sticker[i]),
                     left=int(p.joker_perish[i]))
            x["deb"] = bool(p.joker_debuff[i]) or _perished(x)
            out.append(x)
    return out


def new_record(nm, v, st, ed, disc, stk=0):
    """A joker as it enters: a shop / Buffoon pack one with its stickers (a Perishable one has
    5 rounds), any other with none."""
    return dict(name=nm, v=v, st=st, sell=joker_sell(nm, ed, disc, stk), ed=ed, deb=False, stk=stk,
                left=R.PERISH_ROUNDS if stk & STK_PERISH else 0)


def card_of(p, inst, boss=None, m=None):
    """Instance inst of a host state as a reference Card (debuffed if the boss says so)."""
    i = int(inst)
    cd = R.Card(i, int(p.card_rank[i]), int(p.card_suit[i]), int(p.card_enh[i]), int(p.card_ed[i]),
                int(p.card_seal[i]), float(p.card_bonus[i]))
    if boss is not None:
        cd.deb = R.debuffed(cd, boss, m, bool(p.card_played_ante[i]))
    return cd


def _mods(J):
    return R.mods_of(R.acting_names(J))


def _new_joker(c, tag, w, J, q, k, rarity=None, extra=""):
    """A random joker landed in slot k of q: read it, check it fits. -> appended to J."""
    jid = int(q.jokers[k]) if k < MAX_JOKERS else -1
    c.check(f"{tag}: a new joker{extra}", jid >= 0, jid)
    if jid < 0:
        return
    nm = NAME[jid]
    if rarity is not None:
        c.check(f"{tag}: the new joker's rarity{extra}", int(JOKER_RARITY[jid]) == rarity, nm, rarity)
    ed = int(q.joker_ed[k])
    J.append(new_record(nm, _seed(jid), int(q.joker_state[k]), ed, w.disc))
    w.f["free_rerolls"] += nm == "Chaos the Clown"
    w.f["discards"] += R.D_SIZE.get(nm, 0)


def _can_use(p, J, w, cid, sel, from_slot):
    """card.lua can_use_consumeable."""
    if not 0 <= cid < len(_USE):
        return False
    u = _USE[cid]
    lanes, n_ok = _sel(p, sel, cid)
    room = w.room() + from_slot > 0
    hand_ok = int(p.game_stage) in (STAGE_BLIND, STAGE_PACK)
    in_hand = int(np.asarray(p.hand_valid).sum())
    if _LO[cid] > 0:
        ok = in_hand >= _LO[cid] and hand_ok and n_ok
        if u == U_AURA and lanes:
            ok = ok and int(p.card_ed[int(p.hand[lanes[0]])]) == 0
        return ok
    if _NEED[cid] > 0:
        return in_hand >= _NEED[cid] and hand_ok
    if u in (U_WHEEL, U_ECTOPLASM, U_HEX):
        return any(x["ed"] == 0 for x in J)
    if u in (U_JUDGEMENT, U_WRAITH, U_SOUL):
        return len(J) < int(p.joker_slots)
    if u == U_ANKH:
        return 1 <= len(J) < int(p.joker_slots) and int(p.joker_slots) > 1
    if u == U_FOOL:
        return int(p.last_cons) >= 0 and int(p.last_cons) != CONS_ID["The Fool"] and room
    if u in (U_SPAWN_TAROT, U_SPAWN_PLANET):
        return room
    return True


def _destroyed(c, tag, w, p, q, want_n, inst_pool, mfaces, new_ok=None, n_new=0):
    """Random cards destroyed: read them from q (they must be among inst_pool, want_n of them).
    new_ok: n_new new cards follow in the same step and take the first free slots, so a destroyed
    slot with fewer than n_new old free slots below it comes back live as a new card (new_ok(i):
    q's card i can be one). -> faces among them (Canio)."""
    dead = [i for i in inst_pool if not q.card_live[i] and w.cards["card_live"][i]]
    if new_ok is not None and len(dead) < want_n:
        free0 = ~np.asarray(w.cards["card_live"], bool)
        fields = ("card_rank", "card_suit", "card_enh", "card_ed", "card_seal")
        cand = [i for i in inst_pool if q.card_live[i] and free0[:i].sum() < n_new and new_ok(i)]
        changed = [i for i in cand if any(int(getattr(q, f)[i]) != int(w.cards[f][i]) for f in fields)]
        dead += (changed + [i for i in cand if i not in changed])[:want_n - len(dead)]
    c.eq(f"{tag}: {want_n} cards destroyed", len(dead), want_n)
    for i in dead:
        w.cards["card_live"][i] = False
    return sum(mfaces(i) for i in dead)


def _apply_cons(c, tag, p, q, J, w, cid, sel):
    """use_consumeable on the expected state. -> J (random jokers / cards / choices are read from q)."""
    u, prm = _USE[cid], _PARAM[cid]
    lanes, _ = _sel(p, sel, cid)
    inst = [int(p.hand[l]) for l in lanes]
    hand_inst = [int(p.hand[l]) for l in range(MAX_HAND_SIZE) if p.hand_valid[l]]
    cd = w.cards
    m = _mods(J)
    boss = boss_on(p)
    astro = "Astronomer" in R.acting_names(J)

    def face_of(i):
        card = card_of(p, i, boss, m)
        return R.is_face(card, m)

    if u == U_ENHANCE:
        for i in inst:
            cd["card_enh"][i] = prm
            cd["card_played_ante"][i] = False             # set_ability: a new ability table
    elif u == U_SUIT:
        for i in inst:
            cd["card_suit"][i] = prm
    elif u == U_STRENGTH:
        for i in inst:
            cd["card_rank"][i] = (cd["card_rank"][i] + 1) % 13
    elif u == U_HANGED:
        glass = sum(int(cd["card_enh"][i]) == R.GLASS for i in inst)
        _bump(J, "Canio", 1.0 * sum(face_of(i) for i in inst))
        for i in inst:
            cd["card_live"][i] = False
        _bump(J, "Glass Joker", 0.75 * glass)
    elif u == U_DEATH:
        right = inst[-1]
        for i in inst[:-1]:
            for f in CARD_FIELDS[:-1] + ("card_played_ante",):       # copy_card: the mark too
                cd[f][i] = cd[f][right]
    elif u == U_FOOL:
        w.give(int(p.last_cons), astro)
    elif u == U_HERMIT:
        w.f["money"] += max(0, min(w.f["money"], prm))
    elif u == U_TEMPERANCE:
        w.f["money"] += min(sum(x["sell"] for x in J), prm)
    elif u in (U_SPAWN_TAROT, U_SPAWN_PLANET):
        if u == U_SPAWN_TAROT:
            w.spawn(CK_TAROT, prm)
        else:
            w.spawn(CK_PLANET, prm, astro=astro)
        # the new cards: none that exists (held, in the shop, in the open pack, and the used card
        # itself: used_jokers keeps it until it dissolves), none twice; Showman allows them all
        if "Showman" not in R.acting_names(J):
            shop_c = {int(p.shop_slots[i]) for i in range(MAX_SHOP_SLOTS)
                      if int(p.shop_slots[i]) >= 0 and int(p.shop_kind[i]) in (IK_TAROT, IK_PLANET, IK_SPECTRAL)}
            pk = _PKIND[int(p.pack_id)] if int(p.pack_id) >= 0 else -1
            pack_c = {int(x) for x in p.pack_items if int(x) >= 0} if pk in (PK_ARCANA, PK_CELESTIAL, PK_SPECTRAL) else set()
            new = [int(x) for x in q.consumables if x >= 0][len(w.cons):]
            old = set(w.cons) | shop_c | pack_c | {cid}
            c.check(f"{tag}: {CONS_NAMES[cid]} makes no card that exists (itself too), none twice",
                    not set(new) & old and len(set(new)) == len(new), [CONS_NAMES[x] for x in new])
    elif u == U_WHEEL:
        changed = [i for i, x in enumerate(J) if int(q.joker_ed[i]) != x["ed"]]
        c.check(f"{tag}: Wheel: at most one joker changes", len(changed) <= 1, changed)
        for i in changed:
            c.check(f"{tag}: Wheel: from none to Foil / Holo / Poly", J[i]["ed"] == 0 and
                    int(q.joker_ed[i]) in (1, 2, 3), J[i]["ed"], int(q.joker_ed[i]))
            J[i]["sell"] += (joker_sell(J[i]["name"], int(q.joker_ed[i]), w.disc, J[i]["stk"])
                             - joker_sell(J[i]["name"], 0, w.disc, J[i]["stk"]))
            J[i]["ed"] = int(q.joker_ed[i])
    elif u == U_ECTOPLASM:
        hit = [i for i, x in enumerate(J) if x["ed"] == 0 and int(q.joker_ed[i]) == 4]
        c.eq(f"{tag}: Ectoplasm makes one editionless joker Negative", len(hit), 1)
        for i in hit:
            J[i]["sell"] += joker_sell(J[i]["name"], 4, w.disc, J[i]["stk"]) - joker_sell(J[i]["name"], 0, w.disc, J[i]["stk"])
            J[i]["ed"] = 4
        w.f["base_hand_size"] -= w.f["ecto_minus"]
        w.f["ecto_minus"] += 1
    elif u in (U_HEX, U_ANKH):
        # Hex: a random editionless joker turns Polychrome; Ankh: a random joker (an Eternal one
        # too) is copied (copy_card: its stickers and rounds left too). Every other joker but the
        # Eternal ones is destroyed. The env's pick is read from q: the choice whose survivors
        # (slot order) are q's jokers.
        names = [NAME[int(j)] for j in q.jokers if j >= 0]
        surv = lambda x: [y for y in J if y is x or _eternal(y)]

        def fits(x):
            k = next(i for i, y in enumerate(surv(x)) if y is x)
            return (abs(x["v"] - float(q.joker_value[k])) < 1e-4 and x["stk"] == int(q.joker_sticker[k])
                    and x["left"] == int(q.joker_perish[k]) and (u == U_ANKH or int(q.joker_ed[k]) == 3))
        cand = [x for x in J if (u == U_ANKH or x["ed"] == 0)
                and [y["name"] for y in surv(x)] + ([x["name"]] if u == U_ANKH else []) == names]
        if len(cand) > 1:
            cand = [x for x in cand if fits(x)] or cand
        c.check(f"{tag}: {CONS_NAMES[cid]} keeps a held joker and the Eternal ones", bool(cand), names,
                [x["name"] for x in J])
        if cand:
            x = cand[0]
            for y in J:
                if y is not x and not _eternal(y):
                    w.f["discards"] = max(w.f["discards"] - (0 if y.get("deb") else R.D_SIZE.get(y["name"], 0)), 0)
                    w.f["free_rerolls"] = max(w.f["free_rerolls"] - (y["name"] == "Chaos the Clown" and not _perished(y)), 0)
            astro_old = "Astronomer" in R.acting_names(J)
            if u == U_HEX:
                x["sell"] += joker_sell(x["name"], 3, w.disc, x["stk"]) - joker_sell(x["name"], x["ed"], w.disc, x["stk"])
                x["ed"] = 3
                J[:] = surv(x)
            else:
                ced = 0 if x["ed"] == 4 else x["ed"]
                extra = x["sell"] - joker_sell(x["name"], x["ed"], w.disc, x["stk"])
                cp = dict(x, ed=ced, sell=joker_sell(x["name"], ced, w.disc, x["stk"]) + extra, deb=_perished(x))
                if not _perished(x):            # a perished copy adds nothing (env.copy_joker)
                    w.f["free_rerolls"] += x["name"] == "Chaos the Clown"
                    w.f["discards"] += R.D_SIZE.get(x["name"], 0)
                J[:] = surv(x) + [cp]
            _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))
    elif u in (U_JUDGEMENT, U_WRAITH, U_SOUL):
        _new_joker(c, tag, w, J, q, len(J), {U_WRAITH: 3, U_SOUL: 4}.get(u), f" ({CONS_NAMES[cid]})")
        if u == U_WRAITH:
            w.f["money"] = 0
    elif u == U_PLANET:
        w.level[prm] += 1
        w.planets_used[prm] = True
        _bump(J, "Constellation", 0.1)
    elif u == U_BLACK_HOLE:
        w.level += 1
    elif u == U_SEAL:
        for i in inst:
            cd["card_seal"][i] = prm
    elif u == U_AURA:
        for i in inst:
            c.check(f"{tag}: Aura: Foil / Holo / Poly", int(q.card_ed[i]) in (1, 2, 3), int(q.card_ed[i]))
            cd["card_ed"][i] = int(q.card_ed[i])
    elif u in (U_SIGIL, U_OUIJA):
        f = "card_suit" if u == U_SIGIL else "card_rank"
        vals = {int(getattr(q, f)[i]) for i in hand_inst}
        c.eq(f"{tag}: {CONS_NAMES[cid]}: one {f[5:]} for the hand", len(vals), 1)
        for i in hand_inst:
            cd[f][i] = int(getattr(q, f)[i])
        if u == U_OUIJA:
            w.f["base_hand_size"] -= 1
    elif u in (U_CREATE_CARDS, U_IMMOLATE):
        n = 1 if u == U_CREATE_CARDS else min(prm, len(hand_inst))
        new_ok = None
        if u == U_CREATE_CARDS:
            new_ok = lambda i: ((_RANKS[cid] >> int(q.card_rank[i])) & 1 and int(q.card_enh[i]) in (1, 2, 3, 4, 5, 7, 8)
                                and int(q.card_ed[i]) == 0 and int(q.card_seal[i]) == 0)
        n_new = min(prm, int((~w.cards["card_live"]).sum()) + 1) if u == U_CREATE_CARDS else 0
        faces = _destroyed(c, tag, w, p, q, n, hand_inst, face_of, new_ok, n_new)
        _bump(J, "Canio", 1.0 * faces)
        if u == U_IMMOLATE:
            w.f["money"] += 20
        else:
            ranks = {r for r in range(13) if (_RANKS[cid] >> r) & 1}
            free = int((~w.cards["card_live"]).sum())
            for _ in range(min(prm, free)):
                w.new_cards.append((True, dict(card_rank=frozenset(ranks), card_enh=frozenset({1, 2, 3, 4, 5, 7, 8}),
                                               card_ed=0, card_seal=0)))
            _bump(J, "Hologram", 0.25 * min(prm, free))
    elif u == U_CRYPTID:
        src = card_of(p, inst[0])
        free = int((~w.cards["card_live"]).sum())
        for _ in range(min(2, free)):
            w.new_cards.append((True, dict(card_rank=src.r, card_suit=src.s, card_enh=src.enh, card_ed=src.ed,
                                           card_seal=src.seal, card_bonus=src.bonus,
                                           card_played_ante=float(cd["card_played_ante"][inst[0]]))))
        _bump(J, "Hologram", 0.25 * min(2, free))
    w.f["tarots_used"] += _KIND[cid] == CK_TAROT
    if _KIND[cid] != CK_SPECTRAL:
        w.f["last_cons"] = cid                              # The Fool too: a 2nd Fool cannot copy it
    return J


def _clear_round(c, tag, p, J, w, held_cards, saved=False, played=None):
    """The end of the round (end_round + evaluate_round): held Gold / Blue Seal cards, the
    END_OF_ROUND jokers (not a debuffed one), the cash-out rows, the next blind on deck; after
    a boss the next ante. The cash-out screen follows (the $ is paid at its NEXT); the win
    pays at once. -> (J, cashout)"""
    g = ref_g(p)
    g.update(discards=max(int(p.discards), 0), is_boss=int(p.blind_index) == BLIND_BOSS)
    acting = [x for x in J if not x.get("deb")]
    gift = sum(x["name"] == "Gift Card" for x in acting)
    w.cons_sell = [v + gift for v in w.cons_sell]
    w.new_cons = [(kd, n, e, v + gift, ed) for kd, n, e, v, ed in w.new_cons]
    gold, planets = R.ref_held_end_of_round(J, held_cards)
    w.f["money"] += gold
    if w.f["last_hand"] >= 0:
        w.spawn(CK_PLANET, planets, astro="Astronomer" in R.acting_names(J))
    g["unique_planets"] = int(w.planets_used.sum())
    g["steel"] = int((w.cards["card_live"] & (w.cards["card_enh"] == R.STEEL)).sum())
    g["nines"] = int((w.cards["card_live"] & (w.cards["card_enh"] != R.STONE)
                      & (w.cards["card_rank"] == R.NINE)).sum())
    g["discards"] = max(w.f["discards"], 0)
    J = _drop(J, R.ref_end_of_round(J, g))
    # calculate_perishable (after each joker's end-of-round effect): one round less; at 0 it is
    # debuffed for good, before the cash out ($, To the Moon) reads the jokers
    astro_old = "Astronomer" in R.acting_names(J)
    for x in J:
        if x["stk"] & STK_PERISH and x["left"] > 0:
            x["left"] -= 1
            if x["left"] == 0:
                x["deb"] = x["fresh"] = True              # fresh: it still rolled its own end of round
    _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))
    money = w.f["money"]
    v = w.vouchers
    rate = 1 + sum(x["name"] == "To the Moon" for x in J if not x.get("deb"))
    cap = 20 if "Money Tree" in v else 10 if "Seed Money" in v else 5
    green = w.f["back"] == B_GREEN                         # Green Deck: no interest, $2 / hand, $1 / discard
    interest = rate * min(money // 5, cap) if money >= 5 and not green else 0
    is_boss = int(p.blind_index) == BLIND_BOSS
    invest = sum(t == "Investment Tag" for t, _ in w.tags) if is_boss else 0
    if invest:
        w.tags = [(t, h) for t, h in w.tags if t != "Investment Tag"]
    boss = BOSS_NAMES[int(p.boss)] if int(p.boss) >= 0 else None
    pay = 0 if saved else R.blind_pay(int(p.blind_index), boss if is_boss else None, int(p.stake))
    cashout = (pay + w.f["hands"] * (2 if green else 1) + (max(w.f["discards"], 0) if green else 0) + interest
               + int(R.ref_dollar_bonus(J, g)) + 25 * invest)
    if is_boss and w.f["back"] == B_ANAGLYPH:               # Anaglyph Deck: a Double Tag after a boss
        _add_tag(w, "Double Tag", 0)
    won = is_boss and int(p.ante) >= R.WIN_ANTE
    w.f.update(money=money + (cashout if won else 0), last_cashout=cashout, game_stage=STAGE_WON if won else STAGE_CASHOUT,
               unused_discards=w.f["unused_discards"] + max(w.f["discards"], 0), round_hand_bonus=0,
               boss_disabled=False)
    w.f.update(reroll_cost=5 - 2 * ("Reroll Surplus" in v) - 2 * ("Reroll Glut" in v), reroll_inc=0,
               reroll_d6=False)
    if is_boss:
        played = np.array(p.hand_played) if played is None else played   # this hand counted
        w.f["ox_hand"] = R.ox_hand(played)
        w.cards["card_played_ante"][:] = False
        for _, attrs in w.new_cards:                       # a DNA copy made by this hand forgets too
            if attrs and "card_played_ante" in attrs:
                attrs["card_played_ante"] = 0.0
        if not won:
            w.f["ante"] += 1
        w.f["blind_index"] = 0
    else:
        w.f["blind_index"] += 1
    astro_old = "Astronomer" in R.acting_names(J)
    for x in J:
        x["deb"] = _perished(x)                           # Crimson Heart lets go; a perished one stays
    _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))
    return J, cashout


def _check_new_ante(c, tag, w, p, q):
    """A boss was beaten: the next ante's boss (eligible, the least used), its tags (by min
    ante), its voucher (not redeemed, its first one redeemed)."""
    ante = w.f["ante"]
    b = int(q.boss)
    c.check(f"{tag}: the new boss is a boss", 0 <= b < len(BOSS_NAMES), b)
    if not 0 <= b < len(BOSS_NAMES):
        return
    nm = BOSS_NAMES[b]
    c.check(f"{tag}: the new boss fits ante {ante}", R.boss_eligible(nm, ante), nm, ante)
    used = w.used
    elig = [i for i, n in enumerate(BOSS_NAMES) if R.boss_eligible(n, ante)]
    c.check(f"{tag}: the new boss is one of the least used", used[b] == min(used[i] for i in elig),
            (nm, int(used[b])), min(int(used[i]) for i in elig))
    w.used = used.copy()
    w.used[b] += 1
    w.f["boss"] = b
    w.f["boss_rerolled"] = False
    for k, t in enumerate(q.blind_tags):
        tn = TAG_NAMES[int(t)] if 0 <= int(t) < len(TAG_NAMES) else None
        c.check(f"{tag}: tag {k} fits ante {ante}", tn is not None and R.TAGS[tn] <= ante, tn)
    vis = {h for h in range(N_HAND_TYPES) if h >= 3 or int(p.hand_played[h]) > 0}
    c.check(f"{tag}: the Orbital hands are visible hands", all(int(h) in vis for h in q.orbital_hands), q.orbital_hands)
    av = int(q.ante_voucher)
    vn = VOUCHER_NAMES[av] if 0 <= av < len(VOUCHER_NAMES) else None
    ok = vn is not None and vn not in w.vouchers and (vn not in R.VOUCHER_NEEDS or R.VOUCHER_NEEDS[vn] in w.vouchers)
    c.check(f"{tag}: the ante's voucher can be redeemed", ok or vn == "Blank", vn)


def _check_shop(c, tag, q, w, fresh=True):
    """A fresh shop (or a reroll): the unlocked slots hold a card of their kind, the locked
    ones are empty; the tags it used (D6, Uncommon / Rare, the edition tags, Voucher, Coupon)
    are gone, and the cards they made are free."""
    n_tag = Counter(t for t, _ in w.tags)
    left = Counter(TAG_NAMES[int(t)] for t in q.tags if int(t) >= 0)
    for t in left:
        c.check(f"{tag}: shop: tag {t} kept", left[t] <= n_tag[t], left[t], n_tag[t])
    used = n_tag - left
    for t in used:
        c.check(f"{tag}: shop: tag {t} is a shop tag",
                t in ("D6 Tag", "Uncommon Tag", "Rare Tag", "Voucher Tag", "Coupon Tag") or t in R.TAG_EDITION, t)
    coupon = used["Coupon Tag"] > 0
    size = int(q.shop_size)
    for i in range(MAX_SHOP_SLOTS):
        it, kd = int(q.shop_slots[i]), int(q.shop_kind[i])
        if i < size:
            ok = ((kd == IK_JOKER and 0 <= it < N_JOKERS) or (kd == IK_CARD and it == 0)
                  or (kd in (IK_TAROT, IK_PLANET) and 0 <= it < len(CONS_NAMES) and _KIND[it] == (kd - 1))
                  or (kd == IK_SPECTRAL and 0 <= it < len(CONS_NAMES) and _KIND[it] == CK_SPECTRAL
                      and CONS_NAMES[it] not in ("The Soul", "Black Hole")))
            c.check(f"{tag}: shop slot {i} holds a card of its kind", ok, it, kd)
            if kd == IK_SPECTRAL:
                c.check(f"{tag}: shop slot {i}: Spectral cards need the Ghost Deck", w.f["back"] == B_GHOST,
                        w.f["back"])
            if kd == IK_CARD:
                c.check(f"{tag}: shop slot {i}: playing cards need Magic Trick",
                        bool(w.vouchers & {"Magic Trick", "Illusion"}), w.vouchers)
            if kd == IK_JOKER and 0 <= it < N_JOKERS:
                c.check(f"{tag}: shop slot {i}: no Legendary", int(JOKER_RARITY[it]) != 4, NAME[it])
            _check_stickers(c, f"{tag}: shop slot {i}", NAME[it] if kd == IK_JOKER and 0 <= it < N_JOKERS else None,
                            int(q.shop_sticker[i]), w.f["stake"])
        else:
            c.check(f"{tag}: locked shop slot {i} empty", it == -1, it)
            npeq(c, f"{tag}: locked shop slot {i}: no stickers", q.shop_sticker[i], 0)
    free = [i for i in range(size) if int(q.shop_slots[i]) >= 0 and bool(q.shop_free[i])]
    if not coupon:
        # an Uncommon / Rare Tag makes a free joker; an edition tag puts its edition on a joker
        # without one (the tag's own joker too) and makes it free
        made = used["Uncommon Tag"] + used["Rare Tag"]
        eds = Counter(R.TAG_EDITION[t] for t in R.TAG_EDITION for _ in range(used[t]))
        n_ed = sum(eds.values())
        c.check(f"{tag}: shop: free cards = the cards the tags made", max(made, n_ed) <= len(free) <= made + n_ed,
                len(free), (made, n_ed))
        c.check(f"{tag}: shop: only jokers are free", all(int(q.shop_kind[i]) == IK_JOKER for i in free), free)
        have = Counter(int(q.shop_ed[i]) for i in free)
        c.check(f"{tag}: shop: each edition tag's joker has its edition", all(have[e] >= k for e, k in eds.items()),
                dict(have), dict(eds))
    w.tags = [(TAG_NAMES[int(t)], int(h)) for t, h in zip(q.tags, q.tag_hand) if int(t) >= 0]
    if fresh:
        c.check(f"{tag}: two packs", all(0 <= int(x) < len(PACK_NAMES) for x in q.pack_slots), q.pack_slots)
        n_v = 1 if w.f.get("_ante_voucher", -1) >= 0 else 0
        got_v = [int(x) for x in q.shop_vouchers if int(x) >= 0]
        c.eq(f"{tag}: the shop's vouchers", len(got_v), min(n_v + used["Voucher Tag"], 4))
        c.check(f"{tag}: no voucher twice", len(set(got_v)) == len(got_v), got_v)
        if coupon:
            c.check(f"{tag}: Coupon: every card and pack is free",
                    all(bool(q.shop_free[i]) for i in range(size) if int(q.shop_slots[i]) >= 0)
                    and all(bool(q.pack_free[i]) for i in range(2) if int(q.pack_slots[i]) >= 0))
        want_r = 0 if used["D6 Tag"] else w.f["reroll_cost"]
        w.f.update(reroll_cost=want_r, reroll_d6=bool(used["D6 Tag"]))


def _check_stickers(c, tag, name, stk, stake):
    """A shop / pack card's stickers: only a joker has them, and they fit the joker and the stake
    (ref_balatro.stickers_ok)."""
    if name is None:
        c.eq(f"{tag}: no stickers (not a joker)", stk, 0)
    else:
        c.check(f"{tag}: stickers fit {name} at stake {stake}",
                0 <= stk < 8 and R.stickers_ok(name, stk & STK_ETERNAL, stk & STK_PERISH, stk & STK_RENTAL, stake),
                stk)


def _enter_select(c, tag, p, q, J, w):
    """update_blind_select (env _settle at STAGE_SELECT): the immediate tags in order, then the
    new_blind_choice tags until one opens a pack."""
    if w.f["game_stage"] != STAGE_SELECT:
        return J
    left = []
    for t, h in w.tags:
        if t == "Top-up Tag":
            for _ in range(2):
                if len(J) < w_slots(J, w):
                    _new_joker(c, tag, w, J, q, len(J), 1, " (Top-up Tag)")
        elif t == "Skip Tag":
            w.f["money"] += 5 * w.f["skips"]
        elif t == "Garbage Tag":
            w.f["money"] += w.f["unused_discards"]
        elif t == "Handy Tag":
            w.f["money"] += int(np.array(p.hand_played).sum())
        elif t == "Economy Tag":
            w.f["money"] += min(40, max(0, w.f["money"]))
        elif t == "Orbital Tag":
            w.level[h] += 3
        else:
            left.append((t, h))
    w.tags = left
    left = []
    stopped = False
    rerolls = 0                                          # the Boss Tags before the first pack tag
    for t, h in w.tags:
        if stopped or (t != "Boss Tag" and t not in R.TAG_PACK):
            left.append((t, h))
            continue
        if t == "Boss Tag":
            rerolls += 1
        else:
            stopped = True
            pid = PACK_NAMES.index(R.TAG_PACK[t])
            w.f.update(game_stage=STAGE_PACK, pack_id=pid, pack_left=int(PACK_CHOOSE[pid]), pack_return=STAGE_SELECT)
            _check_pack(c, tag, p, q, w, J, pid)
            w.spawn(CK_TAROT, _eff_count(J, "Hallucination"), exact=False)
    if rerolls:
        _reroll_boss_ref(c, tag, w, q, free=True, n=rerolls)
    w.tags = left
    return J


def w_slots(J, w):
    neg = sum(1 for x in J if x["ed"] == 4)
    return min(5 + int(BACK_JOKER_SLOT[w.f["back"]]) + ("Antimatter" in w.vouchers) + neg, R.SLOT_CAP)


def _reroll_boss_ref(c, tag, w, q, free=False, n=1):
    """A new boss for this ante (the Boss Tag, Director's Cut, Retcon): read from q. n rerolls
    in one step (Boss Tags in a row): only the last boss shows, so the counts come from q."""
    b = int(q.boss)
    nm = BOSS_NAMES[b] if 0 <= b < len(BOSS_NAMES) else None
    ante = w.f["ante"]
    elig = [i for i, n in enumerate(BOSS_NAMES) if R.boss_eligible(n, ante)]
    c.check(f"{tag}: rerolled boss fits ante {ante}", nm is not None and R.boss_eligible(nm, ante), nm)
    if nm is not None and n == 1:
        c.check(f"{tag}: rerolled boss is one of the least used", w.used[b] == min(w.used[i] for i in elig),
                int(w.used[b]), min(int(w.used[i]) for i in elig))
        w.used = w.used.copy()
        w.used[b] += 1
        w.f["boss"] = b
    elif nm is not None:
        diff = np.asarray(q.bosses_used) - np.asarray(w.used)
        c.check(f"{tag}: {n} boss rerolls, each an eligible boss",
                (diff >= 0).all() and diff.sum() == n and diff[b] >= 1
                and all(diff[i] == 0 for i in range(len(diff)) if i not in elig), diff)
        w.used = np.asarray(q.bosses_used).copy()
        w.f["boss"] = b
    w.f["boss_rerolled"] = True


def _check_pack(c, tag, p, q, w, J, pid):
    """A pack just opened: its cards fit its kind; Arcana / Spectral packs deal a hand."""
    npeq(c, f"{tag}: OPEN: pack id", q.pack_id, pid)
    items = [int(x) for x in q.pack_items]
    size = int(PACK_SIZE[pid])
    c.check(f"{tag}: OPEN: {size} cards", all(x >= 0 for x in items[:size]) and all(x < 0 for x in items[size:]), items)
    kind = _PKIND[pid]
    for i in range(5):
        _check_stickers(c, f"{tag}: OPEN: card {i}", NAME[items[i]] if kind == PK_BUFFOON and items[i] >= 0 else None,
                        int(q.pack_sticker[i]), w.f["stake"])
    if kind in (PK_ARCANA, PK_CELESTIAL, PK_SPECTRAL):
        ok_kind = {PK_ARCANA: {CK_TAROT} | ({CK_SPECTRAL} if "Omen Globe" in w.vouchers else set()),
                   PK_CELESTIAL: {CK_PLANET}, PK_SPECTRAL: {CK_SPECTRAL}}[kind]
        soul = {CONS_ID["The Soul"], CONS_ID["Black Hole"]}
        c.check(f"{tag}: OPEN: the right kind", all(_KIND[x] in ok_kind or x in soul for x in items[:size]),
                [CONS_NAMES[x] for x in items[:size]])
        tele = kind == PK_CELESTIAL and "Telescope" in w.vouchers and int(np.array(p.hand_played).max()) > 0
        rest = items[1:size] if tele else items[:size]
        c.check(f"{tag}: OPEN: no repeats", len(set(rest)) == len(rest) or "Showman" in R.acting_names(J))
        if tele:
            want = int(PLANET_OF_HAND[int(np.argmax(np.array(p.hand_played)))])
            npeq(c, f"{tag}: Telescope: the first Planet is the most played hand's", items[0], want)
    if kind in (PK_ARCANA, PK_SPECTRAL):
        want_n = min(w_hand_size(J, w, q), int(np.asarray(q.card_live).sum()))
        npeq(c, f"{tag}: OPEN: an Arcana / Spectral pack deals a hand", int(np.asarray(q.hand_valid).sum()), want_n)
        # from the top of the deck, no shuffle: the cards after the ones p had dealt (a pack that
        # closed in this step put its hand back at the bottom)
        n = int(np.asarray(q.hand_valid).sum())
        hand = [int(x) for x in q.hand[:n]]
        npeq(c, f"{tag}: OPEN: the hand is the top of the deck", hand, [int(x) for x in q.deck[:n]])
        npeq(c, f"{tag}: OPEN: the hand's cards are dealt", int(q.deck_ptr), n)
        ptr, size = int(p.deck_ptr), int(p.deck_size)
        if ptr + n <= size:
            npeq(c, f"{tag}: OPEN: the next cards of the deck", hand, [int(x) for x in p.deck[ptr:ptr + n]])


def _scale(p):
    """The blind targets' scale: target_scale x the deck's ante_scaling (Plasma X2)."""
    return float(np.float32(p.target_scale) * np.float32(BACK_ANTE_SCALING[int(p.back)]))


def w_hand_size(J, w, q):
    return R.ref_budget(J, w.f["base_hand_size"], w.f["base_hands"], w.f["base_discards"],
                        w.f["round_hand_bonus"] - _manacle(q))[0]


def _manacle(q):
    return int(int(q.game_stage) in (STAGE_BLIND, STAGE_END, STAGE_LOST) and int(q.blind_index) == BLIND_BOSS
               and not bool(q.boss_disabled) and int(q.boss) == BOSS_NAMES.index("The Manacle"))


def _disable(c, tag, p, w, J):
    """blind.lua disable on the expected state (the boss is on in p)."""
    boss = boss_on(p)
    if boss is None or w.f["boss_disabled"]:
        return
    if boss == "The Water":
        w.f["discards"] += w.f["boss_sub"]
    if boss == "The Needle":                             # a negative take (0 base hands): down to 0
        w.f["hands"] = max(w.f["hands"] + w.f["boss_sub"], 0)
    w.f["blind_target"] = R.blind_chips(int(p.ante), 2, boss, _scale(p), disabled=True,
                                        stake=int(p.stake))
    astro_old = "Astronomer" in R.acting_names(J)
    for x in J:
        if x.get("deb") and not _perished(x):
            w.f["discards"] += R.D_SIZE.get(x["name"], 0)
            x["deb"] = False
    _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))
    w.f["boss_disabled"] = True


def _blind_start(c, tag, p, q, J, w):
    """NEXT at the blind select: the round starts (new_round, set_blind, setting_blind, the draw)."""
    is_boss = int(p.blind_index) == BLIND_BOSS
    boss = BOSS_NAMES[int(p.boss)] if is_boss and int(p.boss) >= 0 else None
    w.f.update(game_stage=STAGE_BLIND, round=w.f["round"] + 1, chips_scored=0, boss_disabled=False, boss_hand=-1,
               blind_target=R.blind_chips(int(p.ante), int(p.blind_index), boss, _scale(p),
                                          stake=int(p.stake)),
               discards_used=0)
    hs, hc, dc, js = R.ref_budget(J, w.f["base_hand_size"], w.f["base_hands"], w.f["base_discards"])
    w.f["hands"], w.f["discards"] = hc, dc
    # The Needle takes round_resets.hands - 1 (not at least 1: Troubadours can make it 0 or less)
    r_hands = w.f["base_hands"] + sum(R.H_PLAYS.get(x["name"], 0) for x in J if not x.get("deb"))
    w.f["boss_sub"] = dc if boss == "The Water" else r_hands - 1 if boss == "The Needle" else 0
    if boss == "The Water":
        w.f["discards"] = 0
    if boss == "The Needle":
        w.f["hands"] = hc - w.f["boss_sub"]
    # setting_blind, slot order (a joker being destroyed does not act, nor a copy of it)
    names = [x["name"] for x in J]
    eff = R.effective(J)
    sliced, buffer, burgle, riff, carto, marble = set(), 0, 0, 0, 0, 0
    n_held = len(J)
    slots = w_slots(J, w)
    for i, x in enumerate(J):
        t = eff[i]
        if t is None or i in sliced or t in sliced:
            continue
        n = names[t]
        if x["name"] == "Madness" and not is_boss:
            x["v"] += 0.5
            victims = [k for k in range(len(J)) if k != i and k not in sliced and not _eternal(J[k])]
            if victims:
                gone_names = Counter(names) - Counter(NAME[int(j)] for j in q.jokers if j >= 0)
                vic = [k for k in victims if gone_names[names[k]] > 0]
                c.check(f"{tag}: Madness destroyed a joker", bool(vic), names, [NAME[int(j)] for j in q.jokers if j >= 0])
                if vic:
                    sliced.add(vic[0])
        if n == "Burglar":
            burgle += 1
        if n == "Riff-raff":
            make = min(2, max(slots - (n_held + buffer), 0))
            riff += make
            buffer += make
        if n == "Cartomancer":
            carto += 1
        if x["name"] == "Ceremonial Dagger" and i + 1 < len(J) and i + 1 not in sliced and not _eternal(J[i + 1]):
            sliced.add(i + 1)
            x["v"] += 2 * J[i + 1]["sell"]
            buffer -= 1
        if n == "Marble Joker":
            marble += 1
    for k in sorted(sliced, reverse=True):
        y = J.pop(k)
        w.f["discards"] = max(w.f["discards"] - (0 if y.get("deb") else R.D_SIZE.get(y["name"], 0)), 0)
    if burgle:
        w.f["discards"] = 0                            # Burglar: before a Riff-raff Drunkard's +1
    # Riff-raff's jokers (random Commons): read from q after the survivors
    q_names = [NAME[int(j)] for j in q.jokers if j >= 0]
    for _ in range(riff):
        if len(J) < len(q_names):
            _new_joker(c, tag, w, J, q, len(J), 1, " (Riff-raff)")
    w.spawn(CK_TAROT, carto)
    free = int((~w.cards["card_live"]).sum())
    k_m = min(marble, 3, free)
    for _ in range(k_m):
        w.new_cards.append((False, dict(card_enh=R.STONE, card_ed=0, card_seal=0)))
    _bump(J, "Hologram", 0.25 * k_m)
    w.f["hands"] += 3 * burgle
    w.f["free_rerolls"] = sum(x["name"] == "Chaos the Clown" and not x.get("deb") for x in J)
    # Amber Acorn shuffles the jokers (read the order from q); Chicot disables the boss
    if boss == "Amber Acorn" and len(J) > 1:
        order = [NAME[int(j)] for j in q.jokers if j >= 0]
        c.check(f"{tag}: Amber Acorn: the same jokers", sorted(order) == sorted(x["name"] for x in J), order)
        if sorted(order) == sorted(x["name"] for x in J):
            pool = list(J)
            new = []
            for k, nm in enumerate(order):
                cand = [x for x in pool if x["name"] == nm and abs(x["v"] - float(q.joker_value[k])) < 1e-4] or \
                       [x for x in pool if x["name"] == nm]
                new.append(cand[0])
                pool.remove(cand[0])
            J[:] = new
    if boss is not None and "Chicot" in R.acting_names(J):
        p_on = p._replace(game_stage=STAGE_BLIND)
        _disable(c, tag, p_on, w, J)
    # Juggle Tags: +3 hand size this round
    juggle = sum(t == "Juggle Tag" for t, _ in w.tags)
    w.tags = [(t, h) for t, h in w.tags if t != "Juggle Tag"]
    w.f["round_hand_bonus"] += 3 * juggle
    # the draw: hand_size cards (+ Certificate's); at hand size 0 the draw ends the run first
    hs = w_hand_size(J, w, q)
    if hs <= 0:
        w.f["game_stage"] = STAGE_LOST
        npeq(c, f"{tag}: hand size 0: no opening hand", int(np.asarray(q.hand_valid).sum()), 0)
        return J
    n_cert = min(_eff_count(J, "Certificate"), 3, int((~w.cards["card_live"]).sum()) - k_m)
    for _ in range(n_cert):
        w.new_cards.append((True, dict(card_enh=0, card_ed=0)))
    _bump(J, "Hologram", 0.25 * n_cert)
    deck = int(w.cards["card_live"].sum()) + k_m
    want = min(hs, deck) + n_cert
    grow = _heart_pick(c, tag, p, q, J, w)
    want += min(grow, deck - min(hs, deck)) if hs + grow > 1 else 0
    npeq(c, f"{tag}: the opening hand", int(np.asarray(q.hand_valid).sum()), want)
    _check_draw_end(c, tag, q, J)
    return J


def _check_draw_end(c, tag, q, J):
    """drawn_to_hand: Cerulean Bell forces a card of the hand; Crimson Heart disables one joker."""
    boss = boss_on(q)
    if boss == "Cerulean Bell" and np.asarray(q.hand_valid).any():
        held = [int(h) for h, v in zip(q.hand, q.hand_valid) if v]
        c.check(f"{tag}: Cerulean Bell: the forced card is in hand", int(q.forced_card) in held, int(q.forced_card))
    deb = [bool(x) for x in q.joker_debuff[:len(J)]]
    if boss == "Crimson Heart" and J:
        # one joker, or none when no joker can be picked (the only ones left perished)
        c.eq(f"{tag}: Crimson Heart: the disabled jokers", sum(deb), sum(1 for x in J if x.get("deb") and not _perished(x)))
    elif boss != "Crimson Heart":
        c.check(f"{tag}: no joker disabled", not any(bool(x) for x in q.joker_debuff), q.joker_debuff[:8])


def _heart_pick(c, tag, p, q, J, w):
    """Crimson Heart after the first draw / a PLAY: a random joker that is not debuffed (not the
    one before, nor a perished one; with one joker: that one, but a perished one stays as it is),
    read from q; its discards go with it."""
    if boss_on(q) != "Crimson Heart" or not J:
        return 0
    new = [k for k in range(len(J)) if bool(q.joker_debuff[k])]
    old = [k for k, x in enumerate(J) if x.get("deb") and not _perished(x)]
    cand = [k for k, x in enumerate(J) if not _perished(x) and (len(J) < 2 or not x.get("deb"))]
    c.eq(f"{tag}: Crimson Heart: one joker disabled", len(new), 1 if cand else 0)
    c.check(f"{tag}: Crimson Heart: not the same joker again, nor a perished one", all(k in cand for k in new),
            new, cand)
    back = sum(R.D_SIZE.get(J[k]["name"], 0) for k in old) - sum(R.D_SIZE.get(J[k]["name"], 0) for k in new)
    w.f["discards"] = max(w.f["discards"] + back, 0)
    h = lambda x: R.H_SIZE.get(x["name"], 0) + (int(x["v"]) if x["name"] == "Turtle Bean" else 0)
    grow = max(sum(h(J[k]) for k in old), 0) + max(-sum(h(J[k]) for k in new), 0)
    astro_old = "Astronomer" in R.acting_names(J)
    for k, x in enumerate(J):
        x["deb"] = k in new or _perished(x)
    _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))   # set_debuff: Astronomer's set_cost
    return grow


def check_step(c, tag, p, a, q, reward, out=None):
    """Compare post-state q against the reference model of (p, a).
    p, q are host states; a is a host balatroAction; out the host PlayOut of
    (p, a.select) for a PLAY. -> the effective verb."""
    verb, t, d = int(a.verb), int(a.target), int(a.dest)
    sel = np.asarray(a.select).astype(bool)
    stage = int(p.game_stage)
    if verb == V_DISCARD and int(p.discards) <= 0:
        verb = V_PLAY                                       # DISCARD at 0 left is a PLAY
    legal = 0 <= verb < N_VERBS and LEGAL_NP[min(max(stage, 0), STAGES - 1), verb]
    if verb in (V_PLAY, V_DISCARD) and not played_lanes(p, sel):
        legal = False                                       # no card selected: the buttons stay off

    errs = invariant_errors(q)
    c.check(f"{tag}: invariants", not errs, errs)
    if not legal:
        same = all(np.array_equal(getattr(p, f), getattr(q, f)) for f in p._fields)
        c.check(f"{tag}: illegal verb {int(a.verb)} in stage {stage} is a no-op", same)
        npeq(c, f"{tag}: no-op reward", reward, 0)
        return None

    J = records(p)
    n = len(J)
    w = Want(p)
    w.f["_ante_voucher"] = int(p.ante_voucher)
    played, in_round = np.array(p.hand_played), np.array(p.hand_round)
    want_reward = 0
    observe_state = set()                                   # To Do slots whose new hand is random
    boss = boss_on(p)
    m = _mods(J)
    shop_fresh = None                                       # a new shop opened: check it
    check_debuffs = True

    if verb in (V_PLAY, V_DISCARD):
        lanes = played_lanes(p, sel)
        hook = []
        if verb == V_PLAY and boss == "The Hook":
            u = _hand_u(p, K_HOOK)
            held_l = [l for l in range(MAX_HAND_SIZE) if p.hand_valid[l] and l not in lanes]
            hook = sorted(sorted(held_l, key=lambda l: (u[l], l))[:2])
        cards = [card_of(p, p.hand[l], boss, m) for l in lanes]
        held_cards = [card_of(p, p.hand[l], boss, m) for l in range(MAX_HAND_SIZE)
                      if p.hand_valid[l] and l not in lanes and l not in hook]
        disc_l = hook if verb == V_PLAY else lanes
        if disc_l or verb == V_DISCARD:
            dcards = [card_of(p, p.hand[l], boss, m) for l in disc_l]
            g = ref_g(p)
            money, gone, extra = R.ref_discard(J, dcards, g, hook=verb == V_PLAY)
            if extra["burnt"]:
                h, k = extra["burnt"]
                w.level[HAND_NAMES.index(h)] += k
            w.spawn(CK_TAROT, extra["purple"])
            if extra["trading"]:
                w.cards["card_live"][dcards[0].i] = False
                _bump(J, "Canio", 1.0 * R.is_face(dcards[0], m))
                _bump(J, "Glass Joker", 0.75 * (dcards[0].enh == R.GLASS))
            J = _drop(J, gone)
            w.f["money"] += money
    if verb == V_PLAY:
        hname = R.ref_hand(cards, m)[0]
        hi = HAND_NAMES.index(hname)
        tooth = len(cards) if boss == "The Tooth" else 0
        arm = boss == "The Arm" and w.level[hi] > 1
        ox = boss == "The Ox" and hi == int(p.ox_hand)
        debuffed = (boss == "The Psychic" and len(cards) < 5) or (boss == "The Eye" and int(p.hand_round[hi]) > 0) \
            or (boss == "The Mouth" and int(p.boss_hand) >= 0 and int(p.boss_hand) != hi)
        w.f["money"] -= tooth
        if ox:
            w.f["money"] = 0
        if arm:
            w.level[hi] -= 1
        g = ref_g(p)
        g.update(money=w.f["money"], levels={HAND_NAMES[k]: int(w.level[k]) for k in range(N_HAND_TYPES)},
                 hand_debuffed=debuffed, flint=boss == "The Flint", boss_trigger=arm or ox,
                 discards=max(int(p.discards), 0), card_room=int((~w.cards["card_live"]).sum()), boss=boss)
        if "Observatory" in w.vouchers:
            obs = Counter(HAND_NAMES[_PARAM[x]] for x in w.cons if _KIND[x] == CK_PLANET)
            g["observatory"] = obs
        R_ = out.rolls.shape[1]
        res = R.ref_play(J, cards, [(l, card_of(p, p.hand[l], boss, m)) for l in range(MAX_HAND_SIZE)
                                    if p.hand_valid[l] and l not in lanes and l not in hook], g,
                         rolls=lambda k, r, j: bool(out.rolls[k, min(r, R_ - 1), j]),
                         hrolls=lambda l, r, j: bool(out.hrolls[l, min(r, out.hrolls.shape[1] - 1), j]),
                         rnd=lambda j: float(out.rnd[j]),
                         lucky=lambda k, r, wv: bool(out.lucky_rolls[k, min(r, R_ - 1), wv]),
                         glass=lambda k: bool(out.glass_roll[k]),
                         space=lambda j: bool(out.space_roll[j]))
        got = int(q.chips_scored) - int(p.chips_scored)
        capped = min(int(p.chips_scored) + min(res["score"], SCORE_CAP), INT32_MAX) - int(p.chips_scored)
        if capped == res["score"]:
            ok = exact(got, res["score"], res["chips"] * res["mult"])
        else:                                                # past int32: the score and the sum stop
            ok = got == capped
        c.check(f"{tag}: PLAY {res['hand']} scores {res['score']} ({res['chips']} x {res['mult']})",
                ok, got, capped)
        total = int(p.chips_scored) + (got if ok else capped)
        played = np.array([res["played"][h] for h in HAND_NAMES])
        in_round = np.array([res["round"][h] for h in HAND_NAMES])
        hi = HAND_NAMES.index(res["hand"])
        w.level[hi] += res["level_up"]
        w.f["last_hand"] = hi
        if boss == "The Mouth" and int(p.boss_hand) < 0:
            w.f["boss_hand"] = hi
        # the mark (Midas / Vampire: set_ability takes it); in The Pillar's own blind only a mark
        # from before it (a new one debuffs nothing there: the flags were set at the blind's start)
        old_mark = lambda i: bool(p.card_played_ante[i]) or boss != "The Pillar"
        for k, card in enumerate(res["cards"]):
            w.cards["card_enh"][card.i] = card.enh
            w.cards["card_bonus"][card.i] = card.bonus
            w.cards["card_played_ante"][card.i] = k not in res["reset"] and old_mark(card.i)
        first = int(np.asarray(p.hand_round).sum()) == 0
        after = res["cards"]                                 # the cards after BEFORE (Midas, Vampire)
        sixth = ("Sixth Sense" in R.acting_names(J) and first and len(cards) == 1 and not debuffed
                 and R.rank(after[0]) == R.SIX)
        dead = {k for k in res["broken"]} | ({0} if sixth else set())
        for k in dead:
            w.cards["card_live"][cards[k].i] = False
        _bump(J, "Glass Joker", 0.75 * sum(after[k].enh == R.GLASS for k in dead))
        _bump(J, "Canio", 1.0 * sum(R.is_face(after[k], m) for k in dead))
        astro = "Astronomer" in R.acting_names(J)
        w.spawn_seq([CK_TAROT if s == "tarot" else CK_SPECTRAL for s in res["spawns"]]
                    + ([CK_SPECTRAL] if sixth else []), astro=astro)
        # DNA: the 1st hand of the round is one card -> a copy into deck and hand (ref_play made it a
        # held card of this hand and gave Hologram its X0.25)
        if first and len(cards) == 1 and not debuffed:
            for src in res["dna"]:                              # each DNA copies the card as it saw it
                w.new_cards.append((True, dict(card_rank=src.r, card_suit=src.s, card_enh=src.enh,
                                               card_ed=src.ed, card_seal=src.seal, card_bonus=cards[0].bonus,
                                               card_played_ante=float(src.mark and old_mark(cards[0].i)))))
                held_cards.append(R.Card(-1, src.r, src.s, src.enh, src.ed, src.seal, deb=src.deb))
        J = _drop(J, res["gone"])
        w.f.update(chips_scored=total, hands=max(int(p.hands) - 1, 0))
        w.f["money"] += int(res["money"])
        held_after = len(held_cards)
        deck_left = int(p.deck_size) - int(p.deck_ptr)
        stuck = held_after == 0 and deck_left == 0
        over = total >= int(p.blind_target) or int(p.hands) - 1 <= 0
        if not over and held_after == 0 and w_hand_size(J, w, q) <= 0:
            w.f["game_stage"] = STAGE_LOST                      # the draw at hand size 0: GAME_OVER
        elif (over or stuck) and _end_window(p, J, w, held_cards, total):
            w.f["game_stage"] = STAGE_END                       # the joker order can still change the end
        elif over or stuck:
            todo = {id(x) for x in J if x["name"] == "To Do List" and not x.get("deb")}   # a debuffed one misses it
            J, want_reward, shop_fresh = _end_round(c, tag, p, q, J, w, held_cards, total, played)
            if w.f["game_stage"] != STAGE_LOST:
                observe_state = {i for i, x in enumerate(J) if id(x) in todo}
        else:
            _heart_pick(c, tag, p, q, J, w)
            _check_draw_end(c, tag, q, J)
    elif verb == V_DISCARD:
        w.f.update(discards=w.f["discards"] - 1, discards_used=int(p.discards_used) + 1)
        held_after = int(np.asarray(p.hand_valid).sum()) - len(lanes)
        deck_left = int(p.deck_size) - int(p.deck_ptr)
        if held_after == 0 and w_hand_size(J, w, q) <= 0:
            w.f["game_stage"] = STAGE_LOST                      # the draw at hand size 0: GAME_OVER
        elif held_after == 0 and deck_left == 0:
            J, want_reward, shop_fresh = _end_round(c, tag, p, q, J, w, [], int(p.chips_scored))
        else:
            _check_draw_end(c, tag, q, J)
    elif verb == V_BUY:
        item = int(p.shop_slots[t]) if 0 <= t < MAX_SHOP_SLOTS else -1
        kind = int(p.shop_kind[t]) if 0 <= t < MAX_SHOP_SLOTS else 0
        ed = int(p.shop_ed[t]) if 0 <= t < MAX_SHOP_SLOTS else 0
        astro = "Astronomer" in R.acting_names(J)
        stk = int(p.shop_sticker[t]) if 0 <= t < MAX_SHOP_SLOTS else 0
        if kind == IK_JOKER:
            cost = joker_cost_ref(NAME[item], ed, w.disc, stk) if item >= 0 else 0
            room = n < int(p.joker_slots) + (ed == 4 and int(p.joker_slots) < R.SLOT_CAP)
        elif kind == IK_CARD:
            cost = R.cost(1 + int(ED_COST[ed]), w.disc)
            room = not np.asarray(p.card_live).all()
        else:
            cost = cons_cost_ref(item, 0, w.disc, astro) if item >= 0 else 0
            room = w.room() > 0
        # Buy and Use: a Tarot / Planet the full slots cannot take is used at once, if the shop
        # allows its use (can_buy_and_use: can_use_consumeable from outside the slots, no hand)
        buy_use = kind in (IK_TAROT, IK_PLANET, IK_SPECTRAL) and not room and item >= 0
        if buy_use:
            room = _can_use(p, J, w, item, np.zeros(MAX_HAND_SIZE, bool), False)
        if 0 <= t < MAX_SHOP_SLOTS and bool(p.shop_free[t]):
            cost = 0
        floor = -20 * R.acting_names(J).count("Credit Card")
        if 0 <= t < int(p.shop_size) and item >= 0 and (cost <= 0 or int(p.money) - cost >= floor) and room:
            if kind == IK_JOKER:
                nm = NAME[item]
                st = int(sum(p.hand_played)) if nm == "Loyalty Card" else 0
                J.append(new_record(nm, float(R.SEED.get(nm, JOKERS.VALUE[item])), st, ed, w.disc, stk))
                if nm == "To Do List":
                    observe_state = {len(J) - 1}
                w.f.update(free_rerolls=int(p.free_rerolls) + (nm == "Chaos the Clown"),
                           discards=int(p.discards) + R.D_SIZE.get(nm, 0))
            elif kind == IK_CARD:
                w.new_cards.append((False, dict(card_rank=int(p.shop_rank[t]), card_suit=int(p.shop_suit[t]),
                                                card_enh=int(p.shop_enh[t]), card_ed=ed, card_seal=0)))
                _bump(J, "Hologram", 0.25)
            elif not buy_use:
                w.give(item, astro)
            w.f["money"] = int(p.money) - cost
            if buy_use:                                     # paid first: The Hermit sees the money after
                J = _apply_cons(c, tag, p, q, J, w, item, np.zeros(MAX_HAND_SIZE, bool))
            npeq(c, f"{tag}: BUY empties the shop slot", q.shop_slots[t], -1)
            npeq(c, f"{tag}: BUY: the slot's stickers go with the card", q.shop_sticker[t], 0)
    elif verb == V_REDEEM:
        v = int(p.shop_vouchers[t]) if 0 <= t < 4 else -1
        cost = R.cost(10, w.disc)
        floor = -20 * R.acting_names(J).count("Credit Card")
        if v >= 0 and int(p.money) - cost >= floor:
            vn = VOUCHER_NAMES[v]
            old_disc = w.disc
            w.f["money"] -= cost
            w.vouchers.add(vn)
            if vn in ("Overstock", "Overstock Plus"):
                w.f["shop_size"] = min(w.f["shop_size"] + 1, 4)
            if vn in ("Reroll Surplus", "Reroll Glut"):
                w.f["reroll_cost"] = max(w.f["reroll_cost"] - 2, 0)
            if vn == "Crystal Ball":
                w.f["consumable_slots"] += 1
            if vn in ("Grabber", "Nacho Tong"):
                w.f["base_hands"] += 1
            if vn in ("Wasteful", "Recyclomancy"):
                w.f["base_discards"] += 1
            if vn in ("Paint Brush", "Palette"):
                w.f["base_hand_size"] += 1
            if vn in ("Hieroglyph", "Petroglyph"):
                w.f["ante"] -= 1
                w.f["base_hands" if vn == "Hieroglyph" else "base_discards"] -= 1
            if w.disc != old_disc:
                for x in J:
                    x["sell"] += joker_sell(x["name"], x["ed"], w.disc, x["stk"]) - joker_sell(x["name"], x["ed"], old_disc, x["stk"])
                astro = "Astronomer" in R.acting_names(J)
                w.cons_sell = [s + _sell_of(cons_cost_ref(cid, e, w.disc, astro))
                               - _sell_of(cons_cost_ref(cid, e, old_disc, astro))
                               for cid, e, s in zip(w.cons, w.cons_ed, w.cons_sell)]
            npeq(c, f"{tag}: REDEEM: the voucher leaves the shop", q.shop_vouchers[t], -1)
            npeq(c, f"{tag}: REDEEM: no ante voucher after a redeem", q.ante_voucher, -1)
    elif verb == V_SELL:
        if 0 <= t < MAX_JOKERS and int(p.jokers[t]) >= 0 and not _eternal(J[t]):     # can_sell_card
            x0 = J[t]
            nm = x0["name"]
            for i, x in enumerate(J):
                if i != t and x["name"] == "Campfire" and not x.get("deb"):
                    x["v"] += 0.25
            w.f["money"] += int(p.joker_sell_value[t])
            acts = not x0.get("deb")
            tgt = R.effective(J)[t]                         # a copier: its target's selling_self
            en = J[tgt]["name"] if tgt is not None else None
            if acts and en == "Luchador":
                _disable(c, tag, p, w, J)
            if acts and en == "Diet Cola":
                _add_tag(w, "Double Tag", 0)
            dup = None
            if acts and nm == "Invisible Joker" and x0["v"] >= 2 and n > 1 and n <= int(p.joker_slots):   # else "No room!"
                q_names = [NAME[int(j)] for j in q.jokers if j >= 0]
                if len(q_names) == n:
                    dup = q_names[-1]
                    c.check(f"{tag}: Invisible Joker copies another joker", dup in [y["name"] for i, y in enumerate(J) if i != t],
                            dup, [y["name"] for y in J])
            if not x0.get("deb"):
                w.f["discards"] = max(w.f["discards"] - R.D_SIZE.get(nm, 0), 0)
            w.f["free_rerolls"] = max(w.f["free_rerolls"] - (nm == "Chaos the Clown" and not _perished(x0)), 0)
            astro_old = "Astronomer" in R.acting_names(J)
            J = _drop(J, {t})
            if dup is not None:
                # the copy (copy_card): the stickers and rounds left too; read which one from q
                src_c = [y for y in J if y["name"] == dup]
                k = len(J)
                invis = dup == "Invisible Joker"                  # its copy starts at 0 rounds (invis_rounds = 0)
                src = ([y for y in src_c if y["stk"] == int(q.joker_sticker[k]) and y["left"] == int(q.joker_perish[k])
                        and (invis or abs(y["v"] - float(q.joker_value[k])) < 1e-4)] or src_c)[0]
                ced = 0 if src["ed"] == 4 else src["ed"]
                extra = src["sell"] - joker_sell(dup, src["ed"], w.disc, src["stk"])
                J.append(dict(src, ed=ced, sell=joker_sell(dup, ced, w.disc, src["stk"]) + extra, deb=_perished(src),
                              v=0.0 if invis else src["v"]))
                if not _perished(src):                      # a perished copy adds nothing (env.copy_joker)
                    w.f["free_rerolls"] += dup == "Chaos the Clown"
                    w.f["discards"] += R.D_SIZE.get(dup, 0)
            _reprice_cons(w, astro_old, "Astronomer" in R.acting_names(J))
            if boss == "Verdant Leaf":
                _disable(c, tag, p, w, J)
            if w.f["boss_disabled"] and boss is not None and int(p.chips_scored) >= w.f["blind_target"]:
                held_cards = [card_of(p, p.hand[l]) for l in range(MAX_HAND_SIZE) if p.hand_valid[l]]
                J, want_reward, shop_fresh = _end_round(c, tag, p._replace(boss_disabled=True), q, J, w, held_cards,
                                                        int(p.chips_scored))
    elif verb == V_REROLL and stage == STAGE_SHOP:
        free = int(p.free_rerolls) > 0
        cost = 0 if free else int(p.reroll_cost)
        if free or cost <= 0 or int(p.money) - cost >= -20 * R.acting_names(J).count("Credit Card"):
            v = w.vouchers                                  # calculate_reroll_cost: (D6 ? 0 : base) + paid ones
            base = 0 if bool(p.reroll_d6) else 5 - 2 * ("Reroll Surplus" in v) - 2 * ("Reroll Glut" in v)
            w.f.update(money=int(p.money) - cost, free_rerolls=int(p.free_rerolls) - free,
                       reroll_cost=int(p.reroll_cost) if free else base + int(p.reroll_inc) + 1,
                       reroll_inc=int(p.reroll_inc) + (not free))
            _bump(J, "Flash Card", 2)
            _check_shop(c, tag, q, w, fresh=False)
    elif verb == V_REROLL:                                  # the blind select: the boss
        vs = w.vouchers
        can = ("Retcon" in vs or ("Director's Cut" in vs and not bool(p.boss_rerolled))) \
            and int(p.money) - 10 >= -20 * R.acting_names(J).count("Credit Card")
        if can:
            w.f["money"] -= 10
            _reroll_boss_ref(c, tag, w, q)
    elif verb == V_NEXT and stage == STAGE_SHOP:
        # Perkeo (copies too): a Negative copy of a random held consumable each
        k = _eff_count(J, "Perkeo")
        for i in range(k):
            if not w.cons:
                break
            got = int(q.consumables[len(w.cons)]) if len(w.cons) < MAX_CONSUMABLES else -1
            c.check(f"{tag}: Perkeo copies a held consumable", got in w.cons, got, w.cons)
            if got in w.cons:
                # the copy keeps the source's extra sell value (Gift Card); two of the same card
                # may differ there, so the copied one is read from q
                astro = "Astronomer" in R.acting_names(J)
                base = _sell_of(cons_cost_ref(got, 4, w.disc, astro))
                opts = [base + w.cons_sell[j] - _sell_of(cons_cost_ref(got, w.cons_ed[j], w.disc, astro))
                        for j in range(len(w.cons)) if w.cons[j] == got]
                seen = int(q.cons_sell[len(w.cons)])
                w.cons.append(got)
                w.cons_ed.append(4)
                w.cons_sell.append(seen if seen in opts else opts[0])
        w.f["game_stage"] = STAGE_SELECT
    elif verb == V_NEXT and stage == STAGE_END:             # the end of the round, with this joker order
        held_cards = [card_of(p, p.hand[l], boss, m) for l in range(MAX_HAND_SIZE) if p.hand_valid[l]]
        todo = {id(x) for x in J if x["name"] == "To Do List" and not x.get("deb")}
        J, want_reward, shop_fresh = _end_round(c, tag, p, q, J, w, held_cards, int(p.chips_scored))
        if w.f["game_stage"] != STAGE_LOST:
            observe_state = {i for i, x in enumerate(J) if id(x) in todo}
    elif verb == V_NEXT and stage == STAGE_CASHOUT:         # Cash Out: the $, a shuffled deck, the shop
        w.f.update(money=w.f["money"] + int(p.last_cashout), game_stage=STAGE_SHOP)
        npeq(c, f"{tag}: Cash Out: the whole deck, shuffled", int(q.deck_ptr), 0)
        shop_fresh = True
    elif verb == V_NEXT:                                    # the blind select: play the blind
        J = _blind_start(c, tag, p, q, J, w)
        in_round = np.zeros(N_HAND_TYPES, np.int64)
    elif verb == V_MOVE_JOKER:                              # INSERT slot t at slot d
        if 0 <= t < n and 0 <= d < n and t != d:
            J.insert(d, J.pop(t))
    elif verb == V_MOVE_CARD:                               # INSERT lane t at lane d
        hand, valid, down = list(p.hand), list(p.hand_valid), list(p.hand_down)
        size = max(int(p.hand_size), max([l for l in range(MAX_HAND_SIZE) if valid[l]] + [-1]) + 1)
        if 0 <= t < size and 0 <= d < size and t != d and valid[t]:
            for arr in (hand, valid, down):
                arr.insert(d, arr.pop(t))
        npeq(c, f"{tag}: MOVE_CARD hand", q.hand, np.array(hand))
        npeq(c, f"{tag}: MOVE_CARD valid", q.hand_valid, np.array(valid))
        npeq(c, f"{tag}: MOVE_CARD face down", q.hand_down, np.array(down))
    elif verb == V_USE:
        cons = [int(x) for x in p.consumables if x >= 0]
        cid = cons[t] if 0 <= t < len(cons) else -1
        if cid >= 0 and _can_use(p, J, w, cid, sel, True):
            w.extra_room = int(w.cons_ed[t] == 4)
            w.drop_cons(t)
            J = _apply_cons(c, tag, p, q, J, w, cid, sel)
            w.extra_room = 0
            _redraw_ref(c, tag, p, J, w, cid, sel)
            if "Chicot" in R.acting_names(J) and "Chicot" not in R.acting_names(records(p)):
                _disable(c, tag, p, w, J)
    elif verb == V_SELL_CONS:
        cons = [int(x) for x in p.consumables if x >= 0]
        if 0 <= t < len(cons):
            w.f["money"] += w.cons_sell[t]                   # $1, + $1 per Gift Card round
            w.drop_cons(t)
            _bump(J, "Campfire", 0.25)                      # every card sold counts
    elif verb == V_OPEN:
        pid = int(p.pack_slots[t]) if 0 <= t < 2 else -1
        if pid >= 0:
            astro = "Astronomer" in R.acting_names(J)
            cost = 0 if (bool(p.pack_free[t]) or (_PKIND[pid] == PK_CELESTIAL and astro)) else R.cost(int(PACK_COST[pid]), w.disc)
            if cost <= 0 or int(p.money) - cost >= -20 * R.acting_names(J).count("Credit Card"):
                w.f.update(money=int(p.money) - cost, game_stage=STAGE_PACK, pack_id=pid,
                           pack_left=int(PACK_CHOOSE[pid]), pack_return=STAGE_SHOP)
                npeq(c, f"{tag}: OPEN: the pack slot is used", q.pack_slots[t], -1)
                _check_pack(c, tag, p, q, w, J, pid)
                w.spawn(CK_TAROT, _eff_count(J, "Hallucination"), exact=False)
    elif verb == V_PICK:
        pid = int(p.pack_id)
        item = int(p.pack_items[t]) if 0 <= t < 5 else -1
        kind = _PKIND[pid] if pid >= 0 else -1
        ok = item >= 0 and int(p.pack_left) > 0
        cons_k = kind in (PK_ARCANA, PK_CELESTIAL, PK_SPECTRAL)
        if ok and cons_k:
            ok = _can_use(p, J, w, item, sel, False)
        elif ok and kind == PK_BUFFOON:
            ed = int(p.pack_ed[t])
            ok = n < int(p.joker_slots) + (ed == 4 and int(p.joker_slots) < R.SLOT_CAP)
        elif ok and kind == PK_STANDARD:
            ok = not np.asarray(p.card_live).all()
        if ok:
            if cons_k:
                J = _apply_cons(c, tag, p, q, J, w, item, sel)
            elif kind == PK_BUFFOON:
                ed = int(p.pack_ed[t])
                nm = NAME[item]
                st = int(sum(p.hand_played)) if nm == "Loyalty Card" else 0
                J.append(new_record(nm, float(R.SEED.get(nm, JOKERS.VALUE[item])), st, ed, w.disc,
                                    int(p.pack_sticker[t])))
                if nm == "To Do List":
                    observe_state = {len(J) - 1}
                w.f.update(free_rerolls=int(p.free_rerolls) + (nm == "Chaos the Clown"),
                           discards=int(p.discards) + R.D_SIZE.get(nm, 0))
            else:
                w.new_cards.append((False, dict(card_rank=int(p.pack_rank[t]), card_suit=int(p.pack_suit[t]),
                                                card_enh=int(p.pack_enh[t]), card_ed=int(p.pack_ed[t]),
                                                card_seal=int(p.pack_seal[t]))))
                _bump(J, "Hologram", 0.25)
            left = int(p.pack_left) - 1
            w.f["pack_left"] = max(left, 0)
            if left <= 0:
                _close(c, tag, p, q, w)
            else:
                npeq(c, f"{tag}: PICK: the card is taken", q.pack_items[t], -1)
    elif verb == V_SKIP and stage == STAGE_PACK:
        _bump(J, "Red Card", 3)
        _close(c, tag, p, q, w, skip=True)
    elif verb == V_SKIP:                                    # the blind select: skip it
        if int(p.blind_index) < BLIND_BOSS:
            b = int(p.blind_index)
            w.f["skips"] += 1
            _add_tag(w, TAG_NAMES[int(p.blind_tags[b])], int(p.orbital_hands[b]))
            w.f["blind_index"] += 1

    # the blind select: the immediate tags, then the new_blind_choice ones (env _settle)
    J = _enter_select(c, tag, p, q, J, w)
    if shop_fresh:
        _check_shop(c, tag, q, w, fresh=True)
    if w.f["game_stage"] != STAGE_PACK:
        w.f.update(pack_id=-1, pack_left=0)
    # To Do List: a new hand is random; it must be a visible hand (a secret one once played), and
    # never the old one at a round end
    qs = [int(x) for x in q.joker_state]
    for i in observe_state:
        if i < len(J):
            c.check(f"{tag}: To Do List hand visible", 0 <= qs[i] < N_HAND_TYPES
                    and R.HAND_ORDER[qs[i]] in R.visible_hands(q.hand_played), qs[i])
            if verb == V_PLAY or (verb == V_NEXT and stage == STAGE_END):
                c.check(f"{tag}: To Do List hand changes", qs[i] != J[i]["st"], qs[i], J[i]["st"])
            J[i]["st"] = qs[i]
    k = len(J)
    npeq(c, f"{tag}: jokers", q.jokers, np.array([JOKER_ID[x["name"]] for x in J] + [-1] * (MAX_JOKERS - k), np.int32))
    vals = np.array([x["v"] for x in J] + [0.0] * (MAX_JOKERS - k))
    c.check(f"{tag}: joker values", np.allclose(q.joker_value, vals, rtol=1e-5, atol=1e-4),
            q.joker_value[:8], vals[:8])
    npeq(c, f"{tag}: sell values", q.joker_sell_value, np.array([x["sell"] for x in J] + [0] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: joker states", q.joker_state, np.array([x["st"] for x in J] + [0] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: joker editions", q.joker_ed, np.array([x["ed"] for x in J] + [0] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: joker debuffs", q.joker_debuff,
         np.array([bool(x.get("deb")) and not _perished(x) for x in J] + [False] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: joker stickers", q.joker_sticker, np.array([x["stk"] for x in J] + [0] * (MAX_JOKERS - k)))
    npeq(c, f"{tag}: Perishable rounds left", q.joker_perish, np.array([x["left"] for x in J] + [0] * (MAX_JOKERS - k)))
    manacle = _manacle(q)
    hs, hc, dc, js = R.ref_budget(J, w.f["base_hand_size"], w.f["base_hands"], w.f["base_discards"],
                                  w.f["round_hand_bonus"] - manacle,
                                  ("Antimatter" in w.vouchers) + int(BACK_JOKER_SLOT[w.f["back"]]))
    w.f.update(hand_size=hs, handcount=hc, discardcount=dc, joker_slots=js)
    w.f.pop("_ante_voucher", None)
    for f, v in w.f.items():
        npeq(c, f"{tag}: {f}", getattr(q, f), v)
    npeq(c, f"{tag}: hand_played", q.hand_played, played)
    npeq(c, f"{tag}: hand_round", q.hand_round, in_round)
    npeq(c, f"{tag}: hand_level", q.hand_level, w.level)
    npeq(c, f"{tag}: planets_used", q.planets_used, w.planets_used)
    npeq(c, f"{tag}: reward", reward, want_reward)
    npeq(c, f"{tag}: tags", [TAG_NAMES[int(x)] for x in q.tags if int(x) >= 0], [x for x, _ in w.tags])
    npeq(c, f"{tag}: vouchers", sorted(vouchers_of(q)), sorted(w.vouchers))
    npeq(c, f"{tag}: bosses used", q.bosses_used, w.used)
    _check_cons(c, tag, w, q)
    _check_cards(c, tag, w, p, q)
    return verb


def _redraw_ref(c, tag, p, J, w, cid, sel):
    """The Hanged Man / Immolate emptied the hand in a round with cards in the deck: the env
    draws a new hand (game.lua DRAW_TO_HAND). Before any play or discard it is the first
    hand: Certificate adds its card (first_hand_drawn)."""
    if int(p.game_stage) != STAGE_BLIND or _USE[cid] not in (U_HANGED, U_IMMOLATE):
        return
    held = int(np.asarray(p.hand_valid).sum())
    gone = len(_sel(p, sel, cid)[0]) if _USE[cid] == U_HANGED else min(_PARAM[cid], held)
    if held - gone > 0 or int(p.deck_size) - int(p.deck_ptr) <= 0 or w_hand_size(J, w, p) <= 0:
        return
    if int(np.asarray(p.hand_round).sum()) == 0 and int(p.discards_used) == 0:
        n_cert = min(_eff_count(J, "Certificate"), 3, int((~w.cards["card_live"]).sum()))
        for _ in range(n_cert):
            w.new_cards.append((True, dict(card_enh=0, card_ed=0)))
        _bump(J, "Hologram", 0.25 * n_cert)


def _add_tag(w, tag, hand):
    """add_tag: each held Double Tag gives a copy (of a tag that is not a Double Tag) and goes."""
    doubles = sum(t == "Double Tag" for t, _ in w.tags)
    if tag != "Double Tag" and doubles:
        w.tags = [(t, h) for t, h in w.tags if t != "Double Tag"]
        n = 1 + doubles
    else:
        n = 1
    w.tags = (w.tags + [(tag, hand)] * n)[:8]


def _close(c, tag, p, q, w, skip=False):
    """A pack closes: back to where it was opened; the drawn hand goes back to the bottom of
    the deck (the undealt cards stay on top, in order). skip: SKIP closed it (the hand is p's)."""
    w.f.update(game_stage=int(p.pack_return), pack_id=-1, pack_left=0)
    c.check(f"{tag}: the pack closes", int(q.pack_id) == -1 or int(q.game_stage) == STAGE_PACK)
    ptr, size = int(p.deck_ptr), int(p.deck_size)
    top = [int(x) for x in p.deck[ptr:size]]
    npeq(c, f"{tag}: close: the undealt cards on top", [int(x) for x in q.deck[:len(top)]], top)
    rest = [int(x) for x in q.deck[len(top):int(q.deck_size)]]
    live = {i for i in range(MAX_DECK_SIZE) if q.card_live[i]}
    c.check(f"{tag}: close: the hand at the bottom", set(rest) == live - set(top), rest)
    held = [int(p.hand[l]) for l in range(MAX_HAND_SIZE) if p.hand_valid[l]]
    if skip and all(q.card_live[i] for i in held):
        npeq(c, f"{tag}: close: the hand in lane order (the leftmost first)", rest[:len(held)], held)


def _end_window(p, J, w, held_cards, total):
    """env.end_window: after the PLAY that ends a round, the round waits in STAGE_END when the
    joker order can still change its end: won (or Mr. Bones saves it), Mime and Blueprint /
    Brainstorm act, a held card has an end-of-round effect (Gold; a Blue Seal with room)."""
    names = R.acting_names(J)
    beaten = total >= w.f["blind_target"]
    bones = "Mr. Bones" in names and total >= 0.25 * w.f["blind_target"]
    eff = any(not R.as_card(cd).deb and (R.as_card(cd).enh == R.GOLD
                                         or (R.as_card(cd).seal == R.BLUE_SEAL and w.room() > 0))
              for cd in held_cards)
    return bool((beaten or bones) and "Mime" in names and ("Blueprint" in names or "Brainstorm" in names) and eff)


def _end_round(c, tag, p, q, J, w, held_cards, total, played=None):
    """end_round after a PLAY / DISCARD / disable / NEXT in STAGE_END: beaten -> the cash-out
    screen; else Mr. Bones (>= 25 %: destroyed, the blind pays $0) or the run is lost.
    -> (J, reward, a new shop: never, the shop opens at Cash Out)"""
    beaten = total >= w.f["blind_target"]
    bones = [i for i, x in enumerate(J) if x["name"] == "Mr. Bones" and not x.get("deb")]
    saved = not beaten and bones and total >= 0.25 * w.f["blind_target"]
    if beaten or saved:
        w.f["money"] -= R.RENTAL_RATE * sum(1 for x in J if x["stk"] & STK_RENTAL)   # calculate_rental
        if saved:
            J = _drop(J, {bones[0]})
        J, _ = _clear_round(c, tag, p, J, w, held_cards, bool(saved), played)
        J, gros = _deaths(c, tag, J, q)
        for x in J:
            x.pop("fresh", None)
        w.f["gros_extinct"] = bool(p.gros_extinct) or gros
        c.check(f"{tag}: the round targets are cards / suits",
                0 <= int(q.idol_card) < 52 and 0 <= int(q.mail_rank) < 13
                and 0 <= int(q.castle_suit) < 4 and 0 <= int(q.ancient_suit) < 4)
        c.check(f"{tag}: Ancient Joker gets a new suit", int(q.ancient_suit) != int(p.ancient_suit),
                int(q.ancient_suit), int(p.ancient_suit))
        c.check(f"{tag}: the hand goes back to the deck", not np.asarray(q.hand_valid).any())
        if int(p.blind_index) == BLIND_BOSS and w.f["game_stage"] != STAGE_WON:
            _check_new_ante(c, tag, w, p, q)
            w.f["_ante_voucher"] = int(q.ante_voucher)
        return J, w.f["last_cashout"], w.f["game_stage"] == STAGE_SHOP
    w.f["game_stage"] = STAGE_LOST
    return J, 0, False


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
    "wild":     ("2H 5H 7H 9H JC 3C 4D 6S", "11111"),    # 15 x 1   high card JC (a Wild JC: a flush)
    "five":     ("KD KC KH KS QD 2C 3C 4C", "11111"),    # QD made a King: Five of a Kind
    "flush5":   ("2D 3D 4D 5D 6D 2C 3C 4C", "11111"),    # all made Kings: Flush Five
    "flushh":   ("2D 3D 4D 5D 6D 2C 3C 4C", "11111"),    # K K K Q Q of Diamonds: Flush House
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
    (("To Do List",), "pair", dict(joker_state=(10,)), 30, 2),
    (("To Do List", "Bull"), "pair", dict(joker_state=(10, 0)), 38, 2),   # its $4 comes first
    (("To Do List", "Bull"), "pair", dict(joker_state=(9, 0)), 30, 2),
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

    # ======================= GROUP D + card modifiers =======================
    # pair = KD KC played; 2H 3S 4C 6D 7H 8S held. 5H and the rest are in the deck.
    (("Marble Joker",), "pair", {}, 30, 2),
    (("8 Ball",), "pair", {}, 30, 2),
    (("Steel Joker",), "pair", {}, 30, 2),
    (("Steel Joker",), "pair", dict(cards_mod={"5H": dict(enh=5)}), 30, 2.4),     # a Steel card in the deck
    (("Steel Joker",), "pair", dict(cards_mod={"2H": dict(enh=5)}), 30, 3.6),     # held Steel X1.5, then X1.2
    (("Space Joker", "Oops! All 6s", "Oops! All 6s"), "pair", {}, 45, 3),          # 4 in 4: Pair level 2
    (("DNA",), "pair", {}, 30, 2),
    (("Constellation",), "pair", dict(joker_value=(1.5,)), 30, 3),
    (("Hiker",), "pair", {}, 30, 2),
    (("Hiker", "Hanging Chad"), "pair", {}, 65, 2),          # KD x3: 10 + 15 + 20
    (("Superposition",), "pair", {}, 30, 2),
    (("Red Card",), "pair", dict(joker_value=(6.0,)), 30, 8),
    (("Vampire",), "pair", dict(cards_mod={"KD": dict(enh=1)}), 30, 2.2),           # Bonus stripped, X1.1
    (("Vampire",), "pair", {}, 30, 2),
    (("Hologram",), "pair", dict(joker_value=(1.25,)), 30, 2.5),
    (("Vagabond",), "pair", {}, 30, 2),
    (("Midas Mask",), "pair", {}, 30, 2),
    (("Midas Mask", "Golden Ticket"), "pair", {}, 30, 2),     # $ only
    (("Hallucination",), "pair", {}, 30, 2),
    (("Fortune Teller",), "pair", dict(tarots_used=3), 30, 5),
    (("Fortune Teller",), "pair", {}, 30, 2),
    (("Stone Joker",), "pair", dict(cards_mod={"5H": dict(enh=6)}), 55, 2),
    (("Stone Joker",), "pair", {}, 30, 2),
    (("Lucky Cat",), "pair", dict(joker_value=(1.5,)), 30, 3),
    (("Golden Ticket",), "pair", dict(cards_mod={"KD": dict(enh=7)}), 30, 2),
    (("Certificate",), "pair", {}, 30, 2),
    (("Glass Joker",), "pair", dict(joker_value=(1.75,)), 30, 3.5),
    (("Driver's License",), "pair", dict(cards_mod={c: dict(enh=1) for c in
        "2D 3D 4D 5D 7D 8D 9D TD JD QD AD 2S 4S 5S 6S 7S".split()}), 30, 6),
    (("Driver's License",), "pair", dict(cards_mod={c: dict(enh=1) for c in
        "2D 3D 4D 5D 7D 8D 9D TD JD QD AD 2S 4S 5S 6S".split()}), 30, 2),
    (("Cartomancer",), "pair", {}, 30, 2),
    (("Astronomer",), "pair", {}, 30, 2),
    (("Burnt Joker",), "pair", {}, 30, 2),
    (("Satellite",), "pair", {}, 30, 2),
    (("Trading Card",), "pair", {}, 30, 2),
    # card modifiers, levels, editions (no joker)
    ((), "pair", dict(cards_mod={"KD": dict(enh=1)}), 60, 2),            # Bonus +30
    ((), "pair", dict(cards_mod={"KD": dict(enh=2)}), 30, 6),            # Mult +4
    ((), "pair", dict(cards_mod={"KD": dict(enh=4)}), 30, 4),            # Glass X2
    ((), "pair", dict(cards_mod={"2H": dict(enh=5)}), 30, 3),            # Steel held X1.5
    ((), "pair", dict(cards_mod={"2H": dict(enh=5, seal=2)}), 30, 4.5),  # + Red Seal: twice
    ((), "pair", dict(cards_mod={"KC": dict(enh=6)}), 65, 1),            # Stone: high card K + 50
    ((), "pair", dict(cards_mod={"KD": dict(ed=1)}), 80, 2),             # Foil +50
    ((), "pair", dict(cards_mod={"KD": dict(ed=2)}), 30, 12),            # Holographic +10
    ((), "pair", dict(cards_mod={"KD": dict(ed=3)}), 30, 3),             # Polychrome X1.5
    ((), "pair", dict(cards_mod={"KD": dict(seal=2)}), 40, 2),           # Red Seal: KD twice
    ((), "pair", dict(cards_mod={"KD": dict(seal=1)}), 30, 2),           # Gold Seal: $ only
    ((), "pair", dict(cards_mod={"KD": dict(bonus=15.0)}), 45, 2),       # perma +15
    ((), "wild", dict(cards_mod={"JC": dict(enh=3)}), 68, 4),            # Wild: a flush
    ((), "wild", {}, 15, 1),                                              # no Wild: high card J
    (("Oops! All 6s",) * 3, "pair", dict(cards_mod={"KD": dict(enh=8)}), 30, 22),   # Lucky 8 in 5
    ((), "pair", dict(hand_level=[1] * 10 + [3, 1]), 60, 4),             # Pair level 3
    ((), "five", dict(cards_mod={"QD": dict(rank=11)}), 170, 12),        # Five of a Kind
    ((), "flush5", dict(cards_mod={c: dict(rank=11) for c in "2D 3D 4D 5D 6D".split()}), 210, 16),
    ((), "flushh", dict(cards_mod={"2D": dict(rank=11), "3D": dict(rank=11), "4D": dict(rank=11),
                                   "5D": dict(rank=10), "6D": dict(rank=10)}), 190, 14),
    (("Joker",), "pair", dict(joker_ed=(1,)), 80, 6),                    # Foil joker
    (("Joker",), "pair", dict(joker_ed=(2,)), 30, 16),                   # Holographic joker
    (("Joker",), "pair", dict(joker_ed=(3,)), 30, 9),                    # Polychrome joker: (2 + 4) X1.5
    (("The Duo", "Joker"), "pair", dict(joker_ed=(3,)), 30, 10),         # 2 X2 X1.5 + 4
    (("Smeared Joker", "Seeing Double"), "wild", dict(cards_mod={"JC": dict(enh=3)}), 68, 8),
    # group E: the jokers that score (the others change no score: they have scenarios)
    (("Ceremonial Dagger",), "pair", dict(joker_value=(6.0,)), 30, 8),     # +6 Mult (2 x its prey's sell)
    (("Ceremonial Dagger",), "pair", {}, 30, 2),                           # 0 at the start
    (("Madness",), "pair", dict(joker_value=(2.5,)), 30, 5),               # X2.5
    (("Throwback",), "pair", dict(skips=2), 30, 3),                        # X(1 + 0.25 x 2)
    (("Throwback",), "pair", {}, 30, 2),                                   # no skip: X1
    (("Canio",), "pair", dict(joker_value=(3.0,)), 30, 6),                 # X3 (2 faces destroyed)
    (("Yorick",), "pair", dict(joker_value=(2.0,)), 30, 4),                # X2 (23 cards discarded)
    (("Seance",), "sf", {}, 135, 8),                                       # a Spectral card, no score
    (("Sixth Sense", "Riff-raff", "Luchador", "Diet Cola", "Mr. Bones"), "pair", {}, 30, 2),
    (("Matador", "Invisible Joker", "Chicot", "Perkeo"), "pair", {}, 30, 2),
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
                    elif f in ("joker_value", "joker_state", "joker_sell_value", "hand_valid", "joker_ed"):
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
                if len(v) < len(base):
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
          "flush4", "straight4", "gappy", "random", "ace_straight"]
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
          ("Mad Joker",), ("To Do List", "Bull"), ("Hit the Road",), ("Card Sharp",),
          ("Vampire",), ("Midas Mask", "Golden Ticket"), ("Hiker", "Hanging Chad"), ("Lucky Cat",),
          ("Space Joker", "Oops! All 6s"), ("Steel Joker",), ("Stone Joker",), ("Superposition",),
          ("8 Ball",), ("Vagabond",), ("Glass Joker",), ("Driver's License",), ("Fortune Teller",),
          ("Smeared Joker", "Seeing Double"), ("Flower Pot",), ("Blackboard",)]
X_COUNTER = ("Obelisk", "Hit the Road", "Campfire", "Constellation", "Vampire", "Hologram",
             "Lucky Cat", "Glass Joker", "Madness", "Canio", "Yorick")


def _played(rng):
    """Random played cards for a random hand shape. Ids are distinct."""
    shape = SHAPES[rng.integers(len(SHAPES))]
    if shape == "random":
        return list(rng.choice(52, rng.integers(1, 6), replace=False))
    if shape in ("flush", "straight", "sf", "flush4", "straight4", "gappy", "ace_straight"):
        if shape in ("flush", "flush4"):
            ranks = list(rng.choice(13, 5, replace=False))
        elif shape == "gappy":
            lo = int(rng.integers(0, 4))
            ranks = sorted(rng.choice(range(lo, lo + 9), 5, replace=False))
        elif shape == "ace_straight":
            lo = int(rng.choice([-1, 8]))                # A-2-3-4-5 or T-J-Q-K-A
            ranks = [ACE if r < 0 else r for r in range(lo, lo + 5)]
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
            if nm in X_COUNTER:
                values.append(1.0 + float(rng.integers(0, 10)) * 0.25)
            elif nm == "Ramen":
                values.append(1.0 + float(rng.integers(1, 101)) * 0.01)
            elif nm in R.SEED:
                values.append(float(rng.integers(0, 13)) + (nm in ("Ice Cream", "Seltzer", "Turtle Bean", "Popcorn")))
            else:
                values.append(float(JOKERS.VALUE[j]))
            states.append(int(rng.integers(0, N_HAND_TYPES)) if nm == "To Do List" else int(rng.integers(0, 12)) if nm == "Loyalty Card" else 0)
        hp = [int(x) for x in rng.integers(0, 4, N_HAND_TYPES)]
        hr = [min(int(x), h) for x, h in zip(rng.integers(0, 2, N_HAND_TYPES), hp)]
        # card modifiers on the cards in hand (and some in the deck), a duplicate rank now and then
        rank_ = np.array(_BASE.card_rank).copy()
        suit_ = np.array(_BASE.card_suit).copy()
        enh = np.zeros(MAX_DECK_SIZE, np.int32)
        ed = np.zeros(MAX_DECK_SIZE, np.int32)
        seal = np.zeros(MAX_DECK_SIZE, np.int32)
        bonus = np.zeros(MAX_DECK_SIZE, np.float32)
        p_mod = rng.choice([0.0, 0.2, 0.5])
        for x in [h for h in hand if h >= 0]:
            if rng.random() < p_mod:
                enh[x] = rng.integers(0, 9)
            if rng.random() < p_mod / 2:
                ed[x] = rng.integers(1, 4)
            if rng.random() < p_mod / 2:
                seal[x] = rng.integers(1, 5)
            if rng.random() < p_mod / 3:
                bonus[x] = 5 * int(rng.integers(1, 5))
        deck_enh = rng.random() < 0.1 or "Driver's License" in [NAME[j] for j in jokers]
        if deck_enh:
            for x in rng.choice(52, int(rng.integers(5, 30)), replace=False):
                if x not in hand:
                    enh[x] = rng.integers(1, 9)
        if rng.random() < 0.08 and len(played) >= 3:
            r0, s0 = rank_[played[0]], suit_[played[0]]
            for x in played:
                rank_[x] = r0
                if rng.random() < 0.5:
                    suit_[x] = s0
        jed = [int(rng.integers(1, 4)) if rng.random() < 0.15 else 0 for _ in jokers]
        level = [int(x) for x in rng.integers(1, 4, N_HAND_TYPES)] if rng.random() < 0.3 else [1] * N_HAND_TYPES
        cases.append(_case(jokers, hand, valid, sel, values=values, joker_state=states,
                           card_rank=rank_, card_suit=suit_, card_enh=enh, card_ed=ed, card_seal=seal,
                           card_bonus=bonus, joker_ed=jed, hand_level=level,
                           tarots_used=int(rng.integers(0, 4)),
                           joker_sell_value=[int(rng.integers(1, 9)) for _ in jokers],
                           money=int(rng.integers(-20, 120)), discards=int(rng.integers(0, 4)),
                           hands=int(rng.integers(1, 5)), deck_ptr=int(rng.integers(8, 53)),
                           joker_slots=slots, hand_played=hp, hand_round=hr,
                           idol_card=int(rng.choice(played)) if rng.random() < 0.5 else int(rng.integers(52)),
                           mail_rank=int(rng.integers(13)),
                           ancient_suit=int(rng.integers(4)), castle_suit=int(rng.integers(4)),
                           skips=int(rng.integers(0, 4)),
                           key=[int(x) for x in rng.integers(0, 2 ** 31, 2)]))
    return cases


MONEY_ONLY = {"Golden Joker", "Cloud 9", "Mail-In Rebate", "Faceless Joker", "Delayed Gratification",
              "Rocket", "Egg", "Gift Card", "Credit Card", "To the Moon", "Chaos the Clown", "Juggler",
              "Drunkard", "Merry Andy", "Troubadour", "Turtle Bean", "Burglar", "Showman",
              "Oops! All 6s", "Blueprint", "Brainstorm", "Erosion", "Ice Cream", "Seltzer",
              "Marble Joker", "DNA", "Certificate", "Cartomancer", "Hallucination", "Astronomer",
              "Burnt Joker", "Satellite", "Trading Card", "Red Card", "Constellation", "Hologram",
              "Sixth Sense", "Seance", "Riff-raff", "Luchador", "Diet Cola", "Mr. Bones", "Matador",
              "Invisible Joker", "Chicot", "Perkeo"}


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

    # Popcorn -4 a round and eaten at 0; Golden / Cloud 9 money over 6 cleared blinds (the bosses
    # have no ability here).
    s = env_step(reset(jax.random.key(3), 52, 8)._replace(boss=jnp.int32(-1)), act(V_NEXT))[0]
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
        s = drive(c, f"round {r} next", s, [act(V_NEXT), act(V_NEXT)])      # Cash Out, then leave the shop
        s = drive(c, f"round {r} select", s._replace(boss=jnp.int32(-1)), [act(V_NEXT)])
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
    s = drive(c, "budget", s, [act(V_BUY, target=k) for k in range(4)] + [act(V_NEXT), act(V_NEXT)])
    c.eq("budget: hand size 8 + 1 - 1 + 2", s.hand_size, 10)
    c.eq("budget: hands 4 - 1", s.hands, 3)
    c.eq("budget: discards 3 + 3 + 1", s.discards, 7)
    c.eq("budget: 10 cards dealt", s.hand_valid.sum(), 10)
    s = drive(c, "budget sell", s, [act(V_SELL, target=2), act(V_PLAY, "1")])
    c.eq("budget: Troubadour sold, hand size 8", s.hand_size, 8)
    c.check("budget: the extra cards stay", int(s.hand_valid.sum()) >= 8, int(s.hand_valid.sum()))

    # Burglar: +3 hands, no discards
    s = state(hand, jokers=("Burglar", "Blueprint"), game_stage=STAGE_SHOP)
    s = drive(c, "burglar", s, [act(V_MOVE_JOKER, target=1, dest=0), act(V_NEXT), act(V_NEXT)])
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
    base = state(hand, shop_size=4)
    shop_j = lambda s: jax.vmap(lambda k: (lambda o: jnp.where(o.kinds == IK_JOKER, o.ids, -1))(roll_shop(k, s)))(keys)
    ext = shop_j(base._replace(gros_extinct=jnp.bool_(True)))
    alive = shop_j(base)
    c.check("extinct: Cavendish appears", bool((ext == JOKER_ID["Cavendish"]).any()))
    c.check("extinct: Gros Michel never", not bool((ext == JOKER_ID["Gros Michel"]).any()))
    c.check("alive: Cavendish never", not bool((alive == JOKER_ID["Cavendish"]).any()))
    # Showman: held jokers and repeats in the shop
    show = shop_j(state(hand, shop_size=4, jokers=("Joker", "Showman")))
    c.check("showman: a held joker appears", bool((show == JOKER_ID["Joker"]).any()))
    rep = np.asarray(show)
    c.check("showman: a joker twice in one shop", any(len(set(x for x in r if x >= 0)) < (r >= 0).sum() for r in rep))
    # enhancement gates: Steel Joker only with a Steel card in the deck
    gate = shop_j(base)
    c.check("no Steel card: no Steel Joker", not bool((gate == JOKER_ID["Steel Joker"]).any()))
    gate = shop_j(state(hand, shop_size=4, cards_mod={"5H": dict(enh=5)}))
    c.check("a Steel card: Steel Joker may appear", bool((gate == JOKER_ID["Steel Joker"]).any()))
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
    n_cons = int((p.consumables >= 0).sum())
    if u < 0.02:                                             # any verb, any target
        return (int(rng.integers(-1, N_VERBS + 1)), rng.random(MAX_HAND_SIZE) < 0.3, int(rng.integers(-2, 34)),
                int(rng.integers(-2, 34)))
    if len(valid):
        k = int(rng.integers(1, min(5, len(valid)) + 1))
        if rng.random() < 0.05:
            k = len(valid)                                   # more than 5: the first 5 play
        sel[rng.choice(valid, k, replace=False)] = True
    if stage == STAGE_BLIND and len(valid):
        if u < 0.5:
            return V_PLAY, sel, 0, 0
        if u < 0.68:
            return V_DISCARD, sel, 0, 0
        if u < 0.74:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.8:
            return V_USE, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.82:
            return V_SELL_CONS, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.91:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        size = int(p.hand_size)
        return V_MOVE_CARD, sel, int(rng.integers(-1, size + 1)), int(rng.integers(-1, size + 1))
    if stage == STAGE_SHOP:
        if u < 0.3:
            return V_BUY, sel, int(rng.integers(-1, MAX_SHOP_SLOTS)), 0
        if u < 0.42:
            return V_OPEN, sel, int(rng.integers(-1, 3)), 0
        if u < 0.5:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.56:
            return V_USE, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.58:
            return V_SELL_CONS, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.66:
            return V_REROLL, sel, 0, 0
        if u < 0.7:
            return V_REDEEM, sel, int(rng.integers(-1, 3)), 0
        if u < 0.76:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        return V_NEXT, sel, 0, 0
    if stage == STAGE_SELECT:
        if u < 0.25:
            return V_SKIP, sel, 0, 0
        if u < 0.3:
            return V_REROLL, sel, 0, 0
        if u < 0.34:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.38:
            return V_USE, sel, int(rng.integers(-1, n_cons + 1)), 0
        return V_NEXT, sel, 0, 0
    if stage == STAGE_PACK:
        if u < 0.6:
            return V_PICK, sel, int(rng.integers(-1, 6)), 0
        if u < 0.72:
            return V_USE, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.76:
            return V_MOVE_CARD, sel, int(rng.integers(-1, 9)), int(rng.integers(-1, 9))
        if u < 0.8:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        return V_SKIP, sel, 0, 0
    if stage == STAGE_END:
        if u < 0.5:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        if u < 0.55:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0      # illegal there: a no-op
        return V_NEXT, sel, 0, 0
    if stage == STAGE_CASHOUT:
        if u < 0.1:
            return V_SELL, sel, int(rng.integers(-1, held + 1)), 0
        if u < 0.22:
            return V_USE, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.26:
            return V_SELL_CONS, sel, int(rng.integers(-1, n_cons + 1)), 0
        if u < 0.34:
            return V_MOVE_JOKER, sel, int(rng.integers(-1, held + 1)), int(rng.integers(-1, held + 1))
        return V_NEXT, sel, 0, 0
    return int(rng.integers(0, N_VERBS)), sel, 0, 0         # LOST / WON: all no-ops


_batch_reset = jax.jit(jax.vmap(lambda k, stake, back: reset(k, 52, 8, 1.0, stake, back)))
_batch_budget = jax.jit(jax.vmap(apply_budget))
_PAIRS = [(VOUCHER_NAMES.index(a), VOUCHER_NAMES.index(b)) for a, b in R.VOUCHER_PAIRS]


def _soak_batch(rng, key, n_envs):
    """Fresh runs: a random stake and deck, 0-4 random starting jokers (with the stake's stickers;
    a Perishable one 0-5 rounds left), some money, a mix of curriculum scales."""
    stakes = rng.integers(1, 9, n_envs)
    backs = rng.integers(0, N_BACKS, n_envs)
    batch = _batch_reset(jax.random.split(key, n_envs), jnp.asarray(stakes, jnp.int32), jnp.asarray(backs, jnp.int32))
    live_n = np.asarray(batch.card_live).sum(axis=1)          # 52 (the Abandoned Deck: 40)
    cslots = np.asarray(batch.consumable_slots)
    js = np.asarray(batch.joker_slots)
    held_c = np.asarray(batch.consumables)
    held_s = np.asarray(batch.cons_sell)
    J = np.full((n_envs, MAX_JOKERS), -1, np.int32)
    STK = np.zeros((n_envs, MAX_JOKERS), np.int32)
    LEFT = np.zeros((n_envs, MAX_JOKERS), np.int32)
    V = np.zeros((n_envs, MAX_JOKERS), np.float32)
    S = np.zeros((n_envs, MAX_JOKERS), np.int32)
    T = np.zeros((n_envs, MAX_JOKERS), np.int32)
    E = np.zeros((n_envs, MAX_JOKERS), np.int32)
    C = np.full((n_envs, MAX_CONSUMABLES), -1, np.int32)
    enh = np.zeros((n_envs, MAX_DECK_SIZE), np.int32)
    seal = np.zeros((n_envs, MAX_DECK_SIZE), np.int32)
    ced = np.zeros((n_envs, MAX_DECK_SIZE), np.int32)
    for i in range(n_envs):
        for slot, j in enumerate(rng.integers(0, N_JOKERS, rng.integers(0, min(5, int(js[i]) + 1)))):
            E[i, slot] = int(rng.integers(1, 4)) if rng.random() < 0.1 else 0
            cost = int(JOKER_COST[j]) + int(ED_COST[E[i, slot]])
            u, st, nm = rng.random(), int(stakes[i]), NAME[j]
            et = st >= 4 and u > 0.7 and nm not in R.NOT_ETERNAL
            pe = st >= 7 and 0.4 < u <= 0.7 and nm not in R.NOT_PERISHABLE
            re = st >= 8 and rng.random() > 0.7
            STK[i, slot] = et * STK_ETERNAL + pe * STK_PERISH + re * STK_RENTAL
            LEFT[i, slot] = int(rng.integers(0, 6)) if pe else 0
            cost = 1 if re else cost
            J[i, slot], S[i, slot] = j, max(1, cost // 2)
            V[i, slot] = _seed(j)
            T[i, slot] = int(rng.integers(3, N_HAND_TYPES)) if NAME[j] == "To Do List" else 0
        if rng.random() < 0.5:                                   # the deck's own consumables (Magic, Ghost)
            C[i, :] = held_c[i]
        else:
            for slot in range(int(rng.integers(0, min(3, int(cslots[i]) + 1)))):
                C[i, slot] = int(rng.integers(0, len(CONS_NAMES)))
        if rng.random() < 0.5:
            for x in rng.choice(int(live_n[i]), int(rng.integers(1, 20)), replace=False):
                enh[i, x] = rng.integers(0, 9)
                seal[i, x] = rng.integers(0, 5) if rng.random() < 0.3 else 0
                ced[i, x] = rng.integers(0, 4) if rng.random() < 0.2 else 0
        if rng.random() < 0.15:                              # the END window: Mime + a copier, Gold cards
            for slot, nm in enumerate(("Mime", "Blueprint" if rng.random() < 0.6 else "Brainstorm")):
                j = JOKER_ID[nm]
                J[i, slot], S[i, slot], V[i, slot], E[i, slot] = j, max(1, int(JOKER_COST[j]) // 2), _seed(j), 0
                STK[i, slot], LEFT[i, slot], T[i, slot] = 0, 0, 0
            for x in rng.choice(int(live_n[i]), 12, replace=False):
                enh[i, x] = R.GOLD
    scale = rng.choice([0.01, 0.03, 0.1, 0.3], n_envs).astype(np.float32)
    # some vouchers already redeemed (the second of a pair only after the first)
    VCH = np.zeros((n_envs, len(VOUCHER_NAMES)), bool)
    for i in range(n_envs):
        for a, b in _PAIRS:
            if rng.random() < 0.15:
                VCH[i, a] = True
                VCH[i, b] = rng.random() < 0.4
    sells = np.where(C >= 0, rng.integers(1, 4, C.shape), 0)
    sells = np.where((C >= 0) & np.isin(C, np.flatnonzero(np.asarray(CONS_KIND) == CK_SPECTRAL)), sells + 1, sells)
    same = (C == held_c).all(axis=1)
    sells = np.where(same[:, None], held_s, sells)
    # the deck's vouchers stay (Magic, Nebula, Zodiac), with some more
    VCH = VCH | np.asarray(batch.vouchers)
    batch = batch._replace(
        jokers=jnp.asarray(J), joker_value=jnp.asarray(V), joker_sell_value=jnp.asarray(S),
        joker_state=jnp.asarray(T), joker_ed=jnp.asarray(E), consumables=jnp.asarray(C),
        cons_sell=jnp.asarray(sells, jnp.int32),                 # some Gift Card rounds
        card_enh=jnp.asarray(enh), card_seal=jnp.asarray(seal), card_ed=jnp.asarray(ced),
        money=jnp.asarray(rng.integers(0, 30, n_envs), jnp.int32), target_scale=jnp.asarray(scale),
        blind_target=jnp.asarray(np.maximum(np.floor(100 * scale * np.where(backs == B_PLASMA, 2, 1)), 1),
                                 jnp.int32),
        vouchers=jnp.asarray(VCH), joker_sticker=jnp.asarray(STK), joker_perish=jnp.asarray(LEFT))
    batch = _batch_budget(batch)
    # the first blind starts with the budget the jokers give
    return batch._replace(hands=batch.handcount, discards=batch.discardcount)


def _n_perished(h):
    """The held jokers of host state h that perished (Perishable, no round left)."""
    return int((((np.asarray(h.joker_sticker) & STK_PERISH) > 0) & (np.asarray(h.joker_perish) == 0)
                & (np.asarray(h.jokers) >= 0)).sum())


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
                   V_MOVE_CARD: "move_card", V_USE: "use", V_SELL_CONS: "sell_cons", V_OPEN: "open",
                   V_PICK: "pick", V_SKIP: "skip", V_REDEEM: "redeem"}[verb]] += 1
            if verb == V_SKIP and int(p.game_stage) == STAGE_SELECT and int(q.skips) > int(p.skips):
                stats["blind skips"] += 1
            if verb == V_NEXT and int(p.game_stage) == STAGE_SELECT and int(p.blind_index) == BLIND_BOSS:
                stats["bosses"] += 1
            if verb == V_PLAY:
                for j in p.jokers:
                    if j >= 0:
                        held_at_play[NAME[j]] += 1
                if int(q.game_stage) in (STAGE_CASHOUT, STAGE_END, STAGE_WON):
                    stats["clears"] += 1
                    bk = int(p.back)
                    if bk == B_GREEN:
                        stats["green cash outs"] += 1
                    if bk == B_ANAGLYPH and int(p.blind_index) == BLIND_BOSS:
                        stats["anaglyph bosses"] += 1
                if int(p.back) == B_PLASMA:
                    stats["plasma plays"] += 1
                if int(q.game_stage) == STAGE_END:
                    stats["end windows"] += 1
            if int(p.game_stage) == STAGE_CASHOUT and verb not in (None, V_NEXT):
                stats["cash-out actions"] += 1
            if int(q.game_stage) == STAGE_LOST and int(p.game_stage) != STAGE_LOST:
                stats["lost"] += 1
            if int(q.game_stage) == STAGE_WON and int(p.game_stage) != STAGE_WON:
                stats["won"] += 1
            # the stake stickers at work: an Eternal joker kept from a SELL, a joker that perished,
            # a round end with a Rental joker held, a sticker bought or picked
            if verb == V_SELL and 0 <= int(a_host.target[i]) < MAX_JOKERS and int(p.joker_sticker[int(a_host.target[i])]) & STK_ETERNAL \
                    and int(p.jokers[int(a_host.target[i])]) >= 0:
                stats["eternal kept"] += 1
            stats["perished"] += _n_perished(q) > _n_perished(p)
            if int(q.game_stage) in (STAGE_CASHOUT, STAGE_WON) and int(p.game_stage) in (STAGE_BLIND, STAGE_END) \
                    and (np.asarray(p.joker_sticker) & STK_RENTAL).any():
                stats["rent paid"] += 1
            if verb in (V_BUY, V_PICK) and (np.asarray(q.joker_sticker) != 0).sum() > (np.asarray(p.joker_sticker) != 0).sum():
                stats["sticker joker in"] += 1
            tb = int(a_host.target[i])
            if int(q.game_stage) == STAGE_SHOP and ((np.asarray(q.shop_kind) == IK_SPECTRAL)
                                                    & (np.asarray(q.shop_slots) >= 0))[:int(q.shop_size)].any():
                stats["ghost shop spectrals"] += 1
            if verb == V_BUY and 0 <= tb < MAX_SHOP_SLOTS and int(p.shop_kind[tb]) in (IK_TAROT, IK_PLANET, IK_SPECTRAL) \
                    and int(p.shop_slots[tb]) >= 0 > int(q.shop_slots[tb]) \
                    and np.array_equal(np.asarray(p.consumables), np.asarray(q.consumables)):
                stats["buy and use"] += 1
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
    c.check("the soak met the stake stickers (Eternal SELL, perished, rent, a sticker joker in)",
            min(stats["eternal kept"], stats["perished"], stats["rent paid"], stats["sticker joker in"]) > 0,
            {k: stats[k] for k in ("eternal kept", "perished", "rent paid", "sticker joker in")})
    c.check("the soak met the END window and acted on the cash-out screen",
            min(stats["end windows"], stats["cash-out actions"]) > 0,
            {k: stats[k] for k in ("end windows", "cash-out actions")})
    c.check("the soak met Buy and Use (a BUY with the consumable slots full)", stats["buy and use"] > 0,
            stats["buy and use"])
    deck_stats = ("green cash outs", "anaglyph bosses", "plasma plays", "ghost shop spectrals")
    c.check("the soak met the decks' later rules (Green, Anaglyph, Plasma, Ghost)",
            min(stats[k] for k in deck_stats) > 0, {k: stats[k] for k in deck_stats})
    return c.done()


ALL = [run_table_test, run_spec_test, run_misprint_test, run_odds_test, run_order_test,
       run_fuzz_test, run_life_test, run_soak_test]


def run_all():
    return all([f() for f in ALL])


if __name__ == "__main__":
    run_all()
