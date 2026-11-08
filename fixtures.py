"""Env staging as DATA — author a test state in the few fields that matter.

A test state is a ROW: write the handful of fields the scenario is about, and
let every other field default. One generic builder makes the pytree, the same
way one generic evaluator reads a joker row.

WHY: balatroEnvState is a positional NamedTuple with no defaults, so every
fixture spelled out all 16 fields. Adding one state field (shop_slots) broke
four fixtures at once. Here a new field is ONE default in state(), and nothing
downstream moves.

Cards are NAMES, not ids. "KD" is readable; 24 needs a comment explaining that
24 % 13 == 11 == King. Raw ids still work where the id itself is the point:

    state(hand="4C KD 9H KC 7C 9S 7S 3H")     # names
    state(hand=[2, 24, 33, 11, 5, 46, 44, 27])  # ids — same hand
    state(hand="2D 3D . 5D 6D . . 9D")          # "." is an empty lane

Import direction: this sits ABOVE env (it imports the state type) and BELOW
tests / agent scratch blocks. Nothing in the env may import it.
"""

import jax
import jax.numpy as jnp

from env import balatroEnvState, MAX_HAND_SIZE, MAX_DECK_SIZE, MAX_CONSUMABLES, MAX_SHOP_SLOTS
from env import standard_cards, STAGE_PACK, CONS_SELL, run_fields, BASE_JOKER_SLOTS
from items import (BACK_NAMES, N_HAND_TYPES, CONS_ID, PACK_SLOTS, MAX_PACK, ED_NEG, MAX_JOKER_SLOTS, BOSS_ID,
                   TAG_ID, VOUCHER_ID, MAX_TAGS, MAX_SHOP_VOUCHERS, V_ANTIMATTER, HAND_NAMES,
                   STK_ETERNAL, STK_PERISH, STK_RENTAL, PERISH_ROUNDS, RENTAL_COST)
# the card-name convention has ONE definition, in env (show() prints with it)
from env import RANK_CHARS, SUIT_CHARS
from jokers import JOKER_COST
from jokers import (JOKER_ID, J_EMPTY, J_STENCIL, MAX_JOKERS, N_QUERY_KINDS,
                    Q_MONEY_DIV_5, Q_MONEY, Q_DISCARDS_LEFT, Q_NO_DISCARDS_LEFT, Q_DECK_LEFT,
                    Q_JOKER_COUNT, Q_EMPTY_JOKER_SLOTS, rows_for, resolve)

EMPTY = -1
EMPTY_CHARS = {".", "-", "_"}   # any of these authors an empty lane


# ---------------------------------------------------------------------------
# CARD NAMES — the human end of the id convention
# ---------------------------------------------------------------------------
def card_id(name):
    """'KD' -> 24. '.' -> -1. Raises on a name the convention cannot produce."""
    if isinstance(name, (int, jnp.integer)):
        return int(name)
    name = name.strip().upper()
    if name in EMPTY_CHARS:
        return EMPTY
    if len(name) != 2:
        raise ValueError(f"card {name!r}: want two chars, rank then suit (e.g. 'KD')")
    rank, suit = name[0], name[1]
    if rank not in RANK_CHARS:
        raise ValueError(f"card {name!r}: rank {rank!r} not in {RANK_CHARS!r}")
    if suit not in SUIT_CHARS:
        raise ValueError(f"card {name!r}: suit {suit!r} not in {SUIT_CHARS!r}")
    return SUIT_CHARS.index(suit) * 13 + RANK_CHARS.index(rank)


def card_name(cid):
    """24 -> 'KD'. Anything outside 0..51 prints as its raw number."""
    cid = int(cid)
    return RANK_CHARS[cid % 13] + SUIT_CHARS[cid // 13] if 0 <= cid < 52 else str(cid)


def cards(spec):
    """'4C KD .' or [2, 24, -1] -> [2, 24, -1]. The one entry point for both."""
    if isinstance(spec, str):
        spec = spec.split()
    return [card_id(c) for c in spec]


# ---------------------------------------------------------------------------
# LANES — short authored lists into the fixed-width arrays the state holds
# ---------------------------------------------------------------------------
def lanes(vals, width=MAX_HAND_SIZE, fill=EMPTY, dtype=jnp.int32):
    """Pad a short list to full width. Capacity is SHAPE, count is DATA."""
    a = jnp.full(width, fill, dtype=dtype)
    return a.at[:len(vals)].set(jnp.array(vals, dtype=dtype)) if len(vals) else a


def hand_lanes(spec):
    """'4C KD .' -> the full-width hand array. lanes(cards(spec)), the common pair."""
    return lanes(cards(spec))


def mask_lanes(bits, width=MAX_HAND_SIZE):
    """Pad a short 0/1 list to a full-width bool mask; the tail is False."""
    m = jnp.zeros(width, dtype=bool)
    return m.at[:len(bits)].set(jnp.array(bits, dtype=bool)) if len(bits) else m


def deck_after(hand_ids, deck_size=52):
    """The deck a mid-round state implies: dealt cards first, then the rest.

    Positions 0..n-1 hold the cards already in hand (deck_ptr sits past them),
    then every remaining live id ascending, then the dead ids up to capacity.
    Deliberately NOT the identity — deck[8] != 8 — so confusing a deck POSITION
    with the CARD at that position fails visibly instead of passing by luck.
    """
    dealt = [c for c in hand_ids if c != EMPTY]
    rest = [i for i in range(deck_size) if i not in dealt]
    dead = list(range(deck_size, MAX_DECK_SIZE))
    return jnp.array(dealt + rest + dead, dtype=jnp.int32)


def rotated_deck(by, deck_size=52):
    """deck[i] = (i + by) % deck_size, then the dead ids. Also non-identity."""
    live = (jnp.arange(deck_size, dtype=jnp.int32) + by) % deck_size
    return jnp.concatenate([live, jnp.arange(deck_size, MAX_DECK_SIZE, dtype=jnp.int32)])


# ---------------------------------------------------------------------------
# JOKERS — a loadout is authored as NAMES, the same way a hand is authored
# as card names. ("Joker", "Bootstraps") beats [0, 2] with a comment.
# ---------------------------------------------------------------------------
def sticker_bits(spec):
    """'E' / 'P' / 'R' (any mix: 'ER', 'PR'), '' or '.' = none, or the STK_* bits as an int."""
    if isinstance(spec, str):
        bits = {"E": STK_ETERNAL, "P": STK_PERISH, "R": STK_RENTAL}
        return sum(bits[ch] for ch in spec.upper() if ch not in EMPTY_CHARS)
    return int(spec)


def joker_ids(names, width=MAX_JOKERS):
    """('Joker', 'Bootstraps') -> full-width id array, J_EMPTY in vacant slots.

    "." is a vacant slot, the same as in a hand. A name not in the table raises
    KeyError HERE, naming the typo, instead of silently scoring as whatever row
    that id landed on.
    """
    ids = [J_EMPTY if n in EMPTY_CHARS else JOKER_ID[n] if isinstance(n, str) else int(n)
           for n in names]
    return lanes(ids, width=width, fill=J_EMPTY)


def query_ctx(*, money=0, discards=3, deck_left=0, jokers=0, empty_slots=0):
    """The query results resolve() reads. ctx[k] = the count for query kind k.

    env builds this from live state each hand (env.build_ctx). Here a test
    writes only the inputs its case is about; the rest stay at a neutral value.
    """
    ctx = jnp.zeros(N_QUERY_KINDS, jnp.float32)
    return (ctx.at[Q_MONEY_DIV_5].set(money // 5)
               .at[Q_MONEY].set(money)
               .at[Q_DISCARDS_LEFT].set(discards)
               .at[Q_NO_DISCARDS_LEFT].set(discards == 0)
               .at[Q_DECK_LEFT].set(deck_left)
               .at[Q_JOKER_COUNT].set(jokers)
               .at[Q_EMPTY_JOKER_SLOTS].set(empty_slots))


def loadout(names=(), *, money=0, discards=3, deck_left=0, joker_slots=5, joker_value=None):
    """Stage held jokers for a scoring call. -> (rows, active, v)

    names       joker names in slot order; vacant slots are padded in
    money       for Q_MONEY_DIV_5 (Bootstraps) and Q_MONEY (Bull)
    discards    for Q_DISCARDS_LEFT (Banner) and Q_NO_DISCARDS_LEFT (Mystic Summit)
    deck_left   for Q_DECK_LEFT (Blue Joker)
    joker_slots with names, gives Q_JOKER_COUNT and Q_EMPTY_JOKER_SLOTS
    joker_value live COUNTER memory; defaults to the seed a fresh buy gives
                (VALUE in an occupied slot, 0 in a vacant one)

    This is the three-line preamble every joker test would otherwise repeat:
    gather the rows, mark the live slots, resolve the values.
    """
    ids = joker_ids(names)
    active = ids != J_EMPTY
    rows = rows_for(ids)
    if joker_value is None:
        joker_value = jnp.where(active, rows.VALUE, 0.0)
    else:
        joker_value = lanes(joker_value, MAX_JOKERS, 0.0, jnp.float32)
    held = int(active.sum())
    stencils = int((ids == J_STENCIL).sum())
    ctx = query_ctx(money=money, discards=discards, deck_left=deck_left, jokers=held,
                    empty_slots=max(joker_slots - held + stencils, 0))
    return rows, active, resolve(rows, active, joker_value, ctx)


def play_cards(rank, suit, valid, names=(), *, money=0, discards=3, deck_left=0,
               joker_slots=5, joker_value=None, mods=None, key=0):
    """Score 5 played cards with a loadout through the full pipeline (no held
    cards, nothing played before). -> scoring.PlayOut"""
    from scoring import play, PlayIn, Targets
    from jokers import NO_MODS
    ids = joker_ids(names)
    active = ids != J_EMPTY
    rows = rows_for(ids)
    if joker_value is None:
        joker_value = jnp.where(active, rows.VALUE, 0.0)
    else:
        joker_value = lanes(joker_value, MAX_JOKERS, 0.0, jnp.float32)
    held = int(active.sum())
    stencils = int((ids == J_STENCIL).sum())
    ctx = query_ctx(money=money, discards=discards, deck_left=deck_left, jokers=held,
                    empty_slots=max(joker_slots - held + stencils, 0))
    z9 = jnp.zeros(N_HAND_TYPES, jnp.int32)
    p = PlayIn(jnp.asarray(rank), jnp.asarray(suit), jnp.asarray(valid, bool),
               jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32), jnp.zeros(1, bool),
               ids, joker_value, jnp.zeros(MAX_JOKERS, jnp.int32), ctx, jnp.float32(money),
               jnp.int32(1), z9, z9, jnp.bool_(False),
               Targets(jnp.int32(51), jnp.int32(12), jnp.int32(3), jnp.int32(3)))
    return play(p, NO_MODS if mods is None else mods, jax.random.key(key))


# ---------------------------------------------------------------------------
# THE BUILDER — every field defaulted, so a new state field lands only here
# ---------------------------------------------------------------------------
def state(hand="", *, valid=None, hand_size=None, deck=None, deck_size=52,
          deck_ptr=None, hands=3, discards=2, handcount=4, discardcount=3,
          round=1, ante=1, chips_scored=0, blind_target=300,
          shop_slots=(EMPTY, EMPTY, EMPTY, EMPTY), jokers=(), consumables=(),
          joker_value=None, joker_state=(), joker_sell_value=None, joker_slots=5,
          consumable_slots=2, money=0, game_stage=0, key=0, blind_index=0,
          shop_size=2, reroll_cost=5, last_cashout=0, target_scale=1.0,
          base_hand_size=None, base_hands=None, base_discards=None, discards_used=0,
          hand_played=(), hand_round=(), idol_card=51, mail_rank=12, ancient_suit=3,
          castle_suit=3, free_rerolls=0, gros_extinct=False, cards_mod=None, joker_ed=(),
          hand_level=(), last_hand=-1, last_cons=-1, tarots_used=0, planets_used=(),
          shop_kind=(), shop_ed=(), pack_slots=(EMPTY, EMPTY), buffoon_given=True, cons_sell=None,
          boss=-1, boss_disabled=False, vouchers=(), tags=(), tag_hand=(), blind_tags=(EMPTY, EMPTY),
          skips=0, played_ante=(), forced_card=None, hand_down=(), joker_debuff=(), ante_voucher=-1,
          shop_vouchers=(), cons_ed=(), ecto_minus=1, boss_hand=-1, ox_hand=None, unused_discards=0,
          round_hand_bonus=0, pack_return=None, shop_free=(), pack_free=(), shop_rank=(), shop_suit=(),
          shop_enh=(), bosses_used=(), boss_rerolled=False, boss_sub=0, orbital_hands=(11, 11),
          reroll_inc=0, reroll_d6=False, stake=1, joker_sticker=(), joker_perish=None, shop_sticker=(),
          pack_sticker=(), back=0):
    """Author a balatroEnvState from the fields the scenario is actually about.

    hand        card-name string, or a list of ids. '.' is an empty lane.
    valid       defaults to "every authored lane that holds a card"
    hand_size   defaults to the number of authored lanes
    deck        defaults to deck_after(hand) — the deck a mid-round state implies
    deck_ptr    defaults to hand_size, i.e. the opening hand has just been dealt
    joker_value defaults to the fresh-buy seed (VALUE per occupied slot, else 0)
    joker_state defaults to zeros; joker_slots to 5; consumable_slots to 2
    joker_sell_value defaults to what a buy sets: max(1, cost // 2) per held joker
    key         an int seed, or a real PRNG key to pass straight through
    shop_slots  joker NAMES or ids; '.' is an empty slot
    base_hand_size defaults to hand_size minus what the held jokers add
    base_hands / base_discards default to handcount / discardcount
    hand_played / hand_round  per hand type counts (HAND_NAMES order), default 0
    idol_card / mail_rank / ancient_suit / castle_suit  the round targets
    hand_played / hand_round / hand_level / planets_used: 12 entries (items.HAND_NAMES
                order), or the old 9 (straight flush .. high card), which map to the last 9
    cards_mod   {instance or card name: dict(enh=, ed=, seal=, bonus=, rank=, suit=)} — the card
                modifiers (instances 0..51 are the standard deck: card id = instance)
    consumables Tarot / Planet / Spectral names
    cons_sell   $ each consumable sells for; defaults to $1 each (what a new one sells for)
    boss        the ante's Boss Blind (a name); -1 (the default) is a boss with no ability
    vouchers    redeemed voucher names; tags: held tag names (tag_hand: an Orbital Tag's hand)
    played_ante card names played this ante (The Pillar); forced_card: Cerulean Bell's card
    hand_down / joker_debuff  lanes / slots (0/1 lists): face down / disabled by Crimson Heart
    shop_vouchers voucher names in the shop; ante_voucher: this ante's voucher (a name or -1)
    reroll_inc / reroll_d6  paid rerolls in this shop / a D6 Tag made them start at $0
    stake       1 White .. 8 Gold (the Eternal / Perishable / Rental stickers need stake 4 / 7 / 8)
    back        the deck: an items.BACK_NAMES index or name (only its later rules: Green's cash out,
                Ghost's shop, Anaglyph's tag, Plasma's score and targets; the start fields are as given)
    joker_sticker / shop_sticker / pack_sticker  per slot: 'E', 'P', 'R' (a mix: 'ER'), '' = none
    joker_perish a Perishable joker's rounds left; defaults to 5 (a new one)
    """
    ids = cards(hand)
    jids = joker_ids(jokers)
    held = jids != J_EMPTY
    if joker_value is None:
        joker_value = jnp.where(held, rows_for(jids).VALUE, 0.0)
    else:
        joker_value = lanes(joker_value, MAX_JOKERS, 0.0, jnp.float32)
    jstk = lanes([sticker_bits(x) for x in joker_sticker], MAX_JOKERS, 0, jnp.int32)
    if joker_perish is None:
        jper = jnp.where(held & ((jstk & STK_PERISH) > 0), PERISH_ROUNDS, 0).astype(jnp.int32)
    else:
        jper = lanes(list(joker_perish), MAX_JOKERS, 0, jnp.int32)
    if joker_sell_value is None:
        cost = JOKER_COST[jnp.clip(jids, 0, len(JOKER_COST) - 1)]
        cost = jnp.where((jstk & STK_RENTAL) > 0, RENTAL_COST, cost)       # a Rental joker costs $1
        joker_sell_value = jnp.where(held, jnp.maximum(1, cost // 2), 0).astype(jnp.int32)
    else:
        joker_sell_value = lanes(joker_sell_value, MAX_JOKERS, 0, jnp.int32)
    if hand_size is None:
        hand_size = len(ids)
    if valid is None:
        valid = [c != EMPTY for c in ids]
    if deck is None:
        deck = deck_after(ids, deck_size)
    if deck_ptr is None:
        deck_ptr = hand_size
    if not isinstance(key, jax.Array):
        key = jax.random.key(key)
    if base_hand_size is None:
        rows = rows_for(jids)
        add = jnp.where(held, rows.H_SIZE + jnp.where(rows.HSIZE_V, joker_value, 0).astype(jnp.int32), 0).sum()
        base_hand_size = int(hand_size) - int(add)

    def per_hand(vals, fill=0, dtype=jnp.int32):
        vals = list(vals)
        if len(vals) == 9:
            vals = [fill] * (N_HAND_TYPES - 9) + vals
        return lanes(vals, N_HAND_TYPES, fill, dtype)

    cards_ = standard_cards(deck_size)
    for k, mod in (cards_mod or {}).items():
        i = card_id(k)
        for f, name in (("enh", "card_enh"), ("ed", "card_ed"), ("seal", "card_seal"),
                        ("rank", "card_rank"), ("suit", "card_suit")):
            if f in mod:
                cards_[name] = cards_[name].at[i].set(mod[f])
        if "bonus" in mod:
            cards_["card_bonus"] = cards_["card_bonus"].at[i].set(mod["bonus"])
    jed = lanes(joker_ed, MAX_JOKERS, 0, jnp.int32)
    neg = int(((jed == ED_NEG) & held).sum())
    vids = [VOUCHER_ID[v] if isinstance(v, str) else int(v) for v in vouchers]
    if joker_slots == 5 and (neg or V_ANTIMATTER in vids):
        joker_slots = min(5 + neg + (V_ANTIMATTER in vids), MAX_JOKER_SLOTS)
    cons_ids = [CONS_ID[c] if isinstance(c, str) else int(c) for c in consumables]
    if cons_sell is None:
        cons_sell = [CONS_SELL] * len(cons_ids)

    named = lambda table, v: table[v] if isinstance(v, str) else int(v)
    rf = run_fields()
    rf.update(
        boss=jnp.int32(named(BOSS_ID, boss)), boss_disabled=jnp.bool_(boss_disabled),
        vouchers=jnp.zeros(len(VOUCHER_ID), bool).at[jnp.array(vids, jnp.int32)].set(True) if vids
        else rf["vouchers"],
        tags=lanes([named(TAG_ID, t) for t in tags], MAX_TAGS),
        tag_hand=lanes([HAND_NAMES.index(h) if isinstance(h, str) else int(h) for h in tag_hand], MAX_TAGS, 0),
        blind_tags=lanes([named(TAG_ID, t) for t in blind_tags], 2),
        orbital_hands=lanes([HAND_NAMES.index(h) if isinstance(h, str) else int(h) for h in orbital_hands], 2, 11),
        skips=jnp.int32(skips),
        card_played_ante=jnp.zeros(MAX_DECK_SIZE, bool).at[jnp.array(cards(played_ante), jnp.int32)].set(True)
        if played_ante else rf["card_played_ante"],
        forced_card=jnp.int32(-1 if forced_card is None else card_id(forced_card)),
        hand_down=mask_lanes(hand_down), joker_debuff=mask_lanes(joker_debuff, MAX_JOKERS),
        ante_voucher=jnp.int32(named(VOUCHER_ID, ante_voucher)),
        shop_vouchers=lanes([named(VOUCHER_ID, v) for v in shop_vouchers], MAX_SHOP_VOUCHERS),
        cons_ed=lanes(list(cons_ed), MAX_CONSUMABLES, 0), ecto_minus=jnp.int32(ecto_minus),
        boss_hand=jnp.int32(boss_hand),
        ox_hand=jnp.int32(N_HAND_TYPES - 1 if ox_hand is None else
                          (HAND_NAMES.index(ox_hand) if isinstance(ox_hand, str) else ox_hand)),
        unused_discards=jnp.int32(unused_discards), round_hand_bonus=jnp.int32(round_hand_bonus),
        pack_return=jnp.int32(1 if pack_return is None else pack_return),
        shop_free=mask_lanes(shop_free, MAX_SHOP_SLOTS), pack_free=mask_lanes(pack_free, PACK_SLOTS),
        shop_rank=lanes(list(shop_rank), MAX_SHOP_SLOTS, 0), shop_suit=lanes(list(shop_suit), MAX_SHOP_SLOTS, 0),
        shop_enh=lanes(list(shop_enh), MAX_SHOP_SLOTS, 0),
        bosses_used=lanes(list(bosses_used), len(BOSS_ID), 0) if bosses_used else rf["bosses_used"],
        boss_rerolled=jnp.bool_(boss_rerolled), boss_sub=jnp.int32(boss_sub),
        reroll_inc=jnp.int32(reroll_inc), reroll_d6=jnp.bool_(reroll_d6),
        stake=jnp.int32(stake), joker_sticker=jstk, joker_perish=jper,
        back=jnp.int32(BACK_NAMES.index(back) if isinstance(back, str) else back),
        shop_sticker=lanes([sticker_bits(x) for x in shop_sticker], MAX_SHOP_SLOTS, 0, jnp.int32),
        pack_sticker=lanes([sticker_bits(x) for x in pack_sticker], MAX_PACK, 0, jnp.int32),
    )

    return balatroEnvState(
        **cards_,
        joker_ed=jed,
        hand_level=per_hand(hand_level, 1),
        last_hand=jnp.int32(last_hand),
        last_cons=jnp.int32(last_cons),
        tarots_used=jnp.int32(tarots_used),
        planets_used=per_hand(planets_used, False, bool),
        shop_kind=lanes(shop_kind, MAX_SHOP_SLOTS, 0, jnp.int32),
        shop_ed=lanes(shop_ed, MAX_SHOP_SLOTS, 0, jnp.int32),
        pack_slots=lanes(pack_slots, PACK_SLOTS, EMPTY, jnp.int32),
        buffoon_given=jnp.bool_(buffoon_given),
        pack_id=jnp.int32(-1), pack_left=jnp.int32(0),
        pack_items=jnp.full(MAX_PACK, -1, jnp.int32), pack_ed=jnp.zeros(MAX_PACK, jnp.int32),
        pack_rank=jnp.zeros(MAX_PACK, jnp.int32), pack_suit=jnp.zeros(MAX_PACK, jnp.int32),
        pack_enh=jnp.zeros(MAX_PACK, jnp.int32), pack_seal=jnp.zeros(MAX_PACK, jnp.int32),
        jokers=jids,
        joker_value=joker_value,
        joker_state=lanes(joker_state, MAX_JOKERS, 0, jnp.int32),
        joker_sell_value=joker_sell_value,
        joker_slots=jnp.int32(joker_slots),
        consumables=lanes(cons_ids, MAX_CONSUMABLES),
        cons_sell=lanes(list(cons_sell), MAX_CONSUMABLES, 0),
        consumable_slots=jnp.int32(consumable_slots),
        hand=lanes(ids),
        hand_size=jnp.int32(hand_size),
        hand_valid=mask_lanes(valid),
        deck=jnp.asarray(deck, dtype=jnp.int32),
        deck_size=jnp.int32(deck_size),
        deck_ptr=jnp.int32(deck_ptr),
        hands=jnp.int32(hands),
        discards=jnp.int32(discards),
        handcount=jnp.int32(handcount),
        discardcount=jnp.int32(discardcount),
        round=jnp.int32(round),
        ante=jnp.int32(ante),
        blind_index=jnp.int32(blind_index),
        chips_scored=jnp.int32(chips_scored),
        blind_target=jnp.int32(blind_target),
        target_scale=jnp.float32(target_scale),
        key=key,
        shop_slots=joker_ids(shop_slots, width=MAX_SHOP_SLOTS),
        shop_size=jnp.int32(shop_size),
        reroll_cost=jnp.int32(reroll_cost),
        game_stage=jnp.int32(game_stage),
        money=jnp.int32(money),
        last_cashout=jnp.int32(last_cashout),
        base_hand_size=jnp.int32(base_hand_size),
        base_hands=jnp.int32(handcount if base_hands is None else base_hands),
        base_discards=jnp.int32(discardcount if base_discards is None else base_discards),
        discards_used=jnp.int32(discards_used),
        hand_played=per_hand(hand_played),
        hand_round=per_hand(hand_round),
        idol_card=jnp.int32(idol_card), mail_rank=jnp.int32(mail_rank),
        ancient_suit=jnp.int32(ancient_suit), castle_suit=jnp.int32(castle_suit),
        free_rerolls=jnp.int32(free_rerolls),
        gros_extinct=jnp.bool_(gros_extinct),
        **rf,
    )


def select(spec):
    """A select mask for balatroAction: '01011010' or [0,1,0,1,1,0,1,0]."""
    bits = [int(c) for c in spec] if isinstance(spec, str) else list(spec)
    return mask_lanes(bits)


# ---------------------------------------------------------------------------
# COMPARING — the field-by-field diff every state test was repeating
# ---------------------------------------------------------------------------
def diff_states(got, want, label):
    """-> list of failure lines, one per field that differs. Empty list = pass.

    PRNG keys compare by their raw words, because typed keys are opaque objects
    and jnp.array_equal on two of them does not do what you want.
    """
    out = []
    for field in balatroEnvState._fields:
        g, e = getattr(got, field), getattr(want, field)
        if field == "key":
            same = bool(jnp.array_equal(jax.random.key_data(g), jax.random.key_data(e)))
        else:
            same = bool(jnp.array_equal(g, e))
        if not same:
            out.append(f"FAIL  {label}: field '{field}'\n      got:  {g}\n      want: {e}")
    return out


def report(label, failures, passed_msg):
    """Print a suite result. -> True if it passed."""
    if failures:
        print("\n".join(failures))
        print(f"{label}: {len(failures)} check(s) wrong")
        return False
    print(passed_msg)
    return True


if __name__ == "__main__":
    # self-check: the name/id convention must round-trip, and the defaults must
    # reproduce the hand the env_step test was building by hand.
    assert card_id("KD") == 24 and card_name(24) == "KD"
    assert card_id("4C") == 2 and card_id(".") == EMPTY
    assert cards("4C KD 9H KC 7C 9S 7S 3H") == [2, 24, 33, 11, 5, 46, 44, 27]

    s = state(hand="4C KD 9H KC 7C 9S 7S 3H", hands=3, chips_scored=60)
    assert int(s.hand_size) == 8 and int(s.deck_ptr) == 8
    assert bool(s.hand_valid[:8].all()) and not bool(s.hand_valid[8:].any())
    assert [int(c) for c in s.deck[8:12]] == [0, 1, 3, 4]

    holes = state(hand="2D 3D . 5D 6D . . 9D", deck=rotated_deck(13), deck_ptr=8)
    assert [bool(v) for v in holes.hand_valid[:8]] == [1, 1, 0, 1, 1, 0, 0, 1]
    assert [int(c) for c in holes.deck[8:11]] == [21, 22, 23]

    # jokers: names -> ids -> resolved values. Slot 1 is a hole, so the clip in
    # rows_for() must be cancelled by `active` and leave a 0, not a Joker's 4.
    rows, active, v = loadout(("Joker", ".", "Bootstraps"), money=23)
    assert [bool(a) for a in active[:4]] == [1, 0, 1, 0]
    assert [float(x) for x in v[:4]] == [4.0, 0.0, 8.0, 0.0]   # 4, hole, 2 * (23//5)

    print("fixtures: name round-trip, default deck, hole authoring, loadout all OK")
