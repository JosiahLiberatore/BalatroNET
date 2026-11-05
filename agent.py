"""The policy: obs encoding, the token encoder, the masked action heads, and the value head.

ACTION LAYOUT — one flat categorical over every atomic choice (NOTES: "KEEP the
flat masked action vector ... one head, one policy, phase-masked"):

    index   0                PLAY        (+ the card picker)
            1                DISCARD     (+ the card picker)
            2 .. 5           BUY shop slot 0..3 (a joker, a Tarot or a Planet)
            6 .. 37          SELL joker slot 0..31
            38               MOVE_JOKER  (+ the joker src / dst pointer heads)
            39               MOVE_CARD   (+ the card src / dst pointer heads)
            40               REROLL
            41               NEXT
            42 .. 43         OPEN pack slot 0..1
            44 .. 45         USE consumable slot 0..1   (+ the card picker for a card Tarot)
            46 .. 47         SELL_CONS consumable slot 0..1
            48 .. 52         PICK pack card 0..4        (+ the card picker for an Arcana Tarot)
            53               SKIP the pack

plus the card picker, read by PLAY / DISCARD / USE / PICK: given the action, it
picks lanes one at a time (up to MAX_PICK), each pick conditioned on the cards
picked so far, or STOP. The action sets the count range [lo, hi]: PLAY and
DISCARD 1..5, a Tarot its own (The Lovers 1..1, Death 2..2, The Star 1..3), and
0..0 for anything that takes no cards (the picker then STOPs with p = 1). The
query is per verb, plus the token of the consumable / pack card being used.
Then two pointer pairs (src over 32 slots, then dst over 32 slots given src),
read only by the MOVE verbs. legal_mask() and move_masks() mask every choice
the env would turn into a no-op.

OBS LAYOUT — a union of DISJOINT blocks, in this order (offsets O_*):
    IDS         item id + 1 per held joker (32), shop slot (4), consumable slot (2), pack card (5),
                then pack id + 1 per pack slot (2); 0 = empty. An item id is a joker id, or
                N_JOKERS + a consumable id.
    JOKER SLOT  per held slot: live value, sell value, To Do hand, Loyalty phase, edition
    CARDS       per hand lane: rank, suit, enhancement, edition, seal, perma chips
    SHOP SLOT   per shop slot: cost, affordable, kind, edition            (0 outside the shop)
    PACK SLOT   per pack slot: cost, affordable                           (0 outside the shop)
    PACK CARD   per pack card: present, a playing card's rank .. seal, a joker's edition
    CONS SLOT   per consumable slot: present
    GLOBAL      money, ante, blind, stage, scale, round targets, hand counts and levels,
                Planets / Tarots used, budget; hands/discards left, chips vs target; the
                undealt deck (rank x suit, enhancements) and the full deck's enhancements;
                reroll cost; the open pack's kind and cards left

ENCODER — every held joker, shop card, pack slot, pack card, consumable and hand
lane is one token, plus one GLOBAL token. An item token = E[item] + fixed item
features @ Wf (one table for jokers, Tarots and Planets, the same in every
slot); a card token = its card features. Every token adds a learned POSITION
vector for its slot and a TYPE vector. N_LAYERS pre-LN self-attention layers
mix the tokens; an empty slot or a hole is never a key. The trunk reads the
GLOBAL features and the GLOBAL token's output. BUY / SELL / OPEN / USE /
SELL_CONS / PICK add a score per token, the MOVE pointers score the joker /
card tokens, and the picker scores the card tokens.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from env import (balatroAction, balatroEnvState, MAX_HAND_SIZE, MAX_DECK_SIZE, MAX_SHOP_SLOTS,
                 STAGE_BLIND, STAGE_SHOP, STAGE_PACK, V_PLAY, V_DISCARD, V_BUY, V_SELL, V_REROLL,
                 V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, V_USE, V_SELL_CONS, V_OPEN, V_PICK, V_SKIP,
                 WIN_ANTE, live_v, bankrupt_at, reroll_price, item_cost, pack_cost, cons_room,
                 can_use_base, pick_ok, deck_counts, affordable, shop_item_cost, card_lanes)
from items import (N_HAND_TYPES, N_CONS, CONS_KIND, CONS_USE, CONS_PARAM, CONS_LO, CONS_HI, N_ENH,
                   N_ED, N_SEAL, N_PACKS, PACK_KIND, PACK_SLOTS, MAX_PACK, PK_ARCANA, IK_JOKER,
                   MAX_JOKER_SLOTS, ED_NEG, U_ENHANCE, U_SUIT, U_PLANET, MAX_PICK_USE)
from jokers import (MAX_JOKERS, N_JOKERS, J_EMPTY, JOKER_COST, JOKERS, JOKER_RARITY, ST_TODO,
                    ST_LOYALTY, CONDITION_NAMES, C_SUIT, C_RANK_IN, C_HAND_HAS,
                    C_COUNT_CMP, C_FACES_GE, C_LOYALTY, C_POSITION, C_COUNT_EQ)

# --- action layout ------------------------------------------------------------
K_CONS = 2                                  # consumable slots the agent sees (the env has 2)
A_PLAY, A_DISCARD = 0, 1
A_BUY = 2                                   # + shop slot
A_SELL = A_BUY + MAX_SHOP_SLOTS             # + joker slot
A_MOVE_JOKER = A_SELL + MAX_JOKERS          # src / dst from the joker pointer heads
A_MOVE_CARD = A_MOVE_JOKER + 1              # src / dst from the card pointer heads
A_REROLL = A_MOVE_CARD + 1
A_NEXT = A_REROLL + 1
A_OPEN = A_NEXT + 1                         # + pack slot
A_USE = A_OPEN + PACK_SLOTS                 # + consumable slot
A_SELL_CONS = A_USE + K_CONS                # + consumable slot
A_PICK = A_SELL_CONS + K_CONS               # + pack card
A_SKIP = A_PICK + MAX_PACK
N_ACTIONS = A_SKIP + 1                      # 54
N_PTR = MAX_JOKERS                          # pointer width; MAX_JOKERS == MAX_HAND_SIZE == 32
assert MAX_HAND_SIZE == N_PTR
MAX_PICK = 5                                # the most cards one action picks (PLAY / DISCARD)
STOP = MAX_HAND_SIZE                        # the picker's "no more cards" choice
PICK_D = 64                                 # the picker's hidden width
PV_PLAY, PV_DISCARD, PV_USE, PV_PICK = range(4)   # picker query per verb

_idx = jnp.arange(N_ACTIONS)
_in = lambda lo, hi: (_idx >= lo) & (_idx < hi)
ACTION_VERB = jnp.select(
    [_idx == A_PLAY, _idx == A_DISCARD, _in(A_BUY, A_SELL), _in(A_SELL, A_MOVE_JOKER),
     _idx == A_MOVE_JOKER, _idx == A_MOVE_CARD, _idx == A_REROLL, _idx == A_NEXT,
     _in(A_OPEN, A_USE), _in(A_USE, A_SELL_CONS), _in(A_SELL_CONS, A_PICK), _in(A_PICK, A_SKIP)],
    [V_PLAY, V_DISCARD, V_BUY, V_SELL, V_MOVE_JOKER, V_MOVE_CARD, V_REROLL, V_NEXT,
     V_OPEN, V_USE, V_SELL_CONS, V_PICK], V_SKIP).astype(jnp.int32)
ACTION_TARGET = jnp.select(
    [_in(A_BUY, A_SELL), _in(A_SELL, A_MOVE_JOKER), _in(A_OPEN, A_USE), _in(A_USE, A_SELL_CONS),
     _in(A_SELL_CONS, A_PICK), _in(A_PICK, A_SKIP)],
    [_idx - A_BUY, _idx - A_SELL, _idx - A_OPEN, _idx - A_USE, _idx - A_SELL_CONS, _idx - A_PICK],
    0).astype(jnp.int32)


def is_move(a):
    return (a == A_MOVE_JOKER) | (a == A_MOVE_CARD)


def uses_picker(a):
    return (a == A_PLAY) | (a == A_DISCARD) | ((a >= A_USE) & (a < A_SELL_CONS)) | ((a >= A_PICK) & (a < A_SKIP))


def pick_verb(a):
    return jnp.select([a == A_DISCARD, (a >= A_USE) & (a < A_SELL_CONS), (a >= A_PICK) & (a < A_SKIP)],
                      [PV_DISCARD, PV_USE, PV_PICK], PV_PLAY).astype(jnp.int32)


def action_name(a, src=None, dst=None):
    a = int(a)
    if a == A_PLAY:    return "PLAY"
    if a == A_DISCARD: return "DISCARD"
    if a < A_SELL:     return f"BUY {a - A_BUY}"
    if a < A_MOVE_JOKER: return f"SELL {a - A_SELL}"
    if a in (A_MOVE_JOKER, A_MOVE_CARD):
        k = "J" if a == A_MOVE_JOKER else "C"
        return f"MOVE {k}" if src is None else f"MOVE {k}{int(src)}->{k}{int(dst)}"
    if a == A_REROLL:  return "REROLL"
    if a == A_NEXT:    return "NEXT"
    if a < A_USE:      return f"OPEN {a - A_OPEN}"
    if a < A_SELL_CONS: return f"USE {a - A_USE}"
    if a < A_PICK:     return f"SELL_C {a - A_SELL_CONS}"
    if a < A_SKIP:     return f"PICK {a - A_PICK}"
    return "SKIP"


def pick_bounds(state: balatroEnvState, a):
    """(lo, hi): how many cards action a picks. PLAY / DISCARD 1..5; a card Tarot
    (USE, or PICK in an Arcana pack) its own range; anything else 0..0."""
    cid = state.consumables[jnp.clip(a - A_USE, 0, K_CONS - 1)]
    item = state.pack_items[jnp.clip(a - A_PICK, 0, MAX_PACK - 1)]
    arcana = PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)] == PK_ARCANA
    c_use = jnp.clip(cid, 0, N_CONS - 1)
    c_pick = jnp.clip(item, 0, N_CONS - 1)
    use = (a >= A_USE) & (a < A_SELL_CONS)
    pick = (a >= A_PICK) & (a < A_SKIP) & arcana
    card = (a == A_PLAY) | (a == A_DISCARD)
    lo = jnp.select([card, use, pick], [1, CONS_LO[c_use], CONS_LO[c_pick]], 0)
    hi = jnp.select([card, use, pick], [MAX_PICK, CONS_HI[c_use], CONS_HI[c_pick]], 0)
    return lo.astype(jnp.int32), hi.astype(jnp.int32)


def legal_mask(state: balatroEnvState) -> jax.Array:
    """(N_ACTIONS,) bool — the choices env_step would actually carry out.

    Mirrors env_step's repair rules; an unmasked choice the env turns into a
    no-op is not a bug, only a wasted step. When no choice is legal (LOST/WON,
    the episode is over and gets reset), NEXT stays open so the categorical is
    never fully masked.
    """
    blind = state.game_stage == STAGE_BLIND
    shop = state.game_stage == STAGE_SHOP
    pack = state.game_stage == STAGE_PACK
    held = (state.jokers != J_EMPTY)

    slots = jnp.arange(MAX_SHOP_SLOTS)
    item, kind, ed = state.shop_slots, state.shop_kind, state.shop_ed
    cost = jax.vmap(lambda k: shop_item_cost(state, k))(slots)          # a couponed card is $0
    neg = (ed == ED_NEG) & (state.joker_slots < MAX_JOKER_SLOTS)
    room = jnp.where(kind == IK_JOKER, held.sum() < state.joker_slots + neg, cons_room(state) > 0)
    buy = shop & (slots < state.shop_size) & (item >= 0) & affordable(state, cost) & room
    sell = (blind | shop | pack) & held
    move = (blind | shop | pack) & (held.sum() >= 2)
    move_card = (blind | pack) & (card_lanes(state) >= 2) & state.hand_valid.any()
    pid = state.pack_slots
    opn = shop & (pid >= 0) & affordable(state, jax.vmap(lambda p, f: pack_cost(state, p, f))(pid, state.pack_free))
    cons = state.consumables[:K_CONS]
    present = cons >= 0
    use = (blind | shop | pack) & present & jax.vmap(lambda c: can_use_base(state, c, True))(cons)
    sell_c = (blind | shop | pack) & present
    pick = pack & jax.vmap(lambda t: pick_ok(state, t))(jnp.arange(MAX_PACK))

    legal = jnp.concatenate([
        jnp.stack([blind, blind & (state.discards > 0)]),
        buy,
        sell,
        jnp.stack([move, move_card]),
        jnp.stack([shop & ((state.free_rerolls > 0) | affordable(state, reroll_price(state))), shop]),
        opn, use, sell_c, pick, jnp.stack([pack]),
    ])
    return legal.at[A_NEXT].set(legal[A_NEXT] | ~legal.any())


def move_masks(state: balatroEnvState) -> jax.Array:
    """(4, N_PTR) bool — rows: joker src, joker dst, card src, card dst.

    The dst rows are the slots a move may END at; dst_mask() also removes src.
    Joker: any held slot (held jokers are packed). Card: src must hold a card,
    dst is any lane < env.card_lanes (the hand size, or up to the last card past it;
    holes shift like cards).
    """
    held = state.jokers != J_EMPTY
    lanes = jnp.arange(N_PTR) < card_lanes(state)
    return jnp.stack([held, held, state.hand_valid & lanes, lanes])


def dst_mask(base, src):
    return base & (jnp.arange(N_PTR) != src)


# --- observation ----------------------------------------------------------------
N_ITEMS = N_JOKERS + N_CONS                 # one id space: jokers, then Tarots and Planets
ID_N = MAX_JOKERS + MAX_SHOP_SLOTS + K_CONS + MAX_PACK + PACK_SLOTS
JS_F = 2 + N_HAND_TYPES + 1 + N_ED          # per held slot: value, sell, To Do hand, Loyalty, edition
CARD_F = 13 + 4 + N_ENH + 4 + N_SEAL + 1    # rank, suit, enhancement, edition, seal, perma
SHOP_F = 2 + 3 + N_ED                       # cost, affordable, kind, edition
PS_F = 2                                    # pack slot: cost, affordable
PI_F = 1 + CARD_F + N_ED                    # pack card: present, a playing card, a joker's edition
CS_F = 1                                    # consumable slot: present
O_JS = ID_N
O_CARD = O_JS + MAX_JOKERS * JS_F
O_SHOP = O_CARD + MAX_HAND_SIZE * CARD_F
O_PS = O_SHOP + MAX_SHOP_SLOTS * SHOP_F
O_PI = O_PS + PACK_SLOTS * PS_F
O_CS = O_PI + MAX_PACK * PI_F
O_GLOB = O_CS + K_CONS * CS_F
# id offsets inside IDS
I_HELD, I_SHOP = 0, MAX_JOKERS
I_CONS = I_SHOP + MAX_SHOP_SLOTS
I_PACK = I_CONS + K_CONS
I_PSLOT = I_PACK + MAX_PACK


def _card_feats(rank, suit, enh, ed, seal, bonus, ok):
    """(K, CARD_F) playing card features; zero where not ok."""
    f = jnp.concatenate([jax.nn.one_hot(rank, 13), jax.nn.one_hot(suit, 4), jax.nn.one_hot(enh, N_ENH),
                         jax.nn.one_hot(ed, 4), jax.nn.one_hot(seal, N_SEAL),
                         (bonus / 50.0)[:, None]], axis=1)
    return f * ok[:, None]


def _joker_slot_block(state: balatroEnvState) -> jax.Array:
    """(MAX_JOKERS, JS_F) the per-slot joker state; zero for an empty slot."""
    held = state.jokers != J_EMPTY
    kind = JOKERS.STATE[jnp.clip(state.jokers, 0, N_JOKERS - 1)]
    todo = jax.nn.one_hot(state.joker_state, N_HAND_TYPES) * (held & (kind == ST_TODO))[:, None]
    loyal = jnp.where(held & (kind == ST_LOYALTY),
                      jnp.mod(state.hand_played.sum() - state.joker_state, 6) / 5.0, 0.0)
    return jnp.concatenate([(live_v(state) / 10.0)[:, None],
                            (state.joker_sell_value / 10.0)[:, None],
                            todo, loyal[:, None], jax.nn.one_hot(state.joker_ed, N_ED)],
                           axis=1) * held[:, None]


def _shared_block(state: balatroEnvState) -> jax.Array:
    f32 = jnp.float32
    blind = (state.game_stage == STAGE_BLIND).astype(f32)
    shop = (state.game_stage == STAGE_SHOP).astype(f32)
    pack = (state.game_stage == STAGE_PACK).astype(f32)
    held = state.jokers != J_EMPTY
    pk = jnp.clip(state.pack_id, 0, N_PACKS - 1)
    shared = jnp.concatenate([
        jnp.stack([state.money / 25.0, jnp.log1p(jnp.maximum(state.money, 0)) / 4.0]),
        jax.nn.one_hot(jnp.clip(state.ante, 1, WIN_ANTE) - 1, WIN_ANTE),
        jax.nn.one_hot(state.blind_index, 3),
        jnp.stack([blind, shop, pack, state.target_scale]),
        jnp.stack([held.sum() / state.joker_slots, state.joker_slots / 5.0]),
        # the round targets (The Idol, Mail-In Rebate, Ancient Joker, Castle)
        jax.nn.one_hot(state.idol_card % 13, 13), jax.nn.one_hot(state.idol_card // 13, 4),
        jax.nn.one_hot(state.mail_rank, 13), jax.nn.one_hot(state.ancient_suit, 4),
        jax.nn.one_hot(state.castle_suit, 4),
        state.hand_played / 10.0, state.hand_round / 4.0, (state.hand_level - 1) / 5.0,
        state.planets_used.astype(f32), jax.nn.one_hot(state.last_hand, N_HAND_TYPES),
        jnp.stack([state.hand_size / 10.0, state.handcount / 5.0, state.discardcount / 5.0,
                   state.discards_used / 3.0, state.free_rerolls.astype(f32),
                   bankrupt_at(state) / 20.0, state.gros_extinct.astype(f32),
                   state.tarots_used / 10.0, cons_room(state) / 2.0]),
        jax.nn.one_hot(PACK_KIND[pk], 4) * pack, jnp.stack([state.pack_left * pack / 2.0]),
    ]).astype(f32)
    return shared


def _deck_block(state: balatroEnvState) -> jax.Array:
    """The undealt cards (rank x suit counts, enhancement counts) and the full deck's enhancements."""
    pos = jnp.arange(MAX_DECK_SIZE)
    inst = state.deck
    undealt = (pos >= state.deck_ptr) & (pos < state.deck_size)
    card = state.card_rank[inst] + 13 * state.card_suit[inst]
    stone = state.card_enh[inst] == 6
    by_card = jnp.zeros(52).at[card].add(undealt & ~stone)
    by_enh = jnp.zeros(N_ENH).at[state.card_enh[inst]].add(undealt)
    full = jnp.zeros(N_ENH).at[state.card_enh].add(state.card_live)
    return jnp.concatenate([by_card, by_enh / 10.0, full / 10.0,
                            jnp.stack([deck_counts(state)["size"] / 52.0])])


def observe(state: balatroEnvState) -> jax.Array:
    f32 = jnp.float32
    blind = (state.game_stage == STAGE_BLIND).astype(f32)
    shop = (state.game_stage == STAGE_SHOP).astype(f32)
    pack = (state.game_stage == STAGE_PACK).astype(f32)
    in_shop = state.game_stage == STAGE_SHOP

    # IDS (item id + 1, 0 = empty; the shop ids only in the shop, the pack ids only with a pack open)
    held_id = jnp.where(state.jokers != J_EMPTY, state.jokers + 1, 0)
    shop_item = jnp.where(state.shop_kind == IK_JOKER, state.shop_slots, N_JOKERS + state.shop_slots)
    shop_id = jnp.where((state.shop_slots >= 0) & in_shop, shop_item + 1, 0)
    cons = state.consumables[:K_CONS]
    cons_id = jnp.where(cons >= 0, N_JOKERS + cons + 1, 0)
    pk = PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)]
    open_ = state.game_stage == STAGE_PACK
    items = state.pack_items
    pack_item = jnp.select([pk == 3, (pk == 0) | (pk == 1)], [items, N_JOKERS + items], -1)
    pack_id = jnp.where(open_ & (items >= 0) & (pack_item >= 0), pack_item + 1, 0)
    pslot_id = jnp.where((state.pack_slots >= 0) & in_shop, state.pack_slots + 1, 0)
    ids = jnp.concatenate([held_id, shop_id, cons_id, pack_id, pslot_id]).astype(f32)

    # CARDS (the hand)
    i = jnp.clip(state.hand, 0, MAX_DECK_SIZE - 1)
    cards = _card_feats(state.card_rank[i], state.card_suit[i], state.card_enh[i], state.card_ed[i],
                        state.card_seal[i], state.card_bonus[i], state.hand_valid)

    # SHOP SLOT / PACK SLOT (in the shop)
    on = state.shop_slots >= 0
    cost = item_cost(state, state.shop_kind, state.shop_slots, state.shop_ed) * on
    shop_slot = jnp.concatenate([jnp.stack([cost / 10.0, (on & (state.money >= cost)).astype(f32)], axis=1),
                                 jax.nn.one_hot(state.shop_kind, 3) * on[:, None],
                                 jax.nn.one_hot(state.shop_ed, N_ED) * on[:, None]], axis=1) * shop
    pon = state.pack_slots >= 0
    pcost = jax.vmap(lambda p: pack_cost(state, p))(state.pack_slots) * pon
    pack_slot = jnp.stack([pcost / 10.0, (pon & (state.money >= pcost)).astype(f32)], axis=1) * shop

    # PACK CARD (a pack open)
    present = open_ & (items >= 0)
    std = pk == 2
    pcard = _card_feats(state.pack_rank, state.pack_suit, state.pack_enh, jnp.minimum(state.pack_ed, 3),
                        state.pack_seal, jnp.zeros(MAX_PACK), present & std)
    pjed = jax.nn.one_hot(state.pack_ed, N_ED) * (present & (pk == 3))[:, None]
    pack_card = jnp.concatenate([present[:, None].astype(f32), pcard, pjed], axis=1)

    # CONS SLOT
    cons_slot = (cons >= 0).astype(f32)[:, None]

    # GLOBAL
    blind_g = jnp.stack([state.hands / state.handcount,
                         state.discards / state.discardcount,
                         jnp.clip(state.chips_scored / state.blind_target, 0.0, 2.0),
                         jnp.log10(state.blind_target.astype(f32)) / 6.0]) * blind
    shop_g = jnp.stack([reroll_price(state) / 10.0,
                        (state.money >= reroll_price(state)).astype(f32)]) * shop

    return jnp.concatenate([ids, _joker_slot_block(state).ravel(), cards.ravel(), shop_slot.ravel(),
                            pack_slot.ravel(), pack_card.ravel(), cons_slot.ravel(),
                            _shared_block(state), blind_g, _deck_block(state) * (blind + pack),
                            shop_g]).astype(f32)


def _obs_dim():
    from fixtures import state as make_state
    return int(observe(make_state(hand="2C")).shape[0])


def joker_ids(obs):
    """(MAX_JOKERS + MAX_SHOP_SLOTS,) the held and shop item ids, -1 = empty (read back from obs)."""
    return obs[:I_CONS].astype(jnp.int32) - 1


def hand_feats(obs) -> jax.Array:
    """(MAX_HAND_SIZE, CARD_F) the card features per lane, read back from obs
    (all zero for a hole and in the shop). Columns 0..12 rank, 13..16 suit."""
    return obs[O_CARD:O_SHOP].reshape(MAX_HAND_SIZE, CARD_F)


# --- the fixed item features ("the card text") ------------------------------------
_COUNT_CONDS = (C_COUNT_CMP, C_FACES_GE, C_LOYALTY, C_POSITION, C_COUNT_EQ)


def _cond_bits(cond, param):
    """(N, 4 + 13 + N_HAND_TYPES + 1) what COND_PARAM means for its condition:
    the suit, the rank mask, the hand type, or a count."""
    f32 = np.float32
    suit = np.eye(4, dtype=f32)[np.clip(param, 0, 3)] * (cond == C_SUIT)[:, None]
    rank = ((param[:, None] >> np.arange(13)) & 1).astype(f32) * (cond == C_RANK_IN)[:, None]
    hand = (np.eye(N_HAND_TYPES, dtype=f32)[np.clip(param, 0, N_HAND_TYPES - 1)]
            * (cond == C_HAND_HAS)[:, None])
    count = (param / 5.0 * np.isin(cond, _COUNT_CONDS)).astype(f32)[:, None]
    return np.concatenate([suit, rank, hand, count], axis=1)


def _joker_feats():
    """(N_JOKERS, F) fixed features from the compiled rows: what each joker IS
    (trigger, condition and its param, effect, source, query, state, rarity,
    magnitudes, update rules and their conditions, the enhancement it needs)."""
    t = {f: np.asarray(getattr(JOKERS, f)) for f in JOKERS._fields}
    oh = lambda x: np.eye(int(x.max()) + 1, dtype=np.float32)[x]
    n_cond = len(CONDITION_NAMES)
    upd = t["UPD_TRIGGER"] >= 0
    upd_cond = np.concatenate([np.eye(n_cond, dtype=np.float32)[t["UPD_COND"][:, r]]
                               * upd[:, r:r + 1] for r in range(upd.shape[1])], axis=1)
    upd_bits = np.concatenate([_cond_bits(t["UPD_COND"][:, r], t["UPD_PARAM"][:, r])
                               * upd[:, r:r + 1] for r in range(upd.shape[1])], axis=1)
    cols = [oh(t["TRIGGER"]), np.eye(n_cond, dtype=np.float32)[t["CONDITION"]],
            _cond_bits(t["CONDITION"], t["COND_PARAM"]),
            oh(t["EFFECT"]), oh(t["SOURCE"]),
            oh(t["QUERY"]), oh(t["STATE"]), oh(t["PROB"]), oh(np.asarray(JOKER_RARITY)),
            oh(t["POOL"]),
            np.stack([np.log1p(t["VALUE"]) / 3.0, t["VALUE2"] / 4.0, t["COPYABLE"],
                      np.asarray(JOKER_COST) / 10.0, t["H_SIZE"] / 2.0, t["D_SIZE"] / 3.0,
                      t["H_PLAYS"], t["HSIZE_V"], t["EXPIRE_ODDS"] > 0,
                      t["EXPIRE_AT"] > -1e8, t["QBASE"]], axis=1).astype(np.float32),
            upd.astype(np.float32), (t["UPD_DELTA"] * upd / 5.0).astype(np.float32),
            oh(t["UPD_MODE"]).reshape(N_JOKERS, -1) * np.repeat(upd, int(t["UPD_MODE"].max()) + 1, axis=1),
            upd_cond, upd_bits, np.eye(N_ENH + 1, dtype=np.float32)[t["GATE_ENH"] + 1]]
    return np.concatenate(cols, axis=1)


def _cons_feats():
    """(N_CONS, F) a Tarot / Planet: its kind, what USE does, the enhancement /
    suit / hand it targets, how many cards it takes."""
    kind, use, prm = np.asarray(CONS_KIND), np.asarray(CONS_USE), np.asarray(CONS_PARAM)
    f32 = np.float32
    return np.concatenate([
        np.eye(int(kind.max()) + 1, dtype=f32)[kind], np.eye(int(use.max()) + 1, dtype=f32)[use],
        np.eye(N_ENH, dtype=f32)[np.clip(prm, 0, N_ENH - 1)] * (use == U_ENHANCE)[:, None],
        np.eye(4, dtype=f32)[np.clip(prm, 0, 3)] * (use == U_SUIT)[:, None],
        np.eye(N_HAND_TYPES, dtype=f32)[np.clip(prm, 0, N_HAND_TYPES - 1)] * (use == U_PLANET)[:, None],
        np.stack([np.asarray(CONS_LO) / 3.0, np.asarray(CONS_HI) / 3.0, prm / 50.0], axis=1)], axis=1)


def _item_feats():
    j, c = _joker_feats(), _cons_feats()
    top = np.concatenate([j, np.zeros((N_JOKERS, c.shape[1]), np.float32), np.ones((N_JOKERS, 1), np.float32)], 1)
    bot = np.concatenate([np.zeros((N_CONS, j.shape[1]), np.float32), c, np.zeros((N_CONS, 1), np.float32)], 1)
    return jnp.asarray(np.concatenate([top, bot], axis=0))


ITEM_FEATS = _item_feats()
ITEM_F = int(ITEM_FEATS.shape[1])
JOKER_FEATS = ITEM_FEATS[:N_JOKERS]


# --- the token encoder --------------------------------------------------------------
TOK_D = 64                                  # token width
N_HEADS = 4
N_LAYERS = 2
FF_D = 128                                  # the feed-forward width in each layer
# Tokens only for the slots the env can fill (held jokers are packed, lanes <
# hand_size); the obs and the action layout keep all 32. tests_run checks the caps.
J_TOK = 16                                  # held joker tokens (items.MAX_JOKER_SLOTS: 5 slots + Antimatter + Negatives)
C_TOK = 20                                  # hand lane tokens (8 + the largest hand size bonus + added cards)
N_TOK = 1 + J_TOK + MAX_SHOP_SLOTS + PACK_SLOTS + MAX_PACK + K_CONS + C_TOK
T_HELD, T_SHOP, T_PSLOT, T_PACK, T_CONS, T_CARD, T_GLOB = range(7)   # rows of the TYPE table


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * g + b


# a masked choice gets this logit, not -inf: its probability is still exactly 0,
# but p * log p stays 0 * finite in the backward pass (0 * -inf would be NaN)
MASKED = -1e9


@jax.checkpoint     # recompute in the backward pass: the (N_TOK, N_TOK) maps stay off the GPU
def _layer(p, x, key_ok):
    """One pre-LN self-attention + feed-forward layer. x (N_TOK, TOK_D)."""
    dh = TOK_D // N_HEADS
    q, k, v = jnp.split(_ln(x, p["g1"], p["b1"]) @ p["Wqkv"], 3, axis=-1)
    q, k, v = (z.reshape(N_TOK, N_HEADS, dh) for z in (q, k, v))
    s = jnp.einsum("qhd,khd->hqk", q, k) / jnp.sqrt(dh)
    s = jnp.where(key_ok[None, None, :], s, MASKED)
    o = jnp.einsum("hqk,khd->qhd", jax.nn.softmax(s, axis=-1), v).reshape(N_TOK, TOK_D)
    x = x + o @ p["Wo"]
    y = _ln(x, p["g2"], p["b2"])
    return x + jax.nn.gelu(y @ p["Wf1"] + p["bf1"]) @ p["Wf2"] + p["bf2"]


def _item_emb(params, ids):
    """E[item] + fixed features @ Wf; zero for an empty slot (id < 0). The table is
    projected first (N_ITEMS x TOK_D), then gathered: a token never carries the
    ITEM_F-wide features (that gather was a 4 GiB batch buffer)."""
    j = jnp.clip(ids, 0, N_ITEMS - 1)
    table = jnp.asarray(params["E"]) + ITEM_FEATS @ params["Wf"]
    return table[j] * (ids >= 0)[:, None]


def tokens(params, obs):
    """-> (tok (N_TOK, TOK_D), key_ok (N_TOK,)) before the attention layers.
    Order: GLOBAL, held slots, shop slots, pack slots, pack cards, consumables, hand lanes."""
    ids = obs[:ID_N].astype(jnp.int32) - 1
    Ty = jnp.asarray(params["Ty"])
    held_i = ids[I_HELD:I_HELD + J_TOK]
    shop_i = ids[I_SHOP:I_CONS]
    cons_i = ids[I_CONS:I_PACK]
    pack_i = ids[I_PACK:I_PSLOT]
    pslot_i = ids[I_PSLOT:ID_N]
    js = obs[O_JS:O_CARD].reshape(MAX_JOKERS, JS_F)[:J_TOK]
    x = hand_feats(obs)[:C_TOK]
    sf = obs[O_SHOP:O_PS].reshape(MAX_SHOP_SLOTS, SHOP_F)
    psf = obs[O_PS:O_PI].reshape(PACK_SLOTS, PS_F)
    pif = obs[O_PI:O_CS].reshape(MAX_PACK, PI_F)
    held = _item_emb(params, held_i) + js @ params["Wjs"] + params["Pj"] + Ty[T_HELD]
    shop = _item_emb(params, shop_i) + sf @ params["Wsh"] + params["Ps"] + Ty[T_SHOP]
    ep = jnp.asarray(params["Ep"])[jnp.clip(pslot_i, 0, N_PACKS - 1)] * (pslot_i >= 0)[:, None]
    pslot = ep + psf @ params["Wps"] + params["Pps"] + Ty[T_PSLOT]
    pack = _item_emb(params, pack_i) + pif @ params["Wpi"] + params["Ppi"] + Ty[T_PACK]
    cons = _item_emb(params, cons_i) + params["Pcs"] + Ty[T_CONS]
    cards = x @ params["Wcd"] + params["Pc"] + Ty[T_CARD]
    glob = obs[O_GLOB:] @ params["Wg"] + params["bg"] + Ty[T_GLOB]
    tok = jnp.concatenate([glob[None], held, shop, pslot, pack, cons, cards])
    key_ok = jnp.concatenate([jnp.ones(1, bool), held_i >= 0, shop_i >= 0, pslot_i >= 0,
                              pif[:, 0] > 0, cons_i >= 0, x.sum(1) > 0])
    return tok, key_ok


class Encoded(NamedTuple):
    h     : jax.Array   # (HIDDEN,)  the trunk
    jt    : jax.Array   # (MAX_JOKERS, TOK_D)  held joker tokens (zero past J_TOK)
    st    : jax.Array   # (MAX_SHOP_SLOTS, TOK_D)  shop card tokens
    pst   : jax.Array   # (PACK_SLOTS, TOK_D)  pack slot tokens
    pit   : jax.Array   # (MAX_PACK, TOK_D)  pack card tokens
    cst   : jax.Array   # (K_CONS, TOK_D)  consumable tokens
    ct    : jax.Array   # (MAX_HAND_SIZE, TOK_D)  hand lane tokens (zero past C_TOK)


def encode(params, obs) -> Encoded:
    tok, key_ok = tokens(params, obs)
    for p in params["layers"]:
        tok = _layer(p, tok, key_ok)
    tok = _ln(tok, params["gN"], params["bN"])
    h = jnp.tanh(obs[O_GLOB:] @ params["W1"] + tok[0] @ params["W1t"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    cuts = [int(x) for x in np.cumsum([1, J_TOK, MAX_SHOP_SLOTS, PACK_SLOTS, MAX_PACK, K_CONS])]
    pad = lambda t, n: jnp.pad(t, ((0, n - t.shape[0]), (0, 0)))
    jt, st, pst, pit, cst, ct = (tok[a:b] for a, b in zip(cuts, cuts[1:] + [N_TOK]))
    return Encoded(h, pad(jt, MAX_JOKERS), st, pst, pit, cst, pad(ct, MAX_HAND_SIZE))


# --- network --------------------------------------------------------------------
HIDDEN = 256
N_SCORED = 6                                # token-scored action groups: BUY, SELL, OPEN, USE, SELL_CONS, PICK


def init_params(key, obs_dim=None):
    obs_dim = obs_dim or OBS_DIM
    g_dim = obs_dim - O_GLOB
    ks = iter(jax.random.split(key, 80))
    def dense(n_in, n_out, scale):
        return jax.random.normal(next(ks), (n_in, n_out)) * scale / jnp.sqrt(n_in), jnp.zeros(n_out)
    def table(*shape, scale=0.5):
        return jax.random.normal(next(ks), shape) * scale
    D = TOK_D

    def layer():
        Wqkv, _ = dense(D, 3 * D, 1.0)
        Wo, _ = dense(D, D, 1.0 / jnp.sqrt(2 * N_LAYERS))
        Wf1, bf1 = dense(D, FF_D, 1.0)
        Wf2, bf2 = dense(FF_D, D, 1.0 / jnp.sqrt(2 * N_LAYERS))
        return {"g1": jnp.ones(D), "b1": jnp.zeros(D), "Wqkv": Wqkv, "Wo": Wo,
                "g2": jnp.ones(D), "b2": jnp.zeros(D), "Wf1": Wf1, "bf1": bf1, "Wf2": Wf2, "bf2": bf2}

    p = {
        # tokens
        "E": table(N_ITEMS, D),                          # the shared item embedding (jokers, Tarots, Planets)
        "Wf": dense(ITEM_F, D, 0.5)[0],                  # fixed item features -> token
        "Ep": table(N_PACKS, D),                         # the pack embedding
        "Wjs": dense(JS_F, D, 1.0)[0], "Wsh": dense(SHOP_F, D, 1.0)[0],
        "Wps": dense(PS_F, D, 1.0)[0], "Wpi": dense(PI_F, D, 1.0)[0],
        "Wcd": dense(CARD_F, D, 1.0)[0],
        "Wg": dense(g_dim, D, 1.0)[0], "bg": jnp.zeros(D),
        "Pj": table(J_TOK, D), "Ps": table(MAX_SHOP_SLOTS, D), "Pps": table(PACK_SLOTS, D),
        "Ppi": table(MAX_PACK, D), "Pcs": table(K_CONS, D), "Pc": table(C_TOK, D),   # position per slot
        "Ty": table(7, D),
        "layers": [layer() for _ in range(N_LAYERS)],
        "gN": jnp.ones(D), "bN": jnp.zeros(D),
        # trunk
        "W1": dense(g_dim, HIDDEN, 1.0)[0], "W1t": dense(D, HIDDEN, 1.0)[0], "b1": jnp.zeros(HIDDEN),
    }
    p["W2"], p["b2"] = dense(HIDDEN, HIDDEN, 1.0)
    p["Wa"], p["ba"] = dense(HIDDEN, N_ACTIONS, 0.01)          # small: start near-uniform
    p["Wbs"] = dense(HIDDEN, N_SCORED * D, 0.01)[0]            # per-token action queries
    p["Wp"], p["bp"] = dense(HIDDEN, 4 * D, 0.01)              # joker src, joker dst, card src, card dst
    p["Rp"] = table(2, D, D, scale=0.01 / D ** 0.5)            # dst query given the src token
    p["Pm"] = table(2, N_PTR, N_PTR, scale=0.01)               # dst bias given src: [kind, src, dst]
    # the card picker
    p["Wq"], p["bq"] = dense(HIDDEN, 4 * PICK_D, 1.0)          # picker query per verb (PLAY, DISCARD, USE, PICK)
    p["Wqi"] = dense(D, PICK_D, 1.0)[0]                        # + the token of the consumable / pack card used
    p["Wu"], p["bu"] = dense(D, PICK_D, 1.0)                   # one lane's token
    p["Wc"] = dense(D, PICK_D, 1.0)[0]                         # the tokens picked so far (summed)
    p["Wpk"] = dense(D, PICK_D, 1.0)[0]                        # a lane attends over the picked tokens
    p["Wpv"] = dense(D, PICK_D, 1.0)[0]
    p["Wn"] = table(MAX_PICK + 1, PICK_D)                      # how many picked so far
    p["vo"], p["vs"], p["bstop"] = jnp.zeros(PICK_D), jnp.zeros(PICK_D), jnp.zeros(1)   # start uniform
    p["Wv"], p["bv"] = dense(HIDDEN, 1, 1.0)
    return p


class Heads(NamedTuple):
    """Masked logits of one state. ptr rows: joker src, joker dst | src, card src, card dst | src.
    q / qi / u / x / k / w feed the card picker (pick_logits)."""
    a     : jax.Array   # (N_ACTIONS,)
    q     : jax.Array   # (4, PICK_D)  picker query per verb: PLAY, DISCARD, USE, PICK
    qi    : jax.Array   # (K_CONS + MAX_PACK, PICK_D)  + the consumable / pack card token
    u     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane query
    x     : jax.Array   # (MAX_HAND_SIZE, TOK_D)   per-lane token
    k     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane key   (for the lanes picked so far)
    w     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane value (for the lanes picked so far)
    ptr   : jax.Array   # (4, N_PTR)
    value : jax.Array   # ()


def policy(params, obs, src, enc: Encoded = None):
    """obs -> raw (action (N_ACTIONS,), picker query (4, PICK_D), pointer (4, 32), value (), Encoded).
    The two dst rows are conditioned on src (the joker src for row 1, the card
    src for row 3; one src serves both, only the verb's own pair is read).
    enc = encode(params, obs) when the caller already has it."""
    enc = encode(params, obs) if enc is None else enc
    h, jt, st, pst, pit, cst, ct = enc
    D = TOK_D
    qb = (h @ params["Wbs"]).reshape(N_SCORED, D) / jnp.sqrt(D)
    a = h @ params["Wa"] + params["ba"]
    a = (a.at[A_BUY:A_SELL].add(st @ qb[0]).at[A_SELL:A_MOVE_JOKER].add(jt @ qb[1])
          .at[A_OPEN:A_USE].add(pst @ qb[2]).at[A_USE:A_SELL_CONS].add(cst @ qb[3])
          .at[A_SELL_CONS:A_PICK].add(cst @ qb[4]).at[A_PICK:A_SKIP].add(pit @ qb[5]))
    qp = (h @ params["Wp"] + params["bp"]).reshape(4, D)
    s = jnp.clip(src, 0, N_PTR - 1)
    Pm, Rp = jnp.asarray(params["Pm"]), jnp.asarray(params["Rp"])   # loaded params are numpy
    ptr = jnp.stack([jt @ qp[0],
                     jt @ (qp[1] + jt[s] @ Rp[0]) + Pm[0, s],
                     ct @ qp[2],
                     ct @ (qp[3] + ct[s] @ Rp[1]) + Pm[1, s]])
    return (a, (h @ params["Wq"] + params["bq"]).reshape(4, PICK_D), ptr,
            (h @ params["Wv"] + params["bv"])[0], enc)


def masked_logits(params, obs, legal, hand_valid, mmask, src=0, enc: Encoded = None) -> Heads:
    """mmask = move_masks(state); src = the move source the dst rows condition on."""
    a_logits, q, ptr, value, enc = policy(params, obs, src, enc)
    m = mmask.at[1].set(dst_mask(mmask[1], src)).at[3].set(dst_mask(mmask[3], src))
    x = enc.ct
    qi = jnp.concatenate([enc.cst, enc.pit]) @ params["Wqi"]
    return Heads(jnp.where(legal, a_logits, MASKED), q, qi, x @ params["Wu"] + params["bu"], x,
                 x @ params["Wpk"], x @ params["Wpv"], jnp.where(m, ptr, MASKED), value)


# --- the card picker --------------------------------------------------------------
# picks: (MAX_PICK,) int32, lanes in pick order, then STOP for every step after
# the stop. sel = the picked lanes as a (MAX_HAND_SIZE,) mask (what env_step reads).
def pick_query(hd: Heads, a):
    """The picker query of action a: its verb's, plus the consumable / pack card it uses."""
    use = (a >= A_USE) & (a < A_SELL_CONS)
    pick = (a >= A_PICK) & (a < A_SKIP)
    i = jnp.where(use, a - A_USE, K_CONS + a - A_PICK)
    return hd.q[pick_verb(a)] + jnp.where(use | pick, hd.qi[jnp.clip(i, 0, K_CONS + MAX_PACK - 1)], 0.0)


def _pick_init():
    return (jnp.zeros(MAX_HAND_SIZE, bool), jnp.zeros(TOK_D), jnp.int32(0), jnp.bool_(False))


def pick_logits(params, hd: Heads, qv, lo, hi, hand_valid, carry):
    """(MAX_HAND_SIZE + 1,) masked logits of the next pick; the last one is STOP.
    qv = pick_query(hd, a); [lo, hi] = the card count (pick_bounds). STOP needs lo
    cards (or no card left to pick); no lane after hi. Each lane also attends over
    the lanes picked so far (zero before the first pick)."""
    picked, ctx, n, _ = carry
    g = qv + ctx @ params["Wc"] + jnp.asarray(params["Wn"])[n]
    s = jnp.where(picked[None, :], hd.u @ hd.k.T / jnp.sqrt(PICK_D), MASKED)
    r = (jax.nn.softmax(s, axis=-1) @ hd.w) * picked.any()
    lane = jnp.tanh(g + hd.u + r) @ params["vo"]
    stop = jnp.tanh(g) @ params["vs"] + params["bstop"][0]
    lane_ok = hand_valid & ~picked & (n < hi)
    stop_ok = (n >= lo) | ~lane_ok.any()
    return jnp.where(jnp.append(lane_ok, stop_ok), jnp.append(lane, stop), MASKED)


def _pick_update(hd: Heads, carry, c):
    picked, ctx, n, done = carry
    take = ~done & (c < STOP)
    lane = jnp.clip(c, 0, MAX_HAND_SIZE - 1)
    return (picked.at[lane].set(picked[lane] | take), ctx + take * hd.x[lane],
            n + take, done | (c == STOP))


def sample_picks(params, hd: Heads, qv, lo, hi, hand_valid, key, greedy=False):
    """-> (picks (MAX_PICK,), sel (MAX_HAND_SIZE,) bool)."""
    def step(carry, k):
        logits = pick_logits(params, hd, qv, lo, hi, hand_valid, carry)
        c = jnp.argmax(logits) if greedy else jax.random.categorical(k, logits)
        c = jnp.where(carry[3], STOP, c).astype(jnp.int32)
        return _pick_update(hd, carry, c), c
    carry, picks = jax.lax.scan(step, _pick_init(), jax.random.split(key, MAX_PICK))
    return picks, carry[0]


def pick_logp_entropy(params, hd: Heads, qv, lo, hi, hand_valid, picks):
    """log pi(picks | action) and the summed per-step entropy along the taken picks."""
    @jax.checkpoint     # recompute in the backward pass: keeps 5 x (batch, 32, PICK_D) off the GPU
    def step(carry, c):
        lsm = jax.nn.log_softmax(pick_logits(params, hd, qv, lo, hi, hand_valid, carry))
        live = ~carry[3]
        lp = jnp.where(live, lsm[c], 0.0)
        ent = jnp.where(live, -(jnp.exp(lsm) * lsm).sum(), 0.0)
        return _pick_update(hd, carry, c), (lp, ent)
    _, (lp, ent) = jax.lax.scan(step, _pick_init(), picks)
    return lp.sum(), ent.sum()


def _cat_entropy(logits):
    lp = jax.nn.log_softmax(logits)
    return -(jnp.exp(lp) * lp).sum()


def log_prob_entropy(params, hd: Heads, a, picks, src, dst, hand_valid, lo, hi):
    """log pi(a, picks, src, dst) and the entropy. The picks count only for the
    picker actions; the pointer pair only for its own MOVE verb. The dst entropy is
    the one given the taken src; the pick entropy the one along the taken picks."""
    logp_a = jax.nn.log_softmax(hd.a)
    lp_a = logp_a[a]
    uses_sel = uses_picker(a)
    lp_sel, ent_s = pick_logp_entropy(params, hd, pick_query(hd, a), lo, hi, hand_valid, picks)
    lp_ptr = jax.nn.log_softmax(hd.ptr, axis=-1)
    lp_j = lp_ptr[0, src] + lp_ptr[1, dst]
    lp_c = lp_ptr[2, src] + lp_ptr[3, dst]
    lp = (lp_a + uses_sel * lp_sel + (a == A_MOVE_JOKER) * lp_j + (a == A_MOVE_CARD) * lp_c)

    p_a = jnp.exp(logp_a)
    ent_a = -(p_a * logp_a).sum()
    ent_ptr = jax.vmap(_cat_entropy)(hd.ptr)
    ent = (ent_a + uses_sel * ent_s
           + p_a[A_MOVE_JOKER] * (ent_ptr[0] + ent_ptr[1])
           + p_a[A_MOVE_CARD] * (ent_ptr[2] + ent_ptr[3]))
    return lp, ent


def decode(a, sel, src=0, dst=0) -> balatroAction:
    mv = is_move(a)
    return balatroAction(verb=ACTION_VERB[a], select=sel.astype(jnp.int32),
                         target=jnp.where(mv, src, ACTION_TARGET[a]).astype(jnp.int32),
                         dest=jnp.where(mv, dst, 0).astype(jnp.int32))


def sample_src(hd: Heads, a, key):
    """The move source for verb a (the joker row for MOVE_JOKER, else the card row)."""
    kj, kc = jax.random.split(key)
    sj = jax.random.categorical(kj, hd.ptr[0])
    sc = jax.random.categorical(kc, hd.ptr[2])
    return jnp.where(a == A_MOVE_JOKER, sj, sc).astype(jnp.int32)


def sample_dst(hd: Heads, a, key):
    kj, kc = jax.random.split(key)
    dj = jax.random.categorical(kj, hd.ptr[1])
    dc = jax.random.categorical(kc, hd.ptr[3])
    return jnp.where(a == A_MOVE_JOKER, dj, dc).astype(jnp.int32)


class Step(NamedTuple):
    """What act() decided and what the learner needs to replay it."""
    a     : jax.Array
    sel   : jax.Array
    picks : jax.Array
    src   : jax.Array
    dst   : jax.Array
    obs   : jax.Array
    legal : jax.Array
    mmask : jax.Array
    value : jax.Array
    logp  : jax.Array
    lo    : jax.Array
    hi    : jax.Array


def act(params, state: balatroEnvState, key):
    """Sample one action: verb, then the card picks (for its verb), then src,
    then dst given src. -> (balatroAction, Step)"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = observe(state)
    legal = legal_mask(state)
    mmask = move_masks(state)
    enc = encode(params, obs)      # once: the src pass below reuses it
    hd = masked_logits(params, obs, legal, state.hand_valid, mmask, enc=enc)
    a = jax.random.categorical(k1, hd.a).astype(jnp.int32)
    lo, hi = pick_bounds(state, a)
    picks, sel = sample_picks(params, hd, pick_query(hd, a), lo, hi, state.hand_valid, k2)
    src = sample_src(hd, a, k3)
    hd = masked_logits(params, obs, legal, state.hand_valid, mmask, src, enc)
    dst = sample_dst(hd, a, k4)
    logp, _ = log_prob_entropy(params, hd, a, picks, src, dst, state.hand_valid, lo, hi)
    return decode(a, sel, src, dst), Step(a, sel, picks, src, dst, obs, legal, mmask, hd.value,
                                          logp, lo, hi)


OBS_DIM = _obs_dim()


if __name__ == "__main__":
    from fixtures import state as make_state
    s = make_state(hand="4C KD 9H KC 7C 9S 7S 3H", chips_scored=60, money=7,
                   jokers=("Green Joker",), joker_value=(2.0,), consumables=("The Lovers", "Mercury"))
    print("OBS_DIM", OBS_DIM, " N_ACTIONS", N_ACTIONS, " N_TOK", N_TOK)
    print("legal (blind):", [action_name(i) for i in range(N_ACTIONS) if bool(legal_mask(s)[i])])
    shop = s._replace(game_stage=jnp.int32(STAGE_SHOP),
                      shop_slots=jnp.array([0, 3, -1, -1], jnp.int32))
    print("legal (shop): ", [action_name(i) for i in range(N_ACTIONS) if bool(legal_mask(shop)[i])])
    params = init_params(jax.random.key(0))
    print("params", sum(x.size for x in jax.tree.leaves(params)))
    a, st = act(params, s, jax.random.key(1))
    print("sampled", action_name(st.a, st.src, st.dst), "select", st.sel[:8].astype(int),
          "value", float(st.value), "logp", float(st.logp), "lo/hi", int(st.lo), int(st.hi))
