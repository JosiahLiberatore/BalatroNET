"""The policy: obs encoding, the token encoder, the masked action heads, and the value head.

ACTION LAYOUT — one flat categorical over every atomic choice (NOTES: "KEEP the
flat masked action vector ... one head, one policy, phase-masked"):

    index   0                PLAY        (+ the 32 select lanes)
            1                DISCARD     (+ the 32 select lanes)
            2 .. 5           BUY shop slot 0..3
            6 .. 37          SELL joker slot 0..31
            38               MOVE_JOKER  (+ the joker src / dst pointer heads)
            39               MOVE_CARD   (+ the card src / dst pointer heads)
            40               REROLL
            41               NEXT

plus the card picker, read only by PLAY / DISCARD: given the verb, it picks
lanes one at a time (up to MAX_PICK = 5), each pick conditioned on the cards
picked so far, or STOP after at least one card. The env's "first 5 selected
lanes" repair rule never fires, so the lane order carries no selection
information. Then two pointer pairs (src over 32 slots, then dst over 32 slots given src), read
only by the MOVE verbs (their log-probs count only on those steps). A MOVE
INSERTS the item at src at position dst. legal_mask() and move_masks() mask
every choice the env would turn into a no-op, so the policy never spends a
step on one; dst also masks src itself.

OBS LAYOUT — a union of DISJOINT blocks, in this order (offsets O_*):
    JOKER IDS   id + 1 per held slot (32), then per shop slot (4, 0 outside the shop); 0 = empty.
    JOKER SLOT  per held slot: live value, sell value, To Do hand, Loyalty phase  (JS_F each)
    CARDS       per hand lane: rank + suit one-hot (CARD_F each)                  (0 in the shop)
    SHOP SLOT   per shop slot: cost, affordable                                   (0 in a blind)
    GLOBAL      money, ante, blind, stage, scale, round targets, hand counts, budget;
                hands/discards left, chips vs target, undealt mask (0 in the shop);
                reroll cost (0 in a blind)

ENCODER — every held joker, shop offer and hand lane is one token, plus one
GLOBAL token (N_TOK = 1 + J_TOK + 4 + C_TOK: only the slots the env can fill). A joker token = E[id] + fixed row
features @ Wf (the same table for every slot and the shop) + its slot state,
a card token = rank / suit, and every token adds a learned POSITION vector for
its slot or lane and a TYPE vector. All slots share the weights, and the
position vectors keep the order (Blueprint, left-to-right scoring). N_LAYERS
pre-LN self-attention layers mix the tokens; an empty slot or a hole is never
a key. The trunk reads the GLOBAL features and the GLOBAL token's output.
BUY / SELL add a score per offer / held joker token, the MOVE pointers score
the joker / card tokens, and the picker scores the card tokens.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from env import (balatroAction, balatroEnvState, build_ctx, MAX_HAND_SIZE, MAX_DECK_SIZE,
                 MAX_SHOP_SLOTS, STAGE_BLIND, STAGE_SHOP, V_PLAY, V_DISCARD, V_BUY, V_SELL,
                 V_REROLL, V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, WIN_ANTE, live_v, bankrupt_at,
                 reroll_price)
import numpy as np

from jokers import (MAX_JOKERS, N_JOKERS, J_EMPTY, JOKER_COST, JOKERS, JOKER_RARITY, ST_TODO,
                    ST_LOYALTY, N_HAND_TYPES, CONDITION_NAMES, C_SUIT, C_RANK_IN, C_HAND_HAS,
                    C_COUNT_CMP, C_FACES_GE, C_LOYALTY, C_POSITION, C_COUNT_EQ)

# --- action layout ------------------------------------------------------------
A_PLAY, A_DISCARD = 0, 1
A_BUY = 2                                   # + shop slot
A_SELL = A_BUY + MAX_SHOP_SLOTS             # + joker slot
A_MOVE_JOKER = A_SELL + MAX_JOKERS          # src / dst from the joker pointer heads
A_MOVE_CARD = A_MOVE_JOKER + 1              # src / dst from the card pointer heads
A_REROLL = A_MOVE_CARD + 1
A_NEXT = A_REROLL + 1
N_ACTIONS = A_NEXT + 1                      # 42
N_PTR = MAX_JOKERS                          # pointer width; MAX_JOKERS == MAX_HAND_SIZE == 32
assert MAX_HAND_SIZE == N_PTR
MAX_PICK = 5                                # cards per PLAY / DISCARD (the env's limit)
STOP = MAX_HAND_SIZE                        # the picker's "no more cards" choice
CARD_F = 13 + 4                             # rank + suit one-hot per lane
PICK_D = 64                                 # the picker's hidden width

_idx = jnp.arange(N_ACTIONS)
ACTION_VERB = jnp.select(
    [_idx == A_PLAY, _idx == A_DISCARD, _idx < A_SELL, _idx < A_MOVE_JOKER, _idx == A_MOVE_JOKER,
     _idx == A_MOVE_CARD, _idx == A_REROLL],
    [V_PLAY, V_DISCARD, V_BUY, V_SELL, V_MOVE_JOKER, V_MOVE_CARD, V_REROLL], V_NEXT).astype(jnp.int32)
ACTION_TARGET = jnp.select(
    [(_idx >= A_BUY) & (_idx < A_SELL), (_idx >= A_SELL) & (_idx < A_MOVE_JOKER)],
    [_idx - A_BUY, _idx - A_SELL], 0).astype(jnp.int32)


def is_move(a):
    return (a == A_MOVE_JOKER) | (a == A_MOVE_CARD)


def action_name(a, src=None, dst=None):
    a = int(a)
    if a == A_PLAY:    return "PLAY"
    if a == A_DISCARD: return "DISCARD"
    if a < A_SELL:     return f"BUY {a - A_BUY}"
    if a < A_MOVE_JOKER: return f"SELL {a - A_SELL}"
    if a in (A_MOVE_JOKER, A_MOVE_CARD):
        k = "J" if a == A_MOVE_JOKER else "C"
        return f"MOVE {k}" if src is None else f"MOVE {k}{int(src)}->{k}{int(dst)}"
    return "REROLL" if a == A_REROLL else "NEXT"


def legal_mask(state: balatroEnvState) -> jax.Array:
    """(N_ACTIONS,) bool — the choices env_step would actually carry out.

    Mirrors env_step's repair rules; an unmasked choice the env turns into a
    no-op is not a bug, only a wasted step. When no choice is legal (LOST/WON,
    the episode is over and gets reset), NEXT stays open so the categorical is
    never fully masked.
    """
    blind = state.game_stage == STAGE_BLIND
    shop = state.game_stage == STAGE_SHOP
    held = (state.jokers != J_EMPTY)
    room = held.sum() < state.joker_slots

    slots = jnp.arange(MAX_SHOP_SLOTS)
    offer = state.shop_slots
    cost = JOKER_COST[jnp.clip(offer, 0, N_JOKERS - 1)]
    floor = bankrupt_at(state)                    # 0, or below with Credit Card
    buy = shop & (slots < state.shop_size) & (offer >= 0) & (state.money - cost >= floor) & room
    sell = (blind | shop) & held
    move = (blind | shop) & (held.sum() >= 2)
    move_card = blind & (state.hand_size >= 2) & state.hand_valid.any()

    legal = jnp.concatenate([
        jnp.stack([blind, blind & (state.discards > 0)]),
        buy,
        sell,
        jnp.stack([move, move_card]),
        jnp.stack([shop & ((state.free_rerolls > 0) | (state.money - reroll_price(state) >= floor)), shop]),
    ])
    return legal.at[A_NEXT].set(legal[A_NEXT] | ~legal.any())


def move_masks(state: balatroEnvState) -> jax.Array:
    """(4, N_PTR) bool — rows: joker src, joker dst, card src, card dst.

    The dst rows are the slots a move may END at; dst_mask() also removes src.
    Joker: any held slot (held jokers are packed). Card: src must hold a card,
    dst is any lane < hand_size (holes shift like cards).
    """
    held = state.jokers != J_EMPTY
    lanes = jnp.arange(N_PTR) < state.hand_size
    return jnp.stack([held, held, state.hand_valid & lanes, lanes])


def dst_mask(base, src):
    return base & (jnp.arange(N_PTR) != src)


# --- observation ----------------------------------------------------------------
J_SLOTS = MAX_JOKERS + MAX_SHOP_SLOTS       # 36 id fields at the front of obs
JS_F = 2 + N_HAND_TYPES + 1                 # per held slot: value, sell, To Do hand, Loyalty
SHOP_F = 2                                  # per shop slot: cost, affordable
O_JS = J_SLOTS
O_CARD = O_JS + MAX_JOKERS * JS_F
O_SHOP = O_CARD + MAX_HAND_SIZE * CARD_F
O_GLOB = O_SHOP + MAX_SHOP_SLOTS * SHOP_F


def _joker_slot_block(state: balatroEnvState) -> jax.Array:
    """(MAX_JOKERS, JS_F) the per-slot joker state; zero for an empty slot."""
    held = state.jokers != J_EMPTY
    kind = JOKERS.STATE[jnp.clip(state.jokers, 0, N_JOKERS - 1)]
    todo = jax.nn.one_hot(state.joker_state, N_HAND_TYPES) * (held & (kind == ST_TODO))[:, None]
    loyal = jnp.where(held & (kind == ST_LOYALTY),
                      jnp.mod(state.hand_played.sum() - state.joker_state, 6) / 5.0, 0.0)
    return jnp.concatenate([(live_v(state) / 10.0)[:, None],
                            (state.joker_sell_value / 10.0)[:, None],
                            todo, loyal[:, None]], axis=1) * held[:, None]


def _shared_block(state: balatroEnvState) -> jax.Array:
    f32 = jnp.float32
    blind = (state.game_stage == STAGE_BLIND).astype(f32)
    shop = (state.game_stage == STAGE_SHOP).astype(f32)
    held = state.jokers != J_EMPTY
    shared = jnp.concatenate([
        jnp.stack([state.money / 25.0, jnp.log1p(jnp.maximum(state.money, 0)) / 4.0]),
        jax.nn.one_hot(jnp.clip(state.ante, 1, WIN_ANTE) - 1, WIN_ANTE),
        jax.nn.one_hot(state.blind_index, 3),
        jnp.stack([blind, shop, state.target_scale]),
        jnp.stack([held.sum() / state.joker_slots, state.joker_slots / 5.0]),
        # the round targets (The Idol, Mail-In Rebate, Ancient Joker, Castle)
        jax.nn.one_hot(state.idol_card % 13, 13), jax.nn.one_hot(state.idol_card // 13, 4),
        jax.nn.one_hot(state.mail_rank, 13), jax.nn.one_hot(state.ancient_suit, 4),
        jax.nn.one_hot(state.castle_suit, 4),
        state.hand_played / 10.0, state.hand_round / 4.0,
        jnp.stack([state.hand_size / 10.0, state.handcount / 5.0, state.discardcount / 5.0,
                   state.discards_used / 3.0, state.free_rerolls.astype(f32),
                   bankrupt_at(state) / 20.0, state.gros_extinct.astype(f32)]),
    ]).astype(f32)
    return shared


def observe(state: balatroEnvState) -> jax.Array:
    f32 = jnp.float32
    blind = (state.game_stage == STAGE_BLIND).astype(f32)
    shop = (state.game_stage == STAGE_SHOP).astype(f32)

    # JOKER IDS (id + 1, 0 = empty; the shop ids only in the shop)
    held_id = jnp.where(state.jokers != J_EMPTY, state.jokers + 1, 0)
    offer_id = jnp.where((state.shop_slots >= 0) & (state.game_stage == STAGE_SHOP),
                         state.shop_slots + 1, 0)
    ids = jnp.concatenate([held_id, offer_id]).astype(f32)

    # CARDS (rank + suit per lane, in a blind)
    valid = state.hand_valid[:, None]
    cards = jnp.concatenate([jax.nn.one_hot(state.hand % 13, 13) * valid,
                             jax.nn.one_hot(state.hand // 13, 4) * valid], axis=1) * blind

    # SHOP SLOT (in the shop)
    offer = state.shop_slots
    on = offer >= 0
    cost = JOKER_COST[jnp.clip(offer, 0, N_JOKERS - 1)] * on
    shop_slot = jnp.stack([cost / 10.0, (on & (state.money >= cost)).astype(f32)], axis=1) * shop

    # GLOBAL
    pos = jnp.arange(MAX_DECK_SIZE)
    undealt = (pos >= state.deck_ptr) & (pos < state.deck_size)
    undealt_mask = jnp.zeros(52).at[jnp.clip(state.deck, 0, 51)].max(
        (undealt & (state.deck < 52)).astype(f32))
    blind_g = jnp.concatenate([
        jnp.stack([state.hands / state.handcount,
                   state.discards / state.discardcount,
                   jnp.clip(state.chips_scored / state.blind_target, 0.0, 2.0),
                   jnp.log10(state.blind_target.astype(f32)) / 6.0]),
        undealt_mask]) * blind
    shop_g = jnp.stack([reroll_price(state) / 10.0,
                        (state.money >= reroll_price(state)).astype(f32)]) * shop

    return jnp.concatenate([ids, _joker_slot_block(state).ravel(), cards.ravel(),
                            shop_slot.ravel(), _shared_block(state), blind_g, shop_g]).astype(f32)


def _obs_dim():
    from fixtures import state as make_state
    return int(observe(make_state(hand="2C")).shape[0])


def joker_ids(obs):
    """(J_SLOTS,) int32 joker id per held / shop slot, -1 = empty (read back from obs)."""
    return obs[:J_SLOTS].astype(jnp.int32) - 1


def hand_feats(obs) -> jax.Array:
    """(MAX_HAND_SIZE, CARD_F) rank + suit one-hot per lane, read back from obs
    (all zero for a hole and in the shop)."""
    return obs[O_CARD:O_SHOP].reshape(MAX_HAND_SIZE, CARD_F)


# --- the fixed joker features ("the card text") ------------------------------------
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
    magnitudes, update rules and their conditions), so jokers of one kind start
    near each other and the agent can read a joker it has rarely held."""
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
                      t["EXPIRE_AT"] > -1e8], axis=1).astype(np.float32),
            upd.astype(np.float32), (t["UPD_DELTA"] * upd / 5.0).astype(np.float32),
            oh(t["UPD_MODE"]).reshape(N_JOKERS, -1) * np.repeat(upd, 3, axis=1),
            upd_cond, upd_bits]
    return jnp.asarray(np.concatenate(cols, axis=1))


JOKER_FEATS = _joker_feats()
JOKER_F = int(JOKER_FEATS.shape[1])


# --- the token encoder --------------------------------------------------------------
TOK_D = 64                                  # token width
N_HEADS = 4
N_LAYERS = 2
FF_D = 128                                  # the feed-forward width in each layer
# Tokens only for the slots the env can fill (held jokers are packed, lanes <
# hand_size); the obs and the action layout keep all 32. tests_run checks the caps.
J_TOK = 8                                   # held joker tokens (the env has 5 joker slots)
C_TOK = 16                                  # hand lane tokens (8 + the largest hand size bonus)
N_TOK = 1 + J_TOK + MAX_SHOP_SLOTS + C_TOK  # GLOBAL, held, shop, lanes
T_HELD, T_SHOP, T_CARD, T_GLOB = 0, 1, 2, 3   # rows of the TYPE table


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * g + b


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


def tokens(params, obs):
    """-> (tok (N_TOK, TOK_D), key_ok (N_TOK,)) before the attention layers.
    Order: GLOBAL, held slots 0..J_TOK-1, shop slots 0..3, hand lanes 0..C_TOK-1."""
    ids = joker_ids(obs)
    j = jnp.clip(ids, 0, N_JOKERS - 1)
    E, Ty = jnp.asarray(params["E"]), jnp.asarray(params["Ty"])
    jemb = (E[j] + JOKER_FEATS[j] @ params["Wf"]) * (ids >= 0)[:, None]
    js = obs[O_JS:O_CARD].reshape(MAX_JOKERS, JS_F)[:J_TOK]
    sf = obs[O_SHOP:O_GLOB].reshape(MAX_SHOP_SLOTS, SHOP_F)
    x = hand_feats(obs)[:C_TOK]
    held = jemb[:J_TOK] + js @ params["Wjs"] + params["Pj"] + Ty[T_HELD]
    shop = jemb[MAX_JOKERS:] + sf @ params["Wsh"] + params["Ps"] + Ty[T_SHOP]
    cards = x @ params["Wcd"] + params["Pc"] + Ty[T_CARD]
    glob = obs[O_GLOB:] @ params["Wg"] + params["bg"] + Ty[T_GLOB]
    tok = jnp.concatenate([glob[None], held, shop, cards])
    key_ok = jnp.concatenate([jnp.ones(1, bool), ids[:J_TOK] >= 0, ids[MAX_JOKERS:] >= 0,
                              x.sum(1) > 0])
    return tok, key_ok


class Encoded(NamedTuple):
    h     : jax.Array   # (HIDDEN,)  the trunk
    jt    : jax.Array   # (MAX_JOKERS, TOK_D)  held joker tokens (zero past J_TOK)
    st    : jax.Array   # (MAX_SHOP_SLOTS, TOK_D)  shop offer tokens
    ct    : jax.Array   # (MAX_HAND_SIZE, TOK_D)  hand lane tokens (zero past C_TOK)


def encode(params, obs) -> Encoded:
    tok, key_ok = tokens(params, obs)
    for p in params["layers"]:
        tok = _layer(p, tok, key_ok)
    tok = _ln(tok, params["gN"], params["bN"])
    h = jnp.tanh(obs[O_GLOB:] @ params["W1"] + tok[0] @ params["W1t"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    a, b = 1 + J_TOK, 1 + J_TOK + MAX_SHOP_SLOTS
    pad = lambda t, n: jnp.pad(t, ((0, n - t.shape[0]), (0, 0)))
    return Encoded(h, pad(tok[1:a], MAX_JOKERS), tok[a:b], pad(tok[b:], MAX_HAND_SIZE))


# --- network --------------------------------------------------------------------
HIDDEN = 256


def init_params(key, obs_dim=None):
    obs_dim = obs_dim or OBS_DIM
    g_dim = obs_dim - O_GLOB
    ks = iter(jax.random.split(key, 64))
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
        "E": table(N_JOKERS, D),                         # the shared joker embedding
        "Wf": dense(JOKER_F, D, 0.5)[0],                 # fixed row features -> token
        "Wjs": dense(JS_F, D, 1.0)[0], "Wsh": dense(SHOP_F, D, 1.0)[0],
        "Wcd": dense(CARD_F, D, 1.0)[0],
        "Wg": dense(g_dim, D, 1.0)[0], "bg": jnp.zeros(D),
        "Pj": table(J_TOK, D), "Ps": table(MAX_SHOP_SLOTS, D),   # position per slot / lane
        "Pc": table(C_TOK, D), "Ty": table(4, D),
        "layers": [layer() for _ in range(N_LAYERS)],
        "gN": jnp.ones(D), "bN": jnp.zeros(D),
        # trunk
        "W1": dense(g_dim, HIDDEN, 1.0)[0], "W1t": dense(D, HIDDEN, 1.0)[0], "b1": jnp.zeros(HIDDEN),
    }
    p["W2"], p["b2"] = dense(HIDDEN, HIDDEN, 1.0)
    p["Wa"], p["ba"] = dense(HIDDEN, N_ACTIONS, 0.01)          # small: start near-uniform
    p["Wbs"] = dense(HIDDEN, 2 * D, 0.01)[0]                   # BUY / SELL query over the tokens
    p["Wp"], p["bp"] = dense(HIDDEN, 4 * D, 0.01)              # joker src, joker dst, card src, card dst
    p["Rp"] = table(2, D, D, scale=0.01 / D ** 0.5)            # dst query given the src token
    p["Pm"] = table(2, N_PTR, N_PTR, scale=0.01)               # dst bias given src: [kind, src, dst]
    # the card picker
    p["Wq"], p["bq"] = dense(HIDDEN, 2 * PICK_D, 1.0)          # picker query, one per verb (PLAY, DISCARD)
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
    q / u / x / k / w feed the card picker (pick_logits)."""
    a     : jax.Array   # (N_ACTIONS,)
    q     : jax.Array   # (2, PICK_D)  picker query: PLAY, DISCARD
    u     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane query
    x     : jax.Array   # (MAX_HAND_SIZE, TOK_D)   per-lane token
    k     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane key   (for the lanes picked so far)
    w     : jax.Array   # (MAX_HAND_SIZE, PICK_D)  per-lane value (for the lanes picked so far)
    ptr   : jax.Array   # (4, N_PTR)
    value : jax.Array   # ()


def policy(params, obs, src, enc: Encoded = None):
    """obs -> raw (action (42,), picker query (2, PICK_D), pointer (4, 32), value (), Encoded).
    The two dst rows are conditioned on src (the joker src for row 1, the card
    src for row 3; one src serves both, only the verb's own pair is read).
    enc = encode(params, obs) when the caller already has it."""
    enc = encode(params, obs) if enc is None else enc
    h, jt, st, ct = enc
    D = TOK_D
    qb = (h @ params["Wbs"]).reshape(2, D) / jnp.sqrt(D)
    a = h @ params["Wa"] + params["ba"]
    a = a.at[A_BUY:A_SELL].add(st @ qb[0]).at[A_SELL:A_MOVE_JOKER].add(jt @ qb[1])
    qp = (h @ params["Wp"] + params["bp"]).reshape(4, D)
    s = jnp.clip(src, 0, N_PTR - 1)
    Pm, Rp = jnp.asarray(params["Pm"]), jnp.asarray(params["Rp"])   # loaded params are numpy
    ptr = jnp.stack([jt @ qp[0],
                     jt @ (qp[1] + jt[s] @ Rp[0]) + Pm[0, s],
                     ct @ qp[2],
                     ct @ (qp[3] + ct[s] @ Rp[1]) + Pm[1, s]])
    return (a, (h @ params["Wq"] + params["bq"]).reshape(2, PICK_D), ptr,
            (h @ params["Wv"] + params["bv"])[0], enc)


# a masked choice gets this logit, not -inf: its probability is still exactly 0,
# but p * log p stays 0 * finite in the backward pass (0 * -inf would be NaN)
MASKED = -1e9


def masked_logits(params, obs, legal, hand_valid, mmask, src=0, enc: Encoded = None) -> Heads:
    """mmask = move_masks(state); src = the move source the dst rows condition on."""
    a_logits, q, ptr, value, enc = policy(params, obs, src, enc)
    m = mmask.at[1].set(dst_mask(mmask[1], src)).at[3].set(dst_mask(mmask[3], src))
    x = enc.ct
    return Heads(jnp.where(legal, a_logits, MASKED), q, x @ params["Wu"] + params["bu"], x,
                 x @ params["Wpk"], x @ params["Wpv"], jnp.where(m, ptr, MASKED), value)


# --- the card picker --------------------------------------------------------------
# picks: (MAX_PICK,) int32, lanes in pick order, then STOP for every step after
# the stop. sel = the picked lanes as a (MAX_HAND_SIZE,) mask (what env_step reads).
def _pick_init():
    return (jnp.zeros(MAX_HAND_SIZE, bool), jnp.zeros(TOK_D), jnp.int32(0), jnp.bool_(False))


def pick_logits(params, hd: Heads, v, hand_valid, carry):
    """(MAX_HAND_SIZE + 1,) masked logits of the next pick; the last one is STOP.
    v = 0 for PLAY, 1 for DISCARD. STOP needs one card picked (or none left).
    Each lane also attends over the lanes picked so far (zero before the first pick)."""
    picked, ctx, n, _ = carry
    g = hd.q[v] + ctx @ params["Wc"] + jnp.asarray(params["Wn"])[n]
    s = jnp.where(picked[None, :], hd.u @ hd.k.T / jnp.sqrt(PICK_D), MASKED)
    r = (jax.nn.softmax(s, axis=-1) @ hd.w) * picked.any()
    lane = jnp.tanh(g + hd.u + r) @ params["vo"]
    stop = jnp.tanh(g) @ params["vs"] + params["bstop"][0]
    lane_ok = hand_valid & ~picked
    stop_ok = (n >= 1) | ~lane_ok.any()
    return jnp.where(jnp.append(lane_ok, stop_ok), jnp.append(lane, stop), MASKED)


def _pick_update(hd: Heads, carry, c):
    picked, ctx, n, done = carry
    take = ~done & (c < STOP)
    lane = jnp.clip(c, 0, MAX_HAND_SIZE - 1)
    return (picked.at[lane].set(picked[lane] | take), ctx + take * hd.x[lane],
            n + take, done | (c == STOP))


def sample_picks(params, hd: Heads, v, hand_valid, key, greedy=False):
    """-> (picks (MAX_PICK,), sel (MAX_HAND_SIZE,) bool)."""
    def step(carry, k):
        logits = pick_logits(params, hd, v, hand_valid, carry)
        c = jnp.argmax(logits) if greedy else jax.random.categorical(k, logits)
        c = jnp.where(carry[3], STOP, c).astype(jnp.int32)
        return _pick_update(hd, carry, c), c
    carry, picks = jax.lax.scan(step, _pick_init(), jax.random.split(key, MAX_PICK))
    return picks, carry[0]


def pick_logp_entropy(params, hd: Heads, v, hand_valid, picks):
    """log pi(picks | verb) and the summed per-step entropy along the taken picks."""
    @jax.checkpoint     # recompute in the backward pass: keeps 5 x (batch, 32, PICK_D) off the GPU
    def step(carry, c):
        lsm = jax.nn.log_softmax(pick_logits(params, hd, v, hand_valid, carry))
        live = ~carry[3]
        lp = jnp.where(live, lsm[c], 0.0)
        ent = jnp.where(live, -(jnp.exp(lsm) * lsm).sum(), 0.0)
        return _pick_update(hd, carry, c), (lp, ent)
    _, (lp, ent) = jax.lax.scan(step, _pick_init(), picks)
    return lp.sum(), ent.sum()


def _cat_entropy(logits):
    lp = jax.nn.log_softmax(logits)
    return -(jnp.exp(lp) * lp).sum()


def log_prob_entropy(params, hd: Heads, a, picks, src, dst, hand_valid):
    """log pi(a, picks, src, dst) and the entropy. The picks count only for
    PLAY/DISCARD; the pointer pair only for its own MOVE verb. The dst entropy is
    the one given the taken src; the pick entropy the one along the taken picks."""
    logp_a = jax.nn.log_softmax(hd.a)
    lp_a = logp_a[a]
    uses_sel = (a == A_PLAY) | (a == A_DISCARD)
    lp_sel, ent_s = pick_logp_entropy(params, hd, (a == A_DISCARD).astype(jnp.int32),
                                      hand_valid, picks)
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


def act(params, state: balatroEnvState, key):
    """Sample one action: verb, then the card picks (for its verb), then src,
    then dst given src.
    -> (balatroAction, (a, sel, picks, src, dst, obs, legal, mmask, value, logp))"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = observe(state)
    legal = legal_mask(state)
    mmask = move_masks(state)
    enc = encode(params, obs)      # once: the src pass below reuses it
    hd = masked_logits(params, obs, legal, state.hand_valid, mmask, enc=enc)
    a = jax.random.categorical(k1, hd.a).astype(jnp.int32)
    picks, sel = sample_picks(params, hd, (a == A_DISCARD).astype(jnp.int32), state.hand_valid, k2)
    src = sample_src(hd, a, k3)
    hd = masked_logits(params, obs, legal, state.hand_valid, mmask, src, enc)
    dst = sample_dst(hd, a, k4)
    logp, _ = log_prob_entropy(params, hd, a, picks, src, dst, state.hand_valid)
    return decode(a, sel, src, dst), (a, sel, picks, src, dst, obs, legal, mmask, hd.value, logp)


OBS_DIM = _obs_dim()


if __name__ == "__main__":
    from fixtures import state as make_state
    s = make_state(hand="4C KD 9H KC 7C 9S 7S 3H", chips_scored=60, money=7,
                   jokers=("Green Joker",), joker_value=(2.0,))
    print("OBS_DIM", OBS_DIM, " N_ACTIONS", N_ACTIONS)
    print("legal (blind):", [action_name(i) for i in range(N_ACTIONS) if bool(legal_mask(s)[i])])
    shop = s._replace(game_stage=jnp.int32(STAGE_SHOP),
                      shop_slots=jnp.array([0, 3, -1, -1], jnp.int32))
    print("legal (shop): ", [action_name(i) for i in range(N_ACTIONS) if bool(legal_mask(shop)[i])])
    params = init_params(jax.random.key(0))
    a, (ai, sel, picks, src, dst, obs, legal, mm, v, lp) = act(params, s, jax.random.key(1))
    print("sampled", action_name(ai, src, dst), "select", sel[:8].astype(int),
          "value", float(v), "logp", float(lp))
