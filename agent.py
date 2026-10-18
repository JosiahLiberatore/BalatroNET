"""The policy: obs encoding, the masked action heads, and the value head.

ACTION LAYOUT — one flat categorical over every atomic choice (NOTES: "KEEP the
flat masked action vector ... one head, one policy, phase-masked"):

    index   0                PLAY      (+ the 32 select lanes)
            1                DISCARD   (+ the 32 select lanes)
            2 .. 5           BUY shop slot 0..3
            6 .. 37          SELL joker slot 0..31
            38               REROLL
            39               NEXT

plus 32 independent Bernoulli select lanes, read only by PLAY / DISCARD (their
log-prob counts only on those steps). legal_mask() masks every choice the env
would turn into a no-op, so the policy never spends a step on one.

OBS LAYOUT — a union of DISJOINT blocks (NOTES: "obs layout"):
    SHARED      money, ante, blind, stage, curriculum scale, joker loadout
    BLIND-ONLY  hand ranks/suits, hands/discards left, chips vs target, undealt mask  (0 in the shop)
    SHOP-ONLY   shop offers, costs, affordability, reroll cost                        (0 in a blind)
"""

import jax
import jax.numpy as jnp

from env import (balatroAction, balatroEnvState, build_ctx, MAX_HAND_SIZE, MAX_DECK_SIZE,
                 MAX_SHOP_SLOTS, STAGE_BLIND, STAGE_SHOP, V_PLAY, V_DISCARD, V_BUY, V_SELL,
                 V_REROLL, V_NEXT, V_MOVE_JOKER, V_MOVE_CARD, WIN_ANTE)
from jokers import MAX_JOKERS, N_JOKERS, J_EMPTY, JOKER_COST, rows_for, resolve

# --- action layout ------------------------------------------------------------
A_PLAY, A_DISCARD = 0, 1
A_BUY = 2                                   # + shop slot
A_SELL = A_BUY + MAX_SHOP_SLOTS             # + joker slot
A_MOVE_JOKER = A_SELL + MAX_JOKERS          # + joker slot t: swap t and t+1
A_MOVE_CARD = A_MOVE_JOKER + MAX_JOKERS - 1  # + hand lane t: swap t and t+1
A_REROLL = A_MOVE_CARD + MAX_HAND_SIZE - 1
A_NEXT = A_REROLL + 1
N_ACTIONS = A_NEXT + 1                      # 102

_idx = jnp.arange(N_ACTIONS)
ACTION_VERB = jnp.select(
    [_idx == A_PLAY, _idx == A_DISCARD, _idx < A_SELL, _idx < A_MOVE_JOKER, _idx < A_MOVE_CARD,
     _idx < A_REROLL, _idx == A_REROLL],
    [V_PLAY, V_DISCARD, V_BUY, V_SELL, V_MOVE_JOKER, V_MOVE_CARD, V_REROLL], V_NEXT).astype(jnp.int32)
ACTION_TARGET = jnp.select(
    [(_idx >= A_BUY) & (_idx < A_SELL), (_idx >= A_SELL) & (_idx < A_MOVE_JOKER),
     (_idx >= A_MOVE_JOKER) & (_idx < A_MOVE_CARD), (_idx >= A_MOVE_CARD) & (_idx < A_REROLL)],
    [_idx - A_BUY, _idx - A_SELL, _idx - A_MOVE_JOKER, _idx - A_MOVE_CARD], 0).astype(jnp.int32)


def action_name(a):
    a = int(a)
    if a == A_PLAY:    return "PLAY"
    if a == A_DISCARD: return "DISCARD"
    if a < A_SELL:     return f"BUY {a - A_BUY}"
    if a < A_MOVE_JOKER: return f"SELL {a - A_SELL}"
    if a < A_MOVE_CARD:  return f"MOVE J{a - A_MOVE_JOKER}<>J{a - A_MOVE_JOKER + 1}"
    if a < A_REROLL:     return f"MOVE C{a - A_MOVE_CARD}<>C{a - A_MOVE_CARD + 1}"
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
    buy = shop & (slots < state.shop_size) & (offer >= 0) & (state.money >= cost) & room
    sell = (blind | shop) & held
    move = (blind | shop) & held[:-1] & held[1:]
    lane = jnp.arange(MAX_HAND_SIZE - 1)
    v = state.hand_valid
    move_card = blind & (lane + 1 < state.hand_size) & (v[:-1] | v[1:])

    legal = jnp.concatenate([
        jnp.stack([blind, blind & (state.discards > 0)]),
        buy,
        sell,
        move,
        move_card,
        jnp.stack([shop & (state.money >= state.reroll_cost), shop]),
    ])
    return legal.at[A_NEXT].set(legal[A_NEXT] | ~legal.any())


# --- observation ----------------------------------------------------------------
def observe(state: balatroEnvState) -> jax.Array:
    f32 = jnp.float32
    blind = (state.game_stage == STAGE_BLIND).astype(f32)
    shop = (state.game_stage == STAGE_SHOP).astype(f32)

    # SHARED
    held = state.jokers != J_EMPTY
    live_v = resolve(rows_for(state.jokers), held, state.joker_value, build_ctx(state))
    shared = jnp.concatenate([
        jnp.stack([state.money / 25.0, jnp.log1p(jnp.maximum(state.money, 0)) / 4.0]),
        jax.nn.one_hot(jnp.clip(state.ante, 1, WIN_ANTE) - 1, WIN_ANTE),
        jax.nn.one_hot(state.blind_index, 3),
        jnp.stack([blind, shop, state.target_scale]),
        (jax.nn.one_hot(state.jokers, N_JOKERS) * held[:, None]).ravel(),   # -1 -> all zero
        live_v / 10.0,
        state.joker_sell_value / 10.0,
        jnp.stack([held.sum() / state.joker_slots, state.joker_slots / 5.0]),
    ]).astype(f32)

    # BLIND-ONLY
    valid = state.hand_valid[:, None]
    pos = jnp.arange(MAX_DECK_SIZE)
    undealt = (pos >= state.deck_ptr) & (pos < state.deck_size)
    undealt_mask = jnp.zeros(52).at[jnp.clip(state.deck, 0, 51)].max(
        (undealt & (state.deck < 52)).astype(f32))
    blind_block = jnp.concatenate([
        (jax.nn.one_hot(state.hand % 13, 13) * valid).ravel(),
        (jax.nn.one_hot(state.hand // 13, 4) * valid).ravel(),
        jnp.stack([state.hands / state.handcount,
                   state.discards / state.discardcount,
                   jnp.clip(state.chips_scored / state.blind_target, 0.0, 2.0),
                   jnp.log10(state.blind_target.astype(f32)) / 6.0]),
        undealt_mask,
    ]).astype(f32) * blind

    # SHOP-ONLY
    offer = state.shop_slots
    on = offer >= 0
    cost = JOKER_COST[jnp.clip(offer, 0, N_JOKERS - 1)] * on
    shop_block = jnp.concatenate([
        (jax.nn.one_hot(offer, N_JOKERS) * on[:, None]).ravel(),
        cost / 10.0,
        (on & (state.money >= cost)).astype(f32),
        jnp.stack([state.reroll_cost / 10.0, (state.money >= state.reroll_cost).astype(f32)]),
    ]).astype(f32) * shop

    return jnp.concatenate([shared, blind_block, shop_block])


def _obs_dim():
    from fixtures import state as make_state
    return int(observe(make_state(hand="2C")).shape[0])


# --- network --------------------------------------------------------------------
HIDDEN = 256


def init_params(key, obs_dim=None):
    obs_dim = obs_dim or OBS_DIM
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    def dense(k, n_in, n_out, scale):
        return jax.random.normal(k, (n_in, n_out)) * scale / jnp.sqrt(n_in), jnp.zeros(n_out)
    W1, b1 = dense(k1, obs_dim, HIDDEN, 1.0)
    W2, b2 = dense(k2, HIDDEN, HIDDEN, 1.0)
    Wa, ba = dense(k3, HIDDEN, N_ACTIONS, 0.01)      # small: start near-uniform
    Ws, bs = dense(k4, HIDDEN, MAX_HAND_SIZE, 0.01)
    Wv, bv = dense(k5, HIDDEN, 1, 1.0)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "Wa": Wa, "ba": ba, "Ws": Ws, "bs": bs, "Wv": Wv, "bv": bv}


def policy(params, obs):
    """obs -> (action_logits (40,), select_logits (32,), value ())"""
    h = jnp.tanh(obs @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return (h @ params["Wa"] + params["ba"],
            h @ params["Ws"] + params["bs"],
            (h @ params["Wv"] + params["bv"])[0])


# a masked choice gets this logit, not -inf: its probability is still exactly 0,
# but p * log p stays 0 * finite in the backward pass (0 * -inf would be NaN)
MASKED = -1e9


def masked_logits(params, obs, legal, hand_valid):
    a_logits, s_logits, value = policy(params, obs)
    return jnp.where(legal, a_logits, MASKED), jnp.where(hand_valid, s_logits, MASKED), value


def log_prob_entropy(a_logits, s_logits, a, sel, hand_valid):
    """log pi(a, sel) and the entropy. The select lanes count only for PLAY/DISCARD."""
    logp_a = jax.nn.log_softmax(a_logits)
    lp_a = logp_a[a]
    uses_sel = (a == A_PLAY) | (a == A_DISCARD)
    lp_sel = jnp.where(hand_valid,
                       jnp.where(sel, jax.nn.log_sigmoid(s_logits), jax.nn.log_sigmoid(-s_logits)),
                       0.0).sum()
    p_a = jnp.exp(logp_a)
    ent_a = -(p_a * logp_a).sum()
    p = jnp.clip(jax.nn.sigmoid(s_logits), 1e-7, 1 - 1e-7)
    ent_s = jnp.where(hand_valid, -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p)), 0.0).sum()
    p_sel_verb = p_a[A_PLAY] + p_a[A_DISCARD]
    return lp_a + uses_sel * lp_sel, ent_a + p_sel_verb * ent_s


def decode(a, sel) -> balatroAction:
    return balatroAction(verb=ACTION_VERB[a], select=sel.astype(jnp.int32), target=ACTION_TARGET[a])


def act(params, state: balatroEnvState, key):
    """Sample one action. -> (balatroAction, (a, sel, obs, legal, value, logp))"""
    k1, k2 = jax.random.split(key)
    obs = observe(state)
    legal = legal_mask(state)
    a_logits, s_logits, value = masked_logits(params, obs, legal, state.hand_valid)
    a = jax.random.categorical(k1, a_logits).astype(jnp.int32)
    sel = (jax.random.uniform(k2, (MAX_HAND_SIZE,)) < jax.nn.sigmoid(s_logits)) & state.hand_valid
    logp, _ = log_prob_entropy(a_logits, s_logits, a, sel, state.hand_valid)
    return decode(a, sel), (a, sel, obs, legal, value, logp)


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
    a, (ai, sel, obs, legal, v, lp) = act(params, s, jax.random.key(1))
    print("sampled", action_name(ai), "select", sel[:8].astype(int), "value", float(v), "logp", float(lp))
