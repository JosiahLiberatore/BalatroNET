import jax
import jax.numpy as jnp

from env import balatroAction, balatroEnvState
from env import MAX_DECK_SIZE, MAX_HAND_SIZE
from env import _hand_lanes, _valid_lanes, show, dbg

def init_params(key):

    key1, key2 = jax.random.split(key)

    params = {}

    params[0] = jax.random.normal(key1, (601, 128)) * 0.02
    params[1] = jnp.zeros(128)
    params[2] = jax.random.normal(key2, (128, 33)) * 0.02
    params[3] = jnp.zeros(33)

    return params

#@jit
def observe(state: balatroEnvState):

    hand = state.hand
    hand_valid = state.hand_valid

    ranks = (jax.nn.one_hot(hand % 13, 13) * hand_valid[:, None]).astype(float).ravel()
    suits = (jax.nn.one_hot(hand // 13, 4) * hand_valid[:, None]).astype(float).ravel()

    #scalars
    hands = jnp.array([state.hands / state.handcount]).astype(float)
    discards = jnp.array([state.discards / state.discardcount]).astype(float)
    chips_target = jnp.array([state.chips_scored / state.blind_target]).astype(float)
    ante = jnp.array([state.ante / 39])
    round = jnp.array([state.round / 3])

    deck = state.deck
    pos = jnp.arange(MAX_DECK_SIZE)
    undealt = (pos >= state.deck_ptr) & (pos < state.deck_size)
    undealt_mask = jnp.zeros(52).at[state.deck].set(undealt.astype(float))

    # dbg(n = 30, hand = hand, deck=deck, undealt=undealt, undealt_mask=undealt_mask)

    obs = jnp.concatenate([ranks, suits, hands, discards, chips_target, ante, round, undealt_mask])

    # dbg(n = 30,
    #     deck=state.deck,
    #     hand=hand,
    #     hand_valid=hand_valid,
    #     # ranks=ranks,
    #     suits=suits,
    #     )

    return obs


def policy(params, obs):
    # params -> { "W1": (601, 128), "b1": (128, ), "W2" : (128, 33), "b2", (33,)}

    W1 = params[0]
    b1 = params[1]
    W2 = params[2]
    b2 = params[3]

    h = jnp.tanh(obs @ W1 + b1)
    logits = h @ W2 + b2
    return logits


def act(params, state : balatroEnvState, key):

    key1, key2 = jax.random.split(key)
    obs = observe(state)
    logits = policy(params, obs)

    select = logits[1:MAX_HAND_SIZE+1]
    verb = logits[0]

    select_logits = jnp.where(state.hand_valid, select, -jnp.inf)

    verb_logit = jnp.where(state.discards > 0, verb, -jnp.inf)

    select_probs = jax.nn.sigmoid(select_logits)
    verb_prob = jax.nn.sigmoid(verb_logit)

    select = jax.random.uniform(key1, (32,)) < select_probs
    verb = jax.random.uniform(key2) < verb_prob

    temp = jnp.concat([verb_prob[None], select_probs])

    act = balatroAction(verb=verb.astype(jnp.int32), select=select.astype(jnp.int32))

    log_prob = jnp.where(jnp.concat([verb[None], select]) > 0, jnp.log(temp), jnp.log(1-temp)).sum()

    return act, log_prob


if __name__ == "__main__":
    # dev scratch: the known state from run_env_step_test — mid-round, full
    # 8-card hand, two pair of Ks (lanes 1/3) + 7s (lanes 4/6), unselected
    # pair of 9s at lanes 2/5. chips 60/300, 3 hands, 2 discards.
    # Feed it to observe() and eyeball the encoding with dbg().
    hand_ids = [2, 24, 33, 11, 5, 46, 44, 27]
    rest = [i for i in range(52) if i not in hand_ids]
    state = balatroEnvState(
        hand=_hand_lanes(hand_ids),
        hand_size=jnp.int32(8),
        hand_valid=_valid_lanes([1, 1, 1, 1, 1, 1, 1, 1]),
        deck=jnp.array(hand_ids + rest + list(range(52, MAX_DECK_SIZE)),
                       dtype=jnp.int32),
        deck_size=jnp.int32(52),
        deck_ptr=jnp.int32(8),
        hands=jnp.int32(3),
        discards=jnp.int32(2),
        handcount=jnp.int32(4),
        discardcount=jnp.int32(3),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(60),
        blind_target=jnp.int32(300),
        key=jax.random.key(0),
    )
    # the matching two-pair action, if you want to step it: verb 0 = PLAY
    action = balatroAction(verb=jnp.int32(0),
                           select=_valid_lanes([0, 1, 0, 1, 1, 0, 1, 0]))

    # show(state, "agent dev state")

    key = jax.random.key(42)
    init_key, act_key = jax.random.split(key)

    model = init_params(init_key)

    res, log_prob = act(model, state, act_key)
    print(f"verb {res.verb}, log_prob {log_prob}")
    dbg(select=res.select, hand_valid=state.hand_valid)
    