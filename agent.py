import jax
import jax.numpy as jnp

from env import balatroAction, balatroEnvState
from env import MAX_DECK_SIZE, MAX_HAND_SIZE
from env import _hand_lanes, _valid_lanes, show, dbg

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

    dbg(n = 30, hand = hand, deck=deck, undealt=undealt, undealt_mask=undealt_mask)

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
    pass



def act(params, obs, key):
    pass


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

    obs = observe(state)
    if obs is None:
        print("observe: no return yet")
    else:
        print(f"obs shape {obs.shape}, dtype {obs.dtype}")
        dbg(obs=obs)

