"""Behavior inspector for a trained (or untrained) policy, on full runs.

  inspect_run(params, key, scale)     -- one run, step by step: stage, the choice,
                                         the cards, the hand made, money, reward.
  behavior_stats(params, key, scale)  -- many runs at once (vmapped): antes reached,
                                         win rate, $ per run, the action mix, the
                                         hand types played.

    python debug_agent.py               # runs/params_run.pkl if present
    python debug_agent.py fresh         # a random (untrained) policy
    python debug_agent.py 0.25          # at target scale 0.25 (default 1.0 = real game)
"""

import os
import sys
from collections import Counter

# the suit pips are non-cp1252; force utf-8 so the Windows console doesn't choke
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32":
    os.system("")     # ANSI colors on legacy conhost

import jax
import jax.numpy as jnp
import numpy as np

from env import reset, env_step, explain_play, _card_name, STAGE_BLIND, STAGE_WON, STAGE_LOST
from env import BLIND_NAMES
from agent import (init_params, act, action_name, masked_logits, legal_mask, observe,
                   A_PLAY, A_DISCARD, N_ACTIONS)
from jokers import JOKER_NAMES
from scoring import HAND_NAMES
from train import load_params, PARAMS_PATH, DECK, HAND

SUIT_PIP = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
RESET = "\033[0m"
SUIT_COLOR = {"♥": "\033[91m", "♦": "\033[94m", "♣": "\033[92m", "♠": ""}


def _card(cid):
    name = _card_name(int(cid))
    if len(name) == 2 and name[1] in SUIT_PIP:
        pip = SUIT_PIP[name[1]]
        return f"{name[0]}{SUIT_COLOR[pip]}{pip}{RESET}"
    return name


_act = jax.jit(act)
_step = jax.jit(env_step)


@jax.jit
def _probs(params, state):
    a_logits, s_logits, value = masked_logits(params, observe(state), legal_mask(state),
                                              state.hand_valid)
    return jax.nn.softmax(a_logits), value


# ---------- one run, verbose ----------
def inspect_run(params, key, scale=1.0, max_steps=400):
    key, sk = jax.random.split(key)
    state = reset(sk, DECK, HAND, scale)
    print(f"=== one run at target scale {scale} ===")
    total = 0
    for step in range(max_steps):
        key, sk = jax.random.split(key)
        action, (a, sel, _, _, value, logp) = _act(params, state, sk)
        p, _ = _probs(params, state)
        a = int(a)
        where = (f"A{int(state.ante)} {BLIND_NAMES[int(state.blind_index)][:5]}"
                 if int(state.game_stage) == STAGE_BLIND else f"A{int(state.ante)} shop ")
        line = f" {step:3d} {where}  ${int(state.money):3d}  {action_name(a):8s} p={float(p[a]):.2f}"
        if a in (A_PLAY, A_DISCARD):
            cards = " ".join(_card(state.hand[i]) + ("*" if bool(sel[i]) else " ")
                             for i in range(int(state.hand_size)) if bool(state.hand_valid[i]))
            line += f"  {cards}"
            if a == A_PLAY:
                res = explain_play(state, sel)
                line += f"  {HAND_NAMES[int(res.handtype)]} {int(res.score)}" \
                        f"  ({int(state.chips_scored) + int(res.score)}/{int(state.blind_target)})"
        elif a < N_ACTIONS - 2 and a >= 2:
            shop = [JOKER_NAMES[int(j)] if int(j) >= 0 else "-" for j in state.shop_slots]
            held = [JOKER_NAMES[int(j)] for j in state.jokers if int(j) >= 0]
            line += f"  shop {shop}  held {held}"
        state, reward, done = _step(state, action)
        total += int(reward)
        if int(reward):
            line += f"  -> cashout ${int(reward)}"
        print(line + f"  V={float(value):.2f}")
        if bool(done):
            break
    end = {STAGE_WON: "WON", STAGE_LOST: "LOST"}.get(int(state.game_stage), "cut off")
    print(f" -> {end} at ante {int(state.ante)}, ${total} cashed out, "
          f"jokers {[JOKER_NAMES[int(j)] for j in state.jokers if int(j) >= 0]}\n")


# ---------- many runs, aggregate ----------
def _run_batch(params, keys, scale, max_steps):
    def one(key):
        state = reset(key, DECK, HAND, scale)
        def step(c, k):
            s, done = c
            action, (a, sel, *_rest) = act(params, s, k)
            res = explain_play(s, sel)
            nxt, r, d = env_step(s, action)
            live = ~done
            rec = {"a": jnp.where(live, a, -1), "hand": jnp.where(live & (a == A_PLAY), res.handtype, -1),
                   "r": r * live}
            return (jax.tree.map(lambda x, y: jnp.where(done, x, y), s, nxt), done | d), rec
        (final, done), recs = jax.lax.scan(step, (state, jnp.bool_(False)),
                                           jax.random.split(key, max_steps))
        return final, done, recs
    return jax.jit(jax.vmap(one))(keys)


def behavior_stats(params, key, scale=1.0, n=512, max_steps=600):
    final, done, recs = _run_batch(params, jax.random.split(key, n), scale, max_steps)
    final, done, recs = jax.device_get((final, done, recs))
    a = recs["a"][recs["a"] >= 0]
    hands = recs["hand"][recs["hand"] >= 0]
    antes = Counter(int(x) for x in final.ante)
    print(f"=== behavior over {n} runs at target scale {scale} ===")
    print(f" finished            {done.mean():7.1%}   (cut off at {max_steps} steps)")
    print(f" win rate            {(final.game_stage == STAGE_WON).mean():7.1%}")
    print(f" $ cashed out / run  {recs['r'].sum(axis=1).mean():7.1f}")
    print(f" mean ante reached   {final.ante.mean():7.2f}   " +
          "  ".join(f"A{k}:{antes[k]}" for k in sorted(antes)))
    print(f" steps / run         {len(a) / n:7.1f}")
    print(f" jokers held at end  {(final.jokers >= 0).sum(axis=1).mean():7.2f}")
    print("\n action mix:")
    names = Counter(action_name(x).split()[0] for x in a)
    for k, c in names.most_common():
        print(f"   {k:8s} {c / len(a):6.1%}")
    print("\n hand types played:")
    hc = Counter(int(h) for h in hands)
    for h, c in hc.most_common():
        print(f"   {HAND_NAMES[h]:16s} {c / max(len(hands), 1):6.1%}")


if __name__ == "__main__":
    key = jax.random.key(0)
    key, ik = jax.random.split(key)
    args = sys.argv[1:]
    scale = next((float(x) for x in args if x.replace(".", "", 1).isdigit()), 1.0)
    if "fresh" in args:
        params = init_params(ik)
        print("inspecting a FRESH (untrained) policy\n")
    elif os.path.exists(PARAMS_PATH):
        params = load_params()
        print(f"loaded trained params from {PARAMS_PATH}\n")
    else:
        params = init_params(ik)
        print(f"no params at {PARAMS_PATH}; inspecting a FRESH policy instead\n")

    key, k1, k2 = jax.random.split(key, 3)
    inspect_run(params, k1, scale)
    behavior_stats(params, k2, scale)
