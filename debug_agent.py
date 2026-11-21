"""Behavior inspector for a trained (or untrained) policy, on full runs.

  inspect_run(params, key, scale)     -- one run, step by step: stage, the choice,
                                         the cards, the hand made, money, reward.
  behavior_stats(params, key, scale)  -- many runs at once (vmapped): antes reached,
                                         win rate, $ per run, the action mix, the
                                         hand types played.

    python debug_agent.py               # runs/params_run.pkl if present
    python debug_agent.py fresh         # a random (untrained) policy
    python debug_agent.py 0.25          # at target scale 0.25 (default 1.0 = real game)
    python debug_agent.py --deck plasma --stake gold
                                        # the deck and the stake (default: the ones of
                                        # runs/ckpt_run.pkl, else Red Deck, White Stake)
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
from agent import (init_params, act, action_name, masked_logits, legal_mask, move_masks, observe,
                   is_move, A_PLAY, A_DISCARD, A_MOVE_CARD, A_BUY, A_SELL, A_OPEN, A_USE, A_SELL_CONS,
                   N_ACTIONS, value_parts)
from env import STAGE_PACK, STAGE_NAMES, card_id
from items import CONS_NAMES, PACK_NAMES, BACK_NAMES, STAKE_NAMES, IK_JOKER, IK_CARD, B_RED
from jokers import JOKER_NAMES
from scoring import HAND_NAMES
from train import load_params, load_ckpt, parse_deck, parse_stake, PARAMS_PATH, CKPT_PATH, DECK, HAND

BACK, STAKE = B_RED, 1      # the deck and the stake of every run here (set in __main__)


def new_run(key, scale):
    return reset(key, DECK, HAND, scale, stake=STAKE, back=BACK)

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
    hd = masked_logits(params, observe(state), legal_mask(state), state.hand_valid,
                       move_masks(state))
    return jax.nn.softmax(hd.a), hd.value


# ---------- one run, verbose ----------
def inspect_run(params, key, scale=1.0, max_steps=400):
    key, sk = jax.random.split(key)
    state = new_run(sk, scale)
    print(f"=== one run at target scale {scale} ===")
    total = 0
    for step in range(max_steps):
        key, sk = jax.random.split(key)
        action, st = _act(params, state, sk)
        a, sel, src, dst, value = st.a, st.sel, st.src, st.dst, st.value
        p, _ = _probs(params, state)
        a = int(a)
        where = (f"A{int(state.ante)} {BLIND_NAMES[int(state.blind_index)][:5]}"
                 if int(state.game_stage) == STAGE_BLIND else
                 f"A{int(state.ante)} {STAGE_NAMES[int(state.game_stage)].lower()[:5]:5s}")
        name = action_name(a, src, dst) if is_move(a) else action_name(a)
        line = f" {step:3d} {where}  ${int(state.money):3d}  {name:12s} p={float(p[a]):.2f}"
        if a in (A_PLAY, A_DISCARD):
            cards = " ".join(_card(card_id(state, state.hand[i])) + ("*" if bool(sel[i]) else " ")
                             for i in range(int(state.hand_size)) if bool(state.hand_valid[i]))
            line += f"  {cards}"
            if a == A_PLAY:
                res = explain_play(state, sel)
                line += f"  {HAND_NAMES[int(res.handtype)]} {int(res.score)}" \
                        f"  ({int(state.chips_scored) + int(res.score)}/{int(state.blind_target)})"
        elif a == A_MOVE_CARD:
            line += f"  moves {_card(card_id(state, state.hand[int(src)]))}"
        elif A_USE <= a < A_SELL_CONS:
            line += f"  {CONS_NAMES[int(state.consumables[a - A_USE])]}"
        elif A_BUY <= a < A_SELL:
            i = a - A_BUY
            it = int(state.shop_slots[i])
            kind = int(state.shop_kind[i])
            line += "  " + (JOKER_NAMES[it] if kind == IK_JOKER else
                            "23456789TJQKA"[int(state.shop_rank[i])] + "CDHS"[int(state.shop_suit[i])]
                            if kind == IK_CARD else CONS_NAMES[it])
        elif A_OPEN <= a < A_USE:
            line += f"  {PACK_NAMES[int(state.pack_slots[a - A_OPEN])]}"
        state, reward, done = _step(state, action)
        total += int(reward)
        if int(reward):
            line += f"  -> cashout ${int(reward)}"
        vw, vm = value_parts(value)
        print(line + (f"  Vwin={float(vw):.2f}" if vw is not None else "") + f"  V$={float(vm):.2f}")
        if bool(done):
            break
    end = {STAGE_WON: "WON", STAGE_LOST: "LOST"}.get(int(state.game_stage), "cut off")
    print(f" -> {end} at ante {int(state.ante)}, ${total} cashed out, "
          f"jokers {[JOKER_NAMES[int(j)] for j in state.jokers if int(j) >= 0]}\n")


# ---------- many runs, aggregate ----------
def _run_batch(params, keys, scale, max_steps):
    def one(key):
        state = new_run(key, scale)
        def step(c, k):
            s, done = c
            action, st = act(params, s, k)
            a, sel = st.a, st.sel
            res = explain_play(s, sel)
            nxt, r, d = env_step(s, action)
            live = ~done
            rec = {"a": jnp.where(live, a, -1), "hand": jnp.where(live & (a == A_PLAY), res.handtype, -1),
                   "r": r * live, "n": jnp.where(live & ((a == A_PLAY) | (a == A_DISCARD)), sel.sum(), -1)}
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
    pa, n = recs["a"], recs["n"]
    for name, v in (("PLAY", A_PLAY), ("DISCARD", A_DISCARD)):
        k = n[(pa == v) & (n >= 0)]
        print(f" cards per {name:8s} {k.mean() if len(k) else 0:7.2f}   " +
              "  ".join(f"{c}:{(k == c).mean():.0%}" for c in range(1, 6)))
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
    opt = {}
    for flag in ("--deck", "--stake"):
        if flag in args:
            i = args.index(flag)
            opt[flag] = args[i + 1]
            del args[i:i + 2]
    if os.path.exists(CKPT_PATH) and "fresh" not in args:      # the game the params were trained on
        BACK, STAKE = load_ckpt().get("game", (B_RED, 1))
    try:
        BACK = parse_deck(opt["--deck"]) if "--deck" in opt else BACK
        STAKE = parse_stake(opt["--stake"]) if "--stake" in opt else STAKE
    except ValueError as e:
        sys.exit(str(e))
    print(f"game {BACK_NAMES[BACK]}, {STAKE_NAMES[STAKE - 1]}")
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
