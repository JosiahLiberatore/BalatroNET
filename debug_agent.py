"""Behavior inspector for a trained (or untrained) policy.

Two views, both run the agent EAGERLY (plain Python loop, no jit/scan) so every
value is concrete and printable:

  inspect_episode(params, key)  -- one game, step by step: the state, what the
                                   agent chose, the hand it made, the reward.
  behavior_stats(params, key)   -- aggregate over many games: discard rate, cards
                                   selected, hand score, verb probability, entropy.
                                   This is the readout that confirms/refutes a
                                   hypothesis like "it never discards, just grabs
                                   high cards".

Trained params don't survive a training process, so train.py checkpoints them to
a .pkl; load with load_params and point either view at them:

    python debug_agent.py                 # inspects runs/params_latest.pkl if present
    python debug_agent.py fresh           # inspects a random (untrained) policy
"""

import os
import sys
import pickle

# the suit pips (♠♥♦♣) are non-cp1252; force utf-8 so the Windows console
# doesn't choke when printing them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
# on legacy Windows conhost this flips on ANSI escape processing; a no-op in
# Windows Terminal / VS Code where colors already work
if sys.platform == "win32":
    os.system("")

import jax
import jax.numpy as jnp

from env import balatroAction, balatroEnvState, MAX_HAND_SIZE
from env import reset, env_step, _card_name
from agent import init_params, observe, policy, act
from scoring import scoreHand, HAND_NAMES

RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
DEFAULT_CKPT = os.path.join(RUNS_ROOT, "params_latest.pkl")

# env names suits as letters (C D H S); swap in pips for the human-facing readout
SUIT_EMOJI = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
# 4-color deck: hearts red, diamonds blue, clubs green, spades black
RESET = "\033[0m"
SUIT_COLOR = {"♥": "\033[91m", "♦": "\033[94m", "♣": "\033[92m", "♠": "\033[30m"}

def _emoji_name(cid):
    """'K♠'-style name: _card_name gives 'KS', we replace the suit letter."""
    name = _card_name(int(cid))
    if len(name) == 2 and name[1] in SUIT_EMOJI:
        return name[0] + SUIT_EMOJI[name[1]]
    return name

def _color_card(cid):
    """(colored_str, visible_width) for a card cell: colorize just the suit pip
    so ANSI escapes never enter the width-measured length."""
    name = _emoji_name(cid)
    if len(name) == 2 and name[1] in SUIT_COLOR:
        col = SUIT_COLOR[name[1]]
        colored = f"{name[0]}{col}{name[1]}{RESET}" if col else name
        return colored, 2
    return name, len(name)


# ---------- checkpoint io (params is a dict of jax arrays) ----------
def save_params(params, path=DEFAULT_CKPT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)   # -> numpy, portable

def load_params(path=DEFAULT_CKPT):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- probe the policy WITHOUT sampling (deterministic, faithful) ----------
def _policy_probs(params, state):
    """The per-decision probabilities the policy assigns in this state, masked
    exactly the way act() masks them. Returns (verb_prob, select_probs[32])."""
    logits = policy(params, observe(state))
    verb_logit = jnp.where(state.discards > 0, logits[0], -jnp.inf)
    select_logits = jnp.where(state.hand_valid, logits[1:MAX_HAND_SIZE + 1], -jnp.inf)
    return jax.nn.sigmoid(verb_logit), jax.nn.sigmoid(select_logits)


def _binary_entropy(p):
    """Entropy of independent Bernoullis, summed. Masked lanes have p=0 -> 0 bits,
    so this is automatically the entropy over the ACTIVE decisions only."""
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    return float((-(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))).sum())


def _played_hand(state, action):
    """Reproduce env_step's selection to name the hand the agent would make."""
    select = (action.select & state.hand_valid).astype(bool)
    select = select & (jnp.cumsum(select) <= 5)   # env's 5-card cap
    order = jnp.argsort(~select)
    cards = state.hand[order][:5]
    valid5 = select[order][:5]
    handtype, _, score = scoreHand(cards % 13, cards // 13, valid5)
    names = [_card_name(int(c)) for c, v in zip(cards, valid5) if bool(v)]
    return HAND_NAMES[int(handtype)], names, int(score)


# ---------- one episode, verbose ----------
def inspect_episode(params, key):
    key, sk = jax.random.split(key)
    state = reset(sk, 52, 8)
    blind = int(state.blind_target)
    print(f"=== episode  (blind target {blind}) ===")
    print(" st  action   hand (*=selected)                made          sel    rew   chips  h/d   deck   p_dsc   logp")
    total, plays, discards = 0, 0, 0
    for step in range(7):
        verb_prob, select_probs = _policy_probs(params, state)
        key, sk = jax.random.split(key)
        action, log_prob, _ = act(params, state, sk)

        # env forces PLAY when no discards remain; show the EFFECTIVE verb,
        # and the EFFECTIVE selection (valid lanes, first 5, like env_step)
        eff_verb = int(action.verb) if int(state.discards) > 0 else 0
        sel = action.select.astype(bool) & state.hand_valid
        sel = sel & (jnp.cumsum(sel) <= 5)
        hand, _, _ = _played_hand(state, action)       # from the PRE-step state

        n = int(state.hand_size)
        cells, vis = [], 0
        for i in range(n):
            if bool(state.hand_valid[i]):
                card, w = _color_card(int(state.hand[i]))
            else:
                card, w = "--", 2
            cells.append(f"{card}{'*' if bool(sel[i]) else ' '}")
            vis += w + 1
        # pad by VISIBLE width (color escapes are zero-width); the old :31s
        # counted the escapes and would ragged the trailing columns
        hand_str = " ".join(cells)
        hand_str += " " * max(0, 31 - (vis + (n - 1)))

        made = hand if eff_verb == 0 else "--"
        # raw = every lane the policy flipped on; eff = after valid mask + 5-cap
        n_sel = f"{int(action.select.sum()):2d}/{int(sel.sum())}"
        pre_chips = int(state.chips_scored)
        hd = f"{int(state.hands)}/{int(state.discards)}"
        deck = f"{int(state.deck_ptr):2d}/{int(state.deck_size)}"

        state, reward, done = env_step(state, action)
        reward = int(reward)

        if eff_verb == 0:
            plays += 1
        else:
            discards += 1
        total += reward
        verbname = "PLAY" if eff_verb == 0 else "DISCARD"
        print(f" {step:2d}  {verbname:7s}  {hand_str}  {made:12s} {n_sel:5s} {reward:4d}  {pre_chips:3d}/{blind}  {hd:4s} {deck}  {float(verb_prob):5.2f}  {float(log_prob):+6.2f}")
        if bool(done):
            break
    cleared = total >= blind
    print(f" -> {plays} plays, {discards} discards, total {total}, "
          f"cleared: {'YES' if cleared else 'no'}\n")


# ---------- many episodes, aggregate ----------
def behavior_stats(params, key, n=256):
    """Run n episodes eagerly and average the tells. Slow-ish (eager) but this is
    a diagnostic you run occasionally, not in the training loop."""
    tot_reward = tot_plays = tot_discards = tot_selected = 0
    tot_score_when_play = n_plays_scored = 0
    tot_verb_p = tot_entropy = n_decisions = 0
    cleared = 0
    # what hand types does it actually PLAY, and what do they earn? A policy
    # dumping High Cards when pairs are on the board shows up right here.
    hand_counts, hand_reward = {}, {}

    for _ in range(n):
        key, sk = jax.random.split(key)
        state = reset(sk, 52, 8)
        ep_reward = 0
        for step in range(7):
            verb_prob, select_probs = _policy_probs(params, state)
            key, sk = jax.random.split(key)
            action, _, _ = act(params, state, sk)

            eff_verb = int(action.verb) if int(state.discards) > 0 else 0
            # name the hand from the PRE-step state (env_step reassigns `state`)
            played = _played_hand(state, action)[0] if eff_verb == 0 else None
            tot_selected += int((action.select.astype(bool) & state.hand_valid).sum())
            tot_verb_p += float(verb_prob)
            tot_entropy += _binary_entropy(jnp.concatenate([verb_prob[None], select_probs]))
            n_decisions += 1

            state, reward, done = env_step(state, action)
            reward = int(reward)
            ep_reward += reward
            if eff_verb == 0:
                tot_plays += 1
                hand_counts[played] = hand_counts.get(played, 0) + 1
                hand_reward[played] = hand_reward.get(played, 0) + reward
                if reward > 0:
                    tot_score_when_play += reward
                    n_plays_scored += 1
            else:
                tot_discards += 1
            if bool(done):
                break
        tot_reward += ep_reward
        cleared += int(ep_reward >= int(state.blind_target))

    acts = tot_plays + tot_discards
    print(f"=== behavior over {n} episodes ===")
    print(f" mean return          {tot_reward / n:8.1f}   (blind target 300)")
    print(f" blind clear rate     {cleared / n:8.1%}")
    print(f" discard rate         {tot_discards / max(acts,1):8.1%}   ({tot_discards} discards / {acts} actions)")
    print(f" mean cards selected  {tot_selected / max(acts,1):8.2f}")
    print(f" mean score / play    {tot_score_when_play / max(n_plays_scored,1):8.1f}   (two-pair is 108)")
    print(f" mean verb P(discard) {tot_verb_p / max(n_decisions,1):8.2f}")
    print(f" mean entropy         {tot_entropy / max(n_decisions,1):8.2f}   (fresh policy ~ ln2 * active decisions)")

    print(f"\n=== hand types played (over {tot_plays} plays) ===")
    for name in sorted(hand_counts, key=hand_counts.get, reverse=True):
        c = hand_counts[name]
        print(f" {name:16s} {c:5d}  {c / max(tot_plays,1):6.1%}   mean reward {hand_reward[name] / c:6.1f}")


if __name__ == "__main__":
    key = jax.random.key(0)
    key, ik = jax.random.split(key)

    if len(sys.argv) > 1 and sys.argv[1] == "fresh":
        params = init_params(ik)
        print("inspecting a FRESH (untrained) policy\n")
    elif os.path.exists(DEFAULT_CKPT):
        params = load_params()
        print(f"loaded trained params from {DEFAULT_CKPT}\n")
    else:
        params = init_params(ik)
        print(f"no checkpoint at {DEFAULT_CKPT}; inspecting a FRESH policy instead\n")

    key, k1 = jax.random.split(key)
    inspect_episode(params, k1)
    key, k2 = jax.random.split(key)
    behavior_stats(params, k2, n=256)
