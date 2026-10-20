"""A2C + GAE on full Balatro runs, with an adaptive blind-target curriculum.

    python train.py                  resume from runs/ckpt_run.pkl if it exists
    python train.py fresh            start a new run
    python train.py --curriculum     use the target-scale curriculum (off by default)
    python train.py --steps 100000   number of updates (default 20000)
    python train.py --move-alpha 0.5 --move-beta 1.0   the MOVE streak penalty

EPISODE  = one full run (8 antes, the shop included). Envs AUTO-RESET: each
           update steps every env T more steps and a finished run restarts at
           once, so no scan step is wasted on a padded tail.
REWARD   = cashout $ per cleared blind (env_step's contract), plus the MOVE
           streak penalty (see MOVE_ALPHA); x REWARD_SCALE for the learner only.
           The logged $/run is the env's cashout alone.
ADVANTAGE= GAE(gamma, lambda) on the value head; bootstrap from the value of
           the carried state at the end of the segment, cut at run ends.
CURRICULUM: OFF by default: target_scale stays 1.0, the real game.
           With --curriculum, target_scale multiplies every blind target
           (env.blind_target), starting at SCALE_START. The clear rate (blinds
           cleared / blinds ended) moves it: above CLEAR_HI the scale rises,
           below CLEAR_LO it falls. New runs read the current scale at reset.
"""

import datetime
import os
import pickle
import sys

import jax
import jax.numpy as jnp

from env import reset, env_step, tree_where, STAGE_LOST, STAGE_WON, STAGE_SHOP
from agent import (init_params, act, masked_logits, log_prob_entropy, is_move, OBS_DIM,
                   A_BUY, A_SELL, A_MOVE_JOKER, A_REROLL, N_ACTIONS)

# --- config -----------------------------------------------------------------------
N_ENVS        = 1024
T             = 64          # env steps per env per update
GAMMA         = 0.995
LAMBDA        = 0.95
LEARNT_RATE   = 3e-4
B1_RATE       = 0.9
B2_RATE       = 0.999
VF_COEF       = 0.5
ENTROPY_BETA  = 0.01
MAX_GRAD_NORM = 0.5
REWARD_SCALE  = 0.1         # cashout $ -> value-head units

# MOVE streak penalty (learner reward only; the env's money never changes):
# the k-th MOVE in a row gets -MOVE_BETA * (1 - MOVE_ALPHA ** (k - 1)) $.
# k = 1 is free, then -0.50, -0.75, -0.88, ... -> -MOVE_BETA. Any non-MOVE
# action sets k back to 0. Set from --move-alpha / --move-beta.
MOVE_ALPHA    = 0.5
MOVE_BETA     = 1.0         # $ per move at full strength

SCALE_START   = 0.02        # ante-1 small blind = 6 chips: one card clears it
SCALE_MAX     = 1.0
SCALE_UP      = 1.10
SCALE_DOWN    = 1 / 1.05
CLEAR_HI      = 0.92        # EMA clear rate above this -> harder
CLEAR_LO      = 0.80        # below this -> easier
CLEAR_EMA     = 0.95
SCALE_COOLDOWN = 25         # updates between two scale changes

DECK, HAND = 52, 8


# --- rollout ----------------------------------------------------------------------
def init_carry(key, scale):
    """env states + per-env running run stats, all leading axis N_ENVS."""
    keys = jax.random.split(key, N_ENVS)
    states = jax.vmap(lambda k: reset(k, DECK, HAND, scale))(keys)
    zeros = jnp.zeros(N_ENVS)
    return {"state": states, "ret": zeros, "buys": zeros, "steps": zeros,
            "streak": zeros, "pen": zeros}


def move_penalty(streak):
    """$ penalty of the streak-th MOVE in a row (streak >= 1); 0 for streak 0."""
    return jnp.where(streak > 0, -MOVE_BETA * (1.0 - MOVE_ALPHA ** jnp.maximum(streak - 1, 0)), 0.0)


def env_step_auto(params, state, run, key, scale):
    """One policy step on one env; a finished run resets in place.
    run = the per-env running stats (ret, buys, steps, streak, pen)."""
    k_act, k_reset = jax.random.split(key)
    action, (a, sel, src, dst, obs, legal, mmask, value, logp) = act(params, state, k_act)
    nxt, reward, done = env_step(state, action)
    reward = reward.astype(jnp.float32)

    moved = is_move(a)
    streak = jnp.where(moved, run["streak"] + 1, 0.0)
    pen = move_penalty(streak)
    run = {"ret": run["ret"] + reward, "buys": run["buys"] + ((a >= A_BUY) & (a < A_SELL)),
           "steps": run["steps"] + 1, "streak": streak, "pen": run["pen"] + pen}
    finished = {"done": done, "ret": run["ret"], "ante": nxt.ante.astype(jnp.float32),
                "won": nxt.game_stage == STAGE_WON, "buys": run["buys"], "steps": run["steps"],
                "pen": run["pen"],
                "cleared": reward > 0, "failed": done & (nxt.game_stage == STAGE_LOST),
                "in_shop": state.game_stage == STAGE_SHOP,
                "sell": (a >= A_SELL) & (a < A_MOVE_JOKER), "move": moved,   # joker + card moves
                "streak": streak, "reroll": a == A_REROLL}
    fresh = reset(k_reset, DECK, HAND, scale)
    nxt = tree_where(done, fresh, nxt)
    keep = ~done
    traj = {"obs": obs, "legal": legal, "hand_valid": state.hand_valid, "mmask": mmask,
            "a": a, "sel": sel, "src": src, "dst": dst,
            "reward": (reward + pen) * REWARD_SCALE, "done": done, "value": value}
    return nxt, jax.tree.map(lambda x: x * keep, run), traj, finished


def rollout(params, carry, key, scale):
    """T steps on every env. -> (carry, traj [T, N], finished [T, N], last_value [N])"""
    def step(c, k):
        ks = jax.random.split(k, N_ENVS)
        run = {n: c[n] for n in ("ret", "buys", "steps", "streak", "pen")}
        s, run, traj, fin = jax.vmap(env_step_auto, in_axes=(None, 0, 0, 0, None))(
            params, c["state"], run, ks, scale)
        return {"state": s, **run}, (traj, fin)

    carry, (traj, fin) = jax.lax.scan(step, carry, jax.random.split(key, T))
    _, (*_, last_value, _) = jax.vmap(act, in_axes=(None, 0, 0))(
        params, carry["state"], jax.random.split(key, N_ENVS))
    return carry, traj, fin, last_value


def gae(reward, value, done, last_value):
    """Generalized advantage estimation, scanned backward over T. All [T, N]."""
    def back(next_adv_value, x):
        next_adv, next_value = next_adv_value
        r, v, d = x
        nonterminal = 1.0 - d
        delta = r + GAMMA * next_value * nonterminal - v
        adv = delta + GAMMA * LAMBDA * nonterminal * next_adv
        return (adv, v), adv
    _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value),
                          (reward, value, done.astype(jnp.float32)), reverse=True)
    return adv, adv + value


# --- update -----------------------------------------------------------------------
def loss_fn(params, batch):
    hd = jax.vmap(masked_logits, in_axes=(None, 0, 0, 0, 0, 0))(
        params, batch["obs"], batch["legal"], batch["hand_valid"], batch["mmask"], batch["src"])
    logp, ent = jax.vmap(log_prob_entropy)(hd, batch["a"], batch["sel"], batch["src"],
                                           batch["dst"], batch["hand_valid"])
    value = hd.value
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = -(logp * adv).mean()
    v_loss = 0.5 * ((value - batch["ret"]) ** 2).mean()
    entropy = ent.mean()
    loss = pg_loss + VF_COEF * v_loss - ENTROPY_BETA * entropy
    return loss, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}


@jax.jit
def train_step(params, m, v, t, carry, key, scale):
    carry, traj, fin, last_value = rollout(params, carry, key, scale)
    adv, ret = gae(traj["reward"], traj["value"], traj["done"], last_value)
    flat = lambda x: x.reshape((T * N_ENVS,) + x.shape[2:])
    batch = {k: flat(traj[k]) for k in ("obs", "legal", "hand_valid", "mmask", "a", "sel", "src", "dst")}
    batch["adv"], batch["ret"] = flat(adv), flat(ret)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

    # global-norm clip, then hand-rolled Adam (same update as v0.1)
    grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, MAX_GRAD_NORM / (grad_norm + 1e-8)), grads)
    t = t + 1
    b1, b2 = B1_RATE, B2_RATE
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, v, grads)
    params = jax.tree.map(lambda p, m, v: p - LEARNT_RATE * (m / (1 - b1 ** t)) /
                          (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8), params, m, v)

    # instrumentation: nothing below feeds the update
    d = fin["done"]
    n_done = d.sum()
    per_run = lambda x: jnp.where(d, x, 0).sum() / jnp.maximum(n_done, 1)
    n_cleared, n_failed = fin["cleared"].sum(), fin["failed"].sum()
    var_ret = ret.var()
    metrics = {
        "loss": loss, **aux, "grad_norm": grad_norm,
        "explained_var": 1 - (ret - traj["value"]).var() / (var_ret + 1e-8),
        "runs_finished": n_done,
        "run_return": per_run(fin["ret"]),
        "run_ante": per_run(fin["ante"]),
        "run_steps": per_run(fin["steps"]),
        "run_buys": per_run(fin["buys"]),
        "run_move_penalty": per_run(fin["pen"]),          # $ the learner lost to MOVE streaks per run
        "move_streak_max": fin["streak"].max(),
        "win_rate": per_run(fin["won"].astype(jnp.float32)),
        "blinds_cleared": n_cleared,
        "blinds_failed": n_failed,
        "clear_rate": n_cleared / jnp.maximum(n_cleared + n_failed, 1),
        "shop_step_frac": fin["in_shop"].mean(),
        "sell_frac": fin["sell"].mean(),
        "move_frac": fin["move"].mean(),
        "reroll_frac": fin["reroll"].mean(),
    }
    return params, m, v, t, carry, metrics


# --- checkpointing ------------------------------------------------------------------
# A resumable run needs more than params: Adam's m/v/t, the carried RNG key, the
# step counter, the curriculum scale, and the tensorboard run_dir. The env carry
# is NOT saved: a resume starts every env on a fresh run.
CKPT_EVERY = 500
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
CKPT_PATH = os.path.join(RUNS_ROOT, "ckpt_run.pkl")
PARAMS_PATH = os.path.join(RUNS_ROOT, "params_run.pkl")


def save_params(params, path=PARAMS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)


def load_params(path=PARAMS_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_ckpt(params, m, v, t, key, step, run_dir, scale, clear_ema, curriculum, move_cfg=(MOVE_ALPHA, MOVE_BETA), path=CKPT_PATH):
    ckpt = {"params": jax.device_get(params), "m": jax.device_get(m), "v": jax.device_get(v),
            "t": jax.device_get(t),
            "key_data": jax.device_get(jax.random.key_data(key)),   # typed keys don't pickle
            "step": step, "run_dir": run_dir, "scale": scale, "clear_ema": clear_ema,
            "curriculum": curriculum, "move_cfg": tuple(move_cfg),
            "obs_dim": OBS_DIM, "n_actions": N_ACTIONS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, path)   # atomic: a crash mid-save never corrupts the old ckpt


def load_ckpt(path=CKPT_PATH):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["key"] = jax.random.wrap_key_data(jnp.asarray(ckpt.pop("key_data")))
    return ckpt


def next_scale(scale, clear_ema, since_change):
    """The curriculum rule (host side, plain Python). -> (scale, changed)"""
    if since_change < SCALE_COOLDOWN:
        return scale, False
    if clear_ema > CLEAR_HI and scale < SCALE_MAX:
        return min(scale * SCALE_UP, SCALE_MAX), True
    if clear_ema < CLEAR_LO and scale > SCALE_START:
        return max(scale * SCALE_DOWN, SCALE_START), True
    return scale, False


if __name__ == "__main__":
    import argparse
    from tensorboardX import SummaryWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["fresh"], help="fresh = ignore the checkpoint")
    ap.add_argument("--curriculum", action="store_true",
                    help="start blind targets at SCALE_START and raise them with the clear rate")
    ap.add_argument("--steps", type=int, default=20000, help="total updates")
    ap.add_argument("--move-alpha", type=float, default=MOVE_ALPHA,
                    help="MOVE streak decay: the k-th move in a row costs beta * (1 - alpha^(k-1))")
    ap.add_argument("--move-beta", type=float, default=MOVE_BETA,
                    help="MOVE streak penalty at full strength, in $ (0 = moves are free)")
    args = ap.parse_args()
    curriculum = args.curriculum
    MOVE_ALPHA, MOVE_BETA = args.move_alpha, args.move_beta   # read when train_step traces
    move_cfg = (MOVE_ALPHA, MOVE_BETA)

    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    t = jnp.float32(0.0)
    start_step, scale, clear_ema = 0, SCALE_START if curriculum else 1.0, 0.0
    tag = "curr" if curriculum else "real"
    run_dir = os.path.join(RUNS_ROOT, f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{tag}")

    if args.mode != "fresh" and os.path.exists(CKPT_PATH):
        ckpt = load_ckpt()
        if ckpt.get("obs_dim") != OBS_DIM or ckpt.get("n_actions") != N_ACTIONS:
            sys.exit(f"{CKPT_PATH} was saved for a different obs/action layout; "
                     f"run `python train.py fresh`.")
        if ckpt.get("curriculum", True) != curriculum:
            sys.exit(f"{CKPT_PATH} was saved with curriculum={ckpt.get('curriculum', True)}; "
                     f"pass the same setting or run `python train.py fresh`.")
        if tuple(ckpt.get("move_cfg", (None, None))) != move_cfg:
            sys.exit(f"{CKPT_PATH} was saved with (move_alpha, move_beta)={ckpt.get('move_cfg')}; "
                     f"pass the same values or run `python train.py fresh`.")
        params, m, v, t = ckpt["params"], ckpt["m"], ckpt["v"], ckpt["t"]
        key, start_step, run_dir = ckpt["key"], ckpt["step"], ckpt["run_dir"]
        scale, clear_ema = ckpt["scale"], ckpt["clear_ema"]
        print(f"RESUMING from step {start_step} at scale {scale:.3f}  ({CKPT_PATH})")
        print("(run `python train.py fresh` to start over instead)")

    STEPS = args.steps
    writer = SummaryWriter(run_dir)
    print(f"devices {jax.devices()}   envs {N_ENVS} x T {T} = {N_ENVS * T} steps/update")
    print(f"curriculum {'ON' if curriculum else 'OFF (real blind targets)'}   updates {STEPS}   "
          f"move streak penalty alpha {MOVE_ALPHA} beta ${MOVE_BETA}")
    print(f"logging to {run_dir}\nview with:  tensorboard --logdir {RUNS_ROOT}")

    key, ck = jax.random.split(key)
    carry = init_carry(ck, scale)
    completed, since_change = start_step, SCALE_COOLDOWN
    t0 = datetime.datetime.now()
    try:
        for step in range(start_step, STEPS):
            key, subkey = jax.random.split(key)
            params, m, v, t, carry, metrics = train_step(params, m, v, t, carry, subkey,
                                                         jnp.float32(scale))
            metrics = {k: float(x) for k, x in metrics.items()}
            if metrics["blinds_cleared"] + metrics["blinds_failed"] > 0:
                clear_ema = CLEAR_EMA * clear_ema + (1 - CLEAR_EMA) * metrics["clear_rate"]
            if curriculum:
                scale, changed = next_scale(scale, clear_ema, since_change)
                since_change = 0 if changed else since_change + 1
            metrics["target_scale"], metrics["clear_ema"] = scale, clear_ema
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            completed = step + 1
            if step % 10 == 0:
                rate = (step - start_step + 1) * N_ENVS * T / (datetime.datetime.now() - t0).total_seconds()
                print(f"step {step:5d}  $/run {metrics['run_return']:6.1f}  ante {metrics['run_ante']:4.2f}  "
                      f"win {metrics['win_rate']:5.1%}  clear {clear_ema:5.1%}  scale {scale:.3f}  "
                      f"buys {metrics['run_buys']:4.1f}  ent {metrics['entropy']:5.2f}  "
                      f"ev {metrics['explained_var']:5.2f}  move {metrics['move_frac']:4.0%} "
                      f"pen {metrics['run_move_penalty']:5.1f}  {rate / 1e3:5.0f}k st/s")
            if completed % CKPT_EVERY == 0:
                save_ckpt(params, m, v, t, key, completed, run_dir, scale, clear_ema, curriculum, move_cfg)
                save_params(params)
    except KeyboardInterrupt:
        print(f"\ninterrupted -- saving checkpoint at step {completed}")

    writer.close()
    save_ckpt(params, m, v, t, key, completed, run_dir, scale, clear_ema, curriculum, move_cfg)
    save_params(params)
    resume = ("python train.py" + (" --curriculum" if curriculum else "")
              + f" --steps {STEPS} --move-alpha {MOVE_ALPHA} --move-beta {MOVE_BETA}")
    print(f"stopped at step {completed}/{STEPS}. resume with:  {resume}")
    print("inspect with:  python debug_agent.py")
