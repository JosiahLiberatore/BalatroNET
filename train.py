import datetime
import os
import pickle
import sys

import jax
import jax.numpy as jnp
from tensorboardX import SummaryWriter

from env import balatroAction, balatroEnvState
from env import MAX_DECK_SIZE, MAX_HAND_SIZE
from env import show, dbg, reset, env_step
from agent import init_params, observe, policy, act
from debug_agent import save_params

LEARNT_RATE = 1e-3
B1_RATE = 0.9
B2_RATE = 0.999
ENTROPY_BETA = 0.01

def rollout(params, key):

    def step_fn(carry: tuple[balatroEnvState, jax.Array, jax.Array], x):
        state, key, done = carry
        key, subkey = jax.random.split(key)
        action, log_prob, (verb_p, entropy) = act(params, state, subkey)
        log_prob = log_prob * ~done
        state, chips, temp_done = env_step(state, action)
        chips = chips * ~done
        real_done = done | temp_done
        return (state, key, real_done), (chips, log_prob, action, verb_p * ~done, entropy * ~done)
    
    key, subkey = jax.random.split(key)
    state = reset(subkey, 52, 8)
    finalstate, output = jax.lax.scan(step_fn, (state, key, jnp.bool_(False)), xs=None, length=7)

    return output

@jax.jit
def train_step(params, key, m, v, t):
    # runs
    t = t + 1
    keys = jax.random.split(key, 512)
    def loss_fn(params):
        rewards, log_probs, actions, verb_p, entropy = jax.vmap(rollout, in_axes=(None, 0))(params, keys)
        returns = jax.lax.cumsum(rewards, axis=1, reverse=True)
        advantage = returns - returns.mean(axis=0)
        advantage = advantage / (advantage.std() + 1e-9)
        loss = -(log_probs * advantage).mean() - (ENTROPY_BETA * entropy.mean())
        mean_return = rewards.sum(axis=1).mean()
        # probe means are diluted by zeroed post-done steps; trends still read fine
        return loss, (mean_return, verb_p.mean(), entropy.mean())

    (loss, (mean_return, mean_verb_p, mean_entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    b1 = B1_RATE
    b2 = B2_RATE

    m = jax.tree.map(lambda m, grads: b1 * m + (1-b1) * grads, m, grads)
    v = jax.tree.map(lambda v, grads: b2 * v + (1-b2) * grads**2, v, grads)
    m_hat = jax.tree.map(lambda m: m / (1-b1**t), m)
    v_hat = jax.tree.map(lambda v: v / (1-b2**t), v)

    # Adam GD
    params = jax.tree.map(lambda p, m_hat, v_hat: p - (LEARNT_RATE * m_hat / (jnp.sqrt(v_hat) + 1e-8)), params, m_hat, v_hat)

    # SGD
    # params = jax.tree.map(lambda p, g: p - (LEARNT_RATE * g), params, grads)

    # instrumentation only: nothing below feeds the update above
    grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    metrics = {"return": mean_return, "loss": loss, "grad_norm": grad_norm,
               "verb_p": mean_verb_p, "entropy": mean_entropy}
    return params, m, v, t, metrics

# ---------- checkpointing: full training state, so runs can stop and resume ----------
# A resumable run needs more than params: Adam's m/v/t (its memory of the gradient
# landscape), the carried RNG key (so the episode stream continues, not repeats),
# the step counter, and the tensorboard run_dir (so curves stay continuous).
CKPT_EVERY = 1000
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
CKPT_PATH = os.path.join(RUNS_ROOT, "ckpt_latest.pkl")

def save_ckpt(params, m, v, t, key, step, run_dir, path=CKPT_PATH):
    ckpt = {
        "params": jax.device_get(params),
        "m": jax.device_get(m),
        "v": jax.device_get(v),
        "t": jax.device_get(t),
        # typed PRNG keys don't pickle; store the raw uint32 words instead
        "key_data": jax.device_get(jax.random.key_data(key)),
        "step": step,
        "run_dir": run_dir,
    }
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


if __name__ == "__main__":
    # dev scratch: eyeball rollout. Fresh params -> probs ~ 0.5, so every REAL
    # step's log_prob should sit near 9 * log(0.5) = -6.2383 (8 lanes + verb).
    # Steps after the episode ends must be EXACTLY 0.0 in both columns.
    # Episode length varies with the key: 4 plays minimum, +1 per discard taken.
    expected = 9 * jnp.log(0.5)

    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    model = init_params(init_key)

    m = jax.tree.map(jnp.zeros_like, model)
    v = jax.tree.map(jnp.zeros_like, model)
    t = jnp.float32(0.0)
    start_step = 0

    # absolute path anchored to THIS script, so no matter where python is
    # launched from (terminal, Code Runner, ...) runs always land in the same
    # place. Each run gets its own timestamped subfolder so old curves stay
    # for comparison; a RESUMED run reuses its dir so the curve continues.
    run_dir = os.path.join(RUNS_ROOT, f"{datetime.datetime.now():%Y%m%d-%H%M%S}")

    # `python train.py`       -> resume from runs/ckpt_latest.pkl if it exists
    # `python train.py fresh` -> ignore any checkpoint, start a new run
    fresh = len(sys.argv) > 1 and sys.argv[1] == "fresh"
    if not fresh and os.path.exists(CKPT_PATH):
        ckpt = load_ckpt()
        model, m, v, t = ckpt["params"], ckpt["m"], ckpt["v"], ckpt["t"]
        key, start_step, run_dir = ckpt["key"], ckpt["step"], ckpt["run_dir"]
        print(f"RESUMING from step {start_step}  ({CKPT_PATH})")
        print( "(run `python train.py fresh` to start over instead)")

    STEPS = 100000
    if start_step >= STEPS:
        print(f"checkpoint is already at step {start_step} >= STEPS={STEPS}; "
              f"raise STEPS to train further, or run `python train.py fresh`.")
        sys.exit(0)

    writer = SummaryWriter(run_dir)
    print(f"logging to {run_dir}")
    print(f"view with:  tensorboard --logdir {RUNS_ROOT}")

    completed = start_step
    try:
        for step in range(start_step, STEPS):
            key, subkey = jax.random.split(key)
            model, m, v, t, metrics = train_step(model, subkey, m, v, t)
            for name, value in metrics.items():
                writer.add_scalar(name, float(value), step)
            completed = step + 1
            if step % 25 == 0:
                print(f"step {step:4d}   return {float(metrics['return']):7.2f}   "
                      f"loss {float(metrics['loss']):8.3f}   "
                      f"grad_norm {float(metrics['grad_norm']):8.3f}")
            if completed % CKPT_EVERY == 0:
                save_ckpt(model, m, v, t, key, completed, run_dir)
                save_params(model)   # params-only, so debug_agent can peek mid-run
    except KeyboardInterrupt:
        print(f"\ninterrupted -- saving checkpoint at step {completed}")

    writer.close()
    save_ckpt(model, m, v, t, key, completed, run_dir)
    save_params(model)
    print(f"stopped at step {completed}/{STEPS}. resume with:  python train.py")
    print(f"view with:  tensorboard --logdir {RUNS_ROOT}")
    print("inspect with:  python debug_agent.py")

    # print(f"expected real-step log_prob ~ {expected:.4f}\n")
    # for i, k in enumerate(jax.random.split(key, 3)):
    #     rewards, log_probs, actions = rollout(model, k)
    #     print(f"rollout {i}:")
    #     for step in range(7):
    #         verb = "discard" if actions.verb[step] else "play   "
    #         select = "".join(str(int(b)) for b in actions.select[step][:8])
    #         print(f"  step {step}:  {verb} {select}   log_prob {log_probs[step]:+9.4f}   reward {rewards[step]:8.1f}")
    #     print(f"  total reward {rewards.sum():.1f}\n")

