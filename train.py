import datetime
import os

import jax
import jax.numpy as jnp
from tensorboardX import SummaryWriter

from env import balatroAction, balatroEnvState
from env import MAX_DECK_SIZE, MAX_HAND_SIZE
from env import _hand_lanes, _valid_lanes, show, dbg, reset, env_step
from agent import init_params, observe, policy, act

LEARNT_RATE = 5e-2

def rollout(params, key):

    def step_fn(carry: tuple[balatroEnvState, jax.Array, jax.Array], x):
        state, key, done = carry
        key, subkey = jax.random.split(key)
        action, log_prob = act(params, state, subkey)
        log_prob = log_prob * ~done
        state, chips, temp_done = env_step(state, action)
        chips = chips * ~done
        real_done = done | temp_done
        return (state, key, real_done), (chips, log_prob, action)
    
    key, subkey = jax.random.split(key)
    state = reset(subkey, 52, 8)
    finalstate, output = jax.lax.scan(step_fn, (state, key, jnp.bool_(False)), xs=None, length=7)

    return output

@jax.jit
def train_step(params, key):
    # runs
    keys = jax.random.split(key, 512)
    def loss_fn(params):
        rewards, log_probs, actions = jax.vmap(rollout, in_axes=(None, 0))(params, keys)
        returns = jax.lax.cumsum(rewards, axis=1, reverse=True)
        advantage = returns - returns.mean(axis=0)
        advantage = advantage / (advantage.std() + 1e-9)
        loss = -(log_probs * advantage).mean()
        mean_return = rewards.sum(axis=1).mean()
        return loss, mean_return

    (loss, mean_return), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    params = jax.tree.map(lambda p, g: p - (LEARNT_RATE * g), params, grads)

    # instrumentation only: nothing below feeds the update above
    grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    metrics = {"return": mean_return, "loss": loss, "grad_norm": grad_norm}
    return params, metrics

if __name__ == "__main__":
    # dev scratch: eyeball rollout. Fresh params -> probs ~ 0.5, so every REAL
    # step's log_prob should sit near 9 * log(0.5) = -6.2383 (8 lanes + verb).
    # Steps after the episode ends must be EXACTLY 0.0 in both columns.
    # Episode length varies with the key: 4 plays minimum, +1 per discard taken.
    expected = 9 * jnp.log(0.5)

    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    model = init_params(init_key)

    # absolute path anchored to THIS script, so no matter where python is
    # launched from (terminal, Code Runner, ...) runs always land in the same
    # place: D:\Claude\BalatroNET\runs. Each run gets its own timestamped
    # subfolder so old curves stay for comparison.
    runs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_dir = os.path.join(runs_root, f"{datetime.datetime.now():%Y%m%d-%H%M%S}")
    writer = SummaryWriter(run_dir)
    print(f"logging to {run_dir}")
    print(f"view with:  tensorboard --logdir {runs_root}")

    STEPS = 5000
    for step in range(STEPS):
        key, subkey = jax.random.split(key)
        model, metrics = train_step(model, subkey)
        for name, value in metrics.items():
            writer.add_scalar(name, float(value), step)
        if step % 25 == 0:
            print(f"step {step:4d}   return {float(metrics['return']):7.2f}   "
                  f"loss {float(metrics['loss']):8.3f}   "
                  f"grad_norm {float(metrics['grad_norm']):8.3f}")

    writer.close()

    # checkpoint the trained weights so debug_agent.py can inspect them later
    from debug_agent import save_params
    save_params(model)
    print(f"done. view with:  tensorboard --logdir {runs_root}")
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

