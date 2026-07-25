# BalatroNET

A reinforcement learning agent built natively in JAX to play and learn Balatro.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install jax tensorboardX tensorboard
```

JAX runs CPU-only on Windows; GPU needs WSL2 or a cloud runtime.

## Run

```bash
python train.py          # train (resumes from runs/ckpt_latest.pkl)
python train.py fresh     # ignore any checkpoint, start over
python debug_agent.py     # inspect a trained policy (episode + 256-ep stats)
tensorboard --logdir runs # view training curves
python main.py            # run the env/scoring test suites
```

## v0.1 Results

First working end-to-end RL loop: a JAX-native Balatro environment (one blind,
plays + discards, no jokers/antes) with a REINFORCE agent (Adam, entropy bonus,
advantage normalization). Finish line was "the reward curve goes up" — it does,
well past the greedy no-discard local optimum the early runs plateaued at.

Trained policy, measured over 256 eager episodes (`python debug_agent.py`,
blind target 300):

| Metric | Value |
|---|---|
| Mean return | **339.1** chips |
| Blind clear rate | **61.7%** |
| Discard rate | **39.5%** |
| Mean cards selected | 5.33 |
| Mean score / play | 94.3 chips |
| Mean P(discard) | 0.40 |
| Mean policy entropy | 1.20 |

Mean return clears the 300 target, but the 61.7% clear rate reflects a
right-skewed, high-variance outcome distribution (irreducible card-draw
variance): the agent learned genuine discard-and-fish play — a healthy ~40%
discard rate that never collapsed to the greedy all-plays optimum — rather than
just always grabbing the highest cards. Beating the clear rate via coordinated
card selection (a card-token attention policy vs. 32 independent Bernoullis) is
the v0.2 target.
