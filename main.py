"""BalatroNET — a JAX-native RL agent that plays and learns Balatro.

Entry point. Runs the test suites today; becomes the training loop later.
"""

from jax import random

from env import reset
from tests import run_tests

if __name__ == "__main__":
    run_tests()

    seed = 42
    key = random.key(seed)
    state = reset(key, 52)
