# BalatroNET

BalatroNET has three parts:

- A JAX native environment of the full game of Balatro.
- A reinforcement learning agent that learns to play it.
- A GUI to play the environment by hand and evaluate the agent.

The environment is one pure function, `env_step(state, action)`. It works with `jit` and `vmap`, so
1,024 runs go in parallel on one GPU. The rules follow the Lua code of Balatro 1.0.1o.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install jax numpy pygame tensorboardX tensorboard
```

JAX uses only the CPU on Windows. Training on a GPU needs WSL2 or Linux.

## Run

```bash
python gui.py                    # play by hand, or look at the model (--seed N, --params FILE)
python train.py fresh            # start a training run (Red Deck, White Stake)
python train.py                  # resume from runs/ckpt_run.pkl
python train.py --deck plasma --stake gold --steps 100000
python debug_agent.py            # one run step by step, then the numbers of 512 runs
python tests.py                  # all test suites
tensorboard --logdir runs        # the training curves
```

`train.py` writes a checkpoint to `runs/` every 500 updates. `--grad-chunks 16` splits the gradient
pass so that it fits in 8 GB of GPU memory.

## Initial release

### The environment

A run is the full game: 8 antes, each with a Small Blind, a Big Blind and a Boss Blind, with a shop after each round

| Part | Count |
|---|---|
| Jokers | 150 |
| Tarot, Planet and Spectral cards | 52 |
| Booster packs | 15 |
| Vouchers | 32 |
| Tags | 24 |
| Boss Blinds | 28 |
| Decks | 15 (Red to Erratic) |
| Stakes | 8 (White to Gold) |

Also in the environment:

- Card enhancements, editions and seals.
- The joker stickers of the stakes: Eternal, Perishable and Rental.
- The blind select (play, skip for a tag, reroll the boss).
- The shop: buy, sell, reroll, open a pack, redeem a voucher.
- The cash-out screen with interest.

Not in the environment:

- Endless mode.
- Challenge runs.
- Game seeds. The random numbers are from JAX, so a seed of the game does not give the same run.

### Differences from the game

None of these changes a normal run. `NOTES.md` has the full list and the design decisions.

Size limits:

- The hand stops at 32 cards and the deck at 128.
- Joker slots stop at 16. A Negative joker past 16 gives no slot.
- The environment holds 16 consumables, but the agent sees and uses only 8.
- A player holds at most 8 tags, and a shop shows at most 4 vouchers.
- A hand's score stops at about 2.1 billion (int32). The largest target is 2,400,000 so no result changes.

Game bugs that the environment does not copy:

- Ankh or Invisible Joker copies a perished joker: the game still gives the copy's hand size or
  discards. The environment gives nothing.
- Crimson Heart with no joker to pick: the game has a Lua error. The environment picks none.

Rare rule differences:

- The Pillar: when a joker comes or goes in the middle of its blind, the game checks all cards
  again and debuffs a DNA copy made in that blind. The environment does not.
- Orbital Tag: when an ante number comes back (Hieroglyph, Petroglyph), the game gives the same
  hand as before. The environment picks a new one.
- The Ox: a tie between hand types can go the other way after a save and continue in the game.
- Retriggers stop at 12 on a scored card and 6 on a held card. Only Showman duplicates or long
  copy chains get there.
- A pack hand goes back to the deck in a different order. This only becomes an issue if the agent can cycle
  through their whole deck in a shop before the next blind

What the agent sees:

- Amber Acorn: a face-down joker shows its face after any message. In the game, two jokers with
  the same message can stay unclear.
- Two identical face-down cards: the observation does not give their order.

### The tests

- `ref_balatro.py` is a second implementation of the scoring rules in plain Python. It does not
  read the environment's tables. `tests_jokers.py` compares the two on each joker.
- `scenarios.py` has a scenario for each mechanic. Each scenario gives the rule, the steps and the
  expected result. The suite makes 46,227 checks.
- `tests_run.py` tests the run loop, the shop and the agent's interface.

`python tests.py` runs all of them. It takes about 10 minutes on a CPU.

### The GUI

`gui.py` is a pygame front end. Each game action goes through `env_step`, the same function that
the agent trains on.

- Play a run by hand, on any stake and any seed.
- Show the model's odds for each action, take the model's pick, or let the model play.
- Step through each scenario test and see the board after each step.
- Add money, jokers and consumables to set up a situation.

### The agent

- **Observation:** 4,137 numbers. The agent gets only what a player can know. A face-down card
  stays hidden.
- **Actions:** one masked choice out of 70 actions. A card picker selects the cards for a play, a
  discard or a consumable, one card at a time. Pointer heads select the source and the destination
  of a move.
- **Model:** a transformer encoder over tokens (hand cards, jokers, shop items, pack cards,
  consumables, vouchers), 2 layers at width 64, then a policy and a value head. It has 659,144
  parameters and no memory.
- **Training:** A2C with GAE. 1,024 environments step 64 moves for each update, which is 65,536
  moves.
- **Reward:** the dollars of the cash-out after each cleared blind. There is no reward for a win.

### Results

The run uses the Red Deck on White Stake. The hardware is one RTX 2070 SUPER (8 GB), at about 16,000 moves per
second.

| Stake | Updates | Moves | Win rate | Mean ante | $ per run |
|---|---|---|---|---|---|
| White | 100,000 | 6.6 billion | 48.5 % | 6.82 | 342.6 |

The numbers are from the training log at the last update. The win rate was flat for the last
6,000 updates.

What the agent does:

- It clears 97.6 % of its blinds.
- In ante 1 it takes the best possible play in about half of its plays.

The trained weights are not in the repo.

### Next

The goal is an agent that wins about 50% of its runs on Gold Stake. The work in progress:

- A larger model (4 layers at width 128). A test on White Stake is running.
- Maybe look into splitting the reward between passing a blind and money? So the agent is focused on clears instead of maximizing cash

## History

- **v0.1:** the first learning loop. One blind, no jokers, a REINFORCE agent.
- **v1.0:** the full game in the environment, and the A2C agent on full runs.
- **v1.1:** review fixes, the Invisible Joker room rule, the GUI's view of face-down cards.
