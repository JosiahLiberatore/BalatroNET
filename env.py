"""The Balatro environment — EnvState, reset, step.

Pure functional: the state pytree is the only memory. reset births the full
state; env_step takes a full state and returns the next full state. Capacity
is SHAPE (MAX_* constants), current count is DATA (scalars/masks in state).
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple
from scoring import scoreHand, joker_score
from jokers import MAX_JOKERS, J_EMPTY
from jokers import N_QUERY_KINDS, Q_MONEY_DIV_5, Q_EMPTY_JOKER_SLOTS
from jokers import rows_for, resolve, update
from jokers import T_ONCE_MAIN, T_ON_DISCARD
from arrays import first_n, compact, selected_cards

MAX_HAND_SIZE = 32
MAX_DECK_SIZE = 128
MAX_CONSUMABLES = 16
MAX_SHOP_SLOTS = 4
STARTING_MONEY = 4


class balatroEnvState(NamedTuple):
    # jokers
    jokers       : jax.Array   # (32,)  int32   -- joker id per slot, J_EMPTY = vacant
    joker_value  : jax.Array   # (32, ) float   -- the value of the joker in each jokers[i] slot
    joker_state  : jax.Array   # (32, ) int32   -- holds internal states (ex. Obelisk, Ancient Joker, Castle)
    joker_slots  : jax.Array   # scalar int32   -- slot count for non-negative jokers

    consumables  : jax.Array   # (16,)  int32   -- consumable id per slot, -1 = vacant
    consumable_slots :jax.Array # scalar int32  -- current max of non-negative consumables

    # hand
    hand         : jax.Array   # (32,)  int32   -- card id per lane, -1 = empty
    hand_size    : jax.Array   # scalar int32   -- number of lanes currently in play
    hand_valid   : jax.Array   # (32,)  bool    -- which lanes hold a live card

    # deck
    deck         : jax.Array   # (128,) int32   -- permutation of card ids
    deck_size    : jax.Array   # scalar int32   -- live cards; rest are dead slots
    deck_ptr     : jax.Array   # scalar int32   -- next card to deal

    # budget
    handcount    : jax.Array   # scalar int32   -- hands granted this round
    discardcount : jax.Array   # scalar int32   -- discards granted this round
    hands        : jax.Array   # scalar int32   -- hands remaining
    discards     : jax.Array   # scalar int32   -- discards remaining

    # round
    round        : jax.Array   # scalar int32
    ante         : jax.Array   # scalar int32
    chips_scored : jax.Array   # scalar int32   -- progress toward blind
    blind_target : jax.Array   # scalar int32   -- chips needed to clear
    key          : jax.Array   # PRNG key (new-style typed key)

    # shop
    shop_slots  : jax.Array    # (4, ) int32    -- holds ID of shop items
    game_stage  : jax.Array    # scalar int32   -- 0 = blind, 1 = shop
    money       : jax.Array    # scalar int32


class balatroAction(NamedTuple):
    verb   : jax.Array   # scalar int32   -- 0 = Play, 1 = Discard
    select : jax.Array   # (32,)  bool    -- which lanes to act on

@jit
def env_step(state: balatroEnvState, action: balatroAction) -> tuple[balatroEnvState, float, bool]:

    # get the hand type
    cardularcards, valid, select = selected_cards(state.hand, state.hand_valid,
                                                  action.select.astype(bool), 5)
    suit = cardularcards // 13
    rank = cardularcards % 13
    handtype, hand_mask, chips, mult = scoreHand(rank, suit, valid)

    # get joker lookups from IDs
    rows = rows_for(state.jokers)

    # boolean mask for active
    active = state.jokers != J_EMPTY

    # calculate queries
    ctx = build_ctx(state)

    # get the verb (a discard with no discards left becomes a play)
    verb = jnp.where(state.discards > 0, action.verb, 0)

    # update pass BEFORE resolve: Green Joker's +1 counts toward this hand
    trigger = jnp.where(verb == 0, T_ONCE_MAIN, T_ON_DISCARD)
    joker_value, joker_state = update(rows, active, state.joker_value,
                                      state.joker_state, trigger, ctx)

    v = resolve(rows, active, joker_value=joker_value, ctx=ctx)

    chips, mult = joker_score(chips, mult, handtype, rank, suit, hand_mask, rows, active, v)

    score = jnp.floor(chips * mult).astype(jnp.int32)
    score = score * (verb == 0)

    # update the state
    new_hand_valid = state.hand_valid & ~select
    new_state = state._replace(hands=state.hands-(verb == 0),
                            discards=state.discards-(verb == 1),
                            hand_valid=new_hand_valid,
                            chips_scored=state.chips_scored+score,
                            joker_value=joker_value,
                            joker_state=joker_state
                            )

    # deal new hand
    next_state = deal(new_state)

    done = (next_state.chips_scored >= state.blind_target) | (next_state.hands == 0)
    return next_state, score, done

def build_ctx(state) -> jax.Array:
    return (jnp.zeros(N_QUERY_KINDS, jnp.float32)
    .at[Q_MONEY_DIV_5].set(state.money // 5)
    )

@jit
def reset(key, deckSize, handSize):
    key, subkey = jax.random.split(key)

    deck = jax.random.permutation(subkey, MAX_DECK_SIZE)

    dead = deck >= deckSize
    deck = compact(deck, ~dead)          # live cards to the front, dead to the tail

    hand_valid = jnp.arange(MAX_HAND_SIZE) < handSize
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)

    return balatroEnvState(
        jokers=jnp.full(MAX_JOKERS, J_EMPTY, dtype=jnp.int32),
        joker_value=jnp.full(MAX_JOKERS, 0, dtype=float),
        joker_state=jnp.full(MAX_JOKERS, 0, dtype=jnp.int32),
        joker_slots=jnp.int32(5),
        consumables=jnp.full(MAX_CONSUMABLES, -1, dtype=jnp.int32),
        consumable_slots=jnp.int32(2),
        hand=hand,
        hand_size=jnp.int32(handSize),
        hand_valid=hand_valid,
        deck=deck,
        deck_size=jnp.int32(deckSize),
        deck_ptr=jnp.int32(handSize),
        hands=jnp.int32(4),
        discards=jnp.int32(3),
        handcount=jnp.int32(4),
        discardcount=jnp.int32(3),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(0),
        blind_target=jnp.int32(300),
        key=key,
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, dtype=jnp.int32),
        game_stage=jnp.int32(0),
        money=jnp.int32(STARTING_MONEY)
    )

@jit
def deal(state : balatroEnvState):
    deal_count = state.hand_size - state.hand_valid.sum()
    empty = ~state.hand_valid
    # each empty lane's position in the refill queue -> which deck slot feeds it
    src = state.deck_ptr + jnp.cumsum(empty) - 1
    fill = first_n(empty, deal_count)

    new_hand = jnp.where(fill, state.deck[src], state.hand)
    new_hand_valid = (jnp.arange(MAX_HAND_SIZE) < state.hand_size) | fill

    return state._replace(
        hand=new_hand,
        hand_valid=new_hand_valid,
        deck_ptr=state.deck_ptr+deal_count
    )

RANK_CHARS = "23456789TJQKA"   # rank = id % 13, so index 0 is the Two, 12 the Ace
SUIT_CHARS = "CDHS"            # display only — scoring never names suits, ids are the truth

def _card_name(cid):
    """'KH'-style name for a live card id; raw number for anything else."""
    if 0 <= cid < 52:
        return RANK_CHARS[cid % 13] + SUIT_CHARS[cid // 13]
    return str(cid)

def show(state: balatroEnvState, label=""):
    """Pretty-print the FULL state, one section per concern, plus invariant
    checks so a corrupted state announces itself. Plain Python — call it on
    concrete states only (outside jit, or on the result of a jitted
    function), never inside one."""
    n = int(state.hand_size)
    hand = [int(c) for c in state.hand]
    valid = [bool(v) for v in state.hand_valid]
    ptr, size = int(state.deck_ptr), int(state.deck_size)
    deck = [int(c) for c in state.deck]

    def row(name, cells):
        print(f"  {name:<6s}" + "".join(f"{str(c):>4s}" for c in cells))

    title = label or "state"
    print(f"=== {title} " + "=" * max(4, 56 - len(title)))

    # --- hand: lanes as columns, ids + names + mask stacked ---------------
    print(f"hand   ({sum(valid[:n])}/{n} lanes full)")
    row("lane", range(n))
    row("id", hand[:n])
    row("card", [_card_name(c) if v else "." for c, v in zip(hand[:n], valid[:n])])
    row("valid", [int(v) for v in valid[:n]])
    weird = [i for i in range(n, MAX_HAND_SIZE) if valid[i] or hand[i] != -1]
    if weird:
        print(f"  WARN  phantom lanes not empty: {weird}")
    else:
        print(f"  lanes {n}..{MAX_HAND_SIZE - 1}: empty (id -1, valid 0)")

    # --- deck: pointer, undealt window, dead zone -------------------------
    print(f"deck   ptr {ptr}/{size} ({size - ptr} undealt, "
          f"{MAX_DECK_SIZE - size} dead slots parked behind)")
    nxt = deck[ptr:min(ptr + 8, size)]
    row("next", nxt)
    row("card", [_card_name(c) for c in nxt])
    if ptr + 8 < size:
        print(f"  ... {size - ptr - 8} more undealt")

    # --- round counters + blind progress ----------------------------------
    print(f"round  hands {int(state.hands)}   discards {int(state.discards)}"
          f"   round {int(state.round)}   ante {int(state.ante)}"
          f"   money ${int(state.money)}")
    chips, target = int(state.chips_scored), int(state.blind_target)
    frac = 0.0 if target <= 0 else min(chips / target, 1.0)
    filled = int(round(frac * 20))
    print(f"blind  {chips}/{target} chips  [{'#' * filled}{'.' * (20 - filled)}]"
          f"  {int(frac * 100)}%")

    # --- rng ---------------------------------------------------------------
    print(f"key    {[int(x) for x in jax.random.key_data(state.key)]}"
          f"  (raw uint32 pair)")

    # --- invariants ---------------------------------------------------------
    dealt = set(deck[:ptr])
    checks = [
        ("deck is a permutation of 0..127",
         sorted(deck) == list(range(MAX_DECK_SIZE))),
        ("live ids fill deck[:deck_size]",
         all(c < size for c in deck[:size])),
        ("valid hand ids were all dealt (in deck[:ptr])",
         all(hand[i] in dealt for i in range(n) if valid[i])),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        for b in bad:
            print(f"  WARN  invariant broken: {b}")
    else:
        print("checks permutation OK | live zone OK | hand ids dealt OK")


def dbg(n=None, **rows):
    """Workbench printer: stack any raw arrays lane-aligned for eyeballing
    intermediates while building.  Bool rows print as 0 / 1 bitmaps so masks
    read at a glance; int rows as numbers; float rows to 2 decimals.

        dbg(n=10, valid=state.hand_valid, select=action.select, fill=fill)

    n limits how many lanes print (default: longest row). Concrete arrays
    only — inside @jit these are valueless tracers; un-jit while developing."""
    rows = {k: jnp.atleast_1d(v) for k, v in rows.items()}
    if n is None:
        n = max(int(v.shape[0]) for v in rows.values())
    cells = {}
    for name, v in rows.items():
        out = []
        for x in v[:n]:
            if v.dtype == bool:
                out.append("1" if bool(x) else "0")
            elif jnp.issubdtype(v.dtype, jnp.floating):
                out.append(f"{float(x):.2f}")
            else:
                out.append(str(int(x)))
        out += [""] * (n - len(out))          # short rows leave blank lanes
        cells[name] = out
    w = max(3, max(len(c) for row in cells.values() for c in row) + 1)
    label_w = max(6, max(len(k) for k in rows) + 2)
    print(" " * label_w + "".join(f"{i:>{w}d}" for i in range(n)))
    for name, row_cells in cells.items():
        print(f"{name:<{label_w}}" + "".join(f"{c:>{w}s}" for c in row_cells))



if __name__ == "__main__":
    # Dev scratch for the joker pass: one PLAY with a known loadout.
    # The real tests live in tests.py; this block uses env tools only.
    # NOTE: comment out the @jit on env_step while developing — dbg and
    # print then show real arrays instead of tracers.
    from jokers import J_JOKER, J_GREEDY, J_BOOTSTRAP

    ids = jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32).at[:3].set(
        jnp.array([J_JOKER, J_GREEDY, J_BOOTSTRAP], jnp.int32))
    seed = jnp.where(ids != J_EMPTY, rows_for(ids).VALUE, 0.0)

    pre = reset(jax.random.key(0), 52, 8)._replace(
        jokers=ids, joker_value=seed, money=jnp.int32(23))
    show(pre, "before")

    # PLAY the first five lanes: 6S JD QS 2C 8S -> high card, only QS scores.
    # Base: chips 5 + 10 = 15, mult 1 -> score 15.
    # With the joker pass: Joker +4 mult; Greedy 0 (JD is played but does not
    # score); Bootstraps +2 * (23 // 5) = +8 -> mult 13 -> score 15 * 13 = 195.
    action = balatroAction(verb=jnp.int32(0),
                           select=jnp.arange(MAX_HAND_SIZE) < 5)
    nxt, reward, done = env_step(pre, action)
    show(nxt, "after one PLAY")
    print(f"reward {float(reward)}   done {bool(done)}")
    print("want: reward 15 before the pass, 195 after it lands")


