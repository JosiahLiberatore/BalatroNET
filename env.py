"""The Balatro environment — EnvState, reset, step.

Pure functional: the state pytree is the only memory. reset births the full
state; env_step takes a full state and returns the next full state. Capacity
is SHAPE (MAX_* constants), current count is DATA (scalars/masks in state).
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple
from scoring import scoreHand

MAX_HAND_SIZE = 32
MAX_DECK_SIZE = 128


class balatroEnvState(NamedTuple):
    # consumnables = jnp.ndarray
    # jokers = jnp.ndarray
    hand: jnp.ndarray # (32) int
    hand_size : jnp.int32
    hand_valid : jnp.array # (32) bool
    deck : jnp.ndarray # (128) int
    deck_size : jnp.int32
    deck_ptr : jnp.int32
    hands: jnp.int32
    discards: jnp.int32
    round: jnp.int32
    ante: jnp.int32
    chips_scored : jnp.int32
    blind_target : jnp.int32
    key : jax.random.key

class balatroAction(NamedTuple):
    verb : jnp.int32 # 0 = Play, 1 = Discard
    select : jnp.ndarray # (32, ), bool


#@jit
def env_step(state: balatroEnvState, action: balatroAction) -> tuple[balatroEnvState, float, bool]:

    # what was the action?
    # play
    # get scoreHand input (suit[5], rank[5], valid[5])

    hand = state.hand
    select = action.select

    doublebackwardsantiinversion = ~select

    order = jnp.argsort(doublebackwardsantiinversion)

    cumsum = jnp.cumsum(select)

    cardularcards = hand[order][:5]

    valid = select[order][:5]

    jnp.sum(select)


    suit = (cardularcards // 13)[:5]
    rank = (cardularcards % 13)[:5]

    type, fjd, score = scoreHand(suit, rank, valid)
    
    # update the state






    # dbg(hand=state.hand,
    #     valid=state.hand_valid,
    #     select=action.select,
    #     cumsum=cumsum,
    #     cardularcards=cardularcards,
    #     order=order,
    #     playvalid=valid
    #     )


    # show(state)

    next_state = deal(state)

    done = 0
    return next_state, score, done

def _hand_lanes(vals):
    """32-wide hand array from the first few lanes; -1 = empty lane."""
    a = jnp.full(MAX_HAND_SIZE, -1, dtype=jnp.int32)
    return a.at[:len(vals)].set(jnp.array(vals, dtype=jnp.int32))
def _valid_lanes(bits):
    """32-wide bool mask from the first few lanes; rest False."""
    m = jnp.zeros(MAX_HAND_SIZE, dtype=bool)
    return m.at[:len(bits)].set(jnp.array(bits, dtype=bool))
def run_deal_test():
    # Scenario: deck[i] = i + 13 (mod 52) — deliberately NOT the identity, so
    # confusing a deck POSITION (src) with the CARD at that position shows up
    # as visibly wrong values. Opening 8 cards were deck[0..7] = 13..20 and
    # deck_ptr sits at 8. The player then discarded lanes 2, 5, 6 — three
    # holes. deal must fill them from the top of the deck IN ORDER
    # (lane 2 <- deck[8]=21, lane 5 <- deck[9]=22, lane 6 <- deck[10]=23),
    # advance deck_ptr to 11, and touch NOTHING else.
    key = jax.random.key(0)
    pre = balatroEnvState(
        hand=_hand_lanes([13, 14, -1, 16, 17, -1, -1, 20]),
        hand_size=jnp.int32(8),
        hand_valid=_valid_lanes([1, 1, 0, 1, 1, 0, 0, 1]),
        deck=jnp.concatenate([(jnp.arange(52, dtype=jnp.int32) + 13) % 52,
                              jnp.arange(52, MAX_DECK_SIZE, dtype=jnp.int32)]),
        deck_size=jnp.int32(52),
        deck_ptr=jnp.int32(8),
        hands=jnp.int32(3),
        discards=jnp.int32(2),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(60),
        blind_target=jnp.int32(300),
        key=key,
    )
    expected = pre._replace(
        hand=_hand_lanes([13, 14, 21, 16, 17, 22, 23, 20]),
        hand_valid=_valid_lanes([1, 1, 1, 1, 1, 1, 1, 1]),
        deck_ptr=jnp.int32(11),
    )

    got = deal(pre)
def run_env_step_test():
    # Scenario: mid-round, full 8-card hand, agent PLAYs a scattered two pair.
    # Conventions this test pins down: verb 0 = PLAY (1 = DISCARD);
    # rank = id % 13, suit = id // 13; reward = the hand's score (dense, for now);
    # done = (chips_scored >= blind_target) | (hands == 0).
    #
    #   lane:   0    1    2    3    4    5    6    7
    #   id:     2   24   33   11    5   46   44   27
    #   rank:   2   11    7   11    5    7    5    1     (11 = K, 5 = 7, 7 = 9)
    #
    # select lanes 1, 3, 4, 6 -> K K 7 7 = two pair. Scattered on purpose
    # (exercises stable-argsort compaction), and lanes 2/5 hold an UNselected
    # pair of 9s — if step scores the whole hand instead of the selection,
    # the result comes out wrong.
    #
    # Score: ranks [11, 11, 5, 5] is the same two pair as the scoreHand test
    # row -> (20 + 10+10+7+7) * 2 = 108. chips 60 -> 168 (< 300), hands 3 -> 2,
    # so done stays False.
    #
    # Deck: positions 0-7 hold the hand's ids (they WERE dealt — ptr is 8),
    # positions 8+ hold the remaining 44 ids ascending: deck[8..11] = 0, 1, 3, 4.
    # NOT identity (deck[8] != 8), so a src-vs-deck[src] mixup refills with
    # 8, 9, 10, 11 instead and fails visibly. Refill: lane1<-0, lane3<-1,
    # lane4<-3, lane6<-4, ptr -> 12.
    hand_ids = [2, 24, 33, 11, 5, 46, 44, 27]
    rest = [i for i in range(52) if i not in hand_ids]
    key = jax.random.key(0)
    pre = balatroEnvState(
        hand=_hand_lanes(hand_ids),
        hand_size=jnp.int32(8),
        hand_valid=_valid_lanes([1, 1, 1, 1, 1, 1, 1, 1]),
        deck=jnp.array(hand_ids + rest + list(range(52, MAX_DECK_SIZE)),
                       dtype=jnp.int32),
        deck_size=jnp.int32(52),
        deck_ptr=jnp.int32(8),
        hands=jnp.int32(3),
        discards=jnp.int32(2),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(60),
        blind_target=jnp.int32(300),
        key=key,
    )
    action = balatroAction(
        verb=jnp.int32(0),  # PLAY
        select=_valid_lanes([0, 1, 0, 1, 1, 0, 1, 0]),
    )
    expected = pre._replace(
        hand=_hand_lanes([2, 0, 33, 1, 3, 46, 4, 27]),
        hand_valid=_valid_lanes([1, 1, 1, 1, 1, 1, 1, 1]),
        deck_ptr=jnp.int32(12),
        hands=jnp.int32(2),
        chips_scored=jnp.int32(168),
    )
    expected_reward = 108
    expected_done = False

    try:
        got, reward, done = env_step(pre, action)
    except Exception as e:
        print(f"env_step: NOT RUNNABLE yet ({type(e).__name__}: {e})")
        return

    failures = []
    for field in balatroEnvState._fields:
        g, e = getattr(got, field), getattr(expected, field)
        if field == "key":
            same = bool(jnp.array_equal(jax.random.key_data(g), jax.random.key_data(e)))
        else:
            same = bool(jnp.array_equal(g, e))
        if not same:
            failures.append(f"FAIL  env_step: field '{field}'\n      got:  {g}\n      want: {e}")
    if int(reward) != expected_reward:
        failures.append(f"FAIL  env_step: reward\n      got:  {int(reward)}\n      want: {expected_reward}")
    if bool(done) != expected_done:
        failures.append(f"FAIL  env_step: done\n      got:  {bool(done)}\n      want: {expected_done}")
    if failures:
        print("\n".join(failures))
        print(f"env_step: {len(failures)} check(s) wrong")
    else:
        print("env_step: two-pair play test passed")

@jit
def reset(key, deckSize, handSize):
    key, subkey = jax.random.split(key)

    deck = jax.random.permutation(subkey, MAX_DECK_SIZE)

    dead = deck >= deckSize
    deck = deck[jnp.argsort(dead)]

    hand_valid = jnp.arange(MAX_HAND_SIZE) < handSize
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)

    return balatroEnvState(
        hand=hand,
        hand_size=jnp.int32(handSize),
        hand_valid=hand_valid,
        deck=deck,
        deck_size=jnp.int32(deckSize),
        deck_ptr=jnp.int32(handSize),
        hands=jnp.int32(4),
        discards=jnp.int32(3),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        chips_scored=jnp.int32(0),
        blind_target=jnp.int32(300),
        key=key,
    )

@jit
def deal(state : balatroEnvState):
    deal_count = state.hand_size - state.hand_valid.sum()
    empty = ~state.hand_valid
    queue = jnp.cumsum(empty)
    src = state.deck_ptr + queue - 1
    fill = empty & (queue - 1 < deal_count)

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
          f"   round {int(state.round)}   ante {int(state.ante)}")
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
    # dev scratch: `python env.py` runs deal on the test scenario from tests.py
    # (deck[i] = i+13 mod 52 — NOT identity, so src vs deck[src] mixups are
    # visible; opening 8 dealt = cards 13..20, lanes 2/5/6 discarded).
    # NOTE: comment out the @jit on deal while developing — plain print() then
    # shows real arrays instead of tracers.

    run_env_step_test()

    # hand = jnp.full(MAX_HAND_SIZE, -1, dtype=jnp.int32).at[:8].set(
    #     jnp.array([13, 14, -1, 16, 17, -1, -1, 20], dtype=jnp.int32))
    # hand_valid = jnp.zeros(MAX_HAND_SIZE, dtype=bool).at[:8].set(
    #     jnp.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=bool))
    # pre = balatroEnvState(
    #     hand=hand,
    #     hand_size=jnp.int32(8),
    #     hand_valid=hand_valid,
    #     deck=jnp.concatenate([(jnp.arange(52, dtype=jnp.int32) + 13) % 52,
    #                           jnp.arange(52, MAX_DECK_SIZE, dtype=jnp.int32)]),
    #     deck_size=jnp.int32(52),
    #     deck_ptr=jnp.int32(8),
    #     hands=jnp.int32(3),
    #     discards=jnp.int32(2),
    #     round=jnp.int32(1),
    #     ante=jnp.int32(1),
    #     chips_scored=jnp.int32(60),
    #     blind_target=jnp.int32(300),
    #     key=jax.random.key(0),
    # )

    # got = deal(pre)
    # if got is None:
    #     print("--- deal returned None (no return yet) ---")
    # else:
    #     show(got, "deal returned")
    # print("want hand[:8]:   [13 14 21 16 17 22 23 20], valid all 1, ptr 11")


