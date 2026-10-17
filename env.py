"""The Balatro environment — EnvState, reset, step.

Pure functional: the state pytree is the only memory. reset births the full
state; env_step takes a full state and returns the next full state. Capacity
is SHAPE (MAX_* constants), current count is DATA (scalars/masks in state).

A run is blind -> shop -> blind -> shop ... across 8 antes:
    STAGE_BLIND  PLAY / DISCARD / SELL     clear -> cashout -> STAGE_SHOP
                                           out of hands -> STAGE_LOST
    STAGE_SHOP   BUY / SELL / REROLL       NEXT -> the next blind
    STAGE_LOST / STAGE_WON                 every verb is a no-op
Boss blinds have no boss effects yet, and there is no blind-select screen:
NEXT starts the next blind at once.

The return contract is the v0.1 one, so train.py does not change:
    reward = the chips the step scored
    done   = this step ended the blind (cleared or failed)
The run itself ends when game_stage becomes STAGE_LOST or STAGE_WON.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple
from scoring import scoreHand, joker_score
from jokers import MAX_JOKERS, J_EMPTY, N_JOKERS, JOKERS, JOKER_COST, JOKER_RARITY, JOKER_ID
from jokers import N_QUERY_KINDS, Q_MONEY_DIV_5, Q_EMPTY_JOKER_SLOTS
from jokers import COMMON, UNCOMMON, RARE, LEGENDARY
from jokers import rows_for, resolve, update
from jokers import T_ONCE_MAIN, T_ON_DISCARD, T_END_OF_ROUND, T_ON_BLIND_SELECT, T_ON_SOLD
from arrays import first_n, compact, selected_cards

MAX_HAND_SIZE = 32
MAX_DECK_SIZE = 128
MAX_CONSUMABLES = 16
MAX_SHOP_SLOTS = 4
STARTING_MONEY = 4

# --- verbs: one atomic action per step -------------------------------------
V_PLAY    = 0   # select = the cards (first 5 valid lanes)
V_DISCARD = 1   # select = the cards; with 0 discards left it becomes a PLAY
V_BUY     = 2   # target = shop slot
V_SELL    = 3   # target = joker slot; legal in the blind AND the shop
V_REROLL  = 4
V_NEXT    = 5   # leave the shop, start the next blind
N_VERBS   = 6
V_NOOP    = 6   # internal: the repair target of every illegal action
VERB_NAMES = ["PLAY", "DISCARD", "BUY", "SELL", "REROLL", "NEXT"]

# --- stages ------------------------------------------------------------------
STAGE_BLIND = 0
STAGE_SHOP  = 1
STAGE_LOST  = 2
STAGE_WON   = 3
STAGE_NAMES = ["BLIND", "SHOP", "LOST", "WON"]

# LEGAL[stage, verb]: which verbs each stage accepts. Everything else is a no-op.
LEGAL = jnp.array([
    #  PLAY   DISC   BUY    SELL   REROLL NEXT
    [True,  True,  False, True,  False, False],   # BLIND
    [False, False, True,  True,  True,  True ],   # SHOP
    [False, False, False, False, False, False],   # LOST
    [False, False, False, False, False, False],   # WON
])

# --- blinds (white stake) ----------------------------------------------------
BLIND_SMALL, BLIND_BIG, BLIND_BOSS = 0, 1, 2
BLIND_NAMES  = ["Small Blind", "Big Blind", "Boss Blind"]
ANTE_BASE    = jnp.array([100, 300, 800, 2000, 5000, 11000, 20000, 35000, 50000], jnp.int32)
BLIND_MULT   = jnp.array([1.0, 1.5, 2.0], jnp.float32)
BLIND_REWARD = jnp.array([3, 4, 5], jnp.int32)
WIN_ANTE     = 8     # clearing the ante-8 boss wins the run

# --- economy -----------------------------------------------------------------
INTEREST_PER  = 5    # $1 per $5 held ...
INTEREST_CAP  = 5    # ... at most $5
REROLL_BASE   = 5    # +$1 per reroll, back to base on each shop visit
SHOP_UNLOCKED = 2    # joker slots open in the shop at run start (vouchers raise it)

# Shop odds: pick a rarity (70 / 25 / 5 %), then a joker uniform inside it.
# As one weight per joker: rarity weight / jokers of that rarity in the table.
# Legendaries never appear in the shop.
_RARITY_WEIGHT = {COMMON: 70.0, UNCOMMON: 25.0, RARE: 5.0, LEGENDARY: 0.0}
_rarity = [int(r) for r in JOKER_RARITY]
SHOP_WEIGHT = jnp.array([_RARITY_WEIGHT[r] / _rarity.count(r) for r in _rarity], jnp.float32)
J_FALLBACK = JOKER_ID["Joker"]   # what the shop offers when the pool is empty


class balatroEnvState(NamedTuple):
    # jokers (compact: held jokers fill slots 0..n-1, in order)
    jokers       : jax.Array   # (32,)  int32   -- joker id per slot, J_EMPTY = vacant
    joker_value  : jax.Array   # (32, ) float   -- the value of the joker in each jokers[i] slot
    joker_state  : jax.Array   # (32, ) int32   -- holds internal states (ex. Obelisk, Ancient Joker, Castle)
    joker_sell_value : jax.Array # (32,) int32  -- $ a SELL pays; mutable (Egg, Gift Card)
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
    round        : jax.Array   # scalar int32   -- blinds started this run, from 1
    ante         : jax.Array   # scalar int32
    blind_index  : jax.Array   # scalar int32   -- BLIND_SMALL / BIG / BOSS
    chips_scored : jax.Array   # scalar int32   -- progress toward blind
    blind_target : jax.Array   # scalar int32   -- chips needed to clear
    key          : jax.Array   # PRNG key (new-style typed key)

    # shop
    shop_slots  : jax.Array    # (4, ) int32    -- joker id per shop slot, -1 = empty or locked
    shop_size   : jax.Array    # scalar int32   -- unlocked shop slots (lanes >= this stay -1)
    reroll_cost : jax.Array    # scalar int32
    game_stage  : jax.Array    # scalar int32   -- STAGE_*
    money       : jax.Array    # scalar int32
    last_cashout: jax.Array    # scalar int32   -- $ paid at the last blind clear (display)


class balatroAction(NamedTuple):
    verb   : jax.Array                    # scalar int32 -- V_*
    select : jax.Array                    # (32,)  bool  -- which lanes to act on
    target : jax.Array = jnp.int32(0)     # scalar int32 -- shop slot (BUY) / joker slot (SELL)


class PlayResult(NamedTuple):
    """Everything one scored hand produced. The GUI and the tests read this."""
    handtype   : jax.Array   # scalar int32  -- index into HAND_NAMES
    sel        : jax.Array   # (32,)  bool   -- the lanes that were actually played
    cards      : jax.Array   # (5,)   int32  -- the played cards, lane order
    valid      : jax.Array   # (5,)   bool
    mask       : jax.Array   # (5,)   bool   -- which played cards scored
    base_chips : jax.Array   # scalar        -- before the joker pass
    base_mult  : jax.Array
    chips      : jax.Array   # scalar float32 -- after the joker pass
    mult       : jax.Array
    score      : jax.Array   # scalar int32  -- floor(chips * mult)


def tree_where(c, a, b):
    """Select a whole pytree: a where c, else b. The branch-free 'if'."""
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def blind_target(ante, blind_index):
    """Chips needed: ante base x (1 / 1.5 / 2) for small / big / boss."""
    base = ANTE_BASE[jnp.clip(ante, 0, ANTE_BASE.shape[0] - 1)]
    return jnp.floor(base * BLIND_MULT[blind_index]).astype(jnp.int32)


def build_ctx(state) -> jax.Array:
    held = (state.jokers != J_EMPTY).sum()
    return (jnp.zeros(N_QUERY_KINDS, jnp.float32)
    .at[Q_MONEY_DIV_5].set(state.money // 5)
    .at[Q_EMPTY_JOKER_SLOTS].set(jnp.maximum(state.joker_slots - held, 0))
    )


def _jokers(state):
    """(rows, active, ctx) — the three inputs every joker pass needs."""
    return rows_for(state.jokers), state.jokers != J_EMPTY, build_ctx(state)


def score_play(state: balatroEnvState, select, joker_value) -> PlayResult:
    """Score the selected cards with a given joker_value. Pure, no state change."""
    cards, valid, sel = selected_cards(state.hand, state.hand_valid, select.astype(bool), 5)
    suit = cards // 13
    rank = cards % 13
    handtype, mask, base_chips, base_mult = scoreHand(rank, suit, valid)
    rows, active, ctx = _jokers(state)
    v = resolve(rows, active, joker_value=joker_value, ctx=ctx)
    chips, mult = joker_score(base_chips, base_mult, handtype, rank, suit, mask, rows, active, v)
    score = jnp.floor(chips * mult).astype(jnp.int32)
    return PlayResult(handtype, sel, cards, valid, mask, base_chips, base_mult, chips, mult, score)


@jit
def explain_play(state: balatroEnvState, select) -> PlayResult:
    """What PLAY would score, exactly as env_step scores it (update pass first)."""
    rows, active, ctx = _jokers(state)
    joker_value, _ = update(rows, active, state.joker_value, state.joker_state,
                            jnp.int32(T_ONCE_MAIN), ctx)
    return score_play(state, select, joker_value)


@jit
def preview(state: balatroEnvState, select) -> PlayResult:
    """The hand the selection makes, before any update — the in-game preview."""
    return score_play(state, select, state.joker_value)


# ---------------------------------------------------------------------------
# STEP — repair the action, then dispatch on the verb
# ---------------------------------------------------------------------------
@jit
def env_step(state: balatroEnvState, action: balatroAction) -> tuple[balatroEnvState, jax.Array, jax.Array]:
    """One atomic action. -> (next_state, reward, done). Never throws.

    Repair rules (an illegal action is a no-op, never an error):
      - a verb the current stage does not accept    -> no-op
      - DISCARD with 0 discards left                -> PLAY
      - BUY / SELL with a bad target, too little $, or no free slot -> no-op
      - more than 5 cards selected                  -> the first 5 valid lanes
    """
    verb = action.verb.astype(jnp.int32)
    verb = jnp.where((verb == V_DISCARD) & (state.discards <= 0), V_PLAY, verb)
    in_range = (verb >= 0) & (verb < N_VERBS)
    stage = jnp.clip(state.game_stage, 0, LEGAL.shape[0] - 1)
    legal = in_range & LEGAL[stage, jnp.clip(verb, 0, N_VERBS - 1)]
    branch = jnp.where(legal, verb, V_NOOP)
    action = action._replace(verb=verb, target=jnp.asarray(action.target, jnp.int32))
    return jax.lax.switch(branch, _HANDLERS, state, action)


def _no_change(state):
    return state, jnp.int32(0), jnp.bool_(False)


def _blind_action(state: balatroEnvState, action: balatroAction):
    """PLAY or DISCARD (the verb is already repaired)."""
    is_play = action.verb == V_PLAY

    # update pass BEFORE scoring: Green Joker's +1 counts toward this hand
    rows, active, ctx = _jokers(state)
    trigger = jnp.where(is_play, T_ONCE_MAIN, T_ON_DISCARD)
    joker_value, joker_state = update(rows, active, state.joker_value,
                                      state.joker_state, trigger, ctx)

    res = score_play(state, action.select, joker_value)
    score = res.score * is_play

    new_state = state._replace(hands=state.hands - is_play,
                               discards=state.discards - (~is_play),
                               hand_valid=state.hand_valid & ~res.sel,
                               chips_scored=state.chips_scored + score,
                               joker_value=joker_value,
                               joker_state=joker_state)
    next_state = deal(new_state)

    cleared = next_state.chips_scored >= state.blind_target
    failed = ~cleared & (next_state.hands <= 0)
    next_state = tree_where(cleared, _clear_blind(next_state), next_state)
    next_state = next_state._replace(
        game_stage=jnp.where(failed, STAGE_LOST, next_state.game_stage))
    return next_state, score, cleared | failed


def _clear_blind(state: balatroEnvState) -> balatroEnvState:
    """End of round: END_OF_ROUND update, cashout, then open the shop (or win)."""
    rows, active, ctx = _jokers(state)
    joker_value, joker_state = update(rows, active, state.joker_value, state.joker_state,
                                      jnp.int32(T_END_OF_ROUND), ctx)

    # interest is paid on the money held BEFORE this cashout
    interest = jnp.minimum(state.money // INTEREST_PER, INTEREST_CAP)
    cashout = BLIND_REWARD[state.blind_index] + state.hands + interest

    won = (state.blind_index == BLIND_BOSS) & (state.ante >= WIN_ANTE)
    key, sub = jax.random.split(state.key)
    return state._replace(
        joker_value=joker_value,
        joker_state=joker_state,
        money=state.money + cashout,
        last_cashout=cashout,
        game_stage=jnp.where(won, STAGE_WON, STAGE_SHOP),
        shop_slots=roll_shop(sub, state.jokers, state.shop_size),
        reroll_cost=jnp.int32(REROLL_BASE),
        key=key,
    )


def acquire(state: balatroEnvState, jid) -> tuple[balatroEnvState, jax.Array]:
    """Put joker jid in the first free slot. -> (state, ok). No money changes.

    The one place a joker enters the loadout: BUY uses it, and so will packs,
    Judgement and the GUI's dev button. ok is False (state unchanged) when the
    joker slots are full or jid is not a joker.
    """
    safe = jnp.clip(jid, 0, N_JOKERS - 1)
    held = (state.jokers != J_EMPTY).sum()
    ok = (jid >= 0) & (jid < N_JOKERS) & (held < state.joker_slots)

    dest = held   # jokers are compact, so the first free slot is the count
    got = state._replace(
        jokers=state.jokers.at[dest].set(jid),
        joker_value=state.joker_value.at[dest].set(JOKERS.VALUE[safe]),
        joker_state=state.joker_state.at[dest].set(0),
        joker_sell_value=state.joker_sell_value.at[dest].set(jnp.maximum(1, JOKER_COST[safe] // 2)),
    )
    return tree_where(ok, got, state), ok


def _buy(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_SHOP_SLOTS - 1)
    jid = state.shop_slots[slot]
    cost = JOKER_COST[jnp.clip(jid, 0, N_JOKERS - 1)]

    got, has_room = acquire(state, jid)
    ok = (t >= 0) & (t < state.shop_size) & (state.money >= cost) & has_room
    bought = got._replace(money=state.money - cost,
                          shop_slots=state.shop_slots.at[slot].set(J_EMPTY))
    return tree_where(ok, bought, state), jnp.int32(0), jnp.bool_(False)


def _sell(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_JOKERS - 1)
    ok = (t >= 0) & (t < MAX_JOKERS) & (state.jokers[slot] != J_EMPTY)

    # T_ON_SOLD fires for the joker that is sold (Campfire, Invisible Joker later)
    gone = jnp.arange(MAX_JOKERS) == slot
    jokers = jnp.where(gone, J_EMPTY, state.jokers)
    keep = jokers != J_EMPTY      # compact: the jokers to the right shift left
    sold = state._replace(
        jokers=compact(jokers, keep),
        joker_value=compact(jnp.where(gone, 0.0, state.joker_value), keep),
        joker_state=compact(jnp.where(gone, 0, state.joker_state), keep),
        joker_sell_value=compact(jnp.where(gone, 0, state.joker_sell_value), keep),
        money=state.money + state.joker_sell_value[slot],
    )
    return tree_where(ok, sold, state), jnp.int32(0), jnp.bool_(False)


def _reroll(state: balatroEnvState, action: balatroAction):
    ok = state.money >= state.reroll_cost
    key, sub = jax.random.split(state.key)
    rolled = state._replace(
        shop_slots=roll_shop(sub, state.jokers, state.shop_size),
        money=state.money - state.reroll_cost,
        reroll_cost=state.reroll_cost + 1,
        key=key,
    )
    return tree_where(ok, rolled, state), jnp.int32(0), jnp.bool_(False)


def _next(state: balatroEnvState, action: balatroAction):
    """Leave the shop: advance the blind (and the ante after a boss), fresh deck."""
    was_boss = state.blind_index == BLIND_BOSS
    blind_index = jnp.where(was_boss, BLIND_SMALL, state.blind_index + 1)
    ante = state.ante + was_boss

    rows, active, ctx = _jokers(state)
    joker_value, joker_state = update(rows, active, state.joker_value, state.joker_state,
                                      jnp.int32(T_ON_BLIND_SELECT), ctx)

    key, sub = jax.random.split(state.key)
    deck, hand, hand_valid = _fresh_deal(sub, state.deck_size, state.hand_size)
    nxt = state._replace(
        joker_value=joker_value,
        joker_state=joker_state,
        hand=hand,
        hand_valid=hand_valid,
        deck=deck,
        deck_ptr=state.hand_size,
        hands=state.handcount,
        discards=state.discardcount,
        round=state.round + 1,
        ante=ante,
        blind_index=blind_index,
        chips_scored=jnp.int32(0),
        blind_target=blind_target(ante, blind_index),
        key=key,
        shop_slots=jnp.full(MAX_SHOP_SLOTS, J_EMPTY, jnp.int32),
        game_stage=jnp.int32(STAGE_BLIND),
    )
    return nxt, jnp.int32(0), jnp.bool_(False)


def _noop(state: balatroEnvState, action: balatroAction):
    return _no_change(state)


# index = verb; the last entry is V_NOOP
_HANDLERS = [_blind_action, _blind_action, _buy, _sell, _reroll, _next, _noop]


def roll_shop(key, held, shop_size):
    """Fill the unlocked shop slots. -> (MAX_SHOP_SLOTS,) int32, -1 = empty/locked.

    No joker appears twice in one shop, and a held joker never appears (the
    Balatro rule without Showman). When the pool runs dry, the slot gets a
    plain Joker, as in the real game — so an unlocked slot is never empty.
    """
    ids = jnp.arange(N_JOKERS)
    avail = (SHOP_WEIGHT > 0) & ~(ids[:, None] == held[None, :]).any(axis=1)
    logw = jnp.log(jnp.maximum(SHOP_WEIGHT, 1e-30))
    out = jnp.full(MAX_SHOP_SLOTS, J_EMPTY, jnp.int32)
    for i in range(MAX_SHOP_SLOTS):            # static, 4 iterations
        key, sub = jax.random.split(key)
        pick = jax.random.categorical(sub, jnp.where(avail, logw, -jnp.inf)).astype(jnp.int32)
        unlocked = i < shop_size
        fresh = avail.any() & unlocked
        out = out.at[i].set(jnp.where(fresh, pick, jnp.where(unlocked, J_FALLBACK, J_EMPTY)))
        avail = avail & ~((ids == pick) & fresh)
    return out


def _fresh_deal(key, deck_size, hand_size):
    """Shuffle a full deck and deal the opening hand. -> (deck, hand, hand_valid)"""
    deck = jax.random.permutation(key, MAX_DECK_SIZE)

    dead = deck >= deck_size
    deck = compact(deck, ~dead)          # live cards to the front, dead to the tail

    hand_valid = jnp.arange(MAX_HAND_SIZE) < hand_size
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)
    return deck, hand, hand_valid


@jit
def reset(key, deckSize, handSize):
    key, subkey = jax.random.split(key)
    deck, hand, hand_valid = _fresh_deal(subkey, deckSize, handSize)

    return balatroEnvState(
        jokers=jnp.full(MAX_JOKERS, J_EMPTY, dtype=jnp.int32),
        joker_value=jnp.full(MAX_JOKERS, 0, dtype=float),
        joker_state=jnp.full(MAX_JOKERS, 0, dtype=jnp.int32),
        joker_sell_value=jnp.full(MAX_JOKERS, 0, dtype=jnp.int32),
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
        blind_index=jnp.int32(BLIND_SMALL),
        chips_scored=jnp.int32(0),
        blind_target=blind_target(jnp.int32(1), BLIND_SMALL),
        key=key,
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, dtype=jnp.int32),
        shop_size=jnp.int32(SHOP_UNLOCKED),
        reroll_cost=jnp.int32(REROLL_BASE),
        game_stage=jnp.int32(STAGE_BLIND),
        money=jnp.int32(STARTING_MONEY),
        last_cashout=jnp.int32(0),
    )

@jit
def deal(state : balatroEnvState):
    # never deal past the live deck: an empty deck deals fewer cards
    want = state.hand_size - state.hand_valid.sum()
    deal_count = jnp.clip(jnp.minimum(want, state.deck_size - state.deck_ptr), 0)
    empty = ~state.hand_valid
    # each empty lane's position in the refill queue -> which deck slot feeds it
    src = state.deck_ptr + jnp.cumsum(empty) - 1
    fill = first_n(empty, deal_count)

    new_hand = jnp.where(fill, state.deck[jnp.clip(src, 0, MAX_DECK_SIZE - 1)], state.hand)
    new_hand_valid = state.hand_valid | fill

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
    from jokers import JOKER_NAMES
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
    stage = int(state.game_stage)
    print(f"round  {STAGE_NAMES[stage] if 0 <= stage < 4 else stage}   "
          f"ante {int(state.ante)} {BLIND_NAMES[int(state.blind_index)]}   round {int(state.round)}"
          f"   hands {int(state.hands)}   discards {int(state.discards)}"
          f"   money ${int(state.money)}")
    chips, target = int(state.chips_scored), int(state.blind_target)
    frac = 0.0 if target <= 0 else min(chips / target, 1.0)
    filled = int(round(frac * 20))
    print(f"blind  {chips}/{target} chips  [{'#' * filled}{'.' * (20 - filled)}]"
          f"  {int(frac * 100)}%")

    # --- jokers + shop -----------------------------------------------------
    held = [(JOKER_NAMES[int(j)], float(v), int(s))
            for j, v, s in zip(state.jokers, state.joker_value, state.joker_sell_value) if int(j) >= 0]
    print("jokers " + ("  ".join(f"{nm} v={v:g} sell=${s}" for nm, v, s in held) or "(none)")
          + f"   ({len(held)}/{int(state.joker_slots)})")
    shop = [JOKER_NAMES[int(j)] if int(j) >= 0 else "." for j in state.shop_slots[:int(state.shop_size)]]
    print(f"shop   {shop}   reroll ${int(state.reroll_cost)}")

    # --- rng ---------------------------------------------------------------
    print(f"key    {[int(x) for x in jax.random.key_data(state.key)]}"
          f"  (raw uint32 pair)")

    # --- invariants ---------------------------------------------------------
    for b in invariant_errors(state):
        print(f"  WARN  invariant broken: {b}")
    if not invariant_errors(state):
        print("checks all invariants OK")


def invariant_errors(state: balatroEnvState):
    """-> list of broken invariants (empty = healthy). Plain Python, concrete states."""
    n = int(state.hand_size)
    hand = [int(c) for c in state.hand]
    valid = [bool(v) for v in state.hand_valid]
    ptr, size = int(state.deck_ptr), int(state.deck_size)
    deck = [int(c) for c in state.deck]
    jokers = [int(j) for j in state.jokers]
    held = [j for j in jokers if j >= 0]
    dealt = set(deck[:ptr])
    checks = [
        ("deck is a permutation of 0..127", sorted(deck) == list(range(MAX_DECK_SIZE))),
        ("live ids fill deck[:deck_size]", all(c < size for c in deck[:size])),
        ("deck_ptr within the live deck", 0 <= ptr <= size),
        ("valid hand ids were all dealt (in deck[:ptr])",
         all(hand[i] in dealt for i in range(n) if valid[i])),
        ("no valid lane at or past hand_size", not any(valid[n:])),
        ("no duplicate valid card in hand",
         len([hand[i] for i in range(n) if valid[i]]) == len({hand[i] for i in range(n) if valid[i]})),
        ("hands / discards not negative", int(state.hands) >= 0 and int(state.discards) >= 0),
        ("money not negative", int(state.money) >= 0),
        ("jokers compact (no hole before a held joker)", jokers[:len(held)] == held),
        ("held jokers within joker_slots", len(held) <= int(state.joker_slots)),
        ("vacant joker slots hold value 0 and sell 0",
         all(float(state.joker_value[i]) == 0.0 and int(state.joker_sell_value[i]) == 0
             for i in range(MAX_JOKERS) if jokers[i] < 0)),
        ("locked shop lanes are empty",
         all(int(s) == -1 for s in state.shop_slots[int(state.shop_size):])),
        ("game_stage is a STAGE_*", 0 <= int(state.game_stage) <= 3),
    ]
    return [name for name, ok in checks if not ok]


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
