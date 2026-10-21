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

The return contract (NOTES: "reward = CASHOUT dollars"):
    reward = the cashout $ on the step that clears a blind, else 0
             (a failed blind pays $0; chips are never reward)
    done   = this step ended the RUN (game_stage became STAGE_LOST or STAGE_WON)

target_scale is the curriculum dial: every blind target is multiplied by it
(floor 1 chip). reset sets it; 1.0 is the real game.

Randomness inside a step (Bloodstone, Misprint, Gros Michel ...) comes from
fold_in(state.key, purpose); a step that uses it moves state.key on. So
explain_play shows exactly the outcome env_step will produce.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple
from scoring import (play as _play, discard as _discard, PlayIn, Targets, no_info,
                     rule_counts, Cards, effective, HAND_NAMES)
from jokers import MAX_JOKERS, J_EMPTY, N_JOKERS, JOKERS, JOKER_COST, JOKER_RARITY, JOKER_ID
from jokers import (N_QUERY_KINDS, Q_MONEY_DIV_5, Q_EMPTY_JOKER_SLOTS, Q_DECK_NINE_COUNT,
                    Q_DISCARDS_LEFT, Q_NO_DISCARDS_LEFT, Q_DECK_LEFT, Q_JOKER_COUNT, Q_MONEY,
                    Q_OTHER_SELL, Q_DECK_LOST, Q_DELAYED_GRAT,
                    J_STENCIL, J_CHAOS, J_CREDIT, J_MOON, E_ADD_MONEY, E_SELL_VALUE,
                    E_SELL_VALUE_ALL, E_BURGLAR, ST_TODO, ST_LOYALTY, P_ALWAYS, P_NOT_EXTINCT,
                    P_EXTINCT, N_HAND_TYPES)
from jokers import COMMON, UNCOMMON, RARE, LEGENDARY
from jokers import rows_for, resolve, update, update_ex, expired, rule_mods, copy_sources
from jokers import (T_ONCE_MAIN, T_ON_DISCARD, T_END_OF_ROUND, T_ON_BLIND_SELECT,
                    T_ON_REROLL, T_ON_SELL_OTHER)
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
V_MOVE_JOKER = 6  # INSERT the joker in slot target at slot dest (joker order sets XMULT order)
V_MOVE_CARD  = 7  # INSERT the card in lane target at lane dest (lane order = play order, Death)
N_VERBS   = 8
V_NOOP    = 8   # internal: the repair target of every illegal action
VERB_NAMES = ["PLAY", "DISCARD", "BUY", "SELL", "REROLL", "NEXT", "MOVE_JOKER", "MOVE_CARD"]

# --- stages ------------------------------------------------------------------
STAGE_BLIND = 0
STAGE_SHOP  = 1
STAGE_LOST  = 2
STAGE_WON   = 3
STAGE_NAMES = ["BLIND", "SHOP", "LOST", "WON"]

# LEGAL[stage, verb]: which verbs each stage accepts. Everything else is a no-op.
LEGAL = jnp.array([
    #  PLAY   DISC   BUY    SELL   REROLL NEXT   MOVE_J MOVE_C
    [True,  True,  False, True,  False, False, True,  True ],   # BLIND
    [False, False, True,  True,  True,  True,  True,  False],   # SHOP
    [False, False, False, False, False, False, False, False],   # LOST
    [False, False, False, False, False, False, False, False],   # WON
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
INTEREST_CAP  = 5    # ... at most 5 times (To the Moon raises the $ per step)
REROLL_BASE   = 5    # +$1 per reroll, back to base on each shop visit
SHOP_UNLOCKED = 2    # joker slots open in the shop at run start (vouchers raise it)
CREDIT_DEBT   = 20   # Credit Card: go down to -$20 per card

# Shop odds: pick a rarity (70 / 25 / 5 %), then a joker uniform inside it.
# As one weight per joker: rarity weight / jokers of that rarity in the table.
# Legendaries never appear in the shop.
_RARITY_WEIGHT = {COMMON: 70.0, UNCOMMON: 25.0, RARE: 5.0, LEGENDARY: 0.0}
_rarity = [int(r) for r in JOKER_RARITY]
SHOP_WEIGHT = jnp.array([_RARITY_WEIGHT[r] / _rarity.count(r) for r in _rarity], jnp.float32)
J_FALLBACK = JOKER_ID["Joker"]   # what the shop offers when the pool is empty

# fold_in purposes for state.key
K_PLAY, K_ROUND, K_ACQUIRE = 1, 2, 3


class balatroEnvState(NamedTuple):
    # jokers (compact: held jokers fill slots 0..n-1, in order)
    jokers       : jax.Array   # (32,)  int32   -- joker id per slot, J_EMPTY = vacant
    joker_value  : jax.Array   # (32, ) float   -- the value of the joker in each jokers[i] slot
    joker_state  : jax.Array   # (32, ) int32   -- To Do List hand, Loyalty Card start
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
    base_hand_size : jax.Array # scalar int32   -- hand size before jokers (the deck's)
    base_hands   : jax.Array   # scalar int32   -- hands per round before jokers
    base_discards : jax.Array  # scalar int32   -- discards per round before jokers
    discards_used : jax.Array  # scalar int32   -- discards spent this round (Delayed Gratification)

    # round
    round        : jax.Array   # scalar int32   -- blinds started this run, from 1
    ante         : jax.Array   # scalar int32
    blind_index  : jax.Array   # scalar int32   -- BLIND_SMALL / BIG / BOSS
    chips_scored : jax.Array   # scalar int32   -- progress toward blind
    blind_target : jax.Array   # scalar int32   -- chips needed to clear
    target_scale : jax.Array   # scalar float32 -- curriculum: x every blind target (1.0 = real game)
    key          : jax.Array   # PRNG key (new-style typed key)
    hand_played  : jax.Array   # (9,) int32     -- times each hand type was played this run
    hand_round   : jax.Array   # (9,) int32     -- ... this round
    idol_card    : jax.Array   # scalar int32   -- The Idol's card this round (id 0-51)
    mail_rank    : jax.Array   # scalar int32   -- Mail-In Rebate's rank this round
    ancient_suit : jax.Array   # scalar int32   -- Ancient Joker's suit this round
    castle_suit  : jax.Array   # scalar int32   -- Castle's suit this round

    # shop
    shop_slots  : jax.Array    # (4, ) int32    -- joker id per shop slot, -1 = empty or locked
    shop_size   : jax.Array    # scalar int32   -- unlocked shop slots (lanes >= this stay -1)
    reroll_cost : jax.Array    # scalar int32
    free_rerolls : jax.Array   # scalar int32   -- Chaos the Clown rerolls left this shop
    gros_extinct : jax.Array   # scalar bool    -- Gros Michel died: Cavendish may appear
    game_stage  : jax.Array    # scalar int32   -- STAGE_*
    money       : jax.Array    # scalar int32   -- below 0 only with Credit Card
    last_cashout: jax.Array    # scalar int32   -- $ paid at the last blind clear (display)


class balatroAction(NamedTuple):
    verb   : jax.Array                    # scalar int32 -- V_*
    select : jax.Array                    # (32,)  bool  -- which lanes to act on
    target : jax.Array = jnp.int32(0)     # scalar int32 -- shop slot (BUY) / joker slot (SELL) / MOVE source
    dest   : jax.Array = jnp.int32(0)     # scalar int32 -- MOVE destination (read only by the MOVE verbs)


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
    money      : jax.Array   # scalar        -- $ the hand earned (To Do List, Rough Gem ...)


def tree_where(c, a, b):
    """Select a whole pytree: a where c, else b. The branch-free 'if'."""
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def blind_target(ante, blind_index, scale=1.0):
    """Chips needed: ante base x (1 / 1.5 / 2) for small / big / boss, x the
    curriculum scale. Never below 1 chip, so a DISCARD can never clear a blind."""
    base = ANTE_BASE[jnp.clip(ante, 0, ANTE_BASE.shape[0] - 1)]
    return jnp.maximum(jnp.floor(base * BLIND_MULT[blind_index] * scale), 1).astype(jnp.int32)


def build_ctx(state) -> jax.Array:
    held = (state.jokers != J_EMPTY).sum()
    stencils = (state.jokers == J_STENCIL).sum()
    pos = jnp.arange(MAX_DECK_SIZE)
    in_deck = (pos < state.deck_size) & (state.deck >= 0) & (state.deck < 52)
    return (jnp.zeros(N_QUERY_KINDS, jnp.float32)
    .at[Q_MONEY_DIV_5].set(jnp.maximum(state.money // 5, 0))
    # a held Stencil counts as an empty slot ("Joker Stencil included")
    .at[Q_EMPTY_JOKER_SLOTS].set(jnp.maximum(state.joker_slots - held + stencils, 0))
    .at[Q_DECK_NINE_COUNT].set((in_deck & (state.deck % 13 == 7)).sum())
    .at[Q_DISCARDS_LEFT].set(jnp.maximum(state.discards, 0))
    .at[Q_NO_DISCARDS_LEFT].set(state.discards <= 0)
    .at[Q_DECK_LEFT].set(jnp.maximum(state.deck_size - state.deck_ptr, 0))
    .at[Q_JOKER_COUNT].set(held)
    .at[Q_MONEY].set(jnp.maximum(state.money, 0))
    .at[Q_OTHER_SELL].set(state.joker_sell_value.sum())        # minus its own, per slot
    .at[Q_DECK_LOST].set(jnp.maximum(52 - state.deck_size, 0))
    .at[Q_DELAYED_GRAT].set(jnp.where(state.discards_used == 0, jnp.maximum(state.discards, 0), 0))
    )


def live_v(state, joker_value=None) -> jax.Array:
    """v per slot as scoring reads it now: copies resolved, Swashbuckler's own
    sell value removed. -> (MAX_JOKERS,) float32"""
    joker_value = state.joker_value if joker_value is None else joker_value
    erows, eok, ev, _ = effective(state.jokers, joker_value, state.joker_state)
    v = resolve(erows, eok, ev, build_ctx(state))
    own = state.joker_sell_value[jnp.clip(copy_sources(state.jokers), 0, MAX_JOKERS - 1)]
    return v - jnp.where(eok & (erows.QUERY == Q_OTHER_SELL), erows.VALUE * own, 0.0)


def targets(state) -> Targets:
    return Targets(state.idol_card, state.mail_rank, state.ancient_suit, state.castle_suit)


def bankrupt_at(state):
    """The lowest money a purchase may leave: 0, or -$20 per Credit Card."""
    return -CREDIT_DEBT * (state.jokers == J_CREDIT).sum()


def _jokers(state):
    """(rows, active, ctx) — the three inputs every joker pass needs."""
    return rows_for(state.jokers), state.jokers != J_EMPTY, build_ctx(state)


def _play_in(state, select):
    cards, valid, sel = selected_cards(state.hand, state.hand_valid, select.astype(bool), 5)
    held_ok = state.hand_valid & ~sel
    ctx = build_ctx(state)
    return PlayIn(cards % 13, cards // 13, valid, state.hand % 13, state.hand // 13, held_ok,
                  state.jokers, state.joker_value, state.joker_state, ctx,
                  state.money.astype(jnp.float32), state.hands - 1, state.hand_played,
                  state.hand_round, state.blind_index == BLIND_BOSS, targets(state)), sel, cards, valid


def score_play(state: balatroEnvState, select):
    """Score the selected cards. Pure, no state change. -> (PlayResult, PlayOut)"""
    p, sel, cards, valid = _play_in(state, select)
    out = _play(p, rule_mods(state.jokers), jax.random.fold_in(state.key, K_PLAY),
                state.joker_sell_value)
    res = PlayResult(out.handtype, sel, cards, valid, out.mask, out.base_chips, out.base_mult,
                     out.chips, out.mult, out.score, out.money)
    return res, out


@jit
def explain_play(state: balatroEnvState, select) -> PlayResult:
    """What PLAY would score, exactly as env_step scores it."""
    return score_play(state, select)[0]


@jit
def preview(state: balatroEnvState, select) -> PlayResult:
    """The hand the selection makes (the in-game preview). Same as explain_play."""
    return score_play(state, select)[0]


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
      - MOVE_JOKER unless target != dest and both are held slots (< held count) -> no-op
      - MOVE_CARD unless target != dest, both < hand_size, and lane target holds a card -> no-op
      - more than 5 cards selected                  -> the first 5 valid lanes
    """
    verb = action.verb.astype(jnp.int32)
    verb = jnp.where((verb == V_DISCARD) & (state.discards <= 0), V_PLAY, verb)
    in_range = (verb >= 0) & (verb < N_VERBS)
    stage = jnp.clip(state.game_stage, 0, LEGAL.shape[0] - 1)
    legal = in_range & LEGAL[stage, jnp.clip(verb, 0, N_VERBS - 1)]
    branch = jnp.where(legal, verb, V_NOOP)
    action = action._replace(verb=verb, target=jnp.asarray(action.target, jnp.int32),
                             dest=jnp.asarray(action.dest, jnp.int32))
    return jax.lax.switch(branch, _HANDLERS, state, action)


def _no_change(state):
    return state, jnp.int32(0), jnp.bool_(False)


def remove_jokers(state: balatroEnvState, gone) -> balatroEnvState:
    """Take the jokers in the gone slots out (sold, eaten, extinct). The rest
    shift left in order; the budget follows."""
    gone = gone & (state.jokers != J_EMPTY)
    jokers = jnp.where(gone, J_EMPTY, state.jokers)
    keep = jokers != J_EMPTY
    out = state._replace(
        jokers=compact(jokers, keep),
        joker_value=compact(jnp.where(gone, 0.0, state.joker_value), keep),
        joker_state=compact(jnp.where(gone, 0, state.joker_state), keep),
        joker_sell_value=compact(jnp.where(gone, 0, state.joker_sell_value), keep),
    )
    return apply_budget(out)


def apply_budget(state: balatroEnvState) -> balatroEnvState:
    """Hand size, hands and discards per round = the base + every held joker
    (Juggler, Merry Andy, Troubadour, Turtle Bean ...). The cards in hand stay;
    a smaller hand only draws fewer cards."""
    rows, active = rows_for(state.jokers), state.jokers != J_EMPTY
    h = (jnp.where(active, rows.H_SIZE, 0)
         + jnp.where(active & rows.HSIZE_V, state.joker_value, 0.0).astype(jnp.int32)).sum()
    d = jnp.where(active, rows.D_SIZE, 0).sum()
    p = jnp.where(active, rows.H_PLAYS, 0).sum()
    return fit_hand(state._replace(
        hand_size=jnp.clip(state.base_hand_size + h, 1, MAX_HAND_SIZE).astype(jnp.int32),
        discardcount=jnp.maximum(state.base_discards + d, 0).astype(jnp.int32),
        handcount=jnp.maximum(state.base_hands + p, 1).astype(jnp.int32)))


def fit_hand(state: balatroEnvState) -> balatroEnvState:
    """A card in a lane at or past hand_size (the hand size went down): move
    the cards to the front lanes, order kept."""
    lane = jnp.arange(MAX_HAND_SIZE)
    over = (state.hand_valid & (lane >= state.hand_size)).any()
    hand = jnp.where(state.hand_valid, state.hand, -1)
    packed = state._replace(hand=compact(hand, state.hand_valid),
                            hand_valid=compact(state.hand_valid, state.hand_valid))
    return tree_where(over, packed, state)


def _blind_action(state: balatroEnvState, action: balatroAction):
    """PLAY or DISCARD (the verb is already repaired)."""
    is_play = action.verb == V_PLAY
    res, out = score_play(state, action.select)

    # DISCARD: the discarded cards, lane order
    mods = rule_mods(state.jokers)
    dsc = _discard(res.cards % 13, res.cards // 13, res.valid, state.jokers, state.joker_value,
                   state.joker_state, targets(state), mods)

    value = jnp.where(is_play, out.joker_value, dsc.joker_value)
    gone = jnp.where(is_play, out.expired, dsc.expired)
    earned = jnp.where(is_play, out.money, dsc.money)
    score = res.score * is_play
    key, _ = jax.random.split(state.key)

    new_state = state._replace(hands=state.hands - is_play,
                               discards=state.discards - (~is_play),
                               discards_used=state.discards_used + (~is_play),
                               hand_valid=state.hand_valid & ~res.sel,
                               chips_scored=state.chips_scored + score,
                               joker_value=value,
                               money=state.money + jnp.floor(earned).astype(jnp.int32),
                               hand_played=jnp.where(is_play, out.hand_played, state.hand_played),
                               hand_round=jnp.where(is_play, out.hand_round, state.hand_round),
                               key=key)
    new_state = remove_jokers(new_state, gone)
    next_state = deal(new_state)

    cleared = next_state.chips_scored >= state.blind_target
    failed = ~cleared & (next_state.hands <= 0)
    next_state = tree_where(cleared, _clear_blind(next_state), next_state)
    next_state = next_state._replace(
        game_stage=jnp.where(failed, STAGE_LOST, next_state.game_stage))
    reward = jnp.where(cleared, next_state.last_cashout, 0)
    run_over = (next_state.game_stage == STAGE_LOST) | (next_state.game_stage == STAGE_WON)
    return next_state, reward, run_over


def _clear_blind(state: balatroEnvState) -> balatroEnvState:
    """End of round: the END_OF_ROUND jokers, the cashout, then the shop (or the win)."""
    rows, active = rows_for(state.jokers), state.jokers != J_EMPTY
    k_eor, k_todo, k_tg, k_shop, key = jax.random.split(jax.random.fold_in(state.key, K_ROUND), 5)
    mods = rule_mods(state.jokers)
    is_boss = state.blind_index == BLIND_BOSS

    # counters (Popcorn, Turtle Bean, Hit the Road, Campfire, Rocket)
    none = Cards(*[jnp.zeros(1, jnp.int32)] * 2, jnp.zeros(1, bool), jnp.full(1, -1),
                 jnp.zeros(1, bool), jnp.zeros(1, bool))
    counts = rule_counts(rows, none, jnp.zeros(1), no_info(is_boss), targets(state), mods,
                         state.joker_state, False)
    value, _, fired = update_ex(rows, active, state.joker_value, state.joker_state,
                                jnp.int32(T_END_OF_ROUND), counts)
    gone = expired(rows, active, value, fired)
    # Gros Michel / Cavendish: 1 in N to go extinct
    odds = jnp.where(rows.EXPIRE_ODDS > 0, mods.odds / jnp.maximum(rows.EXPIRE_ODDS, 1), 0.0)
    died = active & (jax.random.uniform(k_eor, (MAX_JOKERS,)) < odds)
    extinct = state.gros_extinct | (died & (rows.POOL == P_NOT_EXTINCT)).any()
    gone = gone | died
    # sell value (Egg on itself, Gift Card on every joker)
    sell = state.joker_sell_value + jnp.where(active & (rows.TRIGGER == T_END_OF_ROUND)
                                              & (rows.EFFECT == E_SELL_VALUE), rows.VALUE, 0).astype(jnp.int32)
    gift = jnp.where(active & (rows.TRIGGER == T_END_OF_ROUND) & (rows.EFFECT == E_SELL_VALUE_ALL),
                     rows.VALUE, 0).sum().astype(jnp.int32)
    sell = jnp.where(active, sell + gift, 0)
    # To Do List: a new hand, never the same one
    r = jax.random.randint(k_todo, (MAX_JOKERS,), 0, N_HAND_TYPES - 1)
    todo = jnp.where(r >= state.joker_state, r + 1, r)
    jstate = jnp.where(active & (rows.STATE == ST_TODO), todo, state.joker_state)

    state = remove_jokers(state._replace(joker_value=value, joker_sell_value=sell,
                                         joker_state=jstate, gros_extinct=extinct), gone)

    # interest is paid on the money held BEFORE this cashout
    rate = 1 + (state.jokers == J_MOON).sum()
    interest = jnp.where(state.money >= INTEREST_PER,
                         rate * jnp.minimum(state.money // INTEREST_PER, INTEREST_CAP), 0)
    cashout = (BLIND_REWARD[state.blind_index] + state.hands + interest
               + joker_money(state, state.joker_value))

    won = (state.blind_index == BLIND_BOSS) & (state.ante >= WIN_ANTE)
    tg = new_targets(k_tg, state)
    return state._replace(
        money=state.money + cashout,
        last_cashout=cashout,
        game_stage=jnp.where(won, STAGE_WON, STAGE_SHOP),
        shop_slots=roll_shop(k_shop, state.jokers, state.shop_size, mods.showman, state.gros_extinct),
        reroll_cost=jnp.int32(REROLL_BASE),
        idol_card=tg.idol_card, mail_rank=tg.mail_rank,
        ancient_suit=tg.ancient_suit, castle_suit=tg.castle_suit,
        key=key,
    )


def new_targets(key, state) -> Targets:
    """reset_idol_card / reset_mail_rank / reset_ancient_card / reset_castle_card:
    a random card of the full deck (Idol, Mail-In rank, Castle suit); a new suit
    for Ancient Joker."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    live = jnp.arange(MAX_DECK_SIZE) < state.deck_size
    card = lambda k: state.deck[jax.random.categorical(k, jnp.where(live, 0.0, -jnp.inf))]
    cur = state.ancient_suit                       # -1 at the start: any of the 4 suits
    a = jnp.floor(jax.random.uniform(k3) * jnp.where(cur < 0, 4, 3)).astype(jnp.int32)
    return Targets(card(k1), card(k2) % 13, jnp.where((cur >= 0) & (a >= cur), a + 1, a),
                   card(k4) // 13)


def joker_money(state: balatroEnvState, joker_value) -> jax.Array:
    """$ the held END_OF_ROUND money jokers pay into the cashout (Golden Joker,
    Cloud 9, Rocket, Delayed Gratification). Copies never pay."""
    rows, active, ctx = _jokers(state)
    v = resolve(rows, active, joker_value, ctx)
    pays = active & (rows.TRIGGER == T_END_OF_ROUND) & (rows.EFFECT == E_ADD_MONEY)
    return jnp.floor(jnp.where(pays, v, 0.0).sum()).astype(jnp.int32)


def acquire(state: balatroEnvState, jid) -> tuple[balatroEnvState, jax.Array]:
    """Put joker jid in the first free slot. -> (state, ok). No money changes.

    The one place a joker enters the loadout: BUY uses it, and so will packs,
    Judgement and the GUI's dev button. ok is False (state unchanged) when the
    joker slots are full or jid is not a joker.
    """
    safe = jnp.clip(jid, 0, N_JOKERS - 1)
    held = (state.jokers != J_EMPTY).sum()
    ok = (jid >= 0) & (jid < N_JOKERS) & (held < state.joker_slots)
    kind = JOKERS.STATE[safe]
    todo = jax.random.randint(jax.random.fold_in(state.key, K_ACQUIRE), (), 0, N_HAND_TYPES)
    start = jnp.select([kind == ST_TODO, kind == ST_LOYALTY], [todo, state.hand_played.sum()], 0)

    dest = held   # jokers are compact, so the first free slot is the count
    key, _ = jax.random.split(state.key)
    got = state._replace(
        jokers=state.jokers.at[dest].set(jid),
        joker_value=state.joker_value.at[dest].set(JOKERS.VALUE[safe]),
        joker_state=state.joker_state.at[dest].set(start),
        joker_sell_value=state.joker_sell_value.at[dest].set(jnp.maximum(1, JOKER_COST[safe] // 2)),
        free_rerolls=state.free_rerolls + (jid == J_CHAOS),
        key=key,
    )
    got = apply_budget(got)
    # a new discard joker also adds to the discards left (ease_discard)
    got = got._replace(discards=state.discards + JOKERS.D_SIZE[safe])
    return tree_where(ok, got, state), ok


def _buy(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_SHOP_SLOTS - 1)
    jid = state.shop_slots[slot]
    cost = JOKER_COST[jnp.clip(jid, 0, N_JOKERS - 1)]

    got, has_room = acquire(state, jid)
    ok = (t >= 0) & (t < state.shop_size) & (state.money - cost >= bankrupt_at(state)) & has_room
    bought = got._replace(money=state.money - cost,
                          shop_slots=state.shop_slots.at[slot].set(J_EMPTY))
    return tree_where(ok, bought, state), jnp.int32(0), jnp.bool_(False)


def _sell(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_JOKERS - 1)
    jid = state.jokers[slot]
    ok = (t >= 0) & (t < MAX_JOKERS) & (jid != J_EMPTY)
    gone = jnp.arange(MAX_JOKERS) == slot

    # T_ON_SELL_OTHER fires for every other joker (Campfire)
    rows, active = rows_for(state.jokers), (state.jokers != J_EMPTY) & ~gone
    value, _ = update(rows, active, state.joker_value, state.joker_state, jnp.int32(T_ON_SELL_OTHER))
    safe = jnp.clip(jid, 0, N_JOKERS - 1)
    sold = remove_jokers(state._replace(
        joker_value=value,
        money=state.money + state.joker_sell_value[slot],
        discards=jnp.maximum(state.discards - JOKERS.D_SIZE[safe], 0),
        free_rerolls=jnp.maximum(state.free_rerolls - (jid == J_CHAOS), 0),
    ), gone)
    return tree_where(ok, sold, state), jnp.int32(0), jnp.bool_(False)


def reroll_price(state):
    """$ the next reroll costs (0 while Chaos the Clown has a free one)."""
    return jnp.where(state.free_rerolls > 0, 0, state.reroll_cost)


def _reroll(state: balatroEnvState, action: balatroAction):
    free = state.free_rerolls > 0
    cost = reroll_price(state)
    ok = free | (state.money - cost >= bankrupt_at(state))
    key, sub = jax.random.split(state.key)
    rows, active = rows_for(state.jokers), state.jokers != J_EMPTY
    value, _ = update(rows, active, state.joker_value, state.joker_state, jnp.int32(T_ON_REROLL))
    mods = rule_mods(state.jokers)
    rolled = state._replace(
        shop_slots=roll_shop(sub, state.jokers, state.shop_size, mods.showman, state.gros_extinct),
        money=state.money - cost,
        reroll_cost=state.reroll_cost + (~free),
        free_rerolls=state.free_rerolls - free,
        joker_value=value,
        key=key,
    )
    return tree_where(ok, rolled, state), jnp.int32(0), jnp.bool_(False)


def _next(state: balatroEnvState, action: balatroAction):
    """Leave the shop: advance the blind (and the ante after a boss), fresh deck."""
    was_boss = state.blind_index == BLIND_BOSS
    blind_index = jnp.where(was_boss, BLIND_SMALL, state.blind_index + 1)
    ante = state.ante + was_boss

    rows, active = rows_for(state.jokers), state.jokers != J_EMPTY
    value, jstate = update(rows, active, state.joker_value, state.joker_state,
                           jnp.int32(T_ON_BLIND_SELECT))
    # Burglar (copies too): +3 hands, no discards
    erows, eok, _, _ = effective(state.jokers, state.joker_value, state.joker_state)
    burgle = eok & (erows.TRIGGER == T_ON_BLIND_SELECT) & (erows.EFFECT == E_BURGLAR)
    extra = jnp.where(burgle, erows.VALUE, 0.0).sum().astype(jnp.int32)

    key, sub = jax.random.split(state.key)
    deck, hand, hand_valid = _fresh_deal(sub, state.deck_size, state.hand_size)
    nxt = state._replace(
        joker_value=value,
        joker_state=jstate,
        hand=hand,
        hand_valid=hand_valid,
        deck=deck,
        deck_ptr=state.hand_size,
        hands=state.handcount + extra,
        discards=jnp.where(burgle.any(), 0, state.discardcount),
        discards_used=jnp.int32(0),
        hand_round=jnp.zeros(N_HAND_TYPES, jnp.int32),
        free_rerolls=(state.jokers == J_CHAOS).sum().astype(jnp.int32),
        round=state.round + 1,
        ante=ante,
        blind_index=blind_index,
        chips_scored=jnp.int32(0),
        blind_target=blind_target(ante, blind_index, state.target_scale),
        key=key,
        shop_slots=jnp.full(MAX_SHOP_SLOTS, J_EMPTY, jnp.int32),
        game_stage=jnp.int32(STAGE_BLIND),
    )
    return nxt, jnp.int32(0), jnp.bool_(False)


def insert_perm(n, src, dst):
    """Gather index for 'take the item at src, put it at dst': the items between
    shift one place toward src. new[k] = old[perm[k]]."""
    k = jnp.arange(n)
    down = (src < dst) & (k >= src) & (k < dst)     # src left of dst: the gap fills from the right
    up = (src > dst) & (k > dst) & (k <= src)       # src right of dst: the gap fills from the left
    return jnp.where(k == dst, src, jnp.where(down, k + 1, jnp.where(up, k - 1, k)))


def _move_joker(state: balatroEnvState, action: balatroAction):
    """Insert the joker in slot target at slot dest. Every per-slot array moves
    with its joker. Held jokers are packed (slots 0..held-1), so both ends must
    be < held."""
    i, j = action.target, action.dest
    held = (state.jokers != J_EMPTY).sum()
    ok = (i >= 0) & (j >= 0) & (i < held) & (j < held) & (i != j)
    perm = insert_perm(MAX_JOKERS, jnp.clip(i, 0, MAX_JOKERS - 1), jnp.clip(j, 0, MAX_JOKERS - 1))
    moved = state._replace(
        jokers=state.jokers[perm],
        joker_value=state.joker_value[perm],
        joker_state=state.joker_state[perm],
        joker_sell_value=state.joker_sell_value[perm],
    )
    return tree_where(ok, moved, state), jnp.int32(0), jnp.bool_(False)


def _move_card(state: balatroEnvState, action: balatroAction):
    """Insert the card in lane target at lane dest. Lanes may be holes (an empty
    deck leaves holes); holes shift like cards, but only a card can be moved."""
    i, j = action.target, action.dest
    src = jnp.clip(i, 0, MAX_HAND_SIZE - 1)
    ok = ((i >= 0) & (j >= 0) & (i < state.hand_size) & (j < state.hand_size) & (i != j)
          & state.hand_valid[src])
    perm = insert_perm(MAX_HAND_SIZE, src, jnp.clip(j, 0, MAX_HAND_SIZE - 1))
    moved = state._replace(hand=state.hand[perm], hand_valid=state.hand_valid[perm])
    return tree_where(ok, moved, state), jnp.int32(0), jnp.bool_(False)


def _noop(state: balatroEnvState, action: balatroAction):
    return _no_change(state)


# index = verb; the last entry is V_NOOP
_HANDLERS = [_blind_action, _blind_action, _buy, _sell, _reroll, _next, _move_joker, _move_card, _noop]


def roll_shop(key, held, shop_size, showman=False, extinct=False):
    """Fill the unlocked shop slots. -> (MAX_SHOP_SLOTS,) int32, -1 = empty/locked.

    No joker appears twice in one shop, and a held joker never appears (the
    Balatro rule without Showman; with Showman both are allowed). Gros Michel
    leaves the pool when it goes extinct, and Cavendish joins it then. When the
    pool runs dry, the slot gets a plain Joker, as in the real game — so an
    unlocked slot is never empty.
    """
    ids = jnp.arange(N_JOKERS)
    pool = jnp.select([JOKERS.POOL == P_NOT_EXTINCT, JOKERS.POOL == P_EXTINCT],
                      [~jnp.asarray(extinct), jnp.asarray(extinct)], default=True)
    avail = (SHOP_WEIGHT > 0) & pool & (showman | ~(ids[:, None] == held[None, :]).any(axis=1))
    logw = jnp.log(jnp.maximum(SHOP_WEIGHT, 1e-30))
    out = jnp.full(MAX_SHOP_SLOTS, J_EMPTY, jnp.int32)
    for i in range(MAX_SHOP_SLOTS):            # static, 4 iterations
        key, sub = jax.random.split(key)
        pick = jax.random.categorical(sub, jnp.where(avail, logw, -jnp.inf)).astype(jnp.int32)
        unlocked = i < shop_size
        fresh = avail.any() & unlocked
        out = out.at[i].set(jnp.where(fresh, pick, jnp.where(unlocked, J_FALLBACK, J_EMPTY)))
        avail = avail & ~((ids == pick) & fresh & ~showman)
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
def reset(key, deckSize, handSize, target_scale=1.0):
    key, subkey, tkey = jax.random.split(key, 3)
    deck, hand, hand_valid = _fresh_deal(subkey, deckSize, handSize)

    s = balatroEnvState(
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
        base_hand_size=jnp.int32(handSize),
        base_hands=jnp.int32(4),
        base_discards=jnp.int32(3),
        discards_used=jnp.int32(0),
        round=jnp.int32(1),
        ante=jnp.int32(1),
        blind_index=jnp.int32(BLIND_SMALL),
        chips_scored=jnp.int32(0),
        blind_target=blind_target(jnp.int32(1), BLIND_SMALL, target_scale),
        target_scale=jnp.float32(target_scale),
        key=key,
        hand_played=jnp.zeros(N_HAND_TYPES, jnp.int32),
        hand_round=jnp.zeros(N_HAND_TYPES, jnp.int32),
        idol_card=jnp.int32(51), mail_rank=jnp.int32(12),       # the game's defaults (A of Spades)
        ancient_suit=jnp.int32(-1), castle_suit=jnp.int32(3),
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, dtype=jnp.int32),
        shop_size=jnp.int32(SHOP_UNLOCKED),
        reroll_cost=jnp.int32(REROLL_BASE),
        free_rerolls=jnp.int32(0),
        gros_extinct=jnp.bool_(False),
        game_stage=jnp.int32(STAGE_BLIND),
        money=jnp.int32(STARTING_MONEY),
        last_cashout=jnp.int32(0),
    )
    tg = new_targets(tkey, s)
    return s._replace(idol_card=tg.idol_card, mail_rank=tg.mail_rank,
                      ancient_suit=tg.ancient_suit, castle_suit=tg.castle_suit)

@jit
def deal(state : balatroEnvState):
    # never deal past the live deck: an empty deck deals fewer cards
    state = fit_hand(state)
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


_H_SIZE = [int(x) for x in JOKERS.H_SIZE]
_HSIZE_V = [bool(x) for x in JOKERS.HSIZE_V]


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
        ("no valid lane at or past max(hand_size, cards held)",
         not any(valid[max(n, sum(valid)):])),
        ("no duplicate valid card in hand",
         len([hand[i] for i in range(n) if valid[i]]) == len({hand[i] for i in range(n) if valid[i]})),
        ("hands / discards not negative", int(state.hands) >= 0 and int(state.discards) >= 0),
        ("money not below the Credit Card limit",
         int(state.money) >= -CREDIT_DEBT * sum(j == J_CREDIT for j in jokers)),
        ("jokers compact (no hole before a held joker)", jokers[:len(held)] == held),
        ("held jokers within joker_slots", len(held) <= int(state.joker_slots)),
        ("vacant joker slots hold value 0 and sell 0",
         all(float(state.joker_value[i]) == 0.0 and int(state.joker_sell_value[i]) == 0
             for i in range(MAX_JOKERS) if jokers[i] < 0)),
        ("locked shop lanes are empty",
         all(int(s) == -1 for s in state.shop_slots[int(state.shop_size):])),
        ("game_stage is a STAGE_*", 0 <= int(state.game_stage) <= 3),
        ("hand size = base + held jokers", int(state.hand_size) == min(max(int(state.base_hand_size) + sum(
            _H_SIZE[j] + (int(float(state.joker_value[i])) if _HSIZE_V[j] else 0)
            for i, j in enumerate(jokers) if j >= 0), 1), MAX_HAND_SIZE)),
        ("round targets in range", 0 <= int(state.idol_card) < 52 and 0 <= int(state.mail_rank) < 13
         and 0 <= int(state.ancient_suit) < 4 and 0 <= int(state.castle_suit) < 4),
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
