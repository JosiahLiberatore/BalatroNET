"""Jokers as data: axis enums, the static row table, and the resolve pass.

A joker is one row over independent axes. One generic evaluator reads the rows.
To add a joker, add a row. Do not add a branch to the scoring pipeline.

Import direction (acyclic):
    arrays.py / jokers.py    no project imports
    scoring.py               imports jokers
    env.py                   imports both

Static data and live data are different axes:
    JOKERS[joker_id]     the rule. One constant row per joker. JOKERS.VALUE is
                         the magnitude for CONSTANT, the per-unit for QUERY,
                         and the seed for COUNTER. The update rules live in
                         UPD_TRIGGER / UPD_DELTA / UPD_COND / UPD_PARAM / UPD_MODE.
    joker_value[slot]    (EnvState) the live magnitude v, float32. Seeded from
                         JOKERS.VALUE on acquire. Written by the update pass.
    joker_state[slot]    (EnvState) live discrete control memory, int32
                         (To Do List: its hand; Loyalty Card: hands at buy).

Scoring reads joker_value and applies it by EFFECT. Scoring never reads SOURCE.
SOURCE tells resolve() how to produce v. rows_for() maps id order to slot order.

The rules come from the Balatro 1.0.1o source (card.lua calculate_joker,
state_events.lua evaluate_play, misc_functions.lua evaluate_poker_hand).
"""

import jax.numpy as jnp
import jax
from typing import NamedTuple

# region AXES
# ---------------------------------------------------------------------------
# AXIS 1 — TRIGGER : when the row fires
# ---------------------------------------------------------------------------
# The first three values are the scoring stages, in pipeline order. The order
# makes XMULT correct: (4+4)*3 != (4*3)+4.
T_PER_SCORED      = 0    # per scored card, play order (repetitions included)
T_PER_HELD        = 1    # per card held in hand, lane order
T_ONCE_MAIN       = 2    # once per hand, slot order. As an UPDATE event it is
                         # the "before" step of a PLAY (Green +1, Ride the Bus)
# The values below drive the update pass and the state-write effects.
T_END_OF_ROUND    = 3    # after a blind is beaten, before the cashout
T_ON_DISCARD      = 4    # once per DISCARD action (Green Joker, Faceless Joker)
T_ON_SHOP_ENTER   = 5    # (unused)
T_ON_SHOP_EXIT    = 6    # Perkeo (not authored)
T_ON_BLIND_SELECT = 7    # NEXT starts a blind (Burglar)
T_ON_SOLD         = 8    # the sold joker itself (not authored yet)
T_ON_PACK_OPEN    = 9    # Hallucination (not authored)
T_ON_PACK_SKIP    = 10   # Red Card (not authored)
T_ON_BOSS         = 11   # Chicot, Luchador, Matador (not authored)
T_BEFORE          = 12   # once per PLAY, before any card scores (To Do List $)
T_AFTER           = 13   # once per PLAY, after the score (Ice Cream, Seltzer)
T_ON_DISCARD_CARD = 14   # per discarded card, lane order (Castle, Mail-In Rebate)
T_ON_REROLL       = 15   # a shop reroll (Flash Card)
T_ON_SELL_OTHER   = 16   # another joker is sold (Campfire)
T_PASSIVE         = 17   # no event: a budget, rule or copy joker
T_NONE            = -1   # padding for an unused update rule. No event is -1,
                         # so the rule never fires.

TRIGGER_NAMES = ["PER_SCORED", "PER_HELD", "ONCE_MAIN", "END_OF_ROUND",
                 "ON_DISCARD", "ON_SHOP_ENTER", "ON_SHOP_EXIT",
                 "ON_BLIND_SELECT", "ON_SOLD", "ON_PACK_OPEN", "ON_PACK_SKIP",
                 "ON_BOSS", "BEFORE", "AFTER", "ON_DISCARD_CARD", "ON_REROLL",
                 "ON_SELL_OTHER", "PASSIVE"]


# ---------------------------------------------------------------------------
# AXIS 2 — CONDITION : (kind, param). Which card, or whether the joker fires.
# ---------------------------------------------------------------------------
# PER-CARD kinds test one card (played, held or discarded). GATE kinds test the
# hand or the round once; on a per-card trigger a gate applies to every card.
# Update rules use the same kinds (UPD_COND): a per-card event counts the
# matching cards, a once event counts 1 when the gate is open.
C_NONE           = 0    # always
C_SUIT           = 1    # card: suit == param (Smeared Joker: red/black)
C_RANK_IN        = 2    # card: rank in the 13-bit mask param
C_IS_FACE        = 3    # card: J/Q/K (Pareidolia: every card)
C_HAND_HAS       = 4    # gate: the played hand contains hand type param
C_COUNT_CMP      = 5    # gate: cards played <= param (Half Joker)
C_RANK_SUIT      = 6    # card: the round's Idol card (rank and suit)
C_POSITION       = 7    # card: the param-th scored card (Hanging Chad: 0)
C_FIRST_FACE     = 8    # card: the first scored face card (Photograph)
C_ANCIENT_SUIT   = 9    # card: the round's Ancient Joker suit
C_LOWEST_HELD    = 10   # card: the lowest held card, the last one on a tie (Raised Fist)
C_MAIL_RANK      = 11   # card: the round's Mail-In Rebate rank
C_CASTLE_SUIT    = 12   # card: the round's Castle suit
C_FACES_GE       = 13   # gate: >= param face cards in the discard (Faceless Joker)
C_LAST_HAND      = 14   # gate: this is the last hand of the round (Dusk, Acrobat)
C_HAND_REPEAT    = 15   # gate: this hand type was already played this round (Card Sharp)
C_SEEING_DOUBLE  = 16   # gate: a scored Club and a scored card of another suit
C_FLOWER_POT     = 17   # gate: scored Diamond, Club, Heart and Spade
C_HELD_ALL_BLACK = 18   # gate: every held card is a Club or a Spade (Blackboard)
C_LOYALTY        = 19   # gate: every (param+1)-th hand since the buy (Loyalty Card)
C_TODO_HAND      = 20   # gate: the hand type is the joker's To Do hand (joker_state)
C_COUNT_EQ       = 21   # gate: cards played == param (Square Joker)
C_SCORING_FACE   = 22   # gate: a scored card is a face card (Ride the Bus)
C_MOST_PLAYED    = 23   # gate: the hand type is now played more than every other (Obelisk)
C_BOSS           = 24   # gate: the blind is a Boss Blind (Campfire, Rocket)

CONDITION_NAMES = ["NONE", "SUIT", "RANK_IN", "IS_FACE", "HAND_HAS",
                   "COUNT_CMP", "RANK_SUIT", "POSITION", "FIRST_FACE", "ANCIENT_SUIT",
                   "LOWEST_HELD", "MAIL_RANK", "CASTLE_SUIT", "FACES_GE", "LAST_HAND",
                   "HAND_REPEAT", "SEEING_DOUBLE", "FLOWER_POT", "HELD_ALL_BLACK",
                   "LOYALTY", "TODO_HAND", "COUNT_EQ", "SCORING_FACE", "MOST_PLAYED", "BOSS"]
PER_CARD_CONDS = (C_SUIT, C_RANK_IN, C_IS_FACE, C_RANK_SUIT, C_POSITION, C_FIRST_FACE,
                  C_ANCIENT_SUIT, C_LOWEST_HELD, C_MAIL_RANK, C_CASTLE_SUIT)


# ---------------------------------------------------------------------------
# AXIS 3 — EFFECT : what the row writes. Three families, one enum.
# ---------------------------------------------------------------------------
# Family A — SCORING: change the (chips, mult) carry of the scan.
E_ADD_MULT          = 0
E_ADD_CHIPS         = 1
E_XMULT             = 2    # applied in scan order, not summed
E_RETRIGGER         = 3    # extra repetitions of the card (a COUNTER row: VALUE2 of them)

# Family B — STATE: a side effect applied after scoring.
E_ADD_MONEY         = 4
E_TRANSFORM_CARD    = 5    # Midas Mask (not authored)
E_DESTROY_CARD      = 6    # Sixth Sense (not authored)
E_SPAWN_CARD        = 7    # DNA (not authored)
E_SPAWN_CONSUMABLE  = 8    # 8 Ball, Seance, Vagabond (not authored)
E_SELL_VALUE        = 9    # Egg — the joker's own sell value, not money

# Family C — BUDGET: round scalars. Budget jokers use the H_SIZE / D_SIZE /
# H_PLAYS columns instead (pure functions of the held jokers, recomputed on
# every change, so the sale of Merry Andy restores hand size at once).
E_ADD_HAND          = 10
E_ADD_DISCARD       = 11
E_SET_DISCARD       = 12
E_ADD_HANDSIZE      = 13
E_ADD_REROLL        = 14

# Family A, two-part and special magnitudes
E_NONE              = 15   # no effect of its own (passive, copy)
E_CHIPS_MULT        = 16   # +VALUE chips and +VALUE2 mult (Scholar, Walkie Talkie)
E_MULT_NOMINAL      = 17   # +VALUE x the card's chip value as mult (Raised Fist)
E_MULT_RANDOM       = 18   # +randint(0, VALUE) mult (Misprint)
E_XMULT_UNCOMMON    = 19   # X v after each Uncommon joker's main effect (Baseball Card)
E_SELL_VALUE_ALL    = 20   # +v sell value to every held joker (Gift Card)
E_BURGLAR           = 21   # +v hands, discards = 0 (Burglar)

EFFECT_NAMES = ["ADD_MULT", "ADD_CHIPS", "XMULT", "RETRIGGER", "ADD_MONEY",
                "TRANSFORM_CARD", "DESTROY_CARD", "SPAWN_CARD",
                "SPAWN_CONSUMABLE", "SELL_VALUE", "ADD_HAND", "ADD_DISCARD",
                "SET_DISCARD", "ADD_HANDSIZE", "ADD_REROLL", "NONE", "CHIPS_MULT",
                "MULT_NOMINAL", "MULT_RANDOM", "XMULT_UNCOMMON", "SELL_VALUE_ALL",
                "BURGLAR"]


# ---------------------------------------------------------------------------
# AXIS 4 — SOURCE : how resolve() produces the live value v.
# ---------------------------------------------------------------------------
S_CONSTANT = 0    # v = VALUE                                  (Joker, Duo)
S_COUNTER  = 1    # v = joker_value, evolved by update()       (Green, Obelisk)
S_QUERY    = 2    # v = VALUE * a count from global state      (Bootstraps)

SOURCE_NAMES = ["CONSTANT", "COUNTER", "QUERY"]

# QUERY kinds. resolve() switches over query_kind, never over joker id.
# To add a query, add a kind here.
Q_NONE              = 0
Q_MONEY_DIV_5       = 1    # Bootstraps
Q_EMPTY_JOKER_SLOTS = 2    # Joker Stencil. A held Stencil counts as empty
                           # ("Joker Stencil included"), so the count is >= 1.
Q_DECK_STEEL_COUNT  = 3    # Steel Joker (not authored)
Q_DECK_STONE_COUNT  = 4    # Stone Joker (not authored)
Q_DECK_NINE_COUNT   = 5    # Cloud 9 - 9s in the FULL deck, not the undealt part
Q_UNCOMMON_JOKERS   = 6    # (unused: Baseball Card is E_XMULT_UNCOMMON)
Q_DISCARDS_LEFT     = 7    # Banner
Q_NO_DISCARDS_LEFT  = 8    # Mystic Summit - 1 when discards == 0, else 0
Q_DECK_LEFT         = 9    # Blue Joker - undealt cards in the deck
Q_JOKER_COUNT       = 10   # Abstract Joker - held jokers, itself included
Q_MONEY             = 11   # Bull
Q_OTHER_SELL        = 12   # Swashbuckler - sell value of the other held jokers
Q_HAND_PLAYED       = 13   # Supernova - times this hand type was played this run
Q_DECK_LOST         = 14   # Erosion - cards below the starting 52
Q_DELAYED_GRAT      = 15   # Delayed Gratification - discards left if none was used
N_QUERY_KINDS       = 16   # length of ctx

# A buy or sell during a blind changes a query result. resolve() runs on each
# hand and its result is never cached.


# ---------------------------------------------------------------------------
# AXIS 5 — STATE : the kind of discrete control memory the row keeps (int)
# ---------------------------------------------------------------------------
ST_NONE             = 0
ST_SCALE_COUNTER    = 1    # Green Joker, Ride the Bus, Seltzer countdown
ST_ROTATE_GLOBAL    = 2    # (the round targets live in EnvState, not per joker)
ST_ROTATE_FROM_DECK = 3
ST_TODO             = 4    # To Do List: joker_state = its hand type, new each round
ST_LOYALTY          = 5    # Loyalty Card: joker_state = hands played at the buy

STATE_NAMES = ["NONE", "SCALE_COUNTER", "ROTATE_GLOBAL", "ROTATE_FROM_DECK", "TODO", "LOYALTY"]


# ---------------------------------------------------------------------------
# AXIS 6 — COPYABLE : can Blueprint or Brainstorm copy this row?
# ---------------------------------------------------------------------------
# A copier scores with the row and the live value of its target. Counter
# updates, the cashout money and sell-value effects are never copied (the game
# guards them with "not context.blueprint" or reads them outside
# calculate_joker). COPYABLE is the game's blueprint_compat flag.

# ---------------------------------------------------------------------------
# UPDATE RULES : how a COUNTER evolves. Up to N_UPDATE_RULES per row.
# ---------------------------------------------------------------------------
# A rule is (trigger, delta, cond, param, mode). n = the count of the rule's
# condition on that event (matching cards on a per-card event, 0 / 1 on a
# once event).
#   M_ADD             v += delta * n                      (Castle: +3 per card)
#   M_RESET           if n > 0: v = seed                  (Hit the Road at round end)
#   M_RESET_ELSE_ADD  if n > 0: v = seed, else v += delta (Ride the Bus, Obelisk)
# A rule that fires floors v at 0.
N_UPDATE_RULES = 2
M_ADD, M_RESET, M_RESET_ELSE_ADD = 0, 1, 2

# POOL : when the shop may offer the joker
P_ALWAYS, P_NOT_EXTINCT, P_EXTINCT = 0, 1, 2     # Gros Michel / Cavendish

NO_EXPIRE = -1e9    # EXPIRE_AT for a joker that never runs out

# endregion

class Joker(NamedTuple):
    """The row schema. Scalars = one row. (MAX_JOKERS,) arrays = one loadout.

    The UPD_* columns have one more axis: (N_UPDATE_RULES,) per row,
    (MAX_JOKERS, N_UPDATE_RULES) per loadout.
    """
    TRIGGER    : jnp.ndarray   # int32   — T_*
    CONDITION  : jnp.ndarray   # int32   — C_*
    COND_PARAM : jnp.ndarray   # int32   — suit / rank mask / handtype / ordinal
    EFFECT     : jnp.ndarray   # int32   — E_*
    SOURCE     : jnp.ndarray   # int32   — S_*  (read by resolve() only)
    QUERY      : jnp.ndarray   # int32   — Q_*
    VALUE      : jnp.ndarray   # float32 — magnitude / per-unit / COUNTER seed
    VALUE2     : jnp.ndarray   # float32 — second magnitude (CHIPS_MULT mult, counter RETRIGGER reps)
    PROB       : jnp.ndarray   # int32   — the effect fires 1 in PROB (0 = always)
    STATE      : jnp.ndarray   # int32   — ST_*, the kind of memory, not the memory
    COPYABLE   : jnp.ndarray   # bool    — Blueprint / Brainstorm can copy this row
    UPD_TRIGGER: jnp.ndarray   # int32   — T_* per update rule, T_NONE = unused
    UPD_DELTA  : jnp.ndarray   # float32 — the step of that rule
    UPD_COND   : jnp.ndarray   # int32   — C_* the rule counts
    UPD_PARAM  : jnp.ndarray   # int32   — its param
    UPD_MODE   : jnp.ndarray   # int32   — M_*
    EXPIRE_AT  : jnp.ndarray   # float32 — destroyed when an update leaves v <= this
    EXPIRE_ODDS: jnp.ndarray   # int32   — destroyed 1 in N at the end of a round (0 = never)
    POOL       : jnp.ndarray   # int32   — P_*
    H_SIZE     : jnp.ndarray   # int32   — hand size while held
    D_SIZE     : jnp.ndarray   # int32   — discards per round while held
    H_PLAYS    : jnp.ndarray   # int32   — hands per round while held
    HSIZE_V    : jnp.ndarray   # bool    — hand size += v (Turtle Bean)

# ---------------------------------------------------------------------------
# AUTHORING RECORD — one joker per line. Compiled to columns below.
# ---------------------------------------------------------------------------
# Rows are easy to write. Columns are what JAX needs. Write rows and transpose
# at import. Defaults cover the common case. Set only the unusual axes.
COMMON, UNCOMMON, RARE, LEGENDARY = 1, 2, 3, 4

# Suits, for COND_PARAM (scoring.py: 0=clubs 1=diamonds 2=hearts 3=spades).
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3

# Hand types, for C_HAND_HAS. Index-aligned with scoring.HAND_NAMES (jokers.py
# imports nothing from the project, so the indices are written here).
H_STRAIGHT_FLUSH, H_FOUR, H_FULL_HOUSE, H_FLUSH, H_STRAIGHT = 0, 1, 2, 3, 4
H_THREE, H_TWO_PAIR, H_PAIR, H_HIGH = 5, 6, 7, 8
N_HAND_TYPES = 9


def _ranks(names):
    """'2 4 6 8 T' -> the 13-bit rank mask for C_RANK_IN (bit r = rank r, 0 = two)."""
    order = "23456789TJQKA"
    return sum(1 << order.index(n) for n in names.split())


class JokerDef(NamedTuple):
    """One authored joker. Plain Python. Never traced, never in a pytree."""
    name    : str
    cost    : int
    rarity  : int
    trigger : int
    effect  : int
    value   : float
    cond    : int = C_NONE
    param   : int = 0
    source  : int = S_CONSTANT
    query   : int = Q_NONE
    state   : int = ST_NONE
    copyable: bool = True
    updates : tuple = ()     # ((trigger, delta[, cond, param, mode]), ...) — COUNTER rows only
    desc    : str = ""       # the in-game text. Display only; the env never reads it.
    value2  : float = 0.0
    prob    : int = 0
    expire_at: float = NO_EXPIRE
    expire_odds: int = 0
    pool    : int = P_ALWAYS
    h_size  : int = 0
    d_size  : int = 0
    h_plays : int = 0
    hsize_v : bool = False

# ---------------------------------------------------------------------------
# THE TABLE — one joker per line. The id is the row index.
# ---------------------------------------------------------------------------
# Rows can be reordered. Ids are valid within one training run only. Nothing
# hardcodes an id: J_* constants resolve by name through JOKER_ID.
# Not authored yet: jokers that need card enhancements, consumables, hand
# levels, packs, skips/tags, boss blinds or joker creation/destruction.
#
#        name                     cost  rarity     trigger              effect                value    optional axes
TABLE = [
    JokerDef("Joker",                2, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           4.0,
             desc="+4 Mult"),
    JokerDef("Greedy Joker",         5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=DIAMONDS,
             desc="Played cards with Diamond suit give +3 Mult when scored"),
    JokerDef("Bootstraps",           7, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           2.0, source=S_QUERY, query=Q_MONEY_DIV_5,
             desc="+2 Mult for every $5 you have"),
    JokerDef("Green Joker",          4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +1.0), (T_ON_DISCARD, -1.0)),
             desc="+1 Mult per hand played, -1 Mult per discard"),

    # --- hand-type jokers. HAND_HAS means CONTAINS: a Full House fires Jolly.
    JokerDef("Jolly Joker",          3, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           8.0, cond=C_HAND_HAS, param=H_PAIR,
             desc="+8 Mult if played hand contains a Pair"),
    JokerDef("Zany Joker",           4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          12.0, cond=C_HAND_HAS, param=H_THREE,
             desc="+12 Mult if played hand contains a Three of a Kind"),
    JokerDef("Mad Joker",            4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          10.0, cond=C_HAND_HAS, param=H_TWO_PAIR,
             desc="+10 Mult if played hand contains a Two Pair"),
    JokerDef("Crazy Joker",          4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          12.0, cond=C_HAND_HAS, param=H_STRAIGHT,
             desc="+12 Mult if played hand contains a Straight"),
    JokerDef("Droll Joker",          4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          10.0, cond=C_HAND_HAS, param=H_FLUSH,
             desc="+10 Mult if played hand contains a Flush"),
    JokerDef("Sly Joker",            3, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,         50.0, cond=C_HAND_HAS, param=H_PAIR,
             desc="+50 Chips if played hand contains a Pair"),
    JokerDef("Wily Joker",           4, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,        100.0, cond=C_HAND_HAS, param=H_THREE,
             desc="+100 Chips if played hand contains a Three of a Kind"),
    JokerDef("Clever Joker",         4, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,         80.0, cond=C_HAND_HAS, param=H_TWO_PAIR,
             desc="+80 Chips if played hand contains a Two Pair"),
    JokerDef("Devious Joker",        4, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,        100.0, cond=C_HAND_HAS, param=H_STRAIGHT,
             desc="+100 Chips if played hand contains a Straight"),
    JokerDef("Crafty Joker",         4, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,         80.0, cond=C_HAND_HAS, param=H_FLUSH,
             desc="+80 Chips if played hand contains a Flush"),
    JokerDef("Half Joker",           5, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          20.0, cond=C_COUNT_CMP, param=3,
             desc="+20 Mult if played hand contains 3 or fewer cards"),

    # --- per-card jokers
    JokerDef("Lusty Joker",          5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=HEARTS,
             desc="Played cards with Heart suit give +3 Mult when scored"),
    JokerDef("Wrathful Joker",       5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=SPADES,
             desc="Played cards with Spade suit give +3 Mult when scored"),
    JokerDef("Gluttonous Joker",     5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=CLUBS,
             desc="Played cards with Club suit give +3 Mult when scored"),
    JokerDef("Even Steven",          4, COMMON,    T_PER_SCORED,        E_ADD_MULT,           4.0, cond=C_RANK_IN, param=_ranks("2 4 6 8 T"),
             desc="Played cards with even rank give +4 Mult when scored (10, 8, 6, 4, 2)"),
    JokerDef("Odd Todd",             4, COMMON,    T_PER_SCORED,        E_ADD_CHIPS,         31.0, cond=C_RANK_IN, param=_ranks("A 3 5 7 9"),
             desc="Played cards with odd rank give +31 Chips when scored (A, 9, 7, 5, 3)"),
    JokerDef("Scary Face",           4, COMMON,    T_PER_SCORED,        E_ADD_CHIPS,         30.0, cond=C_IS_FACE,
             desc="Played face cards give +30 Chips when scored"),
    JokerDef("Smiley Face",          4, COMMON,    T_PER_SCORED,        E_ADD_MULT,           5.0, cond=C_IS_FACE,
             desc="Played face cards give +5 Mult when scored"),
    JokerDef("Fibonacci",            8, UNCOMMON,  T_PER_SCORED,        E_ADD_MULT,           8.0, cond=C_RANK_IN, param=_ranks("A 2 3 5 8"),
             desc="Each played Ace, 2, 3, 5, or 8 gives +8 Mult when scored"),
    JokerDef("Arrowhead",            7, UNCOMMON,  T_PER_SCORED,        E_ADD_CHIPS,         50.0, cond=C_SUIT, param=SPADES,
             desc="Played cards with Spade suit give +50 Chips when scored"),
    JokerDef("Onyx Agate",           7, UNCOMMON,  T_PER_SCORED,        E_ADD_MULT,           7.0, cond=C_SUIT, param=CLUBS,
             desc="Played cards with Club suit give +7 Mult when scored"),

    # --- query jokers: v = VALUE * a count from live state
    JokerDef("Banner",               5, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,         30.0, source=S_QUERY, query=Q_DISCARDS_LEFT,
             desc="+30 Chips for each remaining discard"),
    JokerDef("Mystic Summit",        5, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          15.0, source=S_QUERY, query=Q_NO_DISCARDS_LEFT,
             desc="+15 Mult when 0 discards remaining"),
    JokerDef("Blue Joker",           5, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,          2.0, source=S_QUERY, query=Q_DECK_LEFT,
             desc="+2 Chips for each remaining card in deck"),
    JokerDef("Abstract Joker",       4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           3.0, source=S_QUERY, query=Q_JOKER_COUNT,
             desc="+3 Mult for each Joker card"),
    JokerDef("Bull",                 6, UNCOMMON,  T_ONCE_MAIN,         E_ADD_CHIPS,          2.0, source=S_QUERY, query=Q_MONEY,
             desc="+2 Chips for each $1 you have"),
    JokerDef("Joker Stencil",        8, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              1.0, source=S_QUERY, query=Q_EMPTY_JOKER_SLOTS,
             desc="X1 Mult for each empty Joker slot. Joker Stencil included"),

    # --- counter: +20 Mult, -4 at each end of round, eaten when it reaches 0
    JokerDef("Popcorn",              5, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          20.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_END_OF_ROUND, -4.0),), expire_at=0.0,
             desc="+20 Mult, -4 Mult per round played"),

    # --- economy: money at the end of the round, paid in the cashout
    JokerDef("Golden Joker",         6, COMMON,    T_END_OF_ROUND,      E_ADD_MONEY,          4.0, copyable=False,
             desc="Earn $4 at end of round"),
    JokerDef("Cloud 9",              7, UNCOMMON,  T_END_OF_ROUND,      E_ADD_MONEY,          1.0, source=S_QUERY, query=Q_DECK_NINE_COUNT,
             copyable=False, desc="Earn $1 for each 9 in your full deck at end of round"),

    # --- XMULT on a hand type: applied in slot order
    JokerDef("The Duo",              8, RARE,      T_ONCE_MAIN,         E_XMULT,              2.0, cond=C_HAND_HAS, param=H_PAIR,
             desc="X2 Mult if played hand contains a Pair"),
    JokerDef("The Trio",             8, RARE,      T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_HAND_HAS, param=H_THREE,
             desc="X3 Mult if played hand contains a Three of a Kind"),
    JokerDef("The Family",           8, RARE,      T_ONCE_MAIN,         E_XMULT,              4.0, cond=C_HAND_HAS, param=H_FOUR,
             desc="X4 Mult if played hand contains a Four of a Kind"),
    JokerDef("The Order",            8, RARE,      T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_HAND_HAS, param=H_STRAIGHT,
             desc="X3 Mult if played hand contains a Straight"),
    JokerDef("The Tribe",            8, RARE,      T_ONCE_MAIN,         E_XMULT,              2.0, cond=C_HAND_HAS, param=H_FLUSH,
             desc="X2 Mult if played hand contains a Flush"),

    # ======================================================================
    # GROUP A — counters, scaling, money and budget
    # ======================================================================
    JokerDef("Scholar",              4, COMMON,    T_PER_SCORED,        E_CHIPS_MULT,        20.0, value2=4.0, cond=C_RANK_IN, param=_ranks("A"),
             desc="Played Aces give +20 Chips and +4 Mult when scored"),
    JokerDef("Walkie Talkie",        4, COMMON,    T_PER_SCORED,        E_CHIPS_MULT,        10.0, value2=4.0, cond=C_RANK_IN, param=_ranks("T 4"),
             desc="Each played 10 or 4 gives +10 Chips and +4 Mult when scored"),
    JokerDef("Ride the Bus",         6, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +1.0, C_SCORING_FACE, 0, M_RESET_ELSE_ADD),),
             desc="This Joker gains +1 Mult per consecutive hand played without a scoring face card"),
    JokerDef("Spare Trousers",       6, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +2.0, C_HAND_HAS, H_TWO_PAIR),),
             desc="This Joker gains +2 Mult if played hand contains a Two Pair"),
    JokerDef("Square Joker",         4, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,          0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +4.0, C_COUNT_EQ, 4),),
             desc="This Joker gains +4 Chips if played hand has exactly 4 cards"),
    JokerDef("Runner",               5, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,          0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +15.0, C_HAND_HAS, H_STRAIGHT),),
             desc="Gains +15 Chips if played hand contains a Straight"),
    JokerDef("Ice Cream",            5, COMMON,    T_ONCE_MAIN,         E_ADD_CHIPS,        100.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_AFTER, -5.0),), expire_at=0.0,
             desc="+100 Chips, -5 Chips for every hand played"),
    JokerDef("Wee Joker",            8, RARE,      T_ONCE_MAIN,         E_ADD_CHIPS,          0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_PER_SCORED, +8.0, C_RANK_IN, _ranks("2")),),
             desc="This Joker gains +8 Chips when each played 2 is scored"),
    JokerDef("Castle",               6, UNCOMMON,  T_ONCE_MAIN,         E_ADD_CHIPS,          0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ON_DISCARD_CARD, +3.0, C_CASTLE_SUIT),),
             desc="This Joker gains +3 Chips per discarded {castle} card, suit changes every round"),
    JokerDef("Flash Card",           5, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           0.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ON_REROLL, +2.0),),
             desc="This Joker gains +2 Mult per reroll in the shop"),
    JokerDef("Swashbuckler",         4, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           1.0, source=S_QUERY, query=Q_OTHER_SELL,
             desc="Adds the sell value of all other owned Jokers to Mult"),
    JokerDef("Supernova",            5, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           1.0, source=S_QUERY, query=Q_HAND_PLAYED,
             desc="Adds the number of times poker hand has been played this run to Mult"),
    JokerDef("Card Sharp",           6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_HAND_REPEAT,
             desc="X3 Mult if played poker hand has already been played this round"),
    JokerDef("Obelisk",              8, RARE,      T_ONCE_MAIN,         E_XMULT,              1.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ONCE_MAIN, +0.2, C_MOST_PLAYED, 0, M_RESET_ELSE_ADD),),
             desc="This Joker gains X0.2 Mult per consecutive hand played without playing your most played poker hand"),
    JokerDef("Hit the Road",         8, RARE,      T_ONCE_MAIN,         E_XMULT,              1.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ON_DISCARD_CARD, +0.5, C_RANK_IN, _ranks("J")),
                      (T_END_OF_ROUND, 0.0, C_NONE, 0, M_RESET)),
             desc="This Joker gains X0.5 Mult for every Jack discarded this round"),
    JokerDef("Ramen",                6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              2.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ON_DISCARD_CARD, -0.01),), expire_at=1.0,
             desc="X2 Mult, loses X0.01 Mult per card discarded"),
    JokerDef("Campfire",             9, RARE,      T_ONCE_MAIN,         E_XMULT,              1.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_ON_SELL_OTHER, +0.25), (T_END_OF_ROUND, 0.0, C_BOSS, 0, M_RESET)),
             desc="This Joker gains X0.25 Mult for each card sold, resets when Boss Blind is defeated"),
    JokerDef("Acrobat",              6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_LAST_HAND,
             desc="X3 Mult on final hand of round"),
    JokerDef("Loyalty Card",         5, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              4.0, cond=C_LOYALTY, param=5, state=ST_LOYALTY,
             desc="X4 Mult every 6 hands played"),
    JokerDef("Seeing Double",        6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              2.0, cond=C_SEEING_DOUBLE,
             desc="X2 Mult if played hand has a scoring Club card and a scoring card of any other suit"),
    JokerDef("Flower Pot",           6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_FLOWER_POT,
             desc="X3 Mult if poker hand contains a Diamond card, Club card, Heart card, and Spade card"),
    JokerDef("Stuntman",             7, RARE,      T_ONCE_MAIN,         E_ADD_CHIPS,        250.0, h_size=-2,
             desc="+250 Chips, -2 hand size"),
    JokerDef("Erosion",              6, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           4.0, source=S_QUERY, query=Q_DECK_LOST,
             desc="+4 Mult for each card below the starting amount in your full deck"),
    JokerDef("Gros Michel",          5, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,          15.0, expire_odds=6, pool=P_NOT_EXTINCT,
             desc="+15 Mult, 1 in 6 chance this is destroyed at the end of round"),
    JokerDef("Cavendish",            4, COMMON,    T_ONCE_MAIN,         E_XMULT,              3.0, expire_odds=1000, pool=P_EXTINCT,
             desc="X3 Mult, 1 in 1000 chance this card is destroyed at the end of round"),
    JokerDef("Misprint",             4, COMMON,    T_ONCE_MAIN,         E_MULT_RANDOM,       23.0,
             desc="+0-23 Mult"),
    JokerDef("Baseball Card",        8, RARE,      T_ONCE_MAIN,         E_XMULT_UNCOMMON,     1.5,
             desc="Uncommon Jokers each give X1.5 Mult"),
    JokerDef("Blackboard",           6, UNCOMMON,  T_ONCE_MAIN,         E_XMULT,              3.0, cond=C_HELD_ALL_BLACK,
             desc="X3 Mult if all cards held in hand are Spades or Clubs"),
    JokerDef("To Do List",           4, COMMON,    T_BEFORE,            E_ADD_MONEY,          4.0, cond=C_TODO_HAND, state=ST_TODO,
             desc="Earn $4 if poker hand is a {todo}, poker hand changes at end of round"),
    JokerDef("Mail-In Rebate",       4, COMMON,    T_ON_DISCARD_CARD,   E_ADD_MONEY,          5.0, cond=C_MAIL_RANK,
             desc="Earn $5 for each discarded {mail}, rank changes every round"),
    JokerDef("Faceless Joker",       4, COMMON,    T_ON_DISCARD,        E_ADD_MONEY,          5.0, cond=C_FACES_GE, param=3,
             desc="Earn $5 if 3 or more face cards are discarded at the same time"),
    JokerDef("Delayed Gratification", 4, COMMON,   T_END_OF_ROUND,      E_ADD_MONEY,          2.0, source=S_QUERY, query=Q_DELAYED_GRAT,
             copyable=False, desc="Earn $2 per discard if no discards are used by end of the round"),
    JokerDef("Rocket",               6, UNCOMMON,  T_END_OF_ROUND,      E_ADD_MONEY,          1.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_END_OF_ROUND, +2.0, C_BOSS),), copyable=False,
             desc="Earn $1 at end of round. Payout increases by $2 when Boss Blind is defeated"),
    JokerDef("Egg",                  4, COMMON,    T_END_OF_ROUND,      E_SELL_VALUE,         3.0,
             desc="Gains $3 of sell value at end of round"),
    JokerDef("Gift Card",            6, UNCOMMON,  T_END_OF_ROUND,      E_SELL_VALUE_ALL,     1.0, copyable=False,
             desc="Add $1 of sell value to every Joker at end of round"),
    JokerDef("Credit Card",          1, COMMON,    T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Go up to -$20 in debt"),
    JokerDef("To the Moon",          5, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Earn an extra $1 of interest for every $5 you have at end of round"),
    JokerDef("Chaos the Clown",      4, COMMON,    T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="1 free Reroll per shop"),
    JokerDef("Juggler",              4, COMMON,    T_PASSIVE,           E_NONE,               0.0, h_size=1, copyable=False,
             desc="+1 hand size"),
    JokerDef("Drunkard",             4, COMMON,    T_PASSIVE,           E_NONE,               0.0, d_size=1, copyable=False,
             desc="+1 discard each round"),
    JokerDef("Merry Andy",           7, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, d_size=3, h_size=-1, copyable=False,
             desc="+3 discards each round, -1 hand size"),
    JokerDef("Troubadour",           6, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, h_size=2, h_plays=-1, copyable=False,
             desc="+2 hand size, -1 hand each round"),
    JokerDef("Turtle Bean",          6, UNCOMMON,  T_PASSIVE,           E_NONE,               5.0, source=S_COUNTER, state=ST_SCALE_COUNTER,
             updates=((T_END_OF_ROUND, -1.0),), expire_at=0.0, hsize_v=True, copyable=False,
             desc="+5 hand size, reduces by 1 each round"),
    JokerDef("Burglar",              6, UNCOMMON,  T_ON_BLIND_SELECT,   E_BURGLAR,            3.0,
             desc="When Blind is selected, gain +3 Hands and lose all discards"),

    # ======================================================================
    # GROUP B — held cards, random rolls, retriggers
    # ======================================================================
    JokerDef("Hanging Chad",         4, COMMON,    T_PER_SCORED,        E_RETRIGGER,          2.0, cond=C_POSITION, param=0,
             desc="Retrigger first played card used in scoring 2 additional times"),
    JokerDef("Dusk",                 5, UNCOMMON,  T_PER_SCORED,        E_RETRIGGER,          1.0, cond=C_LAST_HAND,
             desc="Retrigger all played cards in final hand of the round"),
    JokerDef("Hack",                 6, UNCOMMON,  T_PER_SCORED,        E_RETRIGGER,          1.0, cond=C_RANK_IN, param=_ranks("2 3 4 5"),
             desc="Retrigger each played 2, 3, 4, or 5"),
    JokerDef("Sock and Buskin",      6, UNCOMMON,  T_PER_SCORED,        E_RETRIGGER,          1.0, cond=C_IS_FACE,
             desc="Retrigger all played face cards"),
    JokerDef("Seltzer",              6, UNCOMMON,  T_PER_SCORED,        E_RETRIGGER,         10.0, value2=1.0, source=S_COUNTER,
             state=ST_SCALE_COUNTER, updates=((T_AFTER, -1.0),), expire_at=0.0,
             desc="Retrigger all cards played for the next 10 hands"),
    JokerDef("Mime",                 5, UNCOMMON,  T_PER_HELD,          E_RETRIGGER,          1.0,
             desc="Retrigger all card held in hand abilities"),
    JokerDef("Photograph",           5, COMMON,    T_PER_SCORED,        E_XMULT,              2.0, cond=C_FIRST_FACE,
             desc="First played face card gives X2 Mult when scored"),
    JokerDef("The Idol",             6, UNCOMMON,  T_PER_SCORED,        E_XMULT,              2.0, cond=C_RANK_SUIT,
             desc="Each played {idol} gives X2 Mult when scored, card changes every round"),
    JokerDef("Ancient Joker",        8, RARE,      T_PER_SCORED,        E_XMULT,              1.5, cond=C_ANCIENT_SUIT,
             desc="Each played card with {ancient} suit gives X1.5 Mult when scored, suit changes at end of round"),
    JokerDef("Bloodstone",           7, UNCOMMON,  T_PER_SCORED,        E_XMULT,              1.5, cond=C_SUIT, param=HEARTS, prob=2,
             desc="1 in 2 chance for played cards with Heart suit to give X1.5 Mult when scored"),
    JokerDef("Rough Gem",            7, UNCOMMON,  T_PER_SCORED,        E_ADD_MONEY,          1.0, cond=C_SUIT, param=DIAMONDS,
             desc="Played cards with Diamond suit earn $1 when scored"),
    JokerDef("Business Card",        4, COMMON,    T_PER_SCORED,        E_ADD_MONEY,          2.0, cond=C_IS_FACE, prob=2,
             desc="Played face cards have a 1 in 2 chance to give $2 when scored"),
    JokerDef("Triboulet",           20, LEGENDARY, T_PER_SCORED,        E_XMULT,              2.0, cond=C_RANK_IN, param=_ranks("K Q"),
             desc="Played Kings and Queens each give X2 Mult when scored"),
    JokerDef("Baron",                8, RARE,      T_PER_HELD,          E_XMULT,              1.5, cond=C_RANK_IN, param=_ranks("K"),
             desc="Each King held in hand gives X1.5 Mult"),
    JokerDef("Shoot the Moon",       5, COMMON,    T_PER_HELD,          E_ADD_MULT,          13.0, cond=C_RANK_IN, param=_ranks("Q"),
             desc="+13 Mult for each Queen held in hand"),
    JokerDef("Raised Fist",          5, COMMON,    T_PER_HELD,          E_MULT_NOMINAL,       2.0, cond=C_LOWEST_HELD,
             desc="Adds double the rank of lowest ranked card held in hand to Mult"),
    JokerDef("Reserved Parking",     6, COMMON,    T_PER_HELD,          E_ADD_MONEY,          1.0, cond=C_IS_FACE, prob=2,
             desc="Each face card held in hand has a 1 in 2 chance to give $1"),

    # ======================================================================
    # GROUP C — rule changes and copies (identified by name: rule_mods / copy_sources)
    # ======================================================================
    JokerDef("Four Fingers",         7, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="All Flushes and Straights can be made with 4 cards"),
    JokerDef("Shortcut",             7, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Allows Straights to be made with gaps of 1 rank (ex: 10 8 6 5 3)"),
    JokerDef("Smeared Joker",        7, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Hearts and Diamonds count as the same suit, Spades and Clubs count as the same suit"),
    JokerDef("Splash",               3, COMMON,    T_PASSIVE,           E_NONE,               0.0,
             desc="Every played card counts in scoring"),
    JokerDef("Pareidolia",           5, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="All cards are considered face cards"),
    JokerDef("Showman",              5, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Joker cards may appear multiple times"),
    JokerDef("Oops! All 6s",         4, UNCOMMON,  T_PASSIVE,           E_NONE,               0.0, copyable=False,
             desc="Doubles all listed probabilities (ex: 1 in 3 -> 2 in 3)"),
    JokerDef("Blueprint",           10, RARE,      T_PASSIVE,           E_NONE,               0.0,
             desc="Copies ability of Joker to the right"),
    JokerDef("Brainstorm",          10, RARE,      T_PASSIVE,           E_NONE,               0.0,
             desc="Copies the ability of leftmost Joker"),
]

# ---------------------------------------------------------------------------
# COMPILE — transpose the authored rows into columns. Validate first.
# ---------------------------------------------------------------------------
def _upd(u):
    """Pad one authored update rule to (trigger, delta, cond, param, mode)."""
    return tuple(u) + (C_NONE, 0, M_ADD)[len(u) - 2:]


def _validate(table):
    """Find transcription errors at import. Name the joker in the error."""
    seen = set()
    for d in table:
        if d.name in seen:
            raise ValueError(f"joker table: duplicate name {d.name!r}")
        seen.add(d.name)
        if not (COMMON <= d.rarity <= LEGENDARY):
            raise ValueError(f"{d.name}: rarity {d.rarity} outside 1..4")
        if d.cost < 0:
            raise ValueError(f"{d.name}: negative cost {d.cost}")
        # SOURCE and QUERY must agree. S_QUERY needs a query kind. No other
        # source may set one.
        if (d.source == S_QUERY) != (d.query != Q_NONE):
            raise ValueError(
                f"{d.name}: SOURCE/QUERY disagree — source={SOURCE_NAMES[d.source]}, "
                f"query={d.query}. S_QUERY needs a query kind; nothing else may set one.")
        if not d.desc:
            raise ValueError(f"{d.name}: no desc (the GUI shows it on the joker card)")
        if d.cond == C_NONE and d.param != 0:
            raise ValueError(f"{d.name}: COND_PARAM {d.param} set but condition is C_NONE")
        # Only a COUNTER reads joker_value, so only a COUNTER may have update
        # rules. A COUNTER without one never changes from its seed.
        if (d.source == S_COUNTER) != (len(d.updates) > 0):
            raise ValueError(
                f"{d.name}: source={SOURCE_NAMES[d.source]} with {len(d.updates)} "
                f"update rule(s). S_COUNTER needs at least one; nothing else may have one.")
        if len(d.updates) > N_UPDATE_RULES:
            raise ValueError(f"{d.name}: {len(d.updates)} update rules, max {N_UPDATE_RULES}")
        for u in d.updates:
            if not (2 <= len(u) <= 5):
                raise ValueError(f"{d.name}: update rule {u} needs 2..5 fields")
            t, _, c, _, m = _upd(u)
            if not (0 <= t < len(TRIGGER_NAMES)):
                raise ValueError(f"{d.name}: update trigger {t} is not a T_* event")
            if not (0 <= c < len(CONDITION_NAMES)) or m not in (M_ADD, M_RESET, M_RESET_ELSE_ADD):
                raise ValueError(f"{d.name}: update rule {u} has a bad cond or mode")
        if d.effect == E_RETRIGGER and d.source == S_COUNTER and d.value2 <= 0:
            raise ValueError(f"{d.name}: a COUNTER retrigger needs VALUE2 = its repetitions")


def _compile(table):
    """Convert row-wise JokerDefs to the column-wise Joker."""
    col = lambda f, dt: jnp.array([getattr(d, f) for d in table], dt)
    # Pad each row's update list to N_UPDATE_RULES with an unused rule.
    pad = lambda d: [_upd(u) for u in d.updates] + [(T_NONE, 0.0, C_NONE, 0, M_ADD)] * (N_UPDATE_RULES - len(d.updates))
    upd = lambda k, dt: jnp.array([[u[k] for u in pad(d)] for d in table], dt)
    return Joker(
        TRIGGER    = col("trigger", jnp.int32),
        CONDITION  = col("cond",    jnp.int32),
        COND_PARAM = col("param",   jnp.int32),
        EFFECT     = col("effect",  jnp.int32),
        SOURCE     = col("source",  jnp.int32),
        QUERY      = col("query",   jnp.int32),
        VALUE      = col("value",   jnp.float32),
        VALUE2     = col("value2",  jnp.float32),
        PROB       = col("prob",    jnp.int32),
        STATE      = col("state",   jnp.int32),
        COPYABLE   = col("copyable", jnp.bool_),
        UPD_TRIGGER= upd(0, jnp.int32),
        UPD_DELTA  = upd(1, jnp.float32),
        UPD_COND   = upd(2, jnp.int32),
        UPD_PARAM  = upd(3, jnp.int32),
        UPD_MODE   = upd(4, jnp.int32),
        EXPIRE_AT  = col("expire_at", jnp.float32),
        EXPIRE_ODDS= col("expire_odds", jnp.int32),
        POOL       = col("pool",    jnp.int32),
        H_SIZE     = col("h_size",  jnp.int32),
        D_SIZE     = col("d_size",  jnp.int32),
        H_PLAYS    = col("h_plays", jnp.int32),
        HSIZE_V    = col("hsize_v", jnp.bool_),
    )


_validate(TABLE)

JOKERS = _compile(TABLE)

# Index-aligned with JOKERS. Costs and rarities come from the game's P_CENTERS.
JOKER_QUERY  = jnp.array([d.query  for d in TABLE], jnp.int32)
JOKER_COST   = jnp.array([d.cost   for d in TABLE], jnp.int32)
JOKER_RARITY = jnp.array([d.rarity for d in TABLE], jnp.int32)

# ---------------------------------------------------------------------------
# Joker ids — table position. -1 = empty slot.
# ---------------------------------------------------------------------------
JOKER_NAMES = [d.name for d in TABLE]
JOKER_DESC  = [d.desc for d in TABLE]   # display only (gui.py)
JOKER_ID    = {name: i for i, name in enumerate(JOKER_NAMES)}
N_JOKERS    = len(TABLE)

J_EMPTY = -1
MAX_JOKERS = 32

# Lookup by name. A typo raises KeyError at import.
J_JOKER     = JOKER_ID["Joker"]
J_GREEDY    = JOKER_ID["Greedy Joker"]
J_BOOTSTRAP = JOKER_ID["Bootstraps"]
J_GREEN     = JOKER_ID["Green Joker"]
J_STENCIL   = JOKER_ID["Joker Stencil"]
J_BLUEPRINT = JOKER_ID["Blueprint"]
J_BRAINSTORM = JOKER_ID["Brainstorm"]
J_CHAOS     = JOKER_ID["Chaos the Clown"]
J_CREDIT    = JOKER_ID["Credit Card"]
J_MOON      = JOKER_ID["To the Moon"]


class RuleMods(NamedTuple):
    """Stage 0: passive jokers change the rules before hand detection runs.

    Detection reads these booleans, not the jokers array. This keeps scoring
    independent of EnvState.
    """
    four_fingers : jnp.ndarray   # bool — flushes and straights need only 4 cards
    shortcut     : jnp.ndarray   # bool — straights can have one-rank gaps
    smeared      : jnp.ndarray   # bool — hearts/diamonds and spades/clubs are one suit
    splash       : jnp.ndarray   # bool — every played card scores
    pareidolia   : jnp.ndarray   # bool — every card counts as a face card
    showman      : jnp.ndarray   # bool — the shop may offer held jokers and repeats
    odds         : jnp.ndarray   # float — probability numerator: 2 ** (Oops! All 6s held)

NO_MODS = RuleMods(*[jnp.bool_(False)] * 6, jnp.float32(1.0))   # default: no rule-mod jokers held

_MOD_NAMES = ["Four Fingers", "Shortcut", "Smeared Joker", "Splash", "Pareidolia", "Showman"]
_MOD_IDS = jnp.array([JOKER_ID[n] for n in _MOD_NAMES], jnp.int32)
_OOPS = JOKER_ID["Oops! All 6s"]


def rows_for(ids) -> Joker:
    """Gather JOKERS from id order into slot order.

    ids : (MAX_JOKERS,) int32  — joker id per held slot, J_EMPTY (-1) for vacant
    ->    Joker, fields (MAX_JOKERS,)  — the static rules for the held jokers

    A vacant slot must gather a harmless row. Clip the index, then let the
    caller's `active` mask (ids != J_EMPTY) zero it. Without the clip, -1 wraps
    to the last row, and the vacant slot scores as that joker.
    """
    safe = jnp.clip(ids, 0, N_JOKERS - 1)
    return jax.tree.map(lambda col: col[safe], JOKERS)


def resolve(rows: Joker, active, joker_value, ctx):
    """Produce the live value v per slot, before scoring. -> (MAX_JOKERS,) float32
        CONSTANT : v = VALUE
        COUNTER  : v = joker_value  (update() has already evolved it)
        QUERY    : v = VALUE * ctx[QUERY]

    ctx : (N_QUERY_KINDS,) float32 — query results, ctx[k] = count for kind k;
          built by env from live state (money, slots, deck) on each hand.
    """
    query_v = rows.VALUE * ctx[rows.QUERY]
    v = jnp.select(
        [rows.SOURCE == S_CONSTANT, rows.SOURCE == S_COUNTER, rows.SOURCE == S_QUERY],
        [rows.VALUE, joker_value, query_v],
        default=0.0)
    return jnp.where(active, v, 0.0)


def rule_mods(ids) -> RuleMods:
    """Derive the stage-0 RuleMods flags from a loadout (by joker id)."""
    held = (ids[:, None] == _MOD_IDS[None, :]).any(axis=0)
    n_oops = (ids == _OOPS).sum()
    return RuleMods(*[held[i] for i in range(len(_MOD_NAMES))],
                    jnp.power(2.0, n_oops).astype(jnp.float32))


def copy_sources(ids) -> jnp.ndarray:
    """(MAX_JOKERS,) int32 — the slot whose row each slot scores with.

    A plain joker scores with itself. Blueprint copies the joker to its right,
    Brainstorm the leftmost joker; a chain of copiers is followed to its end.
    -1 = the slot scores nothing (vacant, a copier with no target, a cycle, or
    a target the game does not let it copy).
    """
    n = ids.shape[0]
    slot = jnp.arange(n)
    copier = (ids == J_BLUEPRINT) | (ids == J_BRAINSTORM)
    step = lambda i: jnp.where(ids[i] == J_BRAINSTORM, 0, i + 1)
    tgt = jnp.where(copier, step(slot), slot)
    for _ in range(n):                            # follow chains; a cycle never settles
        t = jnp.clip(tgt, 0, n - 1)
        more = (tgt < n) & copier[t]
        tgt = jnp.where(more, step(t), tgt)
    t = jnp.clip(tgt, 0, n - 1)
    ok = ((tgt < n) & (ids[t] != J_EMPTY) & ~copier[t] & (t != slot)
          & JOKERS.COPYABLE[jnp.clip(ids[t], 0, N_JOKERS - 1)])
    return jnp.where(ids == J_EMPTY, -1, jnp.where(copier, jnp.where(ok, t, -1), slot))


def update_ex(rows: Joker, active, joker_value, joker_state, trigger, counts=None):
    """Update pass with the counts of each rule's condition.
    -> (joker_value, joker_state, fired (MAX_JOKERS,) bool)

    counts : (MAX_JOKERS, N_UPDATE_RULES) float — n per rule (see UPDATE RULES).
             None = 1 for every rule (an unconditional event).
    A slot whose rules do not listen to this trigger keeps its value exactly.
    """
    hit = (rows.UPD_TRIGGER == trigger) & active[:, None]
    n = jnp.ones(hit.shape, jnp.float32) if counts is None else counts
    v = joker_value
    for k in range(rows.UPD_TRIGGER.shape[1]):     # static, N_UPDATE_RULES, rule order
        mode, d, nk = rows.UPD_MODE[:, k], rows.UPD_DELTA[:, k], n[:, k]
        nv = jnp.select([mode == M_ADD, nk > 0],
                        [v + d * nk, rows.VALUE],                  # reset to the seed
                        default=jnp.where(mode == M_RESET_ELSE_ADD, v + d, v))
        v = jnp.where(hit[:, k], jnp.maximum(nv, 0.0), v)
    return v, joker_state, hit.any(axis=1)


def update(rows: Joker, active, joker_value, joker_state, trigger, ctx=None, counts=None):
    """Update pass for one event. -> (joker_value, joker_state). See update_ex."""
    v, s, _ = update_ex(rows, active, joker_value, joker_state, trigger, counts)
    return v, s


def expired(rows: Joker, active, joker_value, fired):
    """(MAX_JOKERS,) bool — slots an update has just used up (Ice Cream at 0 chips)."""
    return active & fired & (joker_value <= rows.EXPIRE_AT + 1e-4)


if __name__ == "__main__":
    print(f"{N_JOKERS} jokers")
    ids = jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32).at[:3].set(
        jnp.array([J_BLUEPRINT, J_JOKER, J_BRAINSTORM], jnp.int32))
    print("copy sources:", copy_sources(ids)[:4])    # want [1 1 1 -1]
