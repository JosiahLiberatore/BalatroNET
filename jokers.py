"""Jokers as data: axis enums, the static row table, and the resolve pass.

A joker is one row over independent axes. One generic evaluator reads the rows.
To add a joker, add a row. Do not add a branch to the scoring pipeline.

Import direction (acyclic):
    arrays.py / jokers.py    no project imports
    scoring.py               imports jokers
    env.py                   imports both

Static data and live data are different axes:
    JOKERS[joker_id]     the rule. One constant row per joker. JOKERS.VALUE is
                         the parameter of the update rule.
    joker_value[slot]    (EnvState) the live magnitude v, float32. Seeded from
                         JOKERS.VALUE on acquire. Written by the update pass.
    joker_state[slot]    (EnvState) live discrete control memory, int32.

Scoring reads joker_value and applies it by EFFECT. Scoring never reads SOURCE.
SOURCE tells resolve() how to produce v. rows_for() maps id order to slot order.
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
T_PER_SCORED      = 0    # stage 4 — per scored card, arrangement order
T_PER_HELD        = 1    # stage 5 — per held card, arrangement order
T_ONCE_MAIN       = 2    # stage 6 — once per hand, slot order
# The values below drive the update pass and the state-write effects.
T_END_OF_ROUND    = 3    # Golden Joker, Mime
T_ON_DISCARD      = 4    # Mail-In Rebate, Castle
T_ON_SHOP_ENTER   = 5    # Chaos the Clown — per-shop resources reset here
T_ON_SHOP_EXIT    = 6    # Perkeo
T_ON_BLIND_SELECT = 7    # budget-write effects (Drunkard, Burglar)
T_ON_SOLD         = 8    # Campfire, Invisible Joker
T_ON_PACK_OPEN    = 9    # Hallucination
T_ON_PACK_SKIP    = 10   # Red Card
T_ON_BOSS         = 11   # Chicot, Luchador, Matador

TRIGGER_NAMES = ["PER_SCORED", "PER_HELD", "ONCE_MAIN", "END_OF_ROUND",
                 "ON_DISCARD", "ON_SHOP_ENTER", "ON_SHOP_EXIT",
                 "ON_BLIND_SELECT", "ON_SOLD", "ON_PACK_OPEN", "ON_PACK_SKIP",
                 "ON_BOSS"]


# ---------------------------------------------------------------------------
# AXIS 2 — CONDITION : (kind, param). Which card, or whether the joker fires.
# ---------------------------------------------------------------------------
# The stage decides how the condition is read:
#   GATE      the joker is on or off for this hand, checked once (Dusk, Half Joker)
#   PER-CARD  the joker fires on a given card                    (Greedy, Photograph)
C_NONE        = 0    # always fires
C_SUIT        = 1    # param = suit 0-3
C_RANK_IN     = 2    # param = 13-bit rank mask
C_IS_FACE     = 3    # J/Q/K (Pareidolia widens this; see RuleMods)
C_HAND_HAS    = 4    # param = handtype index into scoring.HAND_NAMES
C_COUNT_CMP   = 5    # param = threshold (Half Joker: cards_played <= 3)
C_RANK_SUIT   = 6    # target read from joker_state (The Idol, Mail-In Rebate)
C_POSITION    = 7    # param = ordinal (Photograph, Hanging Chad). Photograph
                     # fires on the first scored face card only. C_IS_FACE
                     # fires on every face card, so it is wrong here.

CONDITION_NAMES = ["NONE", "SUIT", "RANK_IN", "IS_FACE", "HAND_HAS",
                   "COUNT_CMP", "RANK_SUIT", "POSITION"]


# ---------------------------------------------------------------------------
# AXIS 3 — EFFECT : what the row writes. Three families, one enum.
# ---------------------------------------------------------------------------
# Family A — SCORING: change the (chips, mult) carry of the scan.
E_ADD_MULT          = 0
E_ADD_CHIPS         = 1
E_XMULT             = 2    # applied in scan order, not summed
E_RETRIGGER         = 3    # count = 1 + sum of retrigger amounts

# Family B — STATE: a side effect applied after scoring.
E_ADD_MONEY         = 4
E_TRANSFORM_CARD    = 5    # Midas Mask: card becomes gold
E_DESTROY_CARD      = 6    # Sixth Sense
E_SPAWN_CARD        = 7    # DNA (writes into the MAX_CARDS=128 headroom)
E_SPAWN_CONSUMABLE  = 8    # 8 Ball, Seance, Vagabond
E_SELL_VALUE        = 9    # Egg, Gift Card — sell value, not money

# Family C — BUDGET: round scalars that the deck also sets. These are pure
# functions of (deck_base, held jokers), recomputed on every change. This is
# why the sale of Merry Andy restores hand size at once.
# SET and ADD do not commute (Drunkard +1 then Burglar =0 gives 0; the reverse
# gives 1), so the apply order must be defined.
E_ADD_HAND          = 10
E_ADD_DISCARD       = 11
E_SET_DISCARD       = 12
E_ADD_HANDSIZE      = 13
E_ADD_REROLL        = 14   # Chaos the Clown. Free rerolls are consumed, so
                           # this needs a live EnvState field: +VALUE on buy,
                           # floored at 0 on sell.

EFFECT_NAMES = ["ADD_MULT", "ADD_CHIPS", "XMULT", "RETRIGGER", "ADD_MONEY",
                "TRANSFORM_CARD", "DESTROY_CARD", "SPAWN_CARD",
                "SPAWN_CONSUMABLE", "SELL_VALUE", "ADD_HAND", "ADD_DISCARD",
                "SET_DISCARD", "ADD_HANDSIZE", "ADD_REROLL"]


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
Q_EMPTY_JOKER_SLOTS = 2    # Joker Stencil, Abstract Joker(?)
Q_DECK_STEEL_COUNT  = 3    # Steel Joker
Q_DECK_STONE_COUNT  = 4    # Stone Joker
Q_DECK_NINE_COUNT   = 5    # Cloud 9
Q_UNCOMMON_JOKERS   = 6    # Baseball Card
N_QUERY_KINDS       = 7    # length of ctx

# A buy or sell during a blind changes a query result. resolve() runs on each
# hand and its result is never cached.


# ---------------------------------------------------------------------------
# AXIS 5 — STATE : the kind of discrete control memory the row keeps (int)
# ---------------------------------------------------------------------------
ST_NONE             = 0
ST_SCALE_COUNTER    = 1    # Green Joker, Ride the Bus, Seltzer countdown
ST_ROTATE_GLOBAL    = 2    # Ancient Joker — uniform over 4 suits per round
ST_ROTATE_FROM_DECK = 3    # The Idol (card id 0-51), Mail-In Rebate (rank
                           # 0-12). Sampled from values present in the deck,
                           # so the refresh runs after the deck mutations of
                           # that round.

STATE_NAMES = ["NONE", "SCALE_COUNTER", "ROTATE_GLOBAL", "ROTATE_FROM_DECK"]


# ---------------------------------------------------------------------------
# AXIS 6 — COPYABLE : can Blueprint or Brainstorm copy this row?
# ---------------------------------------------------------------------------
# A copier runs the row of its target in the copier's slot, with the target's
# trigger. The other axes do not give this answer, so each row declares it.
# The default is True. A row sets copyable=False only for the exceptions.

# endregion

class Joker(NamedTuple):
    """The row schema. Scalars = one row. (MAX_JOKERS,) arrays = one loadout."""
    TRIGGER    : jnp.ndarray   # int32   — T_*
    CONDITION  : jnp.ndarray   # int32   — C_*
    COND_PARAM : jnp.ndarray   # int32   — suit / rank mask / handtype / ordinal
    EFFECT     : jnp.ndarray   # int32   — E_*
    SOURCE     : jnp.ndarray   # int32   — S_*  (read by resolve() only)
    QUERY      : jnp.ndarray   # int32   — Q_*
    VALUE      : jnp.ndarray   # float32 — seed / per-unit / increment size
    STATE      : jnp.ndarray   # int32   — ST_*, the kind of memory, not the memory
    COPYABLE   : jnp.ndarray   # bool    — Blueprint / Brainstorm can copy this row

# ---------------------------------------------------------------------------
# AUTHORING RECORD — one joker per line. Compiled to columns below.
# ---------------------------------------------------------------------------
# Rows are easy to write. Columns are what JAX needs. Write rows and transpose
# at import. Defaults cover the common case. Set only the unusual axes.
COMMON, UNCOMMON, RARE, LEGENDARY = 1, 2, 3, 4

# Suits, for COND_PARAM (scoring.py: 0=clubs 1=diamonds 2=hearts 3=spades).
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3

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

# ---------------------------------------------------------------------------
# THE TABLE — one joker per line. The id is the row index.
# ---------------------------------------------------------------------------
# Rows can be reordered. Ids are valid within one training run only. Nothing
# hardcodes an id: J_* constants resolve by name through JOKER_ID.
# These three rows give one example per SOURCE kind.
#
#        name                     cost  rarity     trigger              effect                value    optional axes
TABLE = [
    JokerDef("Joker",                2, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           4.0),
    JokerDef("Greedy Joker",         5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=DIAMONDS),
    JokerDef("Bootstraps",           7, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           2.0, source=S_QUERY, query=Q_MONEY_DIV_5),
]

# ---------------------------------------------------------------------------
# COMPILE — transpose the authored rows into columns. Validate first.
# ---------------------------------------------------------------------------
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
        if d.cond == C_NONE and d.param != 0:
            raise ValueError(f"{d.name}: COND_PARAM {d.param} set but condition is C_NONE")


def _compile(table):
    """Convert row-wise JokerDefs to the column-wise Joker."""
    col = lambda f, dt: jnp.array([getattr(d, f) for d in table], dt)
    return Joker(
        TRIGGER    = col("trigger", jnp.int32),
        CONDITION  = col("cond",    jnp.int32),
        COND_PARAM = col("param",   jnp.int32),
        EFFECT     = col("effect",  jnp.int32),
        SOURCE     = col("source",  jnp.int32),
        QUERY      = col("query",   jnp.int32),
        VALUE      = col("value",   jnp.float32),
        STATE      = col("state",   jnp.int32),
        COPYABLE   = col("copyable", jnp.bool_),
    )


_validate(TABLE)

JOKERS = _compile(TABLE)

# Index-aligned with JOKERS. Costs come from the Cost column of jokers_checklist.xlsx.
JOKER_QUERY  = jnp.array([d.query  for d in TABLE], jnp.int32)
JOKER_COST   = jnp.array([d.cost   for d in TABLE], jnp.int32)
JOKER_RARITY = jnp.array([d.rarity for d in TABLE], jnp.int32)

# ---------------------------------------------------------------------------
# Joker ids — table position. -1 = empty slot.
# ---------------------------------------------------------------------------
JOKER_NAMES = [d.name for d in TABLE]
JOKER_ID    = {name: i for i, name in enumerate(JOKER_NAMES)}
N_JOKERS    = len(TABLE)

J_EMPTY = -1
MAX_JOKERS = 32

# Lookup by name. A typo raises KeyError at import.
J_JOKER     = JOKER_ID["Joker"]
J_GREEDY    = JOKER_ID["Greedy Joker"]
J_BOOTSTRAP = JOKER_ID["Bootstraps"]

class RuleMods(NamedTuple):
    """Stage 0: passive jokers change the rules before hand detection runs.

    Detection reads these five booleans, not the jokers array. This keeps
    scoring independent of EnvState.
    """
    four_fingers : jnp.ndarray   # bool — flushes and straights need only 4 cards
    shortcut     : jnp.ndarray   # bool — straights can have one-rank gaps
    smeared      : jnp.ndarray   # bool — hearts/diamonds and spades/clubs are one suit
    splash       : jnp.ndarray   # bool — every played card scores
    pareidolia   : jnp.ndarray   # bool — every card counts as a face card

NO_MODS = RuleMods(*[jnp.bool_(False)] * 5)   # default: no rule-mod jokers held


# ---------------------------------------------------------------------------
# STUBS — signatures and contracts are fixed. Bodies come with the joker pipeline.
# ---------------------------------------------------------------------------
def rows_for(ids) -> Joker:
    """Gather JOKERS from id order into slot order.

    ids : (MAX_JOKERS,) int32  — joker id per held slot, J_EMPTY (-1) for vacant
    ->    Joker, fields (MAX_JOKERS,)  — the static rules for the held jokers

    JOKERS.EFFECT[3] is the effect of joker id 3. rows_for(ids).EFFECT[3] is
    the effect of the joker in slot 3. The scan walks slots, so it needs slot
    order. The output shape comes from ids, not from N_JOKERS.

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

    rows        : Joker, fields (MAX_JOKERS,)  — gathered static rules, slot order
                                                 (reads SOURCE / VALUE / QUERY only)
    active      : (MAX_JOKERS,) bool           — ids != J_EMPTY; zeroes the rows
                                                 that clip left in vacant slots
    joker_value : (MAX_JOKERS,) float32        — live COUNTER memory from EnvState,
                                                 written by update()
    ctx         : (N_QUERY_KINDS,) float32     — query results, ctx[k] = count for
                                                 kind k; built by env from live
                                                 state (money, slots, deck)

    ctx is built on each hand and never cached. A buy during a blind changes
    the result. The output is for this hand only. Do not write it to state.
    """
    query_v = rows.VALUE * ctx[rows.QUERY]

    v = jnp.select(
    [rows.SOURCE == S_CONSTANT,
    rows.SOURCE == S_COUNTER,
    rows.SOURCE == S_QUERY],
    [rows.VALUE,
    joker_value,
    query_v],
    default=0.0,)                    

    return jnp.where(active, v, 0.0)


def rule_mods(rows: Joker, ids, active):
    """Derive the stage-0 RuleMods flags from a loadout. -> RuleMods (5 scalar bools)

    rows   : Joker, fields (MAX_JOKERS,)  — gathered static rules, slot order
    ids    : (MAX_JOKERS,) int32          — joker id per slot (rule-mod jokers are
                                            identified by id, not by axis fields)
    active : (MAX_JOKERS,) bool           — ids != J_EMPTY; a vacant slot must
                                            not raise a flag
    """
    raise NotImplementedError


def update(rows: Joker, joker_value, joker_state, trigger, ctx):
    """Update pass — the second rule each row carries. -> (joker_value, joker_state)

    rows        : Joker, fields (MAX_JOKERS,)  — gathered static rules, slot order
    joker_value : (MAX_JOKERS,) float32        — live COUNTER memory, read-modify-
                                                 write (this pass owns it)
    joker_state : (MAX_JOKERS,) int32          — live discrete control memory
                                                 (ROTATE selector, countdowns)
    trigger     : scalar int32                 — the T_* event that fires now; a
                                                 slot updates only if its update
                                                 rule listens to this trigger
    ctx         : (N_QUERY_KINDS,) float32     — the same query results resolve()
                                                 gets (conditions can read it)

    The update rule is (trigger, condition, delta, reset). Green Joker is
    +1/-1 across two triggers. Ride the Bus and Obelisk reset on a condition.
    """

    raise NotImplementedError


if __name__ == "__main__":
    # Loadout: slot 0 = Joker, slot 1 = Bootstraps, slot 2 = vacant.
    ids    = jnp.array([J_JOKER, J_BOOTSTRAP, J_EMPTY], jnp.int32)
    active = ids != J_EMPTY                                  # [True, True, False]

    rows = rows_for(ids)

    # joker_value is seeded from JOKERS.VALUE on acquire. A vacant slot holds 0.
    joker_value = jnp.where(active, rows.VALUE, 0.0)         # [4.0, 2.0, 0.0]

    # ctx is built by env from live state. Here: money = 23, so ctx[Q_MONEY_DIV_5] = 4.
    ctx = jnp.zeros(N_QUERY_KINDS, jnp.float32).at[Q_MONEY_DIV_5].set(23 // 5)

    # Expected result:
    #   slot 0  CONSTANT  v = VALUE            = 4.0
    #   slot 1  QUERY     v = VALUE * ctx[Q]   = 2.0 * 4 = 8.0
    #   slot 2  vacant    v = 0.0
    v = resolve(rows, active, joker_value, ctx)
    print(v)

