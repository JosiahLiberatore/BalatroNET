"""Jokers as DATA — the axis enums, the static row table, the resolve pass.

The spine (NOTES "jokers-as-data"): a joker is a ROW over orthogonal axes, and
ONE generic evaluator reads the axes. Adding a joker = adding a row here, never
a code branch in the scoring pipeline.

IMPORT DIRECTION (keeps the graph acyclic — do not break it):

    jokers.py    enums + ROWS + resolve          imports NOTHING from the project
    scoring.py   detect / base / staged scan     imports jokers (enums, rows)
    env.py       state / step / shop / phase     imports both

That is why the Joker schema lives HERE and not in env.py: scoring must read the
axis enums without dragging in EnvState, or the 21-row table in tests.py stops
being cheap (every scoring test would have to build a full state pytree first).

TWO THINGS ARE EASY TO CONFLATE — they are not the same:

  STATIC  (this file)  ROWS[joker_id]  — the rule. Compile-time constant, one row
                                        per joker id that exists in the game.
                                        ROWS.VALUE is the update-rule PARAMETER
                                        (seed + increment size).
  DYNAMIC (EnvState)   joker_value[slot]  — the LIVE magnitude v, float32, seeded
                                        from ROWS.VALUE at acquire, read by
                                        scoring, written by the update pass.
                       joker_state[slot]  — live discrete CONTROL memory (ROTATE
                                        selector, Seltzer countdown), int32.

NOTES calls this the "VALUE + STATE unify": scoring ALWAYS reads joker_value and
applies it by EFFECT, with no branch on where the number came from. ROWS.SOURCE
tells the RESOLVE pass how to produce v; it never reaches scoring.
"""

import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# AXIS 1 — TRIGGER : when the row fires
# ---------------------------------------------------------------------------
# The first three are the scoring stages, and their ORDER is the stage order in
# the pipeline (NOTES: the ordered scan is what makes XMULT correct — you cannot
# collect per-joker then combine, because (4+4)*3 != (4*3)+4).
T_PER_SCORED      = 0    # stage 4 — per scored card, arrangement order
T_PER_HELD        = 1    # stage 5 — per held card, arrangement order
T_ONCE_MAIN       = 2    # stage 6 — once per hand, SLOT order
# out-of-scoring triggers: these drive the update pass and the state-write family
T_END_OF_ROUND    = 3    # Golden Joker, *Mime*
T_ON_DISCARD      = 4    # Mail-In Rebate, Castle
# the shop is TWO moments, not one, and different jokers want different ones.
# ENTER is where the per-shop resources reset (Chaos re-arms its free reroll);
# EXIT is a genuine once-per-shop firing (Perkeo). Collapsing them into a single
# ON_SHOP forces one of the two to fire at the wrong end of the visit.
T_ON_SHOP_ENTER   = 5    # Chaos the Clown
T_ON_SHOP_EXIT    = 6    # Perkeo
T_ON_BLIND_SELECT = 7    # the budget-write family lands here (Drunkard, Burglar)
T_ON_SOLD         = 8    # Campfire, Invisible
T_ON_PACK_OPEN    = 9    # Hallucination
T_ON_PACK_SKIP    = 10   # Red Card
T_ON_BOSS         = 11   # Chicot, Luchador, Matador

TRIGGER_NAMES = ["PER_SCORED", "PER_HELD", "ONCE_MAIN", "END_OF_ROUND",
                 "ON_DISCARD", "ON_SHOP_ENTER", "ON_SHOP_EXIT",
                 "ON_BLIND_SELECT", "ON_SOLD", "ON_PACK_OPEN", "ON_PACK_SKIP",
                 "ON_BOSS"]


# ---------------------------------------------------------------------------
# AXIS 2 — CONDITION : (kind, param). Which card / whether the joker fires.
# ---------------------------------------------------------------------------
# NOTES splits this into TWO slots that share this one menu:
#   GATE      — joker on/off this hand, checked once (Dusk=final_hand, Half Joker
#               = cards_played<=3). Shape: hand-level question.
#   PER-CARD  — fires on WHICH card (Greedy=Diamonds, Photograph=first face).
# Same kinds, different slot; the stage decides which one it is asking.
C_NONE        = 0    # always fires
C_SUIT        = 1    # param = suit 0-3
C_RANK_IN     = 2    # param = 13-bit rank mask
C_IS_FACE     = 3    # J/Q/K (Pareidolia is a stage-0 rule-mod that widens this)
C_HAND_HAS    = 4    # param = handtype index into scoring.HAND_NAMES
C_COUNT_CMP   = 5    # param = threshold (Half Joker: cards_played <= 3)
C_RANK_SUIT   = 6    # read the target from joker_state (The Idol, Mail-In Rebate)
C_POSITION    = 7    # param = ordinal. NOTES gap-fix: Photograph (first scored
                     # face, X2 ONCE) and Hanging Chad (retrigger first card).
                     # Without this a naive IS_FACE fires Photograph on EVERY
                     # face -> X2 K * X2 Q = X4, and X16 under retrigger.

CONDITION_NAMES = ["NONE", "SUIT", "RANK_IN", "IS_FACE", "HAND_HAS",
                   "COUNT_CMP", "RANK_SUIT", "POSITION"]


# ---------------------------------------------------------------------------
# AXIS 3 — EFFECT : what the row writes. THREE families, one enum.
# ---------------------------------------------------------------------------
# family A — SCORING writes: mutate the (chips, mult) carry threaded by the scan
E_ADD_MULT          = 0
E_ADD_CHIPS         = 1
E_XMULT             = 2    # must be first-class and scan-threaded, not summed
E_RETRIGGER         = 3    # own axis really: count = 1 + sum of retrigger amounts

# family B — STATE writes: thread a side-effect channel, applied AFTER scoring
E_ADD_MONEY         = 4
E_TRANSFORM_CARD    = 5    # Midas -> gold
E_DESTROY_CARD      = 6    # Sixth Sense
E_SPAWN_CARD        = 7    # DNA (writes into the MAX_CARDS=128 headroom)
E_SPAWN_CONSUMABLE  = 8    # 8 Ball / Seance / Vagabond
E_SELL_VALUE        = 9    # Egg / Gift Card — NOT money

# family C — BUDGET writes: the round scalars the DECK also sets.
# NOTES: budget is a PURE FUNCTION of (deck_base, held jokers), correct
# CONTINUOUSLY — which is exactly why selling Merry Andy mid-round restores hand
# size on the spot. SET does not commute with ADD (Drunkard +1 then Burglar =0
# -> 0; reverse -> 1), so these need a defined order, same lesson as XMULT.
E_ADD_HAND          = 10
E_ADD_DISCARD       = 11
E_SET_DISCARD       = 12
E_ADD_HANDSIZE      = 13
E_ADD_REROLL        = 14   # Chaos the Clown — the ODD ONE OUT of this family.
                           # Hands/discards/hand-size are pure functions of the
                           # held loadout, so they recompute. Free rerolls are
                           # CONSUMED, so "already spent it" is history the
                           # loadout cannot answer — this one needs a live
                           # EnvState field, +VALUE on buy, floored at 0 on sell.

EFFECT_NAMES = ["ADD_MULT", "ADD_CHIPS", "XMULT", "RETRIGGER", "ADD_MONEY",
                "TRANSFORM_CARD", "DESTROY_CARD", "SPAWN_CARD",
                "SPAWN_CONSUMABLE", "SELL_VALUE", "ADD_HAND", "ADD_DISCARD",
                "SET_DISCARD", "ADD_HANDSIZE", "ADD_REROLL"]


# ---------------------------------------------------------------------------
# AXIS 4 — SOURCE : HOW the live value v is resolved. Read ONLY by resolve().
# ---------------------------------------------------------------------------
# The three sources differ only in what resolve() READS. Scoring never sees this
# axis — it reads joker_value and stays agnostic.
S_CONSTANT = 0    # reads nothing, returns VALUE                    (Joker, Duo)
S_COUNTER  = 1    # reads the joker's own memory, evolved by update (Green, Obelisk)
S_QUERY    = 2    # reads GLOBAL state, resolved FRESH each hand    (Bootstraps)

SOURCE_NAMES = ["CONSTANT", "COUNTER", "QUERY"]

# QUERY sub-menu — a selector into a fixed set of state-reductions, so QUERY
# stays pure data: v = VALUE + PARAM * count(query_kind, state). The resolve pass
# switches over query_kind (bounded menu), NEVER over joker id. Grow the MENU.
Q_NONE              = 0
Q_MONEY_DIV_5       = 1    # Bootstraps
Q_EMPTY_JOKER_SLOTS = 2    # Joker Stencil
Q_DECK_STEEL_COUNT  = 3
Q_DECK_STONE_COUNT  = 4
Q_DECK_NINE_COUNT   = 5    # Cloud 9
Q_UNCOMMON_JOKERS   = 6    # Baseball Card

# MID-ROUND correctness: a QUERY over slots/money/deck CHANGES when you buy or
# sell mid-blind, so resolve runs fresh each hand and is NEVER cached. Cheap — a
# reduction over fixed-shape arrays.


# ---------------------------------------------------------------------------
# AXIS 5 — STATE : what discrete control memory the row keeps (all fit an int)
# ---------------------------------------------------------------------------
ST_NONE             = 0
ST_SCALE_COUNTER    = 1    # Green, Ride the Bus, Seltzer countdown
ST_ROTATE_GLOBAL    = 2    # Ancient — uniform over 4 suits/round, DECK-INDEPENDENT
ST_ROTATE_FROM_DECK = 3    # Idol (card id 0-51) / Mail-In (rank 0-12), sampled
                           # from values PRESENT in the deck -> the refresh READS
                           # mutable deck state and must run AFTER that round's
                           # deck mutations.

STATE_NAMES = ["NONE", "SCALE_COUNTER", "ROTATE_GLOBAL", "ROTATE_FROM_DECK"]


# ---------------------------------------------------------------------------
# AXIS 6 — COPYABLE : may Blueprint / Brainstorm copy this row?
# ---------------------------------------------------------------------------
# A copier does NOT have a trigger of its own — it inherits its TARGET's. A
# Blueprint pointed at Greedy Joker fires PER_SCORED; pointed at Golden Joker it
# fires END_OF_ROUND. So the copy is resolved by re-running the target's row in
# the copier's slot, at whatever stage that row belongs to, and TRIGGER cannot
# double as the "is this copyable" answer.
#
# Nor can the answer be DERIVED from the other axes. It correlates with the
# budget family (a loadout-derived effect has nothing to re-fire) but the
# correlation is not the rule — several rows that look perfectly copyable simply
# are not. So it is DECLARED per row: one bool, hand-set, no inference.
#
# Default True, because most jokers are copyable — a row spells out
# copyable=False only when it is one of the exceptions.


# ---------------------------------------------------------------------------
# The row schema
# ---------------------------------------------------------------------------
class Joker(NamedTuple):
    """One joker rule, as data.

    Struct-of-arrays, per the project rule: read as a single row when the fields
    are scalars (a table lookup), and as a whole loadout when they are
    (MAX_JOKERS,) arrays (the gathered rows for every occupied slot). Same class,
    different shape — nothing in the evaluator cares which it got.
    """
    TRIGGER    : jnp.ndarray   # int32   — T_*
    CONDITION  : jnp.ndarray   # int32   — C_*
    COND_PARAM : jnp.ndarray   # int32   — suit / rank mask / handtype / ordinal
    EFFECT     : jnp.ndarray   # int32   — E_*
    SOURCE     : jnp.ndarray   # int32   — S_*   (resolve reads this, scoring never does)
    VALUE      : jnp.ndarray   # float32 — seed / per-unit / increment size
    STATE      : jnp.ndarray   # int32   — ST_*  (More like State Type: KIND of memory, not the memory)
    COPYABLE   : jnp.ndarray   # bool    — may Blueprint / Brainstorm copy this row


# NOTE vs the version currently in env.py: two fields are added and one dtype
# changes. Both are forced by rows NOTES has ALREADY encoded in the xlsx —
# without them the table cannot express its own 65 confirmed rows:
#   COND_PARAM  the sheet writes conditions as (kind, param): "RANK_IN {A}",
#               "SUIT Diamonds". NOTES: 'CONDITION: (kind, param)'.
#   SOURCE      the sheet's final schema is (T, COND, EFFECT, SRC, STATE) and
#               writes "SRC=CONSTANT 2" — source and amount are one axis, two
#               fields. VALUE alone cannot distinguish Duo's flat 2.0 from
#               Bootstraps' 2.0-per-$5.
#   STATE       int32, not float — NOTES: 'discrete CONTROL memory only ...
#               All fit an int'. Magnitudes live in joker_value.


# ---------------------------------------------------------------------------
# AUTHORING RECORD — one joker per LINE. Compiled to columns below.
# ---------------------------------------------------------------------------
# Struct-of-arrays is right for JAX and wrong for humans: adding a joker to a
# column-major literal means editing ten parallel arrays and counting commas to
# keep them aligned, which is exactly how a Greedy row picks up Bootstraps'
# cost. So the TABLE is authored row-wise in plain Python and TRANSPOSED into
# the SoA form at import. Adding a joker = adding one line, and nothing
# downstream notices — ROWS is still columns.
#
# Defaults carry the common case (no condition, constant value, no memory), so
# a plain joker is a short line and only the unusual axes are spelled out.
COMMON, UNCOMMON, RARE, LEGENDARY = 1, 2, 3, 4

# suits, for COND_PARAM readability — the project-pinned convention
# (scoring.py: 0=clubs, 1=diamonds, 2=hearts, 3=spades)
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3


class JokerDef(NamedTuple):
    """One authored joker. Plain Python — never traced, never in a pytree."""
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
# THE TABLE — a joker is one line. Its id is its row index.
# ---------------------------------------------------------------------------
# Reorder freely. An id is a row index, meaningful only WITHIN one training run
# — reordering invalidates older checkpoints, and this project does not warm-
# start from them (if that is ever wanted, it comes from a curriculum, not from
# grafting weights). Nothing in the code hardcodes an id either: J_* look up
# through JOKER_ID by name, so a reorder either keeps working or fails loudly at
# import.
#
# The six required axes are positional and aligned; everything else is keyword
# and shows up only where it differs from the default. Widths are sized for the
# full 150-row import, so pouring in the rest of jokers_checklist.xlsx won't
# reflow the file.
#
# The three rows here are one per SOURCE kind, so resolve() has a live example
# of each branch.
#
#        name                     cost  rarity     trigger              effect                value    optional axes
TABLE = [
    JokerDef("Joker",                2, COMMON,    T_ONCE_MAIN,         E_ADD_MULT,           4.0),
    JokerDef("Greedy Joker",         5, COMMON,    T_PER_SCORED,        E_ADD_MULT,           3.0, cond=C_SUIT, param=DIAMONDS),
    JokerDef("Bootstraps",           7, UNCOMMON,  T_ONCE_MAIN,         E_ADD_MULT,           2.0, source=S_QUERY, query=Q_MONEY_DIV_5),
]


# ---------------------------------------------------------------------------
# COMPILE — transpose the authored rows into columns, validating on the way.
# ---------------------------------------------------------------------------
def _validate(table):
    """Catch transcription errors at IMPORT, pointing at the offending name.

    Worth the twenty lines: 62 more rows are getting hand-copied out of a
    spreadsheet, and a silently malformed row is a joker that quietly scores
    zero rather than one that crashes.
    """
    seen = set()
    for d in table:
        if d.name in seen:
            raise ValueError(f"joker table: duplicate name {d.name!r}")
        seen.add(d.name)
        if not (COMMON <= d.rarity <= LEGENDARY):
            raise ValueError(f"{d.name}: rarity {d.rarity} outside 1..4")
        if d.cost < 0:
            raise ValueError(f"{d.name}: negative cost {d.cost}")
        # SOURCE and QUERY are one axis in two fields — they must agree, or
        # resolve() reads a menu selector that was never set (or ignores one
        # that was).
        if (d.source == S_QUERY) != (d.query != Q_NONE):
            raise ValueError(
                f"{d.name}: SOURCE/QUERY disagree — source={SOURCE_NAMES[d.source]}, "
                f"query={d.query}. S_QUERY needs a query kind; nothing else may set one.")
        if d.cond == C_NONE and d.param != 0:
            raise ValueError(f"{d.name}: COND_PARAM {d.param} set but condition is C_NONE")


def _compile(table):
    """Row-wise JokerDefs -> the column-wise Joker the evaluator gathers from."""
    col = lambda f, dt: jnp.array([getattr(d, f) for d in table], dt)
    return Joker(
        TRIGGER    = col("trigger", jnp.int32),
        CONDITION  = col("cond",    jnp.int32),
        COND_PARAM = col("param",   jnp.int32),
        EFFECT     = col("effect",  jnp.int32),
        SOURCE     = col("source",  jnp.int32),
        VALUE      = col("value",   jnp.float32),
        STATE      = col("state",   jnp.int32),
        COPYABLE   = col("copyable", jnp.bool_),
    )


_validate(TABLE)

ROWS = _compile(TABLE)

# QUERY selector, index-aligned with ROWS. Q_NONE for every non-QUERY row.
ROW_QUERY = jnp.array([d.query for d in TABLE], jnp.int32)

# Shop economics — static per-joker data, so it lives with the rows.
# Costs are the Cost column of jokers_checklist.xlsx.
ROW_COST   = jnp.array([d.cost   for d in TABLE], jnp.int32)
ROW_RARITY = jnp.array([d.rarity for d in TABLE], jnp.int32)


# ---------------------------------------------------------------------------
# Joker ids — table position. -1 = empty slot.
# ---------------------------------------------------------------------------
JOKER_NAMES = [d.name for d in TABLE]
JOKER_ID    = {name: i for i, name in enumerate(JOKER_NAMES)}
N_JOKERS    = len(TABLE)

J_EMPTY = -1

# Constants only for the jokers CODE names directly (shop pool, tests). Looked
# up by name, not hand-numbered, so they cannot drift out of sync with TABLE —
# and a typo'd name is a KeyError at import instead of a wrong joker at runtime.
J_JOKER     = JOKER_ID["Joker"]
J_GREEDY    = JOKER_ID["Greedy Joker"]
J_BOOTSTRAP = JOKER_ID["Bootstraps"]


# ---------------------------------------------------------------------------
# Stage-0 RULE-MODS — passive jokers that patch the CONTEXT before scoring.
# ---------------------------------------------------------------------------
class RuleMods(NamedTuple):
    """The stage-0 patch to the rules, as a handful of flags.

    This is the answer to 'does detect() need the jokers array?' — no. Detection
    needs these booleans, which a separate pass derives FROM the jokers array.
    Keeping it a flag struct is what lets scoring stay independent of EnvState.
    """
    four_fingers : jnp.ndarray   # bool — flushes/straights need only 4 cards
    shortcut     : jnp.ndarray   # bool — straights may have one-rank gaps
    smeared      : jnp.ndarray   # bool — hearts/diamonds and spades/clubs are one suit
    splash       : jnp.ndarray   # bool — every played card scores
    pareidolia   : jnp.ndarray   # bool — every card counts as a face card


NO_MODS = RuleMods(*[jnp.bool_(False)] * 5)   # the v0.2 default: no rule-mods held


# ---------------------------------------------------------------------------
# STUBS — signatures and contracts pinned; bodies land with the joker pipeline.
# ---------------------------------------------------------------------------
def rows_for(ids):
    """Gather the static rows for a loadout.

    ids : (MAX_JOKERS,) int32, joker id per slot, J_EMPTY for vacant.
    ->    Joker whose fields are each (MAX_JOKERS,) — the whole loadout.

    Empty slots must gather a HARMLESS row, not garbage: clamp the index and let
    the caller's `active` mask zero the contribution. Never let a -1 index wrap
    around to the last row (JAX clamps out-of-bounds gathers silently, which is
    exactly how a vacant slot would start scoring like Bootstraps).
    """
    raise NotImplementedError


def resolve(rows, query_kinds, active, ctx):
    """RESOLVE pass — produce the live value v for every slot, BEFORE scoring.

    The one place the three SOURCE kinds differ; downstream everything reads v.

        CONSTANT : v = VALUE
        COUNTER  : v = joker_value  (already evolved by the update pass; passthrough)
        QUERY    : v = VALUE * count(query_kind, ctx)

    ctx carries the global reductions QUERY needs (money, free slots, deck
    counts). Runs fresh every hand — never cached, because a mid-blind buy or
    sell changes the answer.

    -> (MAX_JOKERS,) float32
    """
    raise NotImplementedError


def rule_mods(rows, ids, active):
    """Derive the stage-0 RuleMods flags from a loadout. -> RuleMods"""
    raise NotImplementedError


def update(rows, joker_value, joker_state, trigger, ctx):
    """UPDATE pass — the SECOND rule each joker carries, writing v and memory.

    NOTES is explicit that this is not free: the update rule is genuinely
    (trigger, condition, delta, reset), a real mini-row, because Green is +1/-1
    across two triggers and Ride the Bus / Obelisk RESET on a condition. What it
    buys is the removal of the per-joker hook that made scoring non-agnostic.

    -> (joker_value, joker_state)
    """
    raise NotImplementedError
