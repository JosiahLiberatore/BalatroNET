"""Jokers as DATA — axis enums, the static row table, the resolve pass.

A joker is a ROW over orthogonal axes, read by ONE generic evaluator. Adding a
joker = adding a row, never a branch in the scoring pipeline.

Import direction (acyclic — do not break):
    arrays.py / jokers.py    no project imports
    scoring.py               imports jokers
    env.py                   imports both

STATIC vs DYNAMIC, and note the INDEX — they are different axes:
    JOKERS[joker_id]     the rule. Compile-time constant, one row per joker that
                         EXISTS. JOKERS.VALUE is the update-rule PARAMETER.
    joker_value[slot]    (EnvState) the LIVE magnitude v, float32, seeded from
                         JOKERS.VALUE at acquire, written by the update pass.
    joker_state[slot]    (EnvState) live discrete control memory, int32.

Scoring ALWAYS reads joker_value and applies it by EFFECT, never branching on
where the number came from. SOURCE tells resolve() how to produce v; it never
reaches scoring. rows_for() bridges id order -> slot order.
"""

import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# AXIS 1 — TRIGGER : when the row fires
# ---------------------------------------------------------------------------
# The first three are the scoring stages, in pipeline order. That ORDER is what
# makes XMULT correct — you cannot collect per-joker then combine, because
# (4+4)*3 != (4*3)+4.
T_PER_SCORED      = 0    # stage 4 — per scored card, arrangement order
T_PER_HELD        = 1    # stage 5 — per held card, arrangement order
T_ONCE_MAIN       = 2    # stage 6 — once per hand, SLOT order
# out-of-scoring: these drive the update pass and the state-write family
T_END_OF_ROUND    = 3    # Golden Joker, *Mime*
T_ON_DISCARD      = 4    # Mail-In Rebate, Castle
T_ON_SHOP_ENTER   = 5    # Chaos the Clown — where per-shop resources re-arm
T_ON_SHOP_EXIT    = 6    # Perkeo — a genuine once-per-shop firing
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
# Two slots share this menu; the stage decides which is being asked:
#   GATE      joker on/off this hand, checked once   (Dusk, Half Joker)
#   PER-CARD  fires on WHICH card                    (Greedy, Photograph)
C_NONE        = 0    # always fires
C_SUIT        = 1    # param = suit 0-3
C_RANK_IN     = 2    # param = 13-bit rank mask
C_IS_FACE     = 3    # J/Q/K (Pareidolia widens this — a stage-0 rule-mod)
C_HAND_HAS    = 4    # param = handtype index into scoring.HAND_NAMES
C_COUNT_CMP   = 5    # param = threshold (Half Joker: cards_played <= 3)
C_RANK_SUIT   = 6    # target read from joker_state (The Idol, Mail-In Rebate)
C_POSITION    = 7    # param = ordinal. Photograph (first scored face, X2 ONCE)
                     # and Hanging Chad. A naive IS_FACE would fire Photograph on
                     # EVERY face -> X2 K * X2 Q = X4, and X16 under retrigger.

CONDITION_NAMES = ["NONE", "SUIT", "RANK_IN", "IS_FACE", "HAND_HAS",
                   "COUNT_CMP", "RANK_SUIT", "POSITION"]


# ---------------------------------------------------------------------------
# AXIS 3 — EFFECT : what the row writes. THREE families, one enum.
# ---------------------------------------------------------------------------
# family A — SCORING: mutate the (chips, mult) carry threaded by the scan
E_ADD_MULT          = 0
E_ADD_CHIPS         = 1
E_XMULT             = 2    # scan-threaded, not summed
E_RETRIGGER         = 3    # count = 1 + sum of retrigger amounts

# family B — STATE: a side-effect channel applied AFTER scoring
E_ADD_MONEY         = 4
E_TRANSFORM_CARD    = 5    # Midas -> gold
E_DESTROY_CARD      = 6    # Sixth Sense
E_SPAWN_CARD        = 7    # DNA (writes into the MAX_CARDS=128 headroom)
E_SPAWN_CONSUMABLE  = 8    # 8 Ball / Seance / Vagabond
E_SELL_VALUE        = 9    # Egg / Gift Card — NOT money

# family C — BUDGET: round scalars the DECK also sets. Pure functions of
# (deck_base, held jokers), recomputed CONTINUOUSLY — which is why selling Merry
# Andy restores hand size on the spot. SET does not commute with ADD (Drunkard +1
# then Burglar =0 -> 0; reverse -> 1), so they need a defined order.
E_ADD_HAND          = 10
E_ADD_DISCARD       = 11
E_SET_DISCARD       = 12
E_ADD_HANDSIZE      = 13
E_ADD_REROLL        = 14   # Chaos the Clown — the odd one out: free rerolls are
                           # CONSUMED, so "already spent it" is history the
                           # loadout cannot answer. Needs a live EnvState field,
                           # +VALUE on buy, floored at 0 on sell.

EFFECT_NAMES = ["ADD_MULT", "ADD_CHIPS", "XMULT", "RETRIGGER", "ADD_MONEY",
                "TRANSFORM_CARD", "DESTROY_CARD", "SPAWN_CARD",
                "SPAWN_CONSUMABLE", "SELL_VALUE", "ADD_HAND", "ADD_DISCARD",
                "SET_DISCARD", "ADD_HANDSIZE", "ADD_REROLL"]


# ---------------------------------------------------------------------------
# AXIS 4 — SOURCE : HOW the live value v is resolved. Read ONLY by resolve().
# ---------------------------------------------------------------------------
S_CONSTANT = 0    # reads nothing, returns VALUE                    (Joker, Duo)
S_COUNTER  = 1    # reads its own memory, evolved by update  (Green, Obelisk)
S_QUERY    = 2    # reads GLOBAL state, resolved FRESH each hand    (Bootstraps)

SOURCE_NAMES = ["CONSTANT", "COUNTER", "QUERY"]

# QUERY sub-menu — a selector into a fixed set of state-reductions, so QUERY
# stays pure data: v = VALUE * count(query_kind, state). resolve() switches over
# query_kind (bounded menu), NEVER over joker id. Grow the MENU.
Q_NONE              = 0
Q_MONEY_DIV_5       = 1    # Bootstraps
Q_EMPTY_JOKER_SLOTS = 2    # Joker Stencil
Q_DECK_STEEL_COUNT  = 3
Q_DECK_STONE_COUNT  = 4
Q_DECK_NINE_COUNT   = 5    # Cloud 9
Q_UNCOMMON_JOKERS   = 6    # Baseball Card

# A QUERY over slots/money/deck CHANGES on a mid-blind buy or sell, so resolve
# runs fresh each hand and is NEVER cached.


# ---------------------------------------------------------------------------
# AXIS 5 — STATE : what discrete control memory the row keeps (all fit an int)
# ---------------------------------------------------------------------------
ST_NONE             = 0
ST_SCALE_COUNTER    = 1    # Green, Ride the Bus, Seltzer countdown
ST_ROTATE_GLOBAL    = 2    # Ancient — uniform over 4 suits/round, deck-independent
ST_ROTATE_FROM_DECK = 3    # Idol (card id 0-51) / Mail-In (rank 0-12), sampled
                           # from values PRESENT in the deck, so the refresh must
                           # run AFTER that round's deck mutations.

STATE_NAMES = ["NONE", "SCALE_COUNTER", "ROTATE_GLOBAL", "ROTATE_FROM_DECK"]


# ---------------------------------------------------------------------------
# AXIS 6 — COPYABLE : may Blueprint / Brainstorm copy this row?
# ---------------------------------------------------------------------------
# A copier inherits its TARGET's trigger — Blueprint on Greedy fires PER_SCORED,
# on Golden fires END_OF_ROUND — so the copy re-runs the target's row in the
# copier's slot, and TRIGGER cannot double as the copyable answer. Not derivable
# from the other axes either (it correlates with the budget family but several
# copyable-looking rows simply are not), so it is declared per row.
# Default True; a row spells out copyable=False only for the exceptions.


class Joker(NamedTuple):
    """The row schema. Scalars = one row; (MAX_JOKERS,) arrays = a whole loadout."""
    TRIGGER    : jnp.ndarray   # int32   — T_*
    CONDITION  : jnp.ndarray   # int32   — C_*
    COND_PARAM : jnp.ndarray   # int32   — suit / rank mask / handtype / ordinal
    EFFECT     : jnp.ndarray   # int32   — E_*
    SOURCE     : jnp.ndarray   # int32   — S_*  (resolve reads this, scoring never does)
    VALUE      : jnp.ndarray   # float32 — seed / per-unit / increment size
    STATE      : jnp.ndarray   # int32   — ST_*, the KIND of memory, not the memory
    COPYABLE   : jnp.ndarray   # bool    — may Blueprint / Brainstorm copy this row


# ---------------------------------------------------------------------------
# AUTHORING RECORD — one joker per LINE. Compiled to columns below.
# ---------------------------------------------------------------------------
# SoA is right for JAX and wrong for humans, so author rows and transpose at
# import. Defaults carry the common case; only unusual axes get spelled out.
COMMON, UNCOMMON, RARE, LEGENDARY = 1, 2, 3, 4

# suits, for COND_PARAM readability (scoring.py: 0=clubs 1=diamonds 2=hearts 3=spades)
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
# Reorder freely: ids are meaningful only within one training run, nothing
# hardcodes one (J_* resolve through JOKER_ID by name), and we do not warm-start
# from old checkpoints. The six required axes are positional and aligned; widths
# are sized for the full 150-row import. These three are one per SOURCE kind, so
# resolve() has a live example of each branch.
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
    """Catch transcription errors at IMPORT, naming the offending joker."""
    seen = set()
    for d in table:
        if d.name in seen:
            raise ValueError(f"joker table: duplicate name {d.name!r}")
        seen.add(d.name)
        if not (COMMON <= d.rarity <= LEGENDARY):
            raise ValueError(f"{d.name}: rarity {d.rarity} outside 1..4")
        if d.cost < 0:
            raise ValueError(f"{d.name}: negative cost {d.cost}")
        # SOURCE and QUERY are one axis in two fields; disagreement means
        # resolve() reads a selector nobody set, or ignores one somebody did.
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

JOKERS = _compile(TABLE)

# index-aligned with JOKERS. Costs are the Cost column of jokers_checklist.xlsx.
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

# Looked up by name, so a typo is a KeyError at import, not a wrong joker later.
J_JOKER     = JOKER_ID["Joker"]
J_GREEDY    = JOKER_ID["Greedy Joker"]
J_BOOTSTRAP = JOKER_ID["Bootstraps"]


class RuleMods(NamedTuple):
    """Stage-0: passive jokers patch the RULES before detection runs.

    Detection needs these five booleans, not the jokers array — which is what
    lets scoring stay independent of EnvState.
    """
    four_fingers : jnp.ndarray   # bool — flushes/straights need only 4 cards
    shortcut     : jnp.ndarray   # bool — straights may have one-rank gaps
    smeared      : jnp.ndarray   # bool — hearts/diamonds and spades/clubs are one suit
    splash       : jnp.ndarray   # bool — every played card scores
    pareidolia   : jnp.ndarray   # bool — every card counts as a face card


NO_MODS = RuleMods(*[jnp.bool_(False)] * 5)   # v0.2 default: no rule-mods held


# ---------------------------------------------------------------------------
# STUBS — signatures and contracts pinned; bodies land with the joker pipeline.
# ---------------------------------------------------------------------------
def rows_for(ids):
    """Gather JOKERS from ID order into SLOT order. ids (MAX_JOKERS,) -> Joker.

    One `JOKERS.field[ids]` per field; what changes is the index axis:
    JOKERS.EFFECT[3] is "joker id 3", rows_for(ids).EFFECT[3] is "whatever is in
    SLOT 3". The scan walks slots, so it needs slot order. Output shape comes
    from ids, which is why N_JOKERS never reaches anything downstream.

    Vacant slots must gather a HARMLESS row: clamp the index, then let the
    caller's `active` mask (ids != J_EMPTY) zero it. A raw -1 clamps silently to
    the LAST row — that is how an empty slot starts scoring like Bootstraps.
    """

    
    raise NotImplementedError


def resolve(rows, query_kinds, active, ctx):
    """Produce the live value v per slot, BEFORE scoring. -> (MAX_JOKERS,) float32

        CONSTANT : v = VALUE
        COUNTER  : v = joker_value  (passthrough; update already evolved it)
        QUERY    : v = VALUE * count(query_kind, ctx)

    ctx carries the global reductions QUERY needs (money, free slots, deck
    counts). Fresh every hand, never cached — a mid-blind buy changes the answer.
    """
    raise NotImplementedError


def rule_mods(rows, ids, active):
    """Derive the stage-0 RuleMods flags from a loadout. -> RuleMods"""
    raise NotImplementedError


def update(rows, joker_value, joker_state, trigger, ctx):
    """UPDATE pass — the SECOND rule each row carries. -> (joker_value, joker_state)

    Genuinely (trigger, condition, delta, reset), a mini-row of its own: Green is
    +1/-1 across two triggers, and Ride the Bus / Obelisk RESET on a condition.
    """
    raise NotImplementedError
