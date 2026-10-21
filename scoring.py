"""Poker hand scoring — detection, scoring mask, chips x mult, and the joker pipeline.

Encoding conventions (everything in the project honors these):
- rank: 0=two, 1=three, ... 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- a played hand is fixed shape (5,) + a valid mask (1-5 real cards)

The order is the order of Balatro's evaluate_play (state_events.lua):
  1. detect the hand (with the rule jokers: Four Fingers, Shortcut, Smeared, Splash)
  2. BEFORE: counters that read the hand (Green, Ride the Bus, Obelisk ...), To Do $
  3. each scored card in play order, each repetition: its chips, then every
     joker's per-card effect in slot order (chips, mult, $, then X mult)
  4. each card held in hand, lane order, each repetition (Mime)
  5. each joker's main effect in slot order, then Baseball Card on Uncommons
  6. AFTER: Ice Cream, Seltzer
Mult is affine in each step, mult -> (mult + m) * x, so a whole sequence of
steps folds into one product and one weighted sum (fold_mult).
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple

from jokers import (Joker, RuleMods, NO_MODS, rows_for, resolve, update_ex, copy_sources,
                    JOKER_RARITY, N_JOKERS, J_EMPTY, UNCOMMON, N_HAND_TYPES,
                    C_NONE, C_SUIT, C_RANK_IN, C_IS_FACE, C_HAND_HAS, C_COUNT_CMP, C_RANK_SUIT,
                    C_POSITION, C_FIRST_FACE, C_ANCIENT_SUIT, C_LOWEST_HELD, C_MAIL_RANK,
                    C_CASTLE_SUIT, C_FACES_GE, C_LAST_HAND, C_HAND_REPEAT, C_SEEING_DOUBLE,
                    C_FLOWER_POT, C_HELD_ALL_BLACK, C_LOYALTY, C_TODO_HAND, C_COUNT_EQ,
                    C_SCORING_FACE, C_MOST_PLAYED, C_BOSS, PER_CARD_CONDS,
                    T_PER_SCORED, T_PER_HELD, T_ONCE_MAIN, T_BEFORE, T_AFTER,
                    T_ON_DISCARD, T_ON_DISCARD_CARD,
                    E_ADD_MULT, E_ADD_CHIPS, E_XMULT, E_RETRIGGER, E_ADD_MONEY, E_CHIPS_MULT,
                    E_MULT_NOMINAL, E_MULT_RANDOM, E_XMULT_UNCOMMON, S_COUNTER,
                    Q_MONEY, Q_MONEY_DIV_5, Q_HAND_PLAYED, Q_OTHER_SELL, expired)

# which candidate mask each hand type uses: 0=all, 1=groups, 2=high (kept for tests)
MASK_CHOICE = jnp.array([0, 1, 0, 0, 0, 1, 1, 1, 2])

# index-aligned with HAND_NAMES (and the argmax flag array)
#                       SF  4oak  full  flush strt  3oak  2pair pair  high
BASE_CHIPS = jnp.array([100,  60,   40,   35,   30,   30,   20,   10,   5])
BASE_MULT  = jnp.array([  8,   7,    4,    4,    4,    3,    2,    2,   1])

BASE_CARD_CHIPS = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

HAND_NAMES = ["straight_flush", "four_kind", "full_house", "flush", "straight",
              "three_kind", "two_pair", "pair", "high_card"]

# CONTAINS[played, wanted] for a standard 5-card hand with no rule jokers. The
# pipeline does not read it: detect() computes "contains" from the cards, as
# the game does (a Four Fingers flush can also contain a Pair).
#                        SF 4oak full flush strt 3oak 2pair pair high
CONTAINS = jnp.array([[1,  0,   0,   1,   1,   0,   0,   0,   1],    # straight_flush
                      [0,  1,   0,   0,   0,   1,   0,   1,   1],    # four_kind
                      [0,  0,   1,   0,   0,   1,   1,   1,   1],    # full_house
                      [0,  0,   0,   1,   0,   0,   0,   0,   1],    # flush
                      [0,  0,   0,   0,   1,   0,   0,   0,   1],    # straight
                      [0,  0,   0,   0,   0,   1,   0,   1,   1],    # three_kind
                      [0,  0,   0,   0,   0,   0,   1,   1,   1],    # two_pair
                      [0,  0,   0,   0,   0,   0,   0,   1,   1],    # pair
                      [0,  0,   0,   0,   0,   0,   0,   0,   1]],   # high_card
                     bool)

MAX_REPS = 12        # repetitions of one scored card (1 + Hanging Chad, Sock, Hack, Dusk, Seltzer, copies)
MAX_HELD_REPS = 6    # repetitions of one held card (1 + Mime and its copies)


# ---------------------------------------------------------------------------
# CARD PREDICATES — Card:is_suit / Card:is_face
# ---------------------------------------------------------------------------
def _red(s):
    return (s == 1) | (s == 2)


def suit_is(card_suit, s, smeared):
    """Card:is_suit. Smeared Joker: hearts = diamonds, spades = clubs."""
    return (card_suit == s) | (smeared & (_red(card_suit) == _red(s)))


def face(rank, pareidolia):
    """Card:is_face. Pareidolia: every card."""
    return pareidolia | ((rank >= 9) & (rank <= 11))


# ---------------------------------------------------------------------------
# DETECTION — evaluate_poker_hand + get_poker_hand_info
# ---------------------------------------------------------------------------
class Hand(NamedTuple):
    handtype : jax.Array   # scalar int32 — index into HAND_NAMES
    contains : jax.Array   # (9,) bool — which hand types the played cards contain
    scoring  : jax.Array   # (5,) bool — which played cards score
    n_played : jax.Array   # scalar int32


def _straight(present, need, shortcut):
    """get_straight: walk the ranks A(low), 2 .. K, A. -> (is straight, ranks in it (13,))."""
    length = jnp.int32(0)
    skipped = jnp.bool_(False)
    straight = jnp.bool_(False)
    done = jnp.bool_(False)
    t = jnp.zeros(13, bool)
    for j in range(1, 15):                         # static: 14 steps
        r = 12 if j == 1 else j - 2                # the ace plays low at j = 1
        hit = present[r]
        skip = ~hit & shortcut & ~skipped & (j != 14)
        miss = ~hit & ~skip
        live = ~done
        t = jnp.where(live & hit, t.at[r].set(True), t)
        t = jnp.where(live & miss & ~straight, jnp.zeros(13, bool), t)
        new_len = jnp.where(hit, length + 1, jnp.where(skip, length, 0))
        new_skipped = jnp.where(hit, False, jnp.where(skip, True, False))
        done = done | (live & miss & straight)     # "if straight then break"
        length = jnp.where(live, new_len, length)
        skipped = jnp.where(live, new_skipped, skipped)
        straight = straight | (~done & (length >= need))
    return straight, t


def detect(rank, suit, valid, mods: RuleMods = NO_MODS) -> Hand:
    valid = valid.astype(bool)
    n = valid.sum().astype(jnp.int32)
    need = 5 - mods.four_fingers.astype(jnp.int32)
    size_ok = (n >= need) & (n <= 5)

    counts = (jax.nn.one_hot(rank, 13) * valid[:, None]).sum(axis=0)
    n2 = (counts == 2).sum()
    n3 = (counts == 3).sum()
    has4 = (counts >= 4).any()

    # get_flush: the first suit (Spades, Hearts, Clubs, Diamonds) with enough cards
    order = jnp.array([3, 2, 0, 1])
    in_suit = suit_is(suit[None, :], order[:, None], mods.smeared) & valid[None, :]   # (4, 5)
    enough = in_suit.sum(axis=1) >= need
    is_flush = size_ok & enough.any()
    flush_cards = in_suit[jnp.argmax(enough)] & is_flush

    present = counts > 0
    st, run = _straight(present, need, mods.shortcut)
    is_straight = size_ok & st
    straight_cards = valid & run[rank] & is_straight

    contains = jnp.stack([
        is_flush & is_straight,                    # straight flush
        has4,                                      # four of a kind
        (n3 >= 1) & (n2 >= 1),                     # full house
        is_flush,
        is_straight,
        (n3 >= 1) | has4,                          # three of a kind
        (n2 == 2) | ((n3 == 1) & (n2 == 1)),       # two pair
        (n2 >= 1) | (n3 >= 1) | has4,              # pair
        n >= 1,                                    # high card
    ])
    handtype = jnp.argmax(contains | (jnp.arange(9) == 8)).astype(jnp.int32)

    group = valid & (counts[rank] >= 2)
    top = jnp.argmax(jnp.where(valid, rank, -1))
    high = valid & (jnp.arange(rank.shape[0]) == top)
    scoring = jnp.select(
        [handtype == 0, handtype == 1, handtype == 2, handtype == 3, handtype == 4,
         handtype == 5, handtype == 8],
        [flush_cards | straight_cards, valid & (counts[rank] >= 4), group, flush_cards,
         straight_cards, valid & (counts[rank] == 3), high],
        default=valid & (counts[rank] == 2))       # two pair, pair
    scoring = jnp.where(mods.splash, valid, scoring)
    return Hand(handtype, contains, scoring, n)


@jit
def scoreHand(rank, suit, valid, mods: RuleMods = NO_MODS):
    """-> (handtype, scoring mask, base chips + scored card chips, base mult)."""
    h = detect(rank, suit, valid, mods)
    chips = (BASE_CARD_CHIPS[rank] * h.scoring).sum() + BASE_CHIPS[h.handtype]
    return h.handtype, h.scoring, chips, BASE_MULT[h.handtype]


# ---------------------------------------------------------------------------
# CONDITIONS — one evaluator for scoring and for the update rules
# ---------------------------------------------------------------------------
class Targets(NamedTuple):
    """The round's rotating targets (G.GAME.current_round)."""
    idol_card    : jax.Array   # card id 0-51
    mail_rank    : jax.Array   # rank 0-12
    ancient_suit : jax.Array   # suit 0-3
    castle_suit  : jax.Array   # suit 0-3


class Info(NamedTuple):
    """Everything a GATE condition reads. Built once per event."""
    handtype      : jax.Array
    contains      : jax.Array   # (9,)
    n_played      : jax.Array
    last_hand     : jax.Array   # bool: no hands left after this one
    hand_repeat   : jax.Array   # bool: played_this_round > 1 (Card Sharp)
    most_played   : jax.Array   # bool (Obelisk)
    scoring_face  : jax.Array   # bool (Ride the Bus)
    seeing_double : jax.Array
    flower_pot    : jax.Array
    held_black    : jax.Array
    faces         : jax.Array   # face cards in the discard (Faceless Joker)
    hands_total   : jax.Array   # hands finished before this one (Loyalty Card)
    is_boss       : jax.Array


def no_info(is_boss=False):
    """An Info with every gate closed (for events that are not a PLAY)."""
    z, f = jnp.int32(0), jnp.bool_(False)
    return Info(jnp.int32(8), jnp.zeros(9, bool), z, f, f, f, f, f, f, f, z, z,
                jnp.asarray(is_boss, bool))


class Cards(NamedTuple):
    """The cards a per-card condition tests. (K,) each."""
    rank       : jax.Array
    suit       : jax.Array
    ok         : jax.Array   # bool: a real card in this event
    nth        : jax.Array   # int: its ordinal among the scored cards (-1 if none)
    first_face : jax.Array   # bool: the first scored face card
    lowest     : jax.Array   # bool: the Raised Fist card


def gate(cond, param, info: Info, jstate) -> jax.Array:
    """GATE kinds. cond, param, jstate broadcast together. Per-card kinds -> False."""
    loyal = jnp.mod(param - 1 - (info.hands_total - jstate), param + 1) == param
    return jnp.select(
        [cond == C_NONE, cond == C_HAND_HAS, cond == C_COUNT_CMP, cond == C_COUNT_EQ,
         cond == C_FACES_GE, cond == C_LAST_HAND, cond == C_HAND_REPEAT,
         cond == C_SEEING_DOUBLE, cond == C_FLOWER_POT, cond == C_HELD_ALL_BLACK,
         cond == C_LOYALTY, cond == C_TODO_HAND, cond == C_SCORING_FACE,
         cond == C_MOST_PLAYED, cond == C_BOSS],
        [jnp.ones_like(cond, bool), info.contains[jnp.clip(param, 0, 8)],
         info.n_played <= param, info.n_played == param, info.faces >= param,
         jnp.broadcast_to(info.last_hand, cond.shape), jnp.broadcast_to(info.hand_repeat, cond.shape),
         jnp.broadcast_to(info.seeing_double, cond.shape), jnp.broadcast_to(info.flower_pot, cond.shape),
         jnp.broadcast_to(info.held_black, cond.shape), loyal, info.handtype == jstate,
         jnp.broadcast_to(info.scoring_face, cond.shape), jnp.broadcast_to(info.most_played, cond.shape),
         jnp.broadcast_to(info.is_boss, cond.shape)],
        default=jnp.zeros_like(cond, bool))


def match(cond, param, cards: Cards, info: Info, tg: Targets, mods: RuleMods, jstate):
    """(J, K) bool — does slot j's condition hold on card k? A gate applies to
    every real card. cond / param / jstate are (J,)."""
    c, p = cond[:, None], param[:, None]
    r, s = cards.rank[None, :], cards.suit[None, :]
    shape = (cond.shape[0], cards.rank.shape[0])
    b = lambda x: jnp.broadcast_to(x, shape)
    per = jnp.select(
        [c == C_SUIT, c == C_RANK_IN, c == C_IS_FACE, c == C_RANK_SUIT, c == C_POSITION,
         c == C_FIRST_FACE, c == C_ANCIENT_SUIT, c == C_LOWEST_HELD, c == C_MAIL_RANK,
         c == C_CASTLE_SUIT],
        [b(suit_is(s, p, mods.smeared)), b(((p >> r) & 1) != 0), b(face(r, mods.pareidolia)),
         b((r == tg.idol_card % 13) & suit_is(s, tg.idol_card // 13, mods.smeared)),
         b(cards.nth[None, :] == p), b(cards.first_face[None, :]),
         b(suit_is(s, tg.ancient_suit, mods.smeared)), b(cards.lowest[None, :]),
         b(r == tg.mail_rank), b(suit_is(s, tg.castle_suit, mods.smeared))],
        default=b(gate(cond, param, info, jstate)[:, None]))
    return per & cards.ok[None, :]


def rule_counts(rows: Joker, cards: Cards, weight, info, tg, mods, jstate, per_card):
    """(J, N_UPDATE_RULES) float — n for each update rule on this event.
    per_card: count the matching cards (weighted); else 1 when the gate is open."""
    out = []
    for k in range(rows.UPD_COND.shape[1]):
        if per_card:
            m = match(rows.UPD_COND[:, k], rows.UPD_PARAM[:, k], cards, info, tg, mods, jstate)
            out.append((m * weight[None, :]).sum(axis=1))
        else:
            out.append(gate(rows.UPD_COND[:, k], rows.UPD_PARAM[:, k], info, jstate).astype(jnp.float32))
    return jnp.stack(out, axis=1).astype(jnp.float32)


def fold_mult(mult0, x, b):
    """Apply mult -> mult * x[e] + b[e] for e in order. -> the final mult."""
    after = jnp.concatenate([jnp.cumprod(x[::-1])[::-1][1:], jnp.ones(1)])   # prod_{l>e} x_l
    return mult0 * jnp.prod(x) + (b * after).sum()


def _odds(rows: Joker, mods: RuleMods):
    """P(effect fires) per slot: 1, or G.GAME.probabilities.normal / PROB."""
    return jnp.where(rows.PROB > 0, jnp.minimum(mods.odds / jnp.maximum(rows.PROB, 1), 1.0), 1.0)


# ---------------------------------------------------------------------------
# THE PLAY PIPELINE
# ---------------------------------------------------------------------------
class PlayIn(NamedTuple):
    rank        : jax.Array   # (5,) played cards, lane order
    suit        : jax.Array
    valid       : jax.Array
    held_rank   : jax.Array   # (H,) cards held in hand, lane order
    held_suit   : jax.Array
    held_ok     : jax.Array
    ids         : jax.Array   # (J,) joker ids, slot order
    joker_value : jax.Array
    joker_state : jax.Array
    ctx         : jax.Array   # (N_QUERY_KINDS,) build_ctx before this hand
    money       : jax.Array
    hands_left  : jax.Array   # hands left AFTER this one
    hand_played : jax.Array   # (9,) before this hand
    hand_round  : jax.Array   # (9,) before this hand
    is_boss     : jax.Array
    targets     : Targets


class PlayOut(NamedTuple):
    handtype    : jax.Array
    mask        : jax.Array   # (5,) scored cards
    base_chips  : jax.Array   # hand base + scored card chips (one pass)
    base_mult   : jax.Array
    chips       : jax.Array   # float32 after every joker
    mult        : jax.Array
    score       : jax.Array   # int32 floor(chips * mult)
    money       : jax.Array   # $ earned during the hand (To Do, Rough Gem, Business ...)
    joker_value : jax.Array   # after BEFORE, the per-card counters and AFTER
    expired     : jax.Array   # (J,) bool — used up by AFTER (Ice Cream, Seltzer)
    hand_played : jax.Array   # (9,) after this hand
    hand_round  : jax.Array
    reps        : jax.Array   # (5,) repetitions of each played card (0 = not scored)
    rolls       : jax.Array   # (5, MAX_REPS, J) bool — the 1-in-N draws per scored card (tests)
    hrolls      : jax.Array   # (H, MAX_HELD_REPS, J) bool — per held card
    rnd         : jax.Array   # (J,) Misprint's draw per slot


def effective(ids, joker_value, joker_state):
    """The rows each slot SCORES with (copies resolved). -> (rows, active, value, state)."""
    src = copy_sources(ids)
    ok = src >= 0
    s = jnp.clip(src, 0, ids.shape[0] - 1)
    rows = rows_for(jnp.where(ok, ids[s], J_EMPTY))
    return rows, ok, joker_value[s], joker_state[s]


def play(p: PlayIn, mods: RuleMods, key, sell=None) -> PlayOut:
    """sell: (J,) sell values — Swashbuckler counts the OTHER jokers (total - its own)."""
    k_card, k_held, k_rand = jax.random.split(key, 3)
    ids = p.ids
    J = ids.shape[0]
    rows, active = rows_for(ids), ids != J_EMPTY
    erows, eok, _, estate = effective(ids, p.joker_value, p.joker_state)

    # 1. the hand
    hand = detect(p.rank, p.suit, p.valid, mods)
    ht = hand.handtype
    played = p.hand_played.at[ht].add(1)
    in_round = p.hand_round.at[ht].add(1)
    sc = hand.scoring
    isf = face(p.rank, mods.pareidolia)
    nth = jnp.where(sc, jnp.cumsum(sc) - 1, -1)
    ff = sc & isf
    first_face = ff & (jnp.cumsum(ff) == 1)

    # suits for Seeing Double / Flower Pot (scored cards, Smeared applies)
    has_s = [(sc & suit_is(p.suit, s, mods.smeared)).any() for s in range(4)]
    seeing_double = has_s[0] & (has_s[1] | has_s[2] | has_s[3])
    got = [jnp.bool_(False)] * 4                     # Flower Pot: H, D, S, C in that order
    for k in range(5):
        taken = jnp.bool_(False)
        for s in (2, 1, 3, 0):
            t = sc[k] & ~taken & ~got[s] & suit_is(p.suit[k], s, mods.smeared)
            got[s] = got[s] | t
            taken = taken | t
    flower = got[0] & got[1] & got[2] & got[3]
    held_black = ~(p.held_ok & ~((p.held_suit == 0) | (p.held_suit == 3))).any()
    others = jnp.where(jnp.arange(N_HAND_TYPES) == ht, -1, played)
    info = Info(ht, hand.contains, hand.n_played, p.hands_left <= 0, in_round[ht] > 1,
                (played[ht] > others).all(), (sc & isf).any(), seeing_double, flower,
                held_black, jnp.int32(0), p.hand_played.sum(), jnp.asarray(p.is_boss, bool))
    tg = p.targets
    scored = Cards(p.rank, p.suit, sc, nth, first_face, jnp.zeros(5, bool))

    # 2. BEFORE: counters (real rows only) and To Do List money (copies too)
    counts = rule_counts(rows, scored, sc.astype(jnp.float32), info, tg, mods, p.joker_state, False)
    value, _, _ = update_ex(rows, active, p.joker_value, p.joker_state, T_ONCE_MAIN, counts)
    before = eok & (erows.TRIGGER == T_BEFORE) & (erows.EFFECT == E_ADD_MONEY) \
        & gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    money = jnp.where(before, erows.VALUE, 0.0).sum()

    # 3. scored cards x repetitions x jokers
    m_card = match(erows.CONDITION, erows.COND_PARAM, scored, info, tg, mods, estate) & eok[:, None]
    on_card = erows.TRIGGER == T_PER_SCORED
    retrig = on_card & (erows.EFFECT == E_RETRIGGER)
    rep_n = jnp.where(erows.SOURCE == S_COUNTER, erows.VALUE2, erows.VALUE)
    reps = jnp.where(sc, 1 + (m_card * (retrig * rep_n)[:, None]).sum(axis=0), 0)
    reps = jnp.clip(reps, 0, MAX_REPS).astype(jnp.int32)                      # (5,)
    fx = on_card & ~retrig
    odds = _odds(erows, mods)
    roll = jax.random.uniform(k_card, (5, MAX_REPS, J)) < odds[None, None, :]
    live = (jnp.arange(MAX_REPS)[None, :] < reps[:, None])[:, :, None]       # (5, R, 1)
    on = live & (m_card & fx[:, None]).T[:, None, :] & roll                    # (5, R, J)
    c_j = jnp.where((erows.EFFECT == E_ADD_CHIPS) | (erows.EFFECT == E_CHIPS_MULT), erows.VALUE, 0.0)
    m_j = jnp.select([erows.EFFECT == E_ADD_MULT, erows.EFFECT == E_CHIPS_MULT],
                     [erows.VALUE, erows.VALUE2], default=0.0)
    x_j = jnp.where(erows.EFFECT == E_XMULT, erows.VALUE, 1.0)
    d_j = jnp.where(erows.EFFECT == E_ADD_MONEY, erows.VALUE, 0.0)
    card_chips = BASE_CARD_CHIPS[p.rank]
    base_chips = (card_chips * sc).sum() + BASE_CHIPS[ht]
    base_mult = BASE_MULT[ht].astype(jnp.float32)
    chips = BASE_CHIPS[ht] + (card_chips * reps).sum() + (on * c_j).sum()
    money = money + (on * d_j).sum()
    X = jnp.where(on, x_j, 1.0).ravel()
    mult = fold_mult(base_mult, X, (jnp.where(on, m_j, 0.0)).ravel() * X)

    # per-card counters (Wee Joker), per repetition
    counts = rule_counts(rows, scored, reps.astype(jnp.float32), info, tg, mods, p.joker_state, True)
    value, _, _ = update_ex(rows, active, value, p.joker_state, T_PER_SCORED, counts)

    # 4. cards held in hand x repetitions (Mime) x jokers
    H = p.held_rank.shape[0]
    lo = jnp.where(p.held_ok, p.held_rank, 99)
    lowest = p.held_ok & (jnp.arange(H) == (H - 1 - jnp.argmin(lo[::-1])))   # the last lowest
    held = Cards(p.held_rank, p.held_suit, p.held_ok, jnp.full(H, -1), jnp.zeros(H, bool), lowest)
    m_held = match(erows.CONDITION, erows.COND_PARAM, held, info, tg, mods, estate) & eok[:, None]
    on_held = erows.TRIGGER == T_PER_HELD
    mime = on_held & (erows.EFFECT == E_RETRIGGER)
    hfx = on_held & ~mime
    hroll = jax.random.uniform(k_held, (H, MAX_HELD_REPS, J)) < odds[None, None, :]
    first = (m_held & hfx[:, None]).T & hroll[:, 0, :]                          # (H, J)
    had = first.any(axis=1)
    hreps = 1 + had * (m_held * (mime * erows.VALUE)[:, None]).sum(axis=0)
    hreps = jnp.where(p.held_ok, jnp.clip(hreps, 1, MAX_HELD_REPS), 0).astype(jnp.int32)
    hlive = (jnp.arange(MAX_HELD_REPS)[None, :] < hreps[:, None])[:, :, None]
    hon = hlive & (m_held & hfx[:, None]).T[:, None, :] & hroll                # (H, Rh, J)
    hm = jnp.select([erows.EFFECT == E_ADD_MULT, erows.EFFECT == E_MULT_NOMINAL],
                    [jnp.broadcast_to(erows.VALUE, (H, J)),
                     erows.VALUE[None, :] * BASE_CARD_CHIPS[p.held_rank][:, None]], default=0.0)
    hm = jnp.broadcast_to(hm[:, None, :], hon.shape)
    money = money + (hon * d_j).sum()
    HX = jnp.where(hon, x_j, 1.0).ravel()
    mult = fold_mult(mult, HX, jnp.where(hon, hm, 0.0).ravel() * HX)

    # 5. main effects in slot order, money read live (Bull, Bootstraps)
    now = p.money + money
    ctx = (p.ctx.at[Q_MONEY].set(jnp.maximum(now, 0.0))
                .at[Q_MONEY_DIV_5].set(jnp.maximum(jnp.floor(now / 5), 0.0))
                .at[Q_HAND_PLAYED].set(played[ht]))
    src = copy_sources(ids)
    ev = value[jnp.clip(src, 0, J - 1)]
    v = resolve(erows, eok, ev, ctx)
    if sell is not None:
        own = sell[jnp.clip(src, 0, J - 1)]
        v = v - jnp.where(eok & (erows.QUERY == Q_OTHER_SELL), erows.VALUE * own, 0.0)
    main = eok & (erows.TRIGGER == T_ONCE_MAIN) & gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    rnd = jnp.floor(jax.random.uniform(k_rand, (J,)) * (erows.VALUE + 1))
    mc = jnp.where(main & (erows.EFFECT == E_ADD_CHIPS), v, 0.0)
    mm = jnp.select([main & (erows.EFFECT == E_ADD_MULT), main & (erows.EFFECT == E_MULT_RANDOM)],
                    [v, rnd], default=0.0)
    mx = jnp.where(main & (erows.EFFECT == E_XMULT), v, 1.0)
    base_x = jnp.where(main & (erows.EFFECT == E_XMULT_UNCOMMON), v, 1.0).prod()
    unc = active & (JOKER_RARITY[jnp.clip(ids, 0, N_JOKERS - 1)] == UNCOMMON)
    mx = mx * jnp.where(unc, base_x, 1.0)
    chips = chips + mc.sum()
    mult = fold_mult(mult, mx, mm * mx)

    # 6. AFTER (Ice Cream, Seltzer)
    value, _, fired = update_ex(rows, active, value, p.joker_state, T_AFTER)
    gone = expired(rows, active, value, fired)

    score = jnp.floor(chips * mult).astype(jnp.int32)
    return PlayOut(ht, sc, base_chips, base_mult, chips, mult, score, money, value, gone,
                   played, in_round, reps, roll, hroll, rnd)


class DiscardOut(NamedTuple):
    joker_value : jax.Array
    money       : jax.Array
    expired     : jax.Array


def discard(rank, suit, ok, ids, joker_value, joker_state, targets: Targets, mods: RuleMods):
    """The joker side of one DISCARD (cards in lane order)."""
    rows, active = rows_for(ids), ids != J_EMPTY
    erows, eok, _, estate = effective(ids, joker_value, joker_state)
    K = rank.shape[0]
    faces = (ok & face(rank, mods.pareidolia)).sum()
    info = no_info()._replace(faces=faces)
    cards = Cards(rank, suit, ok, jnp.full(K, -1), jnp.zeros(K, bool), jnp.zeros(K, bool))
    w = ok.astype(jnp.float32)
    counts = rule_counts(rows, cards, w, info, targets, mods, joker_state, True)
    value, _, fired = update_ex(rows, active, joker_value, joker_state, T_ON_DISCARD_CARD, counts)
    gone = expired(rows, active, value, fired)
    counts = rule_counts(rows, cards, w, info, targets, mods, joker_state, False)
    value, _, _ = update_ex(rows, active, value, joker_state, T_ON_DISCARD, counts)
    # money: Mail-In Rebate per card, Faceless Joker once (copies count)
    m = match(erows.CONDITION, erows.COND_PARAM, cards, info, targets, mods, estate) & eok[:, None]
    per = eok & (erows.TRIGGER == T_ON_DISCARD_CARD) & (erows.EFFECT == E_ADD_MONEY)
    once = eok & (erows.TRIGGER == T_ON_DISCARD) & (erows.EFFECT == E_ADD_MONEY) \
        & gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    money = (jnp.where(per, erows.VALUE, 0.0) * m.sum(axis=1)).sum() + jnp.where(once, erows.VALUE, 0.0).sum()
    return DiscardOut(value, money, gone)
