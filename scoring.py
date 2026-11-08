"""Poker hand scoring — detection, scoring mask, chips x mult, and the joker pipeline.

Encoding conventions (everything in the project honors these):
- rank: 0=two, 1=three, ... 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
- a played hand is fixed shape (5,) + a valid mask (1-5 real cards)
- a card also has an enhancement, an edition, a seal and perma bonus chips
  (items.py ENH_ / ED_ / SEAL_); a Stone card has no rank and no suit

The order is the order of Balatro's evaluate_play (state_events.lua):
  1. detect the hand (with the rule jokers: Four Fingers, Shortcut, Smeared,
     Splash; Stone cards always score, Wild cards are every suit)
  2. BEFORE, slot order: counters that read the hand (Green, Ride the Bus, Obelisk ...),
     To Do $, Space Joker (level up), Midas Mask / Vampire, DNA. Ride the Bus and
     DNA see the cards as the Midas Masks / Vampires to their left left them. The boss
     checks a card they change again (set_ability: debuff_card).
     The hand's base chips and mult come from its level after this step.
  3. each scored card in play order, each repetition (Red Seal, then jokers):
     its chips (+Bonus, Stone, perma), its mult (Mult, Lucky), $ (Gold Seal,
     Lucky), X mult (Glass), its edition, then every joker's per-card effect in
     slot order (chips, mult, $, then X mult)
  4. each card held in hand, lane order, each repetition (Red Seal, Mime):
     Steel X1.5, then the jokers' held effects
  5. each joker in slot order: its edition's chips / mult, its main effect,
     Baseball Card on Uncommons, then its Polychrome X1.5
  6. AFTER: Ice Cream, Seltzer; then each scored Glass card breaks 1 in 4
Mult is affine in each step, mult -> (mult + m) * x, so a whole sequence of
steps folds into one product and one weighted sum (fold_mult).

A Boss Blind changes it (env.py works out which and passes it in PlayIn): a
debuffed card scores nothing and is no face card (but it still makes the hand);
a debuffed hand (The Psychic, The Eye, The Mouth) scores 0, and only Matador and
the AFTER step act; The Flint halves the base chips and mult. Observatory: each
held Planet of the hand is X1.5 after the jokers.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple

from items import (ENH_NONE, ENH_BONUS, ENH_MULT, ENH_WILD, ENH_GLASS, ENH_STEEL, ENH_STONE,
                   ENH_GOLD, ENH_LUCKY, ED_FOIL, ED_HOLO, ED_POLY, SEAL_GOLD, SEAL_RED,
                   BONUS_CHIPS, MULT_MULT, GLASS_X, GLASS_ODDS, STEEL_X, STONE_CHIPS,
                   LUCKY_MULT, LUCKY_MULT_ODDS, LUCKY_DOLLARS, LUCKY_DOLLAR_ODDS,
                   FOIL_CHIPS, HOLO_MULT, POLY_X, GOLD_SEAL_DOLLARS, HAND_NAMES, N_HAND_TYPES,
                   hand_chips_mult)
from jokers import (Joker, RuleMods, NO_MODS, rows_for, resolve, update_ex, copy_sources,
                    JOKER_RARITY, N_JOKERS, J_EMPTY, UNCOMMON,
                    C_NONE, C_SUIT, C_RANK_IN, C_IS_FACE, C_HAND_HAS, C_COUNT_CMP, C_RANK_SUIT,
                    C_POSITION, C_FIRST_FACE, C_ANCIENT_SUIT, C_LOWEST_HELD, C_MAIL_RANK,
                    C_CASTLE_SUIT, C_FACES_GE, C_LAST_HAND, C_HAND_REPEAT, C_SEEING_DOUBLE,
                    C_FLOWER_POT, C_HELD_ALL_BLACK, C_LOYALTY, C_TODO_HAND, C_COUNT_EQ,
                    C_SCORING_FACE, C_MOST_PLAYED, C_BOSS, C_ENH, C_LUCKY_HIT, C_ACE_STRAIGHT,
                    C_MONEY_LE, C_NOT_BOSS, C_BOSS_TRIGGERED, PER_CARD_CONDS,
                    T_PER_SCORED, T_PER_HELD, T_ONCE_MAIN, T_BEFORE, T_AFTER,
                    T_ON_DISCARD, T_ON_DISCARD_CARD,
                    E_ADD_MULT, E_ADD_CHIPS, E_XMULT, E_RETRIGGER, E_ADD_MONEY, E_CHIPS_MULT,
                    E_MULT_NOMINAL, E_MULT_RANDOM, E_XMULT_UNCOMMON, E_PERMA_CHIPS,
                    E_SPAWN_CONSUMABLE, E_LEVEL_UP, E_MIDAS, J_VAMPIRE, S_COUNTER,
                    E_SPAWN_SPECTRAL, E_SPAWN_CARD, T_ON_CARD_ADDED, Q_MONEY, Q_MONEY_DIV_5,
                    Q_HAND_PLAYED, Q_OTHER_SELL, Q_DECK_LOST, expired)
from arrays import compact

BASE_CARD_CHIPS = jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

# the hand types, best first (items.HAND_NAMES). Index constants for this file.
FLUSH_FIVE, FLUSH_HOUSE, FIVE_KIND, STRAIGHT_FLUSH, FOUR_KIND, FULL_HOUSE = range(6)
FLUSH, STRAIGHT, THREE_KIND, TWO_PAIR, PAIR, HIGH_CARD = range(6, 12)

MAX_REPS = 12        # repetitions of one scored card (1 + Red Seal, Hanging Chad, Sock, Hack, Dusk, Seltzer, copies)
MAX_HELD_REPS = 6    # repetitions of one held card (1 + Red Seal, Mime and its copies)
SCORE_CAP = 2147483520.0   # the largest float32 below 2^31: a hand's score saturates here (int32)


# ---------------------------------------------------------------------------
# CARD PREDICATES — Card:is_suit / Card:is_face / Card:get_id
# ---------------------------------------------------------------------------
def _red(s):
    return (s == 1) | (s == 2)


def suit_is(card_suit, s, smeared, enh=ENH_NONE):
    """Card:is_suit. Stone: no suit. Wild: every suit. Smeared Joker: hearts =
    diamonds, spades = clubs."""
    plain = (card_suit == s) | (smeared & (_red(card_suit) == _red(s)))
    return (enh != ENH_STONE) & ((enh == ENH_WILD) | plain)


def face(rank, pareidolia, enh=ENH_NONE):
    """Card:is_face. Pareidolia: every card (a Stone card too). A Stone card has no rank."""
    return pareidolia | ((enh != ENH_STONE) & (rank >= 9) & (rank <= 11))


def has_rank(enh):
    """Card:get_id: a Stone card has a random id, so it never matches a rank."""
    return enh != ENH_STONE


def card_chips(rank, enh, bonus):
    """Card:get_chip_bonus: the rank's chips + Bonus +30 (a Stone card: 50 only) + perma."""
    return jnp.where(enh == ENH_STONE, STONE_CHIPS, BASE_CARD_CHIPS[rank]
                     + jnp.where(enh == ENH_BONUS, BONUS_CHIPS, 0)) + bonus


# ---------------------------------------------------------------------------
# DETECTION — evaluate_poker_hand + get_poker_hand_info
# ---------------------------------------------------------------------------
class Hand(NamedTuple):
    handtype : jax.Array   # scalar int32 — index into HAND_NAMES
    contains : jax.Array   # (N_HAND_TYPES,) bool — which hand types the played cards contain
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


def detect(rank, suit, valid, mods: RuleMods = NO_MODS, enh=None) -> Hand:
    valid = valid.astype(bool)
    enh = jnp.zeros_like(rank) if enh is None else enh
    n = valid.sum().astype(jnp.int32)
    need = 5 - mods.four_fingers.astype(jnp.int32)
    size_ok = (n >= need) & (n <= 5)
    ranked = valid & has_rank(enh)                 # a Stone card makes no pair and no straight

    counts = (jax.nn.one_hot(rank, 13) * ranked[:, None]).sum(axis=0)
    n2 = (counts == 2).sum()
    n3 = (counts == 3).sum()
    has4 = (counts >= 4).any()
    has5 = (counts >= 5).any()

    # get_flush: the first suit (Spades, Hearts, Clubs, Diamonds) with enough cards
    order = jnp.array([3, 2, 0, 1])
    in_suit = suit_is(suit[None, :], order[:, None], mods.smeared, enh[None, :]) & valid[None, :]
    enough = in_suit.sum(axis=1) >= need
    is_flush = size_ok & enough.any()
    flush_cards = in_suit[jnp.argmax(enough)] & is_flush

    present = counts > 0
    st, run = _straight(present, need, mods.shortcut)
    is_straight = size_ok & st
    straight_cards = ranked & run[rank] & is_straight

    full = (n3 >= 1) & (n2 >= 1)
    contains = jnp.stack([
        has5 & is_flush,                           # flush five
        full & is_flush,                           # flush house
        has5,                                      # five of a kind
        is_flush & is_straight,                    # straight flush
        has4,                                      # four of a kind (five contains it)
        full,                                      # full house
        is_flush,
        is_straight,
        (n3 >= 1) | has4,                          # three of a kind
        (n2 == 2) | ((n3 == 1) & (n2 == 1)),       # two pair
        (n2 >= 1) | (n3 >= 1) | has4,              # pair
        n >= 1,                                    # high card
    ])
    handtype = jnp.argmax(contains | (jnp.arange(N_HAND_TYPES) == HIGH_CARD)).astype(jnp.int32)

    cnt = counts[rank] * ranked
    group = ranked & (cnt >= 2)
    top = jnp.argmax(jnp.where(ranked, rank, -1))
    high = valid & (jnp.arange(rank.shape[0]) == top)
    scoring = jnp.select(
        [handtype == FLUSH_FIVE, handtype == FLUSH_HOUSE, handtype == FIVE_KIND,
         handtype == STRAIGHT_FLUSH, handtype == FOUR_KIND, handtype == FULL_HOUSE,
         handtype == FLUSH, handtype == STRAIGHT, handtype == THREE_KIND, handtype == HIGH_CARD],
        [group | flush_cards, group | flush_cards, cnt >= 5,
         flush_cards | straight_cards, cnt >= 4, group,
         flush_cards, straight_cards, cnt == 3, high],
        default=cnt == 2)                          # two pair, pair
    scoring = scoring | (valid & (enh == ENH_STONE))   # the "pure" cards: a Stone card always scores
    scoring = jnp.where(mods.splash, valid, scoring)
    return Hand(handtype, contains, scoring, n)


@jit
def scoreHand(rank, suit, valid, mods: RuleMods = NO_MODS):
    """-> (handtype, scoring mask, base chips + scored card chips, base mult) at level 1."""
    h = detect(rank, suit, valid, mods)
    chips0, mult0 = hand_chips_mult(jnp.ones(N_HAND_TYPES, jnp.int32))
    chips = (BASE_CARD_CHIPS[rank] * h.scoring).sum() + chips0[h.handtype]
    return h.handtype, h.scoring, chips, mult0[h.handtype]


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
    contains      : jax.Array   # (N_HAND_TYPES,)
    n_played      : jax.Array
    last_hand     : jax.Array   # bool: 0 hands left after this one (Dusk, Acrobat: hands_left == 0)
    hand_repeat   : jax.Array   # bool: played_this_round > 1 (Card Sharp)
    most_played   : jax.Array   # bool (Obelisk)
    scoring_face  : jax.Array   # bool (Ride the Bus)
    seeing_double : jax.Array
    flower_pot    : jax.Array
    held_black    : jax.Array
    faces         : jax.Array   # face cards in the discard (Faceless Joker)
    hands_total   : jax.Array   # hands finished before this one (Loyalty Card)
    is_boss       : jax.Array
    ace_straight  : jax.Array = jnp.bool_(False)   # a scored Ace in a hand with a Straight (Superposition)
    money         : jax.Array = jnp.float32(0)     # $ before this hand (Vagabond)
    boss_triggered : jax.Array = jnp.bool_(False)  # the hand triggered the boss (Matador)


def no_info(is_boss=False):
    """An Info with every gate closed (for events that are not a PLAY)."""
    z, f = jnp.int32(0), jnp.bool_(False)
    return Info(jnp.int32(HIGH_CARD), jnp.zeros(N_HAND_TYPES, bool), z, f, f, f, f, f, f, f, z, z,
                jnp.asarray(is_boss, bool), f, jnp.float32(1e9))


class Cards(NamedTuple):
    """The cards a per-card condition tests. (K,) each."""
    rank       : jax.Array
    suit       : jax.Array
    ok         : jax.Array   # bool: a real card in this event
    nth        : jax.Array   # int: its ordinal among the scored cards (-1 if none)
    first_face : jax.Array   # bool: the first scored face card
    lowest     : jax.Array   # bool: the Raised Fist card
    enh        : jax.Array = None   # ENH_* (None = no enhancements)
    lucky      : jax.Array = None   # Lucky Card triggers this hand (Lucky Cat)
    debuff     : jax.Array = None   # bool: debuffed by the boss (no per-card condition holds)


def gate(cond, param, info: Info, jstate) -> jax.Array:
    """GATE kinds. cond, param, jstate broadcast together. Per-card kinds -> False."""
    loyal = jnp.mod(param - 1 - (info.hands_total - jstate), param + 1) == param
    b = lambda x: jnp.broadcast_to(x, cond.shape)
    return jnp.select(
        [cond == C_NONE, cond == C_HAND_HAS, cond == C_COUNT_CMP, cond == C_COUNT_EQ,
         cond == C_FACES_GE, cond == C_LAST_HAND, cond == C_HAND_REPEAT,
         cond == C_SEEING_DOUBLE, cond == C_FLOWER_POT, cond == C_HELD_ALL_BLACK,
         cond == C_LOYALTY, cond == C_TODO_HAND, cond == C_SCORING_FACE,
         cond == C_MOST_PLAYED, cond == C_BOSS, cond == C_ACE_STRAIGHT, cond == C_MONEY_LE,
         cond == C_NOT_BOSS, cond == C_BOSS_TRIGGERED],
        [jnp.ones_like(cond, bool), info.contains[jnp.clip(param, 0, N_HAND_TYPES - 1)],
         info.n_played <= param, info.n_played == param, info.faces >= param,
         b(info.last_hand), b(info.hand_repeat), b(info.seeing_double), b(info.flower_pot),
         b(info.held_black), loyal, info.handtype == jstate, b(info.scoring_face),
         b(info.most_played), b(info.is_boss), b(info.ace_straight), info.money <= param,
         b(~info.is_boss), b(info.boss_triggered)],
        default=jnp.zeros_like(cond, bool))


def match(cond, param, cards: Cards, info: Info, tg: Targets, mods: RuleMods, jstate):
    """(J, K) bool — does slot j's condition hold on card k? A gate applies to
    every real card. cond / param / jstate are (J,)."""
    c, p = cond[:, None], param[:, None]
    r, s = cards.rank[None, :], cards.suit[None, :]
    e = jnp.zeros_like(r) if cards.enh is None else cards.enh[None, :]
    lucky = jnp.zeros(cards.rank.shape, bool) if cards.lucky is None else cards.lucky > 0
    ranked = has_rank(e)
    shape = (cond.shape[0], cards.rank.shape[0])
    b = lambda x: jnp.broadcast_to(x, shape)
    per = jnp.select(
        [c == C_SUIT, c == C_RANK_IN, c == C_IS_FACE, c == C_RANK_SUIT, c == C_POSITION,
         c == C_FIRST_FACE, c == C_ANCIENT_SUIT, c == C_LOWEST_HELD, c == C_MAIL_RANK,
         c == C_CASTLE_SUIT, c == C_ENH, c == C_LUCKY_HIT],
        [b(suit_is(s, p, mods.smeared, e)), b(ranked & (((p >> r) & 1) != 0)),
         b(face(r, mods.pareidolia, e)),
         b(ranked & (r == tg.idol_card % 13) & suit_is(s, tg.idol_card // 13, mods.smeared, e)),
         b(cards.nth[None, :] == p), b(cards.first_face[None, :]),
         b(suit_is(s, tg.ancient_suit, mods.smeared, e)), b(cards.lowest[None, :]),
         b(ranked & (r == tg.mail_rank)), b(suit_is(s, tg.castle_suit, mods.smeared, e)),
         b(e == p), b(lucky[None, :])],
        default=b(gate(cond, param, info, jstate)[:, None]))
    if cards.debuff is not None:                    # a debuffed card: no per-card condition holds
        per_card = jnp.isin(c, jnp.array(PER_CARD_CONDS))
        per = per & ~(per_card & cards.debuff[None, :])
    return per & cards.ok[None, :]


def rule_counts(rows: Joker, cards: Cards, weight, info, tg, mods, jstate, per_card):
    """(J, N_UPDATE_RULES) float — n for each update rule on this event.
    per_card: count the matching cards (weighted; a Lucky Cat rule counts the
    Lucky triggers); else 1 when the gate is open."""
    out = []
    lucky = jnp.zeros_like(weight) if cards.lucky is None else cards.lucky.astype(jnp.float32)
    for k in range(rows.UPD_COND.shape[1]):
        cond = rows.UPD_COND[:, k]
        if per_card:
            m = match(cond, rows.UPD_PARAM[:, k], cards, info, tg, mods, jstate)
            w = jnp.where((cond == C_LUCKY_HIT)[:, None], lucky[None, :], weight[None, :])
            out.append((m * w).sum(axis=1))
        else:
            out.append(gate(cond, rows.UPD_PARAM[:, k], info, jstate).astype(jnp.float32))
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
    hand_played : jax.Array   # (N_HAND_TYPES,) before this hand
    hand_round  : jax.Array   # (N_HAND_TYPES,) before this hand
    is_boss     : jax.Array
    targets     : Targets
    enh         : jax.Array = None   # (5,) played card ENH_* (None = all plain)
    ed          : jax.Array = None   # (5,) ED_*
    seal        : jax.Array = None   # (5,) SEAL_*
    bonus       : jax.Array = None   # (5,) perma bonus chips (Hiker)
    held_enh    : jax.Array = None   # (H,)
    held_seal   : jax.Array = None   # (H,)
    level       : jax.Array = None   # (N_HAND_TYPES,) hand levels (None = all 1)
    joker_ed    : jax.Array = None   # (J,) joker editions
    debuff      : jax.Array = None   # (5,) bool: played cards the boss debuffs
    held_debuff : jax.Array = None   # (H,) bool: held cards the boss debuffs
    hand_debuffed : jax.Array = None # bool: The Psychic / The Eye / The Mouth: the hand scores 0
    boss_trigger : jax.Array = None  # bool: The Arm lowered a level / The Ox took the money (Matador)
    flint       : jax.Array = None   # bool: The Flint halves the base chips and mult
    observatory : jax.Array = None   # (N_HAND_TYPES,) held Planets per hand (Observatory: X1.5 each)
    det_enh     : jax.Array = None   # (5,) the enhancements hand detection reads (a debuffed Wild: plain)
    all_ids     : jax.Array = None   # (J,) every held joker, a debuffed one too (Baseball Card counts it)
    card_room   : jax.Array = None   # free card instances (DNA copies: at most this many)
    touch_debuff : jax.Array = None  # (5,) bool: the debuff a played card gets when Midas Mask / Vampire
                                     # touch it (set_ability runs debuff_card again: a plain / Gold card
                                     # without The Pillar mark)
    plasma      : jax.Array = None   # bool: the Plasma Deck balances chips and mult (final_scoring_step)


class PlayOut(NamedTuple):
    handtype    : jax.Array
    mask        : jax.Array   # (5,) scored cards
    base_chips  : jax.Array   # hand base + scored card chips (one pass)
    base_mult   : jax.Array
    chips       : jax.Array   # float32 after every joker
    mult        : jax.Array
    score       : jax.Array   # int32 floor(chips * mult)
    money       : jax.Array   # $ earned during the hand (To Do, Gold Seal, Lucky, Rough Gem ...)
    joker_value : jax.Array   # after BEFORE, the per-card counters and AFTER
    expired     : jax.Array   # (J,) bool — used up by AFTER (Ice Cream, Seltzer)
    hand_played : jax.Array   # (N_HAND_TYPES,) after this hand
    hand_round  : jax.Array
    reps        : jax.Array   # (5,) repetitions of each played card (0 = not scored)
    rolls       : jax.Array   # (5, MAX_REPS, J) bool — the 1-in-N draws per scored card (tests)
    hrolls      : jax.Array   # (H, MAX_HELD_REPS, J) bool — per held card
    rnd         : jax.Array   # (J,) Misprint's draw per slot
    enh         : jax.Array   # (5,) the played cards' enhancements after Midas Mask / Vampire
    broken      : jax.Array   # (5,) bool — Glass cards that broke
    perma       : jax.Array   # (5,) perma bonus chips each played card gained (Hiker)
    level_up    : jax.Array   # scalar int — levels the played hand gained (Space Joker)
    tarots      : jax.Array   # scalar int — Tarots the hand asked for (8 Ball, Superposition, Vagabond)
    lucky_rolls : jax.Array   # (5, MAX_REPS, 2) bool — Lucky Card mult / $ draws (tests)
    lucky_hits  : jax.Array   # (5,) Lucky Card triggers per played card
    glass_roll  : jax.Array   # (5,) bool — the 1 in 4 Glass draws (tests)
    space_roll  : jax.Array   # (J,) bool — the 1 in 4 Space Joker draws (tests)
    dna_enh     : jax.Array   # (J,) DNA copy k's enhancement (copy k = the k-th DNA effect, slot order)
    joker_state : jax.Array   # (J,) after the hand
    spawn_kind  : jax.Array   # (J,) what each slot's main effect creates: CK_TAROT / CK_SPECTRAL, -1 none
    boss_triggered : jax.Array  # bool: the hand triggered the boss (Matador)
    dna_n       : jax.Array   # scalar int: DNA copies made (they were held cards in this hand)
    dna_mark    : jax.Array   # (J,) bool: copy k has The Pillar mark (no Midas / Vampire touched card 0 first)
    reset       : jax.Array   # (5,) bool: Midas Mask / Vampire touched the card (its The Pillar mark goes)


def effective(ids, joker_value, joker_state):
    """The rows each slot SCORES with (copies resolved). -> (rows, active, value, state)."""
    src = copy_sources(ids)
    ok = src >= 0
    s = jnp.clip(src, 0, ids.shape[0] - 1)
    rows = rows_for(jnp.where(ok, ids[s], J_EMPTY))
    return rows, ok, joker_value[s], joker_state[s]


def _before_cards(rows: Joker, ids, active, enh, sc, rank, pareidolia, deb, tdeb):
    """The BEFORE step reads the cards in slot order (one eval_card per joker):
    Midas Mask (face cards -> Gold) and Vampire (strip enhancements, +X0.1 each),
    real rows only (neither copies). A card a Vampire took stays marked for the rest
    of the hand (card.lua v.vampired): a later Vampire skips it, even after a Midas
    Mask makes it Gold again. Ride the Bus and DNA in slot j see the cards as the
    slots to their left left them. A debuffed card is no face card, and Vampire skips it.
    Both call set_ability, which builds a new ability table: the card loses its The Pillar
    mark (played_this_ante, set before the hand), even a Gold card Midas Mask sets again.
    set_ability also runs the boss's debuff_card again: a touched card takes tdeb (a Stone
    card a Vampire strips can have the boss's suit, or be a face card in The Plant), and a
    later Midas Mask / Vampire skips it then.
    -> (enh (5,), stripped (J,), faces (J,) a scored face card when slot j acts,
        enh0 (J,) played card 0's enhancement when slot j acts, reset (5,) the cards
        set_ability touched, reset0 (J,) card 0 was touched before slot j acts,
        deb (5,) the debuffs after the step, deb0 (J,) card 0's debuff when slot j acts)"""
    def step(carry, j):
        e, taken, reset, d = carry
        seen = ((sc & ~d & face(rank, pareidolia, e)).any(), e[0], reset[0], d[0])
        midas = active[j] & (rows.EFFECT[j] == E_MIDAS)
        vamp = active[j] & (ids[j] == J_VAMPIRE)
        hit = vamp & sc & ~d & (e != ENH_NONE) & ~taken
        e = jnp.where(hit, ENH_NONE, e)
        gold = midas & sc & ~d & face(rank, pareidolia, e)
        e = jnp.where(gold, ENH_GOLD, e)
        touched = hit | gold
        return (e, taken | hit, reset | touched, d | (touched & tdeb)), (hit.sum(),) + seen
    z = jnp.zeros_like(sc)
    (enh, _, reset, deb), (stripped, faces, enh0, reset0, deb0) = jax.lax.scan(
        step, (enh, z, z, deb), jnp.arange(active.shape[0]))
    return enh, stripped, faces, enh0, reset, reset0, deb, deb0


def _suit_sets(sc, suit, enh, smeared, deb=None):
    """Seeing Double and Flower Pot on the scored cards (Smeared applies). The game counts
    the cards that are not Wild first, then gives each Wild card the first suit still empty.
    A debuffed card has no suit (Card:is_suit), but Flower Pot asks the plain cards with
    bypass_debuff. -> (seeing_double, flower_pot) bool"""
    deb = jnp.zeros_like(sc) if deb is None else deb
    wild = sc & (enh == ENH_WILD)
    plain = sc & ~wild
    has_s = [(plain & ~deb & suit_is(suit, s, smeared, enh)).any() for s in range(4)]
    for k in range(5):                               # Seeing Double: Wild fills C, D, S, H
        taken = jnp.bool_(False)
        for s in (0, 1, 3, 2):
            t = wild[k] & ~deb[k] & ~taken & ~has_s[s]
            has_s[s] = has_s[s] | t
            taken = taken | t
    seeing_double = has_s[0] & (has_s[1] | has_s[2] | has_s[3])
    got = [jnp.bool_(False)] * 4                     # Flower Pot: H, D, S, C in that order
    for group in (plain, wild & ~deb):
        for k in range(5):
            taken = jnp.bool_(False)
            for s in (2, 1, 3, 0):
                t = group[k] & ~taken & ~got[s] & suit_is(suit[k], s, smeared, enh[k])
                got[s] = got[s] | t
                taken = taken | t
    return seeing_double, got[0] & got[1] & got[2] & got[3]


def play(p: PlayIn, mods: RuleMods, key, sell=None) -> PlayOut:
    """sell: (J,) sell values — Swashbuckler counts the OTHER jokers (total - its own)."""
    k_card, k_held, k_rand, k_lucky, k_glass, k_space = jax.random.split(key, 6)
    ids = p.ids
    J = ids.shape[0]
    H = p.held_rank.shape[0]
    zi = lambda n: jnp.zeros(n, jnp.int32)
    enh0 = zi(5) if p.enh is None else p.enh
    ed = zi(5) if p.ed is None else p.ed
    seal = zi(5) if p.seal is None else p.seal
    bonus = jnp.zeros(5) if p.bonus is None else p.bonus.astype(jnp.float32)
    held_enh = zi(H) if p.held_enh is None else p.held_enh
    held_seal = zi(H) if p.held_seal is None else p.held_seal
    level = jnp.ones(N_HAND_TYPES, jnp.int32) if p.level is None else p.level
    joker_ed = zi(J) if p.joker_ed is None else p.joker_ed
    f = jnp.bool_(False)
    deb = jnp.zeros(5, bool) if p.debuff is None else p.debuff
    tdeb = jnp.zeros(5, bool) if p.touch_debuff is None else p.touch_debuff
    hdeb = jnp.zeros(H, bool) if p.held_debuff is None else p.held_debuff
    hand_debuffed = f if p.hand_debuffed is None else p.hand_debuffed
    boss_trigger = f if p.boss_trigger is None else p.boss_trigger
    flint = f if p.flint is None else p.flint
    obs = zi(N_HAND_TYPES) if p.observatory is None else p.observatory
    rows, active = rows_for(ids), ids != J_EMPTY
    erows, eok, _, estate = effective(ids, p.joker_value, p.joker_state)

    # 1. the hand (a debuffed Wild card has only its own suit for a flush)
    hand = detect(p.rank, p.suit, p.valid, mods, enh0 if p.det_enh is None else p.det_enh)
    ht = hand.handtype
    played = p.hand_played.at[ht].add(1)
    in_round = p.hand_round.at[ht].add(1)
    sc = hand.scoring
    isf0 = face(p.rank, mods.pareidolia, enh0) & ~deb

    # Blackboard: Card:is_suit with flush_calc (a debuffed card keeps its suit, a debuffed Wild only its own)
    hb_enh = jnp.where(hdeb & (held_enh == ENH_WILD), ENH_NONE, held_enh)
    held_black = ~(p.held_ok & ~(suit_is(p.held_suit, 0, mods.smeared, hb_enh)
                                 | suit_is(p.held_suit, 3, mods.smeared, hb_enh))).any()
    others = jnp.where(jnp.arange(N_HAND_TYPES) == ht, -1, played)
    # Seeing Double / Flower Pot / Superposition read the cards after Midas Mask and Vampire
    # (joker_main): they are set below, after BEFORE. Matador: the hand triggered the boss
    # (The Arm, The Ox, The Flint, a debuffed scoring card, a debuffed hand).
    triggered = boss_trigger | flint | (sc & deb).any() | hand_debuffed
    info = Info(ht, hand.contains, hand.n_played, p.hands_left == 0, in_round[ht] > 1,
                (played[ht] > others).all(), (sc & isf0).any(), f, f,
                held_black, jnp.int32(0), p.hand_played.sum(), jnp.asarray(p.is_boss, bool),
                f, jnp.asarray(p.money, jnp.float32), triggered)
    tg = p.targets

    # 2. BEFORE: counters (real rows only), To Do List money and Space Joker (copies too),
    #    Midas Mask / Vampire, then the hand's chips and mult at its level
    scored0 = Cards(p.rank, p.suit, sc, jnp.where(sc, jnp.cumsum(sc) - 1, -1),
                    jnp.zeros(5, bool), jnp.zeros(5, bool), enh0, None, deb)
    enh, stripped, faces_at, enh0_at, reset, reset0_at, deb, deb0_at = _before_cards(
        rows, ids, active, enh0, sc, p.rank, mods.pareidolia, deb, tdeb)
    # from here on the cards are as BEFORE left them: a card it debuffed scores nothing and
    # triggers the boss (evaluate_play's scoring loop: G.GAME.blind.triggered)
    live_sc = sc & ~deb                              # the cards that score (a debuffed one does not)
    triggered = triggered | (sc & deb).any()
    info = info._replace(boss_triggered=triggered)
    counts = rule_counts(rows, scored0, sc.astype(jnp.float32), info, tg, mods, p.joker_state, False)
    # Ride the Bus reads the faces as the Midas Masks / Vampires to its left left them
    counts = jnp.where(rows.UPD_COND == C_SCORING_FACE, faces_at[:, None].astype(jnp.float32), counts)
    value, _, _ = update_ex(rows, active, p.joker_value, p.joker_state, T_ONCE_MAIN, counts)
    gate_ok = gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    before = eok & (erows.TRIGGER == T_BEFORE) & (erows.EFFECT == E_ADD_MONEY) & gate_ok
    money = jnp.where(before, erows.VALUE, 0.0).sum()
    space = eok & (erows.TRIGGER == T_BEFORE) & (erows.EFFECT == E_LEVEL_UP)
    space_roll = jax.random.uniform(k_space, (J,)) < _odds(erows, mods)
    level_up = (space & space_roll).sum().astype(jnp.int32)
    value = value + jnp.where(active & (ids == J_VAMPIRE), rows.UPD_DELTA[:, 0] * stripped, 0.0)
    # DNA (copies too): the round's first hand is one card -> each DNA effect copies it, as the
    # slots to its left left it, onto the end of the hand (G.hand:emplace in BEFORE). So in this
    # hand the copies are held cards (Steel, Baron, Raised Fist, Blackboard, Mime ...), Erosion
    # counts them, and Hologram gains at once (playing_card_joker_effects). copy_card keeps the
    # card's debuff and its The Pillar mark (unless a Midas Mask / Vampire touched it first).
    lane = jnp.arange(H)
    last = jnp.max(jnp.where(p.held_ok, lane, -1))
    room = jnp.int32(H) if p.card_room is None else p.card_room
    dna_on = eok & (erows.TRIGGER == T_BEFORE) & (erows.EFFECT == E_SPAWN_CARD) \
        & (p.hand_round.sum() == 0) & (hand.n_played == 1)
    dna_n = jnp.clip(jnp.minimum(dna_on.sum(), room), 0, H - 1 - last).astype(jnp.int32)
    dna_slot = compact(jnp.arange(J), dna_on)
    dna_enh, dna_mark, dna_deb = enh0_at[dna_slot], ~reset0_at[dna_slot], deb0_at[dna_slot]
    kc = lane - (last + 1)
    is_copy = (kc >= 0) & (kc < dna_n)
    h_ok = p.held_ok | is_copy
    h_rank = jnp.where(is_copy, p.rank[0], p.held_rank)
    h_suit = jnp.where(is_copy, p.suit[0], p.held_suit)
    held_enh = jnp.where(is_copy, dna_enh[jnp.clip(kc, 0, J - 1)], held_enh)
    held_seal = jnp.where(is_copy, seal[0], held_seal)
    hdeb = jnp.where(is_copy, dna_deb[jnp.clip(kc, 0, J - 1)], hdeb)   # copy_card: other.debuff
    v_add, _, _ = update_ex(rows, active, value, p.joker_state, T_ON_CARD_ADDED,
                            jnp.full((J, rows.UPD_COND.shape[1]), dna_n, jnp.float32))
    value = jnp.where(dna_n > 0, v_add, value)
    hb_enh = jnp.where(hdeb & (held_enh == ENH_WILD), ENH_NONE, held_enh)
    held_black = ~(h_ok & ~(suit_is(h_suit, 0, mods.smeared, hb_enh)
                            | suit_is(h_suit, 3, mods.smeared, hb_enh))).any()
    hchips, hmult = hand_chips_mult(level.at[ht].add(level_up))
    # The Flint: the base chips and mult are halved (rounded, mult at least 1)
    h_chips = jnp.where(flint, jnp.maximum(jnp.floor(hchips[ht] * 0.5 + 0.5), 0), hchips[ht])
    h_mult = jnp.where(flint, jnp.maximum(jnp.floor(hmult[ht] * 0.5 + 0.5), 1), hmult[ht])
    seeing_double, flower = _suit_sets(sc, p.suit, enh, mods.smeared, deb)
    ace = (sc & has_rank(enh) & (p.rank == 12)).any()
    info = info._replace(seeing_double=seeing_double, flower_pot=flower, held_black=held_black,
                         ace_straight=ace & hand.contains[STRAIGHT])

    isf = face(p.rank, mods.pareidolia, enh) & ~deb
    nth = jnp.where(sc, jnp.cumsum(sc) - 1, -1)
    ff = sc & isf
    first_face = ff & (jnp.cumsum(ff) == 1)
    scored = Cards(p.rank, p.suit, sc, nth, first_face, jnp.zeros(5, bool), enh, None, deb)

    # 3. scored cards x repetitions x (the card, its edition, the jokers)
    m_card = match(erows.CONDITION, erows.COND_PARAM, scored, info, tg, mods, estate) & eok[:, None]
    on_card = erows.TRIGGER == T_PER_SCORED
    retrig = on_card & (erows.EFFECT == E_RETRIGGER)
    rep_n = jnp.where(erows.SOURCE == S_COUNTER, erows.VALUE2, erows.VALUE)
    reps = 1 + (seal == SEAL_RED) + (m_card * (retrig * rep_n)[:, None]).sum(axis=0)
    reps = jnp.where(live_sc, jnp.clip(reps, 1, MAX_REPS), 0).astype(jnp.int32)     # (5,)
    fx = on_card & ~retrig
    odds = _odds(erows, mods)
    roll = jax.random.uniform(k_card, (5, MAX_REPS, J)) < odds[None, None, :]
    live = (jnp.arange(MAX_REPS)[None, :] < reps[:, None])                        # (5, R)
    on = live[:, :, None] & (m_card & fx[:, None]).T[:, None, :] & roll           # (5, R, J)
    c_j = jnp.where((erows.EFFECT == E_ADD_CHIPS) | (erows.EFFECT == E_CHIPS_MULT), erows.VALUE, 0.0)
    m_j = jnp.select([erows.EFFECT == E_ADD_MULT, erows.EFFECT == E_CHIPS_MULT],
                     [erows.VALUE, erows.VALUE2], default=0.0)
    x_j = jnp.where(erows.EFFECT == E_XMULT, erows.VALUE, 1.0)
    d_j = jnp.where(erows.EFFECT == E_ADD_MONEY, erows.VALUE, 0.0)
    hike = jnp.where(erows.EFFECT == E_PERMA_CHIPS, erows.VALUE, 0.0)            # Hiker, per rep
    spawn = (on & (erows.EFFECT == E_SPAWN_CONSUMABLE)[None, None, :]).sum()    # 8 Ball

    # the card's own step: chips (perma grows by the Hikers of the earlier reps), Mult / Lucky mult,
    # Gold Seal / Lucky $, Glass X; then its edition
    lr = jax.random.uniform(k_lucky, (5, MAX_REPS, 2))
    lucky_m = live & (enh == ENH_LUCKY)[:, None] & (lr[..., 0] < mods.odds / LUCKY_MULT_ODDS)
    lucky_d = live & (enh == ENH_LUCKY)[:, None] & (lr[..., 1] < mods.odds / LUCKY_DOLLAR_ODDS)
    hike_rep = (on * hike[None, None, :]).sum(axis=2)                              # (5, R) perma added per rep
    perma_before = jnp.cumsum(hike_rep, axis=1) - hike_rep
    rep_chips = jnp.where(live, card_chips(p.rank, enh, bonus)[:, None] + perma_before, 0.0)
    own_m = jnp.where(live, jnp.where(enh == ENH_MULT, MULT_MULT, 0)[:, None] + LUCKY_MULT * lucky_m, 0.0)
    own_x = jnp.where(live & (enh == ENH_GLASS)[:, None], GLASS_X, 1.0)
    own_d = live * (GOLD_SEAL_DOLLARS * (seal == SEAL_GOLD)[:, None] + LUCKY_DOLLARS * lucky_d)
    ed_c = live * jnp.where(ed == ED_FOIL, FOIL_CHIPS, 0)[:, None]
    ed_m = live * jnp.where(ed == ED_HOLO, HOLO_MULT, 0)[:, None]
    ed_x = jnp.where(live & (ed == ED_POLY)[:, None], POLY_X, 1.0)

    base_chips = (jnp.where(live_sc, card_chips(p.rank, enh, bonus), 0)).sum() + h_chips
    base_mult = h_mult.astype(jnp.float32)
    chips = h_chips + rep_chips.sum() + ed_c.sum() + (on * c_j).sum()
    money = money + own_d.sum() + (on * d_j).sum()
    # per (card, rep): [own (+m, x glass), edition (+m, x poly), jokers ...]
    steps_x = jnp.concatenate([own_x[..., None], ed_x[..., None], jnp.where(on, x_j, 1.0)], axis=2)
    steps_m = jnp.concatenate([own_m[..., None], ed_m[..., None], jnp.where(on, m_j, 0.0)], axis=2)
    X = steps_x.ravel()
    mult = fold_mult(base_mult, X, steps_m.ravel() * X)

    # per-card counters (Wee Joker per rep, Lucky Cat per Lucky trigger)
    lucky_hits = (lucky_m | lucky_d).sum(axis=1)
    scored = scored._replace(lucky=lucky_hits)
    counts = rule_counts(rows, scored, reps.astype(jnp.float32), info, tg, mods, p.joker_state, True)
    value, _, _ = update_ex(rows, active, value, p.joker_state, T_PER_SCORED, counts)

    # 4. cards held in hand x repetitions (Red Seal, Mime) x (Steel, the jokers)
    lo = jnp.where(h_ok & has_rank(held_enh), h_rank, 99)
    lowest = h_ok & has_rank(held_enh) & (jnp.arange(H) == (H - 1 - jnp.argmin(lo[::-1]))) \
        & (lo.min() < 99)                                                        # the last lowest
    held = Cards(h_rank, h_suit, h_ok, jnp.full(H, -1), jnp.zeros(H, bool), lowest,
                 held_enh, None, hdeb)
    m_held = match(erows.CONDITION, erows.COND_PARAM, held, info, tg, mods, estate) & eok[:, None]
    m_held = m_held & ~hdeb[None, :]                 # a debuffed held card gives nothing (Mime: nothing to repeat)
    on_held = erows.TRIGGER == T_PER_HELD
    mime = on_held & (erows.EFFECT == E_RETRIGGER)
    hfx = on_held & ~mime
    hroll = jax.random.uniform(k_held, (H, MAX_HELD_REPS, J)) < odds[None, None, :]
    steel = h_ok & ~hdeb & (held_enh == ENH_STEEL)
    first = (m_held & hfx[:, None]).T & hroll[:, 0, :]                          # (H, J)
    had = first.any(axis=1) | steel
    hreps = 1 + had * ((held_seal == SEAL_RED) + (m_held * (mime * erows.VALUE)[:, None]).sum(axis=0))
    hreps = jnp.where(h_ok, jnp.clip(hreps, 1, MAX_HELD_REPS), 0).astype(jnp.int32)
    hlive = (jnp.arange(MAX_HELD_REPS)[None, :] < hreps[:, None])                # (H, Rh)
    hon = hlive[:, :, None] & (m_held & hfx[:, None]).T[:, None, :] & hroll      # (H, Rh, J)
    hm = jnp.select([erows.EFFECT == E_ADD_MULT, erows.EFFECT == E_MULT_NOMINAL],
                    [jnp.broadcast_to(erows.VALUE, (H, J)),
                     erows.VALUE[None, :] * BASE_CARD_CHIPS[h_rank][:, None]], default=0.0)
    hm = jnp.broadcast_to(hm[:, None, :], hon.shape)
    money = money + (hon * d_j).sum()
    steel_x = jnp.where(hlive & steel[:, None], STEEL_X, 1.0)
    HX = jnp.concatenate([steel_x[..., None], jnp.where(hon, x_j, 1.0)], axis=2)
    HM = jnp.concatenate([jnp.zeros(steel_x.shape + (1,)), jnp.where(hon, hm, 0.0)], axis=2)
    mult = fold_mult(mult, HX.ravel(), HM.ravel() * HX.ravel())

    # 5. jokers in slot order: edition chips / mult, the main effect, Baseball Card, Polychrome
    now = p.money + money
    ctx = (p.ctx.at[Q_MONEY].set(jnp.maximum(now, 0.0))
                .at[Q_MONEY_DIV_5].set(jnp.maximum(jnp.floor(now / 5), 0.0))
                .at[Q_HAND_PLAYED].set(played[ht])
                .at[Q_DECK_LOST].set(jnp.maximum(p.ctx[Q_DECK_LOST] - dna_n, 0.0)))   # Erosion: the DNA copies
    src = copy_sources(ids)
    ev = value[jnp.clip(src, 0, J - 1)]
    v = resolve(erows, eok, ev, ctx)
    if sell is not None:
        own = sell[jnp.clip(src, 0, J - 1)]
        v = v - jnp.where(eok & (erows.QUERY == Q_OTHER_SELL), erows.VALUE * own, 0.0)
    # Vagabond (info.money) reads G.GAME.dollars: the money before this hand (ease_dollars waits
    # for an event); Bull / Bootstraps (ctx) add the hand's dollar_buffer
    main = eok & (erows.TRIGGER == T_ONCE_MAIN) & gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    # Matador's $ (joker_main, slot order): Bull / Bootstraps to its right count it (dollar_buffer)
    md = jnp.where(main & (erows.EFFECT == E_ADD_MONEY), v, 0.0)
    extra = jnp.maximum(now + jnp.cumsum(md) - md, 0.0)
    v = jnp.where(eok & (erows.QUERY == Q_MONEY), erows.QBASE + erows.VALUE * extra, v)
    v = jnp.where(eok & (erows.QUERY == Q_MONEY_DIV_5), erows.QBASE + erows.VALUE * jnp.floor(extra / 5), v)
    money = money + md.sum()
    rnd = jnp.floor(jax.random.uniform(k_rand, (J,)) * (erows.VALUE + 1))
    mc = jnp.where(main & (erows.EFFECT == E_ADD_CHIPS), v, 0.0)
    mm = jnp.select([main & (erows.EFFECT == E_ADD_MULT), main & (erows.EFFECT == E_MULT_RANDOM)],
                    [v, rnd], default=0.0)
    mx = jnp.where(main & (erows.EFFECT == E_XMULT), v, 1.0)
    spawn_kind = jnp.select([main & (erows.EFFECT == E_SPAWN_CONSUMABLE),     # Superposition, Vagabond
                             main & (erows.EFFECT == E_SPAWN_SPECTRAL)], [0, 2], -1).astype(jnp.int32)   # Seance
    base_x = jnp.where(main & (erows.EFFECT == E_XMULT_UNCOMMON), v, 1.0).prod()
    # Baseball Card: X1.5 after each Uncommon joker, one Crimson Heart disabled too (the joker-on-joker
    # pass runs for every slot)
    all_ids = ids if p.all_ids is None else p.all_ids
    unc = (all_ids != J_EMPTY) & (JOKER_RARITY[jnp.clip(all_ids, 0, N_JOKERS - 1)] == UNCOMMON)
    mx = mx * jnp.where(unc, base_x, 1.0)
    jc = jnp.where(active & (joker_ed == ED_FOIL), FOIL_CHIPS, 0.0)
    jm = jnp.where(active & (joker_ed == ED_HOLO), HOLO_MULT, 0.0)
    jx = jnp.where(active & (joker_ed == ED_POLY), POLY_X, 1.0)
    chips = chips + mc.sum() + jc.sum()
    xs = mx * jx
    mult = fold_mult(mult, xs, (jm + mm) * xs)
    # Observatory: each held Planet of this hand, X1.5 (the consumables after the jokers)
    mult = mult * jnp.power(1.5, obs[ht].astype(jnp.float32))
    # Plasma Deck (final_scoring_step, after the jokers and the consumables): chips = mult =
    # floor((chips + mult) / 2); the score is then n * n, exact in int32 up to SCORE_CAP
    plasma = jnp.bool_(False) if p.plasma is None else p.plasma
    half = jnp.floor((chips + mult) / 2)
    chips = jnp.where(plasma, half, chips)
    mult = jnp.where(plasma, half, mult)

    # 6. AFTER (Ice Cream, Seltzer), then the Glass cards break (a debuffed one does not)
    value, _, fired = update_ex(rows, active, value, p.joker_state, T_AFTER)
    gone = expired(rows, active, value, fired)
    glass_roll = jax.random.uniform(k_glass, (5,)) < mods.odds / GLASS_ODDS
    broken = live_sc & (enh == ENH_GLASS) & glass_roll

    # floor(chips x mult): a few float32 ulps of slack, so 15 x 12.6 is 189 and not 188.99998 -> 188.
    # The game adds in a double; the int32 score stops at SCORE_CAP (env.py caps the round's sum too).
    prod = jnp.nan_to_num(chips * mult, nan=0.0, posinf=SCORE_CAP)
    score = jnp.floor(jnp.minimum(prod * (1 + 2.0 ** -22) + 1e-3, SCORE_CAP)).astype(jnp.int32)
    n = jnp.clip(half, 0, 46340).astype(jnp.int32)                      # 46340^2 < 2^31
    score = jnp.where(plasma, jnp.where(half > 46340, jnp.int32(SCORE_CAP), n * n), score)
    perma = hike_rep.sum(axis=1)
    out = PlayOut(ht, sc, base_chips, base_mult, chips, mult, score, money, value, gone,
                  played, in_round, reps, roll, hroll, rnd, enh, broken, perma, level_up,
                  spawn.astype(jnp.int32), jnp.stack([lucky_m, lucky_d], axis=-1), lucky_hits,
                  glass_roll, space_roll, dna_enh, p.joker_state, spawn_kind, triggered,
                  dna_n, dna_mark, reset)

    # a debuffed hand (The Psychic / The Eye / The Mouth): 0 chips x 0 mult; no BEFORE, no card,
    # no joker effect but Matador's $ (copies too); the AFTER step still acts
    d_value, _, d_fired = update_ex(rows, active, p.joker_value, p.joker_state, T_AFTER)
    matador = (eok & (erows.TRIGGER == T_ONCE_MAIN) & (erows.EFFECT == E_ADD_MONEY)
               & (erows.CONDITION == C_BOSS_TRIGGERED))
    z5 = jnp.zeros(5, bool)
    dead = out._replace(
        base_chips=jnp.zeros_like(base_chips), base_mult=jnp.zeros_like(base_mult),
        chips=jnp.zeros_like(chips), mult=jnp.zeros_like(mult), score=jnp.int32(0),
        money=jnp.where(matador, erows.VALUE, 0.0).sum().astype(money.dtype),
        joker_value=d_value, expired=expired(rows, active, d_value, d_fired),
        reps=jnp.zeros_like(reps), enh=enh0, broken=z5, perma=jnp.zeros_like(perma),
        level_up=jnp.int32(0), tarots=jnp.int32(0), lucky_hits=jnp.zeros_like(lucky_hits),
        dna_n=jnp.int32(0), reset=z5,
        spawn_kind=jnp.full(J, -1, jnp.int32))
    return jax.tree.map(lambda a, b: jnp.where(hand_debuffed, a, b), dead, out)


class DiscardOut(NamedTuple):
    joker_value : jax.Array
    money       : jax.Array
    expired     : jax.Array
    joker_state : jax.Array   # Yorick's count of discarded cards


def discard(rank, suit, ok, ids, joker_value, joker_state, targets: Targets, mods: RuleMods,
            enh=None, debuff=None):
    """The joker side of one DISCARD (cards in lane order). A debuffed card has no suit,
    rank or face for the per-card rules (Castle, Mail-In Rebate, Hit the Road, Faceless
    Joker); Yorick and Ramen count every card."""
    rows, active = rows_for(ids), ids != J_EMPTY
    erows, eok, _, estate = effective(ids, joker_value, joker_state)
    K = rank.shape[0]
    enh = jnp.zeros(K, jnp.int32) if enh is None else enh
    deb = jnp.zeros(K, bool) if debuff is None else debuff
    faces = (ok & ~deb & face(rank, mods.pareidolia, enh)).sum()
    info = no_info()._replace(faces=faces)
    cards = Cards(rank, suit, ok, jnp.full(K, -1), jnp.zeros(K, bool), jnp.zeros(K, bool), enh, None, deb)
    w = ok.astype(jnp.float32)
    counts = rule_counts(rows, cards, w, info, targets, mods, joker_state, True)
    value, jstate, fired = update_ex(rows, active, joker_value, joker_state, T_ON_DISCARD_CARD, counts)
    gone = expired(rows, active, value, fired)
    counts = rule_counts(rows, cards, w, info, targets, mods, joker_state, False)
    value, jstate, _ = update_ex(rows, active, value, jstate, T_ON_DISCARD, counts)
    # money: Mail-In Rebate per card, Faceless Joker once (copies count)
    m = match(erows.CONDITION, erows.COND_PARAM, cards, info, targets, mods, estate) & eok[:, None]
    per = eok & (erows.TRIGGER == T_ON_DISCARD_CARD) & (erows.EFFECT == E_ADD_MONEY)
    once = eok & (erows.TRIGGER == T_ON_DISCARD) & (erows.EFFECT == E_ADD_MONEY) \
        & gate(erows.CONDITION, erows.COND_PARAM, info, estate)
    money = (jnp.where(per, erows.VALUE, 0.0) * m.sum(axis=1)).sum() + jnp.where(once, erows.VALUE, 0.0).sum()
    return DiscardOut(value, money, gone, jstate)
