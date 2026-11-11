"""The Balatro environment — EnvState, reset, step.

Pure functional: the state pytree is the only memory. reset births the full
state; env_step takes a full state and returns the next full state. Capacity
is SHAPE (MAX_* constants), current count is DATA (scalars/masks in state).

A run is three blinds per ante (Small, Big, Boss) across 8 antes, with one of the 15 decks
(state.back, items.BACK_NAMES; the default is the Red Deck: 4 hands, 4 discards, $4; reset
applies a deck's start, and Green / Ghost / Anaglyph / Plasma act later too), at one of the 8 stakes
(state.stake: 1 White .. 8 Gold; each has the rules of the ones below it: no Small Blind
reward, faster target scaling, -1 discard, Eternal / Perishable / Rental shop jokers):
    STAGE_SELECT NEXT (play the blind) / SKIP (Small or Big: take its tag) / REROLL
                 (the boss: Director's Cut, Retcon) / SELL / USE / SELL_CONS / MOVE_JOKER
    STAGE_BLIND  PLAY / DISCARD / SELL / USE / SELL_CONS / MOVE
                 clear -> the end of the round -> STAGE_CASHOUT; out of hands -> STAGE_LOST;
                 a clear when the joker order can change the end of the round -> STAGE_END
    STAGE_END    MOVE_JOKER / NEXT (the end of the round runs with the new joker order)
    STAGE_CASHOUT  USE / SELL / SELL_CONS / MOVE_JOKER; NEXT (Cash Out): the $, the deck is
                 shuffled, the shop opens -> STAGE_SHOP
    STAGE_SHOP   BUY / REDEEM / OPEN / SELL / USE / SELL_CONS / REROLL / MOVE_JOKER
                 NEXT -> STAGE_SELECT (the next blind); OPEN -> STAGE_PACK
    STAGE_PACK   PICK / SKIP / USE / SELL_CONS / SELL / MOVE
                 -> back to the shop, or to STAGE_SELECT (a tag's free pack)
    STAGE_LOST / STAGE_WON                 every verb is a no-op
The ante goes up when its boss is beaten (at the end of the round), and the next
ante's boss, tags and voucher are rolled then.

CARDS are INSTANCES (NOTES: "card-instance migration"): slot i of the card_*
arrays is one physical card (rank, suit, enhancement, edition, seal, perma
bonus chips). deck and hand hold instance indices. deck[:deck_size] are the
live cards in draw order (deck[:deck_ptr] already dealt); the rest of deck are
the free slots. A copied card (Death, DNA, a Standard pack) takes a free slot;
a destroyed card (Glass, The Hanged Man, Trading Card) frees its slot. The
standard deck is instances 0..51 with rank = i % 13, suit = i // 13. The deck is
shuffled only at the run start, at Cash Out and at the blind start; an Arcana /
Spectral pack deals its hand from the top and puts it back at the bottom.

BOSS BLINDS act only while their round is on (STAGE_BLIND, STAGE_END) and not disabled
(Chicot, Luchador, a sold joker in Verdant Leaf). A debuffed card is a pure
function of the boss and the card (card_debuffed), so it needs no state; The
Pillar reads card_played_ante (a card played in The Pillar's own blind gets no
mark: the game set the debuff flags at the blind's start). Face-down cards
(hand_down, joker_down) change nothing in the env: only what a player sees. Two more
fields record what a player can know (the agent obs reads them): card_down, the cards the
deck view still counts as in the deck (a face-down card until it is played or turned face
up, a discarded one too: ability.wheel_flipped), and joker_seen, a face-down joker that
showed a message in this round (Amber Acorn).

BUY of a Tarot / Planet when the consumable slots are full is the game's Buy and
Use: the card is paid and used at once (if it can be used in the shop). With a
free slot, BUY then USE gives the same result, so there is no separate verb.

The return contract (NOTES: "reward = CASHOUT dollars"):
    reward = the cashout $ on the step that runs the end of a won round (the PLAY that
             clears the blind, or NEXT in STAGE_END), else 0. The $ goes to the money at
             Cash Out (a failed blind pays $0; chips are never reward)
    done   = this step ended the RUN (game_stage became STAGE_LOST or STAGE_WON)

target_scale is the curriculum dial: every blind target is multiplied by it
(floor 1 chip). reset sets it; 1.0 is the real game.

Randomness inside a step (Bloodstone, Misprint, Gros Michel ...) comes from
fold_in(state.key, purpose); a step that uses it moves state.key on. So
explain_play shows exactly the outcome env_step will produce.
"""

import os

import jax
import jax.custom_batching
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import NamedTuple
from scoring import (play as _play, discard as _discard, detect, PlayIn, Targets, no_info,
                     rule_counts, Cards, effective, has_rank, _odds, face as is_face_card, suit_is)
from items import (HAND_NAMES, N_HAND_TYPES, SECRET_HANDS, OX_PAIRS_POS, N_CONS, CONS_KIND, CONS_USE, CONS_PARAM,
                   CONS_LO, CONS_HI, CONS_NEED, CONS_RANKS, CONS_BASE_COST, CONS_COST, CONS_EMPTY,
                   C_FOOL, C_SOUL, C_BLACK_HOLE, CK_TAROT, CK_PLANET, CK_SPECTRAL, TAROTS, PLANETS,
                   SPECTRALS, SPECTRAL_POOL, SOUL_ODDS, PLANET_OF_HAND, U_ENHANCE, U_SUIT, U_STRENGTH,
                   U_HANGED, U_DEATH, U_FOOL, U_HERMIT, U_TEMPERANCE, U_SPAWN_TAROT, U_SPAWN_PLANET,
                   U_WHEEL, U_JUDGEMENT, U_PLANET, U_CREATE_CARDS, U_SEAL, U_AURA, U_WRAITH, U_SIGIL,
                   U_OUIJA, U_ECTOPLASM, U_IMMOLATE, U_ANKH, U_HEX, U_CRYPTID, U_SOUL, U_BLACK_HOLE,
                   MAX_PICK_USE, ENH_NONE, ENH_WILD, ENH_STEEL, ENH_STONE, ENH_GOLD, ENH_GLASS, N_ENH,
                   ED_NONE, ED_FOIL, ED_HOLO, ED_POLY, ED_NEG, ED_COST, MAX_JOKER_SLOTS, SEAL_NONE,
                   SEAL_GOLD, SEAL_RED, SEAL_BLUE, SEAL_PURPLE, GOLD_DOLLARS, N_PACKS, PACK_KIND,
                   PACK_SIZE, PACK_CHOOSE, PACK_COST, PACK_LOGW, P_BUFFOON_NORMAL, MAX_PACK,
                   PACK_SLOTS, PK_ARCANA, PK_CELESTIAL, PK_STANDARD, PK_BUFFOON, PK_SPECTRAL,
                   STANDARD_ENHANCED, STANDARD_SEAL, IK_JOKER, IK_TAROT, IK_PLANET, IK_CARD,
                   JOKER_RATE, TAROT_RATE, PLANET_RATE, CARD_BASE_COST, CONS_ID, poll_edition,
                   cost_of, sell_of, N_VOUCHERS, VOUCHER_NEEDS, VOUCHER_COST, MAX_SHOP_VOUCHERS,
                   V_OVERSTOCK, V_OVERSTOCK_PLUS, V_CLEARANCE, V_LIQUIDATION, V_HONE, V_GLOW_UP,
                   V_REROLL_SURPLUS, V_REROLL_GLUT, V_CRYSTAL_BALL, V_OMEN_GLOBE, V_TELESCOPE,
                   V_OBSERVATORY, V_GRABBER, V_NACHO_TONG, V_WASTEFUL, V_RECYCLOMANCY,
                   V_TAROT_MERCHANT, V_TAROT_TYCOON, V_PLANET_MERCHANT, V_PLANET_TYCOON, V_SEED_MONEY,
                   V_MONEY_TREE, V_BLANK, V_ANTIMATTER, V_MAGIC_TRICK, V_ILLUSION, V_HIEROGLYPH,
                   V_PETROGLYPH, V_DIRECTORS_CUT, V_RETCON, V_PAINT_BRUSH, V_PALETTE,
                   N_TAGS, TAG_MIN_ANTE, TAG_PACK, TAG_EDITION, MAX_TAGS, TG_UNCOMMON, TG_RARE,
                   TG_INVESTMENT, TG_VOUCHER, TG_BOSS, TG_HANDY, TG_GARBAGE, TG_COUPON, TG_DOUBLE,
                   TG_JUGGLE, TG_D6, TG_TOP_UP, TG_SKIP, TG_ORBITAL, TG_ECONOMY, SKIP_TAG_DOLLARS,
                   INVESTMENT_DOLLARS, ECONOMY_MAX, ORBITAL_LEVELS, JUGGLE_SIZE,
                   N_BOSSES, BOSS_MIN_ANTE, BOSS_MULT, BOSS_DOLLARS, BOSS_SHOWDOWN, BOSS_SUIT,
                   BK_HOOK, BK_OX, BK_HOUSE, BK_WALL, BK_WHEEL, BK_ARM, BK_FISH, BK_PSYCHIC,
                   BK_WATER, BK_MANACLE, BK_EYE, BK_MOUTH, BK_PLANT, BK_SERPENT, BK_PILLAR,
                   BK_NEEDLE, BK_TOOTH, BK_FLINT, BK_MARK, BK_ACORN, BK_LEAF, BK_VESSEL, BK_HEART,
                   BK_BELL, WHEEL_ODDS, SERPENT_DRAW, BOSS_REROLL_COST,
                   N_STAKES, ST_NO_SMALL_REWARD, ST_SCALING_2, ST_ETERNAL, ST_DISCARD, ST_SCALING_3, ST_PERISH,
                   ST_RENTAL, STK_ETERNAL, STK_PERISH, STK_RENTAL, STICKER_POLL, PERISH_POLL, PERISH_ROUNDS,
                   RENTAL_RATE, RENTAL_COST, IK_SPECTRAL, N_BACKS, B_RED, B_GREEN, B_MAGIC, B_NEBULA,
                   B_GHOST, B_ABANDONED, B_CHECKERED, B_ZODIAC, B_ANAGLYPH, B_PLASMA, B_ERRATIC,
                   BACK_NAMES, BACK_HANDS, BACK_DISCARDS, BACK_DOLLARS, BACK_JOKER_SLOT, BACK_HAND_SIZE,
                   BACK_CONS_SLOT, BACK_ANTE_SCALING, GHOST_SPECTRAL_RATE, GREEN_PER_HAND,
                   GREEN_PER_DISCARD)
from jokers import MAX_JOKERS, J_EMPTY, N_JOKERS, JOKERS, JOKER_COST, JOKER_RARITY, JOKER_ID, J_BLUEPRINT, J_BRAINSTORM
from jokers import JOKER_ETERNAL_OK, JOKER_PERISH_OK
from jokers import (N_QUERY_KINDS, Q_MONEY_DIV_5, Q_EMPTY_JOKER_SLOTS, Q_DECK_NINE_COUNT,
                    Q_DISCARDS_LEFT, Q_NO_DISCARDS_LEFT, Q_DECK_LEFT, Q_JOKER_COUNT, Q_MONEY,
                    Q_OTHER_SELL, Q_DECK_LOST, Q_DELAYED_GRAT, Q_DECK_STEEL_COUNT,
                    Q_DECK_STONE_COUNT, Q_TAROTS_USED, Q_UNIQUE_PLANETS, Q_DRIVER, Q_SKIPS,
                    J_STENCIL, J_CHAOS, J_CREDIT, J_MOON, J_MARBLE, J_CERTIFICATE, J_ASTRONOMER,
                    J_TRADING, J_DAGGER, J_MADNESS, J_SIXTH, J_INVISIBLE, J_CHICOT, J_MR_BONES,
                    E_ADD_MONEY, E_SELL_VALUE, E_SELL_VALUE_ALL, E_BURGLAR,
                    E_SPAWN_CARD, E_SPAWN_CONSUMABLE, E_LEVEL_UP, E_SPAWN_JOKER, E_DISABLE_BOSS,
                    E_ADD_TAG, E_SAVE, E_DUPLICATE, E_NEGATIVE_COPY, E_DESTROY_CARD,
                    ST_TODO, ST_LOYALTY, P_ALWAYS, P_NOT_EXTINCT, P_EXTINCT)
from jokers import COMMON, UNCOMMON, RARE, LEGENDARY
from jokers import rows_for, resolve, update, update_ex, expired, rule_mods, copy_sources
from jokers import (T_ONCE_MAIN, T_ON_DISCARD, T_END_OF_ROUND, T_ON_BLIND_SELECT, T_BEFORE, T_PER_HELD,
                    E_RETRIGGER, T_ON_SOLD, T_ON_SHOP_EXIT, T_AFTER,
                    T_ON_REROLL, T_ON_SELL_OTHER, T_ON_PACK_OPEN, T_ON_PACK_SKIP, T_ON_PLANET,
                    T_ON_CARD_ADDED, T_ON_GLASS, T_ON_FACE_DESTROYED)
from arrays import first_n, compact, selected_cards

# The persistent compile cache (.jax_cache, git-ignored): a later start loads env_step and every
# jit that took over 1 s to compile from disk (tracing still runs). An env change adds ~60 MB of
# new entries; past CACHE_KEEP the oldest go. Delete the folder to clear it. JAX puts this path
# string in every cache key (an XLA option), so realpath gives one spelling: a start as
# "python d:\...\gui.py" and one as "D:\..." must share the entries.
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".jax_cache")
CACHE_KEEP = 1 << 30   # bytes


def _trim_cache(path=CACHE_DIR, keep=CACHE_KEEP):
    """Delete the oldest cache entries (by write time) until the folder holds at most keep bytes."""
    try:
        files = sorted((f.stat().st_mtime, f.stat().st_size, f.path) for f in os.scandir(path))
    except OSError:                    # no cache yet
        return
    size = sum(n for _, n, _ in files)
    for _, n, p in files:
        if size <= keep:
            break
        try:
            os.remove(p)
            size -= n
        except OSError:                # another process has it open
            pass


_trim_cache()
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)

MAX_HAND_SIZE = 32
MAX_DECK_SIZE = 128
MAX_CONSUMABLES = 16
MAX_SHOP_SLOTS = 4
INT32_MAX = 2147483647   # chips_scored stops here (the game adds in a double)
STARTING_MONEY = 4
CONS_SELL = 1        # a Tarot / Planet sells for max(1, $3 // 2); Gift Card adds $1 a round

# --- verbs: one atomic action per step -------------------------------------
V_PLAY    = 0   # select = the cards (first 5 valid lanes)
V_DISCARD = 1   # select = the cards; with 0 discards left it becomes a PLAY
V_BUY     = 2   # target = shop slot (a joker, a Tarot / Planet or a playing card)
V_SELL    = 3   # target = joker slot
V_REROLL  = 4   # the shop; at the blind select: the boss (Director's Cut, Retcon)
V_NEXT    = 5   # the shop: leave it; the blind select: play the blind
V_MOVE_JOKER = 6  # INSERT the joker in slot target at slot dest (joker order sets XMULT order)
V_MOVE_CARD  = 7  # INSERT the card in lane target at lane dest (lane order = play order, Death)
V_USE     = 8   # target = consumable slot; select = the cards a Tarot acts on
V_SELL_CONS = 9  # target = consumable slot
V_OPEN    = 10  # target = pack slot: pay, open, STAGE_PACK
V_PICK    = 11  # target = pack card; select = the cards an Arcana Tarot acts on
V_SKIP    = 12  # close the pack (Red Card); at the blind select: skip the blind, take its tag
V_REDEEM  = 13  # target = voucher slot of the shop
N_VERBS   = 14
V_NOOP    = 14  # internal: the repair target of every illegal action
VERB_NAMES = ["PLAY", "DISCARD", "BUY", "SELL", "REROLL", "NEXT", "MOVE_JOKER", "MOVE_CARD",
              "USE", "SELL_CONS", "OPEN", "PICK", "SKIP", "REDEEM"]

# --- stages ------------------------------------------------------------------
STAGE_BLIND  = 0
STAGE_SHOP   = 1
STAGE_LOST   = 2
STAGE_WON    = 3
STAGE_PACK   = 4
STAGE_SELECT = 5
STAGE_END    = 6     # after the hand that ends a round, before the end of the round: MOVE_JOKER
STAGE_CASHOUT = 7    # the cash-out screen (ROUND_EVAL): the $ is paid at NEXT, then the shop
STAGE_NAMES = ["BLIND", "SHOP", "LOST", "WON", "PACK", "SELECT", "END", "CASH OUT"]

# LEGAL[stage, verb]: which verbs each stage accepts. Everything else is a no-op.
LEGAL = jnp.array([
    #  PLAY   DISC   BUY    SELL   REROLL NEXT   MOVE_J MOVE_C USE    SELL_C OPEN   PICK   SKIP   REDEEM
    [True,  True,  False, True,  False, False, True,  True,  True,  True,  False, False, False, False],  # BLIND
    [False, False, True,  True,  True,  True,  True,  False, True,  True,  True,  False, False, True ],  # SHOP
    [False] * N_VERBS,                                                                                    # LOST
    [False] * N_VERBS,                                                                                    # WON
    [False, False, False, True,  False, False, True,  True,  True,  True,  False, True,  True,  False],  # PACK
    [False, False, False, True,  True,  True,  True,  False, True,  True,  False, False, True,  False],  # SELECT
    # END: the hand is scoring, so the game blocks a sale and a use (can_sell_card / can_use_consumeable:
    # cards in G.play); a joker can be dragged (no lock)
    [False, False, False, False, False, True,  True,  False, False, False, False, False, False, False],  # END
    [False, False, False, True,  False, True,  True,  False, True,  True,  False, False, False, False],  # CASH OUT
])

# --- blinds --------------------------------------------------------------------
BLIND_SMALL, BLIND_BIG, BLIND_BOSS = 0, 1, 2
BLIND_NAMES  = ["Small Blind", "Big Blind", "Boss Blind"]
# the ante's base chips, ante 0..8 (misc_functions.lua get_blind_amount; ante 0 by Hieroglyph):
# row 0 for White / Red, row 1 (scaling 2) from Green, row 2 (scaling 3) from Purple
ANTE_BASE    = jnp.array([[100, 300, 800, 2000, 5000, 11000, 20000, 35000, 50000],
                          [100, 300, 900, 2600, 8000, 20000, 36000, 60000, 100000],
                          [100, 300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]], jnp.int32)
BLIND_MULT   = jnp.array([1.0, 1.5, 2.0], jnp.float32)
BLIND_REWARD = jnp.array([3, 4, 5], jnp.int32)
WIN_ANTE     = 8     # clearing the ante-8 boss wins the run
BONES_SHARE  = 0.25  # Mr. Bones saves a lost round with >= 25 % of the target

# --- economy -----------------------------------------------------------------
INTEREST_PER  = 5    # $1 per $5 held ...
INTEREST_CAP  = 5    # ... at most 5 times (Seed Money 10, Money Tree 20; To the Moon raises the $ per step)
REROLL_BASE   = 5    # +$1 per reroll, back to base on each shop visit (Reroll Surplus / Glut -$2 each)
SHOP_UNLOCKED = 2    # card slots open in the shop at run start (Overstock / Plus +1 each)
CREDIT_DEBT   = 20   # Credit Card: go down to -$20 per card
BASE_JOKER_SLOTS = 5
BASE_CONS_SLOTS = 2
BASE_HANDS = 4
BASE_DISCARDS = 3    # starting_params.discards; the Red Deck adds 1 (BACK_DISCARDS)

# Shop odds for a joker (roll_joker): a rarity 70 / 25 / 5 %, then a joker uniform among the
# available ones of that rarity. Legendaries never appear in the shop (only The Soul makes one).
J_FALLBACK = JOKER_ID["Joker"]   # what the shop offers when the pool is empty
J_MIME = JOKER_ID["Mime"]
# an empty Tarot / Planet / Spectral pool
CONS_FALLBACK = jnp.array([CONS_ID["Strength"], CONS_ID["Pluto"], CONS_ID["Incantation"]], jnp.int32)
_IS_TAROT = jnp.asarray(TAROTS)
_IS_PLANET = jnp.asarray(PLANETS)
_IS_SPECTRAL = jnp.asarray(SPECTRALS)
_SPECTRAL_POOL = jnp.asarray(SPECTRAL_POOL)

# fold_in purposes for state.key
K_PLAY, K_ROUND, K_ACQUIRE, K_USE, K_PACK, K_BLIND, K_SHOP = 1, 2, 3, 4, 5, 6, 7
K_SKIP, K_BOSS, K_TAG, K_VOUCHER, K_HOOK, K_DRAW, K_SELL, K_SHOP_EXIT = 8, 9, 10, 11, 12, 13, 14, 15
K_CASHOUT = 16


class balatroEnvState(NamedTuple):
    # jokers (compact: held jokers fill slots 0..n-1, in order)
    jokers       : jax.Array   # (32,)  int32   -- joker id per slot, J_EMPTY = vacant
    joker_value  : jax.Array   # (32, ) float   -- the value of the joker in each jokers[i] slot
    joker_state  : jax.Array   # (32, ) int32   -- To Do List hand, Loyalty Card start, Yorick's count
    joker_sell_value : jax.Array # (32,) int32  -- $ a SELL pays; mutable (Egg, Gift Card)
    joker_slots  : jax.Array   # scalar int32   -- slot count (5 + Antimatter + held Negatives, up to 8)
    joker_ed     : jax.Array   # (32,)  int32   -- ED_* per slot

    consumables  : jax.Array   # (16,)  int32   -- consumable id per slot (compact), -1 = vacant
    cons_sell    : jax.Array   # (16,)  int32   -- $ a SELL_CONS pays per slot (+ Gift Card); 0 = vacant
    consumable_slots :jax.Array # scalar int32  -- slots before the Negative consumables (2 + Crystal Ball)

    # the card instances (see the module doc)
    card_rank    : jax.Array   # (128,) int32
    card_suit    : jax.Array   # (128,) int32
    card_enh     : jax.Array   # (128,) int32   -- ENH_*
    card_ed      : jax.Array   # (128,) int32   -- ED_* (Negative never)
    card_seal    : jax.Array   # (128,) int32   -- SEAL_*
    card_bonus   : jax.Array   # (128,) float32 -- perma bonus chips (Hiker)
    card_live    : jax.Array   # (128,) bool    -- the slot holds a card of the deck

    # hand
    hand         : jax.Array   # (32,)  int32   -- card instance per lane, -1 = empty
    hand_size    : jax.Array   # scalar int32   -- number of lanes currently in play
    hand_valid   : jax.Array   # (32,)  bool    -- which lanes hold a live card

    # deck
    deck         : jax.Array   # (128,) int32   -- permutation of instance slots, live first
    deck_size    : jax.Array   # scalar int32   -- live cards; rest are free slots
    deck_ptr     : jax.Array   # scalar int32   -- next card to deal
    deck_start   : jax.Array   # scalar int32   -- the cards at the run start (starting_deck_size: Erosion)

    # budget
    handcount    : jax.Array   # scalar int32   -- hands granted this round
    discardcount : jax.Array   # scalar int32   -- discards granted this round
    hands        : jax.Array   # scalar int32   -- hands remaining
    discards     : jax.Array   # scalar int32   -- discards remaining
    base_hand_size : jax.Array # scalar int32   -- hand size before jokers (the deck's, vouchers, Ouija ...)
    base_hands   : jax.Array   # scalar int32   -- hands per round before jokers (+ Grabber ...)
    base_discards : jax.Array  # scalar int32   -- discards per round before jokers (+ Wasteful ...)
    discards_used : jax.Array  # scalar int32   -- discards spent this round (Delayed Gratification)

    # round
    round        : jax.Array   # scalar int32   -- blinds played this run (a skipped one does not count)
    ante         : jax.Array   # scalar int32
    blind_index  : jax.Array   # scalar int32   -- BLIND_SMALL / BIG / BOSS: the one on deck or in play
    chips_scored : jax.Array   # scalar int32   -- progress toward blind
    blind_target : jax.Array   # scalar int32   -- chips needed to clear
    target_scale : jax.Array   # scalar float32 -- curriculum: x every blind target (1.0 = real game)
    key          : jax.Array   # PRNG key (new-style typed key)
    hand_played  : jax.Array   # (12,) int32    -- times each hand type was played this run
    hand_round   : jax.Array   # (12,) int32    -- ... this round
    hand_level   : jax.Array   # (12,) int32    -- each hand type's level (Planets), from 1
    last_hand    : jax.Array   # scalar int32   -- the last hand type played (Blue Seal), -1 = none
    idol_card    : jax.Array   # scalar int32   -- The Idol's card this round (rank + 13 * suit)
    mail_rank    : jax.Array   # scalar int32   -- Mail-In Rebate's rank this round
    ancient_suit : jax.Array   # scalar int32   -- Ancient Joker's suit this round
    castle_suit  : jax.Array   # scalar int32   -- Castle's suit this round

    # run memory of the consumables
    last_cons    : jax.Array   # scalar int32   -- the last Tarot / Planet used (The Fool), -1 = none
    tarots_used  : jax.Array   # scalar int32   -- Fortune Teller
    planets_used : jax.Array   # (12,) bool     -- per hand type: its Planet was used (Satellite)

    # shop
    shop_slots  : jax.Array    # (4, ) int32    -- item id per shop slot (by shop_kind), -1 = empty or locked
    shop_kind   : jax.Array    # (4, ) int32    -- IK_JOKER / IK_TAROT / IK_PLANET / IK_CARD
    shop_ed     : jax.Array    # (4, ) int32    -- a joker's or a playing card's edition
    shop_size   : jax.Array    # scalar int32   -- unlocked shop slots (lanes >= this stay -1)
    pack_slots  : jax.Array    # (2, ) int32    -- pack id per pack slot, -1 = bought
    reroll_cost : jax.Array    # scalar int32
    free_rerolls : jax.Array   # scalar int32   -- Chaos the Clown rerolls left this shop
    gros_extinct : jax.Array   # scalar bool    -- Gros Michel died: Cavendish may appear
    buffoon_given : jax.Array  # scalar bool    -- the first shop's free Buffoon slot is used up
    game_stage  : jax.Array    # scalar int32   -- STAGE_*
    money       : jax.Array    # scalar int32   -- below 0 only with Credit Card (or The Tooth)
    last_cashout: jax.Array    # scalar int32   -- $ paid at the last blind clear (display)

    # the open pack (STAGE_PACK)
    pack_id     : jax.Array    # scalar int32   -- -1 = no pack open
    pack_left   : jax.Array    # scalar int32   -- cards still to take
    pack_items  : jax.Array    # (5,) int32     -- joker id / consumable id per card; -1 = taken or none
    pack_ed     : jax.Array    # (5,) int32     -- a joker's or a playing card's edition
    pack_rank   : jax.Array    # (5,) int32     -- Standard: the playing card (pack_items = 0 for a card)
    pack_suit   : jax.Array    # (5,) int32
    pack_enh    : jax.Array    # (5,) int32
    pack_seal   : jax.Array    # (5,) int32

    # the boss of this ante and its effects in its round
    boss         : jax.Array   # scalar int32   -- this ante's Boss Blind (items.BK_*)
    bosses_used  : jax.Array   # (28,) int32    -- times each boss was chosen (the least used goes next)
    boss_rerolled : jax.Array  # scalar bool    -- rerolled this ante (Director's Cut: once)
    boss_disabled : jax.Array  # scalar bool    -- Chicot / Luchador / Verdant Leaf: off for this round
    boss_hand    : jax.Array   # scalar int32   -- The Mouth: the one hand type this round (-1 none yet)
    ox_hand      : jax.Array   # scalar int32   -- The Ox: the most played hand at the last boss beaten
    boss_sub     : jax.Array   # scalar int32   -- The Water's discards / The Needle's hands, given back on disable
    forced_card  : jax.Array   # scalar int32   -- Cerulean Bell: the card always selected (instance, -1)
    card_played_ante : jax.Array # (128,) bool  -- played this ante (The Pillar)
    hand_down    : jax.Array   # (32,) bool     -- lanes drawn face down (Wheel, House, Mark, Fish)
    card_down    : jax.Array   # (128,) bool    -- the deck view counts it as in the deck: drawn face down
                               #                   this round and not played or turned up (wheel_flipped)
    joker_debuff : jax.Array   # (32,) bool     -- Crimson Heart: the disabled joker
    joker_down   : jax.Array   # (32,) bool     -- Amber Acorn: face-down jokers
    joker_seen   : jax.Array   # (32,) bool     -- a face-down joker that showed a message this round

    # skips and tags
    blind_tags   : jax.Array   # (2,) int32     -- the tag of the Small / Big Blind this ante
    orbital_hands : jax.Array  # (2,) int32     -- an Orbital Tag's hand there
    tags         : jax.Array   # (8,) int32     -- held tags (compact), -1 = none
    tag_hand     : jax.Array   # (8,) int32     -- a held Orbital Tag's hand
    skips        : jax.Array   # scalar int32   -- blinds skipped this run (Throwback, Skip Tag)
    unused_discards : jax.Array # scalar int32  -- discards left at the round ends (Garbage Tag)
    pack_return  : jax.Array   # scalar int32   -- the stage an open pack goes back to
    round_hand_bonus : jax.Array # scalar int32 -- Juggle Tag: hand size for this round

    # vouchers, coupons, shop playing cards, Negative consumables
    vouchers     : jax.Array   # (32,) bool     -- redeemed
    ante_voucher : jax.Array   # scalar int32   -- this ante's voucher in the shop, -1 = redeemed
    shop_vouchers : jax.Array  # (4,) int32     -- the vouchers of this shop, -1 = redeemed / none
    shop_free    : jax.Array   # (4,) bool      -- couponed shop cards (cost $0)
    pack_free    : jax.Array   # (2,) bool      -- couponed packs
    shop_rank    : jax.Array   # (4,) int32     -- a playing card in the shop (Magic Trick)
    shop_suit    : jax.Array   # (4,) int32
    shop_enh     : jax.Array   # (4,) int32
    cons_ed      : jax.Array   # (16,) int32    -- ED_NEG for a Negative consumable (Perkeo), else 0
    ecto_minus   : jax.Array   # scalar int32   -- the hand size the next Ectoplasm takes (1, 2, 3 ...)
    reroll_inc   : jax.Array   # scalar int32   -- paid rerolls this shop (current_round.reroll_cost_increase)
    reroll_d6    : jax.Array   # scalar bool    -- a D6 Tag made this shop's rerolls start at $0 (temp_reroll_cost)

    # the stake and the joker stickers (STK_* bits: Eternal, Perishable, Rental)
    stake        : jax.Array   # scalar int32   -- 1 White .. 8 Gold (items.STAKE_NAMES)
    back         : jax.Array   # scalar int32   -- the deck: 0 Red .. 14 Erratic (items.BACK_NAMES)
    joker_sticker : jax.Array  # (32,) int32    -- the stickers of each held joker
    joker_perish : jax.Array   # (32,) int32    -- a Perishable joker's rounds left (5 .. 1; 0: debuffed for good)
    shop_sticker : jax.Array   # (4,) int32     -- the stickers of a shop joker
    pack_sticker : jax.Array   # (5,) int32     -- the stickers of a Buffoon pack joker


class balatroAction(NamedTuple):
    verb   : jax.Array                    # scalar int32 -- V_*
    select : jax.Array                    # (32,)  bool  -- which lanes to act on
    target : jax.Array = jnp.int32(0)     # scalar int32 -- shop / joker / consumable / pack slot, MOVE source
    dest   : jax.Array = jnp.int32(0)     # scalar int32 -- MOVE destination (read only by the MOVE verbs)


class PlayResult(NamedTuple):
    """Everything one scored hand produced. The GUI and the tests read this."""
    handtype   : jax.Array   # scalar int32  -- index into HAND_NAMES
    sel        : jax.Array   # (32,)  bool   -- the lanes that were actually played
    cards      : jax.Array   # (5,)   int32  -- the played card instances, lane order
    valid      : jax.Array   # (5,)   bool
    mask       : jax.Array   # (5,)   bool   -- which played cards scored
    base_chips : jax.Array   # scalar        -- before the joker pass
    base_mult  : jax.Array
    chips      : jax.Array   # scalar float32 -- after the joker pass
    mult       : jax.Array
    score      : jax.Array   # scalar int32  -- floor(chips * mult)
    money      : jax.Array   # scalar        -- $ the hand earned (To Do List, Gold Seal, Rough Gem ...)


def tree_where(c, a, b):
    """Select a whole pytree: a where c, else b. The branch-free 'if'."""
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


COMPACT_DIV = 8     # compacted: a pass runs f on batch / COMPACT_DIV envs (8 = the fastest rollout at 1024 envs)


def _leafwise(fn, *trees):
    """jax.tree.map(fn, *trees), with PRNG key leaves as their raw key data."""
    def one(*xs):
        if jnp.issubdtype(xs[0].dtype, jax.dtypes.prng_key):
            impl = jax.random.key_impl(xs[0])
            return jax.random.wrap_key_data(fn(*(jax.random.key_data(x) for x in xs)), impl=impl)
        return fn(*xs)
    return jax.tree.map(one, *trees)


def compacted(f):
    """g(need, args) == tree_where(need, f(*args), args[0]), where f(*args) has the structure
    and dtypes of args[0]; f runs only for the envs that need it. One env: lax.cond. Under
    vmap: the envs with need are gathered (in batch order) into a buffer of
    ceil(batch / COMPACT_DIV) envs, vmap(f) runs on the buffer and the results are scattered
    back; a while loop does one pass per buffer, so any count of envs is exact."""
    def run(*a):
        out = f(*a)
        return jax.tree.map(lambda o, x: o if jnp.issubdtype(x.dtype, jax.dtypes.prng_key) else o.astype(x.dtype),
                            out, a[0])

    @jax.custom_batching.custom_vmap
    def g(need, args):
        return jax.lax.cond(need, lambda a: run(*a), lambda a: a[0], args)

    @g.def_vmap
    def _rule(axis_size, in_batched, need, args):
        n_b, a_b = in_batched
        need = need if n_b else jnp.broadcast_to(need, (axis_size,))
        args = jax.tree.map(lambda x, b: x if b else _leafwise(lambda y: jnp.broadcast_to(y, (axis_size,) + y.shape), x),
                            args, a_b)
        k = max(1, -(-axis_size // COMPACT_DIV))
        n = need.sum()
        idx = jnp.nonzero(need, size=axis_size + k, fill_value=0)[0]     # the envs with need first
        f_b = jax.vmap(run)

        def body(carry):
            c, out = carry
            lanes = jax.lax.dynamic_slice(idx, (c * k,), (k,))
            dst = jnp.where(c * k + jnp.arange(k) < n, lanes, axis_size)      # out of range = dropped
            res = f_b(*_leafwise(lambda x: x[lanes], args))
            return c + 1, _leafwise(lambda o, r: o.at[dst].set(r, mode="drop"), out, res)
        _, out = jax.lax.while_loop(lambda carry: carry[0] * k < n, body, (jnp.int32(0), args[0]))
        return out, jax.tree.map(lambda _: True, out)
    return g


def blind_mult(blind_index, boss, disabled=False):
    """x the ante's base: 1 / 1.5 for Small / Big, the boss's own (2, The Wall 4, The Needle 1,
    Violet Vessel 6). A disabled Wall / Vessel drop to 2 (blind.lua disable)."""
    b = BOSS_MULT[jnp.clip(boss, 0, N_BOSSES - 1)]
    b = jnp.where(disabled & ((boss == BK_WALL) | (boss == BK_VESSEL)), 2.0, b)
    return jnp.where(blind_index == BLIND_BOSS, b, BLIND_MULT[jnp.clip(blind_index, 0, 1)])


def blind_target(ante, blind_index, scale=1.0, boss=0, disabled=False, stake=1):
    """Chips needed: ante base (the stake's scaling) x the blind's mult (blind_mult), x the
    curriculum scale. Never below 1 chip, so a DISCARD can never clear a blind. Ante < 1
    (Hieroglyph): 100."""
    row = jnp.where(stake >= ST_SCALING_3, 2, jnp.where(stake >= ST_SCALING_2, 1, 0))
    base = ANTE_BASE[row, jnp.clip(ante, 0, ANTE_BASE.shape[1] - 1)]
    return jnp.maximum(jnp.floor(base * blind_mult(blind_index, boss, disabled) * scale), 1).astype(jnp.int32)


def run_scale(state):
    """The scale on every blind target: the curriculum's target_scale x the deck's ante_scaling
    (Plasma Deck: X2; Blind:set_blind get_blind_amount * mult * ante_scaling)."""
    return state.target_scale * BACK_ANTE_SCALING[jnp.clip(state.back, 0, N_BACKS - 1)]


def blind_dollars(blind_index, boss, stake=1):
    """$ for beating it: 3 / 4 / the boss's (5, a showdown boss 8); from the Red Stake the
    Small Blind pays $0 (no_blind_reward)."""
    pay = jnp.where(blind_index == BLIND_BOSS, BOSS_DOLLARS[jnp.clip(boss, 0, N_BOSSES - 1)],
                    BLIND_REWARD[jnp.clip(blind_index, 0, 2)])
    return jnp.where((blind_index == BLIND_SMALL) & (stake >= ST_NO_SMALL_REWARD), 0, pay)


# ---------------------------------------------------------------------------
# VOUCHERS — what the redeemed ones change (card.lua apply_to_run)
# ---------------------------------------------------------------------------
def discount(state):
    """Clearance Sale 25 %, Liquidation 50 %."""
    v = state.vouchers
    return jnp.where(v[V_LIQUIDATION], 50, jnp.where(v[V_CLEARANCE], 25, 0)).astype(jnp.int32)


def edition_rate(state):
    """Hone 2, Glow Up 4 (G.GAME.edition_rate)."""
    v = state.vouchers
    return jnp.where(v[V_GLOW_UP], 4.0, jnp.where(v[V_HONE], 2.0, 1.0))


def interest_steps(state):
    """The most $1 interest steps: $25 / 5, Seed Money $50 / 5, Money Tree $100 / 5."""
    v = state.vouchers
    return jnp.where(v[V_MONEY_TREE], 20, jnp.where(v[V_SEED_MONEY], 10, INTEREST_CAP)).astype(jnp.int32)


def reroll_base(state):
    """The first reroll's $: 5, -2 per Reroll Surplus / Glut."""
    v = state.vouchers
    return (REROLL_BASE - 2 * v[V_REROLL_SURPLUS] - 2 * v[V_REROLL_GLUT]).astype(jnp.int32)


def base_joker_slots(state):
    """starting_params.joker_slots: 5, Black Deck +1, Painted Deck -1."""
    return BASE_JOKER_SLOTS + BACK_JOKER_SLOT[jnp.clip(state.back, 0, N_BACKS - 1)]


def shop_rates(state):
    """create_card_for_shop, by IK_*: joker 20, Tarot 4 (Merchant 9.6, Tycoon 32), Planet 4 (9.6,
    32), playing card 0 (Magic Trick / Illusion 4), Spectral 0 (Ghost Deck 2)."""
    v = state.vouchers
    tarot = jnp.where(v[V_TAROT_TYCOON], 32.0, jnp.where(v[V_TAROT_MERCHANT], 9.6, TAROT_RATE))
    planet = jnp.where(v[V_PLANET_TYCOON], 32.0, jnp.where(v[V_PLANET_MERCHANT], 9.6, PLANET_RATE))
    card = jnp.where(v[V_MAGIC_TRICK] | v[V_ILLUSION], 4.0, 0.0)
    spectral = jnp.where(state.back == B_GHOST, GHOST_SPECTRAL_RATE, 0.0)
    return jnp.stack([jnp.float32(JOKER_RATE), tarot, planet, card, spectral])


# ---------------------------------------------------------------------------
# CARD INSTANCES
# ---------------------------------------------------------------------------
def card_id(state, inst):
    """rank + 13 * suit of instance(s) inst (-1 stays -1). The display / Idol id."""
    i = jnp.clip(inst, 0, MAX_DECK_SIZE - 1)
    return jnp.where(inst >= 0, state.card_rank[i] + 13 * state.card_suit[i], -1)


def deck_counts(state):
    """Full-deck tallies (G.playing_cards: the hand included)."""
    live = state.card_live
    enh = state.card_enh
    return dict(steel=(live & (enh == ENH_STEEL)).sum(), stone=(live & (enh == ENH_STONE)).sum(),
                enhanced=(live & (enh != ENH_NONE)).sum(),
                nines=(live & (enh != ENH_STONE) & (state.card_rank == 7)).sum(),
                size=live.sum())


def remove_cards(state, gone) -> "balatroEnvState":
    """Destroy the instances in gone (128,): free their slots, take them out of
    the deck order and the hand."""
    gone = gone & state.card_live
    live = state.card_live & ~gone
    pos = jnp.arange(MAX_DECK_SIZE)
    keep = live[state.deck] & (pos < state.deck_size)
    hand_gone = state.hand_valid & gone[jnp.clip(state.hand, 0, MAX_DECK_SIZE - 1)]
    return state._replace(
        card_live=live,
        deck=compact(state.deck, keep),
        deck_size=keep.sum().astype(jnp.int32),
        deck_ptr=(keep & (pos < state.deck_ptr)).sum().astype(jnp.int32),
        hand=jnp.where(hand_gone, -1, state.hand),
        hand_valid=state.hand_valid & ~hand_gone,
        hand_down=state.hand_down & ~hand_gone,
        forced_card=jnp.where(gone[jnp.clip(state.forced_card, 0, MAX_DECK_SIZE - 1)], -1, state.forced_card),
    )


def add_card(state, rank, suit, enh, ed, seal, bonus=0.0, to_hand=False, ok=True, played=False):
    """A new card instance in the first free slot. to_hand: it goes to the end of the
    hand (G.hand:emplace appends): the hand's cards close up, order kept, and the new card
    follows the last one (it may go past hand_size, as in the game); else it goes to the
    bottom of the undealt deck. played: it has The Pillar mark (copy_card of a marked card).
    -> (state, added). No free slot: nothing."""
    free = ~state.card_live
    slot = jnp.argmax(free).astype(jnp.int32)
    held = state.hand_valid
    lane = held.sum().astype(jnp.int32)
    hand_c = compact(jnp.where(held, state.hand, -1), held)
    valid_c, down_c = compact(held, held), compact(state.hand_down & held, held)
    ok = ok & free.any() & (~to_hand | (lane < MAX_HAND_SIZE))
    pos = jnp.arange(MAX_DECK_SIZE)
    at = jnp.where(to_hand, state.deck_ptr, state.deck_size)           # where it enters the order
    # sort keys: the cards from `at` on move one place back, the free slot takes `at`
    keys = jnp.where(state.deck == slot, at, jnp.where(pos < at, pos, pos + 1))
    order = state.deck[jnp.argsort(keys)]
    got = state._replace(
        card_rank=state.card_rank.at[slot].set(rank), card_suit=state.card_suit.at[slot].set(suit),
        card_enh=state.card_enh.at[slot].set(enh), card_ed=state.card_ed.at[slot].set(ed),
        card_seal=state.card_seal.at[slot].set(seal),
        card_bonus=state.card_bonus.at[slot].set(bonus), card_live=state.card_live.at[slot].set(True),
        card_played_ante=state.card_played_ante.at[slot].set(played),
        deck=order, deck_size=state.deck_size + 1,
        deck_ptr=state.deck_ptr + to_hand,
        hand=jnp.where(to_hand, hand_c.at[lane].set(slot), state.hand),
        hand_valid=jnp.where(to_hand, valid_c.at[lane].set(True), state.hand_valid),
        hand_down=jnp.where(to_hand, down_c.at[lane].set(False), state.hand_down),
    )
    return tree_where(ok, got, state), ok


# ---------------------------------------------------------------------------
# THE BOSS — is it on, which cards and jokers it debuffs
# ---------------------------------------------------------------------------
def boss_active(state):
    """The boss's ability is on: its round is being played (STAGE_END too: the end of the round
    comes before evaluate_round's Blind:defeat) and it is not disabled."""
    in_round = (state.game_stage == STAGE_BLIND) | (state.game_stage == STAGE_END)
    return in_round & (state.blind_index == BLIND_BOSS) & ~state.boss_disabled


def boss_is(state, b):
    return boss_active(state) & (state.boss == b)


def acting(state):
    """The joker ids that act: the joker Crimson Heart disabled and a perished one are out
    (they still count for Abstract Joker, Joker Stencil and Swashbuckler)."""
    return jnp.where(state.joker_debuff | perished(state), J_EMPTY, state.jokers)


def has_sticker(stickers, bit):
    return (stickers & bit) > 0


def perished(state):
    """(J,) bool — a Perishable joker after its last round: debuffed for good (set_debuff), its
    hand size / discards / Credit Card ... gone (remove_from_deck), a Negative's slot kept."""
    return (state.jokers != J_EMPTY) & has_sticker(state.joker_sticker, STK_PERISH) & (state.joker_perish <= 0)


def eternal(state):
    """(J,) bool — an Eternal joker: it cannot be sold (can_sell_card) or destroyed (Madness,
    Ceremonial Dagger, Hex, Ankh)."""
    return (state.jokers != J_EMPTY) & has_sticker(state.joker_sticker, STK_ETERNAL)


def roll_sticker(key, stake, jid):
    """create_card's stickers for a shop or Buffoon pack joker: one poll is Eternal above 0.7
    (Black Stake on; never on a joker that cannot be Eternal) else Perishable in 0.4 .. 0.7
    (Orange; with the same rule); a second poll is Rental above 0.7 (Gold; any joker)."""
    k1, k2 = jax.random.split(key)
    u, u2 = jax.random.uniform(k1), jax.random.uniform(k2)
    j = jnp.clip(jid, 0, N_JOKERS - 1)
    et = (stake >= ST_ETERNAL) & (u > STICKER_POLL) & JOKER_ETERNAL_OK[j]
    pe = (stake >= ST_PERISH) & (u > PERISH_POLL) & (u <= STICKER_POLL) & JOKER_PERISH_OK[j]
    re = (stake >= ST_RENTAL) & (u2 > STICKER_POLL)
    return (et * STK_ETERNAL + pe * STK_PERISH + re * STK_RENTAL).astype(jnp.int32)


def mods_of(state):
    """The rule mods of the jokers that act (find_joker skips a debuffed joker)."""
    return rule_mods(acting(state))


def debuff_rule(state, r, s, e, marked, mods=None):
    """(...,) bool — blind.lua debuff_card for a card of rank r, suit s, enhancement e and The
    Pillar mark marked: the suit bosses (Card:is_suit with bypass: a Wild card has every suit, a
    Stone card none; Smeared Joker), The Plant (face cards; Pareidolia: every card), The Pillar
    (marked), Verdant Leaf (every card). Only while the boss is on."""
    mods = mods_of(state) if mods is None else mods
    b = state.boss
    bs = BOSS_SUIT[jnp.clip(b, 0, N_BOSSES - 1)]
    suit_hit = (bs >= 0) & suit_is(s, bs, mods.smeared, e)
    face_hit = (b == BK_PLANT) & is_face_card(r, mods.pareidolia, e)
    pillar = (b == BK_PILLAR) & marked
    return boss_active(state) & (suit_hit | face_hit | pillar | (b == BK_LEAF))


def card_debuffed(state, inst, mods=None):
    """(...,) bool — debuff_rule for card instances inst (The Pillar reads card_played_ante)."""
    i = jnp.clip(inst, 0, MAX_DECK_SIZE - 1)
    return (inst >= 0) & debuff_rule(state, state.card_rank[i], state.card_suit[i], state.card_enh[i],
                                      state.card_played_ante[i], mods)


def build_ctx(state) -> jax.Array:
    held = (state.jokers != J_EMPTY).sum()
    stencils = (state.jokers == J_STENCIL).sum()
    d = deck_counts(state)
    return (jnp.zeros(N_QUERY_KINDS, jnp.float32)
    .at[Q_MONEY_DIV_5].set(jnp.maximum(state.money // 5, 0))
    # a held Stencil counts as an empty slot ("Joker Stencil included"); at 1 or less it gives
    # no Xmult (joker_main needs x_mult > 1), so X1 even with more jokers than slots
    .at[Q_EMPTY_JOKER_SLOTS].set(jnp.maximum(state.joker_slots - held + stencils, 1))
    .at[Q_DECK_NINE_COUNT].set(d["nines"])
    .at[Q_DISCARDS_LEFT].set(jnp.maximum(state.discards, 0))
    .at[Q_NO_DISCARDS_LEFT].set(state.discards <= 0)
    .at[Q_DECK_LEFT].set(jnp.maximum(state.deck_size - state.deck_ptr, 0))
    .at[Q_JOKER_COUNT].set(held)
    .at[Q_MONEY].set(jnp.maximum(state.money, 0))
    .at[Q_OTHER_SELL].set(state.joker_sell_value.sum())        # minus its own, per slot
    .at[Q_DECK_LOST].set(jnp.maximum(state.deck_start - d["size"], 0))
    .at[Q_DELAYED_GRAT].set(jnp.where(state.discards_used == 0, jnp.maximum(state.discards, 0), 0))
    .at[Q_DECK_STEEL_COUNT].set(d["steel"])
    .at[Q_DECK_STONE_COUNT].set(d["stone"])
    .at[Q_TAROTS_USED].set(state.tarots_used)
    .at[Q_UNIQUE_PLANETS].set(state.planets_used.sum())
    .at[Q_DRIVER].set(d["enhanced"] >= 16)
    .at[Q_SKIPS].set(state.skips)
    )


def live_v(state, joker_value=None) -> jax.Array:
    """v per slot as scoring reads it now: copies resolved, Swashbuckler's own
    sell value removed. -> (MAX_JOKERS,) float32"""
    joker_value = state.joker_value if joker_value is None else joker_value
    erows, eok, ev, _ = effective(acting(state), joker_value, state.joker_state)
    v = resolve(erows, eok, ev, build_ctx(state))
    own = state.joker_sell_value[jnp.clip(copy_sources(acting(state)), 0, MAX_JOKERS - 1)]
    return v - jnp.where(eok & (erows.QUERY == Q_OTHER_SELL), erows.VALUE * own, 0.0)


def targets(state) -> Targets:
    return Targets(state.idol_card, state.mail_rank, state.ancient_suit, state.castle_suit)


def affordable(state, cost):
    """can_buy / can_open / can_reroll: a $0 item always; else down to bankrupt_at (with Credit Card
    below $0). The Tooth can leave the money below $0."""
    return (cost <= 0) | (state.money - cost >= bankrupt_at(state))


def bankrupt_at(state):
    """The lowest money a purchase may leave: 0, or -$20 per Credit Card."""
    return -CREDIT_DEBT * (acting(state) == J_CREDIT).sum()


def _jokers(state):
    """(rows, active, ctx) — the three inputs every joker pass needs (the acting jokers)."""
    ids = acting(state)
    return rows_for(ids), ids != J_EMPTY, build_ctx(state)


def _gather(state, inst):
    i = jnp.clip(inst, 0, MAX_DECK_SIZE - 1)
    return (state.card_rank[i], state.card_suit[i], state.card_enh[i], state.card_ed[i],
            state.card_seal[i], state.card_bonus[i])


def forced_lane(state):
    """(32,) bool — the lane of Cerulean Bell's forced card (none when it is off)."""
    on = boss_is(state, BK_BELL) & (state.forced_card >= 0)
    return on & state.hand_valid & (state.hand == state.forced_card)


def fix_select(state, select, n):
    """The lanes an action takes: the first n valid selected lanes; with Cerulean Bell
    the forced card is always one of them (and the first n - 1 others)."""
    forced = forced_lane(state)
    has = forced.any()
    others = first_n(select.astype(bool) & state.hand_valid & ~forced, n - has)
    return others | forced


def observatory(state):
    """(12,) the held Planet cards per hand type, when Observatory is redeemed (each X1.5)."""
    held = state.consumables >= 0
    c = jnp.clip(state.consumables, 0, N_CONS - 1)
    planet = held & (CONS_KIND[c] == CK_PLANET)
    n = jnp.zeros(N_HAND_TYPES, jnp.int32).at[jnp.clip(CONS_PARAM[c], 0, N_HAND_TYPES - 1)].add(planet)
    return jnp.where(state.vouchers[V_OBSERVATORY], n, 0)


def boss_hand_rules(state, ht, n_cards):
    """debuff_hand of the boss for hand type ht of n_cards cards (before the play counts):
    -> (debuffed hand, The Arm lowers a level, The Ox takes the money). Each one triggers
    the boss (Matador)."""
    psychic = boss_is(state, BK_PSYCHIC) & (n_cards < 5)
    eye = boss_is(state, BK_EYE) & (state.hand_round[ht] > 0)
    mouth = boss_is(state, BK_MOUTH) & (state.boss_hand >= 0) & (state.boss_hand != ht)
    arm = boss_is(state, BK_ARM) & (state.hand_level[ht] > 1)
    ox = boss_is(state, BK_OX) & (ht == state.ox_hand)
    return psychic | eye | mouth, arm, ox


def _play_in(state, select):
    """The PlayIn of a PLAY of select, with the boss's rules applied the way the game
    does before scoring: The Hook's discard is NOT here (env_step does it), The Tooth's
    $ and The Ox's $0 and The Arm's level are. -> (PlayIn, sel, cards, valid, hand info)"""
    sel = fix_select(state, select, 5)
    cards, valid = compact(state.hand, sel)[:5], compact(sel, sel)[:5]
    held_ok = state.hand_valid & ~sel
    mods = mods_of(state)
    r, s, e, ed, seal, bonus = _gather(state, cards)
    deb = card_debuffed(state, cards, mods) & valid
    # Midas Mask / Vampire (BEFORE) call set_ability, which runs debuff_card again: the card is
    # then a Gold / plain card (Wild and Stone gone) without The Pillar mark
    touch = debuff_rule(state, r, s, ENH_NONE, False, mods) & valid
    hr, hs, he, _, hseal, _ = _gather(state, state.hand)
    hdeb = card_debuffed(state, state.hand, mods) & held_ok
    # flush detection: a debuffed Wild card has only its own suit (Card:is_suit flush_calc)
    det_e = jnp.where(deb & (e == ENH_WILD), ENH_NONE, e)
    ht = detect(r, s, valid, mods, det_e).handtype
    n = valid.sum()
    debuffed, arm, ox = boss_hand_rules(state, ht, n)
    money = state.money - boss_is(state, BK_TOOTH) * n                 # The Tooth: $1 per card played
    money = jnp.where(ox, 0, money)                                     # The Ox: money to $0
    level = state.hand_level.at[ht].add(-arm.astype(jnp.int32))        # The Arm: -1 level
    ctx = build_ctx(state._replace(money=money))
    ids = acting(state)
    p = PlayIn(r, s, valid, hr, hs, held_ok,
               ids, state.joker_value, state.joker_state, ctx,
               money.astype(jnp.float32), state.hands - 1, state.hand_played,
               state.hand_round, state.blind_index == BLIND_BOSS, targets(state),
               e, ed, seal, bonus, he, hseal, level, jnp.where(state.joker_debuff, ED_NONE, state.joker_ed),
               debuff=deb, held_debuff=hdeb, hand_debuffed=debuffed, boss_trigger=arm | ox,
               flint=boss_is(state, BK_FLINT), observatory=observatory(state),
               det_enh=det_e, all_ids=state.jokers, card_room=(~state.card_live).sum().astype(jnp.int32),
               touch_debuff=touch, plasma=state.back == B_PLASMA)
    return p, sel, cards, valid, dict(ht=ht, debuffed=debuffed, arm=arm, ox=ox, money=money, level=level)


def score_play(state: balatroEnvState, select):
    """Score the selected cards. Pure, no state change. -> (PlayResult, PlayOut, info)
    The Hook's 2 cards leave the hand before the scoring (state_events: press_play),
    so score the state after that discard (see _hooked)."""
    p, sel, cards, valid, info = _play_in(state, select)
    out = _play(p, mods_of(state), jax.random.fold_in(state.key, K_PLAY), state.joker_sell_value)
    res = PlayResult(out.handtype, sel, cards, valid, out.mask, out.base_chips, out.base_mult,
                     out.chips, out.mult, out.score, out.money)
    return res, out, info


def _hooked(state, select):
    """The state after The Hook discards 2 random held cards (if it is on), and the
    discarded lanes. -> (state, hook lanes (32,) bool)"""
    sel = fix_select(state, select, 5)
    held = state.hand_valid & ~sel
    u = jax.random.uniform(jax.random.fold_in(state.key, K_HOOK), (MAX_HAND_SIZE,))
    order = jnp.argsort(jnp.where(held, u, 2.0))                        # random order of the held lanes
    pick = jnp.zeros(MAX_HAND_SIZE, bool).at[order[:2]].set(True) & held
    pick = pick & boss_is(state, BK_HOOK)
    return pick


@jit
def explain_play(state: balatroEnvState, select) -> PlayResult:
    """What PLAY would score, exactly as env_step scores it (The Hook's cards out first)."""
    hook = _hooked(state, select)
    s = _hook_discard(state, hook)
    return score_play(s, select)[0]


@jit
def preview(state: balatroEnvState, select) -> PlayResult:
    """The hand the selection makes (the in-game preview): no Hook discard (the game's
    preview does not know it), the rest as explain_play."""
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
      - BUY / SELL / OPEN / REDEEM with a bad target, too little $, or no free slot -> no-op
      - BUY of a Tarot / Planet with the consumable slots full -> Buy and Use (a no-op when
        the card cannot be used in the shop: a card Tarot, The Emperor, The Fool ...)
      - SELL of an Eternal joker                    -> no-op
      - USE / PICK of a consumable that cannot be used now (wrong card count,
        no room, no hand) -> no-op
      - MOVE_JOKER unless target != dest and both are held slots (< held count) -> no-op
      - MOVE_CARD unless target != dest, both < hand_size, and lane target holds a card -> no-op
      - SKIP at the blind select of a Boss Blind; REROLL there without Director's Cut /
        Retcon (once per ante for Director's Cut) or $10 -> no-op
      - PLAY / DISCARD with no card selected        -> no-op (the game's buttons stay off)
      - more than 5 cards selected                  -> the first 5 valid lanes
        (Cerulean Bell: its forced card is always one of them)
    """
    verb = action.verb.astype(jnp.int32)
    verb = jnp.where((verb == V_DISCARD) & (state.discards <= 0), V_PLAY, verb)
    in_range = (verb >= 0) & (verb < N_VERBS)
    stage = jnp.clip(state.game_stage, 0, LEGAL.shape[0] - 1)
    legal = in_range & LEGAL[stage, jnp.clip(verb, 0, N_VERBS - 1)]
    no_cards = ((verb == V_PLAY) | (verb == V_DISCARD)) & ~fix_select(state, action.select, 5).any()
    legal = legal & ~no_cards
    branch = jnp.where(legal, verb, V_NOOP)
    action = action._replace(verb=verb, target=jnp.asarray(action.target, jnp.int32),
                             dest=jnp.asarray(action.dest, jnp.int32))
    # one branch each for PLAY / DISCARD and USE / PICK / Buy and Use (they share their heavy parts)
    branch = jnp.where(branch == V_DISCARD, V_PLAY, jnp.where(branch == V_PICK, V_USE, branch))
    branch = jnp.where((branch == V_BUY) & buy_and_use(state, action.target), V_USE, branch)
    # the two heavy rare parts run here, out of the switch, each only where its handler uses it
    # (compacted: under vmap only those envs pay for it; a custom_vmap cannot sit in a switch)
    use = _use_parts(state, action)
    pre = _Pre(started=_start_blind_c((branch == V_NEXT) & (_next_branch(state) == 1), (state,)),
               used=_apply_cons_c((branch == V_USE) & use["need"], use["args"]))
    nxt, _, _ = jax.lax.switch(branch, _SWITCH, state, action, pre)
    go_end = legal & (verb == V_NEXT) & (state.game_stage == STAGE_END)
    nxt, reward, over = _settle(nxt, legal & (verb == V_PLAY), go_end)
    nxt = _track_down(nxt)
    was_over = (state.game_stage == STAGE_LOST) | (state.game_stage == STAGE_WON)
    return nxt, reward, over & ~was_over


def _track_down(state):
    """card_down: + the face-down cards in the hand, - the face-up ones (drawn face up, or
    turned up when the boss is disabled), - the destroyed ones; nothing after a round (the
    round end clears ability.wheel_flipped; a lost run keeps its last hand as it was). A
    discarded face-down card stays."""
    on = lambda m: jnp.zeros(MAX_DECK_SIZE, bool).at[jnp.where(m, state.hand, MAX_DECK_SIZE)].set(True, mode="drop")
    down = on(state.hand_valid & state.hand_down)
    up = on(state.hand_valid & ~state.hand_down)
    stage = state.game_stage
    in_round = (stage == STAGE_BLIND) | (stage == STAGE_END) | (stage == STAGE_LOST) | (stage == STAGE_WON)
    return state._replace(card_down=((state.card_down | down) & ~up & state.card_live) & in_round)


def _no_change(state):
    return state, jnp.int32(0), jnp.bool_(False)


def remove_jokers(state: balatroEnvState, gone) -> balatroEnvState:
    """Take the jokers in the gone slots out (sold, eaten, extinct, destroyed). The rest
    shift left in order; the budget follows (remove_from_deck: a Drunkard takes its discard
    back, a Chaos the Clown its free reroll)."""
    gone = gone & (state.jokers != J_EMPTY)
    safe = jnp.clip(state.jokers, 0, N_JOKERS - 1)
    dead = perished(state)                          # it gave its discards / reroll back when it perished
    lost_d = jnp.where(gone & ~state.joker_debuff & ~dead, JOKERS.D_SIZE[safe], 0).sum()
    lost_chaos = (gone & (state.jokers == J_CHAOS) & ~dead).sum()
    jokers = jnp.where(gone, J_EMPTY, state.jokers)
    keep = jokers != J_EMPTY
    out = state._replace(
        jokers=compact(jokers, keep),
        joker_value=compact(jnp.where(gone, 0.0, state.joker_value), keep),
        joker_state=compact(jnp.where(gone, 0, state.joker_state), keep),
        joker_sell_value=compact(jnp.where(gone, 0, state.joker_sell_value), keep),
        joker_ed=compact(jnp.where(gone, ED_NONE, state.joker_ed), keep),
        joker_debuff=compact(state.joker_debuff & ~gone, keep),
        joker_down=compact(state.joker_down & ~gone, keep),
        joker_seen=compact(state.joker_seen & ~gone, keep),
        joker_sticker=compact(jnp.where(gone, 0, state.joker_sticker), keep),
        joker_perish=compact(jnp.where(gone, 0, state.joker_perish), keep),
        discards=jnp.maximum(state.discards - lost_d, 0),
        free_rerolls=jnp.maximum(state.free_rerolls - lost_chaos, 0),
    )
    return reprice(apply_budget(out), state)


def round_hands(state):
    """round_resets.hands: the base hands + every acting joker's (Troubadour: -1 each). It can be
    0 or less; a round still starts with at least 1 hand (new_round: math.max(1, ...))."""
    ids = acting(state)
    rows, active = rows_for(ids), ids != J_EMPTY
    return state.base_hands + jnp.where(active, rows.H_PLAYS, 0).sum()


def apply_budget(state: balatroEnvState) -> balatroEnvState:
    """Hand size, hands and discards per round = the base + every acting joker
    (Juggler, Merry Andy, Troubadour, Turtle Bean ...) + Juggle Tag - The Manacle; joker
    slots = 5 + Antimatter + the held Negatives (up to MAX_JOKER_SLOTS). The cards in hand
    stay; a smaller hand only draws fewer cards."""
    ids = acting(state)
    rows, active = rows_for(ids), ids != J_EMPTY
    h = (jnp.where(active, rows.H_SIZE, 0)
         + jnp.where(active & rows.HSIZE_V, state.joker_value, 0.0).astype(jnp.int32)).sum()
    d = jnp.where(active, rows.D_SIZE, 0).sum()
    neg = ((state.jokers != J_EMPTY) & (state.joker_ed == ED_NEG)).sum()
    manacle = boss_is(state, BK_MANACLE)
    slots = base_joker_slots(state) + state.vouchers[V_ANTIMATTER] + neg
    return fit_hand(state._replace(
        joker_slots=jnp.minimum(slots, MAX_JOKER_SLOTS).astype(jnp.int32),
        hand_size=jnp.clip(state.base_hand_size + h + state.round_hand_bonus - manacle, 0,
                           MAX_HAND_SIZE).astype(jnp.int32),
        discardcount=jnp.maximum(state.base_discards + d, 0).astype(jnp.int32),
        handcount=jnp.maximum(round_hands(state), 1).astype(jnp.int32)))


def fit_hand(state: balatroEnvState) -> balatroEnvState:
    """A card in a lane at or past hand_size (the hand size went down): move
    the cards to the front lanes, order kept."""
    lane = jnp.arange(MAX_HAND_SIZE)
    over = (state.hand_valid & (lane >= state.hand_size)).any()
    hand = jnp.where(state.hand_valid, state.hand, -1)
    packed = state._replace(hand=compact(hand, state.hand_valid),
                            hand_down=compact(state.hand_down & state.hand_valid, state.hand_valid),
                            hand_valid=compact(state.hand_valid, state.hand_valid))
    return tree_where(over, packed, state)


def _event(state, trigger, n=1.0):
    """Every acting joker's update rules on one event, n times (Hologram, Glass Joker ...)."""
    ids = acting(state)
    rows, active = rows_for(ids), ids != J_EMPTY
    counts = jnp.full((MAX_JOKERS, rows.UPD_COND.shape[1]), n, jnp.float32)
    value, jstate, _ = update_ex(rows, active, state.joker_value, state.joker_state, jnp.int32(trigger),
                                 counts)
    return state._replace(joker_value=jnp.where(n > 0, value, state.joker_value),
                          joker_state=jnp.where(n > 0, jstate, state.joker_state))


def _effects(state, trigger, effect):
    """(J,) bool — the slots (copies too) whose acting row has this trigger and effect."""
    erows, eok, _, _ = effective(acting(state), state.joker_value, state.joker_state)
    return eok & (erows.TRIGGER == trigger) & (erows.EFFECT == effect)


def _faces_gone(state, gone, mods=None):
    """Canio: the face cards among the instances gone (a debuffed card is not a face card)."""
    mods = mods_of(state) if mods is None else mods
    inst = jnp.arange(MAX_DECK_SIZE)
    face = is_face_card(state.card_rank, mods.pareidolia, state.card_enh) & ~card_debuffed(state, inst, mods)
    return (gone & state.card_live & face).sum()


def destroy_cards(state, gone):
    """Destroy the instances in gone: Canio gains for the face cards, then they leave."""
    faces = _faces_gone(state, gone)
    return remove_cards(_event(state, T_ON_FACE_DESTROYED, faces.astype(jnp.float32)), gone)


# ---------------------------------------------------------------------------
# PRICES — Card:set_cost (vouchers, editions, Astronomer, coupons)
# ---------------------------------------------------------------------------
def joker_cost(state, jid, ed, sticker=0):
    """A joker's cost + its edition, after the discount; a Rental joker $1."""
    base = JOKER_COST[jnp.clip(jid, 0, N_JOKERS - 1)] + ED_COST[jnp.clip(ed, 0, ED_COST.shape[0] - 1)]
    return jnp.where(has_sticker(sticker, STK_RENTAL), RENTAL_COST, cost_of(base, discount(state)))


def _astro(state):
    return (acting(state) == J_ASTRONOMER).any()


def cons_cost(state, cid, ed=ED_NONE):
    """A Tarot / Planet $3, a Spectral $4 (+ a Negative's $5); Astronomer: a Planet $0."""
    c = jnp.clip(cid, 0, N_CONS - 1)
    base = CONS_BASE_COST[c] + ED_COST[jnp.clip(ed, 0, ED_COST.shape[0] - 1)]
    return jnp.where((CONS_KIND[c] == CK_PLANET) & _astro(state), 0, cost_of(base, discount(state)))


def reprice(state, old):
    """set_cost for every held card after the prices changed (a Clearance Sale, Astronomer
    in or out): a sell value = max(1, cost // 2) + what Egg / Gift Card added, which stays."""
    held_j = state.jokers != J_EMPTY
    base_new = sell_of(joker_cost(state, state.jokers, state.joker_ed, state.joker_sticker))
    base_old = sell_of(joker_cost(old._replace(jokers=state.jokers), state.jokers, state.joker_ed,
                                  state.joker_sticker))
    jsell = jnp.where(held_j, state.joker_sell_value - base_old + base_new, 0)
    held_c = state.consumables != CONS_EMPTY
    cnew = sell_of(cons_cost(state, state.consumables, state.cons_ed))
    cold = sell_of(cons_cost(old._replace(jokers=state.jokers if False else old.jokers), state.consumables,
                             state.cons_ed))
    csell = jnp.where(held_c, state.cons_sell - cold + cnew, 0)
    return state._replace(joker_sell_value=jsell.astype(jnp.int32), cons_sell=csell.astype(jnp.int32))


def shop_item_cost(state, slot):
    """$ to buy shop slot `slot`: a joker's cost + its edition (a Rental joker $1); Tarot / Planet
    $3 (Astronomer: Planets free); a playing card $1 + its edition; all after the discount;
    couponed: $0."""
    s = jnp.clip(slot, 0, MAX_SHOP_SLOTS - 1)
    cost = item_cost(state, state.shop_kind[s], state.shop_slots[s], state.shop_ed[s], state.shop_sticker[s])
    return jnp.where(state.shop_free[s], 0, cost).astype(jnp.int32)


def item_cost(state, kind, item, ed, sticker=0):
    """$ of a shop item outside its slot (no coupon) — the GUI's Add Joker price."""
    return jnp.select([kind == IK_JOKER, kind == IK_CARD],
                      [joker_cost(state, item, ed, sticker),
                       cost_of(CARD_BASE_COST + ED_COST[jnp.clip(ed, 0, ED_COST.shape[0] - 1)], discount(state))],
                      cons_cost(state, item)).astype(jnp.int32)


def pack_cost(state, pid, free=False):
    p = jnp.clip(pid, 0, N_PACKS - 1)
    cost = jnp.where((PACK_KIND[p] == PK_CELESTIAL) & _astro(state), 0, cost_of(PACK_COST[p], discount(state)))
    return jnp.where(free, 0, cost).astype(jnp.int32)


def voucher_cost(state):
    return cost_of(VOUCHER_COST, discount(state))


# ---------------------------------------------------------------------------
# CONSUMABLES — spawn, use
# ---------------------------------------------------------------------------
def cons_capacity(state):
    """Consumable slots: 2 (+ Crystal Ball) + the held Negative ones."""
    neg = ((state.consumables != CONS_EMPTY) & (state.cons_ed == ED_NEG)).sum()
    return state.consumable_slots + neg


def cons_room(state):
    return cons_capacity(state) - (state.consumables != CONS_EMPTY).sum()


def _existing(state):
    """(N_JOKERS,), (N_CONS,) bool: the cards that exist now (held, in the shop,
    in an open pack). create_card skips them unless Showman is held."""
    jok = (jnp.arange(N_JOKERS)[:, None] == state.jokers[None, :]).any(axis=1)
    shop_j = (state.shop_kind == IK_JOKER) & (state.shop_slots >= 0)
    jok = jok | (jnp.arange(N_JOKERS)[:, None] == jnp.where(shop_j, state.shop_slots, -1)[None, :]).any(axis=1)
    pk = PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)]
    open_ = state.pack_id >= 0
    pj = jnp.where(open_ & (pk == PK_BUFFOON), state.pack_items, -1)
    jok = jok | (jnp.arange(N_JOKERS)[:, None] == pj[None, :]).any(axis=1)
    con = (jnp.arange(N_CONS)[:, None] == state.consumables[None, :]).any(axis=1)
    shop_c = ((state.shop_kind == IK_TAROT) | (state.shop_kind == IK_PLANET)) & (state.shop_slots >= 0)
    con = con | (jnp.arange(N_CONS)[:, None] == jnp.where(shop_c, state.shop_slots, -1)[None, :]).any(axis=1)
    pc = jnp.where(open_ & ((pk == PK_ARCANA) | (pk == PK_CELESTIAL) | (pk == PK_SPECTRAL)),
                   state.pack_items, -1)
    con = con | (jnp.arange(N_CONS)[:, None] == pc[None, :]).any(axis=1)
    return jok, con


def cons_pool(state, kind, exclude=None):
    """(N_CONS,) bool — the Tarots, Planets or Spectral cards create_card may make now.
    Planets of the secret hands only after the hand was played; The Soul and Black Hole
    never (they come from the pack rolls only)."""
    showman = mods_of(state).showman
    _, con = _existing(state)
    if exclude is not None:
        con = con | exclude
    secret_ok = state.hand_played[jnp.clip(CONS_PARAM, 0, N_HAND_TYPES - 1)] > 0
    planet_ok = _IS_PLANET & ((CONS_PARAM >= SECRET_HANDS) | secret_ok)
    pool = jnp.select([kind == CK_TAROT, kind == CK_PLANET], [_IS_TAROT, planet_ok], _SPECTRAL_POOL)
    return pool & (showman | ~con)


def roll_cons(key, state, kind, exclude=None):
    """A random Tarot / Planet / Spectral id (uniform over the pool; Strength / Pluto /
    Incantation if it is empty)."""
    pool = cons_pool(state, kind, exclude)
    pick = jax.random.categorical(key, jnp.where(pool, 0.0, -jnp.inf)).astype(jnp.int32)
    return jnp.where(pool.any(), pick, CONS_FALLBACK[jnp.clip(kind, 0, 2)])


def soul_ok(state, cid, exclude=None):
    """The Soul / Black Hole may be rolled: none exists (unless Showman)."""
    _, con = _existing(state)
    if exclude is not None:
        con = con | exclude
    return mods_of(state).showman | ~con[cid]


def roll_pack_cons(key, state, kind, exclude=None):
    """create_card(kind, soulable): 0.3 % The Soul for a Tarot / Spectral, 0.3 % Black Hole
    for a Planet / Spectral (a Spectral rolls both; Black Hole wins), else the pool."""
    k1, k2, k3 = jax.random.split(key, 3)
    cid = roll_cons(k1, state, kind, exclude)
    soul = ((kind == CK_TAROT) | (kind == CK_SPECTRAL)) & soul_ok(state, C_SOUL, exclude) \
        & (jax.random.uniform(k2) > 1 - SOUL_ODDS)
    hole = ((kind == CK_PLANET) | (kind == CK_SPECTRAL)) & soul_ok(state, C_BLACK_HOLE, exclude) \
        & (jax.random.uniform(k3) > 1 - SOUL_ODDS)
    return jnp.where(hole, C_BLACK_HOLE, jnp.where(soul, C_SOUL, cid)).astype(jnp.int32)


def give_cons(state, cid, ok=True, ed=ED_NONE, extra=0):
    """Put consumable cid in the first free consumable slot, if there is room (a Negative
    one brings its own slot). extra: slots still counted (a Negative card being used keeps
    its slot until it dissolves)."""
    n = (state.consumables != CONS_EMPTY).sum()
    ok = ok & (cid >= 0) & (n < MAX_CONSUMABLES) & ((n < cons_capacity(state) + extra) | (ed == ED_NEG))
    got = state._replace(consumables=state.consumables.at[n].set(cid),
                         cons_sell=state.cons_sell.at[n].set(sell_of(cons_cost(state, cid, ed))),
                         cons_ed=state.cons_ed.at[n].set(ed))
    return tree_where(ok, got, state)


def spawn_kinds(state, kinds, n, key, extra=0, exclude=None):
    """Create up to n random cards, the i-th of kind kinds[i] (each needs room). n <= 8.
    exclude: (N_CONS,) more cards that count as existing (the card being used)."""
    def body(i, s):
        return give_cons(s, roll_cons(jax.random.fold_in(key, i), s, kinds[i], exclude), i < n, extra=extra)
    return jax.lax.fori_loop(0, 8, body, state)


def spawn_cons(state, kind, n, key, extra=0, exclude=None):
    """Create up to n random cards of kind (each needs room). n is small (<= 8)."""
    return spawn_kinds(state, jnp.full(8, kind, jnp.int32), n, key, extra, exclude)


def take_cons(state, slot):
    """Remove the consumable in slot (the rest shift left)."""
    gone = jnp.arange(MAX_CONSUMABLES) == slot
    c = jnp.where(gone, CONS_EMPTY, state.consumables)
    keep = c != CONS_EMPTY
    return state._replace(consumables=compact(c, keep), cons_sell=compact(jnp.where(keep, state.cons_sell, 0), keep),
                          cons_ed=compact(jnp.where(keep, state.cons_ed, ED_NONE), keep))


def use_sel(state, select, cid):
    """The lanes a consumable acts on: the first CONS_HI valid selected lanes (Cerulean
    Bell's forced card always in), and whether the count fits [CONS_LO, CONS_HI].
    -> (sel (32,), ok)"""
    c = jnp.clip(cid, 0, N_CONS - 1)
    sel = fix_select(state, select, CONS_HI[c])
    n = sel.sum()
    return sel, (n >= CONS_LO[c]) & (n <= CONS_HI[c])


def _editionless(state):
    return (state.jokers != J_EMPTY) & (state.joker_ed == ED_NONE)


def can_use_base(state, cid, from_slot):
    """card.lua can_use_consumeable for consumable cid (from a slot, or from a
    pack), before the card selection: a card Tarot needs a hand with enough cards."""
    c = jnp.clip(cid, 0, N_CONS - 1)
    use = CONS_USE[c]
    hand_ok = (state.game_stage == STAGE_BLIND) | (state.game_stage == STAGE_PACK)
    in_hand = state.hand_valid.sum()
    has_hand = (in_hand >= jnp.maximum(CONS_LO[c], CONS_NEED[c])) & hand_ok
    room = cons_room(state) + from_slot > 0
    held = state.jokers != J_EMPTY
    n_held = held.sum()
    wheel = _editionless(state).any()
    judge = n_held < state.joker_slots
    fool = (state.last_cons >= 0) & (state.last_cons != C_FOOL) & room
    ankh = (n_held >= 1) & (state.joker_slots > 1) & (n_held < state.joker_slots)
    ok = jnp.select([(CONS_LO[c] > 0) | (CONS_NEED[c] > 0), use == U_WHEEL,
                     (use == U_JUDGEMENT) | (use == U_WRAITH) | (use == U_SOUL), use == U_FOOL,
                     (use == U_SPAWN_TAROT) | (use == U_SPAWN_PLANET),
                     (use == U_ECTOPLASM) | (use == U_HEX), use == U_ANKH],
                    [has_hand, wheel, judge, fool, room, wheel, ankh], default=True)
    return (cid >= 0) & (cid < N_CONS) & ok


def can_use(state, cid, select, from_slot):
    """can_use_base, and the selected cards fit the Tarot's [lo, hi] (Aura: a card
    without an edition)."""
    c = jnp.clip(cid, 0, N_CONS - 1)
    sel, n_ok = use_sel(state, select, cid)
    first = jnp.clip(state.hand[jnp.argmax(sel)], 0, MAX_DECK_SIZE - 1)
    aura = (CONS_USE[c] != U_AURA) | (state.card_ed[first] == ED_NONE)
    return can_use_base(state, cid, from_slot) & ((CONS_LO[c] == 0) | n_ok) & aura


def _random_lanes(key, mask, n):
    """(32,) bool — n random lanes of mask (all of them if fewer)."""
    u = jax.random.uniform(key, mask.shape)
    order = jnp.argsort(jnp.where(mask, u, 2.0))
    rank = jnp.zeros(mask.shape, jnp.int32).at[order].set(jnp.arange(mask.shape[0]))
    return mask & (rank < n)


def _random_slot(key, mask):
    """A random True index of mask (the first index when mask is empty)."""
    return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf)).astype(jnp.int32)


def copy_joker(state, j, strip_neg=True):
    """copy_card of the joker in slot j into a new last slot (its value, state, stickers and
    rounds left, sell value with Egg's extra, edition; a Negative one loses it). No room check
    (Ankh, Invisible Joker make room first). A copy of a perished joker is perished too: it
    adds nothing (the game's add_to_deck still gives its hand size / discards; a divergence)."""
    jid, ed, stk = state.jokers[j], state.joker_ed[j], state.joker_sticker[j]
    new_ed = jnp.where(strip_neg & (ed == ED_NEG), ED_NONE, ed)
    extra = state.joker_sell_value[j] - sell_of(joker_cost(state, jid, ed, stk))
    n = (state.jokers != J_EMPTY).sum()
    lives = ~perished(state)[j]
    got = state._replace(
        jokers=state.jokers.at[n].set(jid), joker_value=state.joker_value.at[n].set(state.joker_value[j]),
        joker_state=state.joker_state.at[n].set(state.joker_state[j]),
        joker_sell_value=state.joker_sell_value.at[n].set(sell_of(joker_cost(state, jid, new_ed, stk)) + extra),
        joker_ed=state.joker_ed.at[n].set(new_ed),
        joker_sticker=state.joker_sticker.at[n].set(stk), joker_perish=state.joker_perish.at[n].set(state.joker_perish[j]),
        free_rerolls=state.free_rerolls + ((jid == J_CHAOS) & lives))
    got = apply_budget(got)
    got = got._replace(discards=got.discards + jnp.where(lives, JOKERS.D_SIZE[jnp.clip(jid, 0, N_JOKERS - 1)], 0))
    return tree_where((jid >= 0) & (n < MAX_JOKERS), reprice(got, state), state)


def set_joker_edition(state, j, ed):
    """set_edition on the joker in slot j: the sell value follows the new cost (+ extra; a
    Rental joker's stays $1)."""
    jid, old, stk = state.jokers[j], state.joker_ed[j], state.joker_sticker[j]
    gain = sell_of(joker_cost(state, jid, ed, stk)) - sell_of(joker_cost(state, jid, old, stk))
    return apply_budget(state._replace(joker_ed=state.joker_ed.at[j].set(ed),
                                       joker_sell_value=state.joker_sell_value.at[j].add(gain)))


def apply_cons(state, cid, select, key, extra=0):
    """use_consumeable: the effect of Tarot / Planet / Spectral cid on state (the card has
    already left its slot). Call it only when can_use holds. extra: 1 when the used card is
    Negative (its slot counts until it dissolves: The Emperor, The High Priestess, The Fool).
    The used card still exists while it acts (G.GAME.used_jokers keeps it until Card:remove
    after the dissolve), so The Emperor never makes The Emperor unless Showman is held."""
    c = jnp.clip(cid, 0, N_CONS - 1)
    use, param = CONS_USE[c], CONS_PARAM[c]
    me = (jnp.arange(N_CONS) == cid) & (cid >= 0)        # the used card, for the new cards' pool
    sel, _ = use_sel(state, select, cid)
    inst = jnp.clip(state.hand, 0, MAX_DECK_SIZE - 1)
    on = jnp.zeros(MAX_DECK_SIZE, bool).at[inst].max(sel)            # (128,) the selected instances
    in_hand = jnp.zeros(MAX_DECK_SIZE, bool).at[inst].max(state.hand_valid)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def enhance(s):                                      # set_ability: The Pillar's mark goes
        return s._replace(card_enh=jnp.where(on, param, s.card_enh),
                          card_played_ante=s.card_played_ante & ~on)

    def suit(s):
        return s._replace(card_suit=jnp.where(on, param, s.card_suit))

    def strength(s):
        return s._replace(card_rank=jnp.where(on, (s.card_rank + 1) % 13, s.card_rank))

    def hanged(s):
        glass = (on & (s.card_enh == ENH_GLASS)).sum()
        return _event(destroy_cards(s, on), T_ON_GLASS, glass.astype(jnp.float32))

    def death(s):
        lanes = jnp.arange(MAX_HAND_SIZE)
        right = s.hand[jnp.argmax(jnp.where(sel, lanes, -1))]          # the rightmost card
        r = jnp.clip(right, 0, MAX_DECK_SIZE - 1)
        cp = lambda a: jnp.where(on & (jnp.arange(MAX_DECK_SIZE) != r), a[r], a)
        return s._replace(card_rank=cp(s.card_rank), card_suit=cp(s.card_suit),
                          card_enh=cp(s.card_enh), card_ed=cp(s.card_ed),
                          card_seal=cp(s.card_seal), card_bonus=cp(s.card_bonus),
                          card_played_ante=cp(s.card_played_ante))      # copy_card: the mark too

    def fool(s):
        return give_cons(s, s.last_cons, extra=extra)

    def hermit(s):
        return s._replace(money=s.money + jnp.clip(s.money, 0, param))

    def temperance(s):
        held = s.jokers != J_EMPTY
        return s._replace(money=s.money + jnp.minimum(jnp.where(held, s.joker_sell_value, 0).sum(), param))

    def spawn_t(s):
        return spawn_cons(s, CK_TAROT, jnp.minimum(param, cons_room(s) + extra), k1, extra, me)

    def spawn_p(s):
        return spawn_cons(s, CK_PLANET, jnp.minimum(param, cons_room(s) + extra), k1, extra, me)

    def wheel(s):
        elig = _editionless(s)
        hit = jax.random.uniform(k1) < mods_of(s).odds / param
        j = _random_slot(k2, elig)
        ed = poll_edition(jax.random.uniform(k3), no_neg=True, guaranteed=True)
        return tree_where(hit & elig.any(), set_joker_edition(s, j, ed), s)

    def judgement(s):
        jid, ed = roll_joker(k1, s, k2)
        return acquire(s, jid, ed)[0]

    def planet(s):
        s = s._replace(hand_level=s.hand_level.at[param].add(1),
                       planets_used=s.planets_used.at[param].set(True))
        return _event(s, T_ON_PLANET)

    # --- Spectral cards ---
    def create_cards(s):
        # Familiar / Grim / Incantation: 1 random card of the hand is destroyed, then param new
        # cards (a rank of the mask, any suit, an enhancement but Stone) come into the hand
        lane = _random_slot(k1, s.hand_valid)
        gone = jnp.zeros(MAX_DECK_SIZE, bool).at[jnp.clip(s.hand[lane], 0, MAX_DECK_SIZE - 1)].set(True)
        s = destroy_cards(s, gone)
        ranks = (CONS_RANKS[c] >> jnp.arange(13)) & 1

        def make(i, carry):                              # Incantation makes 4
            s, added = carry
            kr, ks, ke = jax.random.split(jax.random.fold_in(k2, i), 3)
            r = jax.random.categorical(kr, jnp.where(ranks > 0, 0.0, -jnp.inf))
            e = jnp.array([1, 2, 3, 4, 5, 7, 8], jnp.int32)[jax.random.randint(ke, (), 0, 7)]
            s, ok = add_card(s, r, jax.random.randint(ks, (), 0, 4), e, ED_NONE, SEAL_NONE, to_hand=True,
                             ok=i < param)
            return s, added + ok
        s, added = jax.lax.fori_loop(0, 4, make, (s, jnp.int32(0)))
        return _event(s, T_ON_CARD_ADDED, added.astype(jnp.float32))

    def seal(s):
        return s._replace(card_seal=jnp.where(on, param, s.card_seal))

    def aura(s):
        ed = poll_edition(jax.random.uniform(k1), no_neg=True, guaranteed=True)
        return s._replace(card_ed=jnp.where(on, ed, s.card_ed))

    def wraith(s):
        jid, ed = roll_joker(k1, s, k2, rarity=RARE)
        return acquire(s, jid, ed)[0]._replace(money=jnp.int32(0))

    def sigil(s):
        x = jax.random.randint(k1, (), 0, 4)
        return s._replace(card_suit=jnp.where(in_hand, x, s.card_suit))

    def ouija(s):
        x = jax.random.randint(k1, (), 0, 13)
        return apply_budget(s._replace(card_rank=jnp.where(in_hand, x, s.card_rank),
                                       base_hand_size=s.base_hand_size - 1))

    def ectoplasm(s):
        j = _random_slot(k1, _editionless(s))
        s = set_joker_edition(s, j, ED_NEG)
        return apply_budget(s._replace(base_hand_size=s.base_hand_size - s.ecto_minus,
                                       ecto_minus=s.ecto_minus + 1))

    def immolate(s):
        lanes = _random_lanes(k1, s.hand_valid, param)
        gone = jnp.zeros(MAX_DECK_SIZE, bool).at[jnp.clip(s.hand, 0, MAX_DECK_SIZE - 1)].max(lanes)
        return destroy_cards(s, gone)._replace(money=s.money + 20)

    def ankh(s):
        # a random held joker (an Eternal one too) is copied; every other joker but the Eternal
        # ones is destroyed
        held = s.jokers != J_EMPTY
        j = _random_slot(k1, held)
        slots = jnp.arange(MAX_JOKERS)
        gone = held & (slots != j) & ~eternal(s)
        at = (held & ~gone & (slots < j)).sum()                          # the chosen one's slot after
        return copy_joker(remove_jokers(s, gone), at)

    def hex_(s):
        j = _random_slot(k1, _editionless(s))
        s = set_joker_edition(s, j, ED_POLY)
        return remove_jokers(s, (s.jokers != J_EMPTY) & (jnp.arange(MAX_JOKERS) != j) & ~eternal(s))

    def cryptid(s):
        src = jnp.clip(s.hand[jnp.argmax(sel)], 0, MAX_DECK_SIZE - 1)
        added = jnp.int32(0)
        for i in range(2):                               # static: 2 copies
            s, ok = add_card(s, s.card_rank[src], s.card_suit[src], s.card_enh[src], s.card_ed[src],
                             s.card_seal[src], s.card_bonus[src], to_hand=True, ok=i < param,
                             played=s.card_played_ante[src])
            added = added + ok
        return _event(s, T_ON_CARD_ADDED, added.astype(jnp.float32))

    def soul(s):
        jid, ed = roll_joker(k1, s, k2, rarity=LEGENDARY)
        return acquire(s, jid, ed)[0]

    def black_hole(s):
        return s._replace(hand_level=s.hand_level + 1)

    branches = [enhance, suit, strength, hanged, death, fool, hermit, temperance, spawn_t,
                spawn_p, wheel, judgement, planet,
                create_cards, seal, aura, wraith, sigil, ouija, ectoplasm, immolate, ankh, hex_,
                cryptid, soul, black_hole]
    state = jax.lax.switch(jnp.clip(use, 0, len(branches) - 1), branches, state)
    is_tarot = CONS_KIND[c] == CK_TAROT
    # set_consumeable_usage: the Tarot / Planet used is the new last one, The Fool too (after
    # it made its copy), so a second Fool cannot be used until another Tarot / Planet is. A
    # Spectral card is not a "last one".
    return state._replace(tarots_used=state.tarots_used + is_tarot,
                          last_cons=jnp.where(CONS_KIND[c] == CK_SPECTRAL, state.last_cons, cid))


def buy_and_use(state, t):
    """A BUY of shop slot t is the game's Buy and Use (buy_from_shop, id 'buy_and_use'): the slot
    holds a Tarot / Planet / Spectral (Ghost Deck) and the consumable slots are full. With a free
    slot BUY then USE gives the same result, so only a BUY that cannot take the card uses it."""
    kind = state.shop_kind[jnp.clip(t, 0, MAX_SHOP_SLOTS - 1)]
    return ((kind == IK_TAROT) | (kind == IK_PLANET) | (kind == IK_SPECTRAL)) & (cons_room(state) <= 0)


_apply_cons_c = compacted(apply_cons)


def _use_parts(state: balatroEnvState, action: balatroAction):
    """The common part of _use_pick: the checks, the state before use_consumeable (base) and the
    apply_cons arguments. need: apply_cons's result is kept (an ok USE, Buy and Use, or PICK of
    a consumable). env_step calls it too, for the apply_cons call it makes (see _Pre)."""
    is_use = action.verb == V_USE
    is_buy = action.verb == V_BUY
    t = action.target
    key, sub = jax.random.split(jax.random.fold_in(state.key, K_USE))
    # USE: consumable slot t
    slot = jnp.clip(t, 0, MAX_CONSUMABLES - 1)
    u_cid = state.consumables[slot]
    u_ok = (t >= 0) & (t < MAX_CONSUMABLES) & can_use(state, u_cid, action.select, True)
    # BUY: shop slot t. can_buy_and_use asks can_use_consumeable with the card in the shop (no
    # hand, no slot of its own); the card is paid first (ease_dollars comes before use_card, so
    # The Hermit doubles the money after the cost)
    b = jnp.clip(t, 0, MAX_SHOP_SLOTS - 1)
    b_cid, b_cost = state.shop_slots[b], shop_item_cost(state, b)
    b_ok = (t >= 0) & (t < state.shop_size) & (b_cid >= 0) & affordable(state, b_cost) \
        & can_use_base(state, b_cid, False)
    b_state = state._replace(money=state.money - b_cost, shop_slots=state.shop_slots.at[b].set(-1),
                             shop_free=state.shop_free.at[b].set(False),
                             shop_sticker=state.shop_sticker.at[b].set(0))
    # PICK: pack card t
    i = jnp.clip(t, 0, MAX_PACK - 1)
    p_ok = pick_ok(state, t, action.select)
    item = state.pack_items[i]
    kind = PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)]
    p_state = state._replace(pack_items=state.pack_items.at[i].set(-1), pack_left=state.pack_left - 1,
                             pack_sticker=state.pack_sticker.at[i].set(0))
    cons = (kind == PK_ARCANA) | (kind == PK_CELESTIAL) | (kind == PK_SPECTRAL)

    base = tree_where(is_use, take_cons(state, slot), tree_where(is_buy, b_state, p_state))._replace(key=key)
    neg = is_use & (state.cons_ed[slot] == ED_NEG)
    cid = jnp.where(is_use, u_cid, jnp.where(is_buy, b_cid, item))
    ok = jnp.where(is_use, u_ok, jnp.where(is_buy, b_ok, p_ok))
    need = ok & (is_use | is_buy | ((action.verb == V_PICK) & cons))
    return dict(is_use=is_use, is_buy=is_buy, key=key, i=i, kind=kind, p_state=p_state, cons=cons, ok=ok,
                need=need, args=(base, cid, action.select, sub, neg.astype(jnp.int32)))


def _use_pick(state: balatroEnvState, action: balatroAction, pre=None):
    """USE a held consumable, PICK a pack card, or BUY a Tarot / Planet the full consumable slots
    cannot take (Buy and Use): they share one use_consumeable. A pack's joker / playing card
    goes to the jokers / the deck instead. A Chicot made in a boss round (The Soul) disables the
    boss at once (add_to_deck). pre: env_step's _Pre (its apply_cons result); None = call it here."""
    p = _use_parts(state, action)
    is_use, is_buy, key, i, kind, p_state, cons, ok = (p[n] for n in
                                                      ("is_use", "is_buy", "key", "i", "kind", "p_state", "cons", "ok"))
    item = state.pack_items[i]
    used = apply_cons(*p["args"]) if pre is None else pre.used
    joker = acquire(p_state._replace(key=key), item, state.pack_ed[i], state.pack_sticker[i])[0]
    card, added = add_card(p_state._replace(key=key), state.pack_rank[i], state.pack_suit[i],
                           state.pack_enh[i], state.pack_ed[i], state.pack_seal[i])
    card = _event(card, T_ON_CARD_ADDED, added.astype(jnp.float32))
    picked = tree_where(cons, used, tree_where(kind == PK_BUFFOON, joker, card))
    picked = tree_where(picked.pack_left <= 0, _close_pack(picked), picked)
    got = tree_where(is_use | is_buy, used, picked)
    chicot = (acting(got) == J_CHICOT).any() & ~(acting(state) == J_CHICOT).any()
    got = tree_where(chicot, disable_boss(got), got)
    return tree_where(ok, got, state), jnp.int32(0), jnp.bool_(False)


def _sell_cons(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_CONSUMABLES - 1)
    ok = (t >= 0) & (t < MAX_CONSUMABLES) & (state.consumables[slot] != CONS_EMPTY)
    # every card sold counts for Campfire (G.FUNCS.sell_card), a Tarot / Planet too
    rows, active, _ = _jokers(state)
    value, _ = update(rows, active, state.joker_value, state.joker_state, jnp.int32(T_ON_SELL_OTHER))
    sold = take_cons(state, slot)._replace(money=state.money + state.cons_sell[slot], joker_value=value)
    return tree_where(ok, sold, state), jnp.int32(0), jnp.bool_(False)


# ---------------------------------------------------------------------------
# THE BLIND — PLAY / DISCARD, the draw, the end of the round
# ---------------------------------------------------------------------------
def _discard_cards(state, cards, valid, hook, key):
    """The jokers and seals on a discard of cards (lane order), as
    discard_cards_from_highlighted: per card a Purple Seal (not debuffed: a Tarot) and the
    joker rules (Castle, Mail-In Rebate, Hit the Road skip a debuffed card; Yorick and Ramen
    count it); Burnt Joker (not on The Hook's discard); Trading Card. The cards stay in
    the hand: the caller takes them out."""
    mods = mods_of(state)
    ids = acting(state)
    pr, ps, pe, _, pseal, _ = _gather(state, cards)
    deb = card_debuffed(state, cards, mods) & valid
    first_discard = state.discards_used == 0
    dsc = _discard(pr, ps, valid, ids, state.joker_value, state.joker_state, targets(state), mods, pe, deb)
    burnt = _effects(state, T_ON_DISCARD, E_LEVEL_UP).sum() * first_discard * jnp.logical_not(hook)
    dhand = detect(pr, ps, valid, mods, jnp.where(deb & (pe == ENH_WILD), ENH_NONE, pe)).handtype
    purple = (valid & (pseal == SEAL_PURPLE) & ~deb).sum()
    # Trading Card (not its copies): each one pays $3; the card is destroyed once
    n_trading = (ids == J_TRADING).sum()
    trading = (n_trading > 0) & first_discard & (valid.sum() == 1)
    idx = jnp.where(valid, cards, MAX_DECK_SIZE)
    gone = jnp.zeros(MAX_DECK_SIZE, bool).at[idx].set(valid & trading, mode="drop")
    changed = (dsc.joker_value != state.joker_value) | (dsc.joker_state != state.joker_state)
    s = state._replace(joker_value=dsc.joker_value, joker_state=dsc.joker_state,
                       joker_seen=state.joker_seen | (state.joker_down & changed),
                       money=state.money + jnp.floor(dsc.money + 3 * n_trading * trading).astype(jnp.int32),
                       hand_level=state.hand_level.at[dhand].add(burnt))
    s = spawn_cons(s, CK_TAROT, purple, key)
    glass = (gone & state.card_live & (state.card_enh == ENH_GLASS)).sum()   # it shatters: Glass Joker
    s = _event(destroy_cards(s, gone), T_ON_GLASS, glass.astype(jnp.float32))
    return remove_jokers(s, dsc.expired)


def _discard_lanes(state, lanes, hook, always=False):
    """The cards in lanes are discarded: their effects (_discard_cards), then they leave the
    hand. hook: The Hook's discard (G.FUNCS.discard_cards_from_highlighted(nil, true): no
    discard is used, no Burnt Joker). No lanes: nothing happens unless always (an empty
    DISCARD still fires Green Joker). state.key is not moved."""
    cards, valid = compact(state.hand, lanes)[:5], compact(lanes, lanes)[:5]
    s = _discard_cards(state, cards, valid, hook, jax.random.fold_in(state.key, K_HOOK + 100))
    s = s._replace(hand_valid=s.hand_valid & ~lanes, hand_down=s.hand_down & ~lanes)
    return tree_where(lanes.any() | always, s, state)


def _hook_discard(state, lanes):
    """The Hook: the lanes are discarded before the hand scores."""
    return _discard_lanes(state, lanes, True)


def _bell(state, key):
    """Cerulean Bell (drawn_to_hand): no forced card in hand -> a random card of the hand."""
    on = boss_is(state, BK_BELL)
    held = state.hand_valid & (state.hand == state.forced_card) & (state.forced_card >= 0)
    lane = _random_slot(key, state.hand_valid)
    pick = on & ~held.any() & state.hand_valid.any()
    forced = jnp.where(pick, state.hand[lane], jnp.where(on, state.forced_card, -1))
    return state._replace(forced_card=forced.astype(jnp.int32))


def _joker_h_size(state):
    """(J,) the hand size each held joker adds (Juggler, Turtle Bean, Merry Andy -1, Stuntman -2 ...)."""
    rows = rows_for(state.jokers)
    held = state.jokers != J_EMPTY
    return jnp.where(held, rows.H_SIZE + jnp.where(rows.HSIZE_V, state.joker_value, 0.0).astype(jnp.int32), 0)


def _grow_draw(state, old_size, key, n=None):
    """CardArea:change_size in the round: a larger hand size draws the new cards at once
    (n of them; default the growth), if the hand holds a card and the size is above 1."""
    n = jnp.maximum(state.hand_size - old_size, 0) if n is None else n
    go = (state.game_stage == STAGE_BLIND) & (n > 0) & state.hand_valid.any() & (state.hand_size > 1)
    return tree_where(go, draw(state, key, n=n, settle=False), state)


def _heart(state, key, prepped):
    """Crimson Heart (drawn_to_hand after the first draw and after each PLAY): a random
    joker is disabled, never the one just disabled (unless it is the only joker). A
    disabled Drunkard / Merry Andy takes its discards along (remove_from_deck). The joker
    let go adds its hand size back and the new one takes its own (change_size, one event
    each): a growth draws at once (a Juggler let go: 1 card; Stuntman disabled: 2). set_debuff runs
    add_to_deck / remove_from_deck: an Astronomer let go or disabled sets the prices again
    (reprice: a held Negative Planet sells for $1 / $4)."""
    on = boss_is(state, BK_HEART) & prepped
    k_pick, k_draw = jax.random.split(key)
    held = state.jokers != J_EMPTY
    dead = perished(state)                    # debuffed already: never a pick (unless the only joker,
    cand = jnp.where(held.sum() < 2, held, held & ~state.joker_debuff & ~dead)   # then nothing changes)
    j = _random_slot(k_pick, cand)
    new = held & (jnp.arange(MAX_JOKERS) == j) & cand.any() & ~dead
    d = JOKERS.D_SIZE[jnp.clip(state.jokers, 0, N_JOKERS - 1)]
    back = jnp.where(state.joker_debuff, d, 0).sum() - jnp.where(new, d, 0).sum()
    h = _joker_h_size(state)
    grow = jnp.maximum(jnp.where(state.joker_debuff, h, 0).sum(), 0) + jnp.maximum(-jnp.where(new, h, 0).sum(), 0)
    s = reprice(apply_budget(state._replace(joker_debuff=new, discards=jnp.maximum(state.discards + back, 0))), state)
    s = _grow_draw(s, s.hand_size, k_draw, n=grow)
    return tree_where(on & held.any(), s, state)


def draw(state, key, after_play=False, first=False, n=None, settle=True):
    """Draw cards into the empty lanes: up to hand_size (n: exactly that many), The
    Serpent 3 after a PLAY / DISCARD (even past hand_size). Never past the live deck.
    Face down: The House (the first hand), The Wheel (1 in 7), The Mark (face cards),
    The Fish (after a PLAY). Then Cerulean Bell / Crimson Heart (drawn_to_hand; settle
    False: the caller does it)."""
    state = fit_hand(state)
    k_down, k_bell, k_heart = jax.random.split(key, 3)
    left = state.deck_size - state.deck_ptr
    serpent = boss_is(state, BK_SERPENT) & ~first
    want = jnp.where(serpent, SERPENT_DRAW, state.hand_size - state.hand_valid.sum())
    want = want if n is None else n
    count = jnp.clip(jnp.minimum(want, left), 0)
    empty = ~state.hand_valid
    src = state.deck_ptr + jnp.cumsum(empty) - 1
    fill = first_n(empty, count)
    hand = jnp.where(fill, state.deck[jnp.clip(src, 0, MAX_DECK_SIZE - 1)], state.hand)
    mods = mods_of(state)
    r, _, e, _, _, _ = _gather(state, hand)
    wheel = jax.random.uniform(k_down, (MAX_HAND_SIZE,)) < mods.odds / WHEEL_ODDS
    down = fill & ((boss_is(state, BK_HOUSE) & first) | (boss_is(state, BK_WHEEL) & wheel)
                   | (boss_is(state, BK_MARK) & is_face_card(r, mods.pareidolia, e))
                   | (boss_is(state, BK_FISH) & after_play))
    state = state._replace(hand=hand, hand_valid=state.hand_valid | fill, deck_ptr=state.deck_ptr + count,
                           hand_down=(state.hand_down & state.hand_valid) | down)
    if not settle:
        return state
    return drawn_to_hand(state, k_bell, first | after_play)


def drawn_to_hand(state, key, prepped):
    """Blind:drawn_to_hand: Cerulean Bell, then Crimson Heart (prepped: the first draw and
    the draws after a PLAY)."""
    k_bell, k_heart = jax.random.split(key)
    return _heart(_bell(state, k_bell), k_heart, prepped)


@jit
def deal(state: balatroEnvState):
    """Fill the hand up to hand_size from the deck (no boss rules)."""
    state = fit_hand(state)
    want = state.hand_size - state.hand_valid.sum()
    deal_count = jnp.clip(jnp.minimum(want, state.deck_size - state.deck_ptr), 0)
    empty = ~state.hand_valid
    src = state.deck_ptr + jnp.cumsum(empty) - 1
    fill = first_n(empty, deal_count)
    new_hand = jnp.where(fill, state.deck[jnp.clip(src, 0, MAX_DECK_SIZE - 1)], state.hand)
    return state._replace(hand=new_hand, hand_valid=state.hand_valid | fill,
                          deck_ptr=state.deck_ptr + deal_count)


def disable_boss(state, draw_one=True):
    """blind.lua disable: The Water / The Needle give back what they took, The Wall /
    Violet Vessel go back to x2, The Manacle gives the hand size back (and draws 1 card),
    Cerulean Bell lets go, every card and joker faces up, no debuffs. The Needle's take can be
    negative (0 base hands): then the hands go down, to 0 at least (the game goes below 0, but
    the round still ends only after the next hand, so 0 gives the same play)."""
    on = boss_active(state)
    b = state.boss
    s = state._replace(
        discards=state.discards + jnp.where(b == BK_WATER, state.boss_sub, 0),
        hands=jnp.maximum(state.hands + jnp.where(b == BK_NEEDLE, state.boss_sub, 0), 0),
        blind_target=blind_target(state.ante, BLIND_BOSS, run_scale(state), b, True, state.stake),
        forced_card=jnp.int32(-1), hand_down=jnp.zeros(MAX_HAND_SIZE, bool),
        joker_down=jnp.zeros(MAX_JOKERS, bool), joker_seen=jnp.zeros(MAX_JOKERS, bool),
    )
    d = JOKERS.D_SIZE[jnp.clip(state.jokers, 0, N_JOKERS - 1)]
    s = s._replace(discards=s.discards + jnp.where(state.joker_debuff, d, 0).sum(),
                   joker_debuff=jnp.zeros(MAX_JOKERS, bool), boss_disabled=jnp.bool_(True))
    s = reprice(apply_budget(s), state)                   # Crimson Heart's joker back (Astronomer)
    if draw_one:
        s = tree_where(b == BK_MANACLE, draw(s, jax.random.fold_in(state.key, K_DRAW), n=1), s)
    return tree_where(on, s, state)


def _blind_action(state: balatroEnvState, action: balatroAction):
    """PLAY or DISCARD (the verb is already repaired)."""
    is_play = action.verb == V_PLAY
    sel = fix_select(state, action.select, 5)
    k_cons, k_draw = (jax.random.fold_in(state.key, K_ROUND + 100 + i) for i in range(2))
    key, _ = jax.random.split(state.key)

    # one discard pass: a DISCARD's cards (the jokers, Burnt Joker, Purple Seals, Trading Card),
    # or before a PLAY The Hook's 2 random held cards; then a PLAY scores (The Tooth, The Ox,
    # The Arm and the debuffs are in score_play)
    hook = _hooked(state, action.select) & is_play
    pre = _discard_lanes(state, jnp.where(is_play, hook, sel), is_play, always=~is_play)
    res, out, info = score_play(pre, action.select)
    pr, ps, pe, _, pseal, _ = _gather(pre, res.cards)
    idx = jnp.where(res.valid, res.cards, MAX_DECK_SIZE)              # out of range = dropped
    on = lambda m: jnp.zeros(MAX_DECK_SIZE, bool).at[idx].set(m, mode="drop")
    first_hand = state.hand_round.sum() == 0

    # PLAY: the played cards keep Midas / Vampire / Hiker changes; Glass cards break; Sixth Sense.
    # Sixth Sense and the Glass check read the cards after the BEFORE step (a Stone 6 a Vampire
    # stripped is a 6). The played cards get The Pillar mark (play_cards_from_highlighted), and
    # a card Midas Mask / Vampire touched loses it again (set_ability). In The Pillar's own blind
    # a new mark debuffs nothing in the game: the debuff flags were set at the blind's start
    # (Blind:set_blind) and the marks go when the boss is beaten. So the env gives no new mark
    # there: a face card that breaks counts for Canio, and a DNA copy is not debuffed.
    pillar = boss_is(pre, BK_PILLAR)
    sixth = ((acting(pre) == J_SIXTH).any() & first_hand & (res.valid.sum() == 1)
             & (pr[0] == 4) & (out.enh[0] != ENH_STONE) & ~info["debuffed"])
    gone_cards = on(res.valid & (out.broken | sixth))
    glass_gone = (res.valid & (out.broken | sixth) & (out.enh == ENH_GLASS)).sum()
    marks = (pre.card_played_ante | on(res.valid & ~pillar)) & ~on(res.valid & out.reset)
    played = pre._replace(
        hands=jnp.maximum(pre.hands - 1, 0),            # a PLAY at 0 hands (The Needle's disable)
        hand_valid=pre.hand_valid & ~res.sel, hand_down=pre.hand_down & ~res.sel,
        card_down=pre.card_down & ~on(res.valid),        # played: turned face up (Card:flip)
        joker_seen=pre.joker_seen | (pre.joker_down & out.acted),
        chips_scored=jnp.minimum(pre.chips_scored, INT32_MAX - res.score) + res.score,   # no int32 wrap
        joker_value=out.joker_value, joker_state=out.joker_state,
        money=info["money"] + jnp.floor(out.money).astype(jnp.int32),
        hand_played=out.hand_played, hand_round=out.hand_round,
        hand_level=info["level"].at[out.handtype].add(out.level_up),
        last_hand=out.handtype,
        card_enh=pre.card_enh.at[idx].set(out.enh, mode="drop"),
        card_bonus=pre.card_bonus.at[idx].add(out.perma, mode="drop"),
        card_played_ante=marks,
        boss_hand=jnp.where(boss_is(pre, BK_MOUTH) & (pre.boss_hand < 0), out.handtype, pre.boss_hand),
        forced_card=jnp.int32(-1),
        key=key)
    played = destroy_cards(played, gone_cards)
    played = _event(played, T_ON_GLASS, glass_gone.astype(jnp.float32))
    # new consumables in the game's order: 8 Ball (the cards), the jokers' main effects in slot
    # order (Superposition, Vagabond: a Tarot; Seance: a Spectral card), Sixth Sense
    kinds = jnp.full(8, CK_TAROT, jnp.int32)
    main = compact(out.spawn_kind, out.spawn_kind >= 0)
    n_main = (out.spawn_kind >= 0).sum()
    pos = jnp.arange(8)
    kinds = jnp.where(pos < out.tarots, CK_TAROT,
                      jnp.where(pos < out.tarots + n_main, main[jnp.clip(pos - out.tarots, 0, MAX_JOKERS - 1)],
                                CK_SPECTRAL))
    played = spawn_kinds(played, kinds, out.tarots + n_main + sixth, k_cons)
    # DNA (copies too): the 1st hand of the round is one card -> each DNA effect puts a copy at
    # the end of the hand and into the deck. score_play already counted them as held cards and
    # gave Hologram its X0.25 each; copy k is the card as the Midas Masks / Vampires to the left
    # of the k-th DNA effect left it, with its The Pillar mark (copy_card; in The Pillar's blind
    # only a mark from before the blind, see above).
    c0 = jnp.clip(res.cards[0], 0, MAX_DECK_SIZE - 1)
    c0_mark = pre.card_played_ante[c0] | ~pillar

    def dna_copy(k, s):
        return add_card(s, pre.card_rank[c0], pre.card_suit[c0], out.dna_enh[k], pre.card_ed[c0],
                        pre.card_seal[c0], pre.card_bonus[c0], to_hand=True,
                        played=out.dna_mark[k] & c0_mark)[0]
    played = jax.lax.fori_loop(0, out.dna_n, dna_copy, played)
    played = remove_jokers(played, out.expired)

    discarded = pre._replace(discards=pre.discards - 1, discards_used=pre.discards_used + 1,
                             forced_card=jnp.int32(-1), key=key)
    new_state = tree_where(is_play, played, discarded)

    # the blind is beaten or the hands ran out: no draw (_settle ends the round); else draw
    cleared = new_state.chips_scored >= new_state.blind_target
    out_of_hands = is_play & (new_state.hands <= 0)        # a DISCARD at 0 hands still draws
    drawn = draw(new_state, k_draw, after_play=is_play)
    return tree_where(cleared | out_of_hands, new_state, drawn), jnp.int32(0), jnp.bool_(False)


def end_round(state: balatroEnvState) -> balatroEnvState:
    """end_round: beaten -> the cash out; else Mr. Bones saves it (at least 25 % of the target:
    the first acting Mr. Bones is destroyed, the blind pays $0) or the run is lost. Each Rental
    joker takes $3 here (calculate_rental: every held joker, a debuffed one and the Mr. Bones
    that saves the run too), before the cash out's interest."""
    beaten = state.chips_scored >= state.blind_target
    bones = first_n(acting(state) == J_MR_BONES, 1)       # the first saves it: then game_over is false
    saved = ~beaten & bones.any() & (state.chips_scored >= BONES_SHARE * state.blind_target)
    rent = RENTAL_RATE * ((state.jokers != J_EMPTY) & has_sticker(state.joker_sticker, STK_RENTAL)).sum()
    s = remove_jokers(state._replace(money=state.money - rent), jnp.where(saved, bones, False))
    lost = state._replace(game_stage=jnp.int32(STAGE_LOST))
    return tree_where(beaten | saved, _clear_blind(s, saved), lost)


def _held_end_of_round(state: balatroEnvState, key):
    """The cards still in hand at the end of the round: Gold cards pay $3, Blue
    Seals make the Planet of the last hand played; each repeats for a Red Seal
    and for Mime (and its copies) when the card did something. A debuffed card does nothing."""
    r, s, e, _, seal, _ = _gather(state, state.hand)
    ok = state.hand_valid & ~card_debuffed(state, state.hand)
    gold = ok & (e == ENH_GOLD)
    blue = ok & (seal == SEAL_BLUE)
    erows, eok, _, _ = effective(acting(state), state.joker_value, state.joker_state)
    mime = (eok & (erows.TRIGGER == T_PER_HELD) & (erows.EFFECT == E_RETRIGGER)).sum()
    reps = 1 + (gold | blue) * ((seal == SEAL_RED) + mime)
    money = (gold * reps).sum() * GOLD_DOLLARS
    n_blue = (blue * reps).sum()
    planet = PLANET_OF_HAND[jnp.clip(state.last_hand, 0, N_HAND_TYPES - 1)]
    ok = state.last_hand >= 0
    state = jax.lax.fori_loop(0, 8, lambda i, s: give_cons(s, planet, (i < n_blue) & ok), state)
    return state._replace(money=state.money + money)        # each Blue Seal needs room


def _clear_blind(state: balatroEnvState, saved=False) -> balatroEnvState:
    """End of round: the held cards, the END_OF_ROUND jokers, the cash-out rows
    (evaluate_round: last_cashout), then the cash-out screen (STAGE_CASHOUT: the $ is paid at
    NEXT, _cash_out) or the win (paid at once: the run ends). saved: Mr. Bones (the blind pays
    $0). The boss is still on here (a debuffed Gold card pays nothing, Crimson Heart's joker
    misses it all); it is gone after."""
    k_eor, k_todo, k_tg, k_shop, k_held, k_ante, key = jax.random.split(jax.random.fold_in(state.key, K_ROUND), 7)
    ids = acting(state)
    rows, active = rows_for(ids), ids != J_EMPTY
    # Gift Card (not its copies): $1 of sell value on every joker and every consumable. The jokers'
    # end-of-round effects come before the held cards (end_round), so a Blue Seal's Planet misses it.
    gift = jnp.where(active & (rows.TRIGGER == T_END_OF_ROUND) & (rows.EFFECT == E_SELL_VALUE_ALL),
                     rows.VALUE, 0).sum().astype(jnp.int32)
    state = state._replace(cons_sell=jnp.where(state.consumables != CONS_EMPTY, state.cons_sell + gift, 0))
    state = _held_end_of_round(state, k_held)
    mods = mods_of(state)
    is_boss = state.blind_index == BLIND_BOSS

    # counters (Popcorn, Turtle Bean, Hit the Road, Campfire, Rocket, Invisible Joker)
    none = Cards(*[jnp.zeros(1, jnp.int32)] * 2, jnp.zeros(1, bool), jnp.full(1, -1),
                 jnp.zeros(1, bool), jnp.zeros(1, bool))
    counts = rule_counts(rows, none, jnp.zeros(1), no_info(is_boss), targets(state), mods,
                         state.joker_state, False)
    value, jstate, fired = update_ex(rows, active, state.joker_value, state.joker_state,
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
    sell = jnp.where(state.jokers != J_EMPTY, sell + gift, 0)
    # To Do List: a new hand (a secret one once played), never the same one
    todo = todo_pick(k_todo, state.hand_played, jstate)
    jstate = jnp.where(active & (rows.STATE == ST_TODO), todo, jstate)
    # Perishable (calculate_perishable, after the joker's own end-of-round effect): one round
    # less; at 0 it is debuffed for good, so it pays no cash-out $ (calculate_dollar_bonus) and
    # To the Moon's interest, its hand size / discards ... and Astronomer's prices go
    pre = state
    tick = (state.jokers != J_EMPTY) & has_sticker(state.joker_sticker, STK_PERISH) & (state.joker_perish > 0)
    state = state._replace(joker_perish=jnp.where(tick, state.joker_perish - 1, state.joker_perish))

    money_eor = joker_money(state, value)                 # the acting jokers' cashout $ (before they go)
    state = remove_jokers(state._replace(joker_value=value, joker_sell_value=sell,
                                         joker_state=jstate, gros_extinct=extinct), gone)
    state = reprice(state, pre)

    # interest is paid on the money held BEFORE this cashout
    rate = 1 + (acting(state) == J_MOON).sum()
    # Green Deck: $2 per hand left, $1 per discard left, no interest (modifiers.no_interest)
    green = state.back == B_GREEN
    interest = jnp.where((state.money >= INTEREST_PER) & ~green,
                         rate * jnp.minimum(state.money // INTEREST_PER, interest_steps(state)), 0)
    invest = (state.tags == TG_INVESTMENT) & is_boss                    # Investment Tag: $25 at a boss
    hand_pay = state.hands * jnp.where(green, GREEN_PER_HAND, 1)
    discard_pay = jnp.where(green, jnp.maximum(state.discards, 0) * GREEN_PER_DISCARD, 0)
    cashout = (jnp.where(saved, 0, blind_dollars(state.blind_index, state.boss, state.stake)) + hand_pay
               + discard_pay + interest + money_eor + INVESTMENT_DOLLARS * invest.sum())
    won = is_boss & (state.ante >= WIN_ANTE)

    # the boss is beaten: The Ox's hand (the most played; a tie: the last in pairs() order,
    # items.OX_PAIRS_POS), a new ante, The Pillar forgets, this ante's voucher; at the cash out
    # the next boss and tags
    most = jnp.argmax(state.hand_played * N_HAND_TYPES + OX_PAIRS_POS).astype(jnp.int32)
    ante = state.ante + (is_boss & ~won)                   # the win keeps ante 8
    nxt = jnp.where(is_boss, BLIND_SMALL, state.blind_index + 1)
    tg = new_targets(k_tg, state)
    kv, kb, kt = jax.random.split(k_ante, 3)
    pre_clear = state
    state = state._replace(
        tags=compact(jnp.where(invest, -1, state.tags), ~invest & (state.tags >= 0)),
        tag_hand=compact(state.tag_hand, ~invest & (state.tags >= 0)),
        money=state.money + jnp.where(won, cashout, 0),     # else paid at Cash Out
        last_cashout=cashout,
        unused_discards=state.unused_discards + jnp.maximum(state.discards, 0),
        ox_hand=jnp.where(is_boss, most, state.ox_hand),
        ante=ante, blind_index=nxt,
        card_played_ante=jnp.where(is_boss, False, state.card_played_ante),
        game_stage=jnp.where(won, STAGE_WON, STAGE_CASHOUT),
        reroll_cost=reroll_base(state), reroll_inc=jnp.int32(0), reroll_d6=jnp.bool_(False),
        idol_card=tg.idol_card, mail_rank=tg.mail_rank,
        ancient_suit=tg.ancient_suit, castle_suit=tg.castle_suit,
        hand_valid=jnp.zeros(MAX_HAND_SIZE, bool), hand=jnp.full(MAX_HAND_SIZE, -1, jnp.int32),
        hand_down=jnp.zeros(MAX_HAND_SIZE, bool), forced_card=jnp.int32(-1),
        card_down=jnp.zeros(MAX_DECK_SIZE, bool),
        joker_debuff=jnp.zeros(MAX_JOKERS, bool), joker_down=jnp.zeros(MAX_JOKERS, bool),
        joker_seen=jnp.zeros(MAX_JOKERS, bool),
        round_hand_bonus=jnp.int32(0), boss_disabled=jnp.bool_(False),
        deck_ptr=jnp.int32(0),
        key=key,
    )
    state = reprice(apply_budget(state), pre_clear)      # defeat: Crimson Heart's joker back (Astronomer)
    # Anaglyph Deck: a Double Tag after a Boss Blind (trigger_effect 'eval'; a held Double Tag
    # does not copy it)
    state = tree_where(is_boss & (state.back == B_ANAGLYPH), add_tag(state, TG_DOUBLE, -1), state)
    new_ante = new_ante_rolls(kb, state)
    # the win ends the run: no new ante, no shop; else the shop opens at Cash Out (k_shop is
    # not used: the cash out has its own keys)
    return tree_where(is_boss & ~won,
                      new_ante._replace(ante_voucher=roll_voucher(kv, state, jnp.zeros(N_VOUCHERS, bool))), state)


def _cash_out(state: balatroEnvState) -> balatroEnvState:
    """NEXT at the cash-out screen (G.FUNCS.cash_out): the rows' $ (last_cashout) is paid, the
    deck is shuffled ('cashout'), then the shop opens (update_shop). A USE or SELL on the
    screen came before it, so it changes the first shop's pool."""
    k_deck, k_shop, key = jax.random.split(jax.random.fold_in(state.key, K_CASHOUT), 3)
    deck, _, _, n = _fresh_deal(k_deck, state.card_live, 0)
    s = state._replace(money=state.money + state.last_cashout, deck=deck, deck_size=n, deck_ptr=jnp.int32(0),
                       game_stage=jnp.int32(STAGE_SHOP), key=key)
    return open_shop(k_shop, s)


def end_window(state: balatroEnvState) -> jax.Array:
    """After the hand that ends a round: can the joker order still change the end of the round?
    In the game a joker can be dragged while the hand scores (the joker area has no lock), and
    end_round reads the order after the score animation. Blueprint and Brainstorm copy no
    end-of-round effect but Mime's retrigger of a held card (card.lua: every other one has
    `not context.blueprint`), and Mime retriggers only a card that does something. So: the
    round is won (or Mr. Bones saves it), Mime acts, Blueprint or Brainstorm acts, and a held
    card has an end-of-round effect (a Gold card, or a Blue Seal with a free consumable slot;
    not debuffed)."""
    ids = acting(state)
    beaten = state.chips_scored >= state.blind_target
    bones = (ids == J_MR_BONES).any() & (state.chips_scored >= BONES_SHARE * state.blind_target)
    _, _, e, _, seal, _ = _gather(state, state.hand)
    ok = state.hand_valid & ~card_debuffed(state, state.hand)
    eff = ok & ((e == ENH_GOLD) | ((seal == SEAL_BLUE) & (cons_room(state) > 0)))
    copier = ((ids == J_BLUEPRINT) | (ids == J_BRAINSTORM)).any()
    return (beaten | bones) & (ids == J_MIME).any() & copier & eff.any()


def new_targets(key, state) -> Targets:
    """reset_idol_card / reset_mail_rank / reset_ancient_card / reset_castle_card:
    a random non-Stone card of the full deck (Idol, Mail-In rank, Castle suit);
    a new suit for Ancient Joker."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ok = state.card_live & (state.card_enh != ENH_STONE)
    pick = lambda k: jax.random.categorical(k, jnp.where(ok, 0.0, -jnp.inf))
    any_ok = ok.any()
    i1, i2, i4 = pick(k1), pick(k2), pick(k4)
    cur = state.ancient_suit                       # -1 at the start: any of the 4 suits
    a = jnp.floor(jax.random.uniform(k3) * jnp.where(cur < 0, 4, 3)).astype(jnp.int32)
    return Targets(jnp.where(any_ok, state.card_rank[i1] + 13 * state.card_suit[i1], 51),
                   jnp.where(any_ok, state.card_rank[i2], 12),
                   jnp.where((cur >= 0) & (a >= cur), a + 1, a),
                   jnp.where(any_ok, state.card_suit[i4], 3))


def joker_money(state: balatroEnvState, joker_value) -> jax.Array:
    """$ the acting END_OF_ROUND money jokers pay into the cashout (Golden Joker,
    Cloud 9, Rocket, Delayed Gratification, Satellite). Copies never pay."""
    rows, active, ctx = _jokers(state)
    v = resolve(rows, active, joker_value, ctx)
    pays = active & (rows.TRIGGER == T_END_OF_ROUND) & (rows.EFFECT == E_ADD_MONEY)
    return jnp.floor(jnp.where(pays, v, 0.0).sum()).astype(jnp.int32)


def _settle(state: balatroEnvState, played=False, go_end=False):
    """After every action. At the blind select: the tags that act there (enter_select). In a
    blind, the round ends when the target is reached (a PLAY, or a boss disabled by a sale)
    or a PLAY leaves no hands (update_hand_played checks the hands only after a hand: with 0
    hands after The Needle's disable, one more hand can be played). Else an empty hand is drawn
    again (game.lua: SELECTING_HAND with no
    card and a deck -> DRAW_TO_HAND; a Tarot or Spectral card can empty it), and with no card
    in the deck either the round ends. A draw at hand size 0 with no card loses the run at
    once (draw_from_deck_to_hand: GAME_OVER, so Mr. Bones cannot save it). played: the step
    was a PLAY; when it ends a won round and the joker order can still change the end of the
    round (end_window), the round waits in STAGE_END; go_end: NEXT there runs it.
    -> (state, reward, done)"""
    state = tree_where(state.game_stage == STAGE_SELECT, enter_select(state), state)
    blind = state.game_stage == STAGE_BLIND
    hand_empty = ~state.hand_valid.any()
    deck_left = state.deck_ptr < state.deck_size
    scored_out = (state.chips_scored >= state.blind_target) | ((state.hands <= 0) & played)
    dead = blind & ~scored_out & hand_empty & (state.hand_size <= 0)
    redraw = blind & ~scored_out & ~dead & hand_empty & deck_left
    state = tree_where(redraw, _redraw(state), state)
    ends = blind & ~dead & (scored_out | (~state.hand_valid.any() & (state.deck_ptr >= state.deck_size)))
    window = ends & played & end_window(state)
    run = (ends & ~window) | go_end
    ended = end_round(state)
    s = tree_where(run, ended, state)
    s = s._replace(game_stage=jnp.where(window, STAGE_END, jnp.where(dead, STAGE_LOST, s.game_stage)))
    reward = jnp.where(run & (s.game_stage != STAGE_LOST), s.last_cashout, 0).astype(jnp.int32)
    done = (s.game_stage == STAGE_LOST) | (s.game_stage == STAGE_WON)
    return s, reward, done


def _redraw(state):
    """DRAW_TO_HAND for an empty hand in the round: hand size cards (The Serpent: 3 once a hand
    was played or a card discarded); before any play or discard it is still the first hand
    (The House, Certificate: first_hand_drawn); then drawn_to_hand (Cerulean Bell; Crimson
    Heart only after a PLAY)."""
    key, k_draw, k_cert, k_bell = jax.random.split(jax.random.fold_in(state.key, K_DRAW + 50), 4)
    first = (state.hand_round.sum() == 0) & (state.discards_used == 0)
    s = draw(state._replace(key=key), k_draw, first=first, settle=False)
    s = tree_where(first, _first_hand_drawn(s, k_cert), s)
    return drawn_to_hand(s, k_bell, False)


# ---------------------------------------------------------------------------
# JOKERS IN — acquire, the joker pools
# ---------------------------------------------------------------------------
def acquire(state: balatroEnvState, jid, ed=ED_NONE, sticker=0) -> tuple[balatroEnvState, jax.Array]:
    """Put joker jid (edition ed, stickers: a shop or Buffoon pack joker's) in the first free
    slot. -> (state, ok). No money changes.

    The one place a joker enters the loadout: BUY, Buffoon packs, Judgement, Wraith,
    The Soul, Riff-raff, Top-up Tag and the GUI's dev button. ok is False (state
    unchanged) when the joker slots are full (a Negative brings its own slot) or jid is
    not a joker.
    """
    safe = jnp.clip(jid, 0, N_JOKERS - 1)
    held = (state.jokers != J_EMPTY).sum()
    neg = (ed == ED_NEG) & (state.joker_slots < MAX_JOKER_SLOTS)
    ok = (jid >= 0) & (jid < N_JOKERS) & (held < state.joker_slots + neg)
    kind = JOKERS.STATE[safe]
    todo = todo_pick(jax.random.fold_in(state.key, K_ACQUIRE), state.hand_played, jnp.int32(-1))
    start = jnp.select([kind == ST_TODO, kind == ST_LOYALTY], [todo, state.hand_played.sum()], 0)

    dest = held   # jokers are compact, so the first free slot is the count
    key, _ = jax.random.split(state.key)
    got = state._replace(
        jokers=state.jokers.at[dest].set(jid),
        joker_value=state.joker_value.at[dest].set(JOKERS.VALUE[safe]),
        joker_state=state.joker_state.at[dest].set(start),
        joker_sell_value=state.joker_sell_value.at[dest].set(sell_of(joker_cost(state, jid, ed, sticker))),
        joker_ed=state.joker_ed.at[dest].set(ed),
        joker_debuff=state.joker_debuff.at[dest].set(False),
        joker_down=state.joker_down.at[dest].set(False),
        joker_seen=state.joker_seen.at[dest].set(False),
        joker_sticker=state.joker_sticker.at[dest].set(sticker),
        joker_perish=state.joker_perish.at[dest].set(jnp.where(has_sticker(sticker, STK_PERISH), PERISH_ROUNDS, 0)),
        free_rerolls=state.free_rerolls + (jid == J_CHAOS),
        key=key,
    )
    got = apply_budget(got)
    # a new discard joker also adds to the discards left (ease_discard)
    got = got._replace(discards=state.discards + JOKERS.D_SIZE[safe])
    return tree_where(ok, reprice(got, state), state), ok


def joker_pool(state, held=None, rarity=0):
    """(N_JOKERS,) bool — the jokers create_card may make now: none that exists already
    (unless Showman), Gros Michel / Cavendish by extinction, and Steel / Stone / Lucky /
    Gold / Glass jokers only with such a card in the deck. rarity 0: the shop's (no
    Legendary); else only that rarity (Uncommon Tag, Rare Tag, Wraith, The Soul ...)."""
    showman = mods_of(state).showman
    jok, _ = _existing(state)
    if held is not None:
        jok = jok | held
    extinct = state.gros_extinct
    pool = jnp.select([JOKERS.POOL == P_NOT_EXTINCT, JOKERS.POOL == P_EXTINCT],
                      [~jnp.asarray(extinct), jnp.asarray(extinct)], default=True)
    has_enh = jnp.zeros(N_ENH, bool).at[state.card_enh].max(state.card_live)
    gate = (JOKERS.GATE_ENH < 0) | has_enh[jnp.clip(JOKERS.GATE_ENH, 0, N_ENH - 1)]
    kind = jnp.where(rarity == 0, JOKER_RARITY != LEGENDARY, JOKER_RARITY == rarity)
    return kind & pool & gate & (showman | ~jok)


def roll_joker(key, state, ed_key, taken=None, rarity=0):
    """create_card('Joker'): first the rarity (70 / 25 / 5 %: > 0.95 Rare, > 0.7 Uncommon; or
    the one asked for), then a joker uniform among the available ones of that rarity (the
    pool's UNAVAILABLE entries are resampled), and an edition (Hone / Glow Up). A rarity
    with none available gives Joker. -> (joker id, ED_*)"""
    k_rarity, k_pick = jax.random.split(key)
    u = jax.random.uniform(k_rarity)
    rolled = jnp.where(u > 0.95, RARE, jnp.where(u > 0.7, UNCOMMON, COMMON))
    avail = joker_pool(state, taken, jnp.where(rarity == 0, rolled, rarity))
    pick = jax.random.categorical(k_pick, jnp.where(avail, 0.0, -jnp.inf)).astype(jnp.int32)
    jid = jnp.where(avail.any(), pick, J_FALLBACK)
    return jid, poll_edition(jax.random.uniform(ed_key), rate=edition_rate(state))


# ---------------------------------------------------------------------------
# THE SHOP — cards, vouchers, packs; BUY / REDEEM / SELL / REROLL
# ---------------------------------------------------------------------------
class ShopRoll(NamedTuple):
    ids   : jax.Array   # (4,) item id per slot (by kind), -1 locked
    kinds : jax.Array   # (4,) IK_*
    eds   : jax.Array   # (4,) a joker's / playing card's edition
    rank  : jax.Array   # (4,) a playing card
    suit  : jax.Array
    enh   : jax.Array
    free  : jax.Array   # (4,) bool: couponed by a tag
    tags  : jax.Array   # (8,) the tags left
    tag_hand : jax.Array
    stickers : jax.Array   # (4,) a joker's stickers (the stake's Eternal / Perishable / Rental)


def roll_shop(key, state, fill=None) -> ShopRoll:
    """create_card_for_shop for each unlocked slot in fill (default: all): the first held
    Uncommon / Rare Tag makes the slot a free joker of that rarity; else a joker / Tarot /
    Planet / playing card / Spectral (Ghost Deck) by the rates (shop_rates). The first held
    edition tag (Foil ...
    Negative) puts its edition on a joker without one and makes it free. No card twice
    (without Showman), none that exists already; an empty pool gives Joker / Strength /
    Pluto. Illusion: a playing card is Enhanced 40 %, has an edition 20 %. A joker gets the
    stake's stickers (roll_sticker)."""
    fill = jnp.ones(MAX_SHOP_SLOTS, bool) if fill is None else fill
    logr = jnp.log(shop_rates(state))
    showman = mods_of(state).showman
    illusion = state.vouchers[V_ILLUSION]
    ids, kinds, eds = state.shop_slots, state.shop_kind, state.shop_ed
    rank, suit, enh, free = state.shop_rank, state.shop_suit, state.shop_enh, state.shop_free
    stks = state.shop_sticker
    tags, thand = state.tags, state.tag_hand
    taken_j = jnp.zeros(N_JOKERS, bool)
    taken_c = jnp.zeros(N_CONS, bool)
    s0 = state._replace(shop_slots=jnp.where(fill, -1, state.shop_slots))

    def slot(i, carry):
        key, ids, kinds, eds, rank, suit, enh, free, stks, tags, taken_j, taken_c = carry
        key, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 11)
        live = (i < state.shop_size) & fill[i]
        ur = (tags == TG_UNCOMMON) | (tags == TG_RARE)
        t_ur = jnp.argmax(ur)
        forced = live & ur.any()
        rarity = jnp.where(forced, jnp.where(tags[t_ur] == TG_RARE, RARE, UNCOMMON), 0)
        kind = jnp.where(forced, IK_JOKER, jax.random.categorical(k1, logr)).astype(jnp.int32)
        jid, ed = roll_joker(k2, s0, k3, taken_j, rarity)
        cid = roll_cons(k4, s0, jnp.select([kind == IK_TAROT, kind == IK_SPECTRAL], [CK_TAROT, CK_SPECTRAL],
                                           CK_PLANET), taken_c)
        card = jax.random.randint(k5, (), 0, 52)
        e = jnp.where(illusion & (jax.random.uniform(k6) > 0.6), jax.random.randint(k7, (), 1, N_ENH), ENH_NONE)
        u = jax.random.uniform(k9)
        ced = jnp.where(illusion & (jax.random.uniform(k8) > 0.8),
                        jnp.select([u > 0.85, u > 0.5], [ED_POLY, ED_HOLO], ED_FOIL), ED_NONE)
        tags = jnp.where(forced & (jnp.arange(MAX_TAGS) == t_ur), -1, tags)
        is_j = kind == IK_JOKER
        et = (tags >= 0) & (TAG_EDITION[jnp.clip(tags, 0, N_TAGS - 1)] >= 0)
        t_e = jnp.argmax(et)
        mod = live & is_j & (ed == ED_NONE) & et.any()
        ed = jnp.where(mod, TAG_EDITION[jnp.clip(tags[t_e], 0, N_TAGS - 1)], ed)
        tags = jnp.where(mod & (jnp.arange(MAX_TAGS) == t_e), -1, tags)
        item = jnp.select([is_j, kind == IK_CARD], [jid, 0], cid)
        locked = fill[i] & (i >= state.shop_size)
        ids = ids.at[i].set(jnp.where(live, item, jnp.where(locked, -1, ids[i])))
        kinds = kinds.at[i].set(jnp.where(live, kind, kinds[i]))
        eds = eds.at[i].set(jnp.where(live, jnp.select([is_j, kind == IK_CARD], [ed, ced], ED_NONE),
                                      jnp.where(locked, ED_NONE, eds[i])))
        rank = rank.at[i].set(jnp.where(live, card % 13, rank[i]))
        suit = suit.at[i].set(jnp.where(live, card // 13, suit[i]))
        enh = enh.at[i].set(jnp.where(live & (kind == IK_CARD), e, jnp.where(live, ENH_NONE, enh[i])))
        free = free.at[i].set(jnp.where(fill[i], live & (forced | mod), free[i]))
        stk = jnp.where(is_j, roll_sticker(k10, state.stake, jid), 0)
        stks = stks.at[i].set(jnp.where(live, stk, jnp.where(locked, 0, stks[i])))
        taken_j = taken_j | ((jnp.arange(N_JOKERS) == jid) & live & is_j & ~showman)
        is_c = (kind == IK_TAROT) | (kind == IK_PLANET) | (kind == IK_SPECTRAL)
        taken_c = taken_c | ((jnp.arange(N_CONS) == cid) & live & is_c & ~showman)
        return key, ids, kinds, eds, rank, suit, enh, free, stks, tags, taken_j, taken_c

    carry = (key, ids, kinds, eds, rank, suit, enh, free, stks, tags, taken_j, taken_c)
    out = jax.lax.fori_loop(0, MAX_SHOP_SLOTS, slot, carry)
    key, ids, kinds, eds, rank, suit, enh, free, stks, tags, taken_j, taken_c = out
    keep = tags >= 0
    return ShopRoll(ids, kinds, eds, rank, suit, enh, free, compact(tags, keep), compact(thand, keep), stks)


def _apply_roll(state, roll: ShopRoll):
    return state._replace(shop_slots=roll.ids, shop_kind=roll.kinds, shop_ed=roll.eds,
                          shop_rank=roll.rank, shop_suit=roll.suit, shop_enh=roll.enh,
                          shop_free=roll.free, tags=roll.tags, tag_hand=roll.tag_hand,
                          shop_sticker=roll.stickers)


def roll_voucher(key, state, exclude):
    """get_next_voucher_key: uniform over the vouchers not redeemed whose first one is (if
    any), not in the shop already; Blank when none is left."""
    v = state.vouchers
    needs = VOUCHER_NEEDS
    ok = ~v & ((needs < 0) | v[jnp.clip(needs, 0, N_VOUCHERS - 1)]) & ~exclude
    pick = jax.random.categorical(key, jnp.where(ok, 0.0, -jnp.inf)).astype(jnp.int32)
    return jnp.where(ok.any(), pick, V_BLANK).astype(jnp.int32)


def roll_packs(key, state):
    """get_pack x 2: by the pack weights; the first shop's first pack is a Buffoon Pack."""
    picks = jax.random.categorical(key, PACK_LOGW, shape=(PACK_SLOTS,)).astype(jnp.int32)
    picks = picks.at[0].set(jnp.where(state.buffoon_given, picks[0], P_BUFFOON_NORMAL))
    return picks


def _take_tag(tags, hands, mask):
    """Drop the tags in mask; the rest stay in order."""
    keep = (tags >= 0) & ~mask
    return compact(jnp.where(keep, tags, -1), keep), compact(hands, keep)


def open_shop(key, state):
    """update_shop: the first D6 Tag (rerolls start at $0), the cards (the Uncommon / Rare /
    edition tags), this ante's voucher and one per Voucher Tag, 2 packs, then the first
    Coupon Tag (the cards and packs are free)."""
    k1, k2, k3 = jax.random.split(key, 3)
    d6 = state.tags == TG_D6
    first_d6 = d6 & (jnp.cumsum(d6) == 1)
    tags, hands = _take_tag(state.tags, state.tag_hand, first_d6)
    state = state._replace(tags=tags, tag_hand=hands, reroll_cost=jnp.where(d6.any(), 0, state.reroll_cost),
                           reroll_d6=d6.any(), shop_free=jnp.zeros(MAX_SHOP_SLOTS, bool))
    state = _apply_roll(state, roll_shop(k1, state))
    sv = jnp.full(MAX_SHOP_VOUCHERS, -1, jnp.int32).at[0].set(state.ante_voucher)
    vt = state.tags == TG_VOUCHER
    for i in range(MAX_SHOP_VOUCHERS - 1):               # static: one voucher per Voucher Tag
        in_shop = (jnp.arange(N_VOUCHERS)[:, None] == sv[None, :]).any(axis=1)
        v = roll_voucher(jax.random.fold_in(k2, i), state, in_shop)
        go = vt.sum() > i
        n = (sv >= 0).sum()
        sv = jnp.where(go & (jnp.arange(MAX_SHOP_VOUCHERS) == n), v, sv)
    tags, hands = _take_tag(state.tags, state.tag_hand, vt)
    state = state._replace(shop_vouchers=compact(sv, sv >= 0), tags=tags, tag_hand=hands,
                           pack_slots=roll_packs(k3, state), pack_free=jnp.zeros(PACK_SLOTS, bool),
                           buffoon_given=jnp.bool_(True))
    coupon = state.tags == TG_COUPON
    first_c = coupon & (jnp.cumsum(coupon) == 1)
    tags, hands = _take_tag(state.tags, state.tag_hand, first_c)
    on = coupon.any()
    return state._replace(tags=tags, tag_hand=hands,
                          shop_free=state.shop_free | (on & (state.shop_slots >= 0)),
                          pack_free=jnp.where(on, state.pack_slots >= 0, state.pack_free))


def _buy(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, MAX_SHOP_SLOTS - 1)
    item, kind, ed = state.shop_slots[slot], state.shop_kind[slot], state.shop_ed[slot]
    cost = shop_item_cost(state, slot)
    got_j, room_j = acquire(state, item, ed, state.shop_sticker[slot])
    got_c = give_cons(state, item)
    room_c = cons_room(state) > 0
    got_p, room_p = add_card(state, state.shop_rank[slot], state.shop_suit[slot], state.shop_enh[slot], ed,
                             SEAL_NONE)
    got_p = _event(got_p, T_ON_CARD_ADDED, room_p.astype(jnp.float32))
    branch = jnp.select([kind == IK_JOKER, kind == IK_CARD], [0, 2], 1)
    got = jax.lax.switch(branch, [lambda: got_j, lambda: got_c, lambda: got_p])
    room = jnp.select([kind == IK_JOKER, kind == IK_CARD], [room_j, room_p], room_c)
    ok = (t >= 0) & (t < state.shop_size) & (item >= 0) & affordable(state, cost) & room
    bought = got._replace(money=state.money - cost, shop_slots=state.shop_slots.at[slot].set(-1),
                          shop_free=state.shop_free.at[slot].set(False),
                          shop_sticker=state.shop_sticker.at[slot].set(0))
    return tree_where(ok, bought, state), jnp.int32(0), jnp.bool_(False)


def apply_voucher(state, v, old, key):
    """card.lua apply_to_run of voucher v (redeemed): what changes at once."""
    is_ = lambda *names: jnp.isin(v, jnp.array(names))
    more_shop = is_(V_OVERSTOCK, V_OVERSTOCK_PLUS)
    s = state._replace(
        shop_size=jnp.minimum(state.shop_size + more_shop, MAX_SHOP_SLOTS),
        reroll_cost=jnp.where(is_(V_REROLL_SURPLUS, V_REROLL_GLUT), jnp.maximum(state.reroll_cost - 2, 0),
                              state.reroll_cost),
        consumable_slots=state.consumable_slots + is_(V_CRYSTAL_BALL),
        base_hands=state.base_hands + is_(V_GRABBER, V_NACHO_TONG) - is_(V_HIEROGLYPH),
        base_discards=state.base_discards + is_(V_WASTEFUL, V_RECYCLOMANCY) - is_(V_PETROGLYPH),
        base_hand_size=state.base_hand_size + is_(V_PAINT_BRUSH, V_PALETTE),
        ante=state.ante - is_(V_HIEROGLYPH, V_PETROGLYPH),
    )
    s = reprice(apply_budget(s), old)                     # Clearance Sale / Liquidation
    # Overstock: every empty card slot gets a card at once, a bought one too (change_shop_size:
    # joker_max - #cards new cards)
    fill = (jnp.arange(MAX_SHOP_SLOTS) < s.shop_size) & (state.shop_slots < 0)
    return tree_where(more_shop, _apply_roll(s, roll_shop(key, s, fill)), s)


def _redeem(state: balatroEnvState, action: balatroAction):
    """REDEEM the voucher in slot target. Any redeem takes this ante's voucher out of the
    next shops (Card:redeem sets current_round.voucher to nil)."""
    t = action.target
    slot = jnp.clip(t, 0, MAX_SHOP_VOUCHERS - 1)
    v = state.shop_vouchers[slot]
    cost = voucher_cost(state)
    ok = (t >= 0) & (t < MAX_SHOP_VOUCHERS) & (v >= 0) & (state.money - cost >= bankrupt_at(state))
    key, sub = jax.random.split(jax.random.fold_in(state.key, K_VOUCHER))
    s = state._replace(money=state.money - cost, vouchers=state.vouchers.at[jnp.clip(v, 0, N_VOUCHERS - 1)].set(True),
                       shop_vouchers=state.shop_vouchers.at[slot].set(-1), ante_voucher=jnp.int32(-1), key=key)
    s = apply_voucher(s, v, state, sub)
    return tree_where(ok, s, state), jnp.int32(0), jnp.bool_(False)


def _sell(state: balatroEnvState, action: balatroAction):
    """SELL the joker in slot target (not an Eternal one): the other jokers see a card sold
    (Campfire); its own ability on a sale (Luchador: disable the boss; Diet Cola: a Double Tag;
    Invisible Joker after 2 rounds: a copy of a random other joker); then the $ and it is gone.
    A sold joker disables Verdant Leaf."""
    t = action.target
    slot = jnp.clip(t, 0, MAX_JOKERS - 1)
    jid = state.jokers[slot]
    ok = (t >= 0) & (t < MAX_JOKERS) & (jid != J_EMPTY) & ~eternal(state)[slot]     # can_sell_card
    gone = jnp.arange(MAX_JOKERS) == slot
    acts = ~state.joker_debuff[slot]
    k1, key = jax.random.split(jax.random.fold_in(state.key, K_SELL))

    # T_ON_SELL_OTHER fires for every other acting joker (Campfire)
    ids = acting(state)
    rows, active = rows_for(ids), (ids != J_EMPTY) & ~gone
    value, _ = update(rows, active, state.joker_value, state.joker_state, jnp.int32(T_ON_SELL_OTHER))
    s = state._replace(joker_value=value, key=key)
    # selling_self: a sold Blueprint / Brainstorm runs the copied joker's own sale ability
    # (Luchador, Diet Cola have no blueprint guard; Invisible Joker does: it is not copyable)
    erows, eok, _, _ = effective(ids, state.joker_value, state.joker_state)
    eff = jnp.where(acts & eok[slot] & (erows.TRIGGER[slot] == T_ON_SOLD), erows.EFFECT[slot], -1)
    s = tree_where(eff == E_DISABLE_BOSS, disable_boss(s), s)
    s = tree_where(eff == E_ADD_TAG, add_tag(s, TG_DOUBLE, -1), s)
    others = (s.jokers != J_EMPTY) & ~gone
    dup = (eff == E_DUPLICATE) & (state.joker_value[slot] >= JOKERS.VALUE2[jnp.clip(jid, 0, N_JOKERS - 1)]) \
        & others.any()
    # the copy of an Invisible Joker starts at 0 rounds (card.lua: invis_rounds = 0; Ankh's copy keeps them)
    n_held = (s.jokers != J_EMPTY).sum()
    copied = copy_joker(s, _random_slot(k1, others))
    fresh = copied.jokers[n_held] == J_INVISIBLE
    copied = copied._replace(joker_value=copied.joker_value.at[n_held].set(
        jnp.where(fresh, 0.0, copied.joker_value[n_held])))
    s = tree_where(dup, copied, s)
    sold = remove_jokers(s._replace(money=s.money + state.joker_sell_value[slot]), gone)
    sold = tree_where(boss_is(sold, BK_LEAF), disable_boss(sold), sold)
    # in a round a larger hand size draws at once (Merry Andy / Stuntman sold, a Juggler copied,
    # The Manacle let go: its change_size draws 1 more besides its own draw)
    sold = _grow_draw(sold, state.hand_size, jax.random.fold_in(k1, 7))
    return tree_where(ok, sold, state), jnp.int32(0), jnp.bool_(False)


def reroll_price(state):
    """$ the next reroll costs (0 while Chaos the Clown has a free one)."""
    return jnp.where(state.free_rerolls > 0, 0, state.reroll_cost)


def _reroll_shop(state: balatroEnvState):
    free = state.free_rerolls > 0
    cost = reroll_price(state)
    ok = free | affordable(state, cost)
    key, sub = jax.random.split(state.key)
    rows, active, _ = _jokers(state)
    value, _ = update(rows, active, state.joker_value, state.joker_state, jnp.int32(T_ON_REROLL))
    s = state._replace(shop_free=jnp.zeros(MAX_SHOP_SLOTS, bool))
    rolled = _apply_roll(s, roll_shop(sub, s))._replace(
        money=state.money - cost,
        # calculate_reroll_cost: (D6 ? $0 : the base) + the paid rerolls; a free one changes nothing
        reroll_cost=jnp.where(free, state.reroll_cost,
                              jnp.where(state.reroll_d6, 0, reroll_base(state)) + state.reroll_inc + 1),
        reroll_inc=state.reroll_inc + (~free),
        free_rerolls=state.free_rerolls - free,
        joker_value=value,
        key=key,
    )
    return tree_where(ok, rolled, state)


def _reroll(state: balatroEnvState, action: balatroAction):
    """REROLL: the shop's cards; at the blind select: the boss."""
    return tree_where(state.game_stage == STAGE_SHOP, _reroll_shop(state), _reroll_boss(state)), \
        jnp.int32(0), jnp.bool_(False)


# ---------------------------------------------------------------------------
# BOOSTER PACKS — OPEN / PICK / SKIP
# ---------------------------------------------------------------------------
def most_played(state):
    """The hand played the most this run (a tie: the better hand), -1 if none (Telescope)."""
    h = jnp.argmax(state.hand_played).astype(jnp.int32)
    return jnp.where(state.hand_played.max() > 0, h, -1)


def roll_pack_cards(key, state, pid):
    """card.lua Card:open: the pack's offers. -> (items, eds, rank, suit, enh, seal, stickers), each
    (MAX_PACK,).
    Arcana: Tarots (Omen Globe: each 20 % a Spectral card); Celestial: Planets (Telescope: the
    first is the most played hand's); Spectral: Spectral cards; each 0.3 % The Soul (Tarot,
    Spectral) / Black Hole (Planet, Spectral); no repeats, none that exists. Standard: a
    playing card (40 % enhanced, an edition at double odds, 20 % a seal); Buffoon: jokers as
    in the shop (the stake's stickers too)."""
    p = jnp.clip(pid, 0, N_PACKS - 1)
    kind, size = PACK_KIND[p], PACK_SIZE[p]
    showman = mods_of(state).showman
    rate = edition_rate(state)
    tele_hand = most_played(state)
    items = jnp.full(MAX_PACK, -1, jnp.int32)
    eds = jnp.zeros(MAX_PACK, jnp.int32)
    rank = jnp.zeros(MAX_PACK, jnp.int32)
    suit = jnp.zeros(MAX_PACK, jnp.int32)
    enh = jnp.zeros(MAX_PACK, jnp.int32)
    seal = jnp.zeros(MAX_PACK, jnp.int32)
    stks = jnp.zeros(MAX_PACK, jnp.int32)
    taken_j = jnp.zeros(N_JOKERS, bool)
    taken_c = jnp.zeros(N_CONS, bool)
    is_c = (kind == PK_ARCANA) | (kind == PK_CELESTIAL) | (kind == PK_SPECTRAL)
    is_j = kind == PK_BUFFOON
    is_s = kind == PK_STANDARD

    def card_i(i, carry):
        key, items, eds, rank, suit, enh, seal, stks, taken_j, taken_c = carry
        key, *k = jax.random.split(key, 12)
        live = i < size
        omen = state.vouchers[V_OMEN_GLOBE] & (jax.random.uniform(k[9]) > 0.8)
        ckind = jnp.select([kind == PK_ARCANA, kind == PK_CELESTIAL], [jnp.where(omen, CK_SPECTRAL, CK_TAROT),
                                                                        CK_PLANET], CK_SPECTRAL)
        cid = roll_pack_cons(k[0], state, ckind, taken_c)
        tele = (kind == PK_CELESTIAL) & state.vouchers[V_TELESCOPE] & (i == 0) & (tele_hand >= 0)
        cid = jnp.where(tele, PLANET_OF_HAND[jnp.clip(tele_hand, 0, N_HAND_TYPES - 1)], cid)
        jid, jed = roll_joker(k[1], state, k[2], taken_j)
        card = jax.random.randint(k[3], (), 0, 52)
        e = jnp.where(jax.random.uniform(k[4]) < STANDARD_ENHANCED,
                      jax.random.randint(k[5], (), 1, N_ENH), ENH_NONE)
        ced = poll_edition(jax.random.uniform(k[6]), mod=2.0, no_neg=True, rate=rate)
        u = jax.random.uniform(k[8])
        sl = jnp.where(jax.random.uniform(k[7]) < STANDARD_SEAL,
                       jnp.select([u > 0.75, u > 0.5, u > 0.25], [SEAL_RED, SEAL_BLUE, 1], SEAL_PURPLE),
                       SEAL_NONE)
        items = items.at[i].set(jnp.where(~live, -1, jnp.select([is_c, is_j], [cid, jid], 0)))
        eds = eds.at[i].set(jnp.select([is_j, is_s], [jed, ced], ED_NONE) * live)
        rank = rank.at[i].set(card % 13)
        suit = suit.at[i].set(card // 13)
        enh = enh.at[i].set(jnp.where(is_s, e, ENH_NONE))
        seal = seal.at[i].set(jnp.where(is_s, sl, SEAL_NONE))
        stks = stks.at[i].set(jnp.where(is_j & live, roll_sticker(k[10], state.stake, jid), 0))
        taken_c = taken_c | ((jnp.arange(N_CONS) == cid) & live & is_c & ~showman)
        taken_j = taken_j | ((jnp.arange(N_JOKERS) == jid) & live & is_j & ~showman)
        return key, items, eds, rank, suit, enh, seal, stks, taken_j, taken_c

    out = jax.lax.fori_loop(0, MAX_PACK, card_i, (key, items, eds, rank, suit, enh, seal, stks, taken_j, taken_c))
    _, items, eds, rank, suit, enh, seal, stks, _, _ = out
    return items, eds, rank, suit, enh, seal, stks


def _deal_arcana_hand(state):
    """An Arcana / Spectral pack draws a hand to use the cards on: the next hand-size cards
    from the top of the deck (draw_from_deck_to_hand; no shuffle: the deck is shuffled only at
    the run start, Cash Out and the blind start, so a second pack deals the next cards)."""
    lanes = jnp.arange(MAX_HAND_SIZE)
    n = jnp.clip(jnp.minimum(state.hand_size, state.deck_size - state.deck_ptr), 0, MAX_HAND_SIZE)
    hand_valid = lanes < n
    hand = jnp.where(hand_valid, state.deck[jnp.clip(state.deck_ptr + lanes, 0, MAX_DECK_SIZE - 1)], -1)
    return state._replace(hand=hand, hand_valid=hand_valid, hand_down=jnp.zeros(MAX_HAND_SIZE, bool),
                          deck_ptr=state.deck_ptr + n)


def open_pack(state, pid, key, back):
    """Card:open of pack pid: its cards, STAGE_PACK (back to `back` when it closes); an Arcana
    or Spectral pack deals a hand; Hallucination (copies too): 1 in 2 a Tarot."""
    k_cards, _, k_hall = jax.random.split(key, 3)        # (the 2nd key dealt the hand from a new shuffle)
    p = jnp.clip(pid, 0, N_PACKS - 1)
    items, eds, rank, suit, enh, seal, stks = roll_pack_cards(k_cards, state, pid)
    opened = state._replace(game_stage=jnp.int32(STAGE_PACK), pack_id=jnp.asarray(pid, jnp.int32),
                            pack_left=PACK_CHOOSE[p], pack_items=items, pack_ed=eds,
                            pack_rank=rank, pack_suit=suit, pack_enh=enh, pack_seal=seal, pack_sticker=stks,
                            pack_return=jnp.asarray(back, jnp.int32))
    deals = (PACK_KIND[p] == PK_ARCANA) | (PACK_KIND[p] == PK_SPECTRAL)
    opened = tree_where(deals, _deal_arcana_hand(opened), opened)
    hall = _effects(state, T_ON_PACK_OPEN, E_SPAWN_CONSUMABLE)
    erows, _, _, _ = effective(acting(state), state.joker_value, state.joker_state)
    hits = (hall & (jax.random.uniform(k_hall, (MAX_JOKERS,)) < _odds(erows, mods_of(state)))).sum()
    return spawn_cons(opened, CK_TAROT, hits, jax.random.fold_in(k_hall, 1))


def _open(state: balatroEnvState, action: balatroAction):
    t = action.target
    slot = jnp.clip(t, 0, PACK_SLOTS - 1)
    pid = state.pack_slots[slot]
    cost = pack_cost(state, pid, state.pack_free[slot])
    ok = (t >= 0) & (t < PACK_SLOTS) & (pid >= 0) & affordable(state, cost)
    k_open, key = jax.random.split(jax.random.fold_in(state.key, K_PACK))
    s = state._replace(money=state.money - cost, pack_slots=state.pack_slots.at[slot].set(-1),
                       pack_free=state.pack_free.at[slot].set(False), key=key)
    opened = open_pack(s, pid, k_open, STAGE_SHOP)
    return tree_where(ok, opened, state), jnp.int32(0), jnp.bool_(False)


def _close_pack(state):
    """Back to the shop or the blind select: the drawn hand goes back to the bottom of the deck
    (draw_from_hand_to_deck: the leftmost card first, each one emplaced at the bottom), so the
    undealt cards come next. At the blind select the next pack tag (or Boss Tag) acts
    (end_consumeable): _settle does it."""
    pos = jnp.arange(MAX_DECK_SIZE)
    undealt = (pos >= state.deck_ptr) & (pos < state.deck_size)
    # the hand's cards in lane order (the cards a Tarot made in the hand are dealt too)
    rank = jnp.cumsum(state.hand_valid) - 1
    lane_of = jnp.full(MAX_DECK_SIZE, -1, jnp.int32).at[jnp.where(state.hand_valid, state.hand, MAX_DECK_SIZE)].set(
        rank.astype(jnp.int32), mode="drop")
    back = lane_of[state.deck]
    keys = jnp.where(undealt, pos - state.deck_ptr,
                     jnp.where(back >= 0, state.deck_size - state.deck_ptr + back, MAX_DECK_SIZE + pos))
    deck = state.deck[jnp.argsort(keys)]
    return state._replace(game_stage=state.pack_return, pack_id=jnp.int32(-1),
                          pack_left=jnp.int32(0), pack_items=jnp.full(MAX_PACK, -1, jnp.int32),
                          pack_sticker=jnp.zeros(MAX_PACK, jnp.int32), deck=deck,
                          hand_valid=jnp.zeros(MAX_HAND_SIZE, bool), hand_down=jnp.zeros(MAX_HAND_SIZE, bool),
                          hand=jnp.full(MAX_HAND_SIZE, -1, jnp.int32), deck_ptr=jnp.int32(0))


def pick_ok(state, t, select=None):
    """Can pack card t be taken now? A Tarot / Planet / Spectral card must be usable, a joker
    needs room. select None: before the card selection (can_use_base)."""
    i = jnp.clip(t, 0, MAX_PACK - 1)
    item = state.pack_items[i]
    kind = PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)]
    held = (state.jokers != J_EMPTY).sum()
    neg = (state.pack_ed[i] == ED_NEG) & (state.joker_slots < MAX_JOKER_SLOTS)
    use = can_use_base(state, item, False) if select is None else can_use(state, item, select, False)
    cons = (kind == PK_ARCANA) | (kind == PK_CELESTIAL) | (kind == PK_SPECTRAL)
    ok = jnp.select([cons, kind == PK_BUFFOON], [use, held < state.joker_slots + neg],
                    default=(~state.card_live).any())
    return (t >= 0) & (t < MAX_PACK) & (item >= 0) & (state.pack_left > 0) & ok


def _skip(state: balatroEnvState, action: balatroAction):
    """SKIP: close the pack (Red Card +3 Mult); at the blind select: skip the blind."""
    closed = _close_pack(_event(state, T_ON_PACK_SKIP))
    return tree_where(state.game_stage == STAGE_PACK, closed, _skip_blind(state)), jnp.int32(0), jnp.bool_(False)


# ---------------------------------------------------------------------------
# THE BLIND SELECT — tags, skips, the boss
# ---------------------------------------------------------------------------
def new_boss(key, state):
    """get_new_boss for state.ante: the eligible bosses (min ante; the showdown bosses only at
    ante 8, 16 ...), the least used of them, uniform. -> (boss, bosses_used)"""
    ante = state.ante
    a1 = jnp.maximum(1, ante)
    regular = ~BOSS_SHOWDOWN & (BOSS_MIN_ANTE <= a1) & ((a1 % WIN_ANTE != 0) | (ante < 2))
    show = BOSS_SHOWDOWN & (ante % WIN_ANTE == 0) & (ante >= 2)
    elig = regular | show
    uses = jnp.where(elig, state.bosses_used, 1 << 30)
    pool = elig & (state.bosses_used == uses.min())
    b = jax.random.categorical(key, jnp.where(pool, 0.0, -jnp.inf)).astype(jnp.int32)
    return b, state.bosses_used.at[b].add(1)


def visible_hands(state):
    """The hands the game shows: the 9 plain ones, and a secret one once it was played."""
    return (jnp.arange(N_HAND_TYPES) >= SECRET_HANDS) | (state.hand_played > 0)


def todo_pick(key, hand_played, old):
    """To Do List's new hand (card.lua set_ability / end of round): a random visible hand (a
    secret one once played), never old. old: the hand now (-1 at the buy); an (n,) old gives
    n picks."""
    ok = (jnp.arange(N_HAND_TYPES) >= SECRET_HANDS) | (hand_played > 0)
    ok = ok & (jnp.arange(N_HAND_TYPES) != jnp.asarray(old)[..., None])
    return jax.random.categorical(key, jnp.where(ok, 0.0, -jnp.inf), shape=jnp.shape(old)).astype(jnp.int32)


def new_ante_rolls(key, state):
    """The cash out after a boss (and the run start): the ante's boss, the Small / Big Blind
    tags (by min ante) and their Orbital Tag hands (a visible hand)."""
    kb, k1, k2, k3, k4 = jax.random.split(key, 5)
    boss, used = new_boss(kb, state)
    ok = TAG_MIN_ANTE <= state.ante
    tag = lambda k: jax.random.categorical(k, jnp.where(ok, 0.0, -jnp.inf)).astype(jnp.int32)
    vis = visible_hands(state)
    hand = lambda k: jax.random.categorical(k, jnp.where(vis, 0.0, -jnp.inf)).astype(jnp.int32)
    return state._replace(boss=boss, bosses_used=used, boss_rerolled=jnp.bool_(False),
                          blind_tags=jnp.stack([tag(k1), tag(k2)]), orbital_hands=jnp.stack([hand(k3), hand(k4)]))


def _tag_span(tags):
    """The loop bound for the tag loops: 1 + the last held slot (0 with no tag). An empty slot
    is a no-op in each loop, so the slots after it need no pass (under vmap the loop runs to
    the largest bound in the batch, not to MAX_TAGS)."""
    return jnp.max(jnp.where(tags >= 0, jnp.arange(MAX_TAGS) + 1, 0))


def add_tag(state, tag, hand):
    """add_tag: each held Double Tag gives a copy of a new tag that is not a Double Tag (and is
    used up); the tag and its copies join the end of the list (a full list drops the rest)."""
    doubles = state.tags == TG_DOUBLE
    copies = (tag != TG_DOUBLE) & (tag >= 0)
    keep = (state.tags >= 0) & ~(doubles & copies)
    tags, hands = compact(jnp.where(keep, state.tags, -1), keep), compact(state.tag_hand, keep)
    held = keep.sum()
    n = (tag >= 0) * (1 + jnp.where(copies, doubles.sum(), 0))
    pos = jnp.arange(MAX_TAGS)
    new = (pos >= held) & (pos < held + n)
    return state._replace(tags=jnp.where(new, tag, tags).astype(jnp.int32),
                          tag_hand=jnp.where(new, hand, hands).astype(jnp.int32))


def immediate_tags(state, key):
    """The 'immediate' tags, in order: Top-up (up to 2 Common jokers), Skip ($5 per blind
    skipped), Garbage ($1 per unused discard), Handy ($1 per hand played), Economy (double
    the money, at most +$40), Orbital (+3 levels to its hand). Each is used up."""
    def body(i, s):
        t = s.tags[i]
        k = jax.random.fold_in(key, i)
        k1, k2, k3, k4 = jax.random.split(k, 4)
        top = t == TG_TOP_UP
        for kk in ((k1, k2), (k3, k4)):                  # static: 2 jokers, each into a free slot
            jid, ed = roll_joker(kk[0], s, kk[1], rarity=COMMON)
            room = (s.jokers != J_EMPTY).sum() < s.joker_slots   # a Negative needs a free slot here too
            s = tree_where(top & room, acquire(s, jid, ed)[0], s)
        money = jnp.select([t == TG_SKIP, t == TG_GARBAGE, t == TG_HANDY, t == TG_ECONOMY],
                           [SKIP_TAG_DOLLARS * s.skips, s.unused_discards, s.hand_played.sum(),
                            jnp.minimum(ECONOMY_MAX, jnp.maximum(0, s.money))], 0)
        orb = t == TG_ORBITAL
        level = s.hand_level.at[jnp.clip(s.tag_hand[i], 0, N_HAND_TYPES - 1)].add(ORBITAL_LEVELS * orb)
        used = top | (money > 0) | orb | jnp.isin(t, jnp.array([TG_SKIP, TG_GARBAGE, TG_HANDY, TG_ECONOMY]))
        return s._replace(money=s.money + money, hand_level=level, tags=s.tags.at[i].set(jnp.where(used, -1, t)))
    s = jax.lax.fori_loop(0, _tag_span(state.tags), body, state)
    keep = s.tags >= 0
    return s._replace(tags=compact(s.tags, keep), tag_hand=compact(s.tag_hand, keep))


def new_blind_choice(state):
    """The 'new_blind_choice' tags, in order, until one opens a pack: Boss Tag (reroll the
    boss), Charm / Meteor / Ethereal / Standard / Buffoon Tag (a free pack; when it closes the
    next one acts). Each is used up."""
    key = jax.random.fold_in(state.key, K_TAG)

    def body(i, carry):
        s, stopped = carry
        t = s.tags[i]
        tt = jnp.clip(t, 0, N_TAGS - 1)
        boss = ~stopped & (t == TG_BOSS)
        pack = ~stopped & (t >= 0) & (TAG_PACK[tt] >= 0)
        k1, k2 = jax.random.split(jax.random.fold_in(key, i))
        b, used = new_boss(k1, s)
        s = s._replace(boss=jnp.where(boss, b, s.boss), bosses_used=jnp.where(boss, used, s.bosses_used),
                       boss_rerolled=s.boss_rerolled | boss, tags=s.tags.at[i].set(jnp.where(boss | pack, -1, t)))
        s = tree_where(pack, open_pack(s, TAG_PACK[tt], k2, STAGE_SELECT), s)
        return s, stopped | pack
    s, opened = jax.lax.fori_loop(0, _tag_span(state.tags), body, (state, jnp.bool_(False)))
    keep = s.tags >= 0
    s = s._replace(tags=compact(s.tags, keep), tag_hand=compact(s.tag_hand, keep))
    # a pack moves the key on: the next tag's pack (a Double Tag's copy) is a new roll, even when
    # this one is skipped (a SKIP does not move the key)
    return tree_where(opened, s._replace(key=jax.random.split(state.key)[0]), s)


def enter_select(state):
    """update_blind_select: the immediate tags, then the first new_blind_choice tags."""
    s = immediate_tags(state, jax.random.fold_in(state.key, K_TAG + 50))
    return new_blind_choice(s)


def _skip_blind(state):
    """skip_blind (Small / Big only): +1 skip, its tag (Double Tags copy it), Throwback reads
    the skips, the next blind is on deck; the immediate and new_blind_choice tags act in
    _settle."""
    ok = state.blind_index < BLIND_BOSS
    b = jnp.clip(state.blind_index, 0, 1)
    s = state._replace(skips=state.skips + 1)
    s = add_tag(s, state.blind_tags[b], state.orbital_hands[b])
    key, _ = jax.random.split(state.key)
    s = s._replace(blind_index=state.blind_index + 1, key=key)
    return tree_where(ok, s, state)


def _reroll_boss(state):
    """reroll_boss ($10): Retcon any time, Director's Cut once per ante (a Boss Tag counts)."""
    v = state.vouchers
    can = (v[V_RETCON] | (v[V_DIRECTORS_CUT] & ~state.boss_rerolled)) \
        & (state.money - BOSS_REROLL_COST >= bankrupt_at(state))
    key, sub = jax.random.split(jax.random.fold_in(state.key, K_BOSS))
    b, used = new_boss(sub, state)
    s = state._replace(money=state.money - BOSS_REROLL_COST, boss=b, bosses_used=used,
                       boss_rerolled=jnp.bool_(True), key=key)
    return tree_where(can, s, state)


def _perkeo(state, key):
    """Perkeo (copies too), leaving the shop: a Negative copy of a random held consumable,
    one Perkeo after the other (a later one may copy an earlier copy)."""
    erows, eok, _, _ = effective(acting(state), state.joker_value, state.joker_state)
    n = (eok & (erows.TRIGGER == T_ON_SHOP_EXIT) & (erows.EFFECT == E_NEGATIVE_COPY)).sum()

    def copy(i, state):                                    # Perkeo and copies: up to 4
        held = state.consumables != CONS_EMPTY
        j = _random_slot(jax.random.fold_in(key, i), held)
        cid = state.consumables[j]
        extra = state.cons_sell[j] - sell_of(cons_cost(state, cid, state.cons_ed[j]))
        n_held = held.sum()
        go = (i < n) & held.any()
        s = give_cons(state, cid, go, ED_NEG)
        return s._replace(cons_sell=s.cons_sell.at[n_held].add(jnp.where(go, extra, 0)))
    return jax.lax.fori_loop(0, 4, copy, state)


def _leave_shop(state):
    """NEXT in the shop: Perkeo, then the blind select of the blind on deck."""
    key, k_perkeo = jax.random.split(jax.random.fold_in(state.key, K_SHOP_EXIT))
    return _perkeo(state, k_perkeo)._replace(
        game_stage=jnp.int32(STAGE_SELECT), key=key,
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, jnp.int32), shop_free=jnp.zeros(MAX_SHOP_SLOTS, bool),
        shop_sticker=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32),
        shop_vouchers=jnp.full(MAX_SHOP_VOUCHERS, -1, jnp.int32),
        pack_slots=jnp.full(PACK_SLOTS, -1, jnp.int32), pack_free=jnp.zeros(PACK_SLOTS, bool))


def _setting_blind(state, key):
    """The jokers' setting_blind, in slot order. A joker being destroyed (getting_sliced) does
    not act, and neither does a copy of it. Madness (Small / Big): X0.5 more and a random
    other joker is destroyed; Burglar (copies too): +3 hands, no discards; Riff-raff (copies
    too): up to 2 Common jokers (room, counting the jokers a Dagger destroys); Cartomancer
    (copies too): a Tarot; Ceremonial Dagger: destroys the joker to its right, +2 x its sell
    value as Mult; Marble Joker (copies too): a Stone card into the deck. Then the destroyed
    jokers go and the new ones come. Madness and the Dagger never destroy an Eternal joker."""
    is_boss = state.blind_index == BLIND_BOSS
    src = copy_sources(acting(state))
    n_held = (state.jokers != J_EMPTY).sum()
    sliced = jnp.zeros(MAX_JOKERS, bool)
    value = state.joker_value
    ids = acting(state)
    slots = jnp.arange(MAX_JOKERS)
    et = eternal(state)                                  # never destroyed

    def slot(i, carry):                                  # the held jokers fill slots 0..7
        sliced, value, burgle, riff, carto, marble, buffer = carry
        t = jnp.clip(src[i], 0, MAX_JOKERS - 1)
        real = ids[i]
        name = jnp.where(src[i] >= 0, state.jokers[t], J_EMPTY)
        acts = (real != J_EMPTY) & (src[i] >= 0) & ~sliced[i] & ~sliced[t]
        mad = acts & (real == J_MADNESS) & ~is_boss
        victims = (state.jokers != J_EMPTY) & (slots != i) & ~sliced & ~et
        v_i = _random_slot(jax.random.fold_in(key, i), victims)
        sliced = sliced | (mad & victims.any() & (slots == v_i))
        value = value.at[i].add(jnp.where(mad, 0.5, 0.0))
        burgle = burgle + (acts & (name == JOKER_ID["Burglar"]))
        room = jnp.maximum(state.joker_slots - (n_held + buffer), 0)
        make = jnp.where(acts & (name == JOKER_ID["Riff-raff"]), jnp.minimum(2, room), 0)
        riff = riff + make
        buffer = buffer + make
        carto = carto + (acts & (name == JOKER_ID["Cartomancer"]))
        right = jnp.minimum(i + 1, MAX_JOKERS - 1)
        dag = acts & (real == J_DAGGER) & (state.jokers[right] != J_EMPTY) & ~sliced[right] & ~et[right]
        sliced = sliced | (dag & (slots == right))
        value = value.at[i].add(jnp.where(dag, 2.0 * state.joker_sell_value[right], 0.0))
        buffer = buffer - dag
        marble = marble + (acts & (name == J_MARBLE))
        return sliced, value, burgle, riff, carto, marble, buffer

    z = jnp.int32(0)
    sliced, value, burgle, riff, carto, marble, _ = jax.lax.fori_loop(
        0, MAX_JOKER_SLOTS, slot, (sliced, value, z, z, z, z, z))
    s = state._replace(joker_value=value)
    s = remove_jokers(s, sliced)
    # Burglar takes every discard first: a Drunkard Riff-raff makes below still adds its +1 (the
    # ease_discard events end at 1 in either slot order)
    s = s._replace(discards=jnp.where(burgle > 0, 0, s.discards))
    kr, kc, km = jax.random.split(jax.random.fold_in(key, 99), 3)

    def riffraff(i, s):                                  # Riff-raff and a copy: up to 4 jokers
        k1, k2 = jax.random.split(jax.random.fold_in(kr, i))
        jid, ed = roll_joker(k1, s, k2, rarity=COMMON)
        return tree_where(i < riff, acquire(s, jid, ed)[0], s)
    s = jax.lax.fori_loop(0, 4, riffraff, s)
    s = spawn_cons(s, CK_TAROT, carto, kc)

    def stone(i, carry):                                 # Marble Joker and copies: up to 3 cards
        s, added = carry
        c = jax.random.randint(jax.random.fold_in(km, i), (), 0, 52)
        s, ok = add_card(s, c % 13, c // 13, ENH_STONE, ED_NONE, SEAL_NONE, ok=i < marble)
        return s, added + ok
    s, added = jax.lax.fori_loop(0, 3, stone, (s, jnp.int32(0)))
    s = _event(s, T_ON_CARD_ADDED, added.astype(jnp.float32))
    return s, burgle


def _start_blind(state):
    """NEXT at the blind select: new_round + Blind:set_blind + the jokers' setting_blind, then
    the first draw (DRAW_TO_HAND: Juggle Tag, Certificate, drawn_to_hand)."""
    k_set, k_acorn, k_deal, k_draw, k_cert, key = jax.random.split(jax.random.fold_in(state.key, K_BLIND), 6)
    is_boss = state.blind_index == BLIND_BOSS
    b = state.boss
    s = state._replace(
        game_stage=jnp.int32(STAGE_BLIND), round=state.round + 1, boss_disabled=jnp.bool_(False),
        boss_hand=jnp.int32(-1), forced_card=jnp.int32(-1), chips_scored=jnp.int32(0),
        blind_target=blind_target(state.ante, state.blind_index, run_scale(state), b, stake=state.stake),
        hands=state.handcount, discards=state.discardcount, discards_used=jnp.int32(0),
        hand_round=jnp.zeros(N_HAND_TYPES, jnp.int32), hand_down=jnp.zeros(MAX_HAND_SIZE, bool),
        card_down=jnp.zeros(MAX_DECK_SIZE, bool), joker_seen=jnp.zeros(MAX_JOKERS, bool),
        joker_debuff=jnp.zeros(MAX_JOKERS, bool), joker_down=jnp.zeros(MAX_JOKERS, bool),
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, jnp.int32), pack_slots=jnp.full(PACK_SLOTS, -1, jnp.int32),
        shop_sticker=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32), key=key)
    # set_blind: The Water (no discards), The Needle, Amber Acorn (the jokers face down). The
    # Needle takes round_resets.hands - 1 hands from the round's max(1, ...): 1 hand is left, but
    # with 0 base hands or less (Troubadours) the take is negative: 0 base hands give 2 hands
    water, needle = boss_is(s, BK_WATER), boss_is(s, BK_NEEDLE)
    r = round_hands(s)
    sub = jnp.select([water, needle], [s.discards, r - 1], 0).astype(jnp.int32)
    s = s._replace(boss_sub=sub, discards=jnp.where(water, 0, s.discards),
                   hands=jnp.where(needle, jnp.maximum(r, 1) - sub, s.hands).astype(jnp.int32),
                   joker_down=boss_is(s, BK_ACORN) & (s.jokers != J_EMPTY))
    s = apply_budget(s)                                  # The Manacle: -1 hand size
    # T_ON_BLIND_SELECT update rules, then setting_blind in slot order
    rows, active, _ = _jokers(s)
    value, jstate = update(rows, active, s.joker_value, s.joker_state, jnp.int32(T_ON_BLIND_SELECT))
    s, burgle = _setting_blind(s._replace(joker_value=value, joker_state=jstate), k_set)
    # new_round: a free reroll per Chaos the Clown that acts (find_joker skips a perished one)
    s = s._replace(hands=s.hands + 3 * burgle, free_rerolls=(acting(s) == J_CHAOS).sum().astype(jnp.int32))
    # Amber Acorn shuffles the jokers (after the ones made or destroyed above)
    held = s.jokers != J_EMPTY
    u = jax.random.uniform(k_acorn, (MAX_JOKERS,))
    perm = jnp.argsort(jnp.where(held, u, 2.0 + jnp.arange(MAX_JOKERS)))
    shuffled = s._replace(jokers=s.jokers[perm], joker_value=s.joker_value[perm], joker_state=s.joker_state[perm],
                          joker_sell_value=s.joker_sell_value[perm], joker_ed=s.joker_ed[perm],
                          joker_debuff=s.joker_debuff[perm], joker_down=s.joker_down[perm],
                          joker_seen=s.joker_seen[perm],
                          joker_sticker=s.joker_sticker[perm], joker_perish=s.joker_perish[perm])
    s = tree_where(boss_is(s, BK_ACORN), shuffled, s)
    # Chicot (not a copy): the boss is off before the first draw
    chicot = (acting(s) == J_CHICOT).any() & is_boss
    s = tree_where(chicot, disable_boss(s, draw_one=False), s)
    # DRAW_TO_HAND: Juggle Tags (+3 hand size this round), the shuffled deck, the draw
    juggle = s.tags == TG_JUGGLE
    tags, hands = _take_tag(s.tags, s.tag_hand, juggle)
    s = apply_budget(s._replace(round_hand_bonus=s.round_hand_bonus + JUGGLE_SIZE * juggle.sum(),
                                tags=tags, tag_hand=hands))
    deck, _, _, n = _fresh_deal(k_deal, s.card_live, 0)
    s = s._replace(deck=deck, deck_size=n, deck_ptr=jnp.int32(0), hand=jnp.full(MAX_HAND_SIZE, -1, jnp.int32),
                   hand_valid=jnp.zeros(MAX_HAND_SIZE, bool))
    k_d, k_c = jax.random.split(k_draw)
    s = draw(s, k_d, first=True, settle=False)
    # first_hand_drawn (Certificate), drawn_to_hand; at hand size 0 the draw ends the run first
    # (draw_from_deck_to_hand: GAME_OVER, _settle)
    return tree_where(s.hand_size > 0, drawn_to_hand(_first_hand_drawn(s, k_cert), k_c, True), s)


_start_blind_c = compacted(_start_blind)


def _first_hand_drawn(s, key):
    """first_hand_drawn: Certificate (copies too) adds a card with a seal at the end of the hand."""
    erows, eok, _, _ = effective(acting(s), s.joker_value, s.joker_state)
    src = s.jokers[jnp.clip(copy_sources(acting(s)), 0, MAX_JOKERS - 1)]
    n_cert = (eok & (erows.TRIGGER == T_ON_BLIND_SELECT) & (erows.EFFECT == E_SPAWN_CARD) & (src == J_CERTIFICATE)).sum()

    def cert(i, carry):                                    # Certificate and copies: up to 3 cards
        s, added = carry
        k1, k2 = jax.random.split(jax.random.fold_in(key, i))
        c = jax.random.randint(k1, (), 0, 52)
        u = jax.random.uniform(k2)
        seal = jnp.select([u > 0.75, u > 0.5, u > 0.25], [SEAL_RED, SEAL_BLUE, SEAL_GOLD], SEAL_PURPLE)
        s, ok = add_card(s, c % 13, c // 13, ENH_NONE, ED_NONE, seal, to_hand=True, ok=i < n_cert)
        return s, added + ok
    s, added = jax.lax.fori_loop(0, 3, cert, (s, jnp.int32(0)))
    return _event(s, T_ON_CARD_ADDED, added.astype(jnp.float32))


def _next_branch(state):
    """_next's branch: 0 leave the shop, 1 play the blind on deck, 2 Cash Out, 3 STAGE_END."""
    st = state.game_stage
    return jnp.select([st == STAGE_SHOP, st == STAGE_CASHOUT, st == STAGE_END], [0, 2, 3], 1)


def _next(state: balatroEnvState, action: balatroAction, pre=None):
    """NEXT: leave the shop (to the blind select), play the blind on deck, or Cash Out. In
    STAGE_END the end of the round runs in _settle, with the joker order the player left.
    pre: env_step's _Pre (its _start_blind result); None = call it here."""
    start = _start_blind if pre is None else (lambda s: pre.started)
    return jax.lax.switch(_next_branch(state), [_leave_shop, start, _cash_out, lambda s: s], state), \
        jnp.int32(0), jnp.bool_(False)


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
        joker_ed=state.joker_ed[perm],
        joker_debuff=state.joker_debuff[perm],
        joker_down=state.joker_down[perm],
        joker_seen=state.joker_seen[perm],
        joker_sticker=state.joker_sticker[perm],
        joker_perish=state.joker_perish[perm],
    )
    return tree_where(ok, moved, state), jnp.int32(0), jnp.bool_(False)


def card_lanes(state):
    """The lanes a MOVE_CARD may use: the hand size, or up to the last card when cards sit past
    it (Certificate, DNA, Familiar ... put them at the end; the game drags every hand card)."""
    lanes = jnp.arange(MAX_HAND_SIZE)
    return jnp.maximum(state.hand_size, jnp.max(jnp.where(state.hand_valid, lanes, -1)) + 1)


def _move_card(state: balatroEnvState, action: balatroAction):
    """Insert the card in lane target at lane dest. Lanes may be holes (an empty
    deck leaves holes); holes shift like cards, but only a card can be moved."""
    i, j = action.target, action.dest
    src = jnp.clip(i, 0, MAX_HAND_SIZE - 1)
    n = card_lanes(state)
    ok = (i >= 0) & (j >= 0) & (i < n) & (j < n) & (i != j) & state.hand_valid[src]
    perm = insert_perm(MAX_HAND_SIZE, src, jnp.clip(j, 0, MAX_HAND_SIZE - 1))
    moved = state._replace(hand=state.hand[perm], hand_valid=state.hand_valid[perm],
                           hand_down=state.hand_down[perm])
    return tree_where(ok, moved, state), jnp.int32(0), jnp.bool_(False)


def _noop(state: balatroEnvState, action: balatroAction):
    return _no_change(state)


# index = verb (env_step sends DISCARD to PLAY's and PICK to USE's); the last entry is V_NOOP
_HANDLERS = [_blind_action, _noop, _buy, _sell, _reroll, _next, _move_joker, _move_card,
             _use_pick, _sell_cons, _open, _noop, _skip, _redeem, _noop]


class _Pre(NamedTuple):
    """What env_step computes before the switch: _start_blind(state) and apply_cons(...)."""
    started: "balatroEnvState"
    used: "balatroEnvState"


# env_step's switch: _next and _use_pick take its _Pre, the other handlers do not
_SWITCH = [(lambda s, a, p, h=h: h(s, a, p)) if h in (_next, _use_pick) else (lambda s, a, p, h=h: h(s, a))
           for h in _HANDLERS]


def _fresh_deal(key, card_live, hand_size):
    """Shuffle the live cards and deal the opening hand. -> (deck, hand, hand_valid, deck_size)"""
    perm = jax.random.permutation(key, MAX_DECK_SIZE)
    deck = compact(perm, card_live[perm])           # live cards to the front, free slots to the tail
    n = card_live.sum().astype(jnp.int32)
    hand_valid = jnp.arange(MAX_HAND_SIZE) < jnp.minimum(hand_size, n)
    hand = jnp.where(hand_valid, deck[:MAX_HAND_SIZE], -1)
    return deck, hand, hand_valid, n


def standard_cards(deck_size=52):
    """The card_* arrays of a fresh deck: instance i < deck_size is card i % 52."""
    i = jnp.arange(MAX_DECK_SIZE)
    z = jnp.zeros(MAX_DECK_SIZE, jnp.int32)
    return dict(card_rank=(i % 52) % 13, card_suit=(i % 52) // 13, card_enh=z, card_ed=z,
                card_seal=z, card_bonus=jnp.zeros(MAX_DECK_SIZE, jnp.float32),
                card_live=i < deck_size)


_NO_FACES = np.array([c for c in range(52) if c % 13 not in (9, 10, 11)], np.int32)   # 40 cards, J Q K out


def back_cards(key, back, deck_size=52):
    """game.lua start_run's card_protos for deck back (instance i < the deck's size):
    Abandoned: the 40 cards without J / Q / K; Checkered: Clubs -> Spades, Diamonds -> Hearts;
    Erratic: each card a random one of the 52 (pseudorandom_element(G.P_CARDS, 'erratic'))."""
    cards = standard_cards(deck_size)
    i = jnp.arange(MAX_DECK_SIZE)
    aband = back == B_ABANDONED
    cid = jnp.where(aband, jnp.asarray(_NO_FACES)[i % 40], i % 52)
    cid = jnp.where(back == B_ERRATIC, jax.random.randint(key, (MAX_DECK_SIZE,), 0, 52), cid)
    suit = cid // 13
    suit = jnp.where(back == B_CHECKERED, jnp.where(suit == 0, 3, jnp.where(suit == 1, 2, suit)), suit)
    n = jnp.where(aband, jnp.minimum(deck_size, 40), deck_size)
    return dict(cards, card_rank=(cid % 13).astype(jnp.int32), card_suit=suit.astype(jnp.int32),
                card_live=i < n)


def run_fields():
    """The fields of a fresh run that every builder shares (reset, fixtures): no boss state,
    no tags, no vouchers, White Stake, no stickers. -> dict"""
    return dict(
        boss=jnp.int32(-1), bosses_used=jnp.zeros(N_BOSSES, jnp.int32), boss_rerolled=jnp.bool_(False),
        boss_disabled=jnp.bool_(False), boss_hand=jnp.int32(-1), ox_hand=jnp.int32(N_HAND_TYPES - 1),
        boss_sub=jnp.int32(0), forced_card=jnp.int32(-1),
        card_played_ante=jnp.zeros(MAX_DECK_SIZE, bool), hand_down=jnp.zeros(MAX_HAND_SIZE, bool),
        card_down=jnp.zeros(MAX_DECK_SIZE, bool),
        joker_debuff=jnp.zeros(MAX_JOKERS, bool), joker_down=jnp.zeros(MAX_JOKERS, bool),
        joker_seen=jnp.zeros(MAX_JOKERS, bool),
        blind_tags=jnp.full(2, -1, jnp.int32), orbital_hands=jnp.full(2, N_HAND_TYPES - 1, jnp.int32),
        tags=jnp.full(MAX_TAGS, -1, jnp.int32), tag_hand=jnp.zeros(MAX_TAGS, jnp.int32),
        skips=jnp.int32(0), unused_discards=jnp.int32(0), pack_return=jnp.int32(STAGE_SHOP),
        round_hand_bonus=jnp.int32(0),
        vouchers=jnp.zeros(N_VOUCHERS, bool), ante_voucher=jnp.int32(-1),
        shop_vouchers=jnp.full(MAX_SHOP_VOUCHERS, -1, jnp.int32),
        shop_free=jnp.zeros(MAX_SHOP_SLOTS, bool), pack_free=jnp.zeros(PACK_SLOTS, bool),
        shop_rank=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32), shop_suit=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32),
        shop_enh=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32), cons_ed=jnp.zeros(MAX_CONSUMABLES, jnp.int32),
        ecto_minus=jnp.int32(1), reroll_inc=jnp.int32(0), reroll_d6=jnp.bool_(False),
        stake=jnp.int32(1), back=jnp.int32(B_RED), joker_sticker=jnp.zeros(MAX_JOKERS, jnp.int32),
        joker_perish=jnp.zeros(MAX_JOKERS, jnp.int32), shop_sticker=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32),
        pack_sticker=jnp.zeros(MAX_PACK, jnp.int32),
    )


@jit
def reset(key, deckSize, handSize, target_scale=1.0, stake=1, back=B_RED):
    """A new run at the blind select of the ante-1 Small Blind: the boss, the two tags and
    the first voucher are rolled; no hand yet. stake: 1 White .. 8 Gold (the Blue Stake
    and up: -1 discard). back: the deck (items.BACK_NAMES; back.lua apply_to_run): its hands,
    discards, $, joker / consumable slots, hand size, cards (back_cards), vouchers (redeemed:
    Magic Crystal Ball, Nebula Telescope, Zodiac Tarot Merchant + Planet Merchant + Overstock)
    and consumables (Magic 2 The Fool, Ghost Hex)."""
    key, subkey, tkey, akey, vkey = jax.random.split(key, 5)
    stake = jnp.clip(jnp.asarray(stake, jnp.int32), 1, N_STAKES)
    back = jnp.clip(jnp.asarray(back, jnp.int32), 0, N_BACKS - 1)
    n_disc = (BASE_DISCARDS + BACK_DISCARDS[back] - (stake >= ST_DISCARD)).astype(jnp.int32)
    n_hands = (BASE_HANDS + BACK_HANDS[back]).astype(jnp.int32)
    h_size = (handSize + BACK_HAND_SIZE[back]).astype(jnp.int32)
    cards = back_cards(jax.random.fold_in(subkey, 1), back, deckSize)
    deck, _, _, n = _fresh_deal(subkey, cards["card_live"], 0)
    vouchers = jnp.zeros(N_VOUCHERS, bool)
    vouchers = vouchers.at[V_CRYSTAL_BALL].set(back == B_MAGIC).at[V_TELESCOPE].set(back == B_NEBULA)
    zod = back == B_ZODIAC
    vouchers = vouchers.at[V_TAROT_MERCHANT].set(zod).at[V_PLANET_MERCHANT].set(zod).at[V_OVERSTOCK].set(zod)

    s = balatroEnvState(
        jokers=jnp.full(MAX_JOKERS, J_EMPTY, dtype=jnp.int32),
        joker_value=jnp.full(MAX_JOKERS, 0, dtype=float),
        joker_state=jnp.full(MAX_JOKERS, 0, dtype=jnp.int32),
        joker_sell_value=jnp.full(MAX_JOKERS, 0, dtype=jnp.int32),
        joker_slots=(BASE_JOKER_SLOTS + BACK_JOKER_SLOT[back]).astype(jnp.int32),
        joker_ed=jnp.zeros(MAX_JOKERS, jnp.int32),
        consumables=jnp.full(MAX_CONSUMABLES, CONS_EMPTY, dtype=jnp.int32),
        cons_sell=jnp.zeros(MAX_CONSUMABLES, jnp.int32),
        consumable_slots=(BASE_CONS_SLOTS + BACK_CONS_SLOT[back] + (back == B_MAGIC)).astype(jnp.int32),
        **cards,
        hand=jnp.full(MAX_HAND_SIZE, -1, jnp.int32),
        hand_size=h_size,
        hand_valid=jnp.zeros(MAX_HAND_SIZE, bool),
        deck=deck,
        deck_size=n,
        deck_ptr=jnp.int32(0),
        deck_start=n,
        hands=n_hands,
        discards=n_disc,
        handcount=n_hands,
        discardcount=n_disc,
        base_hand_size=h_size,
        base_hands=n_hands,
        base_discards=n_disc,
        discards_used=jnp.int32(0),
        round=jnp.int32(0),
        ante=jnp.int32(1),
        blind_index=jnp.int32(BLIND_SMALL),
        chips_scored=jnp.int32(0),
        blind_target=blind_target(jnp.int32(1), BLIND_SMALL, target_scale * BACK_ANTE_SCALING[back], stake=stake),
        target_scale=jnp.float32(target_scale),
        key=key,
        hand_played=jnp.zeros(N_HAND_TYPES, jnp.int32),
        hand_round=jnp.zeros(N_HAND_TYPES, jnp.int32),
        hand_level=jnp.ones(N_HAND_TYPES, jnp.int32),
        last_hand=jnp.int32(-1),
        idol_card=jnp.int32(51), mail_rank=jnp.int32(12),       # the game's defaults (A of Spades)
        ancient_suit=jnp.int32(-1), castle_suit=jnp.int32(3),
        last_cons=jnp.int32(-1),
        tarots_used=jnp.int32(0),
        planets_used=jnp.zeros(N_HAND_TYPES, bool),
        shop_slots=jnp.full(MAX_SHOP_SLOTS, -1, dtype=jnp.int32),
        shop_kind=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32),
        shop_ed=jnp.zeros(MAX_SHOP_SLOTS, jnp.int32),
        shop_size=(SHOP_UNLOCKED + zod).astype(jnp.int32),
        pack_slots=jnp.full(PACK_SLOTS, -1, jnp.int32),
        reroll_cost=jnp.int32(REROLL_BASE),
        free_rerolls=jnp.int32(0),
        gros_extinct=jnp.bool_(False),
        buffoon_given=jnp.bool_(False),
        game_stage=jnp.int32(STAGE_SELECT),
        money=(STARTING_MONEY + BACK_DOLLARS[back]).astype(jnp.int32),
        last_cashout=jnp.int32(0),
        pack_id=jnp.int32(-1),
        pack_left=jnp.int32(0),
        pack_items=jnp.full(MAX_PACK, -1, jnp.int32),
        pack_ed=jnp.zeros(MAX_PACK, jnp.int32),
        pack_rank=jnp.zeros(MAX_PACK, jnp.int32),
        pack_suit=jnp.zeros(MAX_PACK, jnp.int32),
        pack_enh=jnp.zeros(MAX_PACK, jnp.int32),
        pack_seal=jnp.zeros(MAX_PACK, jnp.int32),
        **run_fields(),
    )._replace(stake=stake, back=back, vouchers=vouchers)
    # Magic Deck: 2 The Fool; Ghost Deck: Hex (create_card('Tarot', ..., forced key): no edition)
    s = give_cons(s, jnp.where(back == B_GHOST, CONS_ID["Hex"], C_FOOL), (back == B_MAGIC) | (back == B_GHOST))
    s = give_cons(s, C_FOOL, back == B_MAGIC)
    tg = new_targets(tkey, s)
    s = s._replace(idol_card=tg.idol_card, mail_rank=tg.mail_rank,
                   ancient_suit=tg.ancient_suit, castle_suit=tg.castle_suit)
    s = new_ante_rolls(akey, s)
    return s._replace(ante_voucher=roll_voucher(vkey, s, jnp.zeros(N_VOUCHERS, bool)))

RANK_CHARS = "23456789TJQKA"   # rank = id % 13, so index 0 is the Two, 12 the Ace
SUIT_CHARS = "CDHS"            # display only — scoring never names suits, ids are the truth

def _card_name(cid):
    """'KH'-style name for a card id (rank + 13 * suit); raw number for anything else."""
    if 0 <= cid < 52:
        return RANK_CHARS[cid % 13] + SUIT_CHARS[cid // 13]
    return str(cid)


def inst_name(state, inst):
    """'KH'-style name of a card instance (plain Python, concrete state)."""
    inst = int(inst)
    if not 0 <= inst < MAX_DECK_SIZE:
        return str(inst)
    return RANK_CHARS[int(state.card_rank[inst])] + SUIT_CHARS[int(state.card_suit[inst])]


def show(state: balatroEnvState, label=""):
    """Pretty-print the FULL state, one section per concern, plus invariant
    checks so a corrupted state announces itself. Plain Python — call it on
    concrete states only (outside jit, or on the result of a jitted
    function), never inside one."""
    from jokers import JOKER_NAMES
    from items import (CONS_NAMES, ENH_NAMES, ED_NAMES, SEAL_NAMES, BOSS_NAMES, TAG_NAMES,
                       VOUCHER_NAMES, STAKE_NAMES)
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
    row("inst", hand[:n])
    row("card", [inst_name(state, c) if v else "." for c, v in zip(hand[:n], valid[:n])])
    row("valid", [int(v) for v in valid[:n]])
    if any(bool(x) for x in state.hand_down):
        row("down", [int(bool(x)) for x in state.hand_down[:n]])
    mods = [f"{inst_name(state, c)}:{ENH_NAMES[int(state.card_enh[c])]}/{ED_NAMES[int(state.card_ed[c])]}"
            f"/{SEAL_NAMES[int(state.card_seal[c])]}" for c, v in zip(hand[:n], valid[:n])
            if v and (int(state.card_enh[c]) or int(state.card_ed[c]) or int(state.card_seal[c]))]
    if mods:
        print("  mods  " + "  ".join(mods))

    # --- deck: pointer, undealt window, dead zone -------------------------
    print(f"deck   ptr {ptr}/{size} ({size - ptr} undealt, "
          f"{MAX_DECK_SIZE - size} free slots)")
    nxt = deck[ptr:min(ptr + 8, size)]
    row("next", nxt)
    row("card", [inst_name(state, c) for c in nxt])

    # --- round counters + blind progress ----------------------------------
    stage = int(state.game_stage)
    bi = int(state.blind_index)
    boss = BOSS_NAMES[int(state.boss)] + (" (off)" if bool(state.boss_disabled) else "")
    print(f"round  {STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else stage}   "
          f"ante {int(state.ante)} {BLIND_NAMES[bi] if 0 <= bi < 3 else bi}   boss {boss}   round {int(state.round)}"
          f"   hands {int(state.hands)}   discards {int(state.discards)}"
          f"   money ${int(state.money)}   {STAKE_NAMES[min(max(int(state.stake), 1), 8) - 1]}"
          f"   {BACK_NAMES[min(max(int(state.back), 0), N_BACKS - 1)]}")
    chips, target = int(state.chips_scored), int(state.blind_target)
    frac = 0.0 if target <= 0 else min(chips / target, 1.0)
    filled = int(round(frac * 20))
    print(f"blind  {chips}/{target} chips  [{'#' * filled}{'.' * (20 - filled)}]"
          f"  {int(frac * 100)}%")
    lv = [f"{HAND_NAMES[h]} {int(state.hand_level[h])}" for h in range(N_HAND_TYPES) if int(state.hand_level[h]) > 1]
    if lv:
        print("levels " + ", ".join(lv))
    tags = [TAG_NAMES[int(t)] for t in state.tags if int(t) >= 0]
    offer = [TAG_NAMES[int(t)] if int(t) >= 0 else "." for t in state.blind_tags]
    print(f"tags   held {tags}   offered {offer}   skips {int(state.skips)}")
    vs = [VOUCHER_NAMES[i] for i in range(len(VOUCHER_NAMES)) if bool(state.vouchers[i])]
    if vs:
        print(f"vouch  {vs}")

    # --- jokers + consumables + shop ---------------------------------------
    def stickers(st, left=PERISH_ROUNDS):
        st = int(st)
        return "".join(t for b, t in ((STK_ETERNAL, "[E]"), (STK_PERISH, f"[P{int(left)}]"), (STK_RENTAL, "[R]"))
                       if st & b)

    held = [(JOKER_NAMES[int(j)], float(v), int(s), int(e), stickers(st, pl))
            for j, v, s, e, st, pl in zip(state.jokers, state.joker_value, state.joker_sell_value, state.joker_ed,
                                          state.joker_sticker, state.joker_perish) if int(j) >= 0]
    print("jokers " + ("  ".join(f"{nm}{'(' + ED_NAMES[e] + ')' if e else ''}{st} v={v:g} sell=${s}"
                                 for nm, v, s, e, st in held) or "(none)")
          + f"   ({len(held)}/{int(state.joker_slots)})")
    cons = [CONS_NAMES[int(c)] + ("(Negative)" if int(e) == ED_NEG else "")
            for c, e in zip(state.consumables, state.cons_ed) if int(c) >= 0]
    print(f"cons   {cons}   ({len(cons)}/{int(state.consumable_slots)})")
    shop = []
    for i in range(int(state.shop_size)):
        it, k = int(state.shop_slots[i]), int(state.shop_kind[i])
        if it < 0:
            shop.append(".")
        elif k == IK_JOKER:
            shop.append(JOKER_NAMES[it] + stickers(state.shop_sticker[i]))
        elif k == IK_CARD:
            shop.append(RANK_CHARS[int(state.shop_rank[i])] + SUIT_CHARS[int(state.shop_suit[i])])
        else:
            shop.append(CONS_NAMES[it])
    from items import PACK_NAMES
    packs = [PACK_NAMES[int(p)] if int(p) >= 0 else "." for p in state.pack_slots]
    vouch = [VOUCHER_NAMES[int(v)] for v in state.shop_vouchers if int(v) >= 0]
    print(f"shop   {shop}  vouchers {vouch}  packs {packs}   reroll ${int(state.reroll_cost)}")

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
    live = [bool(x) for x in state.card_live]
    jokers = [int(j) for j in state.jokers]
    debuff = [bool(x) for x in state.joker_debuff]
    held = [j for j in jokers if j >= 0]
    stake = int(state.stake)
    stk = [int(x) for x in state.joker_sticker]
    perish = [int(x) for x in state.joker_perish]
    dead = [jokers[i] >= 0 and bool(stk[i] & STK_PERISH) and perish[i] <= 0 for i in range(MAX_JOKERS)]

    def sticker_ok(j, st):
        """Stickers a joker j may have at this stake (never Eternal and Perishable)."""
        e, p, r = bool(st & STK_ETERNAL), bool(st & STK_PERISH), bool(st & STK_RENTAL)
        return (0 <= st < 8 and not (e and p) and (not e or (stake >= ST_ETERNAL and bool(JOKER_ETERNAL_OK[j])))
                and (not p or (stake >= ST_PERISH and bool(JOKER_PERISH_OK[j]))) and (not r or stake >= ST_RENTAL))
    cons = [int(c) for c in state.consumables]
    cons_neg = sum(1 for c, e in zip(cons, state.cons_ed) if c >= 0 and int(e) == ED_NEG)
    dealt = set(deck[:ptr])
    neg = sum(1 for i, j in enumerate(jokers) if j >= 0 and int(state.joker_ed[i]) == ED_NEG)
    antimatter = int(bool(state.vouchers[V_ANTIMATTER]))
    stage = int(state.game_stage)
    manacle = (stage in (STAGE_BLIND, STAGE_END, STAGE_LOST) and int(state.blind_index) == BLIND_BOSS
               and not bool(state.boss_disabled) and int(state.boss) == BK_MANACLE)   # a lost run keeps it
    tags = [int(t) for t in state.tags]
    checks = [
        ("deck is a permutation of 0..127", sorted(deck) == list(range(MAX_DECK_SIZE))),
        ("deck[:deck_size] are the live cards", sorted(deck[:size]) == [i for i in range(MAX_DECK_SIZE) if live[i]]),
        ("deck_ptr within the live deck", 0 <= ptr <= size),
        ("valid hand cards were all dealt (in deck[:ptr])",
         all(hand[i] in dealt for i in range(MAX_HAND_SIZE) if valid[i])),
        ("no valid lane at or past max(hand_size, cards held)",
         not any(valid[max(n, sum(valid)):])),
        ("no duplicate valid card in hand",
         len([hand[i] for i in range(MAX_HAND_SIZE) if valid[i]]) == len({hand[i] for i in range(MAX_HAND_SIZE) if valid[i]})),
        ("a face-down lane holds a card", all(valid[i] for i in range(MAX_HAND_SIZE) if bool(state.hand_down[i]))),
        ("a face-down lane's card is hidden in the deck view (card_down)",
         all(bool(state.card_down[hand[i]]) for i in range(MAX_HAND_SIZE) if valid[i] and bool(state.hand_down[i]))),
        ("a face-up lane's card is not card_down",
         not any(bool(state.card_down[hand[i]]) for i in range(MAX_HAND_SIZE) if valid[i] and not bool(state.hand_down[i]))),
        ("joker_seen only on a face-down joker",
         not any(bool(state.joker_seen[i]) and not bool(state.joker_down[i]) for i in range(MAX_JOKERS))),
        ("hands / discards not negative", int(state.hands) >= 0 and int(state.discards) >= 0),
        ("money not far below the Credit Card limit (The Tooth can take more)",
         int(state.money) >= -CREDIT_DEBT * sum(j == J_CREDIT for j in jokers) - 200),
        ("jokers compact (no hole before a held joker)", jokers[:len(held)] == held),
        # a charged Negative Invisible Joker sold at full slots copies first, then its slot goes:
        # one joker over (card.lua 2381 checks #jokers <= limit before the sale removes it)
        ("held jokers within joker_slots + 1", len(held) <= int(state.joker_slots) + 1),
        ("joker_slots = 5 (+1 Black, -1 Painted) + Antimatter + Negatives (capped)",
         int(state.joker_slots) == min(int(base_joker_slots(state)) + antimatter + neg, MAX_JOKER_SLOTS)),
        ("back 0 .. 14", 0 <= int(state.back) < N_BACKS),
        ("vacant joker slots hold value 0, sell 0, no edition, no flags, no stickers",
         all(float(state.joker_value[i]) == 0.0 and int(state.joker_sell_value[i]) == 0
             and int(state.joker_ed[i]) == 0 and not debuff[i] and not bool(state.joker_down[i])
             and stk[i] == 0 and perish[i] == 0 for i in range(MAX_JOKERS) if jokers[i] < 0)),
        ("stake 1 .. 8", 1 <= stake <= N_STAKES),
        ("held jokers' stickers fit the joker and the stake", all(sticker_ok(j, stk[i]) for i, j in enumerate(jokers)
                                                                 if j >= 0)),
        ("a Perishable joker has 0 .. 5 rounds left, any other 0",
         all((0 <= perish[i] <= PERISH_ROUNDS) if stk[i] & STK_PERISH else perish[i] == 0
             for i in range(MAX_JOKERS) if jokers[i] >= 0)),
        ("shop stickers only on shop jokers, and they fit",
         all(int(st) == 0 or (int(state.shop_kind[i]) == IK_JOKER and int(state.shop_slots[i]) >= 0
                              and sticker_ok(int(state.shop_slots[i]), int(st)))
             for i, st in enumerate(state.shop_sticker))),
        ("pack stickers only on the jokers of an open Buffoon pack, and they fit",
         all(int(st) == 0 or (stage == STAGE_PACK and int(PACK_KIND[jnp.clip(state.pack_id, 0, N_PACKS - 1)]) == PK_BUFFOON
                              and int(state.pack_items[i]) >= 0 and sticker_ok(int(state.pack_items[i]), int(st)))
             for i, st in enumerate(state.pack_sticker))),
        # + 1: a Negative Emperor / High Priestess / Fool fills its own slot before it dissolves
        ("consumables compact and within the slots (+ Negatives, + 1)",
         cons[:len([c for c in cons if c >= 0])] == [c for c in cons if c >= 0]
         and len([c for c in cons if c >= 0]) <= int(state.consumable_slots) + cons_neg + 1),
        ("a held consumable sells for $1 or more, a vacant slot for 0",
         all((int(state.cons_sell[i]) >= 1) if cons[i] >= 0 else int(state.cons_sell[i]) == 0
             for i in range(MAX_CONSUMABLES))),
        ("locked shop lanes are empty",
         all(int(s) == -1 for s in state.shop_slots[int(state.shop_size):])),
        ("game_stage is a STAGE_*", 0 <= stage < len(STAGE_NAMES)),
        ("a pack is open exactly in STAGE_PACK", (int(state.pack_id) >= 0) == (stage == STAGE_PACK)),
        ("hand size = base + acting jokers (not a perished one) + Juggle - Manacle", int(state.hand_size) == min(max(
            int(state.base_hand_size) + int(state.round_hand_bonus) - manacle + sum(
                _H_SIZE[j] + (int(float(state.joker_value[i])) if _HSIZE_V[j] else 0)
                for i, j in enumerate(jokers) if j >= 0 and not debuff[i] and not dead[i]), 0), MAX_HAND_SIZE)),
        ("round targets in range", 0 <= int(state.idol_card) < 52 and 0 <= int(state.mail_rank) < 13
         and 0 <= int(state.ancient_suit) < 4 and 0 <= int(state.castle_suit) < 4),
        ("hand levels >= 1", all(int(x) >= 1 for x in state.hand_level)),
        ("tags compact", tags[:len([t for t in tags if t >= 0])] == [t for t in tags if t >= 0]),
        ("blind_index is Small / Big / Boss", 0 <= int(state.blind_index) <= 2),
        ("only a boss round has boss state (a lost run keeps it)",
         stage in (STAGE_BLIND, STAGE_END, STAGE_LOST)
         or (not any(debuff) and int(state.forced_card) < 0 and not any(bool(x) for x in state.hand_down))),
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
    s = reset(jax.random.key(0), 52, 8)
    show(s, "reset")
    s, reward, done = env_step(s, balatroAction(verb=jnp.int32(V_NEXT), select=jnp.zeros(MAX_HAND_SIZE, bool)))
    show(s, "after NEXT (the Small Blind)")
    nxt, reward, done = env_step(s, balatroAction(verb=jnp.int32(0), select=jnp.arange(MAX_HAND_SIZE) < 5))
    show(nxt, "after one PLAY")
