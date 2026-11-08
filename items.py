"""Card modifiers, hand levels, consumables, booster packs, vouchers, tags and
boss blinds as DATA.

No project imports (same rule as jokers.py): env.py and scoring.py read these
tables, nothing here reads them.

The rules come from the Balatro 1.0.1o source (game.lua P_CENTERS, P_TAGS,
P_BLINDS and G.GAME.hands, card.lua use_consumeable / open / apply_to_run,
tag.lua apply_to_run, blind.lua, common_events.lua create_card / poll_edition /
get_pack / get_new_boss, UI_definitions.lua create_card_for_shop).

PLAYING CARD MODIFIERS — one int per card instance (EnvState.card_enh / card_ed
/ card_seal), 0 = none:
    enhancement  Bonus +30 chips, Mult +4 mult, Wild any suit, Glass X2 mult and
                 1 in 4 to break after scoring, Steel X1.5 mult while held, Stone
                 +50 chips / no rank or suit / always scores, Gold $3 while held at
                 the end of the round, Lucky 1 in 5 +20 mult and 1 in 15 $20
    edition      Foil +50 chips, Holographic +10 mult, Polychrome X1.5 mult
                 (jokers: the same, plus Negative = +1 joker slot)
    seal         Gold $3 when scored, Red retrigger, Blue a Planet for the last
                 hand at the end of the round, Purple a Tarot when discarded

HAND TYPES — Balatro's order, best first; argmax over "contains" picks the hand.
The three secret hands need duplicate or Wild cards.
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

# --- enhancements --------------------------------------------------------------
ENH_NONE, ENH_BONUS, ENH_MULT, ENH_WILD, ENH_GLASS, ENH_STEEL, ENH_STONE, ENH_GOLD, ENH_LUCKY = range(9)
N_ENH = 9
ENH_NAMES = ["", "Bonus", "Mult", "Wild", "Glass", "Steel", "Stone", "Gold", "Lucky"]
BONUS_CHIPS, MULT_MULT, GLASS_X, GLASS_ODDS, STEEL_X = 30, 4, 2.0, 4, 1.5
STONE_CHIPS, GOLD_DOLLARS, LUCKY_MULT, LUCKY_MULT_ODDS, LUCKY_DOLLARS, LUCKY_DOLLAR_ODDS = 50, 3, 20, 5, 20, 15

# --- editions (cards and jokers) --------------------------------------------------
ED_NONE, ED_FOIL, ED_HOLO, ED_POLY, ED_NEG = range(5)
N_ED = 5
ED_NAMES = ["", "Foil", "Holographic", "Polychrome", "Negative"]
FOIL_CHIPS, HOLO_MULT, POLY_X = 50, 10, 1.5
ED_COST = jnp.array([0, 2, 3, 5, 5], jnp.int32)       # set_cost: extra $ on a card with this edition
MAX_JOKER_SLOTS = 16  # divergence: Negative jokers add a slot only up to 16 in all (agent J_TOK)

# --- seals -----------------------------------------------------------------------------
SEAL_NONE, SEAL_GOLD, SEAL_RED, SEAL_BLUE, SEAL_PURPLE = range(5)
N_SEAL = 5
SEAL_NAMES = ["", "Gold", "Red", "Blue", "Purple"]
GOLD_SEAL_DOLLARS = 3

# --- hand types and levels (G.GAME.hands) -----------------------------------------------
HAND_NAMES = ["flush_five", "flush_house", "five_kind", "straight_flush", "four_kind",
              "full_house", "flush", "straight", "three_kind", "two_pair", "pair", "high_card"]
N_HAND_TYPES = len(HAND_NAMES)
#                      FF   FH   5K   SF   4K   FullH Flush Strt 3K   2P   Pair High
HAND_CHIPS = jnp.array([160, 140, 120, 100,  60,  40,   35,   30,  30,  20,  10,  5], jnp.int32)
HAND_MULT  = jnp.array([ 16,  14,  12,   8,   7,   4,    4,    4,   3,   2,   2,  1], jnp.int32)
LEVEL_CHIPS = jnp.array([50,  40,  35,  40,  30,  25,   15,   30,  20,  20,  15, 10], jnp.int32)
LEVEL_MULT  = jnp.array([ 3,   4,   3,   4,   3,   2,    2,    3,   2,   1,   1,  1], jnp.int32)
SECRET_HANDS = 3          # the first three: hidden until played (To Do List skips them, planets soft-locked)
# The Ox on a tie for the most played hand (state_events.lua end_round): the loop over
# pairs(G.GAME.hands) never updates _order, so the LAST tied hand in pairs() order wins. That
# order is LuaJIT 2.0.5's hash order of the game.lua constructor (read from the game's lua51.dll);
# a saved and continued run rebuilds the table in another order (divergence). Position per hand:
OX_PAIRS_POS = jnp.array([8, 0, 9, 5, 11, 1, 2, 6, 10, 7, 3, 4], jnp.int32)


def hand_chips_mult(level):
    """(chips, mult) per hand type at these levels (level_up_hand)."""
    chips = jnp.maximum(HAND_CHIPS + LEVEL_CHIPS * (level - 1), 0)
    mult = jnp.maximum(HAND_MULT + LEVEL_MULT * (level - 1), 1)
    return chips, mult


# --- consumables -----------------------------------------------------------------------------
CK_TAROT, CK_PLANET, CK_SPECTRAL = 0, 1, 2

# what USE does (the Tarot's "effect" + config in P_CENTERS)
U_ENHANCE, U_SUIT, U_STRENGTH, U_HANGED, U_DEATH, U_FOOL, U_HERMIT, U_TEMPERANCE = range(8)
U_SPAWN_TAROT, U_SPAWN_PLANET, U_WHEEL, U_JUDGEMENT, U_PLANET = range(8, 13)
# Spectral cards
U_CREATE_CARDS, U_SEAL, U_AURA, U_WRAITH, U_SIGIL, U_OUIJA, U_ECTOPLASM, U_IMMOLATE = range(13, 21)
U_ANKH, U_HEX, U_CRYPTID, U_SOUL, U_BLACK_HOLE = range(21, 26)


class ConsDef(NamedTuple):
    name  : str
    kind  : int
    use   : int
    param : int = 0      # enhancement / suit / hand type / count / seal
    lo    : int = 0      # cards to select (min_highlighted .. max_highlighted)
    hi    : int = 0
    desc  : str = ""
    need  : int = 0      # cards the hand must hold, none selected (Familiar ... Ouija: more than 1)
    ranks : int = 0      # U_CREATE_CARDS: the 13-bit rank mask of the new cards


CONS_TABLE = [
    ConsDef("The Fool",            CK_TAROT, U_FOOL, desc="Creates the last Tarot or Planet card used this run (The Fool excluded)"),
    ConsDef("The Magician",        CK_TAROT, U_ENHANCE, ENH_LUCKY, 1, 2, "Enhances 2 selected cards to Lucky Cards"),
    ConsDef("The High Priestess",  CK_TAROT, U_SPAWN_PLANET, 2, desc="Creates up to 2 random Planet cards (must have room)"),
    ConsDef("The Empress",         CK_TAROT, U_ENHANCE, ENH_MULT, 1, 2, "Enhances 2 selected cards to Mult Cards"),
    ConsDef("The Emperor",         CK_TAROT, U_SPAWN_TAROT, 2, desc="Creates up to 2 random Tarot cards (must have room)"),
    ConsDef("The Hierophant",      CK_TAROT, U_ENHANCE, ENH_BONUS, 1, 2, "Enhances 2 selected cards to Bonus Cards"),
    ConsDef("The Lovers",          CK_TAROT, U_ENHANCE, ENH_WILD, 1, 1, "Enhances 1 selected card into a Wild Card"),
    ConsDef("The Chariot",         CK_TAROT, U_ENHANCE, ENH_STEEL, 1, 1, "Enhances 1 selected card into a Steel Card"),
    ConsDef("Justice",             CK_TAROT, U_ENHANCE, ENH_GLASS, 1, 1, "Enhances 1 selected card into a Glass Card"),
    ConsDef("The Hermit",          CK_TAROT, U_HERMIT, 20, desc="Doubles money (Max of $20)"),
    ConsDef("The Wheel of Fortune", CK_TAROT, U_WHEEL, 4, desc="1 in 4 chance to add Foil, Holographic, or Polychrome edition to a random Joker"),
    ConsDef("Strength",            CK_TAROT, U_STRENGTH, 0, 1, 2, "Increases rank of up to 2 selected cards by 1"),
    ConsDef("The Hanged Man",      CK_TAROT, U_HANGED, 0, 1, 2, "Destroys up to 2 selected cards"),
    ConsDef("Death",               CK_TAROT, U_DEATH, 0, 2, 2, "Select 2 cards, convert the left card into the right card"),
    ConsDef("Temperance",          CK_TAROT, U_TEMPERANCE, 50, desc="Gives the total sell value of all current Jokers (Max of $50)"),
    ConsDef("The Devil",           CK_TAROT, U_ENHANCE, ENH_GOLD, 1, 1, "Enhances 1 selected card into a Gold Card"),
    ConsDef("The Tower",           CK_TAROT, U_ENHANCE, ENH_STONE, 1, 1, "Enhances 1 selected card into a Stone Card"),
    ConsDef("The Star",            CK_TAROT, U_SUIT, 1, 1, 3, "Converts up to 3 selected cards to Diamonds"),
    ConsDef("The Moon",            CK_TAROT, U_SUIT, 0, 1, 3, "Converts up to 3 selected cards to Clubs"),
    ConsDef("The Sun",             CK_TAROT, U_SUIT, 2, 1, 3, "Converts up to 3 selected cards to Hearts"),
    ConsDef("Judgement",           CK_TAROT, U_JUDGEMENT, desc="Creates a random Joker card (must have room)"),
    ConsDef("The World",           CK_TAROT, U_SUIT, 3, 1, 3, "Converts up to 3 selected cards to Spades"),
    # planets: param = hand type (HAND_NAMES index)
    ConsDef("Mercury",   CK_PLANET, U_PLANET, 10, desc="Level up Pair"),
    ConsDef("Venus",     CK_PLANET, U_PLANET, 8,  desc="Level up Three of a Kind"),
    ConsDef("Earth",     CK_PLANET, U_PLANET, 5,  desc="Level up Full House"),
    ConsDef("Mars",      CK_PLANET, U_PLANET, 4,  desc="Level up Four of a Kind"),
    ConsDef("Jupiter",   CK_PLANET, U_PLANET, 6,  desc="Level up Flush"),
    ConsDef("Saturn",    CK_PLANET, U_PLANET, 7,  desc="Level up Straight"),
    ConsDef("Uranus",    CK_PLANET, U_PLANET, 9,  desc="Level up Two Pair"),
    ConsDef("Neptune",   CK_PLANET, U_PLANET, 3,  desc="Level up Straight Flush"),
    ConsDef("Pluto",     CK_PLANET, U_PLANET, 11, desc="Level up High Card"),
    ConsDef("Planet X",  CK_PLANET, U_PLANET, 2,  desc="Level up Five of a Kind"),
    ConsDef("Ceres",     CK_PLANET, U_PLANET, 1,  desc="Level up Flush House"),
    ConsDef("Eris",      CK_PLANET, U_PLANET, 0,  desc="Level up Flush Five"),
    # spectral cards (cost $4): the Spectral packs, Seance, Sixth Sense, Omen Globe Arcana packs
    ConsDef("Familiar",    CK_SPECTRAL, U_CREATE_CARDS, 3, need=2, ranks=0b0111000000000,
            desc="Destroy 1 random card in your hand, add 3 random Enhanced face cards to your hand"),
    ConsDef("Grim",        CK_SPECTRAL, U_CREATE_CARDS, 2, need=2, ranks=0b1000000000000,
            desc="Destroy 1 random card in your hand, add 2 random Enhanced Aces to your hand"),
    ConsDef("Incantation", CK_SPECTRAL, U_CREATE_CARDS, 4, need=2, ranks=0b0000111111111,
            desc="Destroy 1 random card in your hand, add 4 random Enhanced numbered cards to your hand"),
    ConsDef("Talisman",    CK_SPECTRAL, U_SEAL, 1, 1, 1, "Add a Gold Seal to 1 selected card in your hand"),
    ConsDef("Aura",        CK_SPECTRAL, U_AURA, 0, 1, 1,
            "Add Foil, Holographic, or Polychrome effect to 1 selected card in hand"),
    ConsDef("Wraith",      CK_SPECTRAL, U_WRAITH, desc="Creates a random Rare Joker, sets money to $0"),
    ConsDef("Sigil",       CK_SPECTRAL, U_SIGIL, need=2, desc="Converts all cards in hand to a single random suit"),
    ConsDef("Ouija",       CK_SPECTRAL, U_OUIJA, need=2,
            desc="Converts all cards in hand to a single random rank, -1 hand size"),
    ConsDef("Ectoplasm",   CK_SPECTRAL, U_ECTOPLASM, desc="Add Negative to a random Joker, -1 hand size (-2, -3 ... each time)"),
    ConsDef("Immolate",    CK_SPECTRAL, U_IMMOLATE, 5, need=2, desc="Destroys 5 random cards in hand, gain $20"),
    ConsDef("Ankh",        CK_SPECTRAL, U_ANKH,
            desc="Create a copy of a random Joker, destroy all other Jokers (Removes Negative from copy)"),
    ConsDef("Deja Vu",     CK_SPECTRAL, U_SEAL, 2, 1, 1, "Add a Red Seal to 1 selected card in your hand"),
    ConsDef("Hex",         CK_SPECTRAL, U_HEX, desc="Add Polychrome to a random Joker, and destroy all others"),
    ConsDef("Trance",      CK_SPECTRAL, U_SEAL, 3, 1, 1, "Add a Blue Seal to 1 selected card in your hand"),
    ConsDef("Medium",      CK_SPECTRAL, U_SEAL, 4, 1, 1, "Add a Purple Seal to 1 selected card in your hand"),
    ConsDef("Cryptid",     CK_SPECTRAL, U_CRYPTID, 2, 1, 1, "Create 2 copies of 1 selected card in your hand"),
    ConsDef("The Soul",    CK_SPECTRAL, U_SOUL, desc="Creates a Legendary Joker (Must have room)"),
    ConsDef("Black Hole",  CK_SPECTRAL, U_BLACK_HOLE, desc="Upgrade every poker hand by 1 level"),
]
N_CONS = len(CONS_TABLE)
CONS_NAMES = [d.name for d in CONS_TABLE]
CONS_DESC = [d.desc for d in CONS_TABLE]
CONS_ID = {d.name: i for i, d in enumerate(CONS_TABLE)}
CONS_KIND = jnp.array([d.kind for d in CONS_TABLE], jnp.int32)
CONS_USE = jnp.array([d.use for d in CONS_TABLE], jnp.int32)
CONS_PARAM = jnp.array([d.param for d in CONS_TABLE], jnp.int32)
CONS_LO = jnp.array([d.lo for d in CONS_TABLE], jnp.int32)
CONS_HI = jnp.array([d.hi for d in CONS_TABLE], jnp.int32)
CONS_NEED = jnp.array([d.need for d in CONS_TABLE], jnp.int32)
CONS_RANKS = jnp.array([d.ranks for d in CONS_TABLE], jnp.int32)
CONS_BASE_COST = jnp.array([4 if d.kind == CK_SPECTRAL else 3 for d in CONS_TABLE], jnp.int32)
CONS_COST = 3                                         # every Tarot and Planet (a Spectral card: 4)
C_FOOL = CONS_ID["The Fool"]
C_SOUL = CONS_ID["The Soul"]
C_BLACK_HOLE = CONS_ID["Black Hole"]
TAROTS = np.array([d.kind == CK_TAROT for d in CONS_TABLE])
PLANETS = np.array([d.kind == CK_PLANET for d in CONS_TABLE])
SPECTRALS = np.array([d.kind == CK_SPECTRAL for d in CONS_TABLE])
# The Soul and Black Hole are never in the Spectral pool: they come only from the 0.3 % pack rolls
SPECTRAL_POOL = SPECTRALS & ~np.isin(np.arange(len(CONS_TABLE)), [C_SOUL, C_BLACK_HOLE])
SOUL_ODDS = 0.003                                     # pseudorandom('soul_'..) > 0.997
PLANET_OF_HAND = jnp.array([[i for i, d in enumerate(CONS_TABLE) if d.kind == CK_PLANET and d.param == h][0]
                            for h in range(N_HAND_TYPES)], jnp.int32)
CONS_EMPTY = -1
MAX_PICK_USE = 3                                      # the most cards any consumable selects

# --- booster packs (P_CENTERS p_*) --------------------------------------------------------------
PK_ARCANA, PK_CELESTIAL, PK_STANDARD, PK_BUFFOON, PK_SPECTRAL = range(5)
PACK_KIND_NAMES = ["Arcana", "Celestial", "Standard", "Buffoon", "Spectral"]


class PackDef(NamedTuple):
    name   : str
    kind   : int
    size   : int      # cards offered
    choose : int      # cards taken
    cost   : int
    weight : float    # summed over the art variants of the same pack


PACK_TABLE = [
    PackDef("Arcana Pack",          PK_ARCANA,    3, 1, 4, 4.0),
    PackDef("Jumbo Arcana Pack",    PK_ARCANA,    5, 1, 6, 2.0),
    PackDef("Mega Arcana Pack",     PK_ARCANA,    5, 2, 8, 0.5),
    PackDef("Celestial Pack",       PK_CELESTIAL, 3, 1, 4, 4.0),
    PackDef("Jumbo Celestial Pack", PK_CELESTIAL, 5, 1, 6, 2.0),
    PackDef("Mega Celestial Pack",  PK_CELESTIAL, 5, 2, 8, 0.5),
    PackDef("Standard Pack",        PK_STANDARD,  3, 1, 4, 4.0),
    PackDef("Jumbo Standard Pack",  PK_STANDARD,  5, 1, 6, 2.0),
    PackDef("Mega Standard Pack",   PK_STANDARD,  5, 2, 8, 0.5),
    PackDef("Buffoon Pack",         PK_BUFFOON,   2, 1, 4, 1.2),
    PackDef("Jumbo Buffoon Pack",   PK_BUFFOON,   4, 1, 6, 0.6),
    PackDef("Mega Buffoon Pack",    PK_BUFFOON,   4, 2, 8, 0.15),
    PackDef("Spectral Pack",        PK_SPECTRAL,  2, 1, 4, 0.6),
    PackDef("Jumbo Spectral Pack",  PK_SPECTRAL,  4, 1, 6, 0.3),
    PackDef("Mega Spectral Pack",   PK_SPECTRAL,  4, 2, 8, 0.07),
]
N_PACKS = len(PACK_TABLE)
PACK_NAMES = [d.name for d in PACK_TABLE]
PACK_KIND = jnp.array([d.kind for d in PACK_TABLE], jnp.int32)
PACK_SIZE = jnp.array([d.size for d in PACK_TABLE], jnp.int32)
PACK_CHOOSE = jnp.array([d.choose for d in PACK_TABLE], jnp.int32)
PACK_COST = jnp.array([d.cost for d in PACK_TABLE], jnp.int32)
PACK_LOGW = jnp.log(jnp.array([d.weight for d in PACK_TABLE], jnp.float32))
P_BUFFOON_NORMAL = PACK_NAMES.index("Buffoon Pack")   # the first shop's first pack (get_pack)
# the free packs of the tags (tag.lua): Charm, Meteor, Ethereal, Standard, Buffoon
P_ARCANA_MEGA = PACK_NAMES.index("Mega Arcana Pack")
P_CELESTIAL_MEGA = PACK_NAMES.index("Mega Celestial Pack")
P_SPECTRAL_NORMAL = PACK_NAMES.index("Spectral Pack")
P_STANDARD_MEGA = PACK_NAMES.index("Mega Standard Pack")
P_BUFFOON_MEGA = PACK_NAMES.index("Mega Buffoon Pack")
MAX_PACK = 5                                          # the most cards a pack offers
PACK_SLOTS = 2                                        # packs per shop
STANDARD_ENHANCED = 0.4                               # 'stdset' > 0.6 -> an Enhanced card
STANDARD_SEAL = 0.2                                   # 1 - 0.02 * 10

# --- shop item kinds (create_card_for_shop: joker 20, tarot 4, planet 4, playing card 0) ----------
IK_JOKER, IK_TAROT, IK_PLANET, IK_CARD, IK_SPECTRAL = 0, 1, 2, 3, 4   # IK_SPECTRAL: the Ghost Deck
SHOP_KIND_LOGW = jnp.log(jnp.array([20.0, 4.0, 4.0], jnp.float32))
JOKER_RATE, TAROT_RATE, PLANET_RATE = 20.0, 4.0, 4.0
CARD_BASE_COST = 1                                    # a playing card in the shop (c_base: no cost)

# poll_edition thresholds (top of the roll): negative 0.3 %, polychrome 0.3 %, holo 1.4 %, foil 2 %
EDITION_ODDS = jnp.array([0.003, 0.006, 0.02, 0.04], jnp.float32)


def poll_edition(u, mod=1.0, no_neg=False, guaranteed=False, rate=1.0):
    """common_events.lua poll_edition, on a uniform draw u. -> ED_*
    rate: G.GAME.edition_rate (Hone 2, Glow Up 4); it scales Foil / Holo / Poly, not
    Negative, and a guaranteed roll (Aura, Wheel of Fortune) ignores it."""
    m = 25.0 if guaranteed else mod
    r = 1.0 if guaranteed else rate
    neg = (u > 1 - 0.003 * m) & (not no_neg)
    return jnp.select([neg, u > 1 - 0.006 * r * m, u > 1 - 0.02 * r * m, u > 1 - 0.04 * r * m],
                      [ED_NEG, ED_POLY, ED_HOLO, ED_FOIL], ED_NONE).astype(jnp.int32)


def cost_of(base, discount):
    """Card:set_cost: max(1, floor((base + 0.5) * (100 - discount) / 100)). base includes the
    edition's extra $. Clearance Sale 25 % / Liquidation 50 %."""
    return jnp.maximum(1, jnp.floor((base + 0.5) * (100 - discount) / 100)).astype(jnp.int32)


def sell_of(cost):
    """sell_cost before Egg / Gift Card: max(1, floor(cost / 2))."""
    return jnp.maximum(1, cost // 2).astype(jnp.int32)


# --- vouchers (P_CENTERS v_*: cost $10; the second of each pair needs the first) --------------------
V_OVERSTOCK, V_OVERSTOCK_PLUS, V_CLEARANCE, V_LIQUIDATION, V_HONE, V_GLOW_UP, V_REROLL_SURPLUS, \
    V_REROLL_GLUT, V_CRYSTAL_BALL, V_OMEN_GLOBE, V_TELESCOPE, V_OBSERVATORY, V_GRABBER, V_NACHO_TONG, \
    V_WASTEFUL, V_RECYCLOMANCY, V_TAROT_MERCHANT, V_TAROT_TYCOON, V_PLANET_MERCHANT, V_PLANET_TYCOON, \
    V_SEED_MONEY, V_MONEY_TREE, V_BLANK, V_ANTIMATTER, V_MAGIC_TRICK, V_ILLUSION, V_HIEROGLYPH, \
    V_PETROGLYPH, V_DIRECTORS_CUT, V_RETCON, V_PAINT_BRUSH, V_PALETTE = range(32)
VOUCHER_TABLE = [
    ("Overstock", "+1 card slot available in shop"),
    ("Overstock Plus", "+1 card slot available in shop"),
    ("Clearance Sale", "All cards and packs in shop are 25% off"),
    ("Liquidation", "All cards and packs in shop are 50% off"),
    ("Hone", "Foil, Holographic, and Polychrome cards appear 2X more often"),
    ("Glow Up", "Foil, Holographic, and Polychrome cards appear 4X more often"),
    ("Reroll Surplus", "Rerolls cost $2 less"),
    ("Reroll Glut", "Rerolls cost an additional $2 less"),
    ("Crystal Ball", "+1 consumable slot"),
    ("Omen Globe", "Spectral cards may appear in any of the Arcana Packs"),
    ("Telescope", "Celestial Packs always contain the Planet card for your most played poker hand"),
    ("Observatory", "Planet cards in your consumable area give X1.5 Mult for their specified poker hand"),
    ("Grabber", "Permanently gain +1 hand per round"),
    ("Nacho Tong", "Permanently gain an additional +1 hand per round"),
    ("Wasteful", "Permanently gain +1 discard each round"),
    ("Recyclomancy", "Permanently gain an additional +1 discard each round"),
    ("Tarot Merchant", "Tarot cards appear 2X more frequently in the shop"),
    ("Tarot Tycoon", "Tarot cards appear 4X more frequently in the shop"),
    ("Planet Merchant", "Planet cards appear 2X more frequently in the shop"),
    ("Planet Tycoon", "Planet cards appear 4X more frequently in the shop"),
    ("Seed Money", "Raise the cap on interest earned in each round to $10"),
    ("Money Tree", "Raise the cap on interest earned in each round to $20"),
    ("Blank", "Does nothing?"),
    ("Antimatter", "+1 Joker slot"),
    ("Magic Trick", "Playing cards can be purchased from the shop"),
    ("Illusion", "Playing cards in shop may have an Enhancement or an Edition"),
    ("Hieroglyph", "-1 Ante, -1 hand each round"),
    ("Petroglyph", "-1 Ante, -1 discard each round"),
    ("Director's Cut", "Reroll Boss Blind 1 time per Ante, $10 per roll"),
    ("Retcon", "Reroll Boss Blind unlimited times, $10 per roll"),
    ("Paint Brush", "+1 hand size"),
    ("Palette", "+1 hand size"),
]
N_VOUCHERS = len(VOUCHER_TABLE)
VOUCHER_NAMES = [n for n, _ in VOUCHER_TABLE]
VOUCHER_DESC = [d for _, d in VOUCHER_TABLE]
VOUCHER_ID = {n: i for i, n in enumerate(VOUCHER_NAMES)}
VOUCHER_COST = 10
# the voucher each one needs redeemed first (-1: none): the odd ones need the one before them
VOUCHER_NEEDS = jnp.array([i - 1 if i % 2 else -1 for i in range(N_VOUCHERS)], jnp.int32)
MAX_SHOP_VOUCHERS = 4          # the ante's voucher + up to 3 Voucher Tags (more are dropped)

# --- tags (P_TAGS, in order; min ante; tag.lua apply_to_run) -----------------------------------------
# min ante 0 = none (P_TAGS min_ante nil: any ante, also ante 0 after Hieroglyph / Petroglyph); 2 = from ante 2
TG_UNCOMMON, TG_RARE, TG_NEGATIVE, TG_FOIL, TG_HOLO, TG_POLY, TG_INVESTMENT, TG_VOUCHER, TG_BOSS, \
    TG_STANDARD, TG_CHARM, TG_METEOR, TG_BUFFOON, TG_HANDY, TG_GARBAGE, TG_ETHEREAL, TG_COUPON, \
    TG_DOUBLE, TG_JUGGLE, TG_D6, TG_TOP_UP, TG_SKIP, TG_ORBITAL, TG_ECONOMY = range(24)
TAG_TABLE = [
    ("Uncommon Tag", 0, "Shop has a free Uncommon Joker"),
    ("Rare Tag", 0, "Shop has a free Rare Joker"),
    ("Negative Tag", 2, "Next base edition shop Joker is free and becomes Negative"),
    ("Foil Tag", 0, "Next base edition shop Joker is free and becomes Foil"),
    ("Holographic Tag", 0, "Next base edition shop Joker is free and becomes Holographic"),
    ("Polychrome Tag", 0, "Next base edition shop Joker is free and becomes Polychrome"),
    ("Investment Tag", 0, "Gain $25 after defeating the next Boss Blind"),
    ("Voucher Tag", 0, "Adds one Voucher to the next shop"),
    ("Boss Tag", 0, "Rerolls the Boss Blind"),
    ("Standard Tag", 2, "Gives a free Mega Standard Pack"),
    ("Charm Tag", 0, "Gives a free Mega Arcana Pack"),
    ("Meteor Tag", 2, "Gives a free Mega Celestial Pack"),
    ("Buffoon Tag", 2, "Gives a free Mega Buffoon Pack"),
    ("Handy Tag", 2, "Gives $1 per played hand this run"),
    ("Garbage Tag", 2, "Gives $1 per unused discard this run"),
    ("Ethereal Tag", 2, "Gives a free Spectral Pack"),
    ("Coupon Tag", 0, "Initial cards and booster packs in next shop are free"),
    ("Double Tag", 0, "Gives a copy of the next selected Tag (Double Tag excluded)"),
    ("Juggle Tag", 0, "+3 hand size next round"),
    ("D6 Tag", 0, "Rerolls in next shop start at $0"),
    ("Top-up Tag", 2, "Create up to 2 Common Jokers (Must have room)"),
    ("Skip Tag", 0, "Gives $5 per skipped Blind this run"),
    ("Orbital Tag", 2, "Upgrade a poker hand by 3 levels"),
    ("Economy Tag", 0, "Doubles your money (Max of $40)"),
]
N_TAGS = len(TAG_TABLE)
TAG_NAMES = [n for n, _, _ in TAG_TABLE]
TAG_DESC = [d for _, _, d in TAG_TABLE]
TAG_ID = {n: i for i, n in enumerate(TAG_NAMES)}
TAG_MIN_ANTE = jnp.array([a for _, a, _ in TAG_TABLE], jnp.int32)
# the free pack of each pack tag (-1: none)
TAG_PACK = jnp.array([{TG_STANDARD: P_STANDARD_MEGA, TG_CHARM: P_ARCANA_MEGA, TG_METEOR: P_CELESTIAL_MEGA,
                       TG_BUFFOON: P_BUFFOON_MEGA, TG_ETHEREAL: P_SPECTRAL_NORMAL}.get(i, -1)
                      for i in range(N_TAGS)], jnp.int32)
TAG_EDITION = jnp.array([{TG_NEGATIVE: ED_NEG, TG_FOIL: ED_FOIL, TG_HOLO: ED_HOLO, TG_POLY: ED_POLY}.get(i, -1)
                         for i in range(N_TAGS)], jnp.int32)
MAX_TAGS = 8                   # held tags (a divergence: more are dropped)
SKIP_TAG_DOLLARS, INVESTMENT_DOLLARS, ECONOMY_MAX, ORBITAL_LEVELS, JUGGLE_SIZE = 5, 25, 40, 3, 3

# --- boss blinds (P_BLINDS bl_*: min ante, target x base, $; the showdown bosses only at ante 8) --------
BK_HOOK, BK_OX, BK_HOUSE, BK_WALL, BK_WHEEL, BK_ARM, BK_CLUB, BK_FISH, BK_PSYCHIC, BK_GOAD, BK_WATER, \
    BK_WINDOW, BK_MANACLE, BK_EYE, BK_MOUTH, BK_PLANT, BK_SERPENT, BK_PILLAR, BK_NEEDLE, BK_HEAD, \
    BK_TOOTH, BK_FLINT, BK_MARK, BK_ACORN, BK_LEAF, BK_VESSEL, BK_HEART, BK_BELL = range(28)


class BossDef(NamedTuple):
    name     : str
    min_ante : int
    mult     : float           # the target is mult x the ante's base
    dollars  : int
    showdown : bool
    desc     : str
    suit     : int = -1        # debuffs every card of this suit (Wild cards too)


BOSS_TABLE = [
    BossDef("The Hook", 1, 2, 5, False, "Discards 2 random cards per hand played"),
    BossDef("The Ox", 6, 2, 5, False, "Playing your most played hand sets money to $0"),
    BossDef("The House", 2, 2, 5, False, "First hand is drawn face down"),
    BossDef("The Wall", 2, 4, 5, False, "Extra large blind"),
    BossDef("The Wheel", 2, 2, 5, False, "1 in 7 cards get drawn face down"),
    BossDef("The Arm", 2, 2, 5, False, "Decrease level of played poker hand"),
    BossDef("The Club", 1, 2, 5, False, "All Club cards are debuffed", suit=0),
    BossDef("The Fish", 2, 2, 5, False, "Cards drawn face down after each hand played"),
    BossDef("The Psychic", 1, 2, 5, False, "Must play 5 cards"),
    BossDef("The Goad", 1, 2, 5, False, "All Spade cards are debuffed", suit=3),
    BossDef("The Water", 2, 2, 5, False, "Start with 0 discards"),
    BossDef("The Window", 1, 2, 5, False, "All Diamond cards are debuffed", suit=1),
    BossDef("The Manacle", 1, 2, 5, False, "-1 Hand Size"),
    BossDef("The Eye", 3, 2, 5, False, "No repeat hand types this round"),
    BossDef("The Mouth", 2, 2, 5, False, "Play only 1 hand type this round"),
    BossDef("The Plant", 4, 2, 5, False, "All face cards are debuffed"),
    BossDef("The Serpent", 5, 2, 5, False, "After Play or Discard, always draw 3 cards"),
    BossDef("The Pillar", 1, 2, 5, False, "Cards played previously this Ante are debuffed"),
    BossDef("The Needle", 2, 1, 5, False, "Play only 1 hand"),
    BossDef("The Head", 1, 2, 5, False, "All Heart cards are debuffed", suit=2),
    BossDef("The Tooth", 3, 2, 5, False, "Lose $1 per card played"),
    BossDef("The Flint", 2, 2, 5, False, "Base Chips and Mult are halved"),
    BossDef("The Mark", 2, 2, 5, False, "All face cards are drawn face down"),
    BossDef("Amber Acorn", 8, 2, 8, True, "Flips and shuffles all Joker cards"),
    BossDef("Verdant Leaf", 8, 2, 8, True, "All cards debuffed until 1 Joker sold"),
    BossDef("Violet Vessel", 8, 6, 8, True, "Very large blind"),
    BossDef("Crimson Heart", 8, 2, 8, True, "One random Joker disabled every hand"),
    BossDef("Cerulean Bell", 8, 2, 8, True, "Forces 1 card to always be selected"),
]
N_BOSSES = len(BOSS_TABLE)
BOSS_NAMES = [d.name for d in BOSS_TABLE]
BOSS_DESC = [d.desc for d in BOSS_TABLE]
BOSS_ID = {d.name: i for i, d in enumerate(BOSS_TABLE)}
BOSS_MIN_ANTE = jnp.array([d.min_ante for d in BOSS_TABLE], jnp.int32)
BOSS_MULT = jnp.array([d.mult for d in BOSS_TABLE], jnp.float32)
BOSS_DOLLARS = jnp.array([d.dollars for d in BOSS_TABLE], jnp.int32)
BOSS_SHOWDOWN = jnp.array([d.showdown for d in BOSS_TABLE])
BOSS_SUIT = jnp.array([d.suit for d in BOSS_TABLE], jnp.int32)
WHEEL_ODDS = 7                 # The Wheel: 1 in 7 cards drawn face down
SERPENT_DRAW = 3               # The Serpent: draw 3 after a PLAY or DISCARD
BOSS_REROLL_COST = 10          # Director's Cut / Retcon


# ---------------------------------------------------------------------------
# STAKES (game.lua start_run) — a stake has every rule of the stakes below it — and the
# stickers they put on shop and Buffoon pack jokers (common_events.lua create_card)
# ---------------------------------------------------------------------------
STAKE_NAMES = ["White Stake", "Red Stake", "Green Stake", "Black Stake", "Blue Stake", "Purple Stake",
               "Orange Stake", "Gold Stake"]     # stake 1 .. 8
STAKE_DESC = ["Base difficulty",
              "Small Blind gives no reward money",
              "Required score scales faster for each Ante",
              "Shop can have Eternal Jokers (can't be sold or destroyed)",
              "-1 Discard",
              "Required score scales faster for each Ante",
              "Shop can have Perishable Jokers (debuffed after 5 Rounds)",
              "Shop can have Rental Jokers (costs $3 per round)"]
N_STAKES = len(STAKE_NAMES)

# ---------------------------------------------------------------------------
# DECKS (game.lua P_CENTERS b_*, back.lua apply_to_run / trigger_effect): state.back
# ---------------------------------------------------------------------------
BACK_NAMES = ["Red Deck", "Blue Deck", "Yellow Deck", "Green Deck", "Black Deck", "Magic Deck",
              "Nebula Deck", "Ghost Deck", "Abandoned Deck", "Checkered Deck", "Zodiac Deck",
              "Painted Deck", "Anaglyph Deck", "Plasma Deck", "Erratic Deck"]
BACK_DESC = ["+1 discard every round",
             "+1 hand every round",
             "Start with an extra $10",
             "At end of each Round: $2 per remaining Hand, $1 per remaining Discard. Earn no Interest",
             "+1 Joker slot, -1 hand every round",
             "Start run with the Crystal Ball voucher and 2 copies of The Fool",
             "Start run with the Telescope voucher, -1 consumable slot",
             "Spectral cards may appear in the shop, start with a Hex card",
             "Start run with no Face Cards in your deck",
             "Start run with 26 Spades and 26 Hearts in deck",
             "Start run with Tarot Merchant, Planet Merchant, and Overstock",
             "+2 hand size, -1 Joker slot",
             "After defeating each Boss Blind, gain a Double Tag",
             "Balance Chips and Mult when calculating score for played hand, X2 base Blind size",
             "All Ranks and Suits in deck are randomized"]
N_BACKS = len(BACK_NAMES)
(B_RED, B_BLUE, B_YELLOW, B_GREEN, B_BLACK, B_MAGIC, B_NEBULA, B_GHOST, B_ABANDONED, B_CHECKERED,
 B_ZODIAC, B_PAINTED, B_ANAGLYPH, B_PLASMA, B_ERRATIC) = range(N_BACKS)
# the config of each deck, by state.back (back.lua apply_to_run)
BACK_HANDS = jnp.array([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)
BACK_DISCARDS = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)
BACK_DOLLARS = jnp.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)
BACK_JOKER_SLOT = jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], jnp.int32)
BACK_HAND_SIZE = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], jnp.int32)
BACK_CONS_SLOT = jnp.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], jnp.int32)
BACK_ANTE_SCALING = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], jnp.float32)   # Plasma: X2
GHOST_SPECTRAL_RATE = 2.0          # G.GAME.spectral_rate (0 for the other decks)
GREEN_PER_HAND, GREEN_PER_DISCARD = 2, 1     # modifiers.money_per_hand / money_per_discard
ST_NO_SMALL_REWARD = 2   # Red: no_blind_reward.Small
ST_SCALING_2 = 3         # Green: modifiers.scaling = 2 (get_blind_amount)
ST_ETERNAL = 4           # Black: enable_eternals_in_shop
ST_DISCARD = 5           # Blue: starting_params.discards - 1
ST_SCALING_3 = 6         # Purple: modifiers.scaling = 3
ST_PERISH = 7            # Orange: enable_perishables_in_shop
ST_RENTAL = 8            # Gold: enable_rentals_in_shop
STK_ETERNAL, STK_PERISH, STK_RENTAL = 1, 2, 4     # a joker's stickers (bits)
STICKER_NAMES = [(STK_ETERNAL, "Eternal"), (STK_PERISH, "Perishable"), (STK_RENTAL, "Rental")]
STICKER_POLL = 0.7       # Eternal: a poll > 0.7; Perishable: 0.4 < the same poll <= 0.7; Rental: its own poll > 0.7
PERISH_POLL = 0.4
PERISH_ROUNDS = 5        # G.GAME.perishable_rounds
RENTAL_RATE = 3          # G.GAME.rental_rate: $ each Rental joker takes at the end of a round
RENTAL_COST = 1          # set_cost: a Rental joker costs $1 (its sell value $1)
