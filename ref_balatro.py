"""The test reference — Balatro 1.0.1o rules for the authored jokers, in plain Python.

A second implementation, ported line by line from the game's Lua
(card.lua calculate_joker / calculate_dollar_bonus / get_chip_* / is_suit,
state_events.lua evaluate_play / end_round, misc_functions.lua
evaluate_poker_hand). It does not read jokers.py, items.py or scoring.py: the
joker numbers, the card modifiers, the hand levels, the copy rules, the hand
detection and the scoring order are all written again here.

Random draws are the one input it takes from the env: the env exposes the
1-in-N coin flips, the Lucky / Glass / Space Joker draws and Misprint's draw
(PlayOut.rolls / hrolls / lucky_rolls / glass_roll / space_roll / rnd), and the
reference consumes them by (card, repetition, slot). Everything else is
recomputed.

Cards: a plain id (rank = id % 13, 0 = two .. 12 = ace; suit = id // 13, C D H S),
or a Card with an enhancement, an edition, a seal and perma bonus chips.
"""

from collections import Counter

import numpy as np

TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(13)
CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
NOMINAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]      # base.nominal
HAND_ORDER = ["flush_five", "flush_house", "five_kind", "straight_flush", "four_kind",
              "full_house", "flush", "straight", "three_kind", "two_pair", "pair", "high_card"]
# G.GAME.hands: (s_chips, s_mult, l_chips, l_mult)
HAND_BASE = {"flush_five": (160, 16, 50, 3), "flush_house": (140, 14, 40, 4),
             "five_kind": (120, 12, 35, 3), "straight_flush": (100, 8, 40, 4),
             "four_kind": (60, 7, 30, 3), "full_house": (40, 4, 25, 2), "flush": (35, 4, 15, 2),
             "straight": (30, 4, 30, 3), "three_kind": (30, 3, 20, 2), "two_pair": (20, 2, 20, 1),
             "pair": (10, 2, 15, 1), "high_card": (5, 1, 10, 1)}
BLIND_PAY = [3, 4, 5]
WIN_ANTE = 8
SLOT_CAP = 16     # the env's divergence: Negative jokers add slots only up to 16
SECRET = ("flush_five", "flush_house", "five_kind")     # visible = false until played
# pairs(G.GAME.hands) on a new run: LuaJIT 2.0.5 (the game's lua51.dll) walks the game.lua
# constructor's 12 keys in this order
PAIRS_ORDER = ["flush_house", "full_house", "flush", "pair", "high_card", "straight_flush",
               "straight", "two_pair", "flush_five", "five_kind", "three_kind", "four_kind"]


def visible_hands(played):
    """The hands To Do List may pick: every hand but a secret one not yet played (evaluate_play
    sets visible)."""
    return [h for i, h in enumerate(HAND_ORDER) if h not in SECRET or played[i] > 0]


def ox_hand(played):
    """end_round after a boss: G.GAME.current_round.most_played_poker_hand. The loop keeps a hand
    when v.played > _played or (v.played == _played and _order > v.order); _order stays 100, so
    a tie goes to the later hand in pairs() order. -> its HAND_ORDER index."""
    best, n = "high_card", -1
    for h in PAIRS_ORDER:
        k = int(played[HAND_ORDER.index(h)])
        if k > n or k == n:
            best, n = h, k
    return HAND_ORDER.index(best)

# P_CENTERS m_* / e_* / seals
NONE_, BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, LUCKY = range(9)
FOIL, HOLO, POLY, NEGATIVE = 1, 2, 3, 4
GOLD_SEAL, RED_SEAL, BLUE_SEAL, PURPLE_SEAL = 1, 2, 3, 4

# game.lua P_CENTERS: rarity 2 (Uncommon) among the authored jokers (Baseball Card)
UNCOMMON = {"Bootstraps", "Fibonacci", "Arrowhead", "Onyx Agate", "Bull", "Joker Stencil", "Cloud 9",
            "Spare Trousers", "Castle", "Flash Card", "Card Sharp", "Ramen", "Acrobat",
            "Loyalty Card", "Seeing Double", "Flower Pot", "Erosion", "Blackboard", "Rocket",
            "Gift Card", "To the Moon", "Merry Andy", "Troubadour", "Turtle Bean", "Burglar",
            "Dusk", "Hack", "Sock and Buskin", "Seltzer", "Mime", "The Idol", "Bloodstone",
            "Rough Gem", "Four Fingers", "Shortcut", "Smeared Joker", "Pareidolia", "Showman",
            "Oops! All 6s", "Marble Joker", "Steel Joker", "Space Joker", "Constellation", "Hiker",
            "Vampire", "Hologram", "Midas Mask", "Stone Joker", "Lucky Cat", "Certificate",
            "Glass Joker", "Cartomancer", "Astronomer", "Satellite", "Trading Card",
            "Ceremonial Dagger", "Sixth Sense", "Madness", "Seance", "Luchador", "Diet Cola",
            "Mr. Bones", "Throwback", "Matador"}
# config values the reference needs to seed a bought joker (game.lua config)
SEED = {"Green Joker": 0, "Popcorn": 20, "Ride the Bus": 0, "Spare Trousers": 0, "Square Joker": 0,
        "Runner": 0, "Ice Cream": 100, "Wee Joker": 0, "Castle": 0, "Flash Card": 0,
        "Obelisk": 1, "Hit the Road": 1, "Ramen": 2, "Campfire": 1, "Rocket": 1,
        "Turtle Bean": 5, "Seltzer": 10, "Constellation": 1, "Red Card": 0, "Vampire": 1,
        "Hologram": 1, "Lucky Cat": 1, "Glass Joker": 1, "Ceremonial Dagger": 0, "Madness": 1,
        "Invisible Joker": 0, "Canio": 1, "Yorick": 1}
H_SIZE = {"Juggler": 1, "Merry Andy": -1, "Troubadour": 2, "Stuntman": -2}
D_SIZE = {"Drunkard": 1, "Merry Andy": 3}
H_PLAYS = {"Troubadour": -1}
PROBABLE = {"Bloodstone", "Business Card", "Reserved Parking", "8 Ball"}     # 1 in N per card
COPIERS = ("Blueprint", "Brainstorm")
X_COUNTERS = ("Obelisk", "Hit the Road", "Ramen", "Campfire", "Constellation", "Vampire",
              "Hologram", "Lucky Cat", "Glass Joker", "Madness", "Canio", "Yorick")


class Card:
    """One playing card. A Stone card has no rank and no suit. deb: the boss debuffs it.
    mark: The Pillar's played_this_ante (a DNA copy reports it)."""
    __slots__ = ("i", "r", "s", "enh", "ed", "seal", "bonus", "deb", "mark")

    def __init__(self, i, r=None, s=None, enh=0, ed=0, seal=0, bonus=0.0, deb=False, mark=False):
        self.i, self.enh, self.ed, self.seal, self.bonus, self.deb = i, enh, ed, seal, bonus, deb
        self.r = i % 13 if r is None else r
        self.s = i // 13 if s is None else s
        self.mark = mark

    def copy(self):
        return Card(self.i, self.r, self.s, self.enh, self.ed, self.seal, self.bonus, self.deb, self.mark)

    def __repr__(self):
        return "23456789TJQKA"[self.r] + "CDHS"[self.s] + (f"/{self.enh}" if self.enh else "")


def as_card(c):
    return c if isinstance(c, Card) else Card(c)


def rank(c):
    """get_id: -1 for a Stone card (it matches no rank)."""
    c = as_card(c)
    return -1 if c.enh == STONE else c.r


def suit(c): return as_card(c).s
def red(s): return s in (DIAMONDS, HEARTS)


def mods_of(names):
    """The rule jokers (find_joker): names of the jokers that act (a debuffed one does not)."""
    return dict(ff="Four Fingers" in names, sc="Shortcut" in names, sm="Smeared Joker" in names,
                sp="Splash" in names, pa="Pareidolia" in names, showman="Showman" in names,
                odds=2 ** names.count("Oops! All 6s"), credit=names.count("Credit Card"),
                moon=names.count("To the Moon"), chaos=names.count("Chaos the Clown"))


def is_suit(c, s, m, mode="play"):
    """Card:is_suit. mode "play": a debuffed card has no suit; "flush" (flush_calc): a debuffed
    card keeps its suit, a debuffed Wild card only its own; "bypass" (bypass_debuff)."""
    c = as_card(c)
    if mode == "play" and c.deb:
        return False
    if c.enh == STONE:
        return False
    if c.enh == WILD and not (mode == "flush" and c.deb):
        return True
    return c.s == s or (m["sm"] and red(c.s) == red(s))


def is_face(c, m, from_boss=False):
    """Card:is_face: a debuffed card is no face card (unless the boss asks)."""
    if as_card(c).deb and not from_boss:
        return False
    return m["pa"] or rank(c) in (JACK, QUEEN, KING)


# the suit bosses (blind.lua debuff.suit), the face boss, the others by name
BOSS_SUIT = {"The Club": CLUBS, "The Goad": SPADES, "The Window": DIAMONDS, "The Head": HEARTS}


def debuffed(c, boss, m, played_ante=False):
    """blind.lua debuff_card while boss is on (None: no boss ability)."""
    c = as_card(c)
    if boss in BOSS_SUIT:
        return is_suit(c, BOSS_SUIT[boss], m, "bypass")
    if boss == "The Plant":
        return is_face(c, m, from_boss=True)
    if boss == "The Pillar":
        return played_ante
    return boss == "Verdant Leaf"


def chip_bonus(c):
    """get_chip_bonus."""
    if c.enh == STONE:
        return 50 + c.bonus
    return NOMINAL[c.r] + (30 if c.enh == BONUS else 0) + c.bonus


def hand_base(name, level):
    s_chips, s_mult, l_chips, l_mult = HAND_BASE[name]
    return max(s_chips + l_chips * (level - 1), 0), max(s_mult + l_mult * (level - 1), 1)


# ---------------------------------------------------------------------------
# HAND DETECTION — evaluate_poker_hand / get_poker_hand_info
# ---------------------------------------------------------------------------
def get_flush(hand, m):
    need = 4 if m["ff"] else 5
    if len(hand) > 5 or len(hand) < need:
        return []
    for s in (SPADES, HEARTS, CLUBS, DIAMONDS):
        t = [c for c in hand if is_suit(c, s, m, "flush")]
        if len(t) >= need:
            return [t]
    return []


def get_straight(hand, m):
    need = 4 if m["ff"] else 5
    if len(hand) > 5 or len(hand) < need:
        return []
    ids = {}
    for c in hand:
        if rank(c) >= 0:
            ids.setdefault(rank(c) + 2, []).append(c)
    t, length, straight, skipped = [], 0, False, False
    for j in range(1, 15):
        k = 14 if j == 1 else j
        if k in ids:
            length += 1
            skipped = False
            t += ids[k]
        elif m["sc"] and not skipped and j != 14:
            skipped = True
        else:
            length = 0
            skipped = False
            if not straight:
                t = []
            if straight:
                break
        if length >= need:
            straight = True
    return [t] if straight else []


def get_x_same(num, hand):
    cnt = Counter(rank(c) for c in hand if rank(c) >= 0)
    return [[c for c in hand if rank(c) == r] for r in sorted(cnt, reverse=True) if cnt[r] == num]


def ref_hand(cards, m):
    """-> (hand name, the hand names it contains, the scored cards in lane order)."""
    cards = [as_card(c) for c in cards]
    p5, p4, p3, p2 = (get_x_same(n, cards) for n in (5, 4, 3, 2))
    fl, st = get_flush(cards, m), get_straight(cards, m)
    res = {h: [] for h in HAND_ORDER}
    if p5 and fl:
        res["flush_five"] = p5
    if p3 and p2 and fl:
        res["flush_house"] = [p3[0] + p2[0]]
    if p5:
        res["five_kind"] = p5
    if fl and st:
        res["straight_flush"] = [fl[0] + [c for c in st[0] if c not in fl[0]]]
    if p4:
        res["four_kind"] = p4
    if p3 and p2:
        res["full_house"] = [p3[0] + p2[0]]
    if fl:
        res["flush"] = fl
    if st:
        res["straight"] = st
    if p3:
        res["three_kind"] = p3
    if len(p2) == 2 or (len(p3) == 1 and len(p2) == 1):
        res["two_pair"] = [p2[0] + (p2[1] if len(p2) > 1 else p3[0])]
    if p2:
        res["pair"] = p2
    if cards:
        res["high_card"] = [[max(cards, key=lambda c: -1000 if c.enh == STONE else NOMINAL[c.r] + 0.01 * c.r)]]
    if res["five_kind"]:
        res["four_kind"] = [res["five_kind"][0][:4]]
    if res["four_kind"]:
        res["three_kind"] = [res["four_kind"][0][:3]]
    if res["three_kind"]:
        res["pair"] = [res["three_kind"][0][:2]]
    name = next((h for h in HAND_ORDER if res[h]), "high_card")
    scored = set(res[name][0]) if res[name] else set()
    if name == "four_kind" and p4:
        scored = set(p4[0])
    if name == "three_kind":
        scored = set(p3[0])
    if name == "pair":
        scored = set(p2[0])
    scored |= {c for c in cards if c.enh == STONE}          # the "pure" cards
    if m["sp"]:
        scored = set(cards)
    return name, {h for h in HAND_ORDER if res[h]}, [c for c in cards if c in scored]


# ---------------------------------------------------------------------------
# COPIES — Blueprint / Brainstorm call the target's calculate_joker
# ---------------------------------------------------------------------------
def copy_target(names, i):
    """-> the slot whose effect slot i produces (itself for a plain joker), or None."""
    n, hops, cur = len(names), 0, i
    while names[cur] in COPIERS:
        hops += 1
        if hops > n + 1:
            return None
        o = cur + 1 if names[cur] == "Blueprint" else 0
        if o >= n or o == cur:
            return None
        cur = o
    return cur


def acting_names(J):
    """The names of the jokers that act (Crimson Heart's debuffed joker does not: find_joker)."""
    return [x["name"] for x in J if not x.get("deb")]


def effective(J):
    """Per slot: the slot whose effect it produces (copy_target), None for a debuffed joker or a
    copy of one (calculate_joker returns nil)."""
    names = [x["name"] for x in J]
    out = []
    for i in range(len(J)):
        t = copy_target(names, i)
        out.append(None if J[i].get("deb") or (t is not None and J[t].get("deb")) else t)
    return out


# ---------------------------------------------------------------------------
# A PLAY — evaluate_play
# ---------------------------------------------------------------------------
def ref_play(J, cards, held, g, rolls=None, hrolls=None, rnd=None, mods=None, lucky=None,
             glass=None, space=None):
    """Score one PLAY.
    J      : list of joker records {name, v, st, sell, ed}, slot order (mutated: counters)
    cards  : played cards (ids or Cards), lane order (at most 5). Cards are copied:
             the result carries their enhancement / perma bonus after the hand.
    held   : (lane, card) of the cards left in hand
    g      : dict money, hands_left (after this hand), discards, deck_left, slots, nines,
             deck_size, played (Counter, before), round (Counter, before), targets
             (idol, mail, ancient, castle), is_boss, levels (dict name -> level),
             steel, stone, enhanced (full-deck tallies), tarots_used
    rolls  : fn (card index, rep, slot) -> bool for the 1-in-N draws (default: all hit)
    lucky  : fn (card index, rep, 0 mult / 1 $) -> bool; glass: fn (card index) -> bool;
             space: fn (slot) -> bool  (default: never)
    g may also hold the boss's rules: hand_debuffed (The Psychic / Eye / Mouth: 0 x 0, only
    Matador and AFTER), flint, boss_trigger (The Arm / The Ox did something), observatory
    (hand name -> Planets held), skips. A Card with deb set is debuffed.
    -> dict(hand, chips, mult, score, money, gone (set of slots used up), played, round,
            acted, cards (after), broken (card indices), level_up, tarots, spawns (the
            Tarot / Spectral kinds the hand made, in order), dna)"""
    roll = rolls or (lambda k, r, j: True)
    hroll = hrolls or (lambda k, r, j: True)
    lroll = lucky or (lambda k, r, w: False)
    groll = glass or (lambda k: False)
    sroll = space or (lambda j: False)
    names = [x["name"] for x in J]
    m = mods if mods is not None else mods_of(acting_names(J))
    src = [as_card(c) for c in cards]
    cards = [c.copy() for c in src]
    held = [(lane, as_card(c)) for lane, c in held]
    name, has, scored = ref_hand(cards, m)
    played, rnd_cnt = Counter(g["played"]), Counter(g["round"])
    hands_total = sum(played.values())
    played[name] += 1
    rnd_cnt[name] += 1
    money = g["money"]
    tg = g["targets"]
    eff = effective(J)
    acted = set()
    tarots = 0
    spawns = []
    triggered = (g.get("boss_trigger", False) or g.get("flint", False) or any(c.deb for c in scored)
                 or g.get("hand_debuffed", False))
    if g.get("hand_debuffed", False):
        # debuff_hand: 0 x 0; jokers: only Matador's $ (debuffed_hand), then AFTER
        for i in range(len(J)):
            t = eff[i]
            if t is not None and J[t]["name"] == "Matador":
                money += 8
                acted.add(i)
        gone = _after(J)
        return dict(hand=name, chips=0, mult=0, score=0, money=money - g["money"], gone=gone, played=played,
                    round=rnd_cnt, acted=acted, cards=cards, broken=[], level_up=0, tarots=0, spawns=[],
                    dna=[], triggered=True, reset=[])

    # BEFORE, one joker at a time in slot order (evaluate_play: eval_card per joker). Ride the Bus
    # and DNA see the cards as the Midas Masks / Vampires to their left left them.
    level_up = 0
    vampired = set()                                       # card.lua v.vampired: until the hand ends
    reset = set()                                          # set_ability (Midas Mask, Vampire): The Pillar mark goes
    dna = []                                               # DNA copies (copies too), slot order
    first_hand = sum(Counter(g["round"]).values()) == 0
    for i, x in enumerate(J):
        t = eff[i]
        if t is None:
            continue
        y, bp = J[t], t != i
        n = y["name"]
        if n == "Space Joker":
            if sroll(i):
                level_up += 1
                acted.add(i)
            continue
        if n == "DNA":
            if first_hand and len(cards) == 1 and len(dna) < g.get("card_room", 128):
                # G.hand:emplace in BEFORE: the copy is a held card of this hand, at its end, and
                # Hologram gains at once (playing_card_joker_effects). copy_card keeps the card's
                # debuff and its The Pillar mark (gone if a Midas Mask / Vampire touched it first).
                cp = cards[0].copy()
                cp.mark = cards[0] not in reset
                dna.append(cp)
                held.append((max([lane for lane, _ in held] + [-1]) + 1, cp))
                for z in J:
                    if z["name"] == "Hologram" and not z.get("deb"):
                        z["v"] += 0.25
            continue
        if bp and n not in ("To Do List",):
            continue
        if n == "Spare Trousers" and ("two_pair" in has or "full_house" in has):
            y["v"] += 2
        elif n == "Square Joker" and len(cards) == 4:
            y["v"] += 4
        elif n == "Runner" and "straight" in has:
            y["v"] += 15
        elif n == "To Do List" and name == HAND_ORDER[y["st"]]:
            money += 4
            acted.add(i)
        elif n == "Ride the Bus":
            y["v"] = 0 if any(is_face(c, m) for c in scored) else y["v"] + 1
        elif n == "Obelisk":
            most = all(played[k] < played[name] for k in HAND_ORDER if k != name)
            y["v"] = 1 if most else y["v"] + 0.2
        elif n == "Green Joker":
            y["v"] += 1
        elif n == "Midas Mask":
            for c in scored:
                if is_face(c, m):
                    c.enh = GOLD
                    reset.add(c)
                    acted.add(i)
        elif n == "Vampire":
            k = 0
            for c in scored:
                if c.enh != NONE_ and not c.deb and c not in vampired:
                    c.enh = NONE_
                    vampired.add(c)
                    reset.add(c)
                    k += 1
            y["v"] += 0.1 * k
            if k:
                acted.add(i)

    level = g.get("levels", {}).get(name, 1) + level_up
    chips, mult = hand_base(name, level)
    if g.get("flint", False):                    # The Flint: halved, rounded (mult at least 1)
        mult = max(int(mult * 0.5 + 0.5), 1)
        chips = max(int(chips * 0.5 + 0.5), 0)
    pos = {c: k for k, c in enumerate(cards)}
    first_face = next((c for c in scored if is_face(c, m)), None)
    # scored cards, in order (a debuffed one is skipped: no chips, no effects)
    for c in scored:
        k = pos[c]
        if c.deb:
            continue
        reps = 1 + (c.seal == RED_SEAL)
        for i in range(len(J)):
            t = eff[i]
            if t is None:
                continue
            n = J[t]["name"]
            before = reps
            if n == "Sock and Buskin" and is_face(c, m):
                reps += 1
            elif n == "Hanging Chad" and c is scored[0]:
                reps += 2
            elif n == "Dusk" and g["hands_left"] == 0:
                reps += 1
            elif n == "Seltzer":
                reps += 1
            elif n == "Hack" and rank(c) in (TWO, THREE, FOUR, FIVE):
                reps += 1
            if reps > before:
                acted.add(i)
        for r in range(reps):
            # the card: chips, mult (Mult, Lucky), $ (Gold Seal, Lucky), X (Glass), then its edition
            chips += chip_bonus(c)
            trig = False
            if c.enh == MULT:
                mult += 4
            if c.enh == LUCKY and lroll(k, r, 0):
                mult += 20
                trig = True
            if c.seal == GOLD_SEAL:
                money += 3
            if c.enh == LUCKY and lroll(k, r, 1):
                money += 20
                trig = True
            if c.enh == GLASS:
                mult *= 2
            if c.ed == FOIL:
                chips += 50
            if c.ed == HOLO:
                mult += 10
            if c.ed == POLY:
                mult *= 1.5
            for i in range(len(J)):
                t = eff[i]
                if t is None:
                    continue
                y, bp = J[t], t != i
                n = y["name"]
                e = {}
                if n == "Wee Joker" and rank(c) == TWO and not bp:
                    y["v"] += 8
                elif n == "Lucky Cat" and trig and not bp:
                    y["v"] += 0.25
                    acted.add(i)
                elif n == "Hiker":
                    c.bonus += 5
                    acted.add(i)
                elif n == "Golden Ticket" and c.enh == GOLD:
                    e = {"$": 4}
                elif n == "8 Ball" and rank(c) == EIGHT and roll(k, r, i):
                    tarots += 1
                    acted.add(i)
                elif n == "Photograph" and c is first_face:
                    e = {"x": 2}
                elif n == "The Idol" and rank(c) == tg["idol"] % 13 and is_suit(c, tg["idol"] // 13, m):
                    e = {"x": 2}
                elif n == "Scary Face" and is_face(c, m):
                    e = {"chips": 30}
                elif n == "Smiley Face" and is_face(c, m):
                    e = {"mult": 5}
                elif n == "Scholar" and rank(c) == ACE:
                    e = {"chips": 20, "mult": 4}
                elif n == "Walkie Talkie" and rank(c) in (TEN, FOUR):
                    e = {"chips": 10, "mult": 4}
                elif n == "Business Card" and is_face(c, m) and roll(k, r, i):
                    e = {"$": 2}
                elif n == "Fibonacci" and rank(c) in (TWO, THREE, FIVE, EIGHT, ACE):
                    e = {"mult": 8}
                elif n == "Even Steven" and rank(c) in (TWO, FOUR, SIX, EIGHT, TEN):
                    e = {"mult": 4}
                elif n == "Odd Todd" and rank(c) in (THREE, FIVE, SEVEN, NINE, ACE):
                    e = {"chips": 31}
                elif n in ("Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker") and \
                        is_suit(c, {"Greedy Joker": DIAMONDS, "Lusty Joker": HEARTS,
                                    "Wrathful Joker": SPADES, "Gluttonous Joker": CLUBS}[n], m):
                    e = {"mult": 3}
                elif n == "Rough Gem" and is_suit(c, DIAMONDS, m):
                    e = {"$": 1}
                elif n == "Onyx Agate" and is_suit(c, CLUBS, m):
                    e = {"mult": 7}
                elif n == "Arrowhead" and is_suit(c, SPADES, m):
                    e = {"chips": 50}
                elif n == "Bloodstone" and is_suit(c, HEARTS, m) and roll(k, r, i):
                    e = {"x": 1.5}
                elif n == "Ancient Joker" and is_suit(c, tg["ancient"], m):
                    e = {"x": 1.5}
                elif n == "Triboulet" and rank(c) in (KING, QUEEN):
                    e = {"x": 2}
                chips += e.get("chips", 0)
                mult += e.get("mult", 0)
                money += e.get("$", 0)
                mult *= e.get("x", 1)
                if e:
                    acted.add(i)

    # cards held in hand: Steel, then the jokers; Red Seal and Mime repeat a card that did something
    lo = None
    for lane, c in held:                     # Raised Fist: the last card with the lowest id
        if rank(c) >= 0 and (lo is None or rank(c) <= rank(lo[1])):
            lo = (lane, c)
    for lane, c in held:
        if c.deb:                            # a debuffed held card: nothing (Mime has nothing to repeat)
            continue
        reps, k = 1, 0
        while k < reps:
            any_fx = False
            if c.enh == STEEL:
                mult *= 1.5
                any_fx = True
            for i in range(len(J)):
                t = eff[i]
                if t is None:
                    continue
                n = J[t]["name"]
                e = None
                if n == "Shoot the Moon" and rank(c) == QUEEN:
                    e = {"mult": 13}
                elif n == "Baron" and rank(c) == KING:
                    e = {"x": 1.5}
                elif n == "Reserved Parking" and is_face(c, m) and hroll(lane, k, i):
                    e = {"$": 1}
                elif n == "Raised Fist" and lo is not None and lo[0] == lane:
                    e = {"mult": 2 * NOMINAL[rank(c)]}
                if e is not None:
                    any_fx = True
                    acted.add(i)
                    money += e.get("$", 0)
                    mult += e.get("mult", 0)
                    mult *= e.get("x", 1)
            if k == 0 and any_fx:
                reps += c.seal == RED_SEAL
                for i in range(len(J)):
                    t = eff[i]
                    if t is not None and J[t]["name"] == "Mime":
                        reps += 1
                        acted.add(i)
            k += 1

    # each joker: its edition's chips / mult, its main effect, Baseball Card on it, its Polychrome
    held_cards = [c for _, c in held]
    n_jokers = len(J)
    for i, x in enumerate(J):
        ed = 0 if x.get("deb") else x.get("ed", 0)          # a debuffed joker has no edition (get_edition)
        if ed == FOIL:
            chips += 50
        if ed == HOLO:
            mult += 10
        t = eff[i]
        if t is not None:
            y = J[t]
            n, v = y["name"], y["v"]
            add_m = add_c = 0
            xm = 1
            tm = {"Jolly Joker": ("pair", 8), "Zany Joker": ("three_kind", 12),
                  "Mad Joker": ("two_pair", 10), "Crazy Joker": ("straight", 12),
                  "Droll Joker": ("flush", 10)}
            tc = {"Sly Joker": ("pair", 50), "Wily Joker": ("three_kind", 100),
                  "Clever Joker": ("two_pair", 80), "Devious Joker": ("straight", 100),
                  "Crafty Joker": ("flush", 80)}
            tx = {"The Duo": ("pair", 2), "The Trio": ("three_kind", 3), "The Family": ("four_kind", 4),
                  "The Order": ("straight", 3), "The Tribe": ("flush", 2)}
            if n == "Loyalty Card":
                if (5 - 1 - (hands_total - y["st"])) % 6 == 5:
                    xm = 4
            elif n in X_COUNTERS and v > 1:
                xm = v
            elif n in tx and tx[n][0] in has:
                xm = tx[n][1]
            elif n == "Joker Stencil":
                xm = max(g["slots"] - n_jokers + names.count("Joker Stencil"), 1)
            elif n in tm and tm[n][0] in has:
                add_m = tm[n][1]
            elif n in tc and tc[n][0] in has:
                add_c = tc[n][1]
            elif n == "Half Joker" and len(cards) <= 3:
                add_m = 20
            elif n == "Abstract Joker":
                add_m = 3 * n_jokers
            elif n == "Acrobat" and g["hands_left"] == 0:
                xm = 3
            elif n == "Mystic Summit" and g["discards"] == 0:
                add_m = 15
            elif n == "Misprint":
                add_m = rnd(i) if rnd else 0
            elif n == "Banner" and g["discards"] > 0:
                add_c = 30 * g["discards"]
            elif n == "Stuntman":
                add_c = 250
            elif n == "Supernova":
                add_m = played[name]
            elif n == "Flower Pot":
                got = set()
                for wild in (False, True):
                    for c in scored:
                        if (c.enh == WILD) != wild:
                            continue
                        for s in (HEARTS, DIAMONDS, SPADES, CLUBS):
                            if is_suit(c, s, m, "play" if wild else "bypass") and s not in got:
                                got.add(s)
                                break
                if len(got) == 4:
                    xm = 3
            elif n == "Seeing Double":
                cnt = {s: sum(is_suit(c, s, m) for c in scored if c.enh != WILD) for s in range(4)}
                for c in scored:
                    if c.enh == WILD and not c.deb:
                        for s in (CLUBS, DIAMONDS, SPADES, HEARTS):
                            if cnt[s] == 0:
                                cnt[s] = 1
                                break
                if cnt[CLUBS] > 0 and (cnt[HEARTS] or cnt[DIAMONDS] or cnt[SPADES]):
                    xm = 2
            elif n in ("Wee Joker", "Castle", "Square Joker", "Runner", "Ice Cream") and v > 0:
                add_c = v
            elif n == "Blue Joker" and g["deck_left"] > 0:
                add_c = 2 * g["deck_left"]
            elif n == "Erosion" and 52 - (g["deck_size"] + len(dna)) > 0:          # the DNA copies count
                add_m = 4 * (52 - (g["deck_size"] + len(dna)))
            elif n == "Bull" and money > 0:
                add_c = 2 * money
            elif n == "Blackboard" and all(is_suit(c, CLUBS, m, "flush") or is_suit(c, SPADES, m, "flush")
                                           for c in held_cards):
                xm = 3
            elif n == "Swashbuckler":
                add_m = sum(z["sell"] for k, z in enumerate(J) if k != t)
            elif n == "Joker":
                add_m = 4
            elif n in ("Spare Trousers", "Ride the Bus", "Flash Card", "Popcorn", "Green Joker",
                       "Red Card", "Ceremonial Dagger") and v > 0:
                add_m = v
            elif n == "Throwback" and g.get("skips", 0) > 0:
                xm = 1 + 0.25 * g["skips"]
            elif n == "Matador" and triggered:
                money += 8
                acted.add(i)
            elif n == "Seance" and "straight_flush" in has:
                spawns.append("spectral")
                acted.add(i)
            elif n == "Gros Michel":
                add_m = 15
            elif n == "Cavendish":
                xm = 3
            elif n == "Card Sharp" and rnd_cnt[name] > 1:
                xm = 3
            elif n == "Bootstraps" and money // 5 >= 1:
                add_m = 2 * (money // 5)
            elif n == "Steel Joker" and g.get("steel", 0) > 0:
                xm = 1 + 0.2 * g["steel"]
            elif n == "Stone Joker" and g.get("stone", 0) > 0:
                add_c = 25 * g["stone"]
            elif n == "Driver's License" and g.get("enhanced", 0) >= 16:
                xm = 3
            elif n == "Fortune Teller" and g.get("tarots_used", 0) > 0:
                add_m = g["tarots_used"]
            elif n == "Superposition" and "straight" in has and any(rank(c) == ACE for c in scored):
                spawns.append("tarot")
                acted.add(i)
            elif n == "Vagabond" and g["money"] <= 4:          # G.GAME.dollars: not this hand's $
                spawns.append("tarot")
                acted.add(i)
            mult += add_m
            chips += add_c
            mult *= xm
            if add_m or add_c or xm != 1:
                acted.add(i)
        for k in range(len(J)):                       # Joker on Joker
            t = eff[k]
            if t is not None and J[t]["name"] == "Baseball Card" and t != i and x["name"] in UNCOMMON:
                mult *= 1.5
                acted.add(k)
        if ed == POLY:
            mult *= 1.5
    # Observatory: each held Planet of this hand, X1.5 (the consumables come after the jokers)
    mult *= 1.5 ** g.get("observatory", {}).get(name, 0)

    # AFTER, then the Glass cards break (not a debuffed one)
    gone = _after(J)
    broken = [pos[c] for c in scored if c.enh == GLASS and not c.deb and groll(pos[c])]
    return dict(hand=name, chips=chips, mult=mult, score=int(chips * mult // 1), money=money - g["money"],
                gone=gone, played=played, round=rnd_cnt, acted=acted, cards=cards, broken=broken,
                level_up=level_up, tarots=tarots, spawns=["tarot"] * tarots + spawns, dna=dna,
                triggered=triggered, reset=[k for k, c in enumerate(cards) if c in reset])


def _after(J):
    """The AFTER step: Ice Cream -5, Seltzer -1 (not a debuffed one). -> the slots used up."""
    gone = set()
    for i, x in enumerate(J):
        if x.get("deb"):
            continue
        if x["name"] == "Ice Cream":
            if x["v"] - 5 <= 0:
                gone.add(i)
            else:
                x["v"] -= 5
        elif x["name"] == "Seltzer":
            if x["v"] - 1 <= 0:
                gone.add(i)
            else:
                x["v"] -= 1
    return gone


def ref_discard(J, cards, g, hook=False):
    """One DISCARD of cards (lane order). Mutates the counters. hook: The Hook's discard (no
    Burnt Joker). A debuffed card (Card.deb) counts for no suit / rank / face.
    -> (money, gone slots, dict(burnt levels: hand name or None, purple, trading))"""
    names = acting_names(J)
    m = mods_of(names)
    tg = g["targets"]
    cards = [as_card(c) for c in cards]
    eff = effective(J)
    money, gone = 0, set()
    first = g.get("discards_used", 1) == 0
    burnt = sum(1 for i in range(len(J)) if eff[i] is not None and J[eff[i]]["name"] == "Burnt Joker") \
        if first and not hook else 0
    for ci, c in enumerate(cards):
        last = ci == len(cards) - 1
        for i in range(len(J)):
            t = eff[i]
            if t is None:
                continue
            y, bp = J[t], t != i
            n = y["name"]
            if n == "Ramen" and not bp:
                if y["v"] - 0.01 <= 1 + 1e-9:
                    gone.add(i)
                else:
                    y["v"] -= 0.01
            elif n == "Castle" and not bp and is_suit(c, tg["castle"], m):
                y["v"] += 3
            elif n == "Mail-In Rebate" and rank(c) == tg["mail"] and not c.deb:
                money += 5
            elif n == "Hit the Road" and not bp and rank(c) == JACK and not c.deb:
                y["v"] += 0.5
            elif n == "Yorick" and not bp:
                y["st"] += 1
                if y["st"] == 23:
                    y["st"] = 0
                    y["v"] += 1
            elif n == "Green Joker" and not bp and last:
                y["v"] = max(0, y["v"] - 1)
            elif n == "Faceless Joker" and last and sum(is_face(z, m) for z in cards) >= 3:
                money += 5
    trading = "Trading Card" in names and first and len(cards) == 1
    if trading:
        money += 3 * names.count("Trading Card")              # each one pays (not its copies)
    extra = dict(burnt=(ref_hand(cards, m)[0], burnt) if burnt else None,
                 purple=sum(c.seal == PURPLE_SEAL and not c.deb for c in cards), trading=trading)
    return money, gone, extra


def ref_end_of_round(J, g):
    """end_round jokers (real slots only). Mutates. -> gone slots."""
    gone = set()
    for i, x in enumerate(J):
        n = x["name"]
        if x.get("deb"):
            continue
        if n == "Campfire" and g["is_boss"] and x["v"] > 1:
            x["v"] = 1
        elif n == "Rocket" and g["is_boss"]:
            x["v"] += 2
        elif n == "Turtle Bean":
            if x["v"] - 1 <= 0:
                gone.add(i)
            else:
                x["v"] -= 1
        elif n == "Popcorn":
            if x["v"] - 4 <= 0:
                gone.add(i)
            else:
                x["v"] -= 4
        elif n == "Egg":
            x["sell"] += 3
        elif n == "Gift Card":
            for z in J:
                z["sell"] += 1           # every joker, a debuffed one too
        elif n == "Hit the Road" and x["v"] > 1:
            x["v"] = 1
        elif n == "Invisible Joker":
            x["v"] += 1
    return gone


def ref_held_end_of_round(J, held):
    """The cards left in hand at the end of the round: Gold cards pay $3, Blue
    Seals make a Planet; a card that did something repeats for its Red Seal and
    for each Mime (copies too). -> ($, Planets)"""
    names = [x["name"] for x in J]
    mime = sum(1 for t in effective(J) if t is not None and names[t] == "Mime")
    money = planets = 0
    for c in (as_card(c) for c in held):
        if c.deb:
            continue
        if c.enh == GOLD or c.seal == BLUE_SEAL:
            reps = 1 + (c.seal == RED_SEAL) + mime
            money += 3 * reps * (c.enh == GOLD)
            planets += reps * (c.seal == BLUE_SEAL)
    return money, planets


def ref_dollar_bonus(J, g):
    """calculate_dollar_bonus over the held jokers (copies never pay)."""
    out = 0
    for x in J:
        n = x["name"]
        if x.get("deb"):
            continue
        if n == "Golden Joker":
            out += 4
        elif n == "Cloud 9":
            out += g["nines"]
        elif n == "Rocket":
            out += x["v"]
        elif n == "Delayed Gratification" and g["discards_used"] == 0 and g["discards"] > 0:
            out += 2 * g["discards"]
        elif n == "Satellite":
            out += g.get("unique_planets", 0)
    return out


def ref_budget(J, base_hand, base_hands, base_discards, hand_extra=0, slots_extra=0):
    """-> (hand size, hands per round, discards per round, joker slots). A debuffed joker
    (Crimson Heart: x["deb"]) gives nothing; hand_extra: Juggle Tag - The Manacle; slots_extra:
    Antimatter."""
    act = [x for x in J if not x.get("deb")]
    h = base_hand + hand_extra + sum(H_SIZE.get(x["name"], 0) for x in act) \
        + sum(int(x["v"]) for x in act if x["name"] == "Turtle Bean")
    neg = sum(1 for x in J if x.get("ed", 0) == NEGATIVE)
    return (min(max(h, 0), 32), max(base_hands + sum(H_PLAYS.get(x["name"], 0) for x in act), 1),
            max(base_discards + sum(D_SIZE.get(x["name"], 0) for x in act), 0), min(5 + slots_extra + neg, SLOT_CAP))


# ---------------------------------------------------------------------------
# BLINDS, TAGS, VOUCHERS — game.lua P_BLINDS / P_TAGS / P_CENTERS v_* (written again here)
# ---------------------------------------------------------------------------
# boss: (min ante, x the ante's base, $, showdown)
BOSSES = {"The Hook": (1, 2, 5, False), "The Ox": (6, 2, 5, False), "The House": (2, 2, 5, False),
          "The Wall": (2, 4, 5, False), "The Wheel": (2, 2, 5, False), "The Arm": (2, 2, 5, False),
          "The Club": (1, 2, 5, False), "The Fish": (2, 2, 5, False), "The Psychic": (1, 2, 5, False),
          "The Goad": (1, 2, 5, False), "The Water": (2, 2, 5, False), "The Window": (1, 2, 5, False),
          "The Manacle": (1, 2, 5, False), "The Eye": (3, 2, 5, False), "The Mouth": (2, 2, 5, False),
          "The Plant": (4, 2, 5, False), "The Serpent": (5, 2, 5, False), "The Pillar": (1, 2, 5, False),
          "The Needle": (2, 1, 5, False), "The Head": (1, 2, 5, False), "The Tooth": (3, 2, 5, False),
          "The Flint": (2, 2, 5, False), "The Mark": (2, 2, 5, False), "Amber Acorn": (8, 2, 8, True),
          "Verdant Leaf": (8, 2, 8, True), "Violet Vessel": (8, 6, 8, True), "Crimson Heart": (8, 2, 8, True),
          "Cerulean Bell": (8, 2, 8, True)}
# get_blind_amount by G.GAME.modifiers.scaling (1: White / Red, 2: from Green, 3: from Purple); ante 0: 100
ANTE_BASE = {1: [100, 300, 800, 2000, 5000, 11000, 20000, 35000, 50000],
             2: [100, 300, 900, 2600, 8000, 20000, 36000, 60000, 100000],
             3: [100, 300, 1000, 3200, 9000, 25000, 60000, 110000, 200000]}


def scaling_of(stake):
    """game.lua start_run: stake >= 3 -> scaling 2, stake >= 6 -> scaling 3."""
    return 3 if stake >= 6 else 2 if stake >= 3 else 1


def blind_pay(blind, boss, stake):
    """$ a beaten blind pays: 3 / 4 / the boss's; stake >= 2: no_blind_reward.Small."""
    if blind == 0 and stake >= 2:
        return 0
    return BOSSES[boss][2] if blind == 2 and boss else 5 if blind == 2 else BLIND_PAY[blind]


def boss_eligible(name, ante):
    """get_new_boss: the showdown bosses at ante 8, 16 ...; the others from their min ante."""
    lo, _, _, show = BOSSES[name]
    if show:
        return ante % WIN_ANTE == 0 and ante >= 2
    return lo <= max(1, ante) and (max(1, ante) % WIN_ANTE != 0 or ante < 2)


def blind_chips(ante, blind, boss=None, scale=1.0, disabled=False, stake=1):
    """The target: the ante's base (the stake's scaling) x 1 / 1.5 / the boss's x (a disabled
    Wall / Vessel: x2)."""
    x = [1, 1.5][blind] if blind < 2 else (2 if boss is None else BOSSES[boss][1])
    if blind == 2 and disabled and boss in ("The Wall", "Violet Vessel"):
        x = 2
    # float32, as the env computes it (a scale of 0.03 x 300 is 9 there, not 8.99...)
    t = np.float32(ANTE_BASE[scaling_of(stake)][min(max(ante, 0), 8)]) * np.float32(x) * np.float32(scale)
    return max(int(np.floor(t)), 1)


# tag: min ante (P_TAGS, in order; 0 = min_ante nil: any ante, ante 0 too)
TAGS = {"Uncommon Tag": 0, "Rare Tag": 0, "Negative Tag": 2, "Foil Tag": 0, "Holographic Tag": 0,
        "Polychrome Tag": 0, "Investment Tag": 0, "Voucher Tag": 0, "Boss Tag": 0, "Standard Tag": 2,
        "Charm Tag": 0, "Meteor Tag": 2, "Buffoon Tag": 2, "Handy Tag": 2, "Garbage Tag": 2,
        "Ethereal Tag": 2, "Coupon Tag": 0, "Double Tag": 0, "Juggle Tag": 0, "D6 Tag": 0, "Top-up Tag": 2,
        "Skip Tag": 0, "Orbital Tag": 2, "Economy Tag": 0}
TAG_PACK = {"Standard Tag": "Mega Standard Pack", "Charm Tag": "Mega Arcana Pack",
            "Meteor Tag": "Mega Celestial Pack", "Buffoon Tag": "Mega Buffoon Pack",
            "Ethereal Tag": "Spectral Pack"}
TAG_EDITION = {"Foil Tag": FOIL, "Holographic Tag": HOLO, "Polychrome Tag": POLY, "Negative Tag": NEGATIVE}
IMMEDIATE = ("Top-up Tag", "Skip Tag", "Garbage Tag", "Handy Tag", "Economy Tag", "Orbital Tag")

# vouchers, in pairs: the second needs the first
VOUCHER_PAIRS = [("Overstock", "Overstock Plus"), ("Clearance Sale", "Liquidation"), ("Hone", "Glow Up"),
                 ("Reroll Surplus", "Reroll Glut"), ("Crystal Ball", "Omen Globe"), ("Telescope", "Observatory"),
                 ("Grabber", "Nacho Tong"), ("Wasteful", "Recyclomancy"), ("Tarot Merchant", "Tarot Tycoon"),
                 ("Planet Merchant", "Planet Tycoon"), ("Seed Money", "Money Tree"), ("Blank", "Antimatter"),
                 ("Magic Trick", "Illusion"), ("Hieroglyph", "Petroglyph"), ("Director's Cut", "Retcon"),
                 ("Paint Brush", "Palette")]
VOUCHER_NEEDS = {b: a for a, b in VOUCHER_PAIRS}


def cost(base, discount=0):
    """Card:set_cost: max(1, floor((base + 0.5) * (100 - discount) / 100))."""
    return max(1, int((base + 0.5) * (100 - discount) // 100))


def discount_of(vouchers):
    return 50 if "Liquidation" in vouchers else 25 if "Clearance Sale" in vouchers else 0


# ---------------------------------------------------------------------------
# STAKE STICKERS — create_card (common_events.lua), Card:set_eternal / set_perishable /
# set_rental / calculate_rental / calculate_perishable (card.lua), written again here
# ---------------------------------------------------------------------------
NOT_ETERNAL = {"Gros Michel", "Ice Cream", "Cavendish", "Luchador", "Turtle Bean", "Diet Cola", "Popcorn",
               "Ramen", "Seltzer", "Mr. Bones", "Invisible Joker"}                  # eternal_compat = false
NOT_PERISHABLE = {"Ceremonial Dagger", "Ride the Bus", "Runner", "Constellation", "Green Joker", "Red Card",
                  "Madness", "Square Joker", "Vampire", "Hologram", "Rocket", "Obelisk", "Lucky Cat",
                  "Flash Card", "Spare Trousers", "Castle", "Glass Joker", "Wee Joker"}  # perishable_compat = false
PERISH_ROUNDS = 5        # G.GAME.perishable_rounds
RENTAL_RATE = 3          # G.GAME.rental_rate


def stickers_ok(name, eternal, perishable, rental, stake):
    """Can a shop / Buffoon pack joker `name` have these stickers at `stake`? One poll gives
    Eternal (> 0.7, stake >= 4, eternal_compat) or else Perishable (0.4 .. 0.7, stake >= 7,
    perishable_compat), never both; another poll gives Rental (> 0.7, stake >= 8, any joker)."""
    if eternal and (perishable or stake < 4 or name in NOT_ETERNAL):
        return False
    if perishable and (stake < 7 or name in NOT_PERISHABLE):
        return False
    return not rental or stake >= 8

