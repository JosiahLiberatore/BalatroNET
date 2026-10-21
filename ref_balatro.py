"""The test reference — Balatro 1.0.1o rules for the authored jokers, in plain Python.

A second implementation, ported line by line from the game's Lua
(card.lua calculate_joker / calculate_dollar_bonus, state_events.lua
evaluate_play / end_round, misc_functions.lua evaluate_poker_hand). It does not
read jokers.py or scoring.py: the joker numbers, the copy rules, the hand
detection and the scoring order are all written again here.

Random draws are the one input it takes from the env: the env exposes the
1-in-N coin flips and Misprint's draw (PlayOut.rolls / hrolls / rnd), and the
reference consumes them by (card, repetition, slot). Everything else is
recomputed.

Card ids: rank = id % 13 (0 = two .. 12 = ace), suit = id // 13 (C D H S).
"""

from collections import Counter

TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(13)
CLUBS, DIAMONDS, HEARTS, SPADES = range(4)
NOMINAL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]      # base.nominal
HAND_ORDER = ["straight_flush", "four_kind", "full_house", "flush", "straight",
              "three_kind", "two_pair", "pair", "high_card"]
HAND_BASE = {"straight_flush": (100, 8), "four_kind": (60, 7), "full_house": (40, 4),
             "flush": (35, 4), "straight": (30, 4), "three_kind": (30, 3),
             "two_pair": (20, 2), "pair": (10, 2), "high_card": (5, 1)}
BLIND_PAY = [3, 4, 5]
WIN_ANTE = 8

# game.lua P_CENTERS: rarity 2 (Uncommon) among the authored jokers (Baseball Card)
UNCOMMON = {"Bootstraps", "Fibonacci", "Arrowhead", "Onyx Agate", "Bull", "Joker Stencil", "Cloud 9",
            "Spare Trousers", "Castle", "Flash Card", "Card Sharp", "Ramen", "Acrobat",
            "Loyalty Card", "Seeing Double", "Flower Pot", "Erosion", "Blackboard", "Rocket",
            "Gift Card", "To the Moon", "Merry Andy", "Troubadour", "Turtle Bean", "Burglar",
            "Dusk", "Hack", "Sock and Buskin", "Seltzer", "Mime", "The Idol", "Bloodstone",
            "Rough Gem", "Four Fingers", "Shortcut", "Smeared Joker", "Pareidolia", "Showman",
            "Oops! All 6s"}
# config values the reference needs to seed a bought joker (game.lua config)
SEED = {"Green Joker": 0, "Popcorn": 20, "Ride the Bus": 0, "Spare Trousers": 0, "Square Joker": 0,
        "Runner": 0, "Ice Cream": 100, "Wee Joker": 0, "Castle": 0, "Flash Card": 0,
        "Obelisk": 1, "Hit the Road": 1, "Ramen": 2, "Campfire": 1, "Rocket": 1,
        "Turtle Bean": 5, "Seltzer": 10}
H_SIZE = {"Juggler": 1, "Merry Andy": -1, "Troubadour": 2, "Stuntman": -2}
D_SIZE = {"Drunkard": 1, "Merry Andy": 3}
H_PLAYS = {"Troubadour": -1}
PROBABLE = {"Bloodstone", "Business Card", "Reserved Parking"}     # 1 in 2 per card
COPIERS = ("Blueprint", "Brainstorm")


def rank(c): return c % 13
def suit(c): return c // 13
def red(s): return s in (DIAMONDS, HEARTS)


def mods_of(names):
    return dict(ff="Four Fingers" in names, sc="Shortcut" in names, sm="Smeared Joker" in names,
                sp="Splash" in names, pa="Pareidolia" in names, showman="Showman" in names,
                odds=2 ** names.count("Oops! All 6s"), credit=names.count("Credit Card"),
                moon=names.count("To the Moon"), chaos=names.count("Chaos the Clown"))


def is_suit(c, s, m):
    return suit(c) == s or (m["sm"] and red(suit(c)) == red(s))


def is_face(c, m):
    return m["pa"] or rank(c) in (JACK, QUEEN, KING)


# ---------------------------------------------------------------------------
# HAND DETECTION — evaluate_poker_hand / get_poker_hand_info
# ---------------------------------------------------------------------------
def get_flush(hand, m):
    need = 4 if m["ff"] else 5
    if len(hand) > 5 or len(hand) < need:
        return []
    for s in (SPADES, HEARTS, CLUBS, DIAMONDS):
        t = [c for c in hand if is_suit(c, s, m)]
        if len(t) >= need:
            return [t]
    return []


def get_straight(hand, m):
    need = 4 if m["ff"] else 5
    if len(hand) > 5 or len(hand) < need:
        return []
    ids = {}
    for c in hand:
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
    cnt = Counter(rank(c) for c in hand)
    return [[c for c in hand if rank(c) == r] for r in sorted(cnt, reverse=True) if cnt[r] == num]


def ref_hand(cards, m):
    """-> (hand name, the hand names it contains, the scored cards in lane order)."""
    p5, p4, p3, p2 = (get_x_same(n, cards) for n in (5, 4, 3, 2))
    fl, st = get_flush(cards, m), get_straight(cards, m)
    res = {h: [] for h in HAND_ORDER}
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
        res["high_card"] = [[max(cards, key=rank)]]
    if res["four_kind"]:
        res["three_kind"] = [res["four_kind"][0][:3]]
    if res["three_kind"]:
        res["pair"] = [res["three_kind"][0][:2]]
    name = next((h for h in HAND_ORDER if res[h]), "high_card")
    scored = set(res[name][0]) if res[name] else set()
    if name == "three_kind":
        scored = set(p3[0])
    if name == "pair":
        scored = set(p2[0])
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


# ---------------------------------------------------------------------------
# A PLAY — evaluate_play
# ---------------------------------------------------------------------------
def ref_play(J, cards, held, g, rolls=None, hrolls=None, rnd=None, mods=None):
    """Score one PLAY.
    J      : list of joker records {name, v, st, sell}, slot order (mutated: counters)
    cards  : played card ids, lane order (at most 5)
    held   : (lane, card id) of the cards left in hand
    g      : dict money, hands_left (after this hand), discards, deck_left, slots, nines,
             deck_size, played (Counter, before), round (Counter, before), targets
             (idol, mail, ancient, castle), is_boss
    rolls  : fn (card index, rep, slot) -> bool for the 1-in-N draws (default: all hit)
    -> dict(hand, chips, mult, score, money, gone (set of slots used up))"""
    roll = rolls or (lambda k, r, j: True)
    hroll = hrolls or (lambda k, r, j: True)
    names = [x["name"] for x in J]
    m = mods if mods is not None else mods_of(names)
    name, has, scored = ref_hand(cards, m)
    played, rnd_cnt = Counter(g["played"]), Counter(g["round"])
    hands_total = sum(played.values())
    played[name] += 1
    rnd_cnt[name] += 1
    money = g["money"]
    tg = g["targets"]
    eff = [copy_target(names, i) for i in range(len(J))]
    acted = set()

    # BEFORE
    faces = any(is_face(c, m) for c in scored)
    for i, x in enumerate(J):
        t = eff[i]
        if t is None:
            continue
        y, bp = J[t], t != i
        n = y["name"]
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
            y["v"] = 0 if faces else y["v"] + 1
        elif n == "Obelisk":
            most = all(played[k] < played[name] for k in HAND_ORDER if k != name)
            y["v"] = 1 if most else y["v"] + 0.2
        elif n == "Green Joker":
            y["v"] += 1

    chips, mult = HAND_BASE[name]
    pos = {c: k for k, c in enumerate(cards)}
    first_face = next((c for c in scored if is_face(c, m)), None)
    # scored cards, in order
    for c in scored:
        reps = 1
        for i in range(len(J)):
            t = eff[i]
            if t is None:
                continue
            n = J[t]["name"]
            before = reps
            if n == "Sock and Buskin" and is_face(c, m):
                reps += 1
            elif n == "Hanging Chad" and c == scored[0]:
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
            chips += NOMINAL[rank(c)]
            for i in range(len(J)):
                t = eff[i]
                if t is None:
                    continue
                y, bp = J[t], t != i
                n = y["name"]
                e = {}
                if n == "Wee Joker" and rank(c) == TWO and not bp:
                    y["v"] += 8
                elif n == "Photograph" and c == first_face:
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
                elif n == "Business Card" and is_face(c, m) and roll(pos[c], r, i):
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
                elif n == "Bloodstone" and is_suit(c, HEARTS, m) and roll(pos[c], r, i):
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

    # cards held in hand
    lo = None
    for lane, c in held:                     # Raised Fist: the last card with the lowest id
        if lo is None or rank(c) <= rank(lo[1]):
            lo = (lane, c)
    for lane, c in held:
        reps, k = 1, 0
        while k < reps:
            any_fx = False
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
                for i in range(len(J)):
                    t = eff[i]
                    if t is not None and J[t]["name"] == "Mime":
                        reps += 1
                        acted.add(i)
            k += 1

    # joker main effects, then Baseball Card on each Uncommon joker
    held_cards = [c for _, c in held]
    n_jokers = len(J)
    for i, x in enumerate(J):
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
            elif n in ("Obelisk", "Hit the Road", "Ramen", "Campfire") and v > 1:
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
                for c in scored:
                    for s in (HEARTS, DIAMONDS, SPADES, CLUBS):
                        if is_suit(c, s, m) and s not in got:
                            got.add(s)
                            break
                if len(got) == 4:
                    xm = 3
            elif n == "Seeing Double":
                cnt = {s: sum(is_suit(c, s, m) for c in scored) for s in range(4)}
                if cnt[CLUBS] > 0 and (cnt[HEARTS] or cnt[DIAMONDS] or cnt[SPADES]):
                    xm = 2
            elif n in ("Wee Joker", "Castle", "Square Joker", "Runner", "Ice Cream") and v > 0:
                add_c = v
            elif n == "Blue Joker" and g["deck_left"] > 0:
                add_c = 2 * g["deck_left"]
            elif n == "Erosion" and 52 - g["deck_size"] > 0:
                add_m = 4 * (52 - g["deck_size"])
            elif n == "Bull" and money > 0:
                add_c = 2 * money
            elif n == "Blackboard" and all(suit(c) in (CLUBS, SPADES) for c in held_cards):
                xm = 3
            elif n == "Swashbuckler":
                add_m = sum(z["sell"] for k, z in enumerate(J) if k != t)
            elif n == "Joker":
                add_m = 4
            elif n in ("Spare Trousers", "Ride the Bus", "Flash Card", "Popcorn", "Green Joker") and v > 0:
                add_m = v
            elif n == "Gros Michel":
                add_m = 15
            elif n == "Cavendish":
                xm = 3
            elif n == "Card Sharp" and rnd_cnt[name] > 1:
                xm = 3
            elif n == "Bootstraps" and money // 5 >= 1:
                add_m = 2 * (money // 5)
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

    # AFTER
    gone = set()
    for i, x in enumerate(J):
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
    return dict(hand=name, chips=chips, mult=mult, score=int(chips * mult // 1), money=money - g["money"],
                gone=gone, played=played, round=rnd_cnt, acted=acted)


def ref_discard(J, cards, g):
    """One DISCARD of cards (lane order). Mutates the counters. -> (money, gone slots)."""
    names = [x["name"] for x in J]
    m = mods_of(names)
    tg = g["targets"]
    eff = [copy_target(names, i) for i in range(len(J))]
    money, gone = 0, set()
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
            elif n == "Mail-In Rebate" and rank(c) == tg["mail"]:
                money += 5
            elif n == "Hit the Road" and not bp and rank(c) == JACK:
                y["v"] += 0.5
            elif n == "Green Joker" and not bp and last:
                y["v"] = max(0, y["v"] - 1)
            elif n == "Faceless Joker" and last and sum(is_face(z, m) for z in cards) >= 3:
                money += 5
    return money, gone


def ref_end_of_round(J, g):
    """end_round jokers (real slots only). Mutates. -> (sell value adds, gone slots, to-do slots)."""
    gone = set()
    for i, x in enumerate(J):
        n = x["name"]
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
                z["sell"] += 1
        elif n == "Hit the Road" and x["v"] > 1:
            x["v"] = 1
    return gone


def ref_dollar_bonus(J, g):
    """calculate_dollar_bonus over the held jokers (copies never pay)."""
    out = 0
    for x in J:
        n = x["name"]
        if n == "Golden Joker":
            out += 4
        elif n == "Cloud 9":
            out += g["nines"]
        elif n == "Rocket":
            out += x["v"]
        elif n == "Delayed Gratification" and g["discards_used"] == 0 and g["discards"] > 0:
            out += 2 * g["discards"]
    return out


def ref_budget(J, base_hand, base_hands, base_discards):
    """-> (hand size, hands per round, discards per round)."""
    h = base_hand + sum(H_SIZE.get(x["name"], 0) for x in J) \
        + sum(int(x["v"]) for x in J if x["name"] == "Turtle Bean")
    return (min(max(h, 1), 32), max(base_hands + sum(H_PLAYS.get(x["name"], 0) for x in J), 1),
            max(base_discards + sum(D_SIZE.get(x["name"], 0) for x in J), 0))
