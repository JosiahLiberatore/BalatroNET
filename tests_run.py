"""Test suites for the run loop — repair rules, blind clear / fail, cashout,
the shop (buy / sell / reroll / next), a scripted full run, and a vmapped
random-action fuzz that checks the state invariants.

Staging lives in fixtures.py. Run: python tests_run.py (tests.py also runs it).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from env import (balatroAction, env_step, deal, reset, explain_play, preview, acquire, roll_shop,
                 invariant_errors, blind_target, MAX_HAND_SIZE, MAX_SHOP_SLOTS,
                 V_PLAY, V_DISCARD, V_BUY, V_SELL, V_REROLL, V_NEXT, V_MOVE_JOKER, V_MOVE_CARD,
                 STAGE_BLIND, STAGE_SHOP, STAGE_LOST, STAGE_WON,
                 BLIND_SMALL, BLIND_BIG, BLIND_BOSS)
from jokers import JOKER_ID, J_EMPTY, MAX_JOKERS, N_JOKERS
from fixtures import state, select, mask_lanes, diff_states, report

J_GREEN = JOKER_ID["Green Joker"]
J_JOKER = JOKER_ID["Joker"]
J_BOOT = JOKER_ID["Bootstraps"]

NO_CARDS = mask_lanes([])


def act(verb, sel=NO_CARDS, target=0, dest=0):
    if isinstance(sel, str):
        sel = select(sel)
    return balatroAction(verb=jnp.int32(verb), select=sel, target=jnp.int32(target),
                         dest=jnp.int32(dest))


class Checks:
    """Collect failures for one suite. check() records a line when ok is False."""
    def __init__(self, label):
        self.label, self.failures, self.n = label, [], 0

    def check(self, name, ok, got=None, want=None):
        self.n += 1
        if not bool(ok):
            line = f"FAIL  {self.label}: {name}"
            if got is not None or want is not None:
                line += f"\n      got:  {got}\n      want: {want}"
            self.failures.append(line)

    def eq(self, name, got, want):
        self.check(name, jnp.array_equal(jnp.asarray(got), jnp.asarray(want)), got, want)

    def same_state(self, name, got, want):
        self.failures += diff_states(got, want, f"{self.label}: {name}")
        self.n += 1

    def done(self):
        return report(self.label, self.failures, f"{self.label}: all {self.n} checks passed")


# ---------------------------------------------------------------------------
# REPAIR — an illegal action is a no-op, never an error
# ---------------------------------------------------------------------------
def run_repair_test():
    c = Checks("repair")
    blind = state(hand="4C KD 9H KC 7C 9S 7S 3H", money=20,
                  jokers=("Joker",), shop_slots=("Green Joker",))
    for name, a in [("BUY in a blind", act(V_BUY, target=0)),
                    ("REROLL in a blind", act(V_REROLL)),
                    ("NEXT in a blind", act(V_NEXT)),
                    ("verb 99", act(99, "11")),
                    ("verb -1", act(-1, "11"))]:
        got, r, d = env_step(blind, a)
        c.same_state(name, got, blind)
        c.eq(f"{name} reward", r, 0)
        c.eq(f"{name} done", d, False)

    shop = state(hand="4C KD 9H KC 7C 9S 7S 3H", game_stage=STAGE_SHOP, money=20,
                 shop_slots=("Green Joker", "Joker"))
    for name, a in [("PLAY in the shop", act(V_PLAY, "11")),
                    ("DISCARD in the shop", act(V_DISCARD, "11"))]:
        got, _, _ = env_step(shop, a)
        c.same_state(name, got, shop)

    for stage in (STAGE_LOST, STAGE_WON):
        over = shop._replace(game_stage=jnp.int32(stage))
        for v in range(6):
            got, _, _ = env_step(over, act(v, "11", 0))
            c.same_state(f"stage {stage} verb {v}", got, over)

    # DISCARD with 0 discards left becomes a PLAY: a hand is spent, chips score
    nod = state(hand="4C KD 9H KC 7C 9S 7S 3H", discards=0)
    got, r, _ = env_step(nod, act(V_DISCARD, "01010000"))     # KD KC = pair
    c.eq("0-discard DISCARD spends a hand", got.hands, nod.hands - 1)
    c.eq("0-discard DISCARD scores the pair", got.chips_scored - nod.chips_scored, (10 + 10 + 10) * 2)

    # more than 5 selected: the first 5 valid lanes play
    got, r, _ = env_step(blind._replace(jokers=jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32),
                                        joker_value=jnp.zeros(MAX_JOKERS),
                                        joker_sell_value=jnp.zeros(MAX_JOKERS, jnp.int32)),
                         act(V_PLAY, "11111111"))
    c.eq("8 selected: lanes 5..7 stay in hand (not refilled)",
         got.hand[5:8], blind.hand[5:8])
    return c.done()


# ---------------------------------------------------------------------------
# DEAL at the end of the deck
# ---------------------------------------------------------------------------
def run_deal_empty_test():
    c = Checks("deal empty deck")
    # 10-card deck, 8 dealt, 3 holes: only 2 cards are left to deal
    pre = state(hand="2C 3C . 5C . 7C . 9C", hand_size=8, deck_size=10, deck_ptr=8,
                deck=jnp.concatenate([jnp.array([0, 1, 9, 3, 8, 5, 7, 6, 2, 4]),
                                      jnp.arange(10, 128)]).astype(jnp.int32))
    got = deal(pre)
    c.eq("deck_ptr stops at deck_size", got.deck_ptr, 10)
    c.eq("2 of 3 holes filled", got.hand_valid[:8].sum(), 7)
    c.eq("lane 2 <- deck[8]", got.hand[2], 2)
    c.eq("lane 4 <- deck[9]", got.hand[4], 4)
    c.eq("lane 6 stays a hole", got.hand_valid[6], False)
    c.check("no dead id in a valid lane",
            all(int(h) < 10 for h, v in zip(got.hand[:8], got.hand_valid[:8]) if bool(v)),
            got.hand[:8])
    again = deal(got)
    c.same_state("deal on an empty deck changes nothing", again, got)
    return c.done()


# ---------------------------------------------------------------------------
# BLIND END — clear (cashout + shop), fail, win
# ---------------------------------------------------------------------------
def run_blind_end_test():
    c = Checks("blind end")
    # pair of kings = (10 + 10 + 10) * 2 = 60. 290 + 60 >= 300: clear.
    pre = state(hand="KD KC 2C 4H 6S 8D TC 3S", chips_scored=290, money=12, hands=3)
    got, r, d = env_step(pre, act(V_PLAY, "11000000"))
    cashout = 3 + 2 + 2        # small blind $3 + 2 hands left + interest 12 // 5
    c.eq("chips + 60", got.chips_scored, 350)
    c.eq("reward = the cashout $", r, cashout)
    c.eq("a clear does not end the run", d, False)
    c.eq("stage -> SHOP", got.game_stage, STAGE_SHOP)
    c.eq("last_cashout", got.last_cashout, cashout)
    c.eq("money += cashout", got.money, 12 + cashout)
    c.eq("reroll cost back to base", got.reroll_cost, 5)
    shop = [int(s) for s in got.shop_slots]
    c.check("2 unlocked slots filled", all(0 <= s < N_JOKERS for s in shop[:2]), shop)
    c.check("locked slots empty", all(s == -1 for s in shop[2:]), shop)
    c.check("no duplicate in the shop", shop[0] != shop[1], shop)
    c.check("key advanced", not jnp.array_equal(jax.random.key_data(got.key),
                                                jax.random.key_data(pre.key)))

    # interest caps at $5
    rich = pre._replace(money=jnp.int32(60))
    got, _, _ = env_step(rich, act(V_PLAY, "11000000"))
    c.eq("interest cap: 3 + 2 + 5", got.last_cashout, 10)

    # a held joker never shows up in the shop: three held, 20 rolls, none of
    # the three ever appears
    trio = jnp.array([J_JOKER, J_GREEN, J_BOOT])
    held = pre._replace(jokers=jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32).at[:3].set(trio))
    seen = set()
    for k in range(20):
        got, _, _ = env_step(held._replace(key=jax.random.key(k)), act(V_PLAY, "11000000"))
        seen |= {int(s) for s in got.shop_slots[:2]}
    c.check("held jokers never rolled", not (seen & {int(j) for j in trio}), sorted(seen))

    # the table can hold more jokers than MAX_JOKERS, so the pool tests call
    # roll_shop with a long `held` list. Pool of one: slot 0 = the last joker,
    # slot 1 falls back to a Joker.
    greedy = JOKER_ID["Greedy Joker"]
    all_but = jnp.array([j for j in range(N_JOKERS) if j != greedy], jnp.int32)
    got = roll_shop(jax.random.key(0), all_but, jnp.int32(2))
    c.eq("pool of one: slot 0", got[0], greedy)
    c.eq("pool of one: slot 1 = fallback Joker", got[1], J_JOKER)

    # every joker held: both unlocked slots fall back, the locked ones stay empty
    got = roll_shop(jax.random.key(0), jnp.arange(N_JOKERS, dtype=jnp.int32), jnp.int32(2))
    c.eq("empty pool: Joker, Joker, -, -", got, jnp.array([J_JOKER, J_JOKER, -1, -1]))

    # fail: last hand, not enough chips. 2C high card = 5 + 2 = 7.
    pre = state(hand="2C 4H 6S 8D TC 3S 9D QH", hands=1, money=7)
    got, r, d = env_step(pre, act(V_PLAY, "10000000"))
    c.eq("fail: reward $0", r, 0)
    c.eq("fail: done (the run is over)", d, True)
    c.eq("fail: stage -> LOST", got.game_stage, STAGE_LOST)
    c.eq("fail: no cashout", got.money, 7)

    # clearing on the LAST hand is a clear, not a fail
    pre = state(hand="KD KC 2C 4H 6S 8D TC 3S", hands=1, chips_scored=290)
    got, _, _ = env_step(pre, act(V_PLAY, "11000000"))
    c.eq("clear on the last hand -> SHOP", got.game_stage, STAGE_SHOP)
    c.eq("clear on the last hand: $3 + 0 hands", got.last_cashout, 3)

    # win: the ante-8 boss
    pre = state(hand="KD KC 2C 4H 6S 8D TC 3S", ante=8, blind_index=BLIND_BOSS, blind_target=50)
    got, r, d = env_step(pre, act(V_PLAY, "11000000"))
    c.eq("ante-8 boss clear -> WON", got.game_stage, STAGE_WON)
    c.eq("win: done, reward = the boss cashout", (int(r), bool(d)), (7, True))
    c.eq("boss cashout $5 + 2 hands left", got.last_cashout, 5 + 2)   # fixture hands=3, one spent
    return c.done()


# ---------------------------------------------------------------------------
# SHOP — buy, sell, reroll, next
# ---------------------------------------------------------------------------
def run_shop_test():
    c = Checks("shop")
    shop = state(hand="4C KD 9H KC 7C 9S 7S 3H", game_stage=STAGE_SHOP, money=10,
                 shop_slots=("Green Joker", "Joker"))

    got, _, _ = env_step(shop, act(V_BUY, target=0))          # Green costs 4
    c.eq("buy: joker in slot 0", got.jokers[0], J_GREEN)
    c.eq("buy: Green seeds at 0", got.joker_value[0], 0.0)
    c.eq("buy: sell value 4 // 2", got.joker_sell_value[0], 2)
    c.eq("buy: money -4", got.money, 6)
    c.eq("buy: shop slot emptied", got.shop_slots[0], -1)
    got, _, _ = env_step(got, act(V_BUY, target=1))           # Joker costs 2
    c.eq("buy 2nd: lands in slot 1", got.jokers[:2], jnp.array([J_GREEN, J_JOKER]))
    c.eq("buy 2nd: Joker seeds at 4", got.joker_value[1], 4.0)
    c.eq("buy 2nd: sell value floors at 1", got.joker_sell_value[1], 1)
    c.eq("buy 2nd: money", got.money, 4)

    for name, s, t in [("too poor", shop._replace(money=jnp.int32(3)), 0),
                       ("locked slot", shop, 2),
                       ("target -1", shop, -1),
                       ("target 99", shop, 99),
                       ("empty slot", shop._replace(shop_slots=jnp.array([-1, J_JOKER, -1, -1])), 0),
                       ("joker slots full", state(game_stage=STAGE_SHOP, money=10,
                                                  jokers=("Joker",) * 5,
                                                  shop_slots=("Green Joker",)), 0)]:
        got, _, _ = env_step(s, act(V_BUY, target=t))
        c.same_state(f"buy blocked: {name}", got, s)

    # sell: the jokers to the right shift left, $ = sell value
    three = state(game_stage=STAGE_SHOP, money=0,
                  jokers=("Joker", "Green Joker", "Bootstraps"), joker_value=(4.0, 3.0, 2.0))
    got, _, _ = env_step(three, act(V_SELL, target=1))
    c.eq("sell: compacted", got.jokers[:3], jnp.array([J_JOKER, J_BOOT, J_EMPTY]))
    c.eq("sell: values follow their jokers", got.joker_value[:3], jnp.array([4.0, 2.0, 0.0]))
    c.eq("sell: sell values follow", got.joker_sell_value[:3], jnp.array([1, 3, 0]))
    c.eq("sell: money += 2", got.money, 2)
    for name, t in [("vacant slot", 3), ("target -1", -1), ("target 32", 32)]:
        got, _, _ = env_step(three, act(V_SELL, target=t))
        c.same_state(f"sell blocked: {name}", got, three)
    in_blind = three._replace(game_stage=jnp.int32(STAGE_BLIND))
    got, _, _ = env_step(in_blind, act(V_SELL, target=0))
    c.eq("sell is legal in a blind", got.jokers[0], J_GREEN)

    # move: INSERT the joker in slot i at slot j; value, state and sell value go with it
    three = three._replace(joker_state=jnp.zeros(MAX_JOKERS, jnp.int32).at[1].set(7))
    got, r, _ = env_step(three, act(V_MOVE_JOKER, target=0, dest=2))
    c.eq("move 0->2: inserted", got.jokers[:3], jnp.array([J_GREEN, J_BOOT, J_JOKER]))
    c.eq("move 0->2: values follow", got.joker_value[:3], jnp.array([3.0, 2.0, 4.0]))
    c.eq("move 0->2: joker_state follows", got.joker_state[:3], jnp.array([7, 0, 0]))
    c.eq("move 0->2: sell values follow", got.joker_sell_value[:3], three.joker_sell_value[jnp.array([1, 2, 0])])
    c.eq("move 0->2: money and reward unchanged", (int(got.money), int(r)), (0, 0))
    back, _, _ = env_step(got, act(V_MOVE_JOKER, target=2, dest=0))
    c.same_state("move 2->0 undoes 0->2", back, three)
    got, _, _ = env_step(three, act(V_MOVE_JOKER, target=2, dest=0))
    c.eq("move 2->0: inserted", got.jokers[:3], jnp.array([J_BOOT, J_JOKER, J_GREEN]))
    got, _, _ = env_step(three, act(V_MOVE_JOKER, target=1, dest=2))
    c.eq("move 1->2 = a neighbour swap", got.jokers[:3], jnp.array([J_JOKER, J_BOOT, J_GREEN]))
    for name, i, j in [("src == dst", 1, 1), ("dst vacant", 0, 3), ("src vacant", 3, 0),
                       ("src -1", -1, 0), ("dst -1", 0, -1), ("src 99", 99, 0), ("dst 32", 0, 32)]:
        got, _, _ = env_step(three, act(V_MOVE_JOKER, target=i, dest=j))
        c.same_state(f"move blocked: {name}", got, three)
    got, _, _ = env_step(three._replace(game_stage=jnp.int32(STAGE_BLIND)), act(V_MOVE_JOKER, target=0, dest=1))
    c.eq("move is legal in a blind", got.jokers[0], J_GREEN)
    lost = three._replace(game_stage=jnp.int32(STAGE_LOST))
    got, _, _ = env_step(lost, act(V_MOVE_JOKER, target=0, dest=1))
    c.same_state("move blocked when LOST", got, lost)

    # move card: INSERT the card in lane i at lane j; the play order follows the lanes
    hand = state(hand="4C KD 9H KC 7C 9S 7S 3H")
    got, r, _ = env_step(hand, act(V_MOVE_CARD, target=0, dest=7))
    c.eq("move card 0->7: inserted", got.hand[:8], hand.hand[jnp.array([1, 2, 3, 4, 5, 6, 7, 0])])
    c.eq("move card 0->7: reward 0, deck untouched", (int(r), int(got.deck_ptr)), (0, int(hand.deck_ptr)))
    back, _, _ = env_step(got, act(V_MOVE_CARD, target=7, dest=0))
    c.same_state("move card 7->0 undoes 0->7", back, hand)
    got, _, _ = env_step(hand, act(V_MOVE_CARD, target=5, dest=2))
    c.eq("move card 5->2: inserted", got.hand[:8], hand.hand[jnp.array([0, 1, 5, 2, 3, 4, 6, 7])])
    for name, i, j in [("src == dst", 3, 3), ("dst past hand_size", 0, 8), ("src past hand_size", 8, 0),
                       ("src -1", -1, 2), ("dst -1", 2, -1), ("dst 99", 0, 99)]:
        got, _, _ = env_step(hand, act(V_MOVE_CARD, target=i, dest=j))
        c.same_state(f"move card blocked: {name}", got, hand)
    holed = hand._replace(hand=hand.hand.at[3].set(-1), hand_valid=hand.hand_valid.at[3].set(False))
    got, _, _ = env_step(holed, act(V_MOVE_CARD, target=2, dest=3))
    c.eq("move card into a hole", (int(got.hand[3]), bool(got.hand_valid[2])), (int(hand.hand[2]), False))
    got, _, _ = env_step(holed, act(V_MOVE_CARD, target=3, dest=0))
    c.same_state("move card blocked: src is a hole", got, holed)
    got, _, _ = env_step(holed, act(V_MOVE_CARD, target=0, dest=5))
    c.eq("move card across a hole: the hole shifts too", got.hand_valid[:6],
         jnp.array([True, True, False, True, True, True]))
    c.eq("move card across a hole: the card lands at 5", got.hand[5], hand.hand[0])
    shop_hand = hand._replace(game_stage=jnp.int32(STAGE_SHOP))
    got, _, _ = env_step(shop_hand, act(V_MOVE_CARD, target=0, dest=1))
    c.same_state("move card blocked in the shop", got, shop_hand)

    # acquire: the shared entry point (BUY, and the GUI dev button)
    got, ok = acquire(three, jnp.int32(JOKER_ID["Greedy Joker"]))
    c.eq("acquire: ok", ok, True)
    c.eq("acquire: lands in slot 3", got.jokers[3], JOKER_ID["Greedy Joker"])
    c.eq("acquire: seed 3.0, sell 5 // 2", (float(got.joker_value[3]), int(got.joker_sell_value[3])), (3.0, 2))
    c.eq("acquire: money untouched", got.money, three.money)
    full = state(jokers=("Joker",) * 5)
    got, ok = acquire(full, jnp.int32(J_GREEN))
    c.eq("acquire full: not ok", ok, False)
    c.same_state("acquire full: unchanged", got, full)
    got, ok = acquire(three, jnp.int32(-1))
    c.same_state("acquire -1: unchanged", got, three)

    # reroll
    got, _, _ = env_step(shop, act(V_REROLL))
    c.eq("reroll: money -5", got.money, 5)
    c.eq("reroll: cost +1", got.reroll_cost, 6)
    c.check("reroll: key advanced", not jnp.array_equal(jax.random.key_data(got.key),
                                                        jax.random.key_data(shop.key)))
    got2, _, _ = env_step(got, act(V_REROLL))
    c.same_state("reroll x2 blocked: $5 < $6", got2, got)
    poor = shop._replace(money=jnp.int32(4))
    got, _, _ = env_step(poor, act(V_REROLL))
    c.same_state("reroll blocked: too poor", got, poor)

    # next: small -> big, budget restored, fresh deal, Green keeps its value
    pre = state(hand="4C KD 9H KC 7C 9S 7S 3H", game_stage=STAGE_SHOP, hands=1, discards=0,
                chips_scored=350, round=1, ante=1, blind_index=BLIND_SMALL,
                jokers=("Green Joker",), joker_value=(3.0,), shop_slots=("Joker",))
    got, r, d = env_step(pre, act(V_NEXT))
    c.eq("next: big blind", got.blind_index, BLIND_BIG)
    c.eq("next: target 300 * 1.5", got.blind_target, 450)
    c.eq("next: hands restored", got.hands, 4)
    c.eq("next: discards restored", got.discards, 3)
    c.eq("next: chips reset", got.chips_scored, 0)
    c.eq("next: round + 1", got.round, 2)
    c.eq("next: stage BLIND", got.game_stage, STAGE_BLIND)
    c.eq("next: shop cleared", got.shop_slots, jnp.full(MAX_SHOP_SLOTS, -1))
    c.eq("next: Green keeps 3", got.joker_value[0], 3.0)
    c.eq("next: 8 cards dealt", got.hand_valid.sum(), 8)
    c.eq("next: deck_ptr 8", got.deck_ptr, 8)
    c.eq("next: reward 0, not done", (int(r), bool(d)), (0, False))
    c.check("next: invariants", not invariant_errors(got), invariant_errors(got))

    boss = pre._replace(blind_index=jnp.int32(BLIND_BOSS))
    got, _, _ = env_step(boss, act(V_NEXT))
    c.eq("after the boss: ante 2", got.ante, 2)
    c.eq("after the boss: small blind", got.blind_index, BLIND_SMALL)
    c.eq("after the boss: target 800", got.blind_target, 800)
    return c.done()


def run_targets_test():
    c = Checks("blind targets")
    want = {(1, 0): 300, (1, 1): 450, (1, 2): 600, (2, 0): 800, (3, 2): 4000,
            (8, 0): 50000, (8, 2): 100000}
    for (ante, bi), t in want.items():
        c.eq(f"ante {ante} blind {bi}", blind_target(jnp.int32(ante), bi), t)
    c.eq("reset: ante 1 small = 300", reset(jax.random.key(0), 52, 8).blind_target, 300)
    # curriculum scale: x every target, floor 1 chip; NEXT keeps using it
    c.eq("scale 0.1: ante 1 small = 30", reset(jax.random.key(0), 52, 8, 0.1).blind_target, 30)
    c.eq("scale 0.0: floor of 1 chip", blind_target(jnp.int32(1), 0, 0.0), 1)
    c.eq("scale 0.5: ante 8 boss = 50000", blind_target(jnp.int32(8), BLIND_BOSS, 0.5), 50000)
    shop = state(hand="4C KD", game_stage=STAGE_SHOP, target_scale=0.1)
    got, _, _ = env_step(shop, act(V_NEXT))
    c.eq("NEXT uses the state's scale: big = 45", got.blind_target, 45)
    return c.done()


# ---------------------------------------------------------------------------
# EXPLAIN — the GUI's score readout must be the score env_step gives
# ---------------------------------------------------------------------------
def run_explain_test():
    c = Checks("explain_play")
    pre = state(hand="4C KD 9H KC 7C 9S 7S 3H", jokers=("Green Joker", "Joker"),
                joker_value=(2.0, 4.0))
    sel = select("01011010")
    res = explain_play(pre, sel)
    got, _, _ = env_step(pre, act(V_PLAY, sel))
    c.eq("explain score == env_step chips", res.score, got.chips_scored - pre.chips_scored)
    c.eq("explain mult = 2 + Green 3 + Joker 4", res.mult, 9.0)
    pv = preview(pre, sel)
    c.eq("preview = base hand only", (int(pv.base_chips), int(pv.base_mult)), (54, 2))
    return c.done()


# ---------------------------------------------------------------------------
# A SCRIPTED FULL RUN — every blind, Green Joker held from the first shop
# ---------------------------------------------------------------------------
def best_select(s):
    """The highest-scoring selection of 1..5 valid cards (exhaustive, one vmap)."""
    lanes = [i for i in range(int(s.hand_size)) if bool(s.hand_valid[i])]
    masks = []
    for k in range(1, min(5, len(lanes)) + 1):
        for combo in itertools.combinations(lanes, k):
            m = np.zeros(MAX_HAND_SIZE, bool)
            m[list(combo)] = True
            masks.append(m)
    masks = jnp.asarray(np.stack(masks))
    scores = _batch_explain(s, masks).score
    return masks[int(jnp.argmax(scores))]


_batch_explain = jax.jit(jax.vmap(explain_play, in_axes=(None, 0)))


def run_full_run_test():
    """Drive a whole run. Test staging: each blind's target is cut to 1 chip,
    so every blind clears on its first play and the run must reach WON.

    Policy: on odd rounds discard 2 cards first, then play the best hand.
    Model: Green +1 per play, -1 per discard, floor 0 — checked every step.
    """
    c = Checks("full run")
    s = reset(jax.random.key(7), 52, 8)
    s = s._replace(blind_target=jnp.int32(1))
    green = None          # model of Green's value; None until bought
    clears, steps = 0, 0
    while int(s.game_stage) in (STAGE_BLIND, STAGE_SHOP) and steps < 400:
        steps += 1
        if int(s.game_stage) == STAGE_SHOP:
            if green is None:
                # stage the shop so Green is for sale, and afford it
                s = s._replace(shop_slots=s.shop_slots.at[0].set(J_GREEN),
                               money=jnp.maximum(s.money, 4))
                s, _, _ = env_step(s, act(V_BUY, target=0))
                c.eq("bought Green", s.jokers[0], J_GREEN)
                green = 0.0
            money = int(s.money)
            s, _, _ = env_step(s, act(V_NEXT))
            c.eq("NEXT spends no money", s.money, money)
            s = s._replace(blind_target=jnp.int32(1))
        else:
            if int(s.round) % 2 == 1 and int(s.discards) == int(s.discardcount):
                s, r, _ = env_step(s, act(V_DISCARD, "11"))
                c.eq(f"round {int(s.round)}: discard scores 0", r, 0)
                if green is not None:
                    green = max(green - 1.0, 0.0)
            else:
                money, hands_left = int(s.money), int(s.hands) - 1
                bi = int(s.blind_index)
                s, r, d = env_step(s, act(V_PLAY, best_select(s)))
                if green is not None:
                    green += 1.0
                c.check(f"round {int(s.round)}: the play clears",
                        int(s.game_stage) in (STAGE_SHOP, STAGE_WON))
                clears += 1
                want = [3, 4, 5][bi] + hands_left + min(money // 5, 5)
                c.eq(f"round {int(s.round)}: cashout", s.last_cashout, want)
                c.eq(f"round {int(s.round)}: reward = cashout", r, want)
                c.eq(f"round {int(s.round)}: done only on the win", d, int(s.game_stage) == STAGE_WON)
        if green is not None:
            c.eq(f"step {steps}: Green value", s.joker_value[0], green)
        errs = invariant_errors(s)
        c.check(f"step {steps}: invariants", not errs, errs)
    c.eq("run ends WON", s.game_stage, STAGE_WON)
    c.eq("24 blinds cleared", clears, 24)
    c.eq("ends at ante 8", s.ante, 8)
    return c.done()


# ---------------------------------------------------------------------------
# FUZZ — random actions on a vmapped batch, invariants on every state
# ---------------------------------------------------------------------------
def _host(batch):
    """Device batch -> numpy batch with the PRNG key as raw words."""
    return jax.device_get(batch._replace(key=jax.random.key_data(batch.key)))


def run_fuzz_test(n_envs=128, n_steps=120, seed=0):
    c = Checks("fuzz")
    keys = jax.random.split(jax.random.key(seed), n_envs)
    states = jax.vmap(lambda k: reset(k, 52, 8))(keys)
    # half the batch gets 1-chip targets so the shop gets real traffic
    easy = jnp.arange(n_envs) % 2 == 0
    states = states._replace(blind_target=jnp.where(easy, 1, states.blind_target),
                             money=jnp.where(easy, 40, states.money))
    step = jax.jit(jax.vmap(env_step))
    akey = jax.random.key(seed + 1)
    seen_stage = np.zeros(4, int)
    for t in range(n_steps):
        akey, k1, k2, k3, k4 = jax.random.split(akey, 5)
        verb = jax.random.randint(k1, (n_envs,), -1, 9)            # includes illegal verbs
        sel = jax.random.bernoulli(k2, 0.3, (n_envs, MAX_HAND_SIZE))
        target = jax.random.randint(k3, (n_envs,), -2, 34)
        dest = jax.random.randint(k4, (n_envs,), -2, 10)
        states, r, d = step(states, balatroAction(verb=verb, select=sel, target=target, dest=dest))
        # keep the easy half easy after a NEXT
        states = states._replace(blind_target=jnp.where(easy, 1, states.blind_target))
        host = _host(states)
        for i in range(n_envs):
            one = jax.tree.map(lambda x: x[i], host)
            seen_stage[int(one.game_stage)] += 1
            errs = invariant_errors(one)
            if errs:
                c.check(f"step {t} env {i}", False, errs)
        c.check(f"step {t}: reward >= 0", bool((r >= 0).all()), r)
    c.check("the fuzz reached the shop", seen_stage[STAGE_SHOP] > 0, seen_stage)
    c.check("the fuzz reached LOST", seen_stage[STAGE_LOST] > 0, seen_stage)
    print(f"  fuzz stage visits (BLIND, SHOP, LOST, WON): {seen_stage.tolist()}")
    return c.done()


def run_vmap_determinism_test():
    c = Checks("vmap + determinism")
    a = state(hand="4C KD 9H KC 7C 9S 7S 3H", jokers=("Green Joker",))
    b = state(hand="KD KC 2C 4H 6S 8D TC 3S", chips_scored=290, money=12)
    acts = [act(V_PLAY, "01011010"), act(V_PLAY, "11000000")]
    batch = jax.tree.map(lambda x, y: jnp.stack([x, y]), a, b)
    abatch = jax.tree.map(lambda x, y: jnp.stack([x, y]), *acts)
    got, r, d = jax.vmap(env_step)(batch, abatch)
    for i, (s, a_) in enumerate([(a, acts[0]), (b, acts[1])]):
        one, r1, d1 = env_step(s, a_)
        c.same_state(f"vmap lane {i} == single step", jax.tree.map(lambda x: x[i], got), one)
        c.eq(f"vmap lane {i} reward", r[i], r1)
    s1 = reset(jax.random.key(3), 52, 8)
    s2 = reset(jax.random.key(3), 52, 8)
    for _ in range(5):
        s1, _, _ = env_step(s1, act(V_DISCARD, "111"))
        s2, _, _ = env_step(s2, act(V_DISCARD, "111"))
    c.same_state("same key + same actions -> same state", s1, s2)
    return c.done()


def run_legal_mask_test(n_envs=64, n_steps=60, seed=5):
    """agent.legal_mask must mirror env_step: a legal choice changes the state,
    a masked one is a no-op. Checked on every state of a random-play batch.
    One documented exception: DISCARD with 0 discards left is masked, but the
    env repairs it to a PLAY (it is a duplicate of PLAY, not a no-op).
    The MOVE verbs are checked per (src, dst) pair against move_masks / dst_mask
    (pairs over the first NP slots; a run never holds more than 8 cards or 5 jokers)."""
    from agent import (legal_mask, move_masks, dst_mask, decode, N_ACTIONS, A_DISCARD,
                       A_MOVE_JOKER, A_MOVE_CARD, action_name)
    NP = 10
    c = Checks("legal mask")
    states = jax.vmap(lambda k: reset(k, 52, 8, 0.05))(jax.random.split(jax.random.key(seed), n_envs))
    states = states._replace(money=jnp.full(n_envs, 12, jnp.int32))
    sel = jnp.zeros(MAX_HAND_SIZE, bool).at[:2].set(True)

    def changed(a, b):
        return jnp.any(jnp.stack([jnp.any(x != y) for x, y in
                                  zip(jax.tree.leaves(a._replace(key=jnp.int32(0))),
                                      jax.tree.leaves(b._replace(key=jnp.int32(0))))]))

    pairs = jnp.array([(i, j) for i in range(NP) for j in range(NP)], jnp.int32)

    @jax.jit
    def probe(s):
        def one(a):
            nxt, _, _ = env_step(s, decode(a, sel))
            return changed(nxt, s)

        def one_pair(a, ij):
            nxt, _, _ = env_step(s, decode(a, sel, ij[0], ij[1]))
            return changed(nxt, s)

        legal, mm = legal_mask(s), move_masks(s)
        def pair_legal(a, row):
            ok = jax.vmap(lambda ij: mm[row, ij[0]] & dst_mask(mm[row + 1], ij[0])[ij[1]])(pairs)
            return legal[a] & ok
        pl = jnp.stack([pair_legal(A_MOVE_JOKER, 0), pair_legal(A_MOVE_CARD, 2)])
        pm = jnp.stack([jax.vmap(lambda ij: one_pair(A_MOVE_JOKER, ij))(pairs),
                        jax.vmap(lambda ij: one_pair(A_MOVE_CARD, ij))(pairs)])
        return legal, jax.vmap(one)(jnp.arange(N_ACTIONS)), pl, pm

    @jax.jit
    def advance(s, key):
        legal = legal_mask(s)
        k1, k2, k3 = jax.random.split(key, 3)
        a = jax.random.categorical(k1, jnp.where(legal, 0.0, -1e9))
        mm = move_masks(s)
        row = jnp.where(a == A_MOVE_JOKER, 0, 2)
        src = jax.random.categorical(k2, jnp.where(mm[row], 0.0, -1e9))
        dst = jax.random.categorical(k3, jnp.where(dst_mask(mm[row + 1], src), 0.0, -1e9))
        nxt, _, _ = env_step(s, decode(a, sel, src, dst))
        over = (nxt.game_stage == STAGE_LOST) | (nxt.game_stage == STAGE_WON)
        return jax.tree.map(lambda x, y: jnp.where(over, x, y), s, nxt)   # stay live

    key = jax.random.key(seed + 1)
    bad = 0
    for t in range(n_steps):
        legal, moved, pl, pm = jax.device_get(jax.vmap(probe)(states))
        no_discards = np.asarray(states.discards) <= 0
        for i in range(n_envs):
            for k, a in enumerate((A_MOVE_JOKER, A_MOVE_CARD)):
                if bool(legal[i, a]) != bool(pm[i, k].any()) and bad < 10:
                    bad += 1
                    c.check(f"step {t} env {i}: {action_name(a)} legal={bool(legal[i, a])} "
                            f"but some pair changes={bool(pm[i, k].any())}", False)
                for p in np.nonzero(pl[i, k] != pm[i, k])[0][:max(0, 10 - bad)]:
                    bad += 1
                    c.check(f"step {t} env {i}: {action_name(a, *pairs[p])} legal={bool(pl[i, k, p])} "
                            f"but changed={bool(pm[i, k, p])}", False)
            for a in range(N_ACTIONS):
                if (a == A_DISCARD and no_discards[i]) or a in (A_MOVE_JOKER, A_MOVE_CARD):
                    continue
                if bool(legal[i, a]) != bool(moved[i, a]) and bad < 10:
                    bad += 1
                    c.check(f"step {t} env {i}: {action_name(a)} legal={bool(legal[i, a])} "
                            f"but changed={bool(moved[i, a])}", False)
        key, sub = jax.random.split(key)
        states = jax.vmap(advance)(states, jax.random.split(sub, n_envs))
    c.check("no legal/no-op mismatch", bad == 0)
    return c.done()


def run_move_reach_test(n_perms=40, seed=3):
    """Any order is reachable with at most n - 1 INSERT moves: for place p = 0, 1, ...
    insert the wanted item at p. Jokers (5 slots) and hand cards (8 lanes)."""
    c = Checks("move reach")
    rng = np.random.default_rng(seed)
    names = ("Joker", "The Duo", "Bull", "Banner", "Green Joker")
    base_j = state(game_stage=STAGE_SHOP, jokers=names, joker_value=(4.0, 2.0, 2.0, 30.0, 3.0))
    base_c = state(hand="4C KD 9H KC 7C 9S 7S 3H")
    step = jax.jit(env_step)
    for kind, base, arr, verb, n in (("jokers", base_j, "jokers", V_MOVE_JOKER, 5),
                                     ("cards", base_c, "hand", V_MOVE_CARD, 8)):
        start = [int(x) for x in getattr(base, arr)[:n]]
        worst = 0
        for t in range(n_perms):
            want = [start[i] for i in rng.permutation(n)]
            s, cur, steps = base, list(start), 0
            for p in range(n):
                src = cur.index(want[p], p)
                if src != p:
                    cur.insert(p, cur.pop(src))
                    s, _, _ = step(s, act(verb, target=src, dest=p))
                    steps += 1
            worst = max(worst, steps)
            c.eq(f"{kind} perm {t}: reached", getattr(s, arr)[:n], jnp.array(want))
        c.check(f"{kind}: at most {n - 1} moves (worst {worst})", worst <= n - 1, worst)
    return c.done()


ALL = [run_repair_test, run_deal_empty_test, run_blind_end_test, run_targets_test,
       run_shop_test, run_explain_test, run_vmap_determinism_test, run_full_run_test,
       run_move_reach_test, run_fuzz_test, run_legal_mask_test]


def run_all():
    return all([f() for f in ALL])


if __name__ == "__main__":
    run_all()
