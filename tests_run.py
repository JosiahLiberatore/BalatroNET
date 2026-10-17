"""Test suites for the run loop — repair rules, blind clear / fail, cashout,
the shop (buy / sell / reroll / next), a scripted full run, and a vmapped
random-action fuzz that checks the state invariants.

Staging lives in fixtures.py. Run: python tests_run.py (tests.py also runs it).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from env import (balatroAction, env_step, deal, reset, explain_play, preview, acquire,
                 invariant_errors, blind_target, MAX_HAND_SIZE, MAX_SHOP_SLOTS,
                 V_PLAY, V_DISCARD, V_BUY, V_SELL, V_REROLL, V_NEXT,
                 STAGE_BLIND, STAGE_SHOP, STAGE_LOST, STAGE_WON,
                 BLIND_SMALL, BLIND_BIG, BLIND_BOSS)
from jokers import JOKER_ID, J_EMPTY, MAX_JOKERS, N_JOKERS
from fixtures import state, select, mask_lanes, diff_states, report

J_GREEN = JOKER_ID["Green Joker"]
J_JOKER = JOKER_ID["Joker"]
J_BOOT = JOKER_ID["Bootstraps"]

NO_CARDS = mask_lanes([])


def act(verb, sel=NO_CARDS, target=0):
    if isinstance(sel, str):
        sel = select(sel)
    return balatroAction(verb=jnp.int32(verb), select=sel, target=jnp.int32(target))


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
    c.eq("0-discard DISCARD scores the pair", r, (10 + 10 + 10) * 2)

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
    c.eq("reward = chips scored", r, 60)
    c.eq("done on the clear", d, True)
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

    # a held joker never shows up in the shop: hold three of the four, the
    # only joker left to roll is the fourth, and slot 1 falls back to a Joker
    held = pre._replace(jokers=jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32).at[:3].set(
        jnp.array([J_JOKER, J_GREEN, J_BOOT])))
    got, _, _ = env_step(held, act(V_PLAY, "11000000"))
    c.eq("pool of one: slot 0", got.shop_slots[0], JOKER_ID["Greedy Joker"])
    c.eq("pool of one: slot 1 = fallback Joker", got.shop_slots[1], J_JOKER)

    # every joker held: both unlocked slots fall back, the locked ones stay empty
    held = pre._replace(jokers=jnp.full(MAX_JOKERS, J_EMPTY, jnp.int32).at[:4].set(
        jnp.array([J_JOKER, J_GREEN, J_BOOT, JOKER_ID["Greedy Joker"]])))
    got, _, _ = env_step(held, act(V_PLAY, "11000000"))
    c.eq("empty pool: Joker, Joker, -, -", got.shop_slots, jnp.array([J_JOKER, J_JOKER, -1, -1]))

    # fail: last hand, not enough chips. 2C high card = 5 + 2 = 7.
    pre = state(hand="2C 4H 6S 8D TC 3S 9D QH", hands=1, money=7)
    got, r, d = env_step(pre, act(V_PLAY, "10000000"))
    c.eq("fail: reward still the chips", r, 7)
    c.eq("fail: done", d, True)
    c.eq("fail: stage -> LOST", got.game_stage, STAGE_LOST)
    c.eq("fail: no cashout", got.money, 7)

    # clearing on the LAST hand is a clear, not a fail
    pre = state(hand="KD KC 2C 4H 6S 8D TC 3S", hands=1, chips_scored=290)
    got, _, _ = env_step(pre, act(V_PLAY, "11000000"))
    c.eq("clear on the last hand -> SHOP", got.game_stage, STAGE_SHOP)
    c.eq("clear on the last hand: $3 + 0 hands", got.last_cashout, 3)

    # win: the ante-8 boss
    pre = state(hand="KD KC 2C 4H 6S 8D TC 3S", ante=8, blind_index=BLIND_BOSS, blind_target=50)
    got, _, _ = env_step(pre, act(V_PLAY, "11000000"))
    c.eq("ante-8 boss clear -> WON", got.game_stage, STAGE_WON)
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
    _, r, _ = env_step(pre, act(V_PLAY, sel))
    c.eq("explain score == env_step reward", res.score, r)
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
                c.check(f"round {int(s.round)}: the play clears", bool(d) and int(r) >= 1)
                clears += 1
                want = [3, 4, 5][bi] + hands_left + min(money // 5, 5)
                c.eq(f"round {int(s.round)}: cashout", s.last_cashout, want)
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
        akey, k1, k2, k3 = jax.random.split(akey, 4)
        verb = jax.random.randint(k1, (n_envs,), -1, 7)            # includes illegal verbs
        sel = jax.random.bernoulli(k2, 0.3, (n_envs, MAX_HAND_SIZE))
        target = jax.random.randint(k3, (n_envs,), -2, 34)
        states, r, d = step(states, balatroAction(verb=verb, select=sel, target=target))
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


ALL = [run_repair_test, run_deal_empty_test, run_blind_end_test, run_targets_test,
       run_shop_test, run_explain_test, run_vmap_determinism_test, run_full_run_test,
       run_fuzz_test]


def run_all():
    return all([f() for f in ALL])


if __name__ == "__main__":
    run_all()
