"""Play BalatroNET by hand — a pygame front end over the jitted env.

Every game action goes through env_step, the same function the agent trains
on. The GUI keeps no game rules of its own: it reads the state, draws it,
and turns clicks into balatroAction values. The only exceptions are the dev
buttons (give money / give a joker), which stage the state directly, the way
fixtures.py does in the tests.

    uv run --project D:/Claude/jax-learn python gui.py [--seed N]

Mouse: click cards to select, click a joker to select it (then SELL), click a
shop joker to buy it.
Keys:  1-8 toggle a card   Enter play   D discard   R / S sort by rank / suit
       N new run   Esc quit

Sorting changes the DISPLAY order only. The env plays the selected cards in
lane order; no joker in the table reads the play order yet.
"""

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pygame

from env import (balatroAction, env_step, reset, explain_play, preview, build_ctx, acquire,
                 invariant_errors, MAX_HAND_SIZE, STAGE_BLIND, STAGE_SHOP, STAGE_LOST,
                 STAGE_WON, BLIND_NAMES, BLIND_REWARD, V_PLAY, V_DISCARD, V_BUY, V_SELL,
                 V_REROLL, V_NEXT, WIN_ANTE, INTEREST_PER, INTEREST_CAP, RANK_CHARS)
from jokers import (JOKER_NAMES, JOKER_DESC, JOKER_COST, JOKERS, JOKER_RARITY, J_EMPTY,
                    MAX_JOKERS, rows_for, resolve, COMMON, UNCOMMON, RARE,
                    E_ADD_MULT, E_ADD_CHIPS, E_XMULT, E_ADD_MONEY, EFFECT_NAMES)
from scoring import HAND_NAMES

W, H = 1280, 760

# --- colors (close to the Balatro palette) -----------------------------------
BG        = (36, 74, 60)
PANEL     = (28, 36, 40)
PANEL_HI  = (48, 58, 64)
TEXT      = (238, 238, 238)
DIM       = (150, 160, 165)
CHIPS_C   = (0, 147, 255)
MULT_C    = (254, 95, 85)
MONEY_C   = (243, 185, 68)
CARD_C    = (245, 242, 235)
CARD_EDGE = (190, 185, 175)
SEL_EDGE  = (255, 210, 80)
SUIT_C    = [(35, 89, 85), (240, 107, 63), (240, 52, 100), (60, 67, 104)]   # C D H S
SUIT_GLYPH = ["\u2663", "\u2666", "\u2665", "\u2660"]
RARITY_C  = {COMMON: (0, 147, 255), UNCOMMON: (53, 189, 134), RARE: (254, 95, 85)}
BTN_PLAY  = (0, 147, 255)
BTN_DISC  = (254, 95, 85)
BTN_SHOP  = (243, 140, 50)
BTN_GREEN = (53, 189, 134)

HAND_LABEL = {"straight_flush": "Straight Flush", "four_kind": "Four of a Kind",
              "full_house": "Full House", "flush": "Flush", "straight": "Straight",
              "three_kind": "Three of a Kind", "two_pair": "Two Pair", "pair": "Pair",
              "high_card": "High Card"}

def effect_label(jid, v):
    """The live value v, worded by the joker's EFFECT axis."""
    e = int(JOKERS.EFFECT[jid])
    if e == E_ADD_MULT:
        return f"+{v:g} Mult", MULT_C
    if e == E_ADD_CHIPS:
        return f"+{v:g} Chips", CHIPS_C
    if e == E_XMULT:
        return f"X{v:g} Mult", MULT_C
    if e == E_ADD_MONEY:
        return f"+${v:g}", MONEY_C
    return f"{EFFECT_NAMES[e]} {v:g}", DIM


# ---------------------------------------------------------------------------
# jitted helpers
# ---------------------------------------------------------------------------
@jax.jit
def live_values(state):
    """v per joker slot, as scoring would read it right now (resolve pass)."""
    rows = rows_for(state.jokers)
    return resolve(rows, state.jokers != J_EMPTY, state.joker_value, build_ctx(state))


_acquire = jax.jit(acquire)


def host(state):
    """Device state -> numpy copy for drawing (one transfer per step, not per frame)."""
    return jax.device_get(state._replace(key=jax.random.key_data(state.key)))


def hand_label(i):
    return HAND_LABEL.get(HAND_NAMES[int(i)], HAND_NAMES[int(i)])


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------
class Fonts:
    def __init__(self):
        self.small = pygame.font.SysFont("arial", 15)
        self.body = pygame.font.SysFont("arial", 18)
        self.bold = pygame.font.SysFont("arial", 20, bold=True)
        self.big = pygame.font.SysFont("arial", 30, bold=True)
        self.huge = pygame.font.SysFont("arial", 46, bold=True)
        self.rank = pygame.font.SysFont("arial", 24, bold=True)
        self.glyph = pygame.font.SysFont("segoeuisymbol", 44)
        self.glyph_small = pygame.font.SysFont("segoeuisymbol", 20)


def text(surf, font, s, color, pos, anchor="topleft"):
    img = font.render(str(s), True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def wrap(font, s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def box(surf, rect, color, radius=10, border=None, bw=3):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, bw, border_radius=radius)


class Button:
    def __init__(self, rect, label, color, enabled=True):
        self.rect, self.label, self.color, self.enabled = pygame.Rect(rect), label, color, enabled

    def draw(self, surf, fonts):
        c = self.color if self.enabled else (80, 86, 90)
        box(surf, self.rect, c, 10)
        text(surf, fonts.bold, self.label, TEXT if self.enabled else DIM, self.rect.center, "center")

    def hit(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, seed):
        pygame.init()
        pygame.display.set_caption("BalatroNET")
        self.screen = pygame.display.set_mode((W, H))
        self.fonts = Fonts()
        self.clock = pygame.time.Clock()
        self.seed = seed
        self.sort_by = "rank"
        self.log = []
        self.splash("Compiling the jitted env ...")
        self.new_run(seed)
        # warm the jit caches so the first click does not stall
        explain_play(self.state, jnp.zeros(MAX_HAND_SIZE, bool))
        preview(self.state, jnp.zeros(MAX_HAND_SIZE, bool))
        live_values(self.state)
        env_step(self.state, self.action(V_NEXT))

    # --- state management ---------------------------------------------------
    def splash(self, msg):
        self.screen.fill(BG)
        text(self.screen, self.fonts.big, msg, TEXT, (W // 2, H // 2), "center")
        pygame.display.flip()

    def new_run(self, seed):
        self.seed = seed
        self.state = reset(jax.random.key(seed), 52, 8)
        self.selected = set()          # lanes
        self.joker_sel = None          # joker slot
        self.last_play = None          # (PlayResult on host, cards, pre-state hands)
        self.cashout = None            # breakdown text
        self.refresh()
        self.say(f"New run, seed {seed}. {BLIND_NAMES[0]}: score {int(self.h.blind_target)} chips.")

    def refresh(self):
        self.h = host(self.state)
        self.v = np.asarray(live_values(self.state))
        self.preview = None
        if self.selected:
            self.preview = jax.device_get(preview(self.state, self.sel_mask()))
        errs = invariant_errors(self.h)
        if errs:
            self.say("INVARIANT BROKEN: " + "; ".join(errs))
            print("INVARIANT BROKEN:", errs, file=sys.stderr)

    def say(self, msg):
        self.log.append(msg)
        self.log = self.log[-9:]
        print(msg)

    def sel_mask(self):
        m = np.zeros(MAX_HAND_SIZE, bool)
        m[list(self.selected)] = True
        return jnp.asarray(m)

    def action(self, verb, target=0):
        return balatroAction(verb=jnp.int32(verb), select=self.sel_mask(), target=jnp.int32(target))

    def step(self, verb, target=0):
        pre = self.h
        a = self.action(verb, target)
        if verb == V_PLAY:
            res = jax.device_get(explain_play(self.state, a.select))
            cards = [int(c) for c, ok in zip(res.cards, res.valid) if ok]
            self.last_play = (res, cards)
        self.state, reward, done = env_step(self.state, a)
        self.selected.clear()
        self.refresh()
        h = self.h
        if verb == V_PLAY:
            res, _ = self.last_play
            self.say(f"Played {hand_label(res.handtype)}: {float(res.chips):g} x {float(res.mult):g}"
                     f" = {int(reward)}")
        elif verb == V_DISCARD:
            self.say(f"Discarded. {int(h.discards)} discard(s) left.")
        elif verb == V_BUY:
            if int(h.money) != int(pre.money):
                self.say(f"Bought {JOKER_NAMES[int(pre.shop_slots[target])]} for ${int(pre.money) - int(h.money)}.")
            else:
                self.say("Can't buy that (money, or no free joker slot).")
        elif verb == V_SELL:
            self.say(f"Sold {JOKER_NAMES[int(pre.jokers[target])]} for ${int(h.money) - int(pre.money)}.")
            self.joker_sel = None
        elif verb == V_REROLL:
            self.say(f"Rerolled for ${int(pre.money) - int(h.money)}." if int(h.money) != int(pre.money)
                     else "Not enough money to reroll.")
        elif verb == V_NEXT:
            self.last_play = None
            self.cashout = None
            self.say(f"Ante {int(h.ante)}, {BLIND_NAMES[int(h.blind_index)]}: "
                     f"score {int(h.blind_target)} chips.")

        if int(pre.game_stage) == STAGE_BLIND and int(h.game_stage) in (STAGE_SHOP, STAGE_WON):
            interest = min(int(pre.money) // INTEREST_PER, INTEREST_CAP)
            blind = int(BLIND_REWARD[int(pre.blind_index)])
            self.cashout = (f"Cash out ${int(h.last_cashout)}:  blind ${blind}  +  "
                            f"{int(h.hands)} hand(s) left ${int(h.hands)}  +  interest ${interest}")
            self.say(f"Blind defeated! {self.cashout}")
        if int(h.game_stage) == STAGE_LOST and int(pre.game_stage) != STAGE_LOST:
            self.say("Out of hands. Game over.")
        if int(h.game_stage) == STAGE_WON and int(pre.game_stage) != STAGE_WON:
            self.say("You won the run!")

    # --- dev staging (bypasses env_step on purpose, like fixtures.py) -------
    def give_money(self, n):
        self.state = self.state._replace(money=self.state.money + n)
        self.refresh()
        self.say(f"[dev] +${n}")

    def give_joker(self, jid):
        # env.acquire: the same code path BUY uses, minus the cost
        self.state, ok = _acquire(self.state, jnp.int32(jid))
        self.refresh()
        self.say(f"[dev] gave {JOKER_NAMES[jid]}" if bool(ok) else "[dev] joker slots full")

    # --- layout ---------------------------------------------------------------
    def hand_order(self):
        h = self.h
        lanes = [i for i in range(int(h.hand_size)) if bool(h.hand_valid[i])]
        if self.sort_by == "rank":
            key = lambda i: (-(int(h.hand[i]) % 13), int(h.hand[i]) // 13)
        else:
            key = lambda i: (-(int(h.hand[i]) // 13), -(int(h.hand[i]) % 13))
        return sorted(lanes, key=key)

    def card_rects(self):
        order = self.hand_order()
        n = max(len(order), 1)
        cw, ch = 96, 134
        area_x, area_w = 330, 900
        step = min(cw + 14, (area_w - cw) / max(n - 1, 1))
        x0 = area_x + (area_w - (step * (n - 1) + cw)) / 2
        out = []
        for k, lane in enumerate(order):
            y = 470 - (30 if lane in self.selected else 0)
            out.append((lane, pygame.Rect(int(x0 + k * step), y, cw, ch)))
        return out

    def joker_rects(self):
        held = [i for i in range(MAX_JOKERS) if int(self.h.jokers[i]) >= 0]
        return [(i, pygame.Rect(330 + k * 150, 28, 138, 150)) for k, i in enumerate(held)]

    def shop_rects(self):
        n = int(self.h.shop_size)
        return [(i, pygame.Rect(560 + i * 170, 238, 150, 175)) for i in range(n)]

    def buttons(self):
        h, b = self.h, {}
        stage = int(h.game_stage)
        if stage == STAGE_BLIND:
            n = len(self.selected)
            b["play"] = Button((560, 660, 170, 54), "Play Hand", BTN_PLAY, 1 <= n <= 5)
            b["discard"] = Button((750, 660, 170, 54), "Discard", BTN_DISC,
                                  1 <= n <= 5 and int(h.discards) > 0)
            b["sort_rank"] = Button((940, 668, 110, 40), "Sort Rank", PANEL_HI)
            b["sort_suit"] = Button((1060, 668, 110, 40), "Sort Suit", PANEL_HI)
        if stage == STAGE_SHOP:
            b["next"] = Button((340, 250, 190, 70), "Next Round", BTN_DISC)
            b["reroll"] = Button((340, 334, 190, 70), f"Reroll ${int(h.reroll_cost)}", BTN_GREEN,
                                 int(h.money) >= int(h.reroll_cost))
        if self.joker_sel is not None and stage in (STAGE_BLIND, STAGE_SHOP):
            for i, r in self.joker_rects():
                if i == self.joker_sel:
                    b["sell"] = Button((r.x + 19, r.bottom + 6, 100, 34),
                                       f"Sell ${int(h.joker_sell_value[i])}", BTN_GREEN)
        if stage in (STAGE_LOST, STAGE_WON):
            b["new"] = Button((W // 2 - 110, 420, 220, 60), "New Run", BTN_PLAY)
        # dev strip
        b["dev_money"] = Button((14, 710, 64, 30), "+$10", PANEL_HI)
        for k, name in enumerate(JOKER_NAMES):
            b[f"dev_j{k}"] = Button((84 + k * 54, 710, 50, 30), name.split()[0][:5], PANEL_HI)
        return b

    # --- input ----------------------------------------------------------------
    def click(self, pos):
        h = self.h
        stage = int(h.game_stage)
        for name, btn in self.buttons().items():
            if btn.hit(pos):
                return self.press(name)
        if stage == STAGE_BLIND:
            for lane, r in reversed(self.card_rects()):     # topmost card first
                if r.collidepoint(pos):
                    self.toggle(lane)
                    return
        if stage == STAGE_SHOP:
            for i, r in self.shop_rects():
                if r.collidepoint(pos) and int(h.shop_slots[i]) >= 0:
                    return self.step(V_BUY, i)
        for i, r in self.joker_rects():
            if r.collidepoint(pos):
                self.joker_sel = None if self.joker_sel == i else i
                return
        self.joker_sel = None

    def press(self, name):
        if name == "play":
            self.step(V_PLAY)
        elif name == "discard":
            self.step(V_DISCARD)
        elif name == "sort_rank":
            self.sort_by = "rank"
        elif name == "sort_suit":
            self.sort_by = "suit"
        elif name == "next":
            self.step(V_NEXT)
        elif name == "reroll":
            self.step(V_REROLL)
        elif name == "sell":
            self.step(V_SELL, self.joker_sel)
        elif name == "new":
            self.new_run(self.seed + 1)
        elif name == "dev_money":
            self.give_money(10)
        elif name.startswith("dev_j"):
            self.give_joker(int(name[5:]))

    def toggle(self, lane):
        if lane in self.selected:
            self.selected.discard(lane)
        elif len(self.selected) < 5:
            self.selected.add(lane)
        self.refresh()

    def key(self, k):
        stage = int(self.h.game_stage)
        if k == pygame.K_n:
            self.new_run(self.seed + 1)
        elif stage == STAGE_BLIND:
            if pygame.K_1 <= k <= pygame.K_9:
                order = self.hand_order()
                idx = k - pygame.K_1
                if idx < len(order):
                    self.toggle(order[idx])
            elif k in (pygame.K_RETURN, pygame.K_SPACE) and 1 <= len(self.selected) <= 5:
                self.step(V_PLAY)
            elif k == pygame.K_d and 1 <= len(self.selected) <= 5 and int(self.h.discards) > 0:
                self.step(V_DISCARD)
            elif k == pygame.K_r:
                self.sort_by = "rank"
            elif k == pygame.K_s:
                self.sort_by = "suit"

    # --- drawing --------------------------------------------------------------
    def draw_card(self, rect, cid, selected=False, scored=None):
        surf, f = self.screen, self.fonts
        box(surf, rect.move(0, 5), (20, 44, 34), 9)
        box(surf, rect, CARD_C, 9, SEL_EDGE if selected else CARD_EDGE, 3 if selected else 2)
        rank, suit = cid % 13, cid // 13
        col = SUIT_C[suit]
        r = "10" if RANK_CHARS[rank] == "T" else RANK_CHARS[rank]
        text(surf, f.rank, r, col, (rect.x + 8, rect.y + 4))
        text(surf, f.glyph_small, SUIT_GLYPH[suit], col, (rect.x + 9, rect.y + 30))
        text(surf, f.glyph, SUIT_GLYPH[suit], col, rect.center, "center")
        if scored is False:
            dim = pygame.Surface(rect.size, pygame.SRCALPHA)
            dim.fill((0, 0, 0, 90))
            surf.blit(dim, rect)

    def draw_sidebar(self):
        s, f, h = self.screen, self.fonts, self.h
        box(s, (12, 12, 300, 690), PANEL, 14)
        stage = int(h.game_stage)
        bi = int(h.blind_index)
        blind_c = [(0, 120, 200), (230, 150, 40), (200, 60, 60)][bi]
        box(s, (24, 24, 276, 92), blind_c, 10)
        text(s, f.big, BLIND_NAMES[bi], TEXT, (162, 40), "midtop")
        text(s, f.body, f"Score at least {int(h.blind_target):,}", TEXT, (162, 80), "midtop")

        box(s, (24, 126, 276, 60), PANEL_HI, 10)
        text(s, f.body, "Round score", DIM, (36, 146))
        text(s, f.big, f"{int(h.chips_scored):,}", TEXT, (288, 138), "topright")

        # hand readout: the selection preview, or the last play
        box(s, (24, 196, 276, 112), PANEL_HI, 10)
        label, chips, mult = "", 0, 0
        if self.preview is not None:
            label = hand_label(self.preview.handtype)
            chips, mult = float(self.preview.base_chips), float(self.preview.base_mult)
        elif self.last_play is not None:
            res, _ = self.last_play
            label = hand_label(res.handtype)
            chips, mult = float(res.chips), float(res.mult)
        text(s, f.bold, label, TEXT, (162, 206), "midtop")
        box(s, (36, 240, 114, 56), CHIPS_C, 8)
        box(s, (174, 240, 114, 56), MULT_C, 8)
        text(s, f.big, f"{chips:g}", TEXT, (93, 268), "center")
        text(s, f.big, "X", MULT_C, (162, 268), "center")
        text(s, f.big, f"{mult:g}", TEXT, (231, 268), "center")

        y = 320
        for lbl, val, col in [("Hands", int(h.hands), CHIPS_C), ("Discards", int(h.discards), MULT_C)]:
            box(s, (24 if lbl == "Hands" else 166, y, 134, 64), PANEL_HI, 10)
            x = 91 if lbl == "Hands" else 233
            text(s, f.small, lbl, DIM, (x, y + 6), "midtop")
            text(s, f.big, val, col, (x, y + 26), "midtop")
        y += 74
        box(s, (24, y, 276, 52), PANEL_HI, 10)
        text(s, f.big, f"${int(h.money)}", MONEY_C, (162, y + 26), "center")
        y += 62
        box(s, (24, y, 134, 64), PANEL_HI, 10)
        text(s, f.small, "Ante", DIM, (91, y + 6), "midtop")
        text(s, f.big, f"{int(h.ante)}/{WIN_ANTE}", MONEY_C, (91, y + 26), "midtop")
        box(s, (166, y, 134, 64), PANEL_HI, 10)
        text(s, f.small, "Round", DIM, (233, y + 6), "midtop")
        text(s, f.big, int(h.round), MONEY_C, (233, y + 26), "midtop")

        y += 76
        for line in self.log[-7:]:
            for part in wrap(f.small, line, 270)[:2]:
                text(s, f.small, part, DIM, (26, y))
                y += 17
            if y > 690:
                break

    def draw_jokers(self):
        s, f, h = self.screen, self.fonts, self.h
        held = int((h.jokers != J_EMPTY).sum())
        text(s, f.small, f"Jokers {held}/{int(h.joker_slots)}", DIM, (330, 8))
        for i, r in self.joker_rects():
            jid = int(h.jokers[i])
            name = JOKER_NAMES[jid]
            box(s, r, PANEL, 10, SEL_EDGE if i == self.joker_sel else RARITY_C[int(JOKER_RARITY[jid])], 3)
            text(s, f.bold, name, TEXT, (r.centerx, r.y + 8), "midtop")
            yy = r.y + 36
            for line in wrap(f.small, JOKER_DESC[jid], r.w - 14)[:4]:
                text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
                yy += 17
            label, col = effect_label(jid, float(self.v[i]))
            text(s, f.bold, f"now {label}", col, (r.centerx, r.bottom - 30), "midtop")

    def draw_shop(self):
        s, f, h = self.screen, self.fonts, self.h
        box(s, (322, 222, 920, 210), PANEL, 14)
        text(s, f.bold, "SHOP", MONEY_C, (1230, 230), "topright")
        for i, r in self.shop_rects():
            jid = int(h.shop_slots[i])
            if jid < 0:
                box(s, r, PANEL_HI, 10)
                text(s, f.small, "sold out", DIM, r.center, "center")
                continue
            name = JOKER_NAMES[jid]
            cost = int(JOKER_COST[jid])
            ok = int(h.money) >= cost
            box(s, r, PANEL_HI, 10, RARITY_C[int(JOKER_RARITY[jid])], 3)
            text(s, f.bold, name, TEXT, (r.centerx, r.y + 8), "midtop")
            yy = r.y + 36
            for line in wrap(f.small, JOKER_DESC[jid], r.w - 14)[:4]:
                text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
                yy += 17
            text(s, f.big, f"${cost}", MONEY_C if ok else DIM, (r.centerx, r.bottom - 40), "midtop")
        if self.cashout:
            text(s, f.body, self.cashout, MONEY_C, (782, 445), "midtop")

    def draw_play_area(self):
        s, f, h = self.screen, self.fonts, self.h
        if self.last_play is not None:
            res, cards = self.last_play
            mask = [bool(m) for m, ok in zip(res.mask, res.valid) if ok]
            n = len(cards)
            x0 = 780 - (n * 84) // 2
            for k, cid in enumerate(cards):
                r = pygame.Rect(x0 + k * 84, 240, 76, 106)
                self.draw_card(r, cid, scored=mask[k])
            text(s, f.bold, f"{hand_label(res.handtype)}:  {float(res.chips):g} x "
                            f"{float(res.mult):g}  =  {int(res.score):,}",
                 TEXT, (780, 360), "midtop")
        deck_left = int(h.deck_size) - int(h.deck_ptr)
        text(s, f.small, f"Deck {deck_left}/{int(h.deck_size)}", DIM, (1240, 640), "topright")

    def draw_hand(self):
        for lane, r in self.card_rects():
            self.draw_card(r, int(self.h.hand[lane]), selected=lane in self.selected)

    def draw_overlay(self, title, sub):
        s, f = self.screen, self.fonts
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 160))
        s.blit(veil, (0, 0))
        text(s, f.huge, title, TEXT, (W // 2, 300), "center")
        text(s, f.body, sub, DIM, (W // 2, 360), "center")

    def draw(self):
        s, h = self.screen, self.h
        s.fill(BG)
        stage = int(h.game_stage)
        self.draw_sidebar()
        self.draw_jokers()
        if stage == STAGE_SHOP:
            self.draw_shop()
        else:
            self.draw_play_area()
            if stage == STAGE_BLIND:
                self.draw_hand()
        buttons = self.buttons()
        if stage in (STAGE_LOST, STAGE_WON):
            reached = f"Ante {int(h.ante)}, round {int(h.round)}, seed {self.seed}"
            self.draw_overlay("GAME OVER" if stage == STAGE_LOST else "YOU WIN", reached)
        for btn in buttons.values():
            btn.draw(s, self.fonts)
        text(s, self.fonts.small, "dev", DIM, (14, 694))
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                if e.type == pygame.KEYDOWN:
                    self.key(e.key)
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    App(ap.parse_args().seed).run()
