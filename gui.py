"""Play BalatroNET by hand — a pygame front end over the jitted env.

Every game action goes through env_step, the same function the agent trains
on. The GUI keeps no game rules of its own: it reads the state, draws it,
and turns clicks into balatroAction values. The only exceptions are the dev
buttons (give money / give a joker), which stage the state directly, the way
fixtures.py does in the tests.

    uv run --project D:/Claude/jax-learn python gui.py [--seed N]

Mouse: click cards to select, click a joker to select it (then SELL), click a
shop joker to buy it. Drag a joker or a hand card to a new place (MOVE_JOKER /
MOVE_CARD steps through env_step).
Keys:  1-8 toggle a card   Enter play   D discard   R / S sort by rank / suit
       V deck view   N new run   Left / Right move the selected joker   Esc close a popup / quit
Model: P model view on/off (hides the panel, the odds and the pick outline)
       M take the model's pick   A auto-play on/off   G greedy / sampled pick
       O card odds on/off.  The panel on the right always shows the model's
       odds for the current state, at every stage.
       --params runs/params_run.pkl (default)   --scale 0.2 (the training curriculum scale)
Dev:   +$10, and the Add Joker dropdown (scroll wheel when the list is long)
Seed:  the seed box: click it, type a number, Enter starts that seed (Esc cancels)
       Restart = the same seed again   Random = a random seed   N = seed + 1
       One seed fixes the shuffle and the shops. The same seed with the same
       actions gives the same run. MOVE steps do not use the RNG.

The hand shows the env's lane order, which is the play order (and the order
Death reads). Sort Rank / Sort Suit reorder the hand once, with MOVE_CARD steps;
new cards land in the lanes the played cards left.
"""

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pygame

from env import (balatroAction, env_step, reset, explain_play, preview, build_ctx, acquire,
                 invariant_errors, MAX_HAND_SIZE, STAGE_BLIND, STAGE_SHOP, STAGE_LOST,
                 STAGE_WON, BLIND_NAMES, BLIND_REWARD, V_PLAY, V_DISCARD, V_BUY, V_SELL, V_MOVE_JOKER, V_MOVE_CARD,
                 V_REROLL, V_NEXT, WIN_ANTE, INTEREST_PER, INTEREST_CAP, RANK_CHARS)
from jokers import (JOKER_NAMES, JOKER_DESC, JOKER_COST, JOKERS, JOKER_RARITY, J_EMPTY,
                    MAX_JOKERS, rows_for, resolve, COMMON, UNCOMMON, RARE,
                    E_ADD_MULT, E_ADD_CHIPS, E_XMULT, E_ADD_MONEY, EFFECT_NAMES)
from scoring import HAND_NAMES
from agent import (observe, legal_mask, masked_logits, init_params, action_name, OBS_DIM,
                   N_ACTIONS, ACTION_VERB, ACTION_TARGET, A_PLAY, A_DISCARD, A_BUY, A_SELL,
                   A_MOVE_JOKER, A_MOVE_CARD, A_REROLL)
from train import load_params, PARAMS_PATH, REWARD_SCALE

GW = 1280                  # the game area; the model panel sits to its right
PANEL_W = 320
W, H = GW + PANEL_W, 760
AUTO_MS = 700              # model auto-play: one step per AUTO_MS

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
RARITY_NAME = {COMMON: "Common", UNCOMMON: "Uncommon", RARE: "Rare"}
AI_C      = (90, 220, 230)    # the model's colour: panel edge, its pick
BAR_C     = (40, 110, 130)


def odds_color(p):
    """Dim for unlikely, the model colour for likely."""
    return AI_C if p >= 0.5 else (DIM if p < 0.2 else TEXT)

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


@jax.jit
def model_view(params, state, key):
    """Everything the model panel shows for one state: the probability of every
    choice, every select lane, the value, and a sampled and a greedy pick."""
    legal = legal_mask(state)
    a_logits, s_logits, value = masked_logits(params, observe(state), legal, state.hand_valid)
    pa = jax.nn.softmax(a_logits)
    ps = jax.nn.sigmoid(s_logits) * state.hand_valid
    k1, k2 = jax.random.split(key)
    return {"pa": pa, "ps": ps, "value": value, "legal": legal,
            "a_sample": jax.random.categorical(k1, a_logits),
            "sel_sample": (jax.random.uniform(k2, ps.shape) < ps) & state.hand_valid,
            "a_greedy": jnp.argmax(pa),
            "sel_greedy": (ps > 0.5) & state.hand_valid}


def load_model(path):
    """-> (params, label). A missing file gives a random policy, labelled as such."""
    if path and os.path.exists(path):
        params = load_params(path)
        if params["W1"].shape[0] != OBS_DIM or params["Wa"].shape[1] != N_ACTIONS:
            print(f"{path}: saved for a different obs/action layout; using a random policy")
        else:
            return jax.tree.map(jnp.asarray, params), os.path.basename(path)
    return init_params(jax.random.key(0)), "random (no params file)"


def host(state):
    """Device state -> numpy copy for drawing (one transfer per step, not per frame)."""
    return jax.device_get(state._replace(key=jax.random.key_data(state.key)))


def hand_label(i):
    return HAND_LABEL.get(HAND_NAMES[int(i)], HAND_NAMES[int(i)])


def card_name(cid):
    """Short card name for the log and the model panel: 10♥, K♠."""
    r = "10" if RANK_CHARS[cid % 13] == "T" else RANK_CHARS[cid % 13]
    return r + SUIT_GLYPH[cid // 13]


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
        self.glyph_tiny = pygame.font.SysFont("segoeuisymbol", 14)


def text(surf, font, s, color, pos, anchor="topleft"):
    img = font.render(str(s), True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def fit(fonts, s, width):
    """The bold font, or the body font when the bold one is wider than width."""
    return fonts.bold if fonts.bold.size(s)[0] <= width else fonts.body


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


class Dropdown:
    """A toggle button plus a list that opens upward. pick(i) runs on a row click.

    rows() returns [(label, right_text, stripe_color, enabled)]; it is called on
    every draw, so the list always matches the current state.
    """
    ROW_H, MAX_ROWS = 30, 14

    def __init__(self, rect, label, rows, pick, width=340):
        self.button = Button(rect, label, PANEL_HI)
        self.rows, self.pick, self.width = rows, pick, width
        self.open, self.scroll = False, 0

    def menu_rect(self):
        n = min(len(self.rows()), self.MAX_ROWS)
        b = self.button.rect
        return pygame.Rect(b.x, b.y - 6 - n * self.ROW_H - 8, self.width, n * self.ROW_H + 8)

    def row_rects(self):
        m, rows = self.menu_rect(), self.rows()
        vis = range(self.scroll, min(self.scroll + self.MAX_ROWS, len(rows)))
        return [(i, pygame.Rect(m.x + 4, m.y + 4 + k * self.ROW_H, m.w - 8, self.ROW_H - 2))
                for k, i in enumerate(vis)]

    def wheel(self, dy):
        top = max(len(self.rows()) - self.MAX_ROWS, 0)
        self.scroll = min(max(self.scroll - dy, 0), top)

    def click(self, pos):
        """True when the dropdown used the click (then nothing else gets it)."""
        if self.button.rect.collidepoint(pos):
            self.open = not self.open
            return True
        if not self.open:
            return False
        rows = self.rows()
        for i, r in self.row_rects():
            if r.collidepoint(pos) and rows[i][3]:
                self.pick(i)
                return True
        if not self.menu_rect().collidepoint(pos):
            self.open = False          # a click outside closes the menu
        return True

    def draw(self, surf, fonts):
        self.button.draw(surf, fonts)
        b = self.button.rect                         # arial has no arrow glyph: draw one
        cx, cy = b.right - 14, b.centery
        tri = [(cx - 5, cy + 3), (cx + 5, cy + 3), (cx, cy - 3)] if self.open else \
              [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 3)]
        pygame.draw.polygon(surf, TEXT, tri)
        if not self.open:
            return
        rows, m = self.rows(), self.menu_rect()
        box(surf, m, PANEL, 8, PANEL_HI, 2)
        mouse = pygame.mouse.get_pos()
        for i, r in self.row_rects():
            label, right, stripe, ok = rows[i]
            if ok and r.collidepoint(mouse):
                box(surf, r, PANEL_HI, 6)
            pygame.draw.rect(surf, stripe, (r.x + 4, r.y + 5, 5, r.h - 10), border_radius=2)
            text(surf, fonts.body, label, TEXT if ok else DIM, (r.x + 16, r.centery), "midleft")
            text(surf, fonts.small, right, DIM, (r.right - 8, r.centery), "midright")
        if len(rows) > self.MAX_ROWS:
            text(surf, fonts.small, f"{self.scroll + 1}-{self.scroll + self.MAX_ROWS} of {len(rows)}",
                 DIM, (m.right - 6, m.y - 4), "bottomright")


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, seed, scale=1.0, params_path=PARAMS_PATH):
        self.scale = scale
        self.params, self.model_label = load_model(params_path)
        self.ai_greedy = False         # False: show a sampled pick (what training does)
        self.ai_auto = False
        self.ai_odds = True            # select odds above the cards
        self.show_model = True         # the whole model view: panel, odds, pick outline
        self.ai_n = 0                  # bumps the sampling key after every step
        self.ai_last = 0               # ms of the last auto step
        pygame.init()
        pygame.display.set_caption("BalatroNET")
        self.screen = pygame.display.set_mode((W, H))   # show_model starts on
        self.fonts = Fonts()
        self.clock = pygame.time.Clock()
        self.seed = seed
        self.log = []
        self.show_deck = False
        self.seed_edit = None          # the typed digits while the seed box is open
        self.drag = None               # [kind "joker"/"card", slot or lane, press pos, mouse pos]
        self.joker_menu = Dropdown((84, 710, 170, 30), "Add Joker   ", self.joker_menu_rows,
                                   lambda jid: self.give_joker(jid))
        self.splash("Compiling the jitted env ...")
        self.new_run(seed)
        # warm the jit caches so the first click does not stall
        explain_play(self.state, jnp.zeros(MAX_HAND_SIZE, bool))
        preview(self.state, jnp.zeros(MAX_HAND_SIZE, bool))
        live_values(self.state)
        env_step(self.state, self.action(V_NEXT))
        explain_play(self.state, jnp.zeros(MAX_HAND_SIZE, bool))

    # --- state management ---------------------------------------------------
    def splash(self, msg):
        self.screen.fill(BG)
        text(self.screen, self.fonts.big, msg, TEXT, (W // 2, H // 2), "center")
        pygame.display.flip()

    def new_run(self, seed):
        self.seed = seed
        self.state = reset(jax.random.key(seed), 52, 8, self.scale)
        self.selected = set()          # lanes
        self.joker_sel = None          # joker slot
        self.last_play = None          # (PlayResult on host, cards, pre-state hands)
        self.cashout = None            # breakdown text
        self.refresh()
        self.say(f"New run, seed {seed}, target scale {self.scale:g}. "
                 f"{BLIND_NAMES[0]}: score {int(self.h.blind_target)} chips.")

    def refresh(self):
        self.h = host(self.state)
        self.v = np.asarray(live_values(self.state))
        key = jax.random.fold_in(jax.random.key(self.seed), self.ai_n)
        self.ai = jax.device_get(model_view(self.params, self.state, key))
        self.ai_hand = None            # the hand the model's pick would make (PLAY / DISCARD)
        a, sel = self.ai_choice()
        if a in (A_PLAY, A_DISCARD) and sel.any():
            self.ai_hand = jax.device_get(explain_play(self.state, jnp.asarray(sel)))
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
        self.ai_n += 1
        self.refresh()
        h = self.h
        if verb == V_PLAY:
            res, _ = self.last_play
            self.say(f"Played {hand_label(res.handtype)}: {float(res.chips):g} x {float(res.mult):g}"
                     f" = {int(res.score)}")
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
        elif verb == V_MOVE_CARD:
            self.say(f"Moved {card_name(int(pre.hand[target]))} right of {card_name(int(pre.hand[target + 1]))}."
                     if bool(pre.hand_valid[target]) and bool(pre.hand_valid[target + 1]) else "Moved a card.")
        elif verb == V_MOVE_JOKER:
            self.say(f"Moved {JOKER_NAMES[int(pre.jokers[target])]} right of "
                     f"{JOKER_NAMES[int(pre.jokers[target + 1])]}.")
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
            jm = int(h.last_cashout) - blind - int(h.hands) - interest
            if jm:
                self.cashout += f"  +  jokers ${jm}"
            self.say(f"Blind defeated! {self.cashout}")
        if int(h.game_stage) == STAGE_LOST and int(pre.game_stage) != STAGE_LOST:
            self.say("Out of hands. Game over.")
        if int(h.game_stage) == STAGE_WON and int(pre.game_stage) != STAGE_WON:
            self.say("You won the run!")

    def move_joker(self, i, k):
        """Bring the joker in slot i to slot k with MOVE steps (adjacent swaps)."""
        if i == k or int(self.h.game_stage) not in (STAGE_BLIND, STAGE_SHOP):
            return
        name = JOKER_NAMES[int(self.h.jokers[i])]
        for t in (range(i, k) if k > i else range(i - 1, k - 1, -1)):
            self.state, _, _ = env_step(self.state, self.action(V_MOVE_JOKER, t))
            self.ai_n += 1
        if self.joker_sel == i:
            self.joker_sel = k
        self.refresh()
        self.say(f"Moved {name} to slot {k + 1}.")

    def card_swaps(self, swaps):
        """Run MOVE_CARD steps (lane t <-> t+1). The selection follows its cards."""
        if not swaps or int(self.h.game_stage) != STAGE_BLIND:
            return
        ids = {int(self.h.hand[l]) for l in self.selected}
        for t in swaps:
            self.state, _, _ = env_step(self.state, self.action(V_MOVE_CARD, t))
            self.ai_n += 1
        hand, valid = np.asarray(self.state.hand), np.asarray(self.state.hand_valid)
        self.selected = {l for l in range(MAX_HAND_SIZE) if valid[l] and int(hand[l]) in ids}
        self.refresh()

    def move_card(self, lane, k):
        """Bring the card in `lane` to place k of the visible hand."""
        order = self.hand_order()
        dest = order[k]
        self.card_swaps(list(range(lane, dest)) if dest > lane else list(range(lane - 1, dest - 1, -1)))

    def sort_hand(self, by):
        """Bubble sort the lanes with MOVE_CARD steps. Holes drift to the right."""
        h = self.h
        n = int(h.hand_size)
        cards = [int(h.hand[l]) if bool(h.hand_valid[l]) else None for l in range(n)]
        key = ((lambda c: (-(c % 13), c // 13)) if by == "rank"
               else (lambda c: (-(c // 13), -(c % 13))))
        k = lambda c: (1,) if c is None else (0,) + key(c)
        swaps = []
        for end in range(n - 1, 0, -1):
            for t in range(end):
                if k(cards[t]) > k(cards[t + 1]):
                    cards[t], cards[t + 1] = cards[t + 1], cards[t]
                    swaps.append(t)
        self.card_swaps(swaps)
        self.say(f"Sorted by {by} ({len(swaps)} MOVE_CARD steps).")

    def drag_moved(self, kind):
        """The slot / lane held in a real drag of this kind, else None."""
        d = self.drag
        return d[1] if d and d[0] == kind and abs(d[3][0] - d[2][0]) > 6 else None

    def drop_slot(self, x):
        """The lineup slot under screen x, clamped to the held jokers."""
        n = len(self.joker_rects())
        return min(max(int((x - 330 + 6) // 150), 0), n - 1)

    def drop_place(self, x):
        """The visible hand place whose card center is nearest screen x."""
        rects = self.card_rects()
        return min(range(len(rects)), key=lambda k: abs(rects[k][1].centerx - x))

    def drag_order(self):
        """Joker slots in drawing order: the held one goes to the drop slot."""
        order = [i for i, _ in self.joker_rects()]
        i = self.drag_moved("joker")
        if i is not None:
            order.remove(i)
            order.insert(self.drop_slot(self.drag[3][0]), i)
        return order

    def card_drag_order(self):
        """Hand lanes in drawing order: the held card goes to the drop place."""
        order = self.hand_order()
        lane = self.drag_moved("card")
        if lane is not None:
            order.remove(lane)
            order.insert(self.drop_place(self.drag[3][0]), lane)
        return order

    # --- dev staging (bypasses env_step on purpose, like fixtures.py) -------
    def toggle_model(self):
        """Show or hide the model view. Hidden: no panel, no odds, no auto-play,
        and the window narrows to the game area."""
        self.show_model = not self.show_model
        self.ai_auto = self.ai_auto and self.show_model
        self.screen = pygame.display.set_mode((W if self.show_model else GW, H))

    def give_money(self, n):
        self.state = self.state._replace(money=self.state.money + n)
        self.refresh()
        self.say(f"[dev] +${n}")

    def give_joker(self, jid):
        # env.acquire: the same code path BUY uses, minus the cost
        self.state, ok = _acquire(self.state, jnp.int32(jid))
        self.refresh()
        self.say(f"[dev] gave {JOKER_NAMES[jid]}" if bool(ok) else "[dev] joker slots full")

    # --- the model ------------------------------------------------------------------
    def ai_choice(self):
        """-> (flat action index, select lanes (32,) bool) for the current mode."""
        if self.ai_greedy:
            return int(self.ai["a_greedy"]), np.asarray(self.ai["sel_greedy"])
        return int(self.ai["a_sample"]), np.asarray(self.ai["sel_sample"])

    def ai_played_lanes(self, sel):
        """The lanes env_step really acts on: the first 5 selected valid lanes."""
        return [i for i in range(MAX_HAND_SIZE) if sel[i] and bool(self.h.hand_valid[i])][:5]

    def choice_label(self, a):
        h = self.h
        if A_BUY <= a < A_SELL:
            jid = int(h.shop_slots[a - A_BUY])
            return f"BUY {JOKER_NAMES[jid]} (${int(JOKER_COST[jid])})" if jid >= 0 else action_name(a)
        if A_MOVE_CARD <= a < A_REROLL:
            t = a - A_MOVE_CARD
            if bool(h.hand_valid[t]) and bool(h.hand_valid[t + 1]):
                return f"MOVE {card_name(int(h.hand[t]))} <> {card_name(int(h.hand[t + 1]))}"
            return action_name(a)
        if A_MOVE_JOKER <= a < A_MOVE_CARD:
            i = a - A_MOVE_JOKER
            if int(h.jokers[i]) >= 0 and int(h.jokers[i + 1]) >= 0:
                return f"MOVE {JOKER_NAMES[int(h.jokers[i])]} <> {JOKER_NAMES[int(h.jokers[i + 1])]}"
            return action_name(a)
        if A_SELL <= a < A_MOVE_JOKER:
            i = a - A_SELL
            jid = int(h.jokers[i])
            return f"SELL {JOKER_NAMES[jid]} (+${int(h.joker_sell_value[i])})" if jid >= 0 else action_name(a)
        if a == A_REROLL:
            return f"REROLL (${int(h.reroll_cost)})"
        return action_name(a)

    def model_step(self):
        """Carry out the model's pick through env_step, like any human action."""
        if int(self.h.game_stage) not in (STAGE_BLIND, STAGE_SHOP):
            self.ai_auto = False
            return
        a, sel = self.ai_choice()
        p = float(self.ai["pa"][a])
        verb, target = int(ACTION_VERB[a]), int(ACTION_TARGET[a])
        self.say(f"[model] {self.choice_label(a)}  p={p:.0%}")
        self.selected = set(self.ai_played_lanes(sel)) if a in (A_PLAY, A_DISCARD) else set()
        self.joker_sel = None
        self.step(verb, target)

    def joker_menu_rows(self):
        h = self.h
        full = int((h.jokers != J_EMPTY).sum()) >= int(h.joker_slots)
        held = list(int(j) for j in h.jokers)
        return [(name + (f"  (x{held.count(jid)})" if jid in held else ""),
                 f"${int(JOKER_COST[jid])}  {RARITY_NAME[int(JOKER_RARITY[jid])]}",
                 RARITY_C[int(JOKER_RARITY[jid])], not full)
                for jid, name in enumerate(JOKER_NAMES)]

    # --- layout ---------------------------------------------------------------
    def hand_order(self):
        h = self.h
        return [i for i in range(int(h.hand_size)) if bool(h.hand_valid[i])]   # lane order = play order

    def card_rects(self, order=None):
        order = self.hand_order() if order is None else order
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
            b["new"] = Button((GW // 2 - 110, 420, 220, 60), "New Run", BTN_PLAY)
        # dev strip
        b["dev_money"] = Button((14, 710, 64, 30), "+$10", PANEL_HI)
        editing = self.seed_edit is not None
        b["seed"] = Button((262, 710, 110, 30), f"#{self.seed_edit}_" if editing else f"Seed {self.seed}",
                           AI_C if editing else PANEL_HI)
        b["restart"] = Button((378, 710, 84, 30), "Restart", PANEL_HI)
        b["random"] = Button((468, 710, 84, 30), "Random", PANEL_HI)
        b["deck"] = Button((1150, 710, 116, 36), "Deck (V)", PANEL_HI)
        b["model"] = Button((1020, 710, 120, 36), "Model (P)", AI_C if self.show_model else PANEL_HI)
        if not self.show_model:
            return b
        # model panel
        x = GW + 14
        live = stage in (STAGE_BLIND, STAGE_SHOP)
        b["ai_step"] = Button((x, 616, 140, 40), "Step (M)", BTN_PLAY, live)
        b["ai_auto"] = Button((x + 152, 616, 140, 40), "Stop (A)" if self.ai_auto else "Auto (A)",
                              BTN_DISC if self.ai_auto else BTN_GREEN, live)
        b["ai_mode"] = Button((x, 664, 140, 36), "Greedy (G)" if self.ai_greedy else "Sample (G)",
                              PANEL_HI)
        b["ai_odds"] = Button((x + 152, 664, 140, 36), "Odds on (O)" if self.ai_odds else "Odds off (O)",
                              PANEL_HI)
        return b

    # --- input ----------------------------------------------------------------
    def click(self, pos):
        h = self.h
        stage = int(h.game_stage)
        if self.show_deck:
            self.show_deck = False         # any click closes the deck view
            return
        if self.seed_edit is not None and not self.buttons()["seed"].hit(pos):
            self.seed_edit = None          # a click outside the seed box cancels it
        if self.joker_menu.click(pos):
            return
        for name, btn in self.buttons().items():
            if btn.hit(pos):
                return self.press(name)
        if stage == STAGE_BLIND:
            for lane, r in reversed(self.card_rects()):     # topmost card first
                if r.collidepoint(pos):
                    self.drag = ["card", lane, pos, pos]   # release() selects or moves it
                    return
        if stage == STAGE_SHOP:
            for i, r in self.shop_rects():
                if r.collidepoint(pos) and int(h.shop_slots[i]) >= 0:
                    return self.step(V_BUY, i)
        for i, r in self.joker_rects():
            if r.collidepoint(pos):
                self.drag = ["joker", i, pos, pos]   # release() selects or moves it
                return
        self.joker_sel = None

    def release(self, pos):
        if self.drag is None:
            return
        kind, i, start, _ = self.drag
        self.drag = None
        click = abs(pos[0] - start[0]) <= 6    # no real drag
        if kind == "card":
            if click:
                self.toggle(i)
            else:
                self.move_card(i, self.drop_place(pos[0]))
        elif click:
            self.joker_sel = None if self.joker_sel == i else i
        else:
            self.move_joker(i, self.drop_slot(pos[0]))

    def press(self, name):
        if name == "play":
            self.step(V_PLAY)
        elif name == "discard":
            self.step(V_DISCARD)
        elif name == "sort_rank":
            self.sort_hand("rank")
        elif name == "sort_suit":
            self.sort_hand("suit")
        elif name == "next":
            self.step(V_NEXT)
        elif name == "reroll":
            self.step(V_REROLL)
        elif name == "sell":
            self.step(V_SELL, self.joker_sel)
        elif name == "new":
            self.new_run(self.seed + 1)
        elif name == "seed":
            self.seed_edit = None if self.seed_edit is not None else ""
        elif name == "restart":
            self.new_run(self.seed)
        elif name == "random":
            self.new_run(int.from_bytes(os.urandom(4), "little") >> 1)
        elif name == "dev_money":
            self.give_money(10)
        elif name == "deck":
            self.show_deck = True
        elif name == "model":
            self.toggle_model()
        elif name == "ai_step":
            self.model_step()
        elif name == "ai_auto":
            self.ai_auto = not self.ai_auto
        elif name == "ai_mode":
            self.ai_greedy = not self.ai_greedy
            self.refresh()
        elif name == "ai_odds":
            self.ai_odds = not self.ai_odds

    def toggle(self, lane):
        if lane in self.selected:
            self.selected.discard(lane)
        elif len(self.selected) < 5:
            self.selected.add(lane)
        self.refresh()

    def key(self, k):
        stage = int(self.h.game_stage)
        if self.seed_edit is not None:     # the seed box takes every key while open
            digit = {getattr(pygame, f"K_{d}"): str(d) for d in range(10)}
            digit.update({getattr(pygame, f"K_KP{d}"): str(d) for d in range(10)})
            if k in digit and len(self.seed_edit) < 10:
                self.seed_edit += digit[k]
            elif k == pygame.K_BACKSPACE:
                self.seed_edit = self.seed_edit[:-1]
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                typed, self.seed_edit = self.seed_edit, None
                if typed:
                    self.new_run(int(typed))
            return
        if k == pygame.K_v:
            self.show_deck = not self.show_deck
            return
        if self.show_deck:
            return                         # the deck view eats game keys
        if k == pygame.K_p:
            return self.toggle_model()
        if k in (pygame.K_m, pygame.K_a, pygame.K_g, pygame.K_o) and not self.show_model:
            return                         # the model keys need the model view
        if k == pygame.K_m:
            return self.model_step()
        if k == pygame.K_a:
            self.ai_auto = not self.ai_auto
            return
        if k == pygame.K_g:
            self.ai_greedy = not self.ai_greedy
            return self.refresh()
        if k == pygame.K_o:
            self.ai_odds = not self.ai_odds
            return
        if k in (pygame.K_LEFT, pygame.K_RIGHT) and self.joker_sel is not None:
            n = len(self.joker_rects())
            k2 = self.joker_sel + (1 if k == pygame.K_RIGHT else -1)
            if 0 <= k2 < n:
                self.move_joker(self.joker_sel, k2)
            return
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
                self.sort_hand("rank")
            elif k == pygame.K_s:
                self.sort_hand("suit")

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
        held_slot = self.drag_moved("joker")
        cards = [(i, pygame.Rect(330 + k * 150, 28, 138, 150)) for k, i in enumerate(self.drag_order())]
        if held_slot is not None:          # draw the held card last, under the mouse
            box(s, next(r for i, r in cards if i == held_slot), PANEL_HI, 10)   # its drop gap
            cards = [c for c in cards if c[0] != held_slot]
            dx = self.drag[3][0] - self.drag[2][0]
            home = dict(self.joker_rects())[held_slot]
            cards.append((held_slot, home.move(dx, 8)))
        for i, r in cards:
            jid = int(h.jokers[i])
            name = JOKER_NAMES[jid]
            box(s, r, PANEL, 10, SEL_EDGE if i in (self.joker_sel, held_slot) else RARITY_C[int(JOKER_RARITY[jid])], 3)
            text(s, fit(f, name, r.w - 10), name, TEXT, (r.centerx, r.y + 8), "midtop")
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
            text(s, fit(f, name, r.w - 10), name, TEXT, (r.centerx, r.y + 8), "midtop")
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
        a, sel = self.ai_choice()
        picked = set(self.ai_played_lanes(sel)) if a in (A_PLAY, A_DISCARD) else set()
        held = self.drag_moved("card")
        cards = self.card_rects(self.card_drag_order())
        if held is not None:               # a gap at the drop place; the held card follows the mouse
            box(self.screen, dict(cards)[held], PANEL_HI, 9)
            cards = [c for c in cards if c[0] != held]
            home = dict(self.card_rects())[held]
            cards.append((held, home.move(self.drag[3][0] - self.drag[2][0], -40)))
        for lane, r in cards:
            self.draw_card(r, int(self.h.hand[lane]), selected=lane in self.selected)
            if not (self.ai_odds and self.show_model):
                continue
            if lane in picked:
                pygame.draw.rect(self.screen, AI_C, r.inflate(10, 10), 3, border_radius=12)
            p = float(self.ai["ps"][lane])
            tag = pygame.Rect(0, 0, 52, 22)
            tag.midbottom = (r.centerx, r.y - 7)
            box(self.screen, tag, PANEL, 6, AI_C if lane in picked else None, 2)
            text(self.screen, self.fonts.small, f"{p:.0%}", odds_color(p), tag.center, "center")

    def draw_model_panel(self):
        """The model's view of the CURRENT state: every legal choice with its odds."""
        s, f, h = self.screen, self.fonts, self.h
        x0 = GW + 4
        box(s, (x0, 12, PANEL_W - 14, H - 24), PANEL, 14, AI_C, 2)
        x = x0 + 12
        text(s, f.bold, "MODEL", AI_C, (x, 22))
        text(s, f.small, self.model_label, DIM, (x0 + PANEL_W - 28, 26), "topright")
        text(s, f.small, f"{'greedy' if self.ai_greedy else 'sampled'} pick   "
                         f"V = ${float(self.ai['value']) / REWARD_SCALE:.1f} expected from here",
             DIM, (x, 46))

        a, sel = self.ai_choice()
        live = int(h.game_stage) in (STAGE_BLIND, STAGE_SHOP)
        box(s, (x, 70, PANEL_W - 38, 74), PANEL_HI, 10)
        if live:
            text(s, f.small, "would choose", DIM, (x + 10, 76))
            text(s, f.bold, self.choice_label(a), TEXT, (x + 10, 94))
            text(s, f.bold, f"{float(self.ai['pa'][a]):.0%}", AI_C, (x + PANEL_W - 50, 94), "topright")
            if a in (A_PLAY, A_DISCARD):
                lanes = self.ai_played_lanes(sel)
                extra = int(sel.sum()) - len(lanes)
                note = f"{len(lanes)} card(s)" + (f", {extra} cut (5 max)" if extra > 0 else "")
                if a == A_PLAY and self.ai_hand is not None:
                    note += f"  ->  {hand_label(self.ai_hand.handtype)} {int(self.ai_hand.score)}"
                text(s, f.small, note, DIM, (x + 10, 120))
        else:
            text(s, f.bold, "run over", DIM, (x + 10, 94))

        # every legal choice, most likely first
        text(s, f.small, "choice odds", DIM, (x, 154))
        legal = [i for i in range(N_ACTIONS) if bool(self.ai["legal"][i])] if live else []
        legal.sort(key=lambda i: -float(self.ai["pa"][i]))
        y, bar_w = 174, PANEL_W - 38
        for i in legal[:12]:
            p = float(self.ai["pa"][i])
            r = pygame.Rect(x, y, bar_w, 26)
            box(s, r, PANEL_HI, 6)
            if p > 0:
                box(s, pygame.Rect(x, y, max(int(bar_w * p), 6), 26), BAR_C, 6)
            if i == a:
                pygame.draw.rect(s, AI_C, r, 2, border_radius=6)
            text(s, f.small, self.choice_label(i), TEXT, (x + 8, r.centery), "midleft")
            text(s, f.small, f"{p:.1%}", TEXT, (r.right - 8, r.centery), "midright")
            y += 30
        if len(legal) > 12:
            text(s, f.small, f"+{len(legal) - 12} more", DIM, (x, y))

        # card select odds, in hand order (also drawn above the cards)
        if live and int(h.game_stage) == STAGE_BLIND:
            y = 548
            text(s, f.small, "card odds (if PLAY / DISCARD)", DIM, (x, y - 2))
            order = self.hand_order()
            cell = (PANEL_W - 38) // max(len(order), 1)
            for k, lane in enumerate(order):
                p = float(self.ai["ps"][lane])
                cid = int(h.hand[lane])
                rk = "10" if RANK_CHARS[cid % 13] == "T" else RANK_CHARS[cid % 13]
                cx = x + k * cell + cell // 2
                text(s, f.small, rk, TEXT, (cx - 5, y + 16), "midtop")
                text(s, f.glyph_tiny, SUIT_GLYPH[cid // 13], SUIT_C[cid // 13] if cid // 13 != 3 else TEXT,
                     (cx + 9, y + 15), "midtop")
                text(s, f.small, f"{p:.0%}", odds_color(p), (cx, y + 34), "midtop")

    def draw_deck_view(self):
        """Suit x rank grid. Big number = cards left to draw; small = in the full deck."""
        s, f, h = self.screen, self.fonts, self.h
        ptr, size = int(h.deck_ptr), int(h.deck_size)
        deck = [int(c) for c in h.deck]
        total = np.zeros((4, 13), int)
        left = np.zeros((4, 13), int)
        for c in deck[:size]:
            total[(c // 13) % 4, c % 13] += 1
        for c in deck[ptr:size]:
            left[(c // 13) % 4, c % 13] += 1

        sw = s.get_width()                          # 1600 with the model view, 1280 without
        veil = pygame.Surface((sw, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        s.blit(veil, (0, 0))
        cw, ch = 70, 74
        gx, gy = (sw - (13 * cw + 2 * 70)) // 2 + 70, 190
        box(s, (gx - 90, gy - 110, 13 * cw + 200, 4 * ch + 150), PANEL, 14, PANEL_HI, 2)
        text(s, f.big, "Full deck", TEXT, (sw // 2, gy - 96), "midtop")
        text(s, f.small, f"{size - ptr} left to draw of {size}.  Big number: left to draw.  "
                         f"Small: in the deck.  Click or V to close.", DIM, (sw // 2, gy - 56), "midtop")
        ranks = list(range(12, -1, -1))             # A K Q ... 2, as Balatro shows them
        for k, r in enumerate(ranks):
            lbl = "10" if RANK_CHARS[r] == "T" else RANK_CHARS[r]
            text(s, f.bold, lbl, TEXT, (gx + k * cw + cw // 2, gy - 26), "midtop")
        text(s, f.bold, "Sum", DIM, (gx + 13 * cw + 35, gy - 26), "midtop")
        for suit in (3, 2, 0, 1):                   # S H C D
            row = [3, 2, 0, 1].index(suit)
            y = gy + row * ch
            text(s, f.glyph_small, SUIT_GLYPH[suit], SUIT_C[suit] if suit != 3 else TEXT,
                 (gx - 30, y + ch // 2), "center")
            for k, r in enumerate(ranks):
                cell = pygame.Rect(gx + k * cw + 3, y + 3, cw - 6, ch - 6)
                n, t = int(left[suit, r]), int(total[suit, r])
                box(s, cell, CARD_C if n else PANEL_HI, 8)
                text(s, f.big, n, SUIT_C[suit] if n else DIM, (cell.centerx, cell.y + 6), "midtop")
                text(s, f.small, f"/{t}", (110, 110, 110) if n else DIM, (cell.right - 6, cell.bottom - 4),
                     "bottomright")
            text(s, f.bold, f"{int(left[suit].sum())}/{int(total[suit].sum())}", DIM,
                 (gx + 13 * cw + 35, y + ch // 2), "center")
        y = gy + 4 * ch + 8
        for k, r in enumerate(ranks):
            text(s, f.small, f"{int(left[:, r].sum())}/{int(total[:, r].sum())}", DIM,
                 (gx + k * cw + cw // 2, y), "midtop")

    def draw_overlay(self, title, sub):
        s, f = self.screen, self.fonts
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 160))
        s.blit(veil, (0, 0))
        text(s, f.huge, title, TEXT, (GW // 2, 300), "center")
        text(s, f.body, sub, DIM, (GW // 2, 360), "center")

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
        if self.show_model:
            self.draw_model_panel()
        self.joker_menu.draw(s, self.fonts)        # last, so the open list sits on top
        if self.show_deck:
            self.draw_deck_view()
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and (
                        self.joker_menu.open or self.show_deck or self.seed_edit is not None):
                    self.joker_menu.open = self.show_deck = False   # Esc closes popups first
                    self.seed_edit = None
                    continue
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    self.release(e.pos)
                if e.type == pygame.MOUSEMOTION and self.drag:
                    self.drag[3] = e.pos
                if e.type == pygame.MOUSEWHEEL and self.joker_menu.open:
                    self.joker_menu.wheel(e.y)
                if e.type == pygame.KEYDOWN:
                    self.key(e.key)
            now = pygame.time.get_ticks()
            if self.ai_auto and not self.show_deck and now - self.ai_last >= AUTO_MS:
                self.ai_last = now
                self.model_step()
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="blind target scale (the training curriculum dial); 1.0 = real game")
    ap.add_argument("--params", default=PARAMS_PATH,
                    help="trained params .pkl for the model panel (default runs/params_run.pkl)")
    args = ap.parse_args()
    App(args.seed, args.scale, args.params).run()
