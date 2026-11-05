"""Play BalatroNET by hand — a pygame front end over the jitted env.

Every game action goes through env_step, the same function the agent trains
on. The GUI keeps no game rules of its own: it reads the state, draws it,
and turns clicks into balatroAction values. The only exceptions are the dev
buttons (give money / give a joker), which stage the state directly, the way
fixtures.py does in the tests.

    uv run --project D:/Claude/jax-learn python gui.py [--seed N]

Mouse: click cards to select, click a joker to select it (then SELL), click a
shop card to buy it, click a pack to open it, click a voucher to redeem it, click
a consumable to select it (then Use: a card Tarot or Spectral acts on the selected
hand cards; or Sell). In an open pack, click a card to take it (an Arcana Tarot or
a Spectral card acts on the selected cards), or Skip. Drag a joker or a hand card
to a new place (one MOVE_JOKER / MOVE_CARD insert step through env_step).
Blind select: Play the blind on deck, Skip it (Small / Big Blind) for its tag, or
Reroll the boss (Director's Cut / Retcon, $10). A face-down card shows its back;
a debuffed card (the boss) has a red cross; Cerulean Bell's forced card says so.
Keys:  1-8 toggle a card   Enter play (at the blind select: play the blind)   D discard
       R / S sort by rank / suit
       V deck view   N new run   Left / Right move the selected joker   Esc close a popup / quit
Model: P model view on/off (hides the panel, the odds and the pick outline)
       M take the model's pick   A auto-play on/off   G greedy / sampled pick
       O card odds on/off.  The panel on the right always shows the model's
       odds for the current state, at every stage.
       --params runs/params_run.pkl (default)   --scale 0.2 (the training curriculum scale)
Dev:   +$10, the Add Joker and Add Tarot / Planet dropdowns (scroll wheel when the list is long)
       L shows the hand levels.
Seed:  the seed box: click it, type a number, Enter starts that seed (Esc cancels)
       Restart = the same seed again   Random = a random seed   N = seed + 1
       One seed fixes the shuffle and the shops. The same seed with the same
       actions gives the same run. MOVE steps do not use the RNG.
Tests: T opens the scenario tests (scenarios.py): the panel on the right shows the rule,
       the steps and the checks; the board shows the state after each step.
       Up / Down  the previous / next test     Left / Right  a step back / forward
       Home / End  the start / the end         F  the next failing test
       Find test  search by name. The board stays playable; Home reloads the test.
       T again goes back to the run. A test runs when it is first shown; a background
       thread runs the rest (the count at the top of the panel).

The hand shows the env's lane order, which is the play order (and the order
Death reads). Sort Rank / Sort Suit reorder the hand once, with MOVE_CARD inserts
(at most hand size - 1 steps);
new cards land in the lanes the played cards left.
"""

import argparse
import os
import sys
import threading

import numpy as np
import jax
import jax.numpy as jnp
import pygame

from env import (balatroAction, env_step, reset, explain_play, preview, build_ctx, acquire,
                 give_cons, invariant_errors, MAX_HAND_SIZE, MAX_DECK_SIZE, STAGE_BLIND, STAGE_SHOP,
                 STAGE_LOST, STAGE_WON, STAGE_PACK, STAGE_SELECT, BLIND_NAMES, BLIND_BOSS, V_PLAY, V_DISCARD,
                 V_BUY, V_SELL, V_MOVE_JOKER, V_MOVE_CARD, V_REROLL, V_NEXT, V_USE, V_SELL_CONS, V_OPEN,
                 V_PICK, V_SKIP, V_REDEEM, WIN_ANTE, RANK_CHARS, SUIT_CHARS,
                 live_v, bankrupt_at, reroll_price, item_cost, pack_cost, joker_cost, shop_item_cost,
                 voucher_cost, blind_target, blind_dollars, card_debuffed)
from items import (CONS_NAMES, CONS_DESC, CONS_KIND, CONS_LO, CONS_HI, CK_TAROT, CK_PLANET, PACK_NAMES,
                   PACK_KIND, PACK_CHOOSE, PK_ARCANA, PK_CELESTIAL, PK_STANDARD, PK_BUFFOON, PK_SPECTRAL,
                   ENH_NAMES, ED_NAMES, SEAL_NAMES, IK_JOKER, IK_TAROT, IK_CARD, N_HAND_TYPES, hand_chips_mult,
                   BOSS_NAMES, BOSS_DESC, TAG_NAMES, TAG_DESC, VOUCHER_NAMES, VOUCHER_DESC, VOUCHER_ID,
                   BOSS_REROLL_COST, STAKE_NAMES, STAKE_DESC, N_STAKES, STK_ETERNAL, STK_PERISH, STK_RENTAL,
                   ST_NO_SMALL_REWARD)
from jokers import (JOKER_NAMES, JOKER_DESC, JOKER_COST, JOKERS, JOKER_RARITY, J_EMPTY,
                    MAX_JOKERS, rows_for, resolve, COMMON, UNCOMMON, RARE, LEGENDARY,
                    E_ADD_MULT, E_ADD_CHIPS, E_XMULT, E_ADD_MONEY, EFFECT_NAMES,
                    E_CHIPS_MULT, E_RETRIGGER, E_MULT_NOMINAL, E_MULT_RANDOM, E_XMULT_UNCOMMON,
                    E_NONE, E_SELL_VALUE, E_SELL_VALUE_ALL, E_BURGLAR, S_COUNTER, ST_TODO,
                    ST_LOYALTY)
from scoring import HAND_NAMES
from agent import (observe, encode, legal_mask, move_masks, masked_logits, sample_src, sample_dst, sample_picks,
                   pick_query, pick_bounds, uses_picker, init_params, action_name, is_move, OBS_DIM,
                   O_GLOB, N_ACTIONS, ACTION_VERB, ACTION_TARGET, A_PLAY, A_DISCARD, A_BUY, A_SELL,
                   A_MOVE_JOKER, A_MOVE_CARD, A_REROLL, A_OPEN, A_USE, A_SELL_CONS, A_PICK, A_SKIP)
from train import load_params, PARAMS_PATH, REWARD_SCALE

GW = 1280                  # the game area; the model panel sits to its right
PANEL_W = 320
TEST_W = 470               # the test panel (test mode) takes the model panel's place
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
RARITY_C  = {COMMON: (0, 147, 255), UNCOMMON: (53, 189, 134), RARE: (254, 95, 85),
             LEGENDARY: (178, 108, 187)}
BTN_PLAY  = (0, 147, 255)
BTN_DISC  = (254, 95, 85)
BTN_SHOP  = (243, 140, 50)
BTN_GREEN = (53, 189, 134)
RARITY_NAME = {COMMON: "Common", UNCOMMON: "Uncommon", RARE: "Rare", LEGENDARY: "Legendary"}
AI_C      = (90, 220, 230)    # the model's colour: panel edge, its pick
BAR_C     = (40, 110, 130)
TAROT_C   = (162, 104, 220)
PLANET_C  = (60, 170, 200)
SPECTRAL_C = (90, 110, 230)
VOUCHER_C = (220, 90, 140)
TAG_C     = (120, 200, 120)
BACK_C    = (52, 78, 150)     # a card's back (face down)
DEBUFF_C  = (230, 40, 40)
BLIND_C   = [(0, 120, 200), (230, 150, 40), (200, 60, 60)]   # Small, Big, Boss
PACK_C    = (210, 120, 60)
# enhancement: card face colour (None = the plain card) and a short tag
ENH_FACE  = {1: (205, 228, 250), 2: (250, 214, 210), 3: (240, 232, 250), 4: (215, 240, 248),
             5: (196, 202, 208), 6: (128, 128, 132), 7: (246, 220, 140), 8: (214, 244, 214)}
ENH_TAG   = {1: "+30", 2: "+4 Mult", 3: "Wild", 4: "Glass", 5: "Steel", 6: "Stone", 7: "Gold", 8: "Lucky"}
ED_EDGE   = {1: (120, 170, 255), 2: (255, 120, 170), 3: (255, 200, 60), 4: (40, 40, 40)}
STICKER_C = {STK_ETERNAL: (190, 120, 255), STK_PERISH: (90, 200, 230), STK_RENTAL: (240, 190, 60)}
STAKE_C   = [(225, 225, 225), (230, 70, 70), (70, 190, 110), (60, 60, 70), (70, 120, 230), (160, 80, 200),
             (240, 140, 50), (240, 200, 60)]
SEAL_C    = {1: (243, 185, 68), 2: (230, 60, 60), 3: (60, 120, 230), 4: (150, 80, 200)}


def odds_color(p):
    """Dim for unlikely, the model colour for likely."""
    return AI_C if p >= 0.5 else (DIM if p < 0.2 else TEXT)

HAND_LABEL = {"flush_five": "Flush Five", "flush_house": "Flush House", "five_kind": "Five of a Kind",
              "straight_flush": "Straight Flush", "four_kind": "Four of a Kind",
              "full_house": "Full House", "flush": "Flush", "straight": "Straight",
              "three_kind": "Three of a Kind", "two_pair": "Two Pair", "pair": "Pair",
              "high_card": "High Card"}

SUIT_WORD = ["Club", "Diamond", "Heart", "Spade"]
HAND_WORD = ["Flush Five", "Flush House", "Five of a Kind", "Straight Flush", "Four of a Kind",
             "Full House", "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair", "High Card"]
RANK_WORD = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]


def joker_text(jid, h, slot=None):
    """The joker's in-game text with this round's targets filled in."""
    todo = HAND_WORD[int(h.joker_state[slot]) % N_HAND_TYPES] if slot is not None else "(random hand)"
    idol = int(h.idol_card)
    return (JOKER_DESC[jid].replace("{castle}", SUIT_WORD[int(h.castle_suit)])
            .replace("{ancient}", SUIT_WORD[int(h.ancient_suit) % 4])
            .replace("{mail}", RANK_WORD[int(h.mail_rank)])
            .replace("{idol}", f"{RANK_WORD[idol % 13]} of {SUIT_WORD[idol // 13]}s")
            .replace("{todo}", todo))


def sticker_words(st, left=None):
    """[(word, color)] of a joker's stake stickers; left: a held Perishable joker's rounds left."""
    st, out = int(st), []
    if st & STK_ETERNAL:
        out.append(("Eternal", STICKER_C[STK_ETERNAL]))
    if st & STK_PERISH:
        out.append(("Perishable" if left is None else f"Perish {int(left)}" if left > 0 else "Perished",
                    STICKER_C[STK_PERISH]))
    if st & STK_RENTAL:
        out.append(("Rental", STICKER_C[STK_RENTAL]))
    return out


def effect_label(jid, v, h=None, slot=None):
    """The live value v, worded by the joker's EFFECT axis."""
    e = int(JOKERS.EFFECT[jid])
    counter = int(JOKERS.SOURCE[jid]) == S_COUNTER
    if int(JOKERS.STATE[jid]) == ST_LOYALTY and h is not None:
        left = (5 - (int(h.hand_played.sum()) - int(h.joker_state[slot]))) % 6
        return ("X4 Mult this hand" if left == 0 else f"{left} remaining"), MULT_C
    if e == E_ADD_MULT:
        return f"+{v:g} Mult", MULT_C
    if e == E_ADD_CHIPS:
        return f"+{v:g} Chips", CHIPS_C
    if e == E_XMULT:
        return f"X{v:g} Mult", MULT_C
    if e == E_ADD_MONEY:
        return f"+${v:g}", MONEY_C
    if e == E_CHIPS_MULT:
        return f"+{v:g} Chips +{float(JOKERS.VALUE2[jid]):g} Mult", CHIPS_C
    if e == E_RETRIGGER:
        return (f"{v:g} hands left" if counter else f"retrigger x{v:g}"), DIM
    if e == E_MULT_NOMINAL:
        return "+2x lowest held rank", MULT_C
    if e == E_MULT_RANDOM:
        return f"+0-{v:g} Mult", MULT_C
    if e == E_XMULT_UNCOMMON:
        return f"X{v:g} per Uncommon", MULT_C
    if e in (E_SELL_VALUE, E_SELL_VALUE_ALL):
        return f"+${v:g} sell", MONEY_C
    if e == E_BURGLAR:
        return f"+{v:g} hands", CHIPS_C
    if bool(JOKERS.HSIZE_V[jid]):
        return f"+{v:g} hand size", DIM
    if e == E_NONE:
        return "passive", DIM
    words = {"SPAWN_CARD": "adds a card", "SPAWN_CONSUMABLE": "makes a Tarot", "LEVEL_UP": "levels up",
             "MIDAS": "faces to Gold", "PERMA_CHIPS": f"+{v:g} card Chips", "DESTROY_DISCARD": f"+${v:g}"}
    return words.get(EFFECT_NAMES[e], f"{EFFECT_NAMES[e]} {v:g}"), DIM


# ---------------------------------------------------------------------------
# jitted helpers
# ---------------------------------------------------------------------------
@jax.jit
def live_values(state):
    """v per joker slot, as scoring would read it right now (resolve pass)."""
    return live_v(state)


_acquire = jax.jit(acquire)


@jax.jit
def model_view(params, state, key):
    """Everything the model panel shows for one state: the probability of every
    choice, the chance each card is picked (rows: PLAY, DISCARD, the sampled and
    the greedy choice; from 256 sampled pick sequences), the value, and a sampled
    and a greedy pick."""
    obs, legal, mmask = observe(state), legal_mask(state), move_masks(state)
    enc = encode(params, obs)
    hd = masked_logits(params, obs, legal, state.hand_valid, mmask, enc=enc)
    pa = jax.nn.softmax(hd.a)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    def pick(a, k, greedy=False):
        lo, hi = pick_bounds(state, a)
        return sample_picks(params, hd, pick_query(hd, a), lo, hi, state.hand_valid, k, greedy)[1]
    a_s, a_g = jax.random.categorical(k1, hd.a).astype(jnp.int32), jnp.argmax(pa).astype(jnp.int32)
    ps = jnp.stack([jax.vmap(lambda k: pick(a, k))(jax.random.split(k5, 256)).mean(0)
                    for a in (jnp.int32(A_PLAY), jnp.int32(A_DISCARD), a_s, a_g)])
    verb = lambda a: a
    # the move pointers: src first, then dst given that src (as act() samples them)
    src_s = sample_src(hd, a_s, k3)
    src_g = jnp.where(a_g == A_MOVE_JOKER, jnp.argmax(hd.ptr[0]), jnp.argmax(hd.ptr[2]))
    hd_s = masked_logits(params, obs, legal, state.hand_valid, mmask, src_s, enc)
    hd_g = masked_logits(params, obs, legal, state.hand_valid, mmask, src_g, enc)
    dst_g = jnp.where(a_g == A_MOVE_JOKER, jnp.argmax(hd_g.ptr[1]), jnp.argmax(hd_g.ptr[3]))
    return {"pa": pa, "ps": ps, "value": hd.value, "legal": legal,
            "a_sample": a_s,
            "sel_sample": pick(verb(a_s), k2),
            "src_sample": src_s, "dst_sample": sample_dst(hd_s, a_s, k4),
            "a_greedy": a_g,
            "sel_greedy": pick(verb(a_g), k2, True),
            "src_greedy": src_g, "dst_greedy": dst_g}


def load_model(path):
    """-> (params, label). A missing file gives a random policy, labelled as such."""
    if path and os.path.exists(path):
        params = load_params(path)
        if "Ep" not in params:
            print(f"{path}: saved before the consumables / packs layout; using a random policy")
            return init_params(jax.random.key(0)), "random (old params file)"
        if (params["W1"].shape[0] != OBS_DIM - O_GLOB or params["Wa"].shape[1] != N_ACTIONS
                or "layers" not in params):
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
    """Short card name for the log and the model panel: 10♥, K♠ (cid = rank + 13 * suit)."""
    r = "10" if RANK_CHARS[cid % 13] == "T" else RANK_CHARS[cid % 13]
    return r + SUIT_GLYPH[cid // 13]


class CardView:
    """A playing card as the GUI draws it (a snapshot: a played card may change or break)."""
    __slots__ = ("rank", "suit", "enh", "ed", "seal", "bonus")

    def __init__(self, rank, suit, enh=0, ed=0, seal=0, bonus=0.0):
        self.rank, self.suit, self.enh, self.ed, self.seal, self.bonus = (
            int(rank), int(suit), int(enh), int(ed), int(seal), float(bonus))

    @property
    def cid(self):
        return self.rank + 13 * self.suit

    def name(self):
        return "Stone" if self.enh == 6 else card_name(self.cid)


def inst_view(h, inst):
    """The CardView of card instance inst in host state h."""
    i = int(inst)
    return CardView(h.card_rank[i], h.card_suit[i], h.card_enh[i], h.card_ed[i], h.card_seal[i],
                    h.card_bonus[i])


CONS_WORD = ["Tarot", "Planet", "Spectral"]


def cons_color(cid):
    return [TAROT_C, PLANET_C, SPECTRAL_C][int(CONS_KIND[cid])]


def blind_name(h, i):
    """Small Blind, Big Blind, or the boss's name."""
    b = int(h.boss)
    return BLIND_NAMES[i] if i < BLIND_BOSS or b < 0 else BOSS_NAMES[b]


def tag_text(h, i):
    """Blind i's tag and what it does (an Orbital Tag names its poker hand)."""
    t = int(h.blind_tags[i])
    if t < 0:
        return "no tag", ""
    desc = TAG_DESC[t]
    if TAG_NAMES[t] == "Orbital Tag":
        desc = f"+3 levels: {HAND_WORD[int(h.orbital_hands[i])]}"
    return TAG_NAMES[t], desc


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


def clip(font, s, width):
    """s cut from the right until it fits width (a narrow card: up to 16 jokers in the row)."""
    while s and font.size(s)[0] > width:
        s = s[:-1]
    return s


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

    rows(query) returns [(label, right_text, stripe_color, enabled, key)] for the
    typed search text; it is called on every draw, so the list always matches the
    current state. pick(key) runs on a row click, or on Enter for the top row.
    head(query), if given, returns one more row (or None) that stays above the list
    when the list scrolls. on_open() runs each time the list opens.
    """
    ROW_H, MAX_ROWS, SEARCH_H = 30, 14, 34

    def __init__(self, rect, label, rows, pick, width=340, on_open=None, head=None):
        self.button = Button(rect, label, PANEL_HI)
        self.rows_q, self.pick, self.width, self.on_open, self.head_q = rows, pick, width, on_open, head
        self.open, self.scroll, self.query = False, 0, ""

    def rows(self):
        return self.rows_q(self.query)

    def head(self):
        return self.head_q(self.query) if self.head_q else None

    def menu_rect(self):
        n = max(min(len(self.rows()), self.MAX_ROWS), 1) + (self.head() is not None)
        b = self.button.rect
        h = self.SEARCH_H + n * self.ROW_H + 8
        return pygame.Rect(b.x, b.y - 6 - h, self.width, h)

    def search_rect(self):
        m = self.menu_rect()
        return pygame.Rect(m.x + 4, m.y + 4, m.w - 8, self.SEARCH_H - 6)

    def slots(self):
        """[(row, rect)]: the head row (if any), then the rows of the list that show."""
        m, hd = self.menu_rect(), self.head()
        shown = ([] if hd is None else [hd]) + self.rows()[self.scroll:self.scroll + self.MAX_ROWS]
        y0 = m.y + 4 + self.SEARCH_H
        return [(row, pygame.Rect(m.x + 4, y0 + k * self.ROW_H, m.w - 8, self.ROW_H - 2))
                for k, row in enumerate(shown)]

    def type(self, chars):
        self.query, self.scroll = self.query + chars, 0

    def key(self, k):
        """Search keys while open: Backspace edits, Enter picks the top enabled row."""
        if k == pygame.K_BACKSPACE:
            self.query, self.scroll = self.query[:-1], 0
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            ok = [r for r in self.rows() if r[3]]
            if ok:
                self.pick(ok[0][4])

    def wheel(self, dy):
        top = max(len(self.rows()) - self.MAX_ROWS, 0)
        self.scroll = min(max(self.scroll - dy, 0), top)

    def click(self, pos):
        """True when the dropdown used the click (then nothing else gets it)."""
        if self.button.rect.collidepoint(pos):
            self.open = not self.open
            self.query, self.scroll = "", 0
            if self.open and self.on_open:
                self.on_open()
            return True
        if not self.open:
            return False
        for row, r in self.slots():
            if r.collidepoint(pos) and row[3]:
                self.pick(row[4])
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
        rows, m, hd = self.rows(), self.menu_rect(), self.head()
        box(surf, m, PANEL, 8, PANEL_HI, 2)
        sr = self.search_rect()
        box(surf, sr, BG, 6, SEL_EDGE, 2)
        cursor = "_" if pygame.time.get_ticks() // 500 % 2 else ""
        text(surf, fonts.body, self.query + cursor if self.query else "type to search" + cursor,
             TEXT if self.query else DIM, (sr.x + 10, sr.centery), "midleft")
        if not rows:
            y = m.y + 4 + self.SEARCH_H + (0 if hd is None else self.ROW_H) + self.ROW_H // 2
            text(surf, fonts.small, "no match", DIM, (m.x + 16, y), "midleft")
        mouse = pygame.mouse.get_pos()
        for k, (row, r) in enumerate(self.slots()):
            label, right, stripe, ok, _ = row
            if ok and r.collidepoint(mouse):
                box(surf, r, PANEL_HI, 6)
            pygame.draw.rect(surf, stripe, (r.x + 4, r.y + 5, 5, r.h - 10), border_radius=2)
            text(surf, fonts.body, label, TEXT if ok else DIM, (r.x + 16, r.centery), "midleft")
            text(surf, fonts.small, right, DIM, (r.right - 8, r.centery), "midright")
            if k == 0 and hd is not None:            # a line under the head row: it does not scroll
                pygame.draw.line(surf, PANEL_HI, (r.x, r.bottom + 1), (r.right, r.bottom + 1), 2)
        if len(rows) > self.MAX_ROWS:
            text(surf, fonts.small, f"{self.scroll + 1}-{self.scroll + self.MAX_ROWS} of {len(rows)}",
                 DIM, (m.right - 6, m.y - 4), "bottomright")


# ---------------------------------------------------------------------------
# the test menu: one sub menu per kind of test
# ---------------------------------------------------------------------------
TEST_CATS = ["Jokers", "Enhancements", "Editions", "Seals", "Tarots", "Planets", "Spectrals", "Vouchers",
             "Packs", "Tags", "Bosses", "Stakes", "Rules"]


def test_place(sc):
    """(sub menu, sort key in it) of a scenario. The sub menu is the scenario's group, but the
    Cards group splits into Enhancements / Editions / Seals, and a Scoring row (one PLAY of the
    joker suite) goes where the first name of its subject belongs (a joker edition, as
    "Foil joker": Editions; no joker, Hand levels ...: Rules). Jokers sort by name, card
    modifiers in the game's order, the rest as scenarios.py lists them."""
    parts = sc.subject.split(" + ")
    first = parts[0]
    ed = next((p[:-6] for p in parts if p.endswith(" joker") and p[:-6] in ED_NAMES), None)
    if sc.group == "Scoring" and ed:
        return "Editions", (ED_NAMES.index(ed), "")
    if sc.group in ("Jokers", "Cards", "Scoring") and first in JOKER_NAMES:
        return "Jokers", (0, first.lower())
    if sc.group not in ("Cards", "Scoring"):
        return sc.group, (0, "")
    if first.endswith(" Seal") and first[:-5] in SEAL_NAMES:
        return "Seals", (SEAL_NAMES.index(first[:-5]), "")
    if first in ED_NAMES:
        return "Editions", (ED_NAMES.index(first), "")
    if first in ENH_NAMES:
        return "Enhancements", (ENH_NAMES.index(first), "")
    return "Rules", (0, "")


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
        self.stake = 1                 # 1 White .. 8 Gold (K: the next stake, a new run)
        self.log = []
        self.show_deck = False
        self.seed_edit = None          # the typed digits while the seed box is open
        self.drag = None               # [kind "joker"/"card", slot or lane, press pos, mouse pos]
        self.joker_menu = Dropdown((84, 710, 170, 30), "Add Joker   ", self.joker_menu_rows,
                                   lambda jid: self.give_joker(jid))
        self.cons_menu = Dropdown((558, 710, 150, 30), "Add Card    ", self.cons_menu_rows,
                                  lambda cid: self.give_cons(cid), width=360)
        self.show_levels = False
        self.test_mode = False         # T: the scenario tests
        self.test_scs = []             # the scenarios, listed on the first T, in the test menu's order
        self.test_cats = []            # the sub menu of each test
        self.test_by_cat = {}          # sub menu -> its test indices
        self.test_cat = None           # the sub menu open in the test menu (None: the list of them)
        self.tests = None              # [scenarios.Run or None (not run yet)]
        self.tests_said = False        # the pass count is in the log
        self.test_i = self.test_k = 0  # the test shown, and the state after test_k of its steps
        self.test_dirty = False        # the board was changed by hand in test mode
        self.saved = None              # (state, seed, show_model) of the run test mode left
        self.test_menu = Dropdown((GW + 18, H - 54, 210, 36), "Find test   ", self.test_menu_rows,
                                  self.pick_test, width=TEST_W - 34, on_open=lambda: self.pick_test(None),
                                  head=self.test_menu_head)
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
        self.state = reset(jax.random.key(seed), 52, 8, self.scale, self.stake)
        self.selected = set()          # lanes
        self.joker_sel = None          # joker slot
        self.cons_sel = None           # consumable slot
        self.last_play = None          # (PlayResult on host, cards, pre-state hands)
        self.cashout = None            # breakdown text
        self.refresh()
        self.say(f"New run, seed {seed}, {STAKE_NAMES[self.stake - 1]}, target scale {self.scale:g}. Choose a "
                 f"blind: the {BLIND_NAMES[0]} needs {self.targets[0]:,} chips.")

    def refresh(self):
        self.h = host(self.state)
        self.v = np.asarray(live_values(self.state))
        self.floor = int(bankrupt_at(self.state))
        self.reroll = int(reroll_price(self.state))
        h, st = self.h, self.state
        # the blind select's targets and rewards; the shop's prices; the boss's debuffs
        self.targets = [int(blind_target(st.ante, i, st.target_scale, st.boss, stake=st.stake)) for i in range(3)]
        self.rewards = [int(blind_dollars(i, st.boss, st.stake)) for i in range(3)]
        self.prices = [int(shop_item_cost(st, i)) for i in range(len(h.shop_slots))]
        self.pack_prices = [int(pack_cost(st, h.pack_slots[i], h.pack_free[i])) for i in range(len(h.pack_slots))]
        self.v_price = int(voucher_cost(st))
        self.debuff = np.asarray(card_debuffed(st, jnp.arange(MAX_DECK_SIZE)))
        key = jax.random.fold_in(jax.random.key(self.seed), self.ai_n)
        self.ai = jax.device_get(model_view(self.params, self.state, key))
        self.ai_hand = None            # the hand the model's pick would make (PLAY / DISCARD)
        a, sel, _, _ = self.ai_choice()
        if a == A_PLAY and sel.any():
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

    def action(self, verb, target=0, dest=0):
        return balatroAction(verb=jnp.int32(verb), select=self.sel_mask(), target=jnp.int32(target),
                             dest=jnp.int32(dest))

    def step(self, verb, target=0, dest=0, quiet=False):
        pre = self.h
        self.test_dirty = self.test_mode
        if verb == V_MOVE_CARD:                # the selection follows its cards
            ids = {int(pre.hand[l]) for l in self.selected}
        a = self.action(verb, target, dest)
        if verb == V_PLAY:
            res = jax.device_get(explain_play(self.state, a.select))
            cards = [inst_view(pre, c) for c, ok in zip(res.cards, res.valid) if ok]
            self.last_play = (res, cards)
        self.state, reward, done = env_step(self.state, a)
        self.selected.clear()
        if verb == V_MOVE_CARD:
            hand, valid = np.asarray(self.state.hand), np.asarray(self.state.hand_valid)
            self.selected = {l for l in range(MAX_HAND_SIZE) if valid[l] and int(hand[l]) in ids}
        if verb == V_MOVE_JOKER and self.joker_sel == target:
            self.joker_sel = dest
        if verb in (V_USE, V_SELL_CONS):
            self.cons_sel = None
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
            if not np.array_equal(np.asarray(pre.shop_slots), np.asarray(h.shop_slots)):
                self.say(f"Bought {self.item_name(pre, target)} for ${int(pre.money) - int(h.money)}.")
            else:
                self.say("Can't buy that (money, or no free slot).")
        elif verb == V_OPEN:
            if int(h.game_stage) == STAGE_PACK:
                pid = int(h.pack_id)
                self.say(f"Opened {PACK_NAMES[pid]}: take {int(h.pack_left)} of the cards, or Skip.")
            else:
                self.say("Can't open that pack (money).")
        elif verb == V_PICK:
            took = int(h.pack_left) != int(pre.pack_left) or int(h.game_stage) != STAGE_PACK
            self.say(f"Took {self.pack_item_name(pre, target)}." if took else
                     "Can't take that (no room, or select the cards the Tarot needs).")
        elif verb == V_SKIP and int(pre.game_stage) == STAGE_PACK:
            self.say("Skipped the pack.")
        elif verb == V_SKIP:
            b = int(pre.blind_index)
            if int(h.skips) > int(pre.skips):
                self.say(f"Skipped the {BLIND_NAMES[b]} for the {tag_text(pre, b)[0]}.")
            else:
                self.say("The Boss Blind can't be skipped.")
        elif verb == V_REDEEM:
            v = int(pre.shop_vouchers[target])
            if int(h.money) != int(pre.money) or bool(h.vouchers[max(v, 0)]) != bool(pre.vouchers[max(v, 0)]):
                self.say(f"Redeemed {VOUCHER_NAMES[v]} for ${int(pre.money) - int(h.money)}.")
            else:
                self.say("Can't redeem that (money).")
        elif verb == V_USE:
            cid = int(pre.consumables[target])
            used = not np.array_equal(np.asarray(pre.consumables), np.asarray(h.consumables))
            self.say(f"Used {CONS_NAMES[cid]}." if used else
                     f"Can't use {CONS_NAMES[cid]} now (it needs {int(CONS_LO[cid])}-{int(CONS_HI[cid])} "
                     f"selected cards, room, or a joker).")
        elif verb == V_SELL_CONS:
            self.say(f"Sold {CONS_NAMES[int(pre.consumables[target])]} for ${int(h.money) - int(pre.money)}.")
        elif verb == V_SELL:
            self.say(f"Sold {JOKER_NAMES[int(pre.jokers[target])]} for ${int(h.money) - int(pre.money)}.")
            self.joker_sel = None
        elif verb == V_MOVE_CARD and not quiet:
            moved = not np.array_equal(np.asarray(pre.hand), np.asarray(h.hand))
            self.say(f"Moved {card_name(int(pre.hand[target]))} to lane {dest + 1}." if moved
                     else "Can't move that card.")
        elif verb == V_MOVE_JOKER:
            moved = not np.array_equal(np.asarray(pre.jokers), np.asarray(h.jokers))
            self.say(f"Moved {JOKER_NAMES[int(pre.jokers[target])]} to slot {dest + 1}." if moved
                     else "Can't move that joker.")
        elif verb == V_REROLL and int(pre.game_stage) == STAGE_SELECT:
            if int(h.boss) != int(pre.boss) or int(h.money) != int(pre.money):
                self.say(f"Rerolled the boss for ${int(pre.money) - int(h.money)}: {BOSS_NAMES[int(h.boss)]}.")
            else:
                self.say("Can't reroll the boss (it needs Director's Cut once per ante, or Retcon, and $10).")
        elif verb == V_REROLL:
            self.say(f"Rerolled for ${int(pre.money) - int(h.money)}." if int(h.money) != int(pre.money)
                     else "Not enough money to reroll.")
        elif verb == V_NEXT:
            self.last_play = None
            self.cashout = None
            if int(h.game_stage) == STAGE_SELECT:
                self.say(f"Choose the next blind (ante {int(h.ante)}).")
            else:
                self.say(f"Ante {int(h.ante)}, {blind_name(h, int(h.blind_index))}: "
                         f"score {int(h.blind_target):,} chips.")
        if int(h.game_stage) == STAGE_SELECT and int(pre.game_stage) != STAGE_SELECT and verb != V_NEXT:
            self.say("Back at the blind select.")

        if int(pre.game_stage) == STAGE_BLIND and int(h.game_stage) in (STAGE_SHOP, STAGE_WON):
            res = self.last_play[0] if self.last_play is not None and verb == V_PLAY else None
            beaten = res is not None and int(pre.chips_scored) + int(res.score) >= int(pre.blind_target)
            blind = int(blind_dollars(pre.blind_index, pre.boss, pre.stake)) if beaten else 0
            hands = int(h.hands)
            rest = int(h.last_cashout) - blind - hands
            self.cashout = (f"Cash out ${int(h.last_cashout)}:  blind ${blind}  +  "
                            f"{hands} hand(s) left ${hands}")
            if rest:
                self.cashout += f"  +  interest, jokers, tags ${rest}"
            if not beaten:
                self.cashout += "  (Mr. Bones)"
            self.say(f"Blind defeated! {self.cashout}")
        if int(h.game_stage) == STAGE_LOST and int(pre.game_stage) != STAGE_LOST:
            self.say("Out of hands. Game over.")
        if int(h.game_stage) == STAGE_WON and int(pre.game_stage) != STAGE_WON:
            self.say("You won the run!")

    def move_joker(self, i, k):
        """Insert the joker in slot i at slot k: one MOVE_JOKER step."""
        if i == k or int(self.h.game_stage) not in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT):
            return
        self.step(V_MOVE_JOKER, i, k)

    def move_card(self, lane, k):
        """Insert the card in `lane` at place k of the visible hand: one MOVE_CARD step."""
        dest = self.hand_order()[k]
        if dest != lane and int(self.h.game_stage) in (STAGE_BLIND, STAGE_PACK):
            self.step(V_MOVE_CARD, lane, dest)

    def item_name(self, h, i):
        """The name of shop slot i (a joker with its edition, or a Tarot / Planet)."""
        it, kind, ed = int(h.shop_slots[i]), int(h.shop_kind[i]), int(h.shop_ed[i])
        if it < 0:
            return "-"
        if kind == IK_JOKER:
            return (f"{ED_NAMES[ed]} " if ed else "") + JOKER_NAMES[it]
        if kind == IK_CARD:
            return CardView(h.shop_rank[i], h.shop_suit[i], h.shop_enh[i], ed).name()
        return CONS_NAMES[it]

    def pack_item_name(self, h, i):
        it = int(h.pack_items[i])
        kind = int(PACK_KIND[max(int(h.pack_id), 0)])
        if it < 0:
            return "-"
        if kind == PK_BUFFOON:
            ed = int(h.pack_ed[i])
            return (f"{ED_NAMES[ed]} " if ed else "") + JOKER_NAMES[it]
        if kind == PK_STANDARD:
            return CardView(h.pack_rank[i], h.pack_suit[i], h.pack_enh[i], h.pack_ed[i], h.pack_seal[i]).name()
        return CONS_NAMES[it]

    def sort_hand(self, by):
        """Sort the lanes with MOVE_CARD inserts: place p gets its card, p = 0, 1, ...
        At most hand_size - 1 steps. Holes go to the right."""
        h = self.h
        if int(h.game_stage) not in (STAGE_BLIND, STAGE_PACK):
            return
        valid = [l for l in range(len(h.hand_valid)) if bool(h.hand_valid[l])]
        n = max(int(h.hand_size), valid[-1] + 1 if valid else 0)          # env.card_lanes
        cards = [int(h.hand[l]) if bool(h.hand_valid[l]) else None for l in range(n)]
        rk = lambda i: -1 if int(h.card_enh[i]) == 6 else int(h.card_rank[i])      # Stone: last
        st = lambda i: int(h.card_suit[i])
        key = ((lambda c: (-rk(c), st(c), c)) if by == "rank" else (lambda c: (-st(c), -rk(c), c)))
        k = lambda c: (1,) if c is None else (0,) + key(c)
        want = sorted(cards, key=k)
        moves = []
        for p in range(n):
            if want[p] is None:
                break                          # only holes are left
            src = cards.index(want[p], p)
            if src != p:
                cards.insert(p, cards.pop(src))
                moves.append((src, p))
        for src, p in moves:
            self.step(V_MOVE_CARD, src, p, quiet=True)
        self.say(f"Sorted by {by} ({len(moves)} MOVE_CARD steps).")

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

    def give_cons(self, cid):
        """[dev] a Tarot / Planet into a free consumable slot (env.give_cons, the spawn path)."""
        before = int((self.h.consumables >= 0).sum())
        self.state = give_cons(self.state, jnp.int32(cid))
        self.refresh()
        ok = int((self.h.consumables >= 0).sum()) > before
        self.say(f"[dev] gave {CONS_NAMES[cid]}" if ok else "[dev] consumable slots full")

    def cons_menu_rows(self, query=""):
        h = self.h
        q = query.lower().strip()
        full = int((h.consumables >= 0).sum()) >= int(h.consumable_slots)
        return [(name, CONS_WORD[int(CONS_KIND[cid])], cons_color(cid),
                 not full, cid) for cid, name in enumerate(CONS_NAMES) if q in name.lower()]

    def give_joker(self, jid):
        # env.acquire: the same code path BUY uses, minus the cost
        self.state, ok = _acquire(self.state, jnp.int32(jid))
        self.refresh()
        self.say(f"[dev] gave {JOKER_NAMES[jid]}" if bool(ok) else "[dev] joker slots full")

    # --- test mode: the scenario tests, one step at a time ------------------------
    def load_tests(self):
        """The scenario list, in the order of the test menu (TEST_CATS, then test_place), so
        < Test / Test > go through a sub menu and on to the next one. A test runs when it is
        first shown; a background thread runs the rest, for the pass count and Next fail."""
        import scenarios
        scs = scenarios.all_scenarios()
        place = [test_place(sc) for sc in scs]
        rank = {c: k for k, c in enumerate(TEST_CATS)}
        order = sorted(range(len(scs)), key=lambda i: (rank.get(place[i][0], len(TEST_CATS)), place[i], i))
        self.test_scs = [scs[i] for i in order]
        self.test_cats = [place[i][0] for i in order]
        self.test_by_cat = {}
        for i, c in enumerate(self.test_cats):
            self.test_by_cat.setdefault(c, []).append(i)
        self.tests = [None] * len(self.test_scs)
        threading.Thread(target=lambda: [self.run_test(i) for i in range(len(self.tests))],
                         daemon=True).start()

    def run_test(self, i):
        """The scenarios.Run of test i (it keeps each state); runs it now if it has not run.
        The background thread calls this too: no pygame here."""
        import scenarios
        if self.tests[i] is None:
            self.tests[i] = scenarios.Run(self.test_scs[i])
        return self.tests[i]

    def poll_tests(self):
        """Once the background thread has run every test: the count in the log."""
        if self.tests is None or self.tests_said or any(r is None for r in self.tests):
            return
        self.tests_said = True
        bad = sum(not r.ok for r in self.tests)
        self.say(f"Tests: {len(self.tests) - bad} of {len(self.tests)} pass.")

    def toggle_tests(self):
        if self.test_mode:                         # back to the run test mode left
            self.test_mode = False
            self.state, self.seed, self.show_model = self.saved
            self.test_menu.open = False
            self.screen = pygame.display.set_mode((W if self.show_model else GW, H))
            self.selected, self.joker_sel, self.cons_sel = set(), None, None
            self.last_play = self.cashout = None
            self.refresh()
            return
        self.saved = (self.state, self.seed, self.show_model)
        if self.tests is None:
            self.load_tests()
        self.test_mode, self.show_model, self.ai_auto = True, False, False
        self.screen = pygame.display.set_mode((GW + TEST_W, H))
        self.show_test(self.test_i, 0)

    def show_test(self, i=None, k=None):
        """Show test i after k of its steps (None: keep). The next step's cards are selected."""
        import scenarios
        n = len(self.tests)
        self.test_i = (self.test_i if i is None else i) % n
        run = self.run_test(self.test_i)
        self.test_k = min(max(self.test_k if k is None else k, 0), len(run.raw) - 1) if run.raw else 0
        self.test_dirty = False
        self.selected, self.joker_sel, self.cons_sel = set(), None, None
        self.last_play = self.cashout = None
        if not run.raw:                            # the build failed: nothing to show
            return
        k = self.test_k
        self.state = run.raw[k]
        h = run.views[k].h
        if k < len(run.sc.steps):                  # the next step: its cards and its slot
            st = run.sc.steps[k]
            try:
                self.selected = set(scenarios._lanes(h, st.cards))
            except ValueError:
                pass
            if st.verb in (V_USE, V_SELL_CONS):
                self.cons_sel = st.target
            if st.verb == V_SELL:
                self.joker_sel = st.target
        if k > 0 and run.plays[k - 1] is not None:
            res, pre = run.plays[k - 1], run.views[k - 1].h
            self.last_play = (res, [inst_view(pre, c) for c, ok in zip(res.cards, res.valid) if ok])
        if k > 0 and int(run.views[k - 1].h.game_stage) == STAGE_BLIND and int(h.game_stage) == STAGE_SHOP:
            self.cashout = f"Cash out ${int(h.last_cashout)}"
        self.refresh()

    def test_step(self, dk):
        if self.test_dirty:                        # leave the hand-made state: back to the recorded one
            return self.show_test(k=self.test_k)
        self.show_test(k=self.test_k + dk)

    def next_fail(self):
        n = len(self.tests)
        bad = lambda i: self.tests[i] is not None and not self.tests[i].ok
        nxt = next(((self.test_i + d) % n for d in range(1, n + 1) if bad((self.test_i + d) % n)), None)
        done = sum(r is not None for r in self.tests)
        if nxt is not None:
            self.show_test(nxt, 0)
        elif done < n:
            self.say(f"No failing test so far ({done} of {n} have run).")
        else:
            self.say("Every test passes.")

    def test_menu_head(self, query=""):
        """In a sub menu, with no search text: the back row (it stays at the top of the list)."""
        if query.strip() or self.test_cat is None:
            return None
        return ("< All tests", f"{self.test_cat}, {len(self.test_by_cat[self.test_cat])} tests", PANEL_HI,
                True, None)

    def test_menu_rows(self, query=""):
        """No search text: the sub menus (with their pass state), or the tests of the open one.
        Search text: the matching tests of every sub menu."""
        q = query.lower().strip()
        rows = []
        if not q and self.test_cat is None:
            for cat, ids in self.test_by_cat.items():
                runs = [self.tests[i] for i in ids]
                bad = sum(r is not None and not r.ok for r in runs)
                c = MULT_C if bad else DIM if any(r is None for r in runs) else BTN_GREEN
                rows.append((cat, f"{bad} fail of {len(ids)}  >" if bad else f"{len(ids)} tests  >", c, True, cat))
            return rows
        for i in range(len(self.test_scs)) if q else self.test_by_cat[self.test_cat]:
            r, sc, cat = self.tests[i], self.test_scs[i], self.test_cats[i]
            label = f"{sc.subject}: {sc.title}"
            if q and q not in label.lower() and q not in cat.lower() and q not in sc.group.lower():
                continue
            label = label if len(label) <= 46 else label[:44] + ".."
            right = cat if q else "shown" if i == self.test_i else ""
            rows.append((label, right, DIM if r is None else BTN_GREEN if r.ok else MULT_C, True, i))
        return rows

    def pick_test(self, key):
        """A sub menu name opens that sub menu, scrolled to the test shown if it is in it; None
        (the back row, or the menu opening) shows the list of sub menus; an index shows that test."""
        menu = self.test_menu
        if key is None or isinstance(key, str):
            self.test_cat, menu.query, menu.scroll = key, "", 0
            ids = self.test_by_cat.get(key, [])
            if self.test_i in ids:
                menu.scroll = min(max(ids.index(self.test_i) - 3, 0), max(len(ids) - menu.MAX_ROWS, 0))
            return
        menu.open = False
        self.show_test(key, 0)

    # --- the model ------------------------------------------------------------------
    def ai_choice(self):
        """-> (flat action index, select lanes (32,) bool, move src, move dst) for the current mode."""
        m = "greedy" if self.ai_greedy else "sample"
        return (int(self.ai[f"a_{m}"]), np.asarray(self.ai[f"sel_{m}"]),
                int(self.ai[f"src_{m}"]), int(self.ai[f"dst_{m}"]))

    def ai_played_lanes(self, sel):
        """The lanes env_step really acts on: the first 5 selected valid lanes."""
        return [i for i in range(MAX_HAND_SIZE) if sel[i] and bool(self.h.hand_valid[i])][:5]

    def choice_label(self, a, src=None, dst=None):
        """src / dst = the move pointers; without them a MOVE is named by its verb only."""
        h = self.h
        if A_BUY <= a < A_SELL:
            i = a - A_BUY
            if int(h.shop_slots[i]) < 0:
                return action_name(a)
            return f"BUY {self.item_name(h, i)} (${self.prices[i]})"
        if A_OPEN <= a < A_USE:
            pid = int(h.pack_slots[a - A_OPEN])
            return f"OPEN {PACK_NAMES[pid]} (${self.pack_prices[a - A_OPEN]})" if pid >= 0 else action_name(a)
        if A_USE <= a < A_SELL_CONS:
            cid = int(h.consumables[a - A_USE])
            return f"USE {CONS_NAMES[cid]}" if cid >= 0 else action_name(a)
        if A_SELL_CONS <= a < A_PICK:
            cid = int(h.consumables[a - A_SELL_CONS])
            return f"SELL {CONS_NAMES[cid]} (+${int(h.cons_sell[a - A_SELL_CONS])})" if cid >= 0 else action_name(a)
        if A_PICK <= a < A_SKIP:
            return f"TAKE {self.pack_item_name(h, a - A_PICK)}"
        if a == A_SKIP:
            return "SKIP the pack" if int(h.game_stage) == STAGE_PACK else f"SKIP the {BLIND_NAMES[int(h.blind_index)]}"
        if a == A_MOVE_CARD:
            if src is None:
                return "MOVE a card"
            if bool(h.hand_valid[src]):
                return f"MOVE {inst_view(h, h.hand[src]).name()} to lane {dst + 1}"
            return action_name(a, src, dst)
        if a == A_MOVE_JOKER:
            if src is None:
                return "MOVE a joker"
            if int(h.jokers[src]) >= 0:
                return f"MOVE {JOKER_NAMES[int(h.jokers[src])]} to slot {dst + 1}"
            return action_name(a, src, dst)
        if A_SELL <= a < A_MOVE_JOKER:
            i = a - A_SELL
            jid = int(h.jokers[i])
            return f"SELL {JOKER_NAMES[jid]} (+${int(h.joker_sell_value[i])})" if jid >= 0 else action_name(a)
        if a == A_REROLL:
            return f"REROLL the boss (${BOSS_REROLL_COST})" if int(h.game_stage) == STAGE_SELECT else f"REROLL (${self.reroll})"
        return action_name(a)

    def model_step(self):
        """Carry out the model's pick through env_step, like any human action."""
        if int(self.h.game_stage) not in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT):
            self.ai_auto = False
            return
        a, sel, src, dst = self.ai_choice()
        p = float(self.ai["pa"][a])
        verb, target = int(ACTION_VERB[a]), int(ACTION_TARGET[a])
        if is_move(a):
            target, dest = src, dst
        else:
            dest = 0
        self.say(f"[model] {self.choice_label(a, src, dst)}  p={p:.0%}")
        self.selected = set(self.ai_played_lanes(sel)) if bool(uses_picker(jnp.int32(a))) else set()
        self.joker_sel = None
        self.step(verb, target, dest, quiet=is_move(a))

    def joker_menu_rows(self, query=""):
        h = self.h
        q = query.lower().strip()
        full = int((h.jokers != J_EMPTY).sum()) >= int(h.joker_slots)
        held = list(int(j) for j in h.jokers)
        return [(name + (f"  (x{held.count(jid)})" if jid in held else ""),
                 f"${int(JOKER_COST[jid])}  {RARITY_NAME[int(JOKER_RARITY[jid])]}",
                 RARITY_C[int(JOKER_RARITY[jid])], not full, jid)
                for jid, name in enumerate(JOKER_NAMES) if q in name.lower()]

    # --- layout ---------------------------------------------------------------
    def hand_order(self):
        """Every lane with a card, in lane order (= play order). A card past hand_size (Certificate,
        DNA) shows too."""
        h = self.h
        return [i for i in range(MAX_HAND_SIZE) if bool(h.hand_valid[i])]

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

    def joker_step(self):
        n = max(int((self.h.jokers != J_EMPTY).sum()), 5)
        return min(150, 730 // n)

    def joker_rects(self):
        held = [i for i in range(MAX_JOKERS) if int(self.h.jokers[i]) >= 0]
        st = self.joker_step()
        return [(i, pygame.Rect(330 + k * st, 28, min(138, st - 6), 150)) for k, i in enumerate(held)]

    def cons_rects(self):
        n = int(self.h.consumable_slots)
        return [(i, pygame.Rect(1076 + i * 98, 28, 92, 150)) for i in range(n)]

    def shop_rects(self):
        n = int(self.h.shop_size)
        return [(i, pygame.Rect(560 + i * 170, 238, 150, 175)) for i in range(n)]

    def pack_rects(self):
        return [(i, pygame.Rect(560 + i * 170, 450, 150, 120)) for i in range(len(self.h.pack_slots))]

    def voucher_rects(self):
        """The shop's voucher slots, right of the packs (1 or 2 wide cards, up to 4 narrow ones)."""
        used = [i for i, v in enumerate(self.h.shop_vouchers) if int(v) >= 0]
        n = max(used[-1] + 1 if used else 1, 2 if len(used) > 1 else 1)
        w = (330 - 8 * (n - 1)) // n if n > 2 else 158
        return [(i, pygame.Rect(900 + i * (w + 8), 450, w, 120)) for i in range(n)]

    def select_rects(self):
        """The blind select: Small, Big and Boss Blind panels."""
        return [pygame.Rect(340 + i * 300, 222, 280, 420) for i in range(3)]

    def pack_item_rects(self):
        return [(i, pygame.Rect(492 + i * 148, 238, 140, 175)) for i in range(len(self.h.pack_items))]

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
            b["reroll"] = Button((340, 334, 190, 70), f"Reroll ${self.reroll}", BTN_GREEN,
                                 int(h.free_rerolls) > 0 or int(h.money) - self.reroll >= self.floor)
        if stage == STAGE_SELECT:
            on = int(h.blind_index)
            r = self.select_rects()[on]
            b["select"] = Button((r.x + 14, r.bottom - 64, 120 if on < BLIND_BOSS else 252, 50), "Play", BTN_PLAY)
            if on < BLIND_BOSS:
                b["skip_blind"] = Button((r.x + 146, r.bottom - 64, 120, 50), "Skip", BTN_DISC)
            vs = h.vouchers
            retcon, cut = bool(vs[VOUCHER_ID["Retcon"]]), bool(vs[VOUCHER_ID["Director's Cut"]])
            if retcon or cut:
                br = self.select_rects()[BLIND_BOSS]
                can = (retcon or not bool(h.boss_rerolled)) and int(h.money) - BOSS_REROLL_COST >= self.floor
                b["reroll_boss"] = Button((br.x + 14, br.bottom - 120, 252, 44), f"Reroll Boss ${BOSS_REROLL_COST}",
                                          BTN_GREEN, can)
        if stage == STAGE_PACK:
            b["skip"] = Button((338, 372, 140, 46), "Skip", BTN_DISC)
            if int(h.hand_valid.sum()):
                b["sort_rank"] = Button((940, 668, 110, 40), "Sort Rank", PANEL_HI)
                b["sort_suit"] = Button((1060, 668, 110, 40), "Sort Suit", PANEL_HI)
        if self.cons_sel is not None and stage in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT):
            r = dict(self.cons_rects())[self.cons_sel]
            b["use"] = Button((r.x, r.bottom + 6, 44, 30), "Use", BTN_PLAY)
            b["sell_cons"] = Button((r.x + 48, r.bottom + 6, 44, 30), f"${int(h.cons_sell[self.cons_sel])}", BTN_GREEN)
        if self.joker_sel is not None and stage in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT):
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
        b["levels"] = Button((928, 716, 86, 30), "Levels", AI_C if self.show_levels else PANEL_HI)
        b["tests"] = Button((716, 716, 104, 30), "Tests (T)", AI_C if self.test_mode else PANEL_HI)
        b["stake"] = Button((824, 716, 100, 30), STAKE_NAMES[self.stake - 1].split()[0] + " (K)", PANEL_HI,
                            not self.test_mode)
        b["model"] = Button((1020, 710, 120, 36), "Model (P)", AI_C if self.show_model else PANEL_HI,
                            not self.test_mode)
        if self.test_mode:
            x = GW + 18
            run = self.tests[self.test_i]
            b["t_prev"] = Button((x, H - 100, 104, 36), "< Test", PANEL_HI)
            b["t_next"] = Button((x + 110, H - 100, 104, 36), "Test >", PANEL_HI)
            b["t_back"] = Button((x + 220, H - 100, 104, 36), "< Step", BTN_PLAY, self.test_k > 0 or self.test_dirty)
            b["t_fwd"] = Button((x + 330, H - 100, 104, 36), "Step >", BTN_PLAY,
                                self.test_k < len(run.raw) - 1 or self.test_dirty)
            b["t_fail"] = Button((x + 220, H - 54, 214, 36), "Next fail (F)", BTN_DISC)
        if not self.show_model:
            return b
        # model panel
        x = GW + 14
        live = stage in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT)
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
        if self.show_deck or self.show_levels:
            self.show_deck = self.show_levels = False     # any click closes the popup
            return
        if self.seed_edit is not None and not self.buttons()["seed"].hit(pos):
            self.seed_edit = None          # a click outside the seed box cancels it
        if self.joker_menu.click(pos) or self.cons_menu.click(pos):
            return
        if self.test_mode and self.test_menu.click(pos):
            return
        for name, btn in self.buttons().items():
            if btn.hit(pos):
                return self.press(name)
        if stage in (STAGE_BLIND, STAGE_PACK):
            for lane, r in reversed(self.card_rects()):     # topmost card first
                if r.collidepoint(pos):
                    self.drag = ["card", lane, pos, pos]   # release() selects or moves it
                    return
        if stage == STAGE_SHOP:
            for i, r in self.shop_rects():
                if r.collidepoint(pos) and int(h.shop_slots[i]) >= 0:
                    return self.step(V_BUY, i)
            for i, r in self.pack_rects():
                if r.collidepoint(pos) and int(h.pack_slots[i]) >= 0:
                    return self.step(V_OPEN, i)
            for i, r in self.voucher_rects():
                if r.collidepoint(pos) and int(h.shop_vouchers[i]) >= 0:
                    return self.step(V_REDEEM, i)
        if stage == STAGE_PACK:
            for i, r in self.pack_item_rects():
                if r.collidepoint(pos) and int(h.pack_items[i]) >= 0:
                    return self.step(V_PICK, i)
        for i, r in self.cons_rects():
            if r.collidepoint(pos) and int(h.consumables[i]) >= 0:
                self.cons_sel = None if self.cons_sel == i else i
                return
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
        elif name == "use":
            self.step(V_USE, self.cons_sel)
        elif name == "sell_cons":
            self.step(V_SELL_CONS, self.cons_sel)
        elif name in ("skip", "skip_blind"):
            self.step(V_SKIP)
        elif name == "select":
            self.step(V_NEXT)
        elif name == "reroll_boss":
            self.step(V_REROLL)
        elif name == "levels":
            self.show_levels = not self.show_levels
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
        elif name == "stake":
            self.next_stake()
        elif name == "deck":
            self.show_deck = True
        elif name == "model":
            self.toggle_model()
        elif name == "tests":
            self.toggle_tests()
        elif name == "t_prev":
            self.show_test(self.test_i - 1, 0)
        elif name == "t_next":
            self.show_test(self.test_i + 1, 0)
        elif name == "t_back":
            self.test_step(-1)
        elif name == "t_fwd":
            self.test_step(+1)
        elif name == "t_fail":
            self.next_fail()
        elif name == "ai_step":
            self.model_step()
        elif name == "ai_auto":
            self.ai_auto = not self.ai_auto
        elif name == "ai_mode":
            self.ai_greedy = not self.ai_greedy
            self.refresh()
        elif name == "ai_odds":
            self.ai_odds = not self.ai_odds

    def next_stake(self):
        """The next stake (Gold goes back to White): a new run with the same seed."""
        self.stake = self.stake % N_STAKES + 1
        self.new_run(self.seed)
        self.say(f"{STAKE_NAMES[self.stake - 1]}: {STAKE_DESC[self.stake - 1]}.")

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
        if k == pygame.K_l:
            self.show_levels = not self.show_levels
            return
        if self.show_deck or self.show_levels:
            return                         # a popup eats game keys
        if k == pygame.K_t:
            return self.toggle_tests()
        if self.test_mode:
            nav = {pygame.K_UP: lambda: self.show_test(self.test_i - 1, 0),
                   pygame.K_DOWN: lambda: self.show_test(self.test_i + 1, 0),
                   pygame.K_LEFT: lambda: self.test_step(-1), pygame.K_RIGHT: lambda: self.test_step(+1),
                   pygame.K_HOME: lambda: self.show_test(k=0), pygame.K_END: lambda: self.show_test(k=10 ** 6),
                   pygame.K_f: self.next_fail}
            if k in nav:
                return nav[k]()
            if k in (pygame.K_p, pygame.K_m, pygame.K_a, pygame.K_g, pygame.K_o):
                return                     # no model in test mode
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
        if k == pygame.K_k and not self.test_mode:
            return self.next_stake()
        if k == pygame.K_n:
            self.new_run(self.seed + 1)
        elif stage == STAGE_SELECT:
            if k in (pygame.K_RETURN, pygame.K_SPACE):
                self.step(V_NEXT)
        elif stage in (STAGE_BLIND, STAGE_PACK):
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
    def draw_back(self, rect, selected=False):
        """A card face down: only its back shows."""
        surf = self.screen
        box(surf, rect.move(0, 5), (20, 44, 34), 9)
        box(surf, rect, BACK_C, 9, SEL_EDGE if selected else CARD_EDGE, 3 if selected else 2)
        inner = rect.inflate(-16, -16)
        pygame.draw.rect(surf, (90, 120, 200), inner, 2, border_radius=6)
        pygame.draw.line(surf, (90, 120, 200), inner.topleft, inner.bottomright, 2)
        pygame.draw.line(surf, (90, 120, 200), inner.topright, inner.bottomleft, 2)

    def mark_debuffed(self, rect):
        """The boss debuffs it: a red cross."""
        r = rect.inflate(-18, -18)
        pygame.draw.line(self.screen, DEBUFF_C, r.topleft, r.bottomright, 5)
        pygame.draw.line(self.screen, DEBUFF_C, r.topright, r.bottomleft, 5)

    def draw_card(self, rect, card, selected=False, scored=None, down=False, debuffed=False, forced=False):
        """card: a CardView. The face colour shows the enhancement, the edge the
        edition, the dot (top right) the seal; perma chips sit at the bottom. down: its back;
        debuffed: a red cross; forced: Cerulean Bell's card."""
        surf, f = self.screen, self.fonts
        if forced:
            text(surf, f.small, "forced", DEBUFF_C, (rect.centerx, rect.bottom + 8), "midtop")
        if down:
            return self.draw_back(rect, selected)
        box(surf, rect.move(0, 5), (20, 44, 34), 9)
        edge = SEL_EDGE if selected else ED_EDGE.get(card.ed, CARD_EDGE)
        box(surf, rect, ENH_FACE.get(card.enh, CARD_C), 9, edge, 3 if selected or card.ed else 2)
        if card.ed == 3:                   # Polychrome: a second, inner edge
            pygame.draw.rect(surf, (120, 220, 120), rect.inflate(-8, -8), 2, border_radius=7)
        if card.enh != 6:
            rank, suit = card.rank, card.suit
            col = SUIT_C[suit] if card.enh != 3 else (150, 80, 200)
            r = "10" if RANK_CHARS[rank] == "T" else RANK_CHARS[rank]
            text(surf, f.rank, r, col, (rect.x + 8, rect.y + 4))
            text(surf, f.glyph_small, SUIT_GLYPH[suit], col, (rect.x + 9, rect.y + 30))
            text(surf, f.glyph, SUIT_GLYPH[suit], col, rect.center, "center")
        if card.enh:
            text(surf, f.small, ENH_TAG[card.enh], (40, 40, 40) if card.enh != 6 else TEXT,
                 (rect.centerx, rect.bottom - 22), "midtop")
        if card.seal:
            pygame.draw.circle(surf, SEAL_C[card.seal], (rect.right - 12, rect.y + 12), 7)
        if card.bonus:
            text(surf, f.small, f"+{card.bonus:g}", CHIPS_C, (rect.right - 6, rect.bottom - 40), "topright")
        if scored is False:
            dim = pygame.Surface(rect.size, pygame.SRCALPHA)
            dim.fill((0, 0, 0, 90))
            surf.blit(dim, rect)
        if debuffed:
            self.mark_debuffed(rect)

    def draw_sidebar(self):
        s, f, h = self.screen, self.fonts, self.h
        box(s, (12, 12, 300, 690), PANEL, 14)
        stage = int(h.game_stage)
        bi = int(h.blind_index)
        box(s, (24, 24, 276, 92), BLIND_C[bi], 10)
        if stage == STAGE_SELECT:
            text(s, f.big, "Blind select", TEXT, (162, 40), "midtop")
            text(s, f.body, f"{BLIND_NAMES[bi]} on deck", TEXT, (162, 80), "midtop")
        else:
            name = blind_name(h, bi)
            text(s, fit(f, name, 260) if len(name) > 14 else f.big, name, TEXT, (162, 40), "midtop")
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
        if label and self.preview is not None:
            lvl = int(h.hand_level[int(self.preview.handtype)])
            label += f"  lvl.{lvl}"
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

        y += 70
        idol = int(h.idol_card)
        text(s, f.small, f"Idol {RANK_CHARS[idol % 13]}{SUIT_CHARS[idol // 13]}  Mail {RANK_CHARS[int(h.mail_rank)]}"
                         f"  Ancient {SUIT_CHARS[int(h.ancient_suit) % 4]}  Castle {SUIT_CHARS[int(h.castle_suit)]}",
             DIM, (26, y))
        y += 20
        # the boss of this ante, the held tags, the redeemed vouchers
        info = []
        st = int(h.stake)
        if st > 1:
            info.append((f"{STAKE_NAMES[st - 1]}: {STAKE_DESC[st - 1]}, and the stakes below", STAKE_C[st - 1]))
        if int(h.boss) >= 0:
            off = " (disabled)" if bool(h.boss_disabled) and stage == STAGE_BLIND and bi == BLIND_BOSS else ""
            info.append((f"Boss: {BOSS_NAMES[int(h.boss)]}{off}: {BOSS_DESC[int(h.boss)]}", BLIND_C[2]))
        tags = [TAG_NAMES[int(t)] for t in h.tags if int(t) >= 0]
        if tags:
            info.append(("Tags: " + ", ".join(tags), TAG_C))
        vs = [VOUCHER_NAMES[i] for i in range(len(VOUCHER_NAMES)) if bool(h.vouchers[i])]
        if vs:
            info.append(("Vouchers: " + ", ".join(vs), VOUCHER_C))
        for line, col in info:
            for part in wrap(f.small, line, 270)[:2]:
                text(s, f.small, part, col, (26, y))
                y += 17
        y += 4
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
        st = self.joker_step()
        cards = [(i, pygame.Rect(330 + k * st, 28, min(138, st - 6), 150))
                 for k, i in enumerate(self.drag_order())]
        if held_slot is not None:          # draw the held card last, under the mouse
            box(s, next(r for i, r in cards if i == held_slot), PANEL_HI, 10)   # its drop gap
            cards = [c for c in cards if c[0] != held_slot]
            dx = self.drag[3][0] - self.drag[2][0]
            home = dict(self.joker_rects())[held_slot]
            cards.append((held_slot, home.move(dx, 8)))
        for i, r in cards:
            jid = int(h.jokers[i])
            name = JOKER_NAMES[jid]
            ed = int(h.joker_ed[i])
            if bool(h.joker_down[i]):                       # Amber Acorn: face down
                self.draw_back(r, i in (self.joker_sel, held_slot))
                continue
            box(s, r, PANEL, 10, SEL_EDGE if i in (self.joker_sel, held_slot) else RARITY_C[int(JOKER_RARITY[jid])], 3)
            if ed:
                pygame.draw.rect(s, ED_EDGE[ed], r.inflate(-6, -6), 2, border_radius=8)
                text(s, f.small, clip(f.small, ED_NAMES[ed], r.w - 8), ED_EDGE[ed] if ed != 4 else DIM,
                     (r.centerx, r.bottom - 48), "midtop")
            stk = sticker_words(h.joker_sticker[i], h.joker_perish[i])
            if stk:
                self.draw_stickers(r, stk, r.bottom - (64 if ed else 48))
            text(s, fit(f, name, r.w - 10), name, TEXT, (r.centerx, r.y + 8), "midtop")
            yy = r.y + 36
            for line in wrap(f.small, joker_text(jid, h, i), r.w - 14)[:4]:
                text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
                yy += 17
            label, col = effect_label(jid, float(self.v[i]), h, i)
            if bool(h.joker_debuff[i]):                     # Crimson Heart
                self.mark_debuffed(r)
                label, col = "disabled", DEBUFF_C
            elif int(h.joker_sticker[i]) & STK_PERISH and int(h.joker_perish[i]) <= 0:
                self.mark_debuffed(r)                       # Perishable: debuffed for good
                label, col = "perished", DEBUFF_C
            if f.bold.size(f"now {label}")[0] <= r.w - 4:
                text(s, f.bold, f"now {label}", col, (r.centerx, r.bottom - 30), "midtop")
            else:                                           # a narrow card: the label alone, 2 lines
                for k, line in enumerate(wrap(f.small, label, r.w - 4)[:2]):
                    text(s, f.small, clip(f.small, line, r.w - 4), col, (r.centerx, r.bottom - 31 + 14 * k), "midtop")

    def draw_stickers(self, r, words, y):
        """A joker's stake stickers, one line across the card (each word in its color)."""
        f = self.fonts
        widths = [f.small.size(w)[0] for w, _ in words]
        gap = 6
        while sum(widths) + gap * (len(words) - 1) > r.w - 8 and any(len(w) > 4 for w, _ in words):
            words = [(w[:max(len(w) - 1, 4)], c) for w, c in words]       # a narrow card: shorter words
            widths = [f.small.size(w)[0] for w, _ in words]
        x = r.centerx - (sum(widths) + gap * (len(words) - 1)) // 2
        for (w, c), wd in zip(words, widths):
            text(self.screen, f.small, w, c, (x, y))
            x += wd + gap

    def draw_item(self, r, name, desc, edge, price=None, ok=True, ed=0, tag=None):
        """One shop / pack card: name, text, price."""
        s, f = self.screen, self.fonts
        box(s, r, PANEL_HI, 10, edge, 3)
        if ed:
            pygame.draw.rect(s, ED_EDGE[ed], r.inflate(-6, -6), 2, border_radius=8)
        text(s, fit(f, name, r.w - 10), name, TEXT, (r.centerx, r.y + 8), "midtop")
        yy = r.y + 34
        if tag:
            text(s, f.small, tag, edge, (r.centerx, yy), "midtop")
            yy += 17
        for line in wrap(f.small, desc, r.w - 14)[:4]:
            text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
            yy += 17
        if price is not None:
            text(s, f.big, f"${price}", MONEY_C if ok else DIM, (r.centerx, r.bottom - 40), "midtop")

    def draw_shop(self):
        s, f, h = self.screen, self.fonts, self.h
        box(s, (322, 222, 920, 210), PANEL, 14)
        text(s, f.bold, "SHOP", MONEY_C, (1230, 230), "topright")
        for i, r in self.shop_rects():
            it = int(h.shop_slots[i])
            if it < 0:
                box(s, r, PANEL_HI, 10)
                text(s, f.small, "sold out", DIM, r.center, "center")
                continue
            kind, ed = int(h.shop_kind[i]), int(h.shop_ed[i])
            cost = self.prices[i]
            ok = int(h.money) - cost >= self.floor
            if kind == IK_JOKER:
                self.draw_item(r, JOKER_NAMES[it], joker_text(it, h), RARITY_C[int(JOKER_RARITY[it])],
                               cost, ok, ed, ED_NAMES[ed] if ed else None)
                stk = sticker_words(h.shop_sticker[i])
                if stk:
                    self.draw_stickers(r, stk, r.bottom - 58)
            elif kind == IK_CARD:                            # Magic Trick: a playing card
                box(s, r, PANEL_HI, 10, CARD_EDGE, 2)
                cr = pygame.Rect(0, 0, 76, 106)
                cr.midtop = (r.centerx, r.y + 8)
                card = CardView(h.shop_rank[i], h.shop_suit[i], h.shop_enh[i], ed)
                self.draw_card(cr, card)
                text(s, f.big, f"${cost}", MONEY_C if ok else DIM, (r.centerx, r.bottom - 40), "midtop")
            else:
                self.draw_item(r, CONS_NAMES[it], CONS_DESC[it], cons_color(it), cost, ok,
                               tag=CONS_WORD[int(CONS_KIND[it])])
        box(s, (322, 440, 920, 140), PANEL, 14)
        text(s, f.bold, "PACKS", PACK_C, (338, 452))
        text(s, f.bold, "VOUCHERS", VOUCHER_C, (338, 480))
        text(s, f.small, "click to redeem", DIM, (338, 506))
        for i, r in self.pack_rects():
            pid = int(h.pack_slots[i])
            if pid < 0:
                box(s, r, PANEL_HI, 10)
                text(s, f.small, "opened", DIM, r.center, "center")
                continue
            cost = self.pack_prices[i]
            kind = int(PACK_KIND[pid])
            what = ["Tarot cards", "Planet cards", "playing cards", "Jokers", "Spectral cards"][kind]
            box(s, r, PANEL_HI, 10, PACK_C, 3)
            text(s, fit(f, PACK_NAMES[pid], r.w - 10), PACK_NAMES[pid], TEXT, (r.centerx, r.y + 8), "midtop")
            text(s, f.small, f"choose {int(PACK_CHOOSE[pid])}: {what}", DIM, (r.centerx, r.y + 36), "midtop")
            text(s, f.big, f"${cost}", MONEY_C if int(h.money) - cost >= self.floor else DIM,
                 (r.centerx, r.bottom - 40), "midtop")
        for i, r in self.voucher_rects():                  # click to redeem
            v = int(h.shop_vouchers[i])
            if v < 0:
                box(s, r, PANEL_HI, 10)
                text(s, f.small, "voucher", DIM, r.center, "center")
                continue
            ok = int(h.money) - self.v_price >= self.floor
            box(s, r, PANEL_HI, 10, VOUCHER_C, 3)
            yy = r.y + 6
            for line in wrap(f.small, VOUCHER_NAMES[v], r.w - 8)[:2]:
                text(s, f.small, line, TEXT, (r.centerx, yy), "midtop")
                yy += 16
            if r.w > 120:
                for line in wrap(f.small, VOUCHER_DESC[v], r.w - 10)[:3]:
                    text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
                    yy += 16
            text(s, f.bold, f"${self.v_price}", MONEY_C if ok else DIM, (r.centerx, r.bottom - 26), "midtop")
        if self.cashout:
            text(s, f.body, self.cashout, MONEY_C, (782, 590), "midtop")

    def draw_select(self):
        """The blind select: the Small, Big and Boss Blind of the ante, their targets, rewards
        and tags (Skip takes the tag), the boss's ability."""
        s, f, h = self.screen, self.fonts, self.h
        on = int(h.blind_index)
        for i, r in enumerate(self.select_rects()):
            box(s, r, PANEL, 14, BLIND_C[i] if i == on else PANEL_HI, 4 if i == on else 2)
            name = blind_name(h, i)
            box(s, (r.x + 10, r.y + 10, r.w - 20, 46), BLIND_C[i] if i >= on else PANEL_HI, 10)
            text(s, fit(f, name, r.w - 30), name, TEXT, (r.centerx, r.y + 22), "midtop")
            state = "on deck" if i == on else ("done" if i < on else "upcoming")
            text(s, f.small, state, AI_C if i == on else DIM, (r.centerx, r.y + 62), "midtop")
            text(s, f.small, "Score at least", DIM, (r.centerx, r.y + 86), "midtop")
            text(s, f.big, f"{self.targets[i]:,}", MULT_C, (r.centerx, r.y + 104), "midtop")
            no_reward = i == 0 and int(h.stake) >= ST_NO_SMALL_REWARD
            text(s, f.body, "No reward" if no_reward else f"Reward ${self.rewards[i]}", DIM if no_reward else MONEY_C,
                 (r.centerx, r.y + 144), "midtop")
            yy = r.y + 180
            if on <= i < BLIND_BOSS:                     # a blind not played yet: Skip takes its tag
                tname, tdesc = tag_text(h, i)
                text(s, f.bold, f"Skip: {tname}", TAG_C, (r.centerx, yy), "midtop")
                yy += 26
                for line in wrap(f.small, tdesc, r.w - 30)[:4]:
                    text(s, f.small, line, DIM, (r.centerx, yy), "midtop")
                    yy += 17
            elif i == BLIND_BOSS and int(h.boss) >= 0:
                for line in wrap(f.body, BOSS_DESC[int(h.boss)], r.w - 30)[:4]:
                    text(s, f.body, line, TEXT, (r.centerx, yy), "midtop")
                    yy += 21
        if int(h.skips):
            text(s, f.small, f"Blinds skipped this run: {int(h.skips)}", DIM, (782, 650), "midtop")

    def draw_pack(self):
        """An open pack: its cards (click one to take it), how many are left to take."""
        s, f, h = self.screen, self.fonts, self.h
        pid = int(h.pack_id)
        kind = int(PACK_KIND[pid])
        box(s, (322, 222, 920, 210), PANEL, 14)
        y = 230                                        # the left column: name, cards to take, hint; Skip below
        for line in wrap(f.bold, PACK_NAMES[pid].upper(), 144)[:2]:
            text(s, f.bold, line, PACK_C, (338, y))
            y += 24
        text(s, f.small, f"take {int(h.pack_left)} more", DIM, (338, y + 2))
        y += 22
        if kind in (PK_ARCANA, PK_SPECTRAL):
            for line in wrap(f.small, "select hand cards first for a card Tarot / Spectral", 144)[:3]:
                text(s, f.small, line, DIM, (338, y))
                y += 16
        for i, r in self.pack_item_rects():
            it = int(h.pack_items[i])
            if it < 0:
                continue
            if kind == PK_STANDARD:
                card = CardView(h.pack_rank[i], h.pack_suit[i], h.pack_enh[i], h.pack_ed[i], h.pack_seal[i])
                cr = pygame.Rect(0, 0, 96, 134)
                cr.center = r.center
                self.draw_card(cr, card)
                bits = [x for x in (ENH_NAMES[card.enh], ED_NAMES[card.ed], SEAL_NAMES[card.seal] and
                                    SEAL_NAMES[card.seal] + " Seal") if x]
                text(s, f.small, ", ".join(bits) or "plain", DIM, (r.centerx, r.bottom + 2), "midtop")
            elif kind == PK_BUFFOON:
                ed = int(h.pack_ed[i])
                self.draw_item(r, JOKER_NAMES[it], joker_text(it, h), RARITY_C[int(JOKER_RARITY[it])],
                               ed=ed, tag=ED_NAMES[ed] if ed else None)
                stk = sticker_words(h.pack_sticker[i])
                if stk:
                    self.draw_stickers(r, stk, r.bottom - 24)
            else:
                self.draw_item(r, CONS_NAMES[it], CONS_DESC[it], cons_color(it))

    def draw_consumables(self):
        s, f, h = self.screen, self.fonts, self.h
        n = int((h.consumables >= 0).sum())
        text(s, f.small, f"Consumables {n}/{int(h.consumable_slots)}", DIM, (1076, 8))
        for i, r in self.cons_rects():
            cid = int(h.consumables[i])
            if cid < 0:
                box(s, r, PANEL_HI, 10)
                continue
            box(s, r, PANEL, 10, SEL_EDGE if i == self.cons_sel else cons_color(cid), 3)
            yy = r.y + 8
            for line in wrap(f.small, CONS_NAMES[cid], r.w - 10)[:2]:
                text(s, f.small, line, TEXT, (r.centerx, yy), "midtop")
                yy += 17
            for line in wrap(f.small, CONS_DESC[cid], r.w - 10)[:4 if int(h.cons_ed[i]) == 4 else 5]:
                text(s, f.small, line, DIM, (r.centerx, yy + 4), "midtop")
                yy += 16
            if int(h.cons_ed[i]) == 4:                       # Perkeo's copy: Negative, no slot
                text(s, f.small, "Negative", DIM, (r.centerx, r.bottom - 38), "midtop")
            if int(h.cons_sell[i]) > 1:                      # Gift Card raised it
                text(s, f.small, f"sells ${int(h.cons_sell[i])}", MONEY_C, (r.centerx, r.bottom - 20), "midtop")

    def draw_levels(self):
        """The hand levels: chips x mult per hand type, and times played."""
        s, f, h = self.screen, self.fonts, self.h
        sw = s.get_width()
        veil = pygame.Surface((sw, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        s.blit(veil, (0, 0))
        chips, mult = hand_chips_mult(jnp.asarray(h.hand_level))
        x0, y0 = sw // 2 - 260, 120
        box(s, (x0, y0, 520, 52 + 34 * N_HAND_TYPES), PANEL, 14, PANEL_HI, 2)
        text(s, f.big, "Hand levels", TEXT, (sw // 2, y0 + 8), "midtop")
        for k in range(N_HAND_TYPES):
            y = y0 + 52 + 34 * k
            played = int(h.hand_played[k])
            col = TEXT if (k >= 3 or played) else DIM
            text(s, f.body, f"lvl.{int(h.hand_level[k])}", MONEY_C, (x0 + 20, y))
            text(s, f.body, HAND_WORD[k], col, (x0 + 90, y))
            text(s, f.bold, f"{int(chips[k])}", CHIPS_C, (x0 + 340, y), "topright")
            text(s, f.bold, "X", MULT_C, (x0 + 356, y))
            text(s, f.bold, f"{int(mult[k])}", MULT_C, (x0 + 380, y))
            text(s, f.small, f"played {played}", DIM, (x0 + 500, y + 2), "topright")

    def draw_play_area(self):
        s, f, h = self.screen, self.fonts, self.h
        if self.last_play is not None:
            res, cards = self.last_play
            mask = [bool(m) for m, ok in zip(res.mask, res.valid) if ok]
            n = len(cards)
            x0 = 780 - (n * 84) // 2
            for k, card in enumerate(cards):
                r = pygame.Rect(x0 + k * 84, 240, 76, 106)
                self.draw_card(r, card, scored=mask[k])
            text(s, f.bold, f"{hand_label(res.handtype)}:  {float(res.chips):g} x "
                            f"{float(res.mult):g}  =  {int(res.score):,}",
                 TEXT, (780, 360), "midtop")
        deck_left = int(h.deck_size) - int(h.deck_ptr)
        text(s, f.small, f"Deck {deck_left}/{int(h.deck_size)}", DIM, (1240, 640), "topright")

    def draw_hand(self):
        a, sel, _, _ = self.ai_choice()
        uses = bool(uses_picker(jnp.int32(a)))
        picked = set(self.ai_played_lanes(sel)) if uses else set()
        row = self.odds_row(a)
        held = self.drag_moved("card")
        cards = self.card_rects(self.card_drag_order())
        if held is not None:               # a gap at the drop place; the held card follows the mouse
            box(self.screen, dict(cards)[held], PANEL_HI, 9)
            cards = [c for c in cards if c[0] != held]
            home = dict(self.card_rects())[held]
            cards.append((held, home.move(self.drag[3][0] - self.drag[2][0], -40)))
        h = self.h
        for lane, r in cards:
            inst = int(h.hand[lane])
            self.draw_card(r, inst_view(h, inst), selected=lane in self.selected, down=bool(h.hand_down[lane]),
                           debuffed=bool(self.debuff[inst]), forced=inst == int(h.forced_card))
            if not (self.ai_odds and self.show_model):
                continue
            if lane in picked:
                pygame.draw.rect(self.screen, AI_C, r.inflate(10, 10), 3, border_radius=12)
            p = float(self.ai["ps"][row][lane])
            tag = pygame.Rect(0, 0, 52, 22)
            tag.midbottom = (r.centerx, r.y - 7)
            box(self.screen, tag, PANEL, 6, AI_C if lane in picked else None, 2)
            text(self.screen, self.fonts.small, f"{p:.0%}", odds_color(p), tag.center, "center")

    def odds_row(self, a):
        """The ps row for the card odds: the model's choice when it takes cards, else PLAY."""
        if bool(uses_picker(jnp.int32(a))):
            return 3 if self.ai_greedy else 2
        return 0

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

        a, sel, src, dst = self.ai_choice()
        live = int(h.game_stage) in (STAGE_BLIND, STAGE_SHOP, STAGE_PACK, STAGE_SELECT)
        box(s, (x, 70, PANEL_W - 38, 74), PANEL_HI, 10)
        if live:
            text(s, f.small, "would choose", DIM, (x + 10, 76))
            text(s, f.bold, self.choice_label(a, src, dst), TEXT, (x + 10, 94))
            text(s, f.bold, f"{float(self.ai['pa'][a]):.0%}", AI_C, (x + PANEL_W - 50, 94), "topright")
            if bool(uses_picker(jnp.int32(a))):
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
        if live and bool(h.hand_valid.any()):
            y = 548
            row = self.odds_row(a)
            text(s, f.small, f"card odds (if {action_name(a) if row >= 2 else 'PLAY'})", DIM, (x, y - 2))
            order = self.hand_order()
            cell = (PANEL_W - 38) // max(len(order), 1)
            for k, lane in enumerate(order):
                p = float(self.ai["ps"][row][lane])
                card = inst_view(h, h.hand[lane])
                rk = "St" if card.enh == 6 else ("10" if RANK_CHARS[card.rank] == "T" else RANK_CHARS[card.rank])
                cx = x + k * cell + cell // 2
                text(s, f.small, rk, TEXT, (cx - 5, y + 16), "midtop")
                text(s, f.glyph_tiny, SUIT_GLYPH[card.suit], SUIT_C[card.suit] if card.suit != 3 else TEXT,
                     (cx + 9, y + 15), "midtop")
                text(s, f.small, f"{p:.0%}", odds_color(p), (cx, y + 34), "midtop")

    def draw_test_panel(self):
        """The test shown: its rule, its steps (the state on the board is marked) and its checks."""
        import scenarios
        s, f = self.screen, self.fonts
        x0, w = GW + 4, TEST_W - 14
        box(s, (x0, 12, w, H - 24), PANEL, 14, AI_C, 2)
        x, right, width = x0 + 14, x0 + w - 14, w - 28
        n = len(self.tests)
        done = [r for r in self.tests if r is not None]
        good = sum(r.ok for r in done)
        run = self.tests[self.test_i]
        sc = run.sc
        text(s, f.bold, "TESTS", AI_C, (x, 22))
        if len(done) < n:                              # the background thread is still running them
            text(s, f.bold, f"{good} / {len(done)} pass  ({len(done)} of {n} run)",
                 DIM if good == len(done) else MULT_C, (right, 22), "topright")
        else:
            text(s, f.bold, f"{good} / {n} pass", BTN_GREEN if good == n else MULT_C, (right, 22), "topright")
        ids = self.test_by_cat[self.test_cats[self.test_i]]
        text(s, f.small, f"{self.test_cats[self.test_i]}  -  test {ids.index(self.test_i) + 1} of {len(ids)}",
             DIM, (x, 48))
        text(s, f.bold, "PASS" if run.ok else "FAIL", BTN_GREEN if run.ok else MULT_C, (right, 48), "topright")
        y = 70
        bottom = H - 112                               # the buttons start below this
        for line in wrap(f.bold, sc.subject, width):
            text(s, f.bold, line, TEXT, (x, y))
            y += 24
        for line in wrap(f.body, sc.title, width):
            text(s, f.body, line, TEXT, (x, y))
            y += 22
        y += 4
        for line in wrap(f.small, sc.expect, width):
            text(s, f.small, line, DIM, (x, y))
            y += 17
        if self.test_dirty:
            y += 4
            text(s, f.small, "Changed by hand: Home or a step key reloads the test.", SEL_EDGE, (x, y))
            y += 17
        if run.error:
            y += 4
            for line in wrap(f.small, "ERROR " + run.error, width):
                text(s, f.small, line, MULT_C, (x, y))
                y += 17
        y += 8
        text(s, f.small, "STEPS  (the board shows the marked state)", DIM, (x, y))
        y += 20
        for k, label in enumerate(["start"] + run.labels):
            if y > bottom - 40:
                break
            here = k == self.test_k and not self.test_dirty
            if here:
                box(s, (x - 6, y - 2, width + 12, 22), PANEL_HI, 6, AI_C, 2)
            col = TEXT if k <= self.test_k else DIM
            text(s, f.small, f"{k}" if k else "", DIM, (x, y + 1))
            text(s, f.body if here else f.small, label, col, (x + 22, y))
            y += 24
        y += 6
        if y < bottom - 20:
            text(s, f.small, "CHECKS  (after the last step)", DIM, (x, y))
            y += 20
        for label, got, want, ok in run.results:
            if y > bottom - 18:
                text(s, f.small, "...", DIM, (x, y))
                break
            col = BTN_GREEN if ok else MULT_C
            pygame.draw.circle(s, col, (x + 5, y + 9), 5)
            lines = wrap(f.small, f"{label}:  {scenarios.fmt(got)}" + ("" if ok else f"   want {scenarios.fmt(want)}"),
                         width - 18)
            for line in lines[:2]:
                text(s, f.small, line, TEXT if ok else MULT_C, (x + 16, y))
                y += 17
            y += 3

    def draw_deck_view(self):
        """Suit x rank grid. Big number = cards left to draw; small = in the full deck."""
        s, f, h = self.screen, self.fonts, self.h
        ptr, size = int(h.deck_ptr), int(h.deck_size)
        deck = [int(c) for c in h.deck]
        total = np.zeros((4, 13), int)
        left = np.zeros((4, 13), int)
        stones = [0, 0]
        enh = np.zeros(9, int)
        for k, c in enumerate(deck[:size]):
            e = int(h.card_enh[c])
            enh[e] += 1
            if e == 6:
                stones[0] += 1
                stones[1] += k >= ptr
                continue
            total[int(h.card_suit[c]), int(h.card_rank[c])] += 1
            left[int(h.card_suit[c]), int(h.card_rank[c])] += k >= ptr

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
        mods = "   ".join(f"{ENH_NAMES[e]} {int(enh[e])}" for e in range(1, 9) if enh[e])
        text(s, f.small, (f"Stone cards {stones[1]}/{stones[0]}   " if stones[0] else "")
             + (f"Enhanced: {mods}" if mods else "No enhanced cards"), DIM, (sw // 2, y + 22), "midtop")

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
        self.draw_consumables()
        if stage == STAGE_SHOP:
            self.draw_shop()
        elif stage == STAGE_SELECT:
            self.draw_select()
        elif stage == STAGE_PACK:
            self.draw_pack()
            if bool(h.hand_valid.any()):
                self.draw_hand()
        else:
            self.draw_play_area()
            if stage == STAGE_BLIND:
                self.draw_hand()
        buttons = self.buttons()
        if stage in (STAGE_LOST, STAGE_WON) and self.test_mode:
            text(s, self.fonts.big, "RUN LOST" if stage == STAGE_LOST else "RUN WON", MULT_C, (782, 600), "midtop")
        elif stage in (STAGE_LOST, STAGE_WON):
            reached = f"Ante {int(h.ante)}, round {int(h.round)}, seed {self.seed}"
            self.draw_overlay("GAME OVER" if stage == STAGE_LOST else "YOU WIN", reached)
        if self.show_model:
            self.draw_model_panel()                # before the buttons, so its box does not cover them
        if self.test_mode:
            self.draw_test_panel()
        for btn in buttons.values():
            btn.draw(s, self.fonts)
        text(s, self.fonts.small, "dev", DIM, (14, 694))
        self.joker_menu.draw(s, self.fonts)        # last, so the open list sits on top
        self.cons_menu.draw(s, self.fonts)
        if self.test_mode:
            self.test_menu.draw(s, self.fonts)
        if self.show_deck:
            self.draw_deck_view()
        if self.show_levels:
            self.draw_levels()
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and (
                        self.joker_menu.open or self.cons_menu.open or self.test_menu.open or self.show_deck
                        or self.show_levels or self.seed_edit is not None):
                    self.joker_menu.open = self.cons_menu.open = self.test_menu.open = False   # Esc closes popups first
                    self.show_deck = self.show_levels = False
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
                menu = next((m for m in (self.joker_menu, self.cons_menu, self.test_menu) if m.open), None)
                if e.type == pygame.MOUSEWHEEL and menu:
                    menu.wheel(e.y)
                if e.type == pygame.TEXTINPUT and menu:
                    menu.type(e.text)
                if e.type == pygame.KEYDOWN:
                    if menu:                       # the search box takes every key while open
                        menu.key(e.key)
                    else:
                        self.key(e.key)
            self.poll_tests()
            now = pygame.time.get_ticks()
            if self.ai_auto and not (self.show_deck or self.show_levels) and now - self.ai_last >= AUTO_MS:
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
