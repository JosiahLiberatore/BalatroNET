# BalatroNET — working notes

## next:
env_step(state, action):
1. derive played cards from the action (verb + 32-lane select mask) — compact selected
   slots down to scoreHand's 5 lanes with the SAME stable-argsort trick as the dead-card sort
2. score * (verb == PLAY) — no branch
3. consequences: chips += score, decrement plays/discards, void used slots (hand_valid False)
4. refill (shared helper — reset and step both END with it, neither IS it)
5. reward (chips progress toward blind target + win bonus) and done (plays out | blind beat)
Decide action-REPAIR rules as comments BEFORE coding (agent will emit 6-card selections,
empty selections, illegal discards — clamp/repair at the boundary, never throw).
Sanity target: reset with a known seed, print the hand, score it with scoreHand by hand.

## todo
- [x] poker hand detection (scoreHand) — jitted, 13/13 tests, commit 0f7a9ff
- [x] scoring_mask — dispatch table (MASK_CHOICE), 21/21 tests incl. padding + position traps
- [x] chips × mult base scoring — (BASE_CARD_CHIPS[rank] * scoring_mask).sum() + BASE_CHIPS[handtype],
      times BASE_MULT[handtype]; returns (handtype, scoring_mask, score); 21/21 with expected-score
      column, values hand-computed independently of the code (caught the rank-index-vs-card-value
      off-by-two: rank 3 is the FIVE — always read chips through BASE_CARD_CHIPS, never rank+2 in your head)
- [x] reset first cut — permutation of 128 slot ids, dead slots (id >= deckSize) pushed
      to the back via STABLE argsort on the bool mask (stability preserves shuffle order =
      fair shuffle of live cards); spends subkey, carries key; smoke-tested: clean partition
      at 52, every id exactly once. Returns (deck, key) — not yet a full EnvState
- [x] EnvState NamedTuple (balatroEnvState) + full reset + deal — deal jitted & passing
      (cumsum queue + gather + fill mask); spec test uses a NON-identity deck
      (deck[i] = i+13 mod 52) so src-vs-deck[src] mixups can't pass green (commit e8c8c43)
- [ ] play/discard actions + round loop (blind progression)
- [ ] later: joker pipeline — full provisional spec at bottom of file ("v0.2 joker/scoring engine")
- [ ] later: economy/shop/phase + curriculum — full provisional spec at bottom ("v0.2 economy/shop/phase + curriculum")

## design decisions (so far)
- struct-of-arrays everywhere; staple into one tensor only in observe()
- cards = 5 fields: rank, suit, enhancement, edition, seal — closed set, jokers only read/write these
- CARD ID CONVENTION (pinned): id 0-51 is IDENTITY only — rank = id % 13, suit = id // 13,
  derived on the fly, never stored. Duplicate cards = the same id appearing in multiple lanes.
  Enhancements/editions/seals are INSTANCE properties: later they live in collection arrays
  indexed by a permanent instance slot (card_id[128], enhancement[128], ...), and deck/hand
  switch from holding ids to holding instance indices (one extra gather hop: card_id[hand] % 13).
  Current code is this design with the indirection collapsed (card_id[i] = i), so the
  migration is mechanical when enhancements arrive.
- hand = indices into deck + valid mask; attributes gathered through one index list
- never sort/permute an attribute array alone — permute indices, or collapse to histograms
- branch → compute-all-then-select (where/argmax); search → per-element question (gathers)
- joker thoughts: one line each, zero design work until core loop scores a hand
  (backlog file removed from repo — recover via `git show 4f918d8:jokers-backlog.md`)
- key discipline: `key, subkey = split(key)` — shadowing makes the spent key unreachable;
  spend subkey on the random call, carry key forward (eventually inside EnvState)
- capacity is SHAPE, count is DATA: MAX_HAND=32, MAX_CARDS=128; current hand_size/deck_size are
  scalars in state; env clamps at the cap and counts clamp events (measured divergence, not silent)

## divergences from real Balatro (deliberate)
- hand size capped at 32, deck capped at 128 cards — tail runs beyond these get clamped
- joker slots capped at 32, consumable slots capped at 16 — Negative jokers/consumables can exceed
  these in real Balatro (unbounded consumables are possible); overflow is clamped and counted, same
  SHAPE-is-capacity / count-is-DATA rule as hand/deck
- face-down cards (boss blinds): not implemented for now. When added: state keeps true rank/suit
  (env must score them), obs gets a face_down flag and zeroes that card's rank/suit lanes -> POMDP.
  The sort-inference trick is faithful FOR FREE: SORT reorders using ground truth, obs shows lane
  order, so a hidden card's position between visible cards leaks the same rank interval the
  animation leaks to a human. Only the pixels are lost, not the information.

## v0.2 joker/scoring engine — PROVISIONAL spec
Source: jokers_checklist.xlsx (150-joker taxonomy) + jax-learn lessons 30/31/37/38/39/43/44/45.
PROVISIONAL — expect revisions/additions once the full range of joker effects is nailed down.
The spine is already designed (lesson 30 "jokers-as-data"); this file is the reconciliation +
the JAX-realization work + the small gaps found while pressure-testing it.

### The spine: jokers are DATA, not code (lesson 30)
A joker = a row over orthogonal axes: (TRIGGER, CONDITION, EFFECT, VALUE, STATE).
ONE generic evaluator reads the axes. Adding a joker = adding a row, NOT a code branch.
~30 confirmed jokers in the sheet are pure rows already (Greedy, Fibonacci, Jolly, Scary Face...).
- TRIGGER : ONCE_MAIN | PER_SCORED | PER_HELD  (+ out-of-scoring: END_OF_ROUND, ON_DISCARD,
            ON_SHOP, ON_BLIND_SELECT, ON_SOLD, ON_PACK_OPEN, ON_BOSS)
- CONDITION: (kind, param) — suit | rank_in | is_face | hand_has | count_cmp | rank_suit(FROM_STATE)
            | none | POSITION (NEW, see gaps)
- EFFECT  : ADD_MULT | ADD_CHIPS | XMULT  (combined adds allowed, e.g. Scholar = chips+mult)
- VALUE   : float
- STATE   : None | SCALE_COUNTER (Green, Ride the Bus, Seltzer countdown) | ROTATE (The Idol,
            Mail-In Rebate, Ancient — env flips a rank/suit memory each round). All fit an int.

### The engine: data says WHAT each cell contributes; the ORDERED SCAN says HOW they combine
Key reconciliation: the evaluator's per-joker (add_mult, add_chips, xmult) return is fine for the
flat-add family (adds commute), but the moment XMULT enters PER_SCORED/PER_HELD the (card x joker)
cells must be threaded IN TRIGGER ORDER by a scan — you canNOT collect per-joker then combine.
(4+4)*3 != (4*3)+4 ; lesson 44/45: same non-commuting math three times over (jokers, played, held).
Pipeline stages (each a fixed-shape scan, jit/vmap-safe):
  stage 0  RULE-MODS (passive, patch the context BEFORE scoring): Splash (scored=played),
           Four Fingers, Shortcut, Smeared, Pareidolia, Oops All 6s, Midas (card-write to gold),
           Showman. NOT score-adders — they change the rules/context the later stages read.
  stage 4  PER_SCORED — walk scored cards in ARRANGEMENT order (agent controls it, MOVE_CARD lever).
           per card: retrigger count R = 1 + Σ_jokers retrigger_amount(joker, card) (capped);
           repeat R times: apply card enh/edition/seal + Σ per_scored jokers, threading (chips,mult);
           collect card-writes (Midas gold, Glass shatter, money).
  stage 5  PER_HELD — walk held cards in arrangement order, SAME threaded carry (Baron xmult,
           Raised Fist +2*rank, Shoot the Moon +13/Q, Steel x1.5). Order-dependent — lesson 45.
  stage 6  ONCE_MAIN — independent jokers in SLOT order (Joker, Duo, Green's mult); COPY resolves
           here (Blueprint copies right neighbor, Brainstorm copies leftmost) — slot position matters.
  final    score = floor(chips * mult). mult carry is FLOAT (xmult/poly/steel fractional) — real
           change from the current int chips*mult in scoreHand (must expose base_chips/base_mult
           separately so jokers inject between the add and the multiply).
  update   out-of-scoring triggers mutate joker_state (Green/Ride the Bus counters, Seltzer countdown).

### Retrigger is its own axis; its SCOPE reuses the condition menu
count = 1 + Σ_jokers retrigger_amount(joker, card). Each retrigger joker carries a scope condition:
  Hanging Chad = (position, 0) +2   Sock&Buskin = (is_face) +1   Hack = (rank_in {2,3,4,5}) +1
  Seltzer = (none) +1 [gated counter>0]   Dusk = (none) +1 [gated final_hand]   Mime = PER_HELD (none) +1
Photograph X4 under Seltzer / X8 under Chad falls straight out of the scan — the retrigger just
raises the trigger count on the first-face card; Photograph's condition re-passes each trigger.

### TWO condition slots (both pure-data)
- GATE (joker on/off this hand, checked once vs game/own state): Dusk=final_hand, Seltzer=counter>0,
  Half Joker=cards_played<=3. (hand_has / count_cmp are already this shape.)
- PER-CARD condition (fires on WHICH card): Photograph=first_face, Greedy=suit Diamonds.

### GAPS found while pressure-testing (small, stay pure-data)
- POSITION condition kind — ("position", 0) / first_face. Needed by Photograph (first scored face,
  X2 once) and Hanging Chad (retrigger first scored card). Without it a naive is_face fires Photograph
  on EVERY face (X2 K + X2 Q = X4/X16 under retrigger) instead of once. Both are marked "No" in the sheet.
- XMULT must be first-class in EFFECT and threaded by the scan (lesson-30 evaluator only sums per-joker).
- Categories beyond the (T,C,E,V) tuple, each its own small handler (NOT the generic evaluator):
  PASSIVE rule-mods (stage 0), COPY (Blueprint/Brainstorm), BOSS hooks (Matador ADD_MONEY / Mr.Bones
  PREVENT_DEATH / Luchador+Chicot DISABLE_BOSS), econ/shop mods (deferred with the shop layer).

### State-write EFFECT family (spawns / transforms / destroys / money)
The EFFECT axis has two kinds. Scoring effects write the (chips,mult) carry (ADD_MULT/ADD_CHIPS/XMULT).
STATE-WRITE effects write game state instead — they thread a side-effect channel alongside the carry
and apply to state AFTER scoring resolves. Same category as Midas (transform to gold) / Glass (destroy).
Still pure (T,C,E,V) rows; the effect enum just names a state-write, not a carry-write:
  TRANSFORM_CARD (Midas)   DESTROY_CARD (Glass)   ADD_MONEY (econ)
  SPAWN_CARD (DNA)         SPAWN_CONSUMABLE (8 Ball / Séance / Vagabond)
  DNA      = (ONCE_MAIN, gate first_hand & played==1, SPAWN_CARD copy_played,   1,    None)
  8 Ball   = (PER_SCORED, rank_in{A},                 SPAWN_CONSUMABLE Tarot,    0.25, None)  # prob, rng from ctx
  Séance   = (ONCE_MAIN,  hand_has Straight Flush,     SPAWN_CONSUMABLE Spectral, 1,    None)
  Vagabond = (ONCE_MAIN,  gate money<=4,               SPAWN_CONSUMABLE Tarot,    1,    None)
Gates for spawns: "has room" (free consumable/hand slot) + probability (rng in ctx; Oops!All6s is a
stage-0 rule-mod that DOUBLES the prob param). DNA fits TODAY — spawning a real card is an instance
write into the MAX_CARDS=128 headroom (copy played instance's attrs to next free slot, bump deck_size,
place in hand); no reshape. SPAWNING a consumable is cheap (write an id into a slot); USING a
consumable (what a Tarot/Spectral/Planet does) is a whole separate deferred subsystem, shop-tier.

### STATE SHAPE to bake NOW (shape changes = the breaking refactor: they invalidate obs dim,
### network input width, and every checkpoint. Functions grow forever; shape is frozen once.)
### CAPS: MAX_JOKERS=32 (== hand), MAX_CONSUMABLES=16, MAX_CARDS=128, MAX_HAND=32. capacity is SHAPE,
### count is DATA — clamp beyond the cap and count the divergence (see divergences list).
  -- OBSERVED (scoring-relevant; freeze these early, they set obs dim / network width) --
  jokers         : (MAX_JOKERS,) int32   — id per slot; slot ORDER = trigger order (COPY needs it)
  joker_state    : (MAX_JOKERS,) int32   — SCALE_COUNTER / ROTATE / gate counters (store int, hook
                                           computes any float, e.g. Obelisk xmult = 1 + 0.2*count)
  joker_edition  : (MAX_JOKERS,) int32   — foil/holo/poly/negative (cheap, include even if unused)
  joker_disabled : (MAX_JOKERS,) bool    — SHARED debuff mask the engine reads at stage 0. Sources:
                                           perishable-expiry, boss-blind debuff, ... (all dormant in
                                           v0.2: no bosses, single round -> all False; bake it anyway
                                           so boss/perishable land with no reshape). obs reflects it.
  card_enh       : (MAX_CARDS,) int32    — enhancement per instance slot (bonus/mult/glass/steel/...)
  card_edition   : (MAX_CARDS,) int32    — foil/holo/poly per instance
  card_seal      : (MAX_CARDS,) int32    — red/blue/gold/purple. Keyed by PERMANENT instance slot
                                           (MAX_CARDS=128), matching the NOTES card-instance migration:
                                           deck/hand hold instance indices, attrs gathered through them,
                                           so attributes persist across deal/discard for free. Current
                                           code collapses the indirection (instance i = id i); this makes
                                           the shape already correct so the migration stays mechanical.
  arrangement    : action must set play-order AND held-order (ordered selection, not a set mask)
  -- UNOBSERVED for scoring (economy / dormant until the shop+sell layer; can be added later too, but
     baked now for tidiness. NOT encoded in observe() until the sell/shop action exists — that layer
     reshapes the network anyway) --
  money            : scalar               — Gold cards, Golden Joker, Rocket, interest; shop layer
  consumables      : (MAX_CONSUMABLES,) int32 — id per slot, empty sentinel; spawn target. Use = deferred.
  joker_sell_value : (MAX_JOKERS,) int32  — MUTABLE, not derivable from cost. Egg (+3/round) and Gift
                                           Card (+1/round to all) are END_OF_ROUND writes here.
  joker_sticker    : (MAX_JOKERS,) int32  — enum: none/eternal/perishable/rental. Eternal = pure
                                           sell-action constraint (never scoring). Rental = -$1/round.
  joker_perish     : (MAX_JOKERS,) int32  — perishable countdown (init 5). Own array, NOT folded into
                                           joker_state (a perishable Green Joker needs both at once).
                                           END_OF_ROUND masked tick, routes debuff into joker_disabled:
                                             is_perish = (sticker==PERISHABLE) & ~disabled
                                             perish   -= is_perish            # non-perishables never tick
                                             disabled |= is_perish & (perish<=0)
                                           Dormant in single-round v0.2 (tick never fires). Countdown
                                           enters obs only in MULTI-round episodes (planning signal).

INVARIANT that protects the project: the stage pipeline + state shape are FROZEN; every joker is
pure data (a row) or a switch branch in one of the special handlers. Adding any future joker never
touches the pipeline or the shape. Get the shape + the axis menus right now; grow the rows forever.
REFINED freeze rule: only OBSERVED state must be frozen early (obs dim -> network width -> param-ckpt
compatibility). UNOBSERVED env-state arrays can grow whenever — jit just retraces, nothing breaks, no
retrain. So economy state (money/consumables/sell/sticker/perish) is baked now for tidiness but carries
no urgency; it enters observe() only when the shop+sell action layer lands (which reshapes anyway).

## v0.2 economy / shop / phase + curriculum — PROVISIONAL
Decision: the env is the FULL 8-ante, all-mechanics machine at full shape; curriculum GATES disable
pieces (jokers, shop, antes, decks) to recover a v0.1-simple problem, then flip on as the agent earns
capability. Blind target = pure formula (ante, blind_index, stake) -> chips (fixed shape, scalars in).
Shape frozen once; behavior gated; never a reshape between curriculum stages.

### money + joker slots enter observe() (the "breaking refactor", triggered on purpose)
money and joker slots (jokers / joker_edition / joker_sell_value) leave the UNOBSERVED bucket and enter
obs. obs dim grows, params[0] reshapes, v0.1 ckpt retires. Was always going to happen when the shop
landed — this is the moment. The state SHAPES don't change (arrays were baked at final shape for exactly
this); it's "add them to the concat", not "reshape state".

### PHASE: env becomes blind <-> shop, alternating (the real architectural change, bigger than "add money")
new phase flag in state. Heterogeneous action space: play/discard in a blind, buy/reroll/skip in a shop.
KEEP the flat masked action vector (act() already does exactly this) — add shop lanes to the SAME vector,
mask by phase. One head, one policy, phase-masked. No multi-head machinery.

### static gates (chosen over dynamic)
gates = Python bools closed over at jit time (ENABLE_JOKERS / ENABLE_SHOP / ... , MAX_ANTE). The traced
graph OMITS disabled work -> zero runtime cost. Flipping a gate = recompile / new run; curriculum = a
sequence of runs. (Blind target stays DYNAMIC — it's a scalar formula fed at reset. Only whole SUBSYSTEMS
gate statically.)
RULE that makes it safe: gates control env DYNAMICS + scan length, NEVER obs LAYOUT. obs is always full-
shape; a gated-OFF subsystem still occupies its slots reading zeros. If a gate ever removed features from
obs, weights couldn't carry across stages and the curriculum loses its point.

### episode length is DATA; scan length is SHAPE (same rule as MAX_HAND, applied to time)
scan runs a fixed MAX_STEPS (shape); a done flag flips mid-scan, tail masked to no-ops (v0.1 ALREADY does
this: chips*~done, log_prob*~done). v0.1's flat length=7 wasn't "the episode is 7 steps" — it was 4 hands
+ 3 discards, the default blind's max action budget, sized so no masking showed. MAX_STEPS is a per-stage
STATIC constant: stage 0 (single blind) ~9; full 8-ante run ~several hundred (24 blinds x ~14 action-steps
+ 24 shops x ~15 ~= 700 worst case; done-mask means a typical run uses a fraction).
KEY: MAX_STEPS does NOT touch param shape (net sees one obs -> one action per step; scan length is a
rollout concern only) -> it's the FREEST dial: lengthen it every stage, weights carry over untouched.
Latent v0.1 gap: a Blue deck (5+3=8) > length 7 would silently TRUNCATE -> MAX_STEPS must cover the max
enabled hands+discards the moment decks turn on.

### deck type = reset scalars, not a shape change
Blue = 5 hands, Red = 4 discards, Black = 3 hands + 1 joker slot. All COUNTS into fixed shapes
(handcount / discardcount / joker-slot-count — already state scalars in v0.1). Deck just picks their init
values at reset. Zero reshape. (count is DATA, capacity is SHAPE — the same rule as hand/deck/jokers.)

### obs layout: append-only union of DISJOINT blocks (NO lane reuse)
  obs = [ SHARED     money, jokers(id/edition/sell), ante, round, phase_flag ]   -- live in both phases
      + [ BLIND-ONLY hand ranks/suits, hand_valid, hands/discards left, chips-vs-target, undealt mask ]  -- 0 in shop
      + [ SHOP-ONLY  shop offerings (2 slots), reroll cost, reroll count ]        -- 0 in blind
No lane ever carries two meanings. Reusing card lanes for shop data would make the net score shop items
with card-PLAYING weights (disambiguated only by a phase flag it must learn to respect) — the exact
collision to avoid.
CONTRACT: append at the TAIL only; never reorder or repurpose a lane. Break it and every prior ckpt still
LOADS but means the wrong things (silent poison). To reuse v0.1's (601,128): put the 601 blind lanes FIRST
in the same order, then append the shop block; graft = copy rows[0:601] from v0.1, small-init rows[601:].

### phase isolation is (nearly) free on the current MLP — but it's a SHALLOW-net property
grad of a W1 row is proportional to its input; off-phase lanes are 0 -> W1 is FULLY phase-partitioned
(blind rows move only in blinds, shop rows only in shops). W2 columns for off-phase ACTIONS are
where(-inf)-masked -> gradient stopped -> W2 partitioned too. The ONLY parameter both phases update is the
128 hidden biases b1. So enabling the shop can't drift the blind policy except through 128 numbers —
"prior focused training preserved" is true up to a 128-dim coupling, not "mostly".
CAVEAT (aimed at the v0.2 attention plan): this clean partition is a property of the ONE hidden layer. A
deeper / attention trunk has nonzero activations for BOTH phases past layer 1 -> its internal weights are
fully shared -> the isolation is gone (transfer still helps, but "preserved" -> "co-adapts"). Bank on the
append contract (survives architecture swaps), NOT on the isolation.

### economy math (base Balatro — for the blind formula + the shop)
blind reward at cashout: Small $3 / Big $4 / Boss $5.   interest: $1 per $5 held, cap $5 (saturates $25).
reroll: base $5, +$1 per reroll this visit, resets each shop (Chaos the Clown = 1 free reroll/shop).
shop card slots: 2 (basic shop = 2 buy slots + reroll; vouchers/booster packs deferred).
joker price by rarity: use the Cost column in jokers_checklist.xlsx.

### the attention swap is a MEASUREMENT, not a plan step
MLP weights do NOT graft onto an attention policy (different topology — (601,128) has no counterpart in
token embeddings + QKV). That swap retrains from scratch. But weights were never the durable asset: the
ENV + obs/action SEMANTICS are architecture-agnostic and carry 100%. The append-only union re-expresses
in attention as TYPED TOKENS (card / shop-item / joker / phase-ctx): include/omit tokens instead of zero-
padding lanes -> variable set sizes for free (a shop with N offerings is exactly a variable token set).
SEQUENCING: build shop / jokers / multi-ante on the MLP FIRST. Its lasting value is as (a) the env's test
harness and (b) a ceiling-probe — a policy you already trust, so if the reward curve moves the new
subsystem is right, if not it's broken (one unknown at a time). Toggle complexity up until the MLP
demonstrably plateaus; the toggle that breaks it = the spec for the next policy (v0.1 already named one:
32 independent Bernoullis can't COORDINATE a selection -> the attention thesis). Swap because you hit a
named limit with data, not a hunch. Don't decide MLP-vs-attention now — the toggle env is the same build
either way and defers the call until there's data to make it.
