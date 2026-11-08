# BalatroNET — working notes

## next:
env_step(state, action):
1. derive played cards from the action (verb + 32-lane select mask) — compact selected
   slots down to scoreHand's 5 lanes with the SAME stable-argsort trick as the dead-card sort
2. score * (verb == PLAY) — no branch
3. consequences: chips += score, decrement plays/discards, void used slots (hand_valid False)
4. refill (shared helper — reset and step both END with it, neither IS it)
5. reward = cashout $ on clear, $0 on fail (see "reward = CASHOUT dollars" — supersedes the old
   chips-progress reward) and done (plays out | blind beat)
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
- [x] play/discard actions + round loop (2026-10-17) — blind -> cashout -> shop -> NEXT across 8 antes
      (STAGE_BLIND/SHOP/LOST/WON, ANTE_BASE x 1/1.5/2 targets, cashout = blind $ + hands left +
      interest). Action fork decided: balatroAction(verb, select, target), verbs PLAY/DISCARD/BUY/SELL/
      REROLL/NEXT, phase-legal table LEGAL[stage, verb], illegal -> no-op via lax.switch. Shop rolls
      jokers by rarity (70/25/5), no duplicates, no held jokers. deal() stops at the end of the deck.
      env_step return contract unchanged (reward = chips, done = blind over) so train.py still works.
      gui.py = pygame-ce front end over env_step; tests_run.py = repair/shop/run/fuzz suites.
      STILL OPEN: empty PLAY spends a hand (repair rule undecided); no boss effects, no blind
      select/skip, no tags/packs/vouchers; sort is display-only (lane order = play order).
- [x] training on full runs (2026-10-17) — env_step contract CHANGED: reward = cashout $ on a clear (0 on
      fail), done = the RUN ended (LOST/WON). state.target_scale = curriculum dial on every blind target
      (floor 1 chip), set at reset. agent.py: disjoint obs blocks (836), ONE flat masked categorical
      (PLAY, DISCARD, BUY x4, SELL x32, REROLL, NEXT = 40) + 32 select Bernoullis (count only for
      PLAY/DISCARD), value head; legal_mask mirrors env repair (tests_run legal-mask suite). Masked
      logits = -1e9, not -inf (0 * -inf = NaN in the entropy grad). train.py: A2C + GAE, 1024 auto-reset
      envs x 64 steps, hand-rolled Adam + grad clip, adaptive scale (clear-rate EMA > 0.92 -> x1.10,
      < 0.80 -> /1.05). Runs on the render node GPU (WSL, ~336k env steps/s while training).
      gui.py model panel: the model's odds for every legal choice + card select odds, Step/Auto,
      greedy/sampled (--params, --scale).
      OPEN: PPO (several epochs per rollout) if A2C is too sample-hungry; eval at scale 1.0.
- [x] joker table 4 -> 39 (2026-10-17) — all pure rows: hand-type +Mult/+Chips (Jolly..Crafty), Half,
      suit/rank/face per-card (Lusty..Onyx Agate, Fibonacci), queries (Banner, Mystic Summit, Blue,
      Abstract, Bull, Stencil), Popcorn counter, Golden/Cloud 9 money, The Duo..Tribe XMULT.
      Engine: C_HAND_HAS = CONTAINS (scoring.CONTAINS table; 4oak has Pair + 3oak, not Two Pair),
      C_COUNT_CMP gate (joker_score takes `valid`), ONCE_MAIN XMULT in slot order (closed form, no
      scan), END_OF_ROUND E_ADD_MONEY paid in the cashout (env.joker_money), 5 new query kinds.
      OBS_DIM 836 -> 2096 (32 x 39 loadout one-hot): old params do not load; retrain fresh.
      NOT YET: random rolls (Misprint, Business Card, Gros Michel), two-effect rows (Scholar, Walkie
      Talkie), held cards, retriggers, PER_SCORED XMULT, Baseball Card (power, not product).
- [x] MOVE_JOKER + MOVE_CARD verbs (2026-10-17; REPLACED 2026-10-18 by INSERT moves, below) — target t swaps joker slots / hand lanes t and t+1.
      MOVE_JOKER: value, state, sell value move with the joker; BLIND + SHOP. MOVE_CARD: BLIND only;
      one of the two lanes may be a hole (empty deck), so a card can step across it. Both free.
      Lane order is the play order (XMULT order for jokers; Death / play-order jokers later).
      N_ACTIONS 40 -> 102 (A_MOVE_JOKER + 31, A_MOVE_CARD + 31): old params do not load.
      train logs move_frac (joker + card moves; watch for free-swap step waste).
      GUI: the hand shows lane order; drag a card or a joker; Sort Rank / Suit = MOVE_CARD steps
      (once per press, not sticky: new cards land in the lanes the played cards left).
- [x] joker test suite tests_jokers.py (2026-10-17) — REF = a second, plain-Python model of every
      joker (hand detection, card-then-slot scoring order, counters, cashout $); never reads the
      jokers.py columns. Layers: table (every joker needs a REF row + a spec case), spec (~130
      hand-computed cases), order (all 120 slot orders of a mix), fuzz (6000 random cases, each
      scoring joker must change >= 25 scores), life (scripted env_step runs), soak (64 envs x 400
      steps, every step through the reference transition model; finished runs swap for fresh ones).
      Mutation check: 5 planted bugs (XMULT order, a suit, Blue off-by-one, counter floor,
      Popcorn round update) each fail several suites. ~30 s. A new joker = a REF row + spec cases.
- [x] curriculum OFF by default (2026-10-18) — train.py --curriculum turns it on; --steps N. A run
      dir ends in -real / -curr; the ckpt records the setting. First real-target run on the node
      (100k updates, ~6 h): win 0.9% at 10k -> ~53% at 96k (fixed 512-run eval: 53.5%, mean ante
      5.39). Early wins were the Bull + Bootstraps money engine (legal; no boss effects here).
      That policy spent 60% of its steps on free MOVEs -> the two changes below.
      Node: C:\Users\Josiah\.wslconfig vmIdleTimeout=-1 + instanceIdleTimeout=-1, else WSL stops
      the VM (and tmux) minutes after the last SSH client exits.
- [x] MOVE = INSERT with pointer heads (2026-10-18) — balatroAction.dest added. MOVE_JOKER /
      MOVE_CARD take the item at target and insert it at dest (env.insert_perm; the items between
      shift; Balatro's drag). Legal: target != dest; jokers both < held; cards both < hand_size and
      a card at target (holes shift like cards). Any order needs <= n - 1 moves (was n(n-1)/2
      neighbour swaps). agent: N_ACTIONS 102 -> 42 (one verb each); pointer heads Wm (src + dst
      for jokers and cards, 4 x 32) + Pm (dst bias given src, [kind, src, dst]); act samples src,
      then dst given src; move_masks / dst_mask mask empty slots and src. Old params do not load.
      GUI: one drag = one MOVE step; Sort = inserts (<= hand_size - 1 steps).
      tests: tests_run move tests, move-reach (random orders in <= n - 1), legal-mask checks every
      (src, dst) pair; tests_jokers reference model inserts.
- [x] MOVE streak penalty (2026-10-18) — learner reward only (env money unchanged): the k-th MOVE in
      a row gets -beta * (1 - alpha^(k-1)) $; any other action resets k. Default alpha 0.5,
      beta $1 (0, -0.50, -0.75, -0.88, ... -> -1). train.py --move-alpha / --move-beta; logged
      run_move_penalty ($ per run) + move_streak_max. Tuning plan: mask MOVE on the old eval to
      price a move, then sweep (alpha, beta) on short runs.
- [x] Sequential card picker (2026-10-19) — the 72k insert run (59.6% win) still spent 34% of its
      steps on MOVEs: the 32 independent Bernoulli select lanes selected 7-8 cards, and the env's
      "first 5 selected lanes" repair made MOVE the way to choose cards (masking card MOVEs on that
      policy: win 58.4% -> 1.9%). Now PLAY / DISCARD pick lanes one at a time, up to 5, then STOP
      (STOP needs >= 1 card); each pick sees the verb (query Wq per verb), the lane's card (Wu over
      rank + suit, read back from obs by hand_feats), the cards picked so far (Wc over their summed
      rank + suit) and the count (Wn); scores vo / vs / bstop start at 0 = uniform. act returns
      picks (5,) (STOP padded) besides sel; the loss teacher-forces the picks; pick entropy = the
      per-step entropies along the taken picks. Env unchanged. Old params do not load (Ws -> Wq...).
      train logs play_cards (cards per PLAY); debug_agent prints cards per PLAY / DISCARD; the GUI
      card odds = P(card picked) from 256 sampled sequences, for the chosen verb.
      tests_run picker test: hand_feats, all pick sequences sum to 1, 1..5 distinct valid lanes,
      sampled logp = teacher-forced logp.
      GPU memory: the first node run hit OOM (train_step temp 7.93 GiB on an 8 GiB 2070S);
      jax.checkpoint on the pick_logp_entropy step -> 2.59 GiB, same math. ~2.7 updates/s.
- [x] Joker pool groups A + B + C (2026-10-20) — 39 -> 109 jokers, rules read from the Balatro
      1.0.1o Lua (card.lua calculate_joker, state_events.lua evaluate_play / end_round,
      misc_functions.lua evaluate_poker_hand). The 100k picker run (84.8% win) leaned on Bull +
      Bootstraps and never learned flushes; a wider pool dilutes them, as in the real shop.
      A: counters / scaling / money / budget (Scholar ... Burglar). B: held cards (Baron, Shoot the
      Moon, Raised Fist, Reserved Parking), 1-in-N rolls, retriggers (Hanging Chad, Dusk, Hack,
      Sock and Buskin, Seltzer, Mime). C: Four Fingers, Shortcut, Smeared, Splash, Pareidolia,
      Showman, Oops! All 6s, Blueprint, Brainstorm. Not yet: enhancements / tarots / planets /
      packs / skips + tags / boss effects / joker creation (D-H in the plan).
      ENGINE: scoring.play() is the evaluate_play order — detect (rule mods), BEFORE (counters,
      To Do $), scored cards x repetitions x jokers, held cards x Mime repetitions, main effects in
      slot order + Baseball on Uncommons, AFTER. Mult is affine per step, so each sequence folds
      into one product + weighted sum (fold_mult); no scan. "Contains" is computed from the cards
      (a Four Fingers flush can contain a Pair), not from the CONTAINS table. Copies: copy_sources
      follows Blueprint / Brainstorm chains; copies score but never update counters or pay cashout.
      Update rules gained (cond, param, mode): M_ADD / M_RESET / M_RESET_ELSE_ADD.
      ENV: new state = hand counts (run, round), discards_used, the round targets (Idol card,
      Mail-In rank, Ancient suit, Castle suit), free_rerolls, gros_extinct, base hand size / hands /
      discards (hand size etc. = base + held jokers, recomputed on every loadout change; a smaller
      hand keeps its cards and draws fewer). Money may go to -$20 per Credit Card. Jokers can be
      destroyed (Ice Cream, Seltzer, Popcorn and Turtle Bean at 0, Ramen at X1, Gros Michel 1 in 6,
      Cavendish 1 in 1000). A PLAY / DISCARD now moves state.key on (rolls come from
      fold_in(key, purpose), so explain_play = env_step).
      BEHAVIOUR CHANGES for old jokers: Popcorn is now eaten at 0 (was: stays at 0); Jolly etc. read
      "contains" from the cards. Interest needs $5+ (same result as before).
      TESTS: ref_balatro.py = an independent port of the Lua; tests_jokers uses the env's coin flips
      (PlayOut.rolls / hrolls / rnd). spec 1235, misprint, odds, order 1507, fuzz 10000 cases,
      life, soak (every step through the transition model). Ramen goes at the 100th card (the game,
      by float error, at the 101st).
      AGENT: obs 4999 dims (joker one-hots x 109, To Do / Loyalty state, round targets, hand
      counts, budget). Old params do not load: a new training run is needed.
- [x] 109-joker run on the one-hot obs (2026-10-20, stopped at 86.9k of 100k) — first try OOM at
      update 0 (3.24 GiB alloc; the obs batch alone is 65536 x 4999 f32 = 1.3 GB). train.batch_grads:
      the loss / grad over 4 equal chunks (lax.scan), mean of the chunk grads = the full-batch grad
      (advantage normalized over the full batch first; checked: max rel grad diff 1.5e-6). 3.4 GiB,
      ~2.07 updates/s (136k env steps/s), 100k ~ 13.4 h.
      Curve: win 11% @5k, 23% @10k, 30% @30k, 39% @50k, 42% @70k, ~43% @86k (39-joker run: 84-88%).
      Fixed eval (debug_agent, 512 runs, scale 1.0): win 41.0%, $239.1 / run, ante 6.41, 141 steps /
      run; MOVE 30.2% of actions; hands pair 45%, two pair 21%, high card 20%, flush 0.1%.
      Stopped because of the obs: each (slot, joker) pair is its own one-hot input (32 x 109 held +
      4 x 109 shop = 3924 of 4999 dims), so a joker is learned again per slot and 2.8x the jokers
      means ~2.8x less data per row. Next: a shared joker embedding. Params kept:
      node runs/old_onehot_abc_20261020/, local runs/params_onehot_abc.pkl.
- [x] Shared joker embedding (2026-10-21) — obs 4999 -> 1111. The 36 (slot, joker) one-hot blocks
      (32 held + 4 shop, x 109) are now 36 id fields at the front of obs (id + 1, 0 = empty; shop
      ids only in the shop). W1 does not read them: policy() looks each id up in ONE table
      e = E[id] (109 x 32, learned) + JOKER_FEATS[id] @ Wf (117 fixed row features: trigger,
      condition, effect, source, query, state, prob, rarity, pool, magnitudes, update rule), the same
      for every slot and the shop. h1 = tanh(rest @ W1 + concat_slots(e) @ W1j + b1); W1j has one
      (32, 256) block per slot, so slot order still matters (Blueprint, slot-order mult). The
      (slot, joker) input is now the rank-32 product E[j] @ W1j_s: what the agent learns about a
      joker in one slot (or in the shop) trains the same E[j]. Params 725k (one-hot: ~1.6M).
      tests_run embed test: ids round-trip, same joker = same vector in any slot / the shop, empty
      = 0, slot order changes the policy, E[j] gets a gradient from any slot. Old params do not load.
      GPU memory: 1 grad chunk OOMs (one 3.89 GiB buffer in the loss pass, not the obs);
      GRAD_CHUNKS = 2 -> 5.7 GiB, 130k env steps/s (4 chunks: 124k), 100k ~ 14 h.
      Run (100k, 2026-10-21): win 20% @10k, 37% @30k, 44% @40k (one-hot: ~65k), 47% @60k,
      ~45-50% @85-100k. Fixed eval (512 runs, scale 1.0): win 49.0% (one-hot 41.0%), $277 / run
      ($239), ante 6.78 (6.41), MOVE 21.6% of actions (30.2%). Hands still pair 46%, high card 23%,
      two pair 20%, flush 0.1%. Params: node runs/old_embed_20261021/, local runs/params_embed.pkl.
- [x] Token encoder (2026-10-27). Why: the agent cannot coordinate 5-card hands (flush 0.1%,
      straight 0.4%). The old picker scored each card alone (its rank / suit + a query from the
      trunk + the sum of the cards picked so far): a card never saw the other cards in the hand,
      so "one of 5 hearts" or "completes 4-5-6-7-8" had to come through one 64-wide query.
      AGENT (training method, reward and env unchanged; A2C as before):
      - Tokens: GLOBAL + held jokers + shop offers + hand lanes. Joker token = E[id] + feats @ Wf
        + slot state (live value, sell value, To Do, Loyalty); offer = the same E / Wf + cost,
        affordable; card = rank / suit. Each adds a learned POSITION vector (slot / lane) and a
        TYPE vector: weights are shared across slots, order stays visible (Blueprint, scoring order).
      - 2 pre-LN self-attention layers (64 wide, 4 heads, FF 128); an empty slot / a hole is never
        a key. Tokens only where the env can put something: J_TOK 8 held (5 joker slots, packed),
        C_TOK 16 lanes (8 + the largest hand size bonus); tests_run checks both caps.
      - Trunk = tanh(GLOBAL feats @ W1 + GLOBAL token @ W1t). W1j (one block per slot) is gone.
      - BUY / SELL logits add a score per offer / held joker token. MOVE src / dst pointers score
        the joker / card tokens (dst also from the src token, + the old Pm bias).
      - Picker: lane query / key / value from the card tokens; each lane also attends over the lanes
        picked so far (ctx = the sum of the picked tokens, as before).
      - JOKER_FEATS 117 -> 252: COND_PARAM decoded (suit, rank mask, hand type, count) for the
        condition and for each update rule, + the update rule's condition one-hot. Droll Joker
        now says "flush", Greedy "diamonds", Fibonacci "A 2 3 5 8".
      obs: same fields (1111), regrouped per slot / lane (offsets O_*). Params 393k (725k).
      Speed on the node (2070S): rollout 0.65 s + grad 1.35 s per update (4 or 8 chunks: same) =
      ~33k env steps/s, 100k ~ 56 h (embedding run: 130k / s). All 32 tokens per side: 3.5 s.
      Without jax.checkpoint on the layers the grad pass OOMs. --grad-chunks N (default 8).
      Run: 100k, --grad-chunks 4, node runs/train_tokens.log (2026-10-27 to 10-30; a Blender job
      on the node shared the GPU for ~13 h and the run did ~1.1k updates in that time).
      Curve (mean win over 3k updates): 49% @43k (embedding run: 40%), 51% @50k, 55% @65k,
      57% @80k, 61% @90k, 65% @97-100k; still rising at the end (embedding run: flat from 60k).
      Fixed eval (512 runs, scale 1.0): win 67.0% (embedding 49.0%), $306 / run ($277), ante 7.03
      (6.78); 36 of 512 runs die in ante 1. MOVE 13.2%, DISCARD 7.5%, REROLL 16.0%. Cards per
      PLAY 4.66 (5: 84%). Hands: pair 41%, high card 32%, two pair 17%, three 5.5%, full house
      4.0%, straight 0.4%, flush 0.0%. Better play, same hand strategy: no flushes / straights.
      Params: node runs/params_run.pkl, local runs/params_tokens.pkl (= runs/params_run.pkl).
- [x] Card modifiers, consumables, hand levels, packs, copied cards (2026-11-01). Rules from the
      1.0.1o Lua (card.lua get_chip_* / is_suit / use_consumeable / open, state_events.lua
      evaluate_play / end_round, common_events.lua create_card / poll_edition / get_pack,
      UI_definitions.lua create_card_for_shop, game.lua P_CENTERS / G.GAME.hands). New items.py.
      ENV RULES (all new):
      - Card instances (the pinned migration): card_rank / suit / enh / ed / seal / bonus / live,
        (128,) each; deck and hand hold instance slots; deck[:deck_size] = the live cards. A copy
        takes a free slot (add_card), a destroyed card frees it (remove_cards). Instances 0..51 =
        the standard deck, so card id == instance in the fixtures.
      - Enhancements: Bonus +30, Mult +4, Wild (every suit), Glass X2 + 1 in 4 to break, Steel X1.5
        held, Stone (+50, no rank / suit, always scores), Gold $3 held at round end, Lucky (1 in 5
        +20 Mult, 1 in 15 $20). Editions on cards and jokers: Foil +50, Holo +10, Poly X1.5;
        Negative jokers +1 slot. Seals: Gold $3 when scored, Red retrigger (held too, if the card
        did something), Blue a Planet of the last hand at round end, Purple a Tarot on discard.
        Scoring order per card / rep: chips, Mult/Lucky mult, Gold Seal/Lucky $, Glass X, edition,
        then the jokers; jokers: edition chips/mult, main effect, Baseball, Polychrome.
      - 12 hand types (Flush Five, Flush House, Five of a Kind first) and hand levels
        (s + l x (level - 1)); HAND_NAMES / H_* indices shifted by 3.
      - Consumables: 22 Tarots + 12 Planets, 2 slots. Verbs USE (target slot, select = the cards a
        Tarot acts on, its [lo, hi]) and SELL_CONS ($1). Shop slots roll joker 20 / Tarot 4 /
        Planet 4; secret Planets only after the hand was played; nothing that exists already
        (unless Showman). Buying a Tarot / Planet puts it in a slot (no "buy and use").
      - Packs: 2 pack slots per shop (the first shop's first one a Buffoon Pack), weights / sizes /
        costs of Arcana, Celestial, Standard, Buffoon (normal / jumbo / mega). OPEN -> STAGE_PACK
        (an Arcana pack deals a hand from the shuffled deck); PICK (a Tarot / Planet is used at
        once, a playing card goes into the deck, a joker into a slot); SKIP. Astronomer: Planets
        and Celestial packs free.
      - 26 new jokers (group D): Marble, 8 Ball, Steel, Space, DNA, Constellation, Hiker,
        Superposition, Red Card, Vampire, Hologram, Vagabond, Midas Mask, Hallucination, Fortune
        Teller, Stone, Lucky Cat, Golden Ticket, Certificate, Glass Joker, Driver's License,
        Cartomancer, Astronomer, Burnt Joker, Satellite, Trading Card (135 jokers). Steel / Stone /
        Lucky Cat / Golden Ticket / Glass Joker only in the shop with that enhancement in the deck.
      - End of round: the hand goes back to the deck when the shop opens (gold / blue counted on the
        hand as the last hand left it, before any refill).
      - Seeing Double / Flower Pot: Wild cards fill the empty suits after the plain cards (Lua order).
      - floor(chips x mult) with a few float32 ulps of slack (15 x 12.6 = 189, not 188).
      DIVERGENCES (new): joker slots stop at 8 (5 + 3 Negatives; agent J_TOK); To Do List never picks
      a secret hand; no Spectral cards / The Soul / Black Hole / vouchers / tags / boss blinds; packs
      only in the shop.
      CONSUMABLE SELL VALUE (state cons_sell, per slot): a Tarot / Planet sells for $1, and Gift Card
      adds $1 to every held one at the end of the round, before the held cards act (end_round), so
      a Blue Seal's new Planet misses that round; Blueprint does not copy Gift Card.
      RULE FIXES (found by the fuzz test and the Lua): Seeing Double / Flower Pot / Superposition read
      the cards after Midas Mask and Vampire (a Wild card stripped by Vampire is no longer every suit);
      Campfire also gains when a Tarot / Planet is sold (G.FUNCS.sell_card); Vagabond reads the money
      before the hand (G.GAME.dollars; the hand's own $ is paid later), Bull / Bootstraps still add it;
      The Fool becomes the last card used (set_consumeable_usage), so a 2nd Fool cannot copy again;
      a card a Vampire took stays marked until the hand ends (v.vampired), so a later Vampire skips
      it, even after a Midas Mask makes it Gold again (two Vampires need Showman); the BEFORE step
      goes one joker at a time, so Ride the Bus and DNA see the cards as the Midas Masks / Vampires
      to their left left them (a Vampire left of Ride the Bus can make a Stone King a face card).
      AGENT: N_ACTIONS 42 -> 54 (OPEN x2, USE x2, SELL_CONS x2, PICK x5, SKIP). The picker serves
      PLAY / DISCARD / USE / PICK with a query per verb + the token of the card used, and a count
      range [lo, hi] per action (pick_bounds; 0..0 = STOP at once, logp 0). Tokens 29 -> 42: + shop
      cards of any kind, 2 pack slots, 5 pack cards, 2 consumables; C_TOK 16 -> 20. One item id
      space (jokers, then Tarots / Planets) with fixed item features (ITEM_FEATS: the joker
      features + what a consumable does). Card tokens: rank, suit, enhancement, edition, seal,
      perma chips. Global: hand levels, Planets / Tarots used, the undealt deck by card and by
      enhancement, the pack stage. obs 1111 -> 2301; params 393k -> 534k. Old params do not load.
      TESTS: ref_balatro extended (Card objects, 12 hands, levels, modifiers, joker editions, new
      jokers); tests_jokers check_step models every verb (USE / OPEN / PICK / SKIP included, the
      card instances, the consumables); soak 64 x 600 steps: 1.55M checks pass. New spec cases for
      every new joker and modifier. GUI: card modifiers, consumables (Use / $1), Tarot / Planet
      shop cards, pack slots, the pack screen, hand levels (L), Add Card dev dropdown.
      SCENARIO TESTS (scenarios.py): a start state, a few env_step actions and the game's values
      written by hand with the rule in words, for every joker, Tarot, Planet, pack, enhancement,
      edition and seal, plus the cash out / shop / blind rules (177 scenarios) and the 305 SPEC rows
      as the "Scoring" group. Each step also goes through check_step. A coverage check fails when an
      item has no scenario (a SPEC row counts only when its joker changes the score). tests.py runs
      it. GUI test mode (T): the test panel shows the rule, the steps and the checks; the board
      shows the state after each step (Left / Right), Up / Down change the test, F the next failure,
      Find test searches. A test runs when it is shown (T opens at once); a thread runs the rest
      for the pass count (10 s). GUI fixes: the pack screen layout (5 cards + Skip), a card past the hand
      size (Certificate) shows, new effect labels (adds a card, makes a Tarot, levels up ...).
      tests.py is env only now: each suite prints its time, 4.6 min in all (the soak 79 s is the
      longest). The agent tests (legal mask, picker, embed: eager JAX, the slow part) are out of
      the default run: python tests_run.py agent.
- [x] The rest of Balatro (2026-11-02): the blind select, skips and tags, the 28 boss blinds, the
      32 vouchers, the 18 Spectral cards and 3 Spectral packs, the last 15 jokers (150 in all).
      Rules from the 1.0.1o Lua (blind.lua, tag.lua, card.lua apply_to_run / use_consumeable /
      calculate_joker, button_callbacks.lua select_blind / skip_blind / reroll_boss / cash_out,
      state_events.lua end_round, common_events.lua get_new_boss / create_card / poll_edition).
      ENV RULES (all new):
      - Flow: reset -> STAGE_SELECT (5). At the blind select NEXT plays the blind on deck, SKIP
        (Small / Big Blind) takes its tag, REROLL rerolls the boss (Director's Cut once per ante,
        Retcon any time, $10); SELL, SELL_CONS, USE and MOVE_JOKER work there too. NEXT in the shop
        goes to the blind select (Perkeo acts first). The cash out of a boss starts the next ante:
        a new boss (the least used of the eligible ones), 2 new tags (an Orbital Tag's hand too) and
        the ante's voucher. The win (the ante 8 boss) ends the run: no new ante, no shop.
      - Bosses (BOSS_TABLE: min ante, x mult, $; the 5 showdown bosses only at ante 8): The Hook,
        Ox, House, Wall, Wheel, Arm, Club, Fish, Psychic, Goad, Water, Window, Manacle, Eye, Mouth,
        Plant, Serpent, Pillar, Needle, Head, Tooth, Flint, Mark, Amber Acorn, Verdant Leaf, Violet
        Vessel, Crimson Heart, Cerulean Bell. A debuffed card gives no chips, mult, edition, seal or
        retrigger, no per-card joker effect, and is no face card; a debuffed hand (Psychic, Eye,
        Mouth) scores 0 and only Matador pays. New state: boss, bosses_used, boss_disabled,
        hand_down (face-down lanes), joker_down (Amber Acorn), joker_debuff (Crimson Heart: the
        joker acts as absent), forced_card (Cerulean Bell), card_played_ante (The Pillar),
        boss_hand (The Mouth), ox_hand. Chicot (at the blind start), a sold Luchador and a sold
        joker under Verdant Leaf disable the boss: its effects end (The Wall / Violet Vessel go to
        x2; The Water, The Needle and The Manacle give their hands / discards / hand size back).
      - Tags (24): the immediate ones at the blind select (Top-up, Skip, Garbage, Handy, Economy,
        Orbital); new_blind_choice (Boss Tag rerolls the boss; Standard / Charm / Meteor / Buffoon /
        Ethereal open a free pack, which closes back to the blind select, then the next tag acts);
        the shop ones (Uncommon / Rare: a free joker; Foil / Holographic / Polychrome / Negative: the
        edition and free; Voucher: one more voucher; Coupon: the first cards and packs free; D6:
        rerolls from $0); Investment ($25 at a boss cash out); Juggle (+3 hand size for 1 round);
        Double (copies the next tag). Held tags: state tags (MAX_TAGS 8).
      - Vouchers (32, $10; the second of a pair needs the first): verb REDEEM (13). The shop shows
        this ante's voucher until it is redeemed (Voucher Tags add more). Shop slots, Clearance Sale
        / Liquidation (prices and sell values), Hone / Glow Up, Reroll Surplus / Glut, Crystal Ball,
        Omen Globe, Telescope, Observatory (X1.5 per held Planet of the hand), Grabber / Nacho Tong,
        Wasteful / Recyclomancy, Tarot / Planet Merchant and Tycoon, Seed Money / Money Tree,
        Blank / Antimatter, Magic Trick / Illusion (playing cards in the shop), Hieroglyph /
        Petroglyph (ante -1; ante 0 needs 100 chips), Director's Cut / Retcon, Paint Brush / Palette.
      - Spectral cards (18, $4) and Spectral packs (normal / jumbo / mega): USE and PICK as for
        Tarots (a Spectral pack deals a hand). The Soul / Black Hole: 0.3% per card of an Arcana /
        Spectral / Celestial pack. A Negative consumable (Perkeo's copy) takes no slot.
      - 15 jokers: Ceremonial Dagger, Sixth Sense, Madness, Seance, Riff-raff, Luchador, Diet Cola,
        Mr. Bones, Throwback, Matador, Invisible Joker, Canio, Yorick, Chicot, Perkeo.
      - Reward unchanged: the cash out $ of a cleared blind (the Investment Tag's $ is part of it);
        a skip gives 0. Mr. Bones: the blind pays $0, the hands left and the interest still pay.
      DIVERGENCES (new): The Ox breaks a tie for the most played hand by the better hand (the game
      uses its table order); at most 8 held tags; joker slots still stop at 8; a lost run keeps the
      boss's face-down / debuff state (nothing reads it).
      RULE FIXES: the BEFORE step goes one joker at a time (Ride the Bus, DNA: see above).
      COMPILE: env_step ~36 s (trace 6.5 s + compile 30 s; it was ~120 s while building): the tags
      of the blind select act in one post-step, fori_loop for each repeated part, one switch
      branch for PLAY / DISCARD and one for USE / PICK.
      COMPILE CACHE: env.py sets a persistent compile cache (.jax_cache, git-ignored; every jit
      over 1 s of compile; past 1 GB the oldest entries go). GUI start ~70 s -> ~28 s (tracing
      ~11 s and ~500 small compiles stay; the env_step entry loads in ~5 s); tests.py 7.8 -> 4.3 min.
      JAX puts the cache path string in every key (xla_gpu_per_fusion_autotune_cache_dir), so
      env.py takes the realpath: "python d:\...\gui.py" and "D:\..." share the entries. Before
      this milestone the GUI started in ~23 s with no cache (env_step 22k ops; now 53k).
      TESTS: check_step models the blind select, skips, tags, bosses (debuffs, face down, the hand
      rules), vouchers, Spectral cards and the new jokers; the soak plays them all (redeems, skips,
      boss rerolls). 343 scenarios (the coverage check now also needs every boss, tag and voucher):
      groups Bosses, Tags, Vouchers, Spectrals; rates checked over 8000 shops / 4000 packs.
      GUI: the blind select screen (3 blinds: target, reward, tag, the boss's ability; Play / Skip /
      Reroll Boss), the boss / tags / vouchers in the sidebar, the vouchers in the shop (click to
      redeem), card backs for face-down cards and jokers, a red cross on debuffed cards and the
      disabled joker, Cerulean Bell's forced card, Spectral cards and packs, playing cards in the
      shop, Negative consumables. Test menu (Find test): a sub menu per kind (Jokers, Enhancements,
      Editions, Seals, Tarots, Planets, Spectrals, Vouchers, Packs, Tags, Bosses, Rules; a Scoring
      row goes to its joker / modifier, gui.test_place); the test list is in that order, so
      < Test / Test > go through a sub menu; typed text searches every test.
      AGENT (not changed; it only imports again: 2 feature widths now follow the tables): it has no
      REDEEM action, its legal mask has no blind select (there NEXT is its only choice: it never
      skips or rerolls the boss), and its obs shows no boss, tags, vouchers, face-down or debuffed
      flags. Old params do not load (150 jokers, new item features).
- [x] Rule fixes (2026-11-04), rules from the 1.0.1o Lua:
      - PLAY / DISCARD with no card selected (after the repair: valid lanes, Cerulean Bell's card)
        is a no-op, as the game's buttons stay off (was: an empty PLAY used a hand and scored a
        High Card; an empty DISCARD used a discard). The agent never sampled one (the picker needs
        >= 1 card); the empty hand + empty deck case still ends the round in _settle.
      - Joker slots stop at 16 (was 8): items.MAX_JOKER_SLOTS; agent J_TOK 8 -> 16 (tokens
        42 -> 50, so a training step costs more; old params do not load anyway).
      - To Do List picks from the visible hands (env.todo_pick): the 9 plain ones, and a secret
        one once it was played (evaluate_play sets visible); at the round end never the old one.
        Secret hands themselves were already in (scoring, levels; Planet X / Ceres / Eris only in
        the shop and packs after the hand was played).
      - The Ox on a tie (state_events.lua end_round): the loop never updates _order, so the last
        tied hand in pairs(G.GAME.hands) order wins (was: the better hand). The game's lua51.dll
        is LuaJIT 2.0.5 (no hash seed), so the order of a new run is fixed: Flush House, Full
        House, Flush, Pair, High Card, Straight Flush, Straight, Two Pair, Flush Five, Five of a
        Kind, Three of a Kind, Four of a Kind (the last wins). items.OX_PAIRS_POS;
        ref_balatro.PAIRS_ORDER. A continued run (save + load) rebuilds the table in another order.
      Scenarios: Play / Discard with no card (Rules), 16 slots and the cap (Negative), To Do List
      secret hands with pick rates, two Ox ties.
- [x] Lua review fixes (2026-11-04). A full review of the env against the 1.0.1o Lua (10 review
      agents by area; every finding checked again in the Lua; the list with the Lua lines and the
      decisions is the claude.ai doc "BalatroNET Env Review"). ENV RULES:
      - An empty hand with cards left in the deck is drawn again in a round (game.lua 2544:
        SELECTING_HAND with no card -> DRAW_TO_HAND; The Serpent: 3 once a hand was played; before
        any play or discard it is still the first hand: The House, Certificate). Was: the round
        could not go on: The Hanged Man / Immolate could empty the hand, and since the empty-PLAY
        no-op of the fix above, Play and Discard did nothing there (an endless episode).
      - chips_scored stops at 2^31 - 1 (was: an int32 wrap made a won blind a loss).
      - The run starts with the Red Deck, the game's default deck: 4 discards (was 3, no real deck).
        The fixtures keep their own values (a mid-round blind() has 3 discards per round).
      - create_card('Joker'): the rarity first (70 / 25 / 5 %), then a joker uniform among the
        available ones of it; none left: Joker (was one weighted draw: held, shown and gated jokers
        moved the shares, 70.4 / 24.4 / 5.2 at the run start). Shops, Buffoon packs, Judgement.
      - Overstock / Overstock Plus refill every empty card slot, a bought one too (change_shop_size).
      - MOVE_CARD reaches every card (env.card_lanes: the hand size, or up to the last card past it).
      - A larger hand size in a round draws the new cards at once (CardArea:change_size): a sold Merry
        Andy / Stuntman, an Invisible Joker's copy of a hand-size joker, Crimson Heart letting a
        Juggler go (or disabling a Stuntman), The Manacle disabled by a sale (+1, and its own draw:
        2 cards). A consumable's new joker (Judgement, The Soul) does not draw (state PLAY_TAROT).
      - A tag's free pack moves state.key: a Double Tag's second pack is new when the first is skipped.
      - DNA: the copy joins the end of the hand in BEFORE (G.hand:emplace): a held card in its own hand
        (Steel, Baron, Raised Fist, Blackboard, Mime), Erosion counts it, Hologram gains at once;
        every DNA effect makes one (Blueprint + DNA + Brainstorm: 3; was at most 2).
      - The Pillar's mark lives in card.ability: Midas Mask, Vampire and the enhancement Tarots clear
        it (set_ability builds a new table), Death, Cryptid and DNA copy it (copy_card).
      - A $0 buy / pack / reroll works at any money (can_buy / can_open / can_reroll: cost > 0 to refuse).
      - Rerolls: (D6 Tag ? $0 : the base) + the paid rerolls of this shop (new state reroll_inc,
        reroll_d6); Reroll Surplus / Glut in a D6 shop take $2 off the next reroll only.
      - Joker Stencil gives X1 at 1 or less (more jokers than slots; was X0).
      - Selling a Blueprint / Brainstorm runs the copied joker's sale ability (Luchador, Diet Cola).
      - A Glass card Trading Card destroys shatters: Glass Joker gains.
      - A Negative Emperor / High Priestess / Fool keeps its own slot while it acts.
      - Two Mr. Bones: only the first saves the run and goes.
      - The 15 tags with no minimum ante (P_TAGS min_ante nil) also come at ante 0 (was: the Uncommon
        Tag only). items.TAG_TABLE min ante 0.
      - Sixth Sense and the Glass check read the cards after BEFORE (a Stone 6 a Vampire strips).
      - Burglar takes the discards before a Riff-raff Drunkard adds its +1 (1 discard, was 0).
      - Hand size 0 with no card: the draw ends the run (GAME_OVER) before Certificate and Mr. Bones.
      - A card added to the hand goes to its end (the cards close up, order kept), as G.hand:emplace.
      - Invariants: jokers <= slots + 1 (a charged Negative Invisible Joker sold at full slots),
        consumables <= slots + Negatives + 1 (a Negative Emperor's / Fool's cards).
      AGENT (no training change): the legal mask follows the env ($0 items; a couponed card or pack
      costs $0 there; MOVE_CARD over env.card_lanes). The obs still showed a couponed card's full
      price (fixed in the next entry).
      TESTS: 36 new scenarios (Rules: Deck, Draw, Hand size, Move, Chips, Shop; the jokers, Tarots,
      bosses, tags and vouchers above); the reference model (ref_balatro.ref_play, tests_jokers.
      check_step) follows every change.
- [x] Agent obs prices (2026-11-04). The price of a shop card and a pack in the obs is the price
      BUY / OPEN charge (env.shop_item_cost; pack_cost with pack_free): a couponed card or pack
      shows $0. The can-pay flags of the cards, the packs and the reroll use env.affordable, as the
      legal mask does ($0 always; Credit Card below $0). Was: the full price and money >= price,
      so the agent saw a free card as a paid one it could not pay for. Obs size unchanged (2301).
      Agent tests: the same failures as before (the blind select mask, the picker), none new.
- [x] Stakes (2026-11-04): all 8, White to Gold (game.lua start_run; each stake has the rules of the
      ones below it). state.stake (1 .. 8); reset(key, deck, hand, scale, stake=1).
      ENV RULES (all new):
      - Red: the Small Blind pays $0 (no_blind_reward.Small; blind_dollars).
      - Green: get_blind_amount scaling 2 (300, 900, 2600, 8000, 20000, 36000, 60000, 100000);
        Purple: scaling 3 (300, 1000, 3200, 9000, 25000, 60000, 110000, 200000); ante 0: 100.
        env.ANTE_BASE is a (3, 9) table.
      - Blue: -1 discard (the Red Deck's 4 -> 3).
      - Black / Orange / Gold: create_card gives a shop joker (an Uncommon / Rare Tag's too) or a
        Buffoon pack joker its stickers (env.roll_sticker): one poll > 0.7 Eternal (stake 4+,
        eternal_compat), else 0.4 .. 0.7 Perishable (stake 7+, perishable_compat), never both;
        a second poll > 0.7 Rental (stake 8, any joker). jokers.NOT_ETERNAL (11 jokers) /
        NOT_PERISHABLE (18). Judgement, Wraith, The Soul, Riff-raff, Top-up Tag: no stickers.
      - Eternal: no SELL (can_sell_card); Madness, Ceremonial Dagger, Hex and Ankh skip it (Ankh can
        still pick it to copy). Perishable: 5 rounds; at each round end (after the joker's own
        end-of-round effect) one less; at 0 it is debuffed for good (env.perished, acting() leaves
        it out): no cash-out $ in that round already, its hand size / discards / Chaos reroll go
        (remove_from_deck), a Negative one keeps its slot, Crimson Heart never picks it. Rental: costs
        $1 (sells for $1), $3 at each round end (every held one, a debuffed one and a saving Mr.
        Bones too) before the cash out's interest; the money can go below $0.
      - copy_card (Ankh, Invisible Joker) copies the stickers and the rounds left.
      - New state: stake, joker_sticker / joker_perish (per joker slot), shop_sticker, pack_sticker
        (STK_ETERNAL 1 | STK_PERISH 2 | STK_RENTAL 4). Invariants: the stickers fit the joker and the
        stake, a Perishable joker has 0 .. 5 rounds.
      DIVERGENCES (new): an Ankh / Invisible Joker copy of a perished joker is perished and adds
      nothing (the game's add_to_deck still gives its hand size / discards). Crimson Heart with no
      joker to pick (2 jokers: a perished one and the last pick): the game passes an empty list to
      pseudorandom_element, which indexes a nil entry (a Lua error); the env picks none.
      AGENT (no training change; train.py still plays the White Stake): the legal mask never sells
      an Eternal joker. obs 2301 -> 2464: the stake (one-hot 8), the stickers of each held, shop and
      Buffoon pack joker, a held Perishable joker's rounds left / 5. Old params do not load.
      TESTS: the reference model (ref_balatro: the stake tables, stickers_ok; tests_jokers.
      check_step: the stickers through BUY / PICK / SELL / Ankh / Hex / Madness / Dagger / Crimson
      Heart, rent, the perish count, the cash out, the sticker rules of every new shop and pack);
      the soak plays random stakes with random starting stickers (Eternal SELLs, perishes, rent);
      28 scenarios in a new Stakes group (GUI test menu: Stakes); every stake and sticker needs one.
      GUI: K (or the stake button) starts the next stake; the stickers on held, shop and pack jokers
      (a perished joker crossed out); the stake in the sidebar; "No reward" on the Small Blind.
- [x] The end of the round and the cash-out screen (2026-11-05), rules from the 1.0.1o Lua:
      ENV RULES:
      - The cash-out screen (ROUND_EVAL): new STAGE_CASHOUT after a won round. end_round and the
        cash-out rows (evaluate_round: last_cashout) still run at the winning hand, and the reward
        stays on that step; the $ goes to the money at NEXT (G.FUNCS.cash_out), which then shuffles
        the deck ('cashout') and opens the shop. USE (cards that need no hand), SELL, SELL_CONS and
        MOVE_JOKER work on the screen and come before the shop roll: a joker sold there can come in
        the first shop, The Hermit there doubles the money before the $. The win (the ante 8 boss)
        pays at once and ends the run. Each won round takes one more step (Cash Out).
      - The window after the winning hand: new STAGE_END. In the game a joker can be dragged while
        the hand scores (the joker area has no drag lock; with a mouse interrupt.focus is cleared
        each frame), and end_round reads the joker order after the score animation. Only Mime's
        retrigger of a held card can change with the order (card.lua: every other end-of-round
        effect has `not context.blueprint`), so the env stops in END only when the order can matter
        (env.end_window): the round is won or Mr. Bones saves it, Mime acts, Blueprint or Brainstorm
        acts, and a held card has an end-of-round effect (a Gold card, or a Blue Seal with a free
        slot; not debuffed). MOVE_JOKER and NEXT only (the game blocks a sale or a use while the
        cards are in the play area); NEXT runs the end of the round, and that round's reward comes
        on that NEXT. The boss is still on in END (boss_active, the Manacle's hand size).
      - Pack hands from the deck: an Arcana / Spectral pack deals the next hand-size cards from the
        top (draw_from_deck_to_hand) and puts the hand back at the bottom when it closes
        (draw_from_hand_to_deck takes the leftmost card first; CardArea:emplace puts a deck card at
        index 1, the bottom). The deck is shuffled only at the run start, at Cash Out and at the
        blind start, so a second pack in one shop (or a tag's pack after it) deals the next cards.
        Was: every pack dealt from a new shuffle.
      - The Emperor never makes The Emperor: the used card stays in the pool's "exists" set while it
        acts (used_jokers keeps it until Card:remove after the dissolve); Showman allows it. Was:
        about 1 in 21 of its Tarots. Packs and shops already skipped existing cards.
      - Trading Card: $3 for each acting Trading Card (two with Showman: $6; the card is destroyed
        once). Was: $3 in all.
      - Invisible Joker: its copy of an Invisible Joker starts at 0 rounds (invis_rounds = 0; Ankh's
        copy keeps the count). Was: the copy kept the count and could be sold again at once.
      AGENT (no training change): the legal mask allows MOVE_JOKER and NEXT in END, and USE / SELL /
      SELL_CONS / MOVE_JOKER / NEXT on the cash-out screen; obs 2464 -> 2467 (the END and CASH OUT
      flags, the cash out's $ on its screen). Old params do not load.
      TESTS: the reference model follows every change (END and its NEXT, Cash Out and the shop it
      opens, the pack deal from the top and the close order, the cards The Emperor / The High
      Priestess make, Trading Card, the copy's count). The soak gives 15 % of its runs Mime +
      Blueprint / Brainstorm and 12 Gold cards, and checks that it met the END window and acted on
      the cash-out screen. 29 old scenarios changed (25 got a Cash Out step: the money and the
      shop come after it; 4 expect the cash-out screen after the hand); 13 new ones (Rules: the
      cash-out screen, the window, the pack deal; Tarots: The Emperor; Jokers: Trading Card,
      Invisible Joker). tests_run: the blind end and the full run press Cash Out.
      GUI: the cash-out screen (the $ and the Cash Out button; Use / Sell work there); END keeps the
      hand on the table (drag the jokers, then "Next: the end of the round"); in the shop and on
      the cash-out screen the sidebar shows the target of the blind on deck (it showed the old one).
- [x] Buy and Use, the boss check after Midas Mask / Vampire, The Pillar's new marks (2026-11-05),
      rules from the 1.0.1o Lua:
      ENV RULES:
      - Buy and Use (buy_from_shop, id 'buy_and_use'): a BUY of a Tarot / Planet when the
        consumable slots are full pays the card and uses it at once, if can_use_consumeable allows
        it with the card still in the shop (no hand, no slot of its own: a Planet, The Hermit,
        Temperance, The Wheel of Fortune, Judgement; not a card Tarot, The Emperor, The High
        Priestess, The Fool). The cost is paid first (The Hermit doubles the money after it). With
        a free slot BUY then USE gives the same result, so there is no separate verb. Was: a no-op.
      - Midas Mask / Vampire: set_ability runs the boss's debuff_card again on each card they
        change (a plain / Gold card without The Pillar mark). A Stone card a Vampire strips can
        have the boss's suit (The Club ...) or be a face card (The Plant): it is debuffed, scores
        nothing and triggers the boss (Matador); a later Midas Mask / Vampire skips it, and a DNA
        copy of it is debuffed too. Was: the debuffs from before the hand.
      - The Pillar: a card played in The Pillar's own blind gets no new mark. The game marks it
        (played_this_ante) but does not run debuff_card, so its flag stays as the blind's start set
        it, and the marks go when the boss is beaten. So a Glass face card first played there breaks
        as a face card (Canio), and a DNA copy made there is not debuffed. Was: debuffed at once.
      AGENT (no training change): the BUY mask allows a Tarot / Planet the full slots cannot take
      when the shop can use it; a playing card in the shop (Magic Trick) needs a free card slot,
      not a free consumable slot (a mask bug). The obs and the action layout do not change.
      TESTS: the reference model follows (Buy and Use through its USE effects; ref_play's BEFORE
      checks a changed card again and sets the boss trigger after it; the marks; a DNA copy's
      debuff). The soak checks that it met Buy and Use. 1 old scenario changed (Mercury with the
      slots full is now Buy and Use), 6 new ones (Rules: Buy and Use x 2, refused; Bosses: The
      Pillar + Canio, The Pillar + DNA, The Club + Vampire, The Plant + Vampire + Midas Mask).
      GUI: a shop Tarot / Planet shows "buy and use" (or "no room") when the slots are full; the
      message says "Bought and used".
- [x] The Needle with 0 base hands (2026-11-05), rules from the 1.0.1o Lua:
      - set_blind takes round_resets.hands - 1 hands (env.round_hands: the base + the acting
        Troubadours, not at least 1) from new_round's max(1, ...): 0 base hands give 2 hands, -1
        give 3. Was: always 1.
      - disable gives the take back, so a negative take lowers the hands (the env stops at 0).
        update_hand_played checks the hands only after a hand: the round goes on, one more hand
        can be played, and a DISCARD at 0 hands still draws. Was: the round ended at 0 hands after
        any step. Dusk / Acrobat act only at hands_left == 0 (was <= 0; no change in a normal
        round).
      TESTS: the reference model follows; 2 new scenarios (Bosses: The Needle with 0 base hands,
      The Needle + Luchador).
- [x] Crimson Heart + Astronomer (2026-11-08), rule from the 1.0.1o Lua: set_debuff runs
      remove_from_deck / add_to_deck, and Astronomer's runs set_cost on every card (card.lua
      616-621, 673-678; held Planets too: a Negative Planet sells for $1 with Astronomer, $4
      without). The env now reprices when Crimson Heart disables or lets go a joker (_heart), when
      the boss is disabled (disable_boss) and at the round end (defeat). Was: the price stayed.
      Kept (the game rule): Astronomer sets the cost of held Planets to $0 too.
      TESTS: the reference model follows; 3 new scenarios (Bosses: Crimson Heart + Astronomer).
- [x] The 15 decks (2026-11-08), rules from the 1.0.1o Lua (game.lua P_CENTERS b_*, back.lua
      apply_to_run / trigger_effect, game.lua start_run card_protos, state_events.lua
      evaluate_play / evaluate_round, UI_definitions.lua create_card_for_shop):
      - state.back (items.BACK_NAMES, 0 Red .. 14 Erratic); reset(..., back=0). The deck is not an
        agent choice and not in the obs: each deck is a separate training run.
      - reset: Red +1 discard (BASE_DISCARDS is now 3, the game's base), Blue +1 hand, Yellow +$10,
        Black +1 joker slot -1 hand, Painted +2 hand size -1 joker slot, Nebula -1 consumable slot;
        vouchers redeemed at the start (Magic Crystal Ball, Nebula Telescope, Zodiac Tarot Merchant
        + Planet Merchant + Overstock; the shop never offers them, their second tier can come);
        consumables (Magic 2 The Fool, Ghost Hex); cards (Abandoned: 40, no J / Q / K; Checkered:
        Clubs -> Spades, Diamonds -> Hearts; Erratic: each card a random one of the 52).
      - later rules: Green (cash out $2 per hand left, $1 per discard left, no interest), Ghost
        (Spectral cards in the shop at rate 2: IK_SPECTRAL, Buy and Use too), Anaglyph (a Double
        Tag after each beaten Boss Blind), Plasma (chips = mult = floor((chips + mult) / 2) after
        the jokers and Observatory, not on a debuffed hand; X2 every blind target: env.run_scale).
      - the joker slots come from the deck (env.base_joker_slots) when apply_budget counts them.
      GUI: B (or the deck button under the consumables) = the next deck, a new run; the deck's
      rule in the info panel.
      TESTS: the reference model follows (cash out, targets, scoring, tags, shop kinds, joker
      slots); the soak starts each run with a random deck and checks that Green, Anaglyph, Plasma
      and Ghost acted; 17 new scenarios (Decks).
- [x] Agent obs v2 + the missing actions (2026-11-08). The obs shows only what a player can see
      or know; the agent has no memory, so what a player remembers is in the obs too. Training
      method, reward and env rules unchanged; the deck stays out of the obs (one run per deck).
      - OBS ERRORS FIXED: a shop playing card read as The Fool (item 0; now no item id, its rank /
        suit / enhancement in the shop slot); a Spectral pack showed no cards (and PICK of an
        Aura / Talisman / Deja Vu ... took no card: pick_bounds now reads every consumable pack);
        face-down cards and jokers showed their real features.
      - FACE-DOWN CARDS (The House, The Wheel, The Mark, The Fish): the lane shows only present +
        face down, and the cards it can be (LANE block, 52 + Stone): the game keeps the hand sorted
        by the real cards (CardArea:sort, get_nominal) and Sort Rank / Sort Suit are free, so the
        face-up cards next to it in each sort bound it; and the deck view (view_deck) counts a
        face-down card as still in the deck, so it is one of those cards. The deck block shows the
        deck view (undealt + face down), not the undealt cards only. Checked: a hidden QH and a
        hidden KH that a player cannot tell apart give the same obs.
      - FACE-DOWN JOKERS (Amber Acorn): a slot shows only present + hidden; the hidden jokers' ids
        go in a sorted list (no slot order) that the GLOBAL token sums (what the player saw before
        the blind). A slot shows its joker after it shows a message in a hand or a discard.
      - ENV (no rule change; what a player can know): card_down (128,) = ability.wheel_flipped (set
        when drawn face down, cleared when played / turned up / at the round end; a discarded
        face-down card stays: env._track_down after each step); joker_seen (32,) (moves with the
        jokers like joker_down; set from scoring.PlayOut.acted on a PLAY, from a counter change on
        a discard; cleared with joker_down). 3 new state invariants.
      - NEW OBS: the stage SELECT; the boss (disabled, rerolled, can reroll, The Mouth / The Ox
        hands); the 3 blinds' targets and rewards; the skip tags (+ Orbital hands), the held tags,
        skips; the redeemed vouchers; the shop vouchers (tokens: Ev embedding, cost, affordable);
        the last consumable used (The Fool); per lane debuffed and Cerulean Bell's card; per joker
        slot Crimson Heart's disable; per consumable Negative and sell value; consumable room.
      - ACTIONS: N_ACTIONS 54 -> 70: REDEEM x4 (token-scored); K_CONS 2 -> 8 (USE / SELL_CONS x8);
        the blind select is in the legal mask: NEXT plays, SKIP (Small / Big) takes the tag,
        REROLL rerolls the boss (Director's Cut / Retcon, $10), SELL / USE / SELL_CONS / MOVE_JOKER.
      - SIZES: OBS_DIM 2301 -> 4137 (the LANE block is 20 x 57); tokens 50 -> 60; params 534k ->
        659k. Old params do not load: retrain fresh.
      - train.py logs run_skips and run_redeems.
      TESTS: tests_run agent: run_hidden_obs_test (new: the sort bounds, Stone, the same-obs check,
      hidden joker order, a seen joker, the shop playing card, the Spectral pack, the blind select);
      the legal mask test now plays from the blind select (skips, boss rerolls, redeems). 6 new
      scenarios (Bosses: The Wheel x3, Amber Acorn x3).
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
- joker slots stop at 16: a Negative past 16 slots gives no slot (agent J_TOK = 16)
- The Ox: a tie uses the pairs() order of a new run; after a save + continue the game uses another
- at most 8 held tags (MAX_TAGS)
- no endless mode (every stake and every deck is in)
- an Ankh / Invisible Joker copy of a perished joker is perished and adds nothing (the game's
  add_to_deck still gives its hand size / discards; a game bug) (kept, 2026-11-05)
- Crimson Heart with no joker to pick (a perished joker and the last pick): the env picks none
  (the game: pseudorandom_element on an empty list, a Lua error)
- the round's chips stop at 2^31 - 1, a hand's score at 2,147,483,520 (int32; the game: a double,
  up to ~1e308). No result changes: the run ends at ante 8, whose largest target is 300,000
- found in the Lua review (2026-11-04), kept as they are:
  - the score floor is float32 (+-1 chip only at an exact target); retrigger caps 12 scored /
    6 held (only Showman duplicates or long copy chains pass them)
  - a USE with too many cards selected takes the first ones (the game turns the button off)
- the Orbital Tag's hand when an ante number comes back (Hieroglyph / Petroglyph): the game gives
  the hand it gave that ante and blind before (G.GAME.orbital_choices has the ante number as its
  key); the env picks a new one (kept, 2026-11-05)
- The Pillar: when a joker comes or goes later in its blind (a sale, Seltzer used up, Judgement
  ...), the game checks every card again (set_blind reset), and a DNA copy made in that blind is
  then debuffed (a suit / rank Tarot on the copy does it too); the env does not debuff the copy
  in that round (kept, 2026-11-05)
- a pack hand goes back to the bottom in lane order (the game: the hand's order, sorted by rank
  unless the player moved cards); only a full pass through the deck between two shuffles shows it
  (about 6 Arcana / Spectral Packs between Cash Out and the next blind). Non-issue (2026-11-08):
  no real run gets there
- the END window comes only after a PLAY (a round that a sale ends, a boss disabled by a sale,
  has none; the game waits 0.2 s before end_round reads the jokers) (kept, 2026-11-05); the next
  ante's boss and tags are rolled at the end of the round, not at Cash Out (nothing on the
  cash-out screen changes them)
- at most 4 vouchers in one shop (the ante's voucher and 3 from Voucher Tags; more are dropped)
- the agent sees 8 consumable slots (K_CONS; the env holds up to 16: more Negatives are not
  usable by the agent)
- face-down obs (2026-11-08): a tie of two identical cards in a sort goes by the instance order
  (the game: unique_val); the obs gives the same bounds a player gets from both sort buttons,
  but not the order of two face-down cards next to each other
- Amber Acorn's reveal (2026-11-08): a slot shows its joker after any message; the game shows
  only the message, so two jokers with the same message can look alike there. A discard reveals
  only a counter change (not a joker's $ on a discard: Mail-In Rebate, Faceless Joker); a message
  at the blind select or the round end reveals nothing (the round end turns them up anyway)

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
            ON_SHOP, ON_BLIND_SELECT, ON_SOLD, ON_PACK_OPEN, ON_PACK_SKIP (Red Card +3 mult), ON_BOSS)
- CONDITION: (kind, param) — suit | rank_in | is_face | hand_has | count_cmp | rank_suit(FROM_STATE)
            | none | POSITION (NEW, see gaps)
- EFFECT  : ADD_MULT | ADD_CHIPS | XMULT  (combined adds allowed, e.g. Scholar = chips+mult)
- VALUE   : float — the SEED of joker_value on acquire (CONSTANT: the amount; QUERY: per-unit;
            COUNTER: the start value). Increments live in the update rule (UPD_DELTA), not here.
            (see "VALUE + STATE unify" subsection below)
- STATE   : None | SCALE_COUNTER (Green, Ride the Bus, Seltzer countdown)
            | ROTATE_GLOBAL   (Ancient — uniform 25% over the 4 suits each round, DECK-INDEPENDENT,
              pure RNG; state = suit 0-3)
            | ROTATE_FROM_DECK (The Idol = a card id 0-51; Mail-In Rebate = a rank 0-12 — sampled from
              the values PRESENT in the current deck each round; refresh READS mutable deck state and
              must run AFTER that round's deck mutations). All fit an int.

### VALUE + STATE unify: scoring reads ONE live value `v` (source-agnostic)
The scoring mechanism is already agnostic to WHERE a number comes from — Duo hands it 2.0, Obelisk hands
it 1+0.2*count, the multiply doesn't care. But splitting the number across VALUE (static) vs STATE
(derived) forces the evaluator to BRANCH ("read VALUE, or compute f(STATE)?") — which re-breaks the very
agnosticism scoring already has. Fix: give every joker a mutable live value v (float), seeded from VALUE
at acquire. Scoring ALWAYS reads v and applies it by EFFECT — no branch.
  Duo:     v = 2.0, never changes
  Obelisk: v starts 1.0, +0.2 on its trigger
  Green:   v = running mult, +1/hand, -1/discard
VALUE then DEMOTES from "the amount" to the SEED of v. The increment is NOT in VALUE: one number cannot
hold both (Obelisk = seed 1.0 + step 0.2; Green = seed 0 with two deltas, +1 and -1). Increments live in
the update rule.
BUILT (2026-10-17): update rule = N_UPDATE_RULES=2 (UPD_TRIGGER, UPD_DELTA) pairs per row, padded with
T_NONE=-1; authored as JokerDef(updates=((T_ONCE_MAIN, +1.0), (T_ON_DISCARD, -1.0))). update() sums the
deltas of the rules that fire and floors at 0. condition + reset (Ride the Bus, Obelisk) still to add.
So a memory-joker = TWO rules, not a tuple-with-a-hook:
  scoring rule  (TRIGGER, CONDITION, EFFECT)                    -- READS v
  update rule   (update_trigger, update_condition, delta/reset) -- WRITES v
A static joker (Duo) has the identity update rule (v = seed forever). This kills the per-joker hook — the
thing that made scoring non-agnostic. Cost (honest): the update rule isn't always a scalar — Green is
+1/-1 across two triggers, Ride the Bus / Obelisk RESET on a condition — so it genuinely needs
(trigger, condition, delta, reset), a real second mini-row, not a number. You trade the VALUE/STATE split
for a scoring-rule/update-rule split: more uniform, kills the branch, not free.
The split that SURVIVES: live scoring magnitude (v, float, read uniformly) vs discrete CONTROL memory
(int: ROTATE selector, Seltzer countdown, Dusk gate) — those aren't magnitudes scoring multiplies, they
select/gate. A cleaner boundary than VALUE-vs-STATE, which cut across both.
=> joker_value : (MAX_JOKERS,) float32 holds the live v per slot; joker_state (int) keeps the discrete
   control memory. Scoring reads joker_value; the update pass writes it.

VALUE-SOURCE axis (HOW v is resolved) — one mechanism, v = resolve(joker, state), run in a RESOLVE pass
BEFORE scoring; scoring stays agnostic (reads joker_value). The three sources differ only in what resolve
READS:
  CONSTANT : reads nothing, returns the seed              (Duo 2.0)
  COUNTER  : reads the joker's own stored memory          (Obelisk, Green — evolved by the update rule)
  QUERY    : reads GLOBAL state, resolved FRESH each hand  (Joker Stencil, Bootstraps, Steel/Stone Joker)
QUERY stays pure DATA: a query_kind selector into a fixed menu of state-reductions,
v = base + per_unit * count(query_kind, state). The resolve pass switches over query_kind (a bounded
menu), NOT over joker id.
  query_kind menu: EMPTY_JOKER_SLOTS (Stencil), MONEY_DIV_5 (Bootstraps), DECK_STEEL_COUNT (Steel),
                   DECK_STONE_COUNT (Stone), UNCOMMON_JOKERS (Baseball), ... (grow the MENU, not the code)
  Joker Stencil = (ONCE_MAIN, none, XMULT, SOURCE=QUERY, query_kind=EMPTY_JOKER_SLOTS, per_unit=1.0)
                  ("included" = count slots NOT held by a non-Stencil joker)
MID-ROUND: a QUERY over slots/money/deck CHANGES when you sell/buy mid-blind -> resolve fresh each hand,
cache NEVER (cheap: a reduction over fixed-shape arrays). Same "continuous pure function" rule as the
budget scalars, now on scoring magnitudes.

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
- ROTATE_FROM_DECK refresh needs a "sample uniformly over a boolean present-mask" primitive (52 ids for
  Idol, 13 ranks for Mail-In): masked-argmax over random keys, argmax(where(present, uniform, -inf)).
  Deck-coupled -> run the refresh AFTER that round's deck mutations. Ancient (ROTATE_GLOBAL) skips it: a
  flat randint(0,4).

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
ADD_MONEY is a BIG family and takes the SAME VALUE-SOURCE axis as scoring effects, across every trigger:
  Golden Joker  = (END_OF_ROUND, none, ADD_MONEY, SRC=CONSTANT 4)
  Cloud 9       = (END_OF_ROUND, none, ADD_MONEY, SRC=QUERY DECK_NINE_COUNT per=1)
  Rocket        = (END_OF_ROUND, none, ADD_MONEY, SRC=COUNTER seed 1; +2 on boss defeat)
  Business Card = (PER_SCORED,  IS_FACE, ADD_MONEY, SRC=CONSTANT 2, prob=0.5)
Distinct from SELL_VALUE writes (Egg / Gift Card raise joker_sell_value, NOT money).

### Budget-write EFFECT family (round-setup scalars) + mid-round SELL / USE actions
Third EFFECT family (sibling of scoring-writes and state-writes): budget-writes touch the ROUND budget
scalars the DECK also sets — handcount / discardcount / hand_size.
  Drunkard   = (ON_BLIND_SELECT, none, ADD_DISCARD, 1, None)                        passive while held
  Merry Andy = (ON_BLIND_SELECT, none, ADD_DISCARD +3 & SUB_HANDSIZE 1, ..., None)  compound row
  Burglar    = (ON_BLIND_SELECT, none, ADD_HAND +3 & SET_DISCARD 0,     ..., None)  SET, not ADD
MODEL (corrects "recompute at round start"): budget is a PURE FUNCTION of (deck_base, held jokers),
correct CONTINUOUSLY. handcount/discardcount/hand_size = deck_base + Σ held budget-jokers, read whenever.
That is exactly why selling Merry Andy MID-ROUND restores hand size on the spot — its input to the
function left. A hand_size INCREASE mid-round just runs the shared deal/refill helper (deal up to the new
capacity = "+1 card now"). Recompute-on-change also makes SELL correct for free (no leak on sell).
Burglar breaks pure summation: "+3 hands AND discards=0" is two writes, one a SET; SET-vs-ADD don't
commute (Drunkard +1 then Burglar =0 -> 0; reverse -> 1). So budget rows carry a small delta-vector (or a
tiny handler) + a defined order — the XMULT-threading lesson, now on budget scalars.

SELL is a mid-BLIND action (Balatro sells anytime), NOT shop-only. A sell = compound handler
(money += joker_sell_value; clear slot; recompute budget -> maybe deal); special-handler bucket with
COPY/BOSS, not a (T,C,E,V) row. CONSUMABLES get the SAME treatment: USE and SELL are mid-round actions
too (a Tarot/Planet/Spectral fires mid-blind between hands), each a compound handler writing state
(enhance cards / level hands / money) and consuming the slot — the ACTION exists in the space now, even
though the full USE EFFECT (what each consumable does) stays the deferred subsystem.
So the BLIND-phase action space is: play, discard, sell-joker-k, use-consumable-k, sell-consumable-k —
not just play/discard. Sell/use lanes live in BOTH phases, masked to occupied slots.
ACTION-SPACE FORK (env owner's call): a step that both sells AND plays is ambiguous. Either verb goes
multi-valued (play / discard / sell-k / use-k + slot selector), or each action-type gets its own masked
lanes and a step is one atomic type. Current act() = binary verb + 32 select bits; sell/use is the first
push past binary. Also nudges MAX_STEPS up (sells/uses are steps, bounded by MAX_JOKERS + MAX_CONSUMABLES).

### STATE SHAPE to bake NOW (shape changes = the breaking refactor: they invalidate obs dim,
### network input width, and every checkpoint. Functions grow forever; shape is frozen once.)
### CAPS: MAX_JOKERS=32 (== hand), MAX_CONSUMABLES=16, MAX_CARDS=128, MAX_HAND=32. capacity is SHAPE,
### count is DATA — clamp beyond the cap and count the divergence (see divergences list).
  -- OBSERVED (scoring-relevant; freeze these early, they set obs dim / network width) --
  jokers         : (MAX_JOKERS,) int32   — id per slot; slot ORDER = trigger order (COPY needs it)
  joker_value    : (MAX_JOKERS,) float32 — the LIVE scoring magnitude v per slot (seeded from the row's
                                           VALUE). Scoring reads THIS, source-agnostic; the update pass
                                           writes it. (see "VALUE + STATE unify")
  joker_state    : (MAX_JOKERS,) int32   — discrete CONTROL memory only: ROTATE selector, Seltzer/Dusk
                                           gate counters. NOT scoring magnitudes (those live in
                                           joker_value).
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
Also: budget-JOKERS (Drunkard / Merry Andy / Burglar) raise the plays+discards ceiling, so MAX_STEPS is a
function of the ENABLED JOKER POOL, not the deck alone — when that pool gates on, recompile with a bigger
bound. (joker gate -> scan length; see the budget-write family in the joker-engine section.)

### deck type = reset scalars, not a shape change
Blue = 5 hands, Red = 4 discards, Black = 3 hands + 1 joker slot. All COUNTS into fixed shapes
(handcount / discardcount / joker-slot-count — already state scalars in v0.1). Deck just picks their init
values at reset. Zero reshape. (count is DATA, capacity is SHAPE — the same rule as hand/deck/jokers.)

### obs layout: a union of DISJOINT blocks (NO lane reuse)
  obs = [ SHARED     money, jokers(id/edition/sell), ante, round, phase_flag ]   -- live in both phases
      + [ BLIND-ONLY hand ranks/suits, hand_valid, hands/discards left, chips-vs-target, undealt mask ]  -- 0 in shop
      + [ SHOP-ONLY  shop offerings (4 lanes, locked ones 0), reroll cost, reroll count ]  -- 0 in blind
No lane ever carries two meanings. Reusing card lanes for shop data would make the net score shop items
with card-PLAYING weights (disambiguated only by a phase flag it must learn to respect) — the exact
collision to avoid.
The DISJOINTNESS is the contract, and it is permanent — it is about what a lane MEANS, not about weights.
Lane ORDER is not a contract: reorder/repurpose at will. That rule only ever bought ckpt grafting (reorder
and an old ckpt still LOADS while meaning the wrong things), and we are NOT warm-starting from v0.1's flat-
hands runs. If a cold start on a bigger env ever proves too hard, the answer is a CURRICULUM — ramp the env
gates, one policy, learned in order — not a weight graft. That also keeps the obs free to be re-cut as the
env grows, which is worth more than a warm start we would use once.

### phase isolation is (nearly) free on the current MLP — but it's a SHALLOW-net property
grad of a W1 row is proportional to its input; off-phase lanes are 0 -> W1 is FULLY phase-partitioned
(blind rows move only in blinds, shop rows only in shops). W2 columns for off-phase ACTIONS are
where(-inf)-masked -> gradient stopped -> W2 partitioned too. The ONLY parameter both phases update is the
128 hidden biases b1. So enabling the shop can't drift the blind policy except through 128 numbers —
"prior focused training preserved" is true up to a 128-dim coupling, not "mostly".
CAVEAT (aimed at the v0.2 attention plan): this clean partition is a property of the ONE hidden layer. A
deeper / attention trunk has nonzero activations for BOTH phases past layer 1 -> its internal weights are
fully shared -> the isolation is gone (transfer still helps, but "preserved" -> "co-adapts"). So do not
bank on phase isolation for anything; it is a nice property of THIS net, not a design guarantee.

### economy math (base Balatro — for the blind formula + the shop)
blind reward at cashout: Small $3 / Big $4 / Boss $5.   interest: $1 per $5 held, cap $5 (saturates $25).
reroll: base $5, +$1 per reroll this visit, resets each shop (Chaos the Clown = 1 free reroll/shop).
shop card slots: capacity 4, 2 UNLOCKED at run start. Overstock -> 3, Overstock Plus -> 4 (vouchers).
  capacity is SHAPE, unlocked count is DATA — the same rule as hand/deck/jokers. shop_slots stays (4,)
  for the whole run; locked lanes hold -1, and mask out of obs and out of the buy action. So buying
  Overstock does NOT reshape obs or params — which is the whole reason 4 is baked in now.
  (booster packs deferred.)
joker price by rarity: use the Cost column in jokers_checklist.xlsx.

### reward = CASHOUT dollars. Chips are NOT reward (pinned)
Reward is what the game pays and nothing else:
  cashout = blind reward (Small $3 / Big $4 / Boss $5)
          + $1 per hand remaining
          + interest (rate x $ held/5, capped)  -- included: it's what makes hoard-vs-spend learnable
  interest rate/cap are DERIVED scalars, not constants — budget-family rule (pure function of base +
  held jokers + owned vouchers, recomputed at read, never cached):
    rate = $1 per $5  + $1 per held To the Moon (ADD_INTEREST_RATE, family-C row, stateless)
    cap  = $5 base -> $10 Seed Money -> $20 Money Tree (vouchers)
    interest = rate * min(floor(money/5), cap)  -- cap counts INCREMENTS, so rate scales max payout
    checked: $25+Moon=$10, $50+Seed+Moon=$20, $100+Tree+Moon=$40, $41+Seed+Moon=$16 (floor: 8 incr)
  a VOUCHER = a joker that can't be sold and occupies no slot: permanent for the run, one bool per
  voucher (vouchers_owned), same effect enums, summed alongside held jokers in the derived scalars.
  failed blind = $0. No chip term, no partial credit, no near-miss shaping — the game has none either.
WHY (each load-bearing):
- EFFICIENCY GRADIENT: under chip-sum reward, four pairs totaling 320 == one 316-chip straight. Under
  cashout the straight pays strictly more ($1 per leftover hand) -> a built-in gradient toward the
  coordinated hands v0.1 refused to play, with ZERO hand-type shaping.
- ONE CURRENCY: blind phase and shop phase optimize the same number. Joker value is priced in the
  reward's own units (buy -> clear faster -> more hands left -> more $) — no separate signal needed.
- SCORE-FORMULA-PROOF: reward never reads the score, so decks that change the formula (Plasma:
  ((chips+mult)/2)^2, x2 blinds) can't distort reward scale. (chips_scored stays int32 in state/obs;
  the Plasma overflow question is separate and still open.)
COST (accepted, eyes open): reward is a step function at the blind boundary — clearing at 299/300 pays
like scoring zero. Sparse reward + cold start is handled by the CURRICULUM (ramp blind targets up from
trivial so early policies clear often), never by re-adding a chip term. If a run stalls, the diagnostic
is CLEAR RATE at the current gate — not entropy, not lr.

### the attention swap is a MEASUREMENT, not a plan step
MLP weights do NOT graft onto an attention policy (different topology — (601,128) has no counterpart in
token embeddings + QKV). That swap retrains from scratch. But weights were never the durable asset: the
ENV + obs/action SEMANTICS are architecture-agnostic and carry 100%. The disjoint union re-expresses
in attention as TYPED TOKENS (card / shop-item / joker / phase-ctx): include/omit tokens instead of zero-
padding lanes -> variable set sizes for free (a shop with N offerings is exactly a variable token set).
SEQUENCING: build shop / jokers / multi-ante on the MLP FIRST. Its lasting value is as (a) the env's test
harness and (b) a ceiling-probe — a policy you already trust, so if the reward curve moves the new
subsystem is right, if not it's broken (one unknown at a time). Toggle complexity up until the MLP
demonstrably plateaus; the toggle that breaks it = the spec for the next policy (v0.1 already named one:
32 independent Bernoullis can't COORDINATE a selection -> the attention thesis). Swap because you hit a
named limit with data, not a hunch. Don't decide MLP-vs-attention now — the toggle env is the same build
either way and defers the call until there's data to make it.
