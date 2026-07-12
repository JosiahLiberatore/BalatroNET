# BalatroNET — working notes

## next:
Finish reset (grow (deck, key) into a full EnvState: hand slots + valid mask via refill,
plays/discards/chips counters, key INSIDE state), then env_step(state, action):
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
- [ ] EnvState NamedTuple + finish reset — hand slots + valid, plays/discards/chips scalars,
      key inside state; refill() helper (fill empty slots while under hand_size: cumsum trick)
- [ ] play/discard actions + round loop (blind progression)
- [ ] later: joker pipeline (trigger points, lax.switch table, slot state, modifiers/aura struct, card-writers via carry)

## design decisions (so far)
- struct-of-arrays everywhere; staple into one tensor only in observe()
- cards = 5 fields: rank, suit, enhancement, edition, seal — closed set, jokers only read/write these
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
- face-down cards (boss blinds): not implemented for now. When added: state keeps true rank/suit
  (env must score them), obs gets a face_down flag and zeroes that card's rank/suit lanes -> POMDP.
  The sort-inference trick is faithful FOR FREE: SORT reorders using ground truth, obs shows lane
  order, so a hidden card's position between visible cards leaks the same rank interval the
  animation leaks to a human. Only the pixels are lost, not the information.
