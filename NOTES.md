# BalatroNET — working notes

## next:
scoring_mask — second return value from scoreHand. Dispatch-table form:
- mask_all    = valid (as bool)                           (straight / flush / full house / SF)
- mask_groups = (rank_counts[rank] >= 2) & valid          (pair / two pair / trips / quads)
- mask_high   = one_hot(argmax(where(valid, rank, -1)), 5) & valid
- MASK_CHOICE = [0,1,0,0,0,1,1,1,2]  # per HAND_NAMES slot: 0=all 1=groups 2=high
- scoring_mask = jnp.stack([mask_all, mask_groups, mask_high])[MASK_CHOICE[handtype]]
All three rows of the stack must be the same dtype (bool) or stack promotes silently.
THREE things now share HAND_NAMES order: argmax flag array, HAND_NAMES, MASK_CHOICE.
Trap test to add: 2-card deuce pair, ranks [0,0,0,0,0], valid [1,1,0,0,0]
→ mask must be [T,T,F,F,F], padding slots excluded (the & valid seatbelt).
Extend TEST_HANDS with an expected-mask column, compare with jnp.array_equal.
Later, Splash / Stone cards OR into this mask as a post-step — base layer stays as-is.

## todo
- [x] poker hand detection (scoreHand) — jitted, 13/13 tests, commit 0f7a9ff
- [x] scoring_mask — dispatch table (MASK_CHOICE), 21/21 tests incl. padding + position traps
- [ ] chips × mult base scoring (current) — BASE_CHIPS/BASE_MULT tables in, need BASE_CARD_CHIPS[rank]
      summed over scoring_mask; return handtype, scoring_mask, score; add expected-score test column
- [ ] EnvState dataclass + reset/deal — parallel card arrays (rank/suit/enhancement/edition/seal/location/draw_order,
      MAX_CARDS=64), round scalars as 0-dim arrays; shuffle = permute indices; deal 8 to HAND
- [ ] play/discard actions + round loop (blind progression)
- [ ] later: joker pipeline (trigger points, lax.switch table, slot state, modifiers/aura struct, card-writers via carry)

## design decisions (so far)
- struct-of-arrays everywhere; staple into one tensor only in observe()
- cards = 5 fields: rank, suit, enhancement, edition, seal — closed set, jokers only read/write these
- hand = indices into deck + valid mask; attributes gathered through one index list
- never sort/permute an attribute array alone — permute indices, or collapse to histograms
- branch → compute-all-then-select (where/argmax); search → per-element question (gathers)
- joker thoughts go to jokers-backlog.md, one line, zero design work until core loop scores a hand
