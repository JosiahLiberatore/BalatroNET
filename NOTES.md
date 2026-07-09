# BalatroNET — working notes

## next:
scoring_mask — second return value from scoreHand.
Three candidate masks (compute all, select with nested where — five_card before grouped):
- mask_all    = valid                                    (straight / flush / full house / SF)
- mask_groups = (rank_counts[rank] >= 2) & valid          (pair / two pair / trips / quads)
- mask_high   = one_hot(argmax(where(valid, rank, -1))) & valid
Trap test to add: 2-card deuce pair, ranks [0,0,0,0,0], valid [1,1,0,0,0]
→ mask must be [T,T,F,F,F], padding slots excluded (the & valid seatbelt).
Extend TEST_HANDS with an expected-mask column, compare with jnp.array_equal.

## todo
- [x] poker hand detection (scoreHand) — jitted, 13/13 tests, commit 0f7a9ff
- [ ] scoring_mask (current)
- [ ] chips × mult base scoring — BASE_CHIPS[rank] summed over scoring_mask, per-hand-type base table
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
