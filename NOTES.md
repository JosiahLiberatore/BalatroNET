# BalatroNET — working notes

## next:
EnvState + reset/deal. Struct-of-arrays deck (rank/suit/... parallel arrays or int codes
0..51 with rank = i % 13, suit = i // 13), key threaded through state, shuffle = permute
indices (never an attribute array alone), deal 8 to HAND via gather.
Sanity target: reset with a known seed, print the hand, score it with scoreHand by hand.

## todo
- [x] poker hand detection (scoreHand) — jitted, 13/13 tests, commit 0f7a9ff
- [x] scoring_mask — dispatch table (MASK_CHOICE), 21/21 tests incl. padding + position traps
- [x] chips × mult base scoring — (BASE_CARD_CHIPS[rank] * scoring_mask).sum() + BASE_CHIPS[handtype],
      times BASE_MULT[handtype]; returns (handtype, scoring_mask, score); 21/21 with expected-score
      column, values hand-computed independently of the code (caught the rank-index-vs-card-value
      off-by-two: rank 3 is the FIVE — always read chips through BASE_CARD_CHIPS, never rank+2 in your head)
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
