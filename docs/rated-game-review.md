# Search refinement

Implemented in `1f08516`, using the neural engine at `df86be8` (v4) as the reference.

## Changes

- **Tactical leaf generation:** generate captures and promotions directly in quiescence.
  When none exist, check for a legal quiet move before returning a static score to preserve
  stalemate detection. En passant and all promotion choices remain supported.
- **Guarded null-move pruning:** test whether a speculative pass exceeds a narrow search
  bound. Disable it in check, pawn-only endings, near draw limits and after another pass.
  Synthetic lines cannot access real-game score caches or repetition history. Interrupted
  probes restore side-to-move and en passant state.

Across three depth-five benchmark positions, tactical generation reduced total search time
from **0.579 s to 0.327 s**, with unchanged sampled moves and scores.

## Match results

Paired openings and equal clocks against v4 unless noted; candidate wins/draws/losses.

| Candidate | Clock | W / D / L |
| --- | --- | --- |
| Tactical generation | 3 s + 0.1 s | 5 / 5 / 6 |
| Tactical generation + null-move pruning | 3 s + 0.1 s | 7 / 6 / 3 |
| Same, four additional openings | 10 s + 0.1 s | 4 / 1 / 3 |
| Same, round-86 opening | 120 s + 0.5 s | 0 / 1 / 1 |
| Additional late-move reduction, against guarded search | 3 s + 0.1 s | 7 / 3 / 6 |

Tactical generation and guarded null-move pruning were retained. The additional reduction
was excluded. The short-clock improvement did not carry through to the competition-clock pair.

## Tactical weaknesses

V4's rounds 72–90 ended **7/2/10**. All losses were checkmates, with successful initialisation
and 6.3–47.3 seconds remaining. Recorded Stockfish 16 depth-16 analysis identified:

| Position | Played | Suggested | Evaluation loss |
| --- | --- | --- | ---: |
| Round 76, White move 16 | Nxf7 | h4 | 158 cp |
| Round 79, Black move 10 | gxf6 | Bxf6 | 139 cp |
| Round 85, Black move 22 | f5 | Qg6 | 235 cp |
| Round 86, Black move 12 | g5 | Be7 | 143 cp |
| Round 86, Black move 18 | Bxf4 | Ng4 | 140 cp |
| Round 86, Black move 20 | Ne3+ | b5 | 873 cp |

The decisive round-86 error, `20...Ne3+`, persisted in replays at the recorded clock.
Higher throughput did not resolve the tactical weakness.
