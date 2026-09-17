# King safety and check extensions

Two experiments against v5 (`1f08516`), retaining the same trained evaluator.
Neither change was kept.

## Candidates

- **King danger:** add an evaluation term for coordinated attacks around the king.
- **Check extension:** extend checked positions by one ply in the main search. Quiescence
  already searches check evasions; repetition, deadline and maximum-ply guards remain active.

## Results

Sixteen paired games per candidate at 3 s + 0.1 s.

| Candidate | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| King danger | 4 | 2 | 10 |
| Check extension | 5 | 5 | 6 |

Neither candidate advanced to longer-clock testing.

## Position replays

The target failures were `20...hxg5` in round 96, opening an h-file attack, and `16...Qc8`
in round 97, allowing a damaging queen exchange. Recorded Stockfish 16 depth-16 analysis
suggested `...Ne8` and `...h5`, respectively.

Both v5 and the check-extension candidate still selected `20...hxg5`. The extension changed
`16...Qc8` to `...Nf6`, without demonstrating a correction. Replays used the recorded clocks
but fresh processes, so they lacked the original search cache and history.
