# Evaluator experiments

Reference: compiled classical engine `a3c1cd0`. Selected evaluator: `df86be8`.

## Architecture and training

- Sum 32-dimensional piece-square embeddings for each of two perspectives.
- Apply ReLU and a shared output layer; subtract perspectives to enforce antisymmetry.
- Apply tanh to bound the correction to the classical score at ±200 centipawns.
- Train from random weights with AdamW, sigmoid-space squared error, seed 9371 and 30 epochs.

The dataset came from a 128 MiB prefix of the
[CC0 Lichess evaluation export](https://database.lichess.org/#evals): 1,580,923 input rows.
Filtering selected the deepest available analysis at depth 16 or greater, its first principal
variation's centipawn score within ±1500, and valid non-terminal positions outside check.
Positions were discarded when classical quiescence differed from static evaluation by more
than 40 cp or the filtering search timed out.

Canonical feature deduplication left **832,883 training and 94,083 validation positions**.
Opposite perspectives share a partition. The prefix is not a random database sample;
related positions can cross partitions because source-game identifiers are unavailable.

Mean validation error was approximately **103 cp**, versus **131 cp** for classical evaluation.
An eight-region king-conditioned model reached **99 cp** but lost its match comparison.

## Match results

Equal clocks, paired colours and the unmodified harness. Initial screens used 3 s + 0.1 s;
confirmation used four additional openings at 10 s + 0.1 s. Results are candidate wins/draws/losses.

| Candidate | Opponent | W / D / L |
| --- | --- | --- |
| Small quiet-position network | Classical reference | 5 / 1 / 10 |
| Learnt tapered piece-square corrections | Classical reference | 7 / 2 / 7 |
| Late quiet-move reductions | Classical reference | 12 / 0 / 4 |
| Reductions, confirmation | Classical reference | 4 / 1 / 3 |
| Larger cached network | Classical reference | 5 / 2 / 1 |
| Cached network, confirmation | Classical reference | 5 / 1 / 2 |
| Cached network, 10 s + 0.1 s | Reductions alone | 10 / 2 / 4 |
| King-conditioned network | Classical reference | 1 / 3 / 4 |
| Cached network + reductions | Reductions alone | 2 / 1 / 5 |

The larger cached network was retained without reductions. It scored **20/5/7** across
32 games against the two classical versions.

The reduction experiment searched late quiet moves one ply shallower, excluding checks,
check evasions, promotions, captures and killers. Moves improving alpha were searched again
at full depth. Combining it with the neural evaluator weakened the measured result.

## Inference checks

Numba loads the exported NumPy weights and caches both perspective sums. Only features for
changed pieces are updated relative to the last evaluated board.

- Cached sums matched full recomputation over 10,000 board transitions.
- Exported inference agreed with PyTorch within 0.001 cp on 200 positions.
- An earlier evaluator probe measured 0.825 microseconds per evaluation versus 0.325 for
  classical evaluation; its depth-five search also visited more nodes.

## Reproduction

See [training instructions](../training/README.md). The trainer reproduced the released
weights from the retained prepared dataset. SHA-256 identifiers:

- Source prefix: `3989b7568bcbccf23414bbc0004d2ccadd9325e4613da10600e5e51da96d835a`.
- Prepared dataset: `e9e44c471525a70eb6799656f844f2839c6a2a799894f349a0f1012a45d64fa6`.
- Weights: `caecd4c672ea4a14d8896cb6ed47f2e3755a19cbed94b5a4b9e4fdbdc451a3fd`.

## References

- [Tan and Watkinson Medina](https://arxiv.org/html/2412.17948v1): filtering unstable
  training positions in Xiangqi; motivation for the quiet-position filter.
- [NNUE training concepts](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md):
  sparse accumulators and sigmoid-space losses.
