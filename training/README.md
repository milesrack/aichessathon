# Training and experiments

Run commands from the repository root after `make setup`.

## Dataset

The training pipeline reads a 128 MiB prefix of the
[CC0 Lichess evaluation export](https://database.lichess.org/#evals), selects the deepest
analysis with depth ≥16 and a centipawn score within ±1500, and removes invalid positions,
terminal positions and checks. Quiet-position filtering uses the classical evaluator at
`a3c1cd0`, rejecting positions where quiescence differs from static evaluation by more than
40 cp. Features are deduplicated and split by hash, keeping opposite perspectives together.

```sh
mkdir -p .agents/training/baseline
git show a3c1cd0:agent.py > .agents/training/baseline/agent.py
curl -fsSL --range 0-134217727 --max-filesize 134217728 \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  -o .agents/training/evals-prefix.zst
uv run python -m training.prepare .agents/training/evals-prefix.zst \
  .agents/training/quiet.npz --baseline .agents/training/baseline
```

The original run produced 832,883 training and 94,083 validation positions from 1,580,923
rows. The split is position-based, so related positions can cross partitions. Regenerating
it depends on the current export and the filter's 10 ms search deadline.

## Optimisation

[train.py](train.py) trains the residual network from random weights using AdamW
(learning rate 0.003, weight decay 0.01), batch size 512, seed 9371 and 30 epochs.
The loss is squared error between sigmoid-transformed predicted and target scores,
with a scale of 240 cp. The checkpoint with the lowest validation loss is exported to
NumPy weights alongside dataset hashes and per-epoch metrics.

```sh
uv run python -m training.train .agents/training/quiet.npz \
  .agents/training/evaluator.npz
```

Numba inference agreed with PyTorch within 0.001 cp on 200 positions; cached feature sums
matched full recomputation over 10,000 board transitions. The trainer reproduced the
packaged weights from the original prepared dataset. SHA-256 identifiers:

| Artifact | SHA-256 |
| --- | --- |
| Source prefix | `3989b7568bcbccf23414bbc0004d2ccadd9325e4613da10600e5e51da96d835a` |
| Prepared dataset | `e9e44c471525a70eb6799656f844f2839c6a2a799894f349a0f1012a45d64fa6` |
| Evaluator weights | `caecd4c672ea4a14d8896cb6ed47f2e3755a19cbed94b5a4b9e4fdbdc451a3fd` |

## Experiments

Comparisons used the unmodified harness with equal clocks and both colours for each opening.
Screens ran at 3 s + 0.1 s; confirmation used additional openings at 10 s + 0.1 s unless
specified. W/D/L is from the candidate's perspective.

| Candidate | Reference | W/D/L |
| --- | --- | --- |
| Small quiet-position network | Classical (`a3c1cd0`) | 5/1/10 |
| Learnt piece-square corrections | Classical | 7/2/7 |
| Late quiet-move reductions | Classical | 12/0/4 |
| Reductions, confirmation | Classical | 4/1/3 |
| Larger cached network | Classical | 5/2/1 |
| Cached network, confirmation | Classical | 5/1/2 |
| Cached network, 10 s + 0.1 s | Reductions alone | 10/2/4 |
| King-conditioned network | Classical | 1/3/4 |
| Cached network + reductions | Reductions alone | 2/1/5 |
| Tactical leaf generation | Neural (`df86be8`) | 5/5/6 |
| Tactical generation + null-move pruning | Neural | 7/6/3 |
| Same, confirmation | Neural | 4/1/3 |
| Same, 120 s + 0.5 s | Neural | 0/1/1 |
| Additional late-move reduction | Guarded search (`1f08516`) | 7/3/6 |
| King-danger term | Guarded search | 4/2/10 |
| Check extension | Guarded search | 5/5/6 |
| Opening book + tablebases | Guarded search | 7/1/0 |
| Same, 120 s + 0.5 s | Guarded search | 3/1/4 |

The cached network reduced validation MAE from 131 to 103 cp and outperformed both
classical variants. King conditioning lowered MAE to 99 cp but weakened match results.
Tactical generation cut a three-position depth-five benchmark from 0.579 s to 0.327 s;
the subsequent search changes showed mixed results at longer clocks. The implementation
uses the cached network, tactical generation and guarded null-move pruning, with opening
and endgame lookup added in the final local candidate.

## Opening book and tablebases

[build_book.py](build_book.py) imports positions with at least 24 pieces and depth-20
analysis from the evaluation export, then expands early positions from supplied PGNs with
Stockfish. The original build analysed 12,000 positions at 80 ms each using Stockfish 19,
one thread, 128 MiB hash and three variations. Continuations stop at move 20, producing
757,293 Polyglot entries. Timed analysis makes the generated book hardware-dependent.

```sh
uv run python -m training.build_book \
  --evaluations .agents/training/evals-prefix.zst \
  --games /path/to/pgn-directory \
  --engine /path/to/stockfish \
  --output weights/openings.bin --positions 12000 --seconds 0.08
uv run python -m training.fetch_tables
```

[fetch_tables.py](fetch_tables.py) downloads the 70 three- and four-piece Syzygy files
from the Lichess mirror. Sources and checksums are recorded in the
[book manifest](../docs/opening-book-manifest.json) and
[tablebase manifest](../docs/tablebase-manifest.json).
