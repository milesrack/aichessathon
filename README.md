# AI Chessathon

A chess engine built for [AI Chessathon](https://aichessathon.com), with Numba-compiled
search and a neural evaluator trained from scratch in PyTorch.

## How it works

`agent.get_move(fen, time_left_ms)` takes a position and remaining clock time, then returns
a legal UCI move such as `e2e4`.

1. **Restore history.** Parse the FEN with `python-chess` and reconstruct the opponent's move
   when possible, preserving repetition history between calls.
2. **Try prepared moves.** Probe three- and four-piece Syzygy endgames, then the Polyglot
   opening book through move 20. Missing coverage and draw-sensitive boundaries fall back
   to search.
3. **Search deeper.** Iterative deepening runs principal variation search with alpha–beta
   bounds. A transposition table reuses previous results; captures, killer moves and quiet-move
   history guide move ordering. Guarded null-move pruning skips unpromising branches.
4. **Resolve tactics.** Quiescence extends leaf positions through captures, promotions and
   check evasions, while retaining stalemate and draw detection.
5. **Evaluate positions.** Tapered material, piece-square, pawn-structure and king-shelter
   terms form the classical score. A neural evaluator adds a correction bounded to ±200
   centipawns.
6. **Return within budget.** The clock limits new iterations and is checked during search.
   An interrupted search restores board state and returns the last completed iteration's
   move, with a legal fallback if no iteration finishes.

### Compiled search

Move generation, reversible board updates, evaluation and search run in Numba over fixed
NumPy buffers. The board uses a 0x88 layout: sixteen-slot rows make off-board detection
cheap. Weights load and search compilation runs at import; game state persists between moves.

### Neural evaluator

Trained from random weights in PyTorch on **832,883 quiet positions**, with **94,083 held
out**, from the Lichess evaluation export. Shared 32-dimensional piece-square embeddings
are summed for each side's perspective, passed through ReLU and subtracted before producing
the bounded correction. Numba evaluates the exported weights during play, updating cached
feature sums only for changed pieces. See [training](training/README.md) for reproduction.

## Setup and usage

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and dependencies are pinned.

```sh
make setup   # Install dependencies
make test    # Run correctness tests
make gate    # Lint, type-check, test and play two verification games
make play    # Play against the greedy baseline
make arena   # Run a 16-game comparison
make zip     # Package agent.zip and smoke-test its contents
```

## Documentation

- [Training](training/README.md): dataset preparation and evaluator training.
- [Evaluator experiments](docs/evaluator-research.md): models and search comparisons.
- [Rated-game review](docs/rated-game-review.md): game analysis and search improvements.
- [Rejected candidates](docs/iteration-2026-09-10.md): king-safety and check-extension tests.
- [Opening book and endgames](docs/final-build.md): asset preparation and validation.

## Acknowledgements

Harness and baselines from [Advit Arora's starter](https://github.com/advitrocks9/aichessathon-starter).

## Licence

Licensed under the [MIT License](LICENSE).
