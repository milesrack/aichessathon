# AI Chessathon

Chess engine with Numba-compiled search and a residual neural evaluator trained in PyTorch,
built for [AI Chessathon](https://aichessathon.com).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
make setup
make play    # Play against the greedy baseline
make arena   # 16 games with paired colours
make gate    # Ruff, mypy, unit tests and two verification games
make zip     # Build agent.zip and test the extracted package
```

## Architecture

[agent.py](agent.py) exposes `get_move(fen: str, time_left_ms: int) -> str`, returning a
UCI move. It uses `python-chess` to parse positions and reconstruct game history, with
move generation, evaluation and recursive search compiled by Numba over a 0x88 board
and preallocated NumPy buffers.

Search combines iterative deepening and principal variation search with alpha–beta bounds,
a transposition table, capture ordering, killer moves and history heuristics. Quiescence
search handles captures, promotions and check evasions. Guarded null-move pruning reduces
work in narrow windows, excluding checks, pawn endings and positions near draw limits.
A deadline interrupts search safely, preserving the best move from the last completed depth.

Evaluation adds a learnt correction to tapered material, piece-square, pawn-structure and
king-safety terms. The network sums shared 32-dimensional piece-square embeddings from
both players' perspectives, applies ReLU and subtracts the outputs. A tanh bounds the
correction to ±200 centipawns. Cached feature sums update as pieces change, keeping inference
inside the compiled search. Model loading and JIT compilation run at import.

[repertoire.py](repertoire.py) provides Polyglot lookup through move 20 and Syzygy WDL/DTZ
lookup for three- and four-piece endings. Repetition and halfmove counters constrain move
selection; uncovered positions return to search. The packaged assets contain 757,293 book
entries and 70 tablebase files.

## Training and evaluation

The evaluator was trained from random weights on **832,883 positions**, with **94,083 held
out**, using quiet positions from the Lichess evaluation database. Validation MAE was
approximately **103 cp**, compared with **131 cp** for the classical evaluator.

Local matches used paired colours and equal clocks. The neural evaluator scored **10/3/3**
against the classical engine across 3 s + 0.1 s and 10 s + 0.1 s games. Adding tactical
move generation and guarded null-move pruning scored **11/7/6** against the neural baseline
at those shorter clocks, followed by **0/1/1** in a 120 s + 0.5 s pair. Opening and endgame
lookup scored **3/1/4** at 120 s + 0.5 s. Results are wins/draws/losses from small local samples.

See [training and experiments](training/README.md) for data preparation, hyperparameters,
comparisons and asset generation.

## Tests

`make test` covers perft, legal move generation, reversible state updates, special moves,
repetition and draw-sensitive caches, interrupted search, neural-cache consistency and
prepared-move selection. Move generation is checked against `python-chess`; endgame tests
include queen, rook, bishop-and-knight and pawn conversions.

## Acknowledgements

Harness and baselines from [Advit Arora's starter](https://github.com/advitrocks9/aichessathon-starter).
Evaluation data and tablebases from [Lichess](https://database.lichess.org/).

## Licence

Licensed under the [MIT License](LICENSE).
