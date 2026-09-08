# Prompt and Circumstance

An original chess engine with trained neural evaluation for [AI Chessathon](https://aichessathon.com),
maintained by Miles Rack. The submission contains `agent.py` and its weights; the harness comes from
[Advit Arora's starter](https://github.com/advitrocks9/aichessathon-starter).

```sh
git clone https://github.com/milesrack/aichessathon.git
cd aichessathon
make setup
make gate
make play
```

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12. The lockfile includes the
competition's supported packages; this engine uses NumPy, Numba, `python-chess` and the standard
library.

## Engine

The platform imports `agent.py` and calls:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    ...  # Return a legal UCI move, such as "e2e4" or "e7e8q".
```

The engine implements:

- Iterative deepening with principal variation search and alpha–beta bounds.
- Original move generation, reversible board updates, evaluation and recursive search compiled
  by Numba. The board uses sixteen-slot rows with off-board padding.
- Quiescence search with check evasions, captures and all promotion choices.
- A bounded transposition table, capture ordering, killer moves and quiet-move history.
- Tapered material and piece-square evaluation, pawn structure, passed pawns, rook files,
  bishop pairs and king shelter, plus a trained residual evaluator. Cached feature sums
  avoid recomputing the entire network at each evaluated position.
- Opponent-move reconstruction to retain repetition history between calls. Cached scores check
  position and reversible-history hashes, halfmove count and game ply.
- A soft budget for starting another iteration and a deadline checked every 256 search nodes.
  Interrupted searches restore the board and retain the last completed move; very short
  clocks use a legal fallback.

Search runs inside `get_move`. There are no background workers, runtime downloads, external
engines or position-score lookup data. Weights load and compilation runs at import using the
same signatures used during play. `python-chess` parses incoming positions, reconstructs history and checks
the chosen root move; the search itself stays in compiled code. The model was trained from
random weights in PyTorch and exported for Numba inference. See [training](training/README.md).

## Verification

```sh
make test                 # Search, draw handling, special moves and interruption checks
make gate                 # Ruff, strict mypy, tests and two fast games against random
make play                 # One game against greedy at 120 s + 0.5 s
make arena                # Sixteen paired games against greedy at 10 s + 0.1 s
make zip                  # Build agent.zip and play two short games from its contents
```

For the stronger supplied baseline:

```sh
uv run python -m harness.arena --opponent baselines/minimax --games 16 --pgn-dir games
uv run python -m harness.arena --opponent baselines/minimax --games 2 \
  --base-ms 120000 --increment-ms 500 --pgn-dir games-full
```

On 7–8 September 2026, the neural candidate scored **10 wins, three draws and three losses**
against the submitted compiled engine at `a3c1cd0`. Eight games used 3 s + 0.1 s; eight used
10 s + 0.1 s from four additional openings. Every opening was played with both colours.

A separate comparison against an improved classical search scored **10 wins, two draws and
four losses** in 16 games at 10 s + 0.1 s. No game in either comparison failed through an
illegal move, crash or clock overrun. These are local results, not competition Elo or a
measurement on the platform CPU. See [the research record](docs/evaluator-research.md).

Correctness checks include standard perft totals, move generation and state transitions compared
with `python-chess`, special moves, draw-sensitive cache use, mate distance, interrupted search
and neural-cache agreement with full recomputation.

The harness reuses opening positions and seeded baseline tie-breaks. Our wall-clock search can
finish at different depths between runs, so its results are not guaranteed to replay exactly.
Rated opening positions are unpublished; the eight local openings are only a sample.

## Packaging and platform rules

`make zip` packages `agent.py` at the archive root and `weights/evaluator.npz`. The packager also
discovers root-level Python files, imported local packages and `weights/` when present. Keep scratch work
in the ignored `.agents/` directory. Additional assets need an explicit `--include` argument.
The smoke check extracts the archive and runs two short games from it.

Read the current [agent contract](https://aichessathon.com/docs/agent-contract.md) and
[competition rules](https://aichessathon.com/docs/rules.md) before submitting. The platform's
validation log decides acceptance. The local harness reproduces the protocol and clock, but
does not enforce the container's memory limit, read-only filesystem or network isolation.

`harness/` is preserved from the starter. `baselines/` contains random, greedy, two-ply minimax
and a variant with compiled evaluation. `tests/` covers engine invariants. `docs/IDEAS.md`
contains the starter's engine-development guidance.

## Licence

[MIT](LICENSE). Copyright 2026 Advit Arora for the starter and Miles Rack for this fork's
contributions. The original attribution and permission notice are retained.
