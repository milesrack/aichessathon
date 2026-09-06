# Prompt and Circumstance

An original classical chess engine for [AI Chessathon](https://aichessathon.com), maintained by
Miles Rack. The submission is `agent.py`; the local harness and baselines come from
[Advit Arora's starter](https://github.com/advitrocks9/aichessathon-starter).

```sh
git clone https://github.com/milesrack/aichessathon.git
cd aichessathon
make setup
make gate
make play
```

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12. The lockfile includes the
competition's supported packages; this engine uses only `python-chess` and the standard library.

## Engine

The platform imports `agent.py` and calls:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    ...  # Return a legal UCI move, such as "e2e4" or "e7e8q".
```

The engine implements:

- Iterative deepening with principal variation search and alpha–beta bounds.
- Quiescence search with check evasions, captures and all promotion choices.
- A bounded transposition table, capture ordering, killer moves and quiet-move history.
- Tapered material and piece-square evaluation, pawn structure, passed pawns, rook files,
  bishop pairs and king shelter.
- Opponent-move reconstruction to retain repetition history between calls. Cached scores
  require matching repetition history, halfmove count and game ply.
- A soft budget for starting another iteration and a hard deadline checked at every search
  node. Interrupted searches restore the board and retain the last completed move; very short
  clocks use a legal fallback.

Search runs inside `get_move`. There are no background workers, runtime downloads, external
engines, neural networks or lookup data. This first implementation searches through
`python-chess` in Python; move generation and evaluation remain the main optimisation targets.

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

The first local run on 6 September 2026 scored **16 wins, no draws and no losses** against the
supplied two-ply minimax baseline at 10 s + 0.1 s, playing both colours from each of eight sample
openings. Every game ended in checkmate. This measures baseline performance on the development
machine; it does not establish a competition Elo or performance on the platform's CPU.

A further pair from the English Opening at 120 s + 0.5 s also finished in two checkmate wins.

The harness reuses opening positions and seeded baseline tie-breaks. Our wall-clock search can
finish at different depths between runs, so its results are not guaranteed to replay exactly.
Rated opening positions are unpublished; the eight local openings are only a sample.

## Packaging and platform rules

`make zip` currently packages only `agent.py` at the archive root. The packager also discovers
root-level Python files, imported local packages and `weights/` when present. Keep scratch work
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
