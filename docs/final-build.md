# Opening book and endgames

Implemented in `4643ae0`, on top of v5 (`1f08516`). Search and evaluator weights are unchanged.
This candidate was packaged locally but not submitted before the competition closed.

## Runtime

[repertoire.py](../repertoire.py) probes Syzygy endgames, then the Polyglot book, before
falling back to search. History and search state are initialised before either lookup.

- **Opening book:** legal moves through the incoming FEN's move 20; avoid repetitions and
  decline lookup when the halfmove counter reaches 80.
- **Endgames:** three- and four-piece win/draw/loss (WDL) and distance-to-zeroing (DTZ)
  tables. Preserve the result and prefer progress towards a capture, pawn move or mate.
- **Fallback:** missing coverage, uncertain DTZ near the fifty-move boundary, or insufficient
  time returns control to search. Prepared moves are skipped at 200 ms or less.

## Assets

| Asset | Contents | Bytes |
| --- | --- | ---: |
| `weights/openings.bin` | 757,293 Polyglot entries | 12,116,688 |
| `weights/syzygy/` | 70 files covering 35 material sets | 4,346,080 |

The book starts from a 128 MiB prefix of the
[CC0 Lichess evaluation export](https://database.lichess.org/#evals). Requiring at least
24 pieces and analysis depth 20 retained 726,369 position keys from 1,580,923 rows.

Stockfish 19 expanded 12,000 positions offline: one thread, 128 MiB hash, three variations
and 80 ms per position. Early positions from public competition PGNs seeded the queue.
Continuations include up to ten plies, stop at move 20 and require remaining nominal depth
of at least 12. The book stores moves without scores.

The tablebase downloader retrieves three- and four-piece files from the Lichess mirror.
Source URLs and checksums are in the [book manifest](opening-book-manifest.json) and
[tablebase manifest](tablebase-manifest.json).

## Build

After `make setup`, supply the evaluation prefix, PGN directory and local Stockfish executable:

```sh
uv run python -m training.build_book \
  --evaluations /path/to/evaluations-prefix.zst \
  --games /path/to/pgn-directory \
  --engine /path/to/stockfish \
  --output weights/openings.bin --positions 12000 --seconds 0.08
uv run python -m training.fetch_tables
make zip
```

Timed analysis is not bitwise reproducible; manifest checksums identify the build.

## Results

Paired games against v5, with equal clocks and four openings per comparison:

| Clock | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| 3 s + 0.1 s | 7 | 1 | 0 |
| 120 s + 0.5 s | 3 | 1 | 4 |

The longer comparison used additional openings from rounds 98, 99, 100 and 103, absent from
the targeted seed set. The short-clock advantage did not persist at the longer time control.

## Verification

- Tests cover lookup boundaries, move encoding, missing coverage, history preservation and
  queen, rook, bishop-and-knight and pawn endgame conversion.
- Asset checks probed 700 valid positions across all 35 material sets for colour symmetry
  and WDL/DTZ sign agreement.
- The archive contained 74 files and 16,592,457 unpacked bytes, each matched to its source.
