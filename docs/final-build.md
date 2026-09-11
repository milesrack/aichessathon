# Opening and endgame preparation, 10 September 2026

The reference is active v5, source `1f08516`, archive SHA-256
`bde67bd922087e2e8db6b1d2c747539ce829bdd1136067f9c8f30d9a9855fc14`.
The authenticated dashboard reported rating 1575, rank 234 of 452 and
26 wins, six draws and 30 losses through round 105. V5's rounds 91–105
comprised seven wins, one draw and seven losses; the losses were checkmates.

## Runtime changes

`repertoire.py` reads a local Polyglot book only when the incoming FEN's
fullmove number is at most 20. It checks legality through python-chess,
avoids repeating a position, and declines positions near the fifty-move limit.
A miss returns to the existing compiled search and trained evaluator.

Three- and four-piece Syzygy WDL and DTZ tables provide endgame moves.
Move selection preserves the result and prefers progress towards a pawn move,
capture or mate. Actual repetition and draw counters are respected. Missing
tables or uncertain rounded distances near the fifty-move boundary return to
search. The engine retains its game history through prepared moves.

The middlegame search and trained network are unchanged. Opening lookup cannot
run at move 21 or later. Tablebase lookup cannot run with more than four pieces.
The runtime contains no external-engine executable, published neural network,
network request or subprocess call.

## Data and reproduction

The [current competition rules](https://aichessathon.com/docs/rules.md), fetched
on 10 September, expressly permit tables answering positions at move 20 or
below and endgames of at most seven pieces, regardless of what generated them.

The book uses the retained 128 MiB prefix of the
[CC0 Lichess evaluation export](https://database.lichess.org/#evals), with at
least 24 pieces and analysis depth at least 20. These filters retained 726,369
unique position keys from 1,580,923 rows. The completed 12,000-position expansion
produced 757,293 entries (12,116,688 bytes). The exact data and executable hashes
are recorded in `opening-book-manifest.json`. Only moves are exported, without
scores or source engine code. Additional opening analysis uses Stockfish 19
as an offline development tool, one thread, 128 MiB hash, three variations,
and 80 ms per position. Its executable and network are not packaged.

Observed early positions from the retained public competition PGNs seed the
analysis queue. Up to ten plies of each principal variation are considered;
alternative root moves are not stored as recommendations. Every continuation
is bounded by move 20, with a remaining nominal depth of at least 12.
New rounds 98–105 are excluded from this targeted expansion. The initial
positions of rounds 98, 99, 100 and 103 were absent from the targeted seed set.

```sh
.venv/bin/python -m training.build_book \
  --evaluations /private/tmp/chess-research-v3/evaluations-prefix.zst \
  --games artifacts \
  --engine /private/tmp/chess-stockfish/stockfish/stockfish-macos-universal \
  --output weights/openings.bin --positions 12000 --seconds 0.08
.venv/bin/python -m training.fetch_tables
```

The tablebase downloader selects every three- and four-piece file from the
Lichess mirror, totalling 70 files and 4,346,080 bytes. Exact source URLs and
SHA-256 checksums are recorded in `tablebase-manifest.json`. The book's timed
analysis is not bitwise reproducible; the released asset checksum identifies
the actual build.

## Verification

Tests cover the move-20 boundary, castling and underpromotion encoding, missing
coverage, state restoration, and complete conversion of queen, rook,
bishop-and-knight and pawn endings. All original search tests remain enabled.
A separate asset check probed 700 valid positions across all 35 material sets,
checking colour symmetry and agreement between WDL and DTZ signs.

The completed short-clock screen scored seven wins, one draw and no losses
against active v5 across four paired openings at 3 s + 0.1 s.

The local runner terminates processes without flushing buffered stdout, so
prepared-move counts in the initial comparison log are not reliable. PGNs and
W/D/L outcomes are authoritative for these local games.

The full-clock comparison scored three wins, one draw and four losses across
eight games from four additional opening positions at 120 s + 0.5 s.
All games ended normally. This does not establish a competition-clock gain.
The short-clock result must not be presented as a proven rating improvement.

Diagnostics now flush explicitly so subsequent local and platform runs retain
them. This changes logging, not chess decisions.

## Delivery status

No replacement was uploaded. Authentication expired, refresh returned
`refresh_token_already_used`, and the available browser was signed out. A fresh
cookie was requested. At the final status check on 11 September, 14:37 UTC,
the user's 11:00 BST submission deadline had passed. The last authenticated
dashboard showed v5 active; its continued final status has not been verified.
The local archive is an unsubmitted candidate, not an accepted final entry.
Local matches do not establish competition Elo or podium strength.

## Local archive

Final Ruff, strict mypy, all 26 tests, two gate games and both extracted-archive
smoke games passed. The archive contains 74 files and
16,592,457 bytes unpacked. Every archive member was compared byte for
byte with its source. The Polyglot records are uniquely keyed and sorted.

Archive SHA-256: `1682ea58a85e0ba2d9382a88513f009b255b7d942b3533aa29ca08733294df1c`.

The root `agent.zip` is this unsubmitted candidate. The exact prior active
archive is preserved in
`artifacts/2026-09-10/final-push/baseline-agent.zip`.
