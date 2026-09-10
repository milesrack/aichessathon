# Rated games and search refinement, 9 September 2026

The v4 neural submission is identified by SHA-256
`37a543f81e7962ba1da41c2078907c122ca203ec01afd7af4ae6189de1661ffc`.
The upload shown as 8 September 18:59 on the dashboard is 17:59 UTC.

## Competition evidence

The authenticated dashboard provided nineteen completed games since that upload,
rounds 72–90: seven wins, two draws and ten losses. All ten losses were
checkmates. Initialisation succeeded; the losing games ended with 6.3–47.3 seconds
remaining. The latest snapshot, at 00:09 UTC on 10 September, showed rating 1496 and rank 270.

Repeated weaknesses are exposed kings, premature pawn advances and attacks whose
failure lies beyond the completed search. In round 86, 12...g5 allows 13.Nxg5 and
opens lines against the castled king. The log stays near equality through our
sixteenth move, then drops from +75 to -403 and -1341 cp. Mate leaves 47.3 seconds
unused. This motivates deeper tactical verification rather than more aggressive
clock spending.

The round-81 podium snapshot was Patzer 2.3, Poincare conjecture and PSL God Matt
Bomer. Their team pages supplied 189 distinct public PGNs. Patzer's round-81 win
shows a long conversion through a rook ending, control of an enemy passed pawn,
and eventual underpromotion to mate. Poincare's round-77 win opens the centre
against a stranded king before a forcing attack. PSL's round-71 win gains the
exchange, advances an outside passer and converts the remaining rook ending.
These are observations of their moves, not deductions about private source code.

The full download manifest, individual PGNs, private logs and detailed game notes
are retained in `artifacts/2026-09-09/`. Credentials are excluded. The ladder and
public game histories change; these statements refer to the recorded snapshots.

## Method and completed comparisons

Every opening is played with both colours at equal clocks through the unmodified
harness. Timed matches run separately from training and CPU benchmarks. Screening
uses the eight harness openings at 3 s + 0.1 s. Confirmation uses four different
rated opening positions at 10 s + 0.1 s. These are local selection experiments,
not measurements on the competition CPU or estimates of podium strength.

| Candidate | Opponent | Clock | Games | Wins / draws / losses |
| --- | --- | --- | ---: | --- |
| Tactical leaf generation | Submitted NN | 3 s + 0.1 s | 16 | 5 / 5 / 6 |
| Tactical generation plus guarded null-move pruning | Submitted NN | 3 s + 0.1 s | 16 | 7 / 6 / 3 |
| Guarded null-move candidate, confirmation | Submitted NN | 10 s + 0.1 s | 8 | 4 / 1 / 3 |

Tactical leaf generation alone reduced total depth-five benchmark time from
0.579 s to 0.327 s across three positions, with unchanged scores and chosen moves.
It did not demonstrate a strength gain in its match screen. The guarded candidate
scored 11 wins, seven draws and six losses across the two completed comparisons.
The sample is small; uncertainty and selection effects remain substantial.

## Search changes

Quiescence generates legal captures and promotions without checking every quiet
move. If no tactical move exists, it still checks for a legal quiet move before
returning a static score. This preserves stalemate detection, en passant and all
promotion choices.

Null-move pruning tests whether passing still exceeds a narrow search bound. It
is selective, not an exact chess result. It is disabled in check, in pawn-only
endings, near fifty-move and game-length limits, and immediately after another
pass. Synthetic lines cannot read or write real-game cached scores or use real
repetition history. Interrupted probes restore side-to-move and en passant.

## Independent game review

The site now provides Stockfish 16 analysis at depth 16. Its evaluation arrays and
best-move suggestions for all nineteen games are retained under `reviews/` in the
artifact directory. They are analysis evidence only and are not shipped as lookup data.
Finite-depth evaluation can still be wrong; these scores provide an independent
check on the engine's own optimistic logs.

| Position | Played | Site suggestion | Evaluation loss |
| --- | --- | --- | ---: |
| Round 76, White move 16 | Nxf7 | h4 | 158 cp |
| Round 79, Black move 10 | gxf6 | Bxf6 | 139 cp |
| Round 85, Black move 22 | f5 | Qg6 | 235 cp |
| Round 86, Black move 12 | g5 | Be7 | 143 cp |
| Round 86, Black move 18 | Bxf4 | Ng4 | 140 cp |
| Round 86, Black move 20 | Ne3+ | b5 | 873 cp |

In round 86, the decisive error is 20...Ne3+, not just the earlier king weakening.
White's position rises from +53 to +926 cp in the site's review. The local replay
reaches depth eight with the new search but still selects Nxf7 in round 76 and g5
in round 86. Increased depth is not evidence that all observed weaknesses are fixed.

The additional late-move reduction scored 7 wins, three draws and six losses
against the guarded candidate at 3 s + 0.1 s. That marginal result did not justify
including another selective heuristic in this release.

## Refreshed podium, round 90

The final ladder check has AlphaFish (Emile Andrieu) first, stockfih (what even
is en passant) second, and PSL God Matt Bomer (JBG fam) third. Their ten latest
games each comprise 26 distinct PGNs, saved separately in `podium-round-90/`.
The earlier podium collection remains useful historical evidence.

- AlphaFish, round 90 against Sobriety: meets a kingside attack with central play
  and a queenside passer. 37.b8=Q draws the rook away, followed by simplification
  into a favourable minor-piece ending. It eventually executes bishop-and-knight
  mate on move 183. This is evidence of conversion ability, though the long
  manoeuvring also shows that the leader does not convert every advantage quickly.
- stockfih, round 85 against Gijs Smit: sacrifices the exchange with
  19...Rxg5, creates an advanced e-pawn and coordinates queen, bishop and knight.
  After 62...Ng3+ the exchanges lead to a bishop ending: ...Bb8 stops White's
  b-pawn while the king escorts the f-pawn to promotion. Mate follows on move 84.
- PSL, round 90 against Ryan Vincent: wins central pawns, exchanges queens and
  creates connected queenside passers. The threat of b-pawn promotion competes
  with a kingside passer; 66.g8=Q leads to mate on move 72.

The common lesson is conversion and defensive calculation, not indiscriminate
king attacks. The PGNs do not reveal their search depths or private implementations.

## Final validation and unresolved weaknesses

The production candidate passed Ruff, strict mypy, all nineteen correctness tests
and both gate games. The final full-clock comparison at 120 s + 0.5 s from the
round-86 opening scored **zero wins, one draw and one loss** against the original
submission. Two games are insufficient to estimate strength, but this result does
not confirm the favourable short-clock screening result at competition time controls.

Replays of round 86 at the recorded remaining clocks still choose 18...Bxf4 and
20...Ne3+. The latter decisive blunder therefore remains unresolved. The change
improves search throughput and the short-clock sample; it is not a demonstrated
solution to the rated losses or a proven route to first place. The original
submission is preserved for comparison and rollback.
