# Search iteration, 10 September 2026

The active v5 archive has SHA-256
`bde67bd922087e2e8db6b1d2c747539ce829bdd1136067f9c8f30d9a9855fc14`;
its source is commit `1f08516`. The dashboard snapshot showed four wins, one
draw and two losses since upload, rating 1649 and rank 200. Both losses were
checkmates, with successful initialisation and time remaining.

In round 96 against Vanish Onigiri, queenside pawn captures allowed a kingside
attack. The site's Stockfish 16 depth-16 review gives 20...hxg5 an evaluation
loss of 255 cp and suggests ...Ne8. In round 97 against Defense, 16...Qc8
allowed a damaging queen exchange: the review changes from +171 to +518 cp
for White and suggests ...h5. These are finite-depth analysis results.

The round-97 ladder snapshot puts BINI Alpha (Opus Carlsen, Exeter) first at
2950. Its win against Gijs Smit converts an outside passed pawn after a rook
sacrifice. Public moves do not reveal its private implementation.

## Experiments

Both candidates retain the team's trained weights. A new king-danger term
measuring coordinated piece attacks scored **4 wins, 2 draws and 10 losses**
against v5 in 16 paired games at 3 s + 0.1 s. It was rejected.

The second candidate extends checked positions by one ply during the main
search. Quiescence already searches check evasions. Existing repetition,
deadline and maximum-ply guards remain active. Reported iterative depths are
nominal; forcing lines can run deeper.

Sources, PGNs, candidate snapshots and local results are preserved in
`artifacts/2026-09-10/`. The harness is unchanged. Timed comparisons run
sequentially, without concurrent training or benchmarks. Local timings do
not reproduce the platform CPU; small match samples cannot establish an Elo gain.

## Decision

Check extensions scored **5 wins, 5 draws and 6 losses** in the same 16-game
screen. Neither candidate demonstrated an improvement. All 32 games ended
normally; no illegal moves, crashes or clock failures were reported.

Isolated-FEN replays used the clocks recorded before each selected move.
Both v5 and the check-extension candidate still chose 20...hxg5 in round 96.
The latter changed round 97's 16...Qc8 to ...Nf6, but this was not the site's
suggestion and is not established as a correction. Both locally found
13...Be4 instead of the played ...Kf8; these fresh-process replays lack the
original transposition table and history and run on different hardware.

**Neither candidate is promoted.** Production source and the canonical
`agent.zip` remain the exact v5 build. Candidate sources and tests are retained
only with the experiment artifacts. No upload was performed. Longer-clock
matches were not run because neither candidate passed the initial screen.

The next useful investigation is quiet defensive calculation around the
round-96 h-file attack, supported by a broader tactical test set. These results
do not justify adding a king penalty or unconditional check extensions.
