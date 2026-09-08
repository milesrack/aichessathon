# Evaluator and search experiments

Experiments ran on 7–8 September 2026. The reference is the compiled engine at
`a3c1cd0`, not the original Python engine. Its submitted archive has SHA-256
`d4461dd5bc7bf8f52e412b2bba166790d047286c260ebcd204690dd749ad26a8`.

## Method

Each comparison uses the unmodified harness, equal clocks and both colours from every
opening. Training and benchmarks run separately from timed matches. The initial screens
use 3 s + 0.1 s; confirmation uses four other openings at 10 s + 0.1 s. These small local
comparisons select candidates; they do not establish a competition rating.

| Candidate | Opponent | Games | Wins / draws / losses |
| --- | --- | ---: | --- |
| Quiet-position neural evaluator, small dataset | Reference | 16 | 5 / 1 / 10 |
| Learnt tapered piece-square corrections | Reference | 16 | 7 / 2 / 7 |
| Late quiet-move reductions | Reference | 16 | 12 / 0 / 4 |
| Late quiet-move reductions, confirmation | Reference | 8 | 4 / 1 / 3 |
| Larger dataset, cached neural evaluation | Reference | 8 | 5 / 2 / 1 |
| Cached neural evaluation, confirmation | Reference | 8 | 5 / 1 / 2 |
| Cached neural evaluation, direct comparison at 10 s + 0.1 s | Reductions alone | 16 | 10 / 2 / 4 |
| Larger dataset, king-conditioned neural evaluation | Reference | 8 | 1 / 3 / 4 |
| Cached neural evaluation plus reductions | Reductions alone | 8 | 2 / 1 / 5 |

No game in these comparisons failed through an illegal move, crash or clock overrun.

The selected release uses the larger, unconditioned neural evaluator with the original
search. It scored positively against both classical versions. Combining the network with
reductions made it weaker, so the reductions are absent from this release. The king-conditioned
network and learnt-table variant are also excluded.

## Neural training

Every network starts from random weights in PyTorch. The
[competition rules](https://aichessathon.com/docs/rules.md) permit training on engine-labelled
positions but prohibit shipping a published chess network, including a fine-tuned one.

The data comes from the [CC0 Lichess evaluation export](https://database.lichess.org/#evals).
A bounded 128 MiB prefix supplied 1,580,923 rows. We select the deepest available analysis
of at least depth 16, using its first principal variation's centipawn score and excluding
scores outside ±1500 cp. Invalid positions, checks and terminal positions are excluded.
Positions whose classical quiescence score differs from static evaluation by more than
40 cp are also excluded. Interrupted filtering searches are discarded.

Canonical feature deduplication leaves 926,966 quiet positions: 832,883 for training and
94,083 for validation. Opposite perspectives stay in the same partition. This is a prefix
sample with a position-based split; source-game identifiers are unavailable, so related
positions can cross partitions. Competition PGNs are not training data.

The main network sums 32-dimensional piece-square embeddings, applies ReLU and produces a
scalar. Subtracting the two perspectives enforces antisymmetry. A tanh bounds its correction
to the classical score at ±200 cp. Training uses sigmoid-space squared error, AdamW and
30 epochs, with seed 9371. Neither the sigmoid scale nor the quietness threshold is a
universal chess constant.

The larger model reaches roughly 103 cp mean validation error against 131 cp for the
classical evaluator on the same set. Conditioning features on eight king-location regions
reduces this to about 99 cp, but that model loses its match comparison. Lower score error
is therefore insufficient evidence of better play.

## Inference and search

Inference uses our own Numba code, loading the trained NumPy weights at import. It retains
the last evaluated board and both feature sums, updating only changed piece features.
Tests compare this cache with full recomputation over 10,000 board transitions. Exported
inference agrees with PyTorch within 0.001 cp on 200 positions.

A probe of the earlier evaluator measured 0.825 microseconds per evaluation versus
0.325 for classical evaluation. Its sampled depth-five search also visited more nodes.
Evaluation cost and changes to the search tree both matter when assessing a model.

The search experiment reduces late quiet moves by one ply, excluding checks, check evasions,
promotions, captures and killers. Any reduced search that improves alpha is repeated at
normal depth. This saves work but can miss a quiet move that appears unpromising at reduced
depth. Its positive match results justify further use; they do not make selective search exact.

## Research basis

- [Tan and Watkinson Medina](https://arxiv.org/html/2412.17948v1) investigate filtering
  unstable training positions. Their experiments concern Xiangqi; their measured gains and
  thresholds cannot be assumed to transfer to this engine.
- [NNUE training concepts](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
  explain sparse accumulators and sigmoid-space losses. We use the concepts, not an engine
  implementation, port or published network.
- [DeepMind's searchless chess work](https://arxiv.org/abs/2402.04494) explores transformers
  with up to 270 million parameters. That scale is a poor fit for this competition's storage
  and single-core constraints.

Training scripts, input data, weights, frozen candidates, logs and PGNs are retained locally
under `.agents/iterations/2026-09-07-neural/`. They are research evidence, not submission assets.

## Reproducibility

The maintained training script reproduces every released weight exactly from the retained
prepared dataset. Relevant SHA-256 hashes:

- Source prefix: `3989b7568bcbccf23414bbc0004d2ccadd9325e4613da10600e5e51da96d835a`.
- Prepared dataset: `e9e44c471525a70eb6799656f844f2839c6a2a799894f349a0f1012a45d64fa6`.
- Released weights: `caecd4c672ea4a14d8896cb6ed47f2e3755a19cbed94b5a4b9e4fdbdc451a3fd`.

See [the training instructions](../training/README.md). The selected neural model's combined
record is 20 wins, five draws and seven losses across 32 games against the two classical
versions. No published network, engine or position-score lookup database is shipped.
