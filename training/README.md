# Training the evaluator

The released network was trained from random weights on 832,883 quiet positions, with 94,083
held out. It uses two perspectives, shared 32-dimensional piece-square embeddings and a
bounded residual relative to the frozen classical evaluator. PyTorch trains the model;
Numba runs the exported weights during play.

The scripts require the repository environment (`make setup`). `zstandard` is a development
dependency used only to decompress training data; the submitted agent does not import it.

## Prepare data

Freeze the classical reference before preparing data. Using the neural engine to filter its
own training data changes the experiment.

```sh
mkdir -p .agents/training/baseline
git show a3c1cd0:agent.py > .agents/training/baseline/agent.py
curl -fsSL --range 0-134217727 --max-filesize 134217728 \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  -o .agents/training/evals-prefix.zst
uv run python -m training.prepare .agents/training/evals-prefix.zst \
  .agents/training/quiet.npz --baseline .agents/training/baseline
```

This uses a bounded prefix of the [CC0 Lichess evaluation export](https://database.lichess.org/#evals),
not a random sample of the whole database. Filtering keeps valid, non-terminal positions
outside check, with deep centipawn labels and little difference between static evaluation
and quiescence search. A canonical feature hash deduplicates positions and assigns the split.
Related positions can cross partitions because source-game identifiers are unavailable.

Filtering has a 10 ms per-position deadline. Different machines can therefore retain slightly
different rows. The exact prepared dataset used for the release is retained with the local
research evidence; use that snapshot when reproducing the weights.

## Train and compare

```sh
uv run python -m training.train .agents/training/quiet.npz \
  .agents/training/evaluator.npz
```

The script fixes the seed, architecture, optimiser and 30-epoch schedule. It selects the best
validation checkpoint and writes the weights plus a JSON record containing dataset and weight
hashes and per-epoch metrics. It never loads pretrained weights.

A lower validation loss is not grounds to replace `weights/evaluator.npz`. Freeze a candidate
with the new weights, check inference against PyTorch and full recomputation, then play paired
equal-clock games against the released agent. Include additional openings and longer clocks
before packaging. Only the selected `.npz` belongs in `weights/`; datasets, logs and training
metadata stay in the ignored research directory.

See [the experiment results](../docs/evaluator-research.md) for the comparisons that selected
this model and rejected other candidates.
