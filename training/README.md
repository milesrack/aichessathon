# Evaluator training

Run `make setup` to install the training dependencies.

## Prepare data

Use the frozen classical evaluator for quiet-position filtering:

```sh
mkdir -p .agents/training/baseline
git show a3c1cd0:agent.py > .agents/training/baseline/agent.py
curl -fsSL --range 0-134217727 --max-filesize 134217728 \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  -o .agents/training/evals-prefix.zst
uv run python -m training.prepare .agents/training/evals-prefix.zst \
  .agents/training/quiet.npz --baseline .agents/training/baseline
```

Preparation reads a 128 MiB prefix of the
[CC0 Lichess evaluation export](https://database.lichess.org/#evals):

- Keep valid, non-terminal positions outside check with analysis depth ≥16 and scores within ±1500 cp.
- Reject positions where classical quiescence and static evaluation differ by more than 40 cp.
- Deduplicate by canonical features and assign training/validation partitions by hash.
- Keep opposite perspectives in the same partition.

The retained dataset has **832,883 training and 94,083 validation positions**. Source-game
identifiers are unavailable, so related positions can cross partitions. The upstream export
can change, and the 10 ms filtering deadline can retain different rows on different machines.
Exact weight reproduction requires the retained dataset and hashes in the
[experiment record](../docs/evaluator-research.md).

## Train

```sh
uv run python -m training.train .agents/training/quiet.npz \
  .agents/training/evaluator.npz
```

Training uses random initialisation, seed 9371, AdamW and 30 epochs.

The best validation checkpoint is exported as `.npz`, alongside JSON metadata containing
dataset and weight hashes and per-epoch metrics.

## Evaluate

1. Check exported inference against PyTorch and cached sums against full recomputation.
2. Compare against the reference engine with both colours and equal clocks.
3. Include additional openings and competition-length clocks before selecting weights.

See [evaluator experiments](../docs/evaluator-research.md) for architecture and results.
