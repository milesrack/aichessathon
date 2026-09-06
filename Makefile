SHELL := /bin/bash

.PHONY: setup play arena zip test gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy

zip:
	uv run python -m harness.package --out agent.zip

test:
	uv run python -m unittest discover -s tests -v

gate:
	uv run ruff check .
	uv run mypy
	$(MAKE) test
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
