"""Filter an official Lichess evaluation-export prefix using a frozen classical agent."""

import argparse
import hashlib
import importlib
import io
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard


def features_for(board: chess.Board) -> list[list[int]]:
    result = []
    for colour in (board.turn, not board.turn):
        indices = sorted(
            (piece.piece_type - 1) * 64
            + (square if colour else square ^ 56)
            + (0 if piece.color == colour else 384)
            for square, piece in board.piece_map().items()
        )
        result.append(indices + [768] * (32 - len(indices)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=1_000_000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.baseline.resolve()))
    baseline = importlib.import_module("agent")
    work = baseline.workspace(chess.Board())
    features = np.zeros((args.max_positions, 2, 32), dtype=np.int16)
    bases = np.zeros(args.max_positions, dtype=np.int16)
    targets = np.zeros(args.max_positions, dtype=np.int16)
    validation = np.zeros(args.max_positions, dtype=bool)
    seen = set()
    count = examined = tactical_rejections = 0
    with (
        args.source.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as reader,
    ):
        for line in io.TextIOWrapper(reader):
            if not line.endswith("\n"):
                break  # A bounded download may end within a JSON record.
            row = json.loads(line)
            examined += 1
            analyses = [
                entry for entry in row["evals"] if entry["depth"] >= 16 and "cp" in entry["pvs"][0]
            ]
            if not analyses:
                continue
            best = max(analyses, key=lambda entry: (entry["depth"], entry["knodes"]))
            cp = best["pvs"][0]["cp"]
            if abs(cp) > 1500:
                continue
            board = chess.Board(row["fen"] + " 0 1")
            if not board.is_valid() or board.is_check() or board.is_game_over():
                continue
            position, state = baseline.encode(board)
            work.board[:] = position
            work.state[:] = state
            work.stats[:] = [0, 0, 0, 1]
            static = baseline.evaluate(position, state)
            quiet = baseline.search(
                work, 0, -baseline.INF, baseline.INF, 0, time.monotonic() + 0.01
            )
            if work.stats[1] or abs(quiet - static) > 40:
                tactical_rejections += 1
                continue
            encoded = features_for(board)
            key = np.asarray(sorted(encoded), dtype=np.int16).tobytes()
            digest = hashlib.sha256(key).digest()
            if digest in seen:
                continue
            seen.add(digest)
            features[count] = encoded
            bases[count] = static
            targets[count] = cp * (1 if board.turn else -1)
            validation[count] = digest[0] % 10 == 0
            count += 1
            if count % 100_000 == 0:
                print(f"{count} quiet positions from {examined} rows", flush=True)
            if count == args.max_positions:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        x=features[:count],
        base=bases[:count],
        target=targets[:count],
        valid=validation[:count],
    )
    print(f"Saved {count} positions; rejected {tactical_rejections} as unstable.")


if __name__ == "__main__":
    main()
