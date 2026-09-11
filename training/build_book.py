"""Build a Polyglot opening book from CC0 evaluations and offline analysis.

Only the data file ships. Runtime lookup is restricted to fullmove 20 or below.
The external analysis engine is a development tool, never part of the agent.
"""

import argparse
import collections
import io
import json
import struct
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.polyglot
import zstandard


def encoded(board: chess.Board, move: chess.Move) -> int:
    target = move.to_square
    if board.is_castling(move):
        target = chess.square(7 if target > move.from_square else 0, chess.square_rank(target))
    return target | move.from_square << 6 | ((move.promotion or 1) - 1) << 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=12000)
    parser.add_argument("--seconds", type=float, default=0.08)
    args = parser.parse_args()
    if args.positions <= 0 or args.seconds <= 0:
        parser.error("positions and seconds must be positive")
    entries: dict[int, tuple[int, int]] = {}
    rows = 0
    with args.evaluations.open("rb") as source:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(source))
        for text in reader:
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                break  # A bounded download can end in the middle of a row.
            rows += 1
            board = chess.Board(row["fen"])
            if len(board.piece_map()) < 24:
                continue
            analysis = max(row["evals"], key=lambda item: item["depth"])
            if analysis["depth"] < 20:
                continue
            move = chess.Move.from_uci(analysis["pvs"][0]["line"].split()[0])
            if move not in board.legal_moves:
                continue
            key = chess.polyglot.zobrist_hash(board)
            if key not in entries or analysis["depth"] > entries[key][1]:
                entries[key] = encoded(board, move), analysis["depth"]
    print(f"Imported {len(entries)} entries from {rows} evaluations", flush=True)
    queue: collections.deque[chess.Board] = collections.deque()
    seen: set[str] = set()

    def enqueue(board: chess.Board) -> None:
        key = " ".join(board.fen().split()[:4])
        if board.fullmove_number <= 20 and key not in seen and not board.is_game_over():
            seen.add(key)
            queue.append(board.copy(stack=False))

    # Seed all observed early positions, not just our own games or our own moves.
    for path in sorted(args.games.rglob("*.pgn")):
        if any(part in str(path) for part in ("screen", "holdout", "full-clock", "final-push")):
            continue
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        if game is None:
            continue
        board = game.board()
        enqueue(board)
        for move in game.mainline_moves():
            board.push(move)
            if board.fullmove_number > 20:
                break
            enqueue(board)
    from harness.rules import OPENINGS

    for _, fen in OPENINGS:
        enqueue(chess.Board(fen))
    seeds = len(queue)
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        engine.configure({"Threads": 1, "Hash": 128})
        for count in range(args.positions):
            if not queue:
                break
            board = queue.popleft()
            lines = engine.analyse(board, chess.engine.Limit(time=args.seconds), multipv=3)
            best = lines[0]
            entries[chess.polyglot.zobrist_hash(board)] = encoded(board, best["pv"][0]), 65535
            for line in lines:
                child = board.copy(stack=False)
                for ply, move in enumerate(line["pv"][:10]):
                    if child.fullmove_number > 20:
                        break
                    # Alternative root moves are not recommendations. Their
                    # principal continuations are conditional best responses.
                    quality = max(1, line.get("depth", 1) - ply)
                    key = chess.polyglot.zobrist_hash(child)
                    if (
                        ply > 0
                        and quality >= 12
                        and (key not in entries or quality > entries[key][1])
                    ):
                        entries[key] = encoded(child, move), quality
                    child.push(move)
                    if ply == 0:
                        enqueue(child)
            if count % 500 == 0:
                print(f"Analysed {count + 1}; queued {len(queue)}", flush=True)
        engine_name = engine.id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for key, (packed_move, _) in sorted(entries.items()):
            output.write(struct.pack(">QHHI", key, packed_move, 1, 0))
    print(
        json.dumps(
            {
                "entries": len(entries),
                "seeds": seeds,
                "engine": engine_name,
                "analysed": count + 1,
                "seconds_per_position": args.seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
