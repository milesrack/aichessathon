"""Independent legality, adjudication and clock checks for the compiled engine."""

import contextlib
import io
import random
import time
import unittest

import chess
import numpy as np

import agent
from agent import (
    INF,
    MATE,
    Array,
    Engine,
    Work,
    compiled,
    decode,
    encode,
    evaluate,
    generate,
    insufficient,
    make,
    position_hash,
    search,
    unmake,
    workspace,
)


@compiled
def perft(board: Array, state: Array, depth: int) -> int:
    if depth == 0:
        return 1
    moves = np.zeros(512, dtype=np.int64)
    undo = np.zeros(10, dtype=np.int64)
    count = generate(board, state, moves, undo)
    if depth == 1:
        return count
    total = 0
    for index in range(count):
        make(board, state, moves[index], undo)
        total += perft(board, state, depth - 1)
        unmake(board, state, moves[index], undo)
    return total


def score(work: Work, depth: int) -> int:
    work.stats[:3] = 0
    return search(work, depth, -INF, INF, 0, time.monotonic() + 10)


class SearchTests(unittest.TestCase):
    def test_null_bound_is_not_reported_as_a_proven_mate(self) -> None:
        work = workspace(chess.Board("7k/7p/5K2/8/8/8/8/6RQ w - - 0 1"))
        value = search(work, 5, 0, 1, 1, time.monotonic() + 10)
        self.assertGreaterEqual(value, 1)
        self.assertLess(value, MATE - 256)

    def test_synthetic_search_keeps_game_cache_unchanged(self) -> None:
        work = workspace(chess.Board())
        score(work, 2)
        slot = position_hash(work.board, work.state) & (len(work.table_keys) - 1)
        work.table[slot, 1] = 12345
        before = work.table.copy(), work.table_keys.copy()
        result = search(work, 2, -INF, INF, 0, time.monotonic() + 10, True, False)
        self.assertNotEqual(result, 12345)
        np.testing.assert_array_equal(work.table, before[0])
        np.testing.assert_array_equal(work.table_keys, before[1])

    def test_interrupted_null_probe_restores_en_passant_and_side(self) -> None:
        work = workspace(chess.Board("7k/8/8/3pP3/8/8/3Q4/4K3 w - d6 0 1"))
        before = work.board.copy(), work.state.copy()
        work.stats[0] = 254
        search(work, 3, -1, 0, 1, time.monotonic() - 1)
        self.assertEqual(work.stats[1], 1)
        np.testing.assert_array_equal(work.board, before[0])
        np.testing.assert_array_equal(work.state, before[1])

    def test_pawn_endgame_search_does_not_use_null_pruning(self) -> None:
        position = chess.Board("8/8/8/8/2k5/2p5/2K5/8 w - - 0 1")
        self.assertFalse(position.is_game_over())
        enabled, disabled = workspace(position), workspace(position)
        expected = search(disabled, 5, -1001, -1000, 1, time.monotonic() + 10, False, False)
        actual = search(enabled, 5, -1001, -1000, 1, time.monotonic() + 10, False, True)
        self.assertEqual(actual, expected)
        self.assertEqual(enabled.stats[0], disabled.stats[0])

    def test_tactical_generation_matches_legal_captures_and_promotions(self) -> None:
        rng = random.Random(19823)
        positions = [chess.Board(fen) for fen in (
            chess.STARTING_FEN,
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "1r5k/P7/8/8/8/8/8/6K1 w - - 0 1",
        )]
        board = chess.Board()
        for _ in range(500):
            if board.is_game_over():
                board.reset()
            board.push(rng.choice(list(board.legal_moves)))
            if not board.is_check():
                positions.append(board.copy())
        for position in positions:
            work = workspace(position)
            before = work.board.copy(), work.state.copy()
            count = generate(work.board, work.state, work.moves[0], work.undo[0], True)
            legal = list(position.legal_moves)
            expected = {m for m in legal if position.is_capture(m) or m.promotion}
            self.assertEqual(count == -1, not legal, position.fen())
            self.assertEqual(
                {decode(int(m)) for m in work.moves[0, :max(0, count)]}, expected,
                position.fen(),
            )
            np.testing.assert_array_equal(work.board, before[0])
            np.testing.assert_array_equal(work.state, before[1])

    def test_standard_perft_totals_and_state_restoration(self) -> None:
        cases = (
            (chess.STARTING_FEN, 4, 197281),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),
        )
        for fen, depth, expected in cases:
            with self.subTest(fen=fen):
                board, state = encode(chess.Board(fen))
                before = board.copy(), state.copy()
                self.assertEqual(perft(board, state, depth), expected)
                np.testing.assert_array_equal(board, before[0])
                np.testing.assert_array_equal(state, before[1])

    def test_move_generation_and_transitions_match_python_chess(self) -> None:
        positions = [
            chess.Board(fen)
            for fen in (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                "k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                "1r5k/P7/8/8/8/8/8/6K1 w - - 0 1",
            )
        ]
        rng = random.Random(1944)
        position = chess.Board()
        for _ in range(200):
            if position.is_game_over():
                position.reset()
            position.push(rng.choice(list(position.legal_moves)))
            positions.append(position.copy(stack=False))
        for position in positions:
            with self.subTest(fen=position.fen()):
                board, state = encode(position)
                before = board.copy(), state.copy()
                moves = np.zeros(512, dtype=np.int64)
                undo = np.zeros(10, dtype=np.int64)
                count = generate(board, state, moves, undo)
                self.assertEqual({decode(int(m)) for m in moves[:count]}, set(position.legal_moves))
                self.assertEqual(insufficient(board), position.is_insufficient_material())
                for move in moves[:count]:
                    make(board, state, move, undo)
                    reference = position.copy(stack=False)
                    reference.push(decode(int(move)))
                    expected_board, expected_state = encode(reference)
                    np.testing.assert_array_equal(board, expected_board)
                    np.testing.assert_array_equal(state, expected_state)
                    unmake(board, state, move, undo)
                    np.testing.assert_array_equal(board, before[0])
                    np.testing.assert_array_equal(state, before[1])

    def test_terminal_positions_at_main_and_quiescence_horizons(self) -> None:
        cases = (
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 100 300", -MATE),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
            ("7k/5K2/5NN1/8/8/8/8/8 b - - 0 1", -MATE),
        )
        for fen, expected in cases:
            for depth in (0, 2):
                with self.subTest(fen=fen, depth=depth):
                    self.assertEqual(score(workspace(chess.Board(fen)), depth), expected)

    def test_quiescence_searches_quiet_check_evasions(self) -> None:
        position = chess.Board("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertTrue(position.is_check())
        self.assertFalse(list(position.generate_legal_captures()))
        work = workspace(position)
        result = score(work, 0)
        self.assertGreater(int(work.stats[0]), 1)
        self.assertGreater(result, -MATE + 100)
        self.assertLess(result, 0)

    def test_underpromotion_and_mate(self) -> None:
        for fen, expected in (
            ("8/k1P5/2K5/8/8/8/8/8 w - - 0 1", "c7c8r"),
            ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", None),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                move = Engine().get_move(fen, 5000)
            if expected is not None:
                self.assertEqual(move, expected)
            else:
                board = chess.Board(fen)
                board.push_uci(move)
                self.assertTrue(board.is_checkmate())

    def test_timeout_restores_search_state(self) -> None:
        work = workspace(chess.Board())
        score(work, 1)
        completed_move = int(work.stats[2])
        work.stats[:2] = 0
        before = work.board.copy(), work.state.copy()
        search(work, 12, -INF, INF, 0, time.monotonic() - 1)
        self.assertEqual(work.stats[1], 1)
        np.testing.assert_array_equal(work.board, before[0])
        np.testing.assert_array_equal(work.state, before[1])
        self.assertEqual(work.stats[2], completed_move)

    def test_cache_rejects_different_repetition_history(self) -> None:
        board = chess.Board()
        keys = [position_hash(*encode(board))]
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
            keys.append(position_hash(*encode(board)))
        work = workspace(board)
        score(work, 2)
        slot = keys[-1] & (len(work.table_keys) - 1)
        work.table[slot, 1] = 12345
        self.assertEqual(score(work, 1), 12345)
        work.path[: len(keys)] = keys
        work.stats[3] = len(keys)
        self.assertNotEqual(score(work, 1), 12345)

    def test_cached_mate_distance_changes_with_search_ply(self) -> None:
        work = workspace(chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
        self.assertEqual(score(work, 1), MATE - 1)
        self.assertEqual(search(work, 1, -INF, INF, 2, time.monotonic() + 1), MATE - 3)

    def test_opponent_synchronisation_preserves_threefold_history(self) -> None:
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            board.push_uci(move)
        engine = Engine()
        engine.board = board.copy()
        board.push_uci("f6g8")
        with contextlib.redirect_stdout(io.StringIO()):
            engine.get_move(board.fen(), 5000)
        self.assertEqual(engine.work.stats[3], 9)
        self.assertEqual(score(engine.work, 2), 0)

    def test_draw_counters_prevent_cached_wins(self) -> None:
        work = workspace(chess.Board("7k/8/8/8/8/8/2Q5/K7 w - - 0 1"))
        self.assertGreater(score(work, 2), 500)
        for counters in ("98 1", "0 300"):
            work.board[:], work.state[:] = encode(
                chess.Board("7k/8/8/8/8/8/2Q5/K7 w - - " + counters)
            )
            self.assertEqual(score(work, 2), 0)

    def test_position_hash_respects_legal_en_passant_and_castling(self) -> None:
        for fen, same in (
            ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", True),
            ("k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1", False),
        ):
            board, state = encode(chess.Board(fen))
            before = board.copy()
            key = position_hash(board, state)
            np.testing.assert_array_equal(board, before)
            without = position_hash(*encode(chess.Board(fen.replace("d6", "-"))))
            self.assertEqual(key == without, same)
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        self.assertNotEqual(
            position_hash(*encode(chess.Board(fen))),
            position_hash(*encode(chess.Board(fen.replace("KQkq", "-")))),
        )

    def test_colour_symmetry(self) -> None:
        board = chess.Board()
        for move in ("e2e4", "c7c5", "g1f3", "b8c6", "f1b5", "g8f6"):
            board.push_uci(move)
            self.assertEqual(evaluate(*encode(board)), evaluate(*encode(board.mirror())))

    def test_low_clock_returns_legal_move_without_search(self) -> None:
        engine = Engine()
        with contextlib.redirect_stdout(io.StringIO()):
            move = engine.get_move(chess.STARTING_FEN, 10)
        self.assertIn(chess.Move.from_uci(move), chess.Board().legal_moves)
        self.assertEqual(engine.work.stats[0], 0)

    def test_neural_cache_matches_full_recomputation(self) -> None:
        positions = []
        for fen in (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "8/P1k5/8/8/8/8/5K1p/8 w - - 0 1",
        ):
            board = chess.Board(fen)
            positions.append(board.copy())
            for move in board.legal_moves:
                child = board.copy()
                child.push(move)
                positions.append(child)
        board = chess.Board()
        randomiser = random.Random(9381)
        for _ in range(150):
            if board.is_game_over():
                board = chess.Board()
            board.push(randomiser.choice(list(board.legal_moves)))
            positions.append(board.copy())
        positions.extend(reversed(positions.copy()))
        work = workspace(chess.Board())
        for board in positions:
            encoded, state = encode(board)
            work.board[:] = encoded
            work.state[:] = state
            hidden = []
            for colour in (chess.WHITE, chess.BLACK):
                indices = [
                    (piece.piece_type - 1) * 64
                    + (square if colour else square ^ 56)
                    + (0 if piece.color == colour else 384)
                    for square, piece in board.piece_map().items()
                ]
                hidden.append(np.maximum(agent.NN_EMBED[indices].sum(axis=0), 0))
            expected = float(200 * np.tanh((hidden[0] - hidden[1]) @ agent.NN_OUT / 2))
            expected *= 1 if board.turn else -1
            self.assertAlmostEqual(agent.residual(work), expected, places=7)
            np.testing.assert_array_equal(work.board, encoded)
            np.testing.assert_array_equal(work.state, state)
            work.state[0] *= -1
            self.assertAlmostEqual(agent.residual(work), -expected, places=7)


if __name__ == "__main__":
    unittest.main()
