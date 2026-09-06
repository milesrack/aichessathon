"""Search invariants that affect legal play, adjudication and clock safety."""

import contextlib
import io
import time
import unittest
from unittest.mock import patch

import chess

from agent import (
    EXACT,
    INFINITY,
    MATE,
    TABLE_SIZE,
    Engine,
    Entry,
    Search,
    SearchStopped,
    _load_score,
    _store_score,
    evaluate,
    position_key,
)


def searcher(board: chess.Board, table: list[Entry | None] | None = None) -> Search:
    return Search(board, table if table is not None else [None] * TABLE_SIZE, time.monotonic() + 10)


class SearchTests(unittest.TestCase):
    def test_terminal_positions_at_main_and_quiescence_horizons(self) -> None:
        cases = (
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 100 300", -MATE),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
            ("7k/5K2/5NN1/8/8/8/8/8 b - - 0 1", -MATE),
        )
        for fen, expected in cases:
            for depth in (0, 2):
                with self.subTest(fen=fen, depth=depth):
                    board = chess.Board(fen)
                    self.assertTrue(board.is_valid())
                    self.assertEqual(searcher(board).search(depth, -INFINITY, INFINITY), expected)

    def test_quiescence_searches_quiet_check_evasions(self) -> None:
        board = chess.Board("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertTrue(board.is_check())
        self.assertFalse(list(board.generate_legal_captures()))

        def quiet_evaluation(position: chess.Board) -> int:
            self.assertFalse(position.is_check())
            return evaluate(position)

        with patch("agent.evaluate", side_effect=quiet_evaluation) as evaluator:
            score = searcher(board).search(0, -INFINITY, INFINITY)
        self.assertGreater(evaluator.call_count, 0)
        self.assertGreater(score, -MATE + 100)
        self.assertLess(score, 0)

    def test_rook_underpromotion_avoids_stalemate(self) -> None:
        fen = "8/k1P5/2K5/8/8/8/8/8 w - - 0 1"
        engine = Engine()
        with contextlib.redirect_stdout(io.StringIO()):
            move = engine.get_move(fen, 5000)
        self.assertEqual(move, "c7c8r")
        queen = chess.Board(fen)
        queen.push_uci("c7c8q")
        self.assertTrue(queen.is_stalemate())

    def test_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        with contextlib.redirect_stdout(io.StringIO()):
            board.push_uci(Engine().get_move(board.fen(), 1000))
        self.assertTrue(board.is_checkmate())

    def test_timeout_restores_board_and_repetition_counts(self) -> None:
        board = chess.Board()
        search = searcher(board)
        original_counts = search.repetitions.copy()

        def interrupt() -> None:
            if search.nodes == 100:
                raise SearchStopped

        with (
            patch.object(search, "check_time", side_effect=interrupt),
            self.assertRaises(SearchStopped),
        ):
            search.search(5, -INFINITY, INFINITY)
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertEqual(board.move_stack, [])
        self.assertEqual(search.repetitions, original_counts)

    def test_special_moves_restore_state(self) -> None:
        cases = (
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("8/k1P5/2K5/8/8/8/8/8 w - - 0 1", "c7c8r"),
        )
        for fen, uci in cases:
            with self.subTest(uci=uci):
                board = chess.Board(fen)
                search = searcher(board)
                before = search.repetitions.copy()
                previous = search._push(board.parse_uci(uci))
                search._pop(previous)
                self.assertEqual(board.fen(), fen)
                self.assertEqual(search.repetitions, before)

    def test_repetition_history_survives_opponent_synchronisation(self) -> None:
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            board.push_uci(uci)
        engine = Engine()
        engine.board = board.copy()
        board.push_uci("f6g8")
        synced = engine._synchronise(chess.Board(board.fen()), time.monotonic() + 1)
        self.assertEqual(len(synced.move_stack), 8)
        self.assertEqual(searcher(synced).search(2, -INFINITY, INFINITY), 0)
        self.assertFalse(chess.Board(board.fen()).is_repetition(3))

    def test_transposition_rejects_different_repetition_history(self) -> None:
        board = chess.Board()
        key = position_key(board)
        table: list[Entry | None] = [None] * TABLE_SIZE
        table[hash(key) & (TABLE_SIZE - 1)] = Entry(
            key, 8, 12345, EXACT, chess.Move.from_uci("e2e4"), 0, 0, frozenset()
        )
        score = searcher(board, table).search(1, -INFINITY, INFINITY)
        self.assertNotEqual(score, 12345)

    def test_fifty_move_and_ply_cap_contexts_are_not_cached_wins(self) -> None:
        table: list[Entry | None] = [None] * TABLE_SIZE
        ordinary = chess.Board("7k/8/8/8/8/8/2Q5/K7 w - - 0 1")
        self.assertGreater(searcher(ordinary, table).search(2, -INFINITY, INFINITY), 500)
        for counters in ("98 1", "0 300"):
            with self.subTest(counters=counters):
                board = chess.Board("7k/8/8/8/8/8/2Q5/K7 w - - " + counters)
                self.assertEqual(searcher(board, table).search(2, -INFINITY, INFINITY), 0)

    def test_mate_scores_keep_distance_across_root_changes(self) -> None:
        self.assertEqual(_load_score(_store_score(MATE - 8, 3), 1), MATE - 6)
        self.assertEqual(_load_score(_store_score(-MATE + 8, 3), 1), -MATE + 6)
        self.assertEqual(_load_score(_store_score(120, 3), 1), 120)

    def test_position_identity_respects_rights_and_legal_en_passant(self) -> None:
        pinned = chess.Board("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        no_target = chess.Board(pinned.fen(en_passant="fen").replace("d6", "-"))
        self.assertEqual(position_key(pinned), position_key(no_target))
        legal = chess.Board("k7/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        self.assertNotEqual(
            position_key(legal), position_key(chess.Board(legal.fen().replace("d6", "-")))
        )
        castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        without = chess.Board(castling.fen().replace("KQkq", "-"))
        self.assertNotEqual(position_key(castling), position_key(without))

    def test_colour_symmetry(self) -> None:
        board = chess.Board()
        for uci in ("e2e4", "c7c5", "g1f3", "b8c6", "f1b5", "g8f6"):
            board.push_uci(uci)
            self.assertEqual(evaluate(board), evaluate(board.mirror()))

    def test_low_clock_returns_legal_move_without_search(self) -> None:
        for fen in (
            chess.STARTING_FEN,
            "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "k4r2/8/8/8/8/8/8/4K2R w K - 0 1",
        ):
            with self.subTest(fen=fen), patch.object(Search, "search") as search:
                with contextlib.redirect_stdout(io.StringIO()):
                    move = Engine().get_move(fen, 10)
                search.assert_not_called()
                self.assertIn(chess.Move.from_uci(move), chess.Board(fen).legal_moves)


if __name__ == "__main__":
    unittest.main()
