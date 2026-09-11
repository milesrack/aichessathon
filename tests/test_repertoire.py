"""Opening boundaries and real tablebase conversion, independent of search."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.polyglot
import chess.syzygy

import repertoire
from training.build_book import encoded


class RepertoireTests(unittest.TestCase):
    def test_book_does_not_repeat_a_known_position(self) -> None:
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1"):
            board.push_uci(uci)
        before = board.fen()
        move = chess.Move.from_uci("f6g8")
        with patch.object(repertoire, "_BOOK") as book:
            book.find_all.return_value = [chess.polyglot.Entry(0, 0, 1, 0, move)]
            self.assertIsNone(repertoire.opening_move(board))
        self.assertEqual(board.fen(), before)
        self.assertEqual(len(board.move_stack), 3)

    def test_rounded_distance_near_fifty_moves_falls_back(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/6R1/6K1 w - - 95 60")
        before = board.fen()
        with patch.object(repertoire, "_TABLES") as tables:
            tables.probe_wdl.return_value = -2
            tables.probe_dtz.return_value = -10
            self.assertIsNone(repertoire.endgame_move(board))
        self.assertEqual(board.fen(), before)

    def test_opening_limit_uses_incoming_fullmove(self) -> None:
        with patch.object(repertoire, "_BOOK") as book:
            book.find_all.return_value = []
            board = chess.Board()
            board.fullmove_number = 20
            repertoire.opening_move(board)
            book.find_all.assert_called_once()
            book.reset_mock()
            board.fullmove_number = 21
            repertoire.opening_move(board)
            book.find_all.assert_not_called()

    def test_book_castling_and_promotion_encoding(self) -> None:
        import struct

        for fen, uci in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 10", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 10", "e8c8"),
            ("7k/P7/8/8/8/8/8/7K w - - 0 10", "a7a8n"),
        ]:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "book.bin"
                path.write_bytes(
                    struct.pack(
                        ">QHHI", chess.polyglot.zobrist_hash(board), encoded(board, move), 1, 0
                    )
                )
                with (
                    chess.polyglot.open_reader(path) as book,
                    patch.object(repertoire, "_BOOK", book),
                ):
                    self.assertEqual(repertoire.opening_move(board), move)
                    self.assertEqual(board.fen(), fen)

    def test_no_tablebase_probe_above_four_pieces(self) -> None:
        with patch.object(repertoire, "_TABLES") as tables:
            self.assertIsNone(repertoire.endgame_move(chess.Board()))
            tables.probe_wdl.assert_not_called()

    def test_missing_table_restores_position_and_falls_back(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/6Q1/6K1 w - - 0 30")
        fen = board.fen()
        with patch.object(repertoire, "_TABLES") as tables:
            tables.probe_wdl.side_effect = KeyError("missing")
            self.assertIsNone(repertoire.endgame_move(board))
        self.assertEqual(board.fen(), fen)
        self.assertFalse(board.move_stack)

    def test_tablebase_converts_mating_material(self) -> None:
        for fen in [
            "7k/8/8/8/8/8/6Q1/6K1 w - - 0 30",
            "7k/8/8/8/8/8/6R1/6K1 w - - 0 30",
            "7k/8/8/8/8/8/5BN1/6K1 w - - 0 30",
            "8/8/4K3/4P3/4k3/8/8/8 w - - 0 30",
        ]:
            board = chess.Board(fen)
            for _ in range(200):
                if board.is_game_over(claim_draw=True):
                    break
                before = board.fen()
                move = repertoire.endgame_move(board)
                self.assertEqual(board.fen(), before)
                self.assertIsNotNone(move)
                assert move is not None
                self.assertIn(move, board.legal_moves)
                board.push(move)
            self.assertTrue(board.is_checkmate(), board.fen())
            self.assertEqual(board.turn, chess.BLACK)


if __name__ == "__main__":
    unittest.main()
