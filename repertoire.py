"""Local opening and endgame lookups, bounded by the competition rules."""

from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy

_ASSETS = Path(__file__).parent / "weights"
_BOOK = (
    chess.polyglot.open_reader(_ASSETS / "openings.bin")
    if (_ASSETS / "openings.bin").exists()
    else None
)
_TABLES = (
    chess.syzygy.open_tablebase(str(_ASSETS / "syzygy")) if (_ASSETS / "syzygy").is_dir() else None
)


def opening_move(board: chess.Board) -> chess.Move | None:
    """Never consult the opening table beyond the supplied FEN's move 20."""
    if _BOOK is None or board.fullmove_number > 20 or board.halfmove_clock >= 80:
        return None
    for entry in _BOOK.find_all(board):
        board.push(entry.move)
        repeats = board.is_repetition(2)
        board.pop()
        if not repeats:
            return entry.move
    return None


def endgame_move(board: chess.Board) -> chess.Move | None:
    """Prefer a preserved win with progress towards a capture, pawn move or mate.

    Missing coverage leaves the position to search. Near the fifty-move boundary,
    allow a spare ply for Syzygy's rounded distance. History draws take precedence.
    """
    if _TABLES is None or len(board.piece_map()) > 4 or board.castling_rights:
        return None
    ranked: list[tuple[int, int, str, chess.Move]] = []
    for move in list(board.legal_moves):
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            if board.is_checkmate():
                return move
            if (
                board.is_game_over()
                or board.is_repetition(3)
                or board.halfmove_clock >= 100
                or board.ply() >= 600
            ):
                value, distance = 0, 0
            else:
                value = -_TABLES.probe_wdl(board)
                dtz = -_TABLES.probe_dtz(board)
                distance = 1 if zeroing else abs(dtz) + 1
                if abs(value) == 1:
                    value = 0
                elif value and not zeroing and abs(dtz) + board.halfmove_clock + 1 > 100:
                    # Rounded distances at advanced clocks are not draw proofs.
                    # Leave this boundary to search rather than misrank a loss.
                    return None
                # Prefer quick progress when winning; delay it when losing.
                distance = -distance if value > 0 else distance if value < 0 else 0
            ranked.append((value, distance, move.uci(), move))
        except KeyError:
            return None
        finally:
            board.pop()
    return max(ranked, key=lambda row: row[:3])[3] if ranked else None
