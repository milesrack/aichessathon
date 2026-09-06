"""Prompt and Circumstance: a time-bounded classical chess search."""

import time
from dataclasses import dataclass

import chess

MATE = 30_000
MATE_THRESHOLD = MATE - 256
INFINITY = 32_000
MAX_PLY = 96
TABLE_SIZE = 8192
EXACT, LOWER, UPPER = 0, 1, 2
MATERIAL = (0, 100, 320, 335, 500, 950, 0)
PHASE = (0, 0, 1, 1, 2, 4, 0)

# Position identity excludes move counters and unusable en-passant targets.
type Position = tuple[int, int, int, int, int, int, int, int, bool, int, int | None]
type History = frozenset[tuple[Position, int]]


def position_key(board: chess.Board) -> Position:
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.castling_rights,
        board.ep_square if board.has_legal_en_passant() else None,
    )


def _piece_square(piece: int, square: int) -> tuple[int, int]:
    rank, file = divmod(square, 8)
    centre = min(file, 7 - file) + min(rank, 7 - rank)
    if piece == chess.PAWN:
        return 100 + 7 * rank + 3 * min(file, 7 - file), 120 + 12 * rank
    if piece == chess.KNIGHT:
        return 285 + 12 * centre, 285 + 9 * centre
    if piece == chess.BISHOP:
        return 315 + 6 * centre, 325 + 6 * centre
    if piece == chess.ROOK:
        return 500 + 3 * rank, 510 + 3 * centre
    if piece == chess.QUEEN:
        return 940 + 3 * centre, 940 + 5 * centre
    shelter = 25 if square in (chess.B1, chess.C1, chess.G1) else 0
    return shelter - 14 * centre - 8 * rank, 16 * centre


PST = tuple(
    tuple(_piece_square(piece, square) for square in chess.SQUARES) for piece in chess.PIECE_TYPES
)
PASSED_MASKS = tuple(
    tuple(
        sum(
            chess.BB_SQUARES[target]
            for target in chess.SQUARES
            if abs(chess.square_file(target) - chess.square_file(square)) <= 1
            and (
                chess.square_rank(target) > chess.square_rank(square)
                if colour
                else chess.square_rank(target) < chess.square_rank(square)
            )
        )
        for square in chess.SQUARES
    )
    for colour in (chess.BLACK, chess.WHITE)
)


def evaluate(board: chess.Board) -> int:
    """Return a tapered centipawn score from the side-to-move's perspective."""
    middlegame = endgame = phase = 0
    for colour in (chess.WHITE, chess.BLACK):
        sign = 1 if colour else -1
        ours = board.occupied_co[colour]
        pawns = board.pawns & ours
        enemy_pawns = board.pawns & board.occupied_co[not colour]
        for piece in chess.PIECE_TYPES:
            squares = board.pieces_mask(piece, colour)
            phase += PHASE[piece] * squares.bit_count()
            for square in chess.scan_forward(squares):
                relative = square if colour else square ^ 56
                mg, eg = PST[piece - 1][relative]
                middlegame += sign * mg
                endgame += sign * eg
                if piece == chess.PAWN:
                    file = chess.square_file(square)
                    neighbours = chess.BB_FILES[file - 1] if file else 0
                    neighbours |= chess.BB_FILES[file + 1] if file < 7 else 0
                    if not pawns & neighbours:
                        middlegame -= sign * 12
                        endgame -= sign * 10
                    if not enemy_pawns & PASSED_MASKS[colour][square]:
                        advance = chess.square_rank(relative)
                        middlegame += sign * advance * advance * 2
                        endgame += sign * advance * advance * 5
                elif piece == chess.ROOK and not pawns & chess.BB_FILES[square & 7]:
                    middlegame += sign * (12 if enemy_pawns & chess.BB_FILES[square & 7] else 22)
        for file_mask in chess.BB_FILES:
            doubled = max(0, (pawns & file_mask).bit_count() - 1)
            middlegame -= sign * 10 * doubled
            endgame -= sign * 15 * doubled
        if (board.bishops & ours).bit_count() >= 2:
            middlegame += sign * 25
            endgame += sign * 40
        king = board.king(colour)
        enemy_king = board.king(not colour)
        if king is not None and enemy_king is not None:
            shield_rank = chess.square_rank(king) + (1 if colour else -1)
            if 0 <= shield_rank < 8:
                shield = chess.BB_KING_ATTACKS[king] & chess.BB_RANKS[shield_rank] & pawns
                middlegame += sign * 12 * shield.bit_count()
            if board.occupied_co[not colour] == chess.BB_SQUARES[enemy_king]:
                # King proximity helps convert a material advantage into a mating net.
                endgame += sign * (14 - chess.square_distance(king, enemy_king)) * 5
    phase = min(24, phase)
    total = middlegame * phase + endgame * (24 - phase)
    score = (abs(total) // 24) * (1 if total >= 0 else -1)
    return (score if board.turn else -score) + 8


@dataclass(slots=True)
class Entry:
    position: Position
    depth: int
    score: int
    bound: int
    move: chess.Move
    halfmoves: int
    game_ply: int
    history: History | None


def _store_score(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _load_score(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


class SearchStopped(Exception):
    pass


class Search:
    def __init__(self, board: chess.Board, table: list[Entry | None], deadline: float) -> None:
        self.board = board
        self.table = table
        self.deadline = deadline
        self.nodes = 0
        self.killers: list[list[chess.Move]] = [[] for _ in range(MAX_PLY)]
        self.quiet_history: dict[tuple[bool, int, int], int] = {}
        self.repetitions: dict[Position, int] = {}
        previous = board.copy()
        while True:
            key = position_key(previous)
            self.repetitions[key] = self.repetitions.get(key, 0) + 1
            if not previous.move_stack or previous.halfmove_clock == 0:
                break
            previous.pop()

    def check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            raise SearchStopped

    def _visit(self) -> None:
        self.nodes += 1
        # Check every node, including quiescence, rather than estimating time from node rate.
        self.check_time()

    def _draw(self, key: Position) -> bool:
        board = self.board
        return (
            board.halfmove_clock >= 100
            or board.ply() >= 600
            or self.repetitions.get(key, 0) >= 3
            or board.is_insufficient_material()
        )

    def _push(self, move: chess.Move) -> dict[Position, int] | None:
        old_rights = self.board.castling_rights
        self.board.push(move)
        previous = None
        if self.board.halfmove_clock == 0 or self.board.castling_rights != old_rights:
            previous = self.repetitions
            self.repetitions = {}
        key = position_key(self.board)
        self.repetitions[key] = self.repetitions.get(key, 0) + 1
        return previous

    def _pop(self, previous: dict[Position, int] | None) -> None:
        if previous is not None:
            self.repetitions = previous
        else:
            key = position_key(self.board)
            count = self.repetitions[key] - 1
            if count:
                self.repetitions[key] = count
            else:
                del self.repetitions[key]
        self.board.pop()

    def _ordered(
        self, moves: list[chess.Move], hint: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        board = self.board

        def priority(move: chess.Move) -> int:
            if move == hint:
                return 2_000_000
            victim = board.piece_type_at(move.to_square)
            if victim is not None or board.is_en_passant(move) or move.promotion:
                gain = MATERIAL[victim or chess.PAWN]
                attacker = MATERIAL[board.piece_type_at(move.from_square) or chess.PAWN]
                return 1_000_000 + 16 * gain - attacker + MATERIAL[move.promotion or 0]
            if move in self.killers[ply]:
                return 900_000
            return self.quiet_history.get((board.turn, move.from_square, move.to_square), 0)

        return sorted(moves, key=priority, reverse=True)

    def quiescence(self, alpha: int, beta: int, ply: int) -> int:
        self._visit()
        board = self.board
        in_check = board.is_check()
        # Stalemate and mate take precedence over draw counters and static evaluation.
        if not any(board.generate_legal_moves()):
            return -MATE + ply if in_check else 0
        if self._draw(position_key(board)):
            return 0
        if ply >= MAX_PLY - 1:
            # Abandon this iteration; do not manufacture a static score while in check.
            raise SearchStopped
        if in_check:
            moves = list(board.legal_moves)
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = list(board.generate_legal_captures())
            # Quiet promotions can decide an ending and are not legal captures.
            promotion_pawns = board.pawns & board.occupied_co[board.turn]
            promotion_pawns &= chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
            if promotion_pawns:
                moves.extend(
                    move
                    for move in board.generate_legal_moves(from_mask=promotion_pawns)
                    if move.promotion and not board.is_capture(move)
                )
        for move in self._ordered(moves, None, ply):
            previous = self._push(move)
            try:
                score = -self.quiescence(-beta, -alpha, ply + 1)
            finally:
                self._pop(previous)
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def search(self, depth: int, alpha: int, beta: int, ply: int = 0) -> int:
        if depth <= 0:
            return self.quiescence(alpha, beta, ply)
        self._visit()
        board = self.board
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if board.is_check() else 0
        key = position_key(board)
        if self._draw(key):
            return 0
        if ply >= MAX_PLY - 1:
            raise SearchStopped
        slot = hash(key) & (TABLE_SIZE - 1)
        entry = self.table[slot]
        hint = None
        if entry is not None and entry.position == key:
            hint = entry.move
            # Full repetition context avoids reusing a bound from a different draw history.
            if (
                entry.depth >= depth
                and entry.halfmoves == board.halfmove_clock
                and entry.game_ply == board.ply()
                and entry.history is not None
                and entry.history == frozenset(self.repetitions.items())
            ):
                cached = _load_score(entry.score, ply)
                if (
                    entry.bound == EXACT
                    or (entry.bound == LOWER and cached >= beta)
                    or (entry.bound == UPPER and cached <= alpha)
                ):
                    return cached
        original_alpha = alpha
        best_score = -INFINITY
        best_move = moves[0]
        for index, move in enumerate(self._ordered(moves, hint, ply)):
            quiet = not board.is_capture(move) and move.promotion is None
            previous = self._push(move)
            try:
                if index == 0:
                    score = -self.search(depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.search(depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.search(depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._pop(previous)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                if quiet:
                    killers = self.killers[ply]
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                    history_key = (board.turn, move.from_square, move.to_square)
                    old = self.quiet_history.get(history_key, 0)
                    self.quiet_history[history_key] = min(800_000, old + depth * depth)
                break
        bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
        # Keep long-history nodes as ordering hints without allocating unbounded context sets.
        context = frozenset(self.repetitions.items()) if len(self.repetitions) <= 16 else None
        self.table[slot] = Entry(
            key,
            depth,
            _store_score(best_score, ply),
            bound,
            best_move,
            board.halfmove_clock,
            board.ply(),
            context,
        )
        return best_score


class Engine:
    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.table: list[Entry | None] = [None] * TABLE_SIZE

    def _synchronise(self, incoming: chess.Board, deadline: float) -> chess.Board:
        previous = self.board
        if previous is not None and incoming.ply() == previous.ply() + 1:
            for move in previous.legal_moves:
                if time.monotonic() >= deadline:
                    break
                previous.push(move)
                if previous.fen() == incoming.fen():
                    return previous
                previous.pop()
        # The contract starts a new process per game; this also supports unrelated local FENs.
        self.table = [None] * TABLE_SIZE
        return incoming

    def get_move(self, fen: str, time_left_ms: int) -> str:
        started = time.monotonic()
        incoming = chess.Board(fen)
        legal = list(incoming.legal_moves)
        if not legal:
            raise ValueError("get_move requires a position with a legal move")
        best = legal[0]
        remaining = max(0.0, time_left_ms / 1000.0)
        reserve = min(0.05, remaining * 0.2)
        allocation = min(remaining / 35 + 0.2, remaining * 0.08)
        deadline = started + max(0.0, min(allocation * 2, remaining - reserve))
        board = self._synchronise(incoming, min(deadline, started + allocation * 0.1))
        completed = nodes = 0
        score = 0
        if len(legal) > 1 and time_left_ms > 20:
            search = Search(board, self.table, deadline)
            for depth in range(1, MAX_PLY):
                if time.monotonic() >= started + allocation:
                    break
                try:
                    candidate_score = search.search(depth, -INFINITY, INFINITY)
                except SearchStopped:
                    break
                entry = self.table[hash(position_key(board)) & (TABLE_SIZE - 1)]
                if (
                    entry is not None
                    and entry.position == position_key(board)
                    and entry.move in legal
                ):
                    best, score, completed = entry.move, candidate_score, depth
                if abs(score) >= MATE_THRESHOLD:
                    break
            nodes = search.nodes
        board.push(best)
        self.board = board
        print(
            f"depth={completed} nodes={nodes} score={score} "
            f"ms={(time.monotonic() - started) * 1000:.0f}"
        )
        return best.uci()


_ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move within the remaining wall-clock budget."""
    return _ENGINE.get_move(fen, time_left_ms)
