"""Original compiled chess search for Prompt and Circumstance.

Squares use sixteen-slot rows: square & 0x88 identifies the off-board padding.
Moves pack origin, destination and promotion into 7, 7 and 3 bits respectively.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import chess
import numpy as np
from numba import njit, objmode
from numpy.typing import NDArray

Array = NDArray[np.int64]


def compiled[Function: Callable[..., Any]](function: Function) -> Function:
    """Keep Python type signatures while Numba compiles the numerical implementation."""
    return cast(Function, njit(cache=False)(function))


MATE = 30_000
INF = 32_000
LIMIT = 96
SIZE = 1 << 17
VALUE = np.array([0, 100, 320, 335, 500, 950, 0], dtype=np.int64)
PHASE = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int64)
KNIGHT = (-33, -31, -18, -14, 14, 18, 31, 33)
KING = (-17, -16, -15, -1, 1, 15, 16, 17)
PST = np.zeros((7, 128, 2), dtype=np.int64)
for sq in range(128):
    if sq & 0x88:
        continue
    rank, file = divmod(sq, 16)
    centre = min(file, 7 - file) + min(rank, 7 - rank)
    PST[1, sq] = 100 + 7 * rank + 3 * min(file, 7 - file), 120 + 12 * rank
    PST[2, sq] = 285 + 12 * centre, 285 + 9 * centre
    PST[3, sq] = 315 + 6 * centre, 325 + 6 * centre
    PST[4, sq] = 500 + 3 * rank, 510 + 3 * centre
    PST[5, sq] = 940 + 3 * centre, 940 + 5 * centre
    PST[6, sq] = (25 if sq in (1, 2, 6) else 0) - 14 * centre - 8 * rank, 16 * centre
ZOBRIST = np.random.default_rng(18271).integers(1, 2**63 - 1, (14, 128), dtype=np.int64)


with np.load(Path(__file__).parent / "weights" / "evaluator.npz", allow_pickle=False) as weights:
    NN_EMBED = weights["embedding"].astype(np.float64)
    NN_OUT = weights["output"].astype(np.float64)
if NN_EMBED.shape != (769, 32) or NN_OUT.shape != (32,):
    raise ValueError("Unexpected evaluator dimensions")
if not np.isfinite(NN_EMBED).all() or not np.isfinite(NN_OUT).all():
    raise ValueError("Evaluator weights must be finite")


class Work(NamedTuple):
    """Fixed buffers shared by compiled calls; array rows are indexed by search ply.

    State: side, castling rights, en-passant square, halfmoves, kings, game ply.
    Undo: the seven state fields, captured square, captured piece, moving piece.
    Table: depth, score, bound, move, halfmoves, game ply, reversible-history hash.
    Stats: visited nodes, stop flag, completed root move, root history length.
    Neural buffers remember the last evaluated board, which can differ from the search board.
    """

    board: Array
    state: Array
    undo: Array
    moves: Array
    order: Array
    path: Array
    table_keys: Array
    table: Array
    killers: Array
    history: Array
    stats: Array
    nn_previous: Array
    nn_hidden: NDArray[np.float64]


@compiled
def residual(w: Work) -> float:
    # Cache the last evaluated board. Only changed piece features alter the sums.
    for square in range(128):
        if square & 0x88 or w.board[square] == w.nn_previous[square]:
            continue
        for old in range(2):
            piece = w.nn_previous[square] if old else w.board[square]
            if not piece:
                continue
            for view in range(2):
                side = 1 if view == 0 else -1
                relative = square if side > 0 else square ^ 112
                feature = (abs(piece) - 1) * 64 + relative // 16 * 8 + relative % 16
                if piece * side < 0:
                    feature += 384
                for neuron in range(32):
                    w.nn_hidden[view, neuron] += NN_EMBED[feature, neuron] * (-1 if old else 1)
        w.nn_previous[square] = w.board[square]
    result = 0.0
    for neuron in range(32):
        result += (max(0.0, w.nn_hidden[0, neuron]) - max(0.0, w.nn_hidden[1, neuron])) * NN_OUT[
            neuron
        ]
    return float(200.0 * np.tanh(result / 2.0) * w.state[0])


@compiled
def attacked(board: Array, square: int, colour: int) -> bool:
    for delta in (-17, -15):
        origin = square + delta * colour
        if not origin & 0x88 and board[origin] == colour:
            return True
    for delta in KNIGHT:
        origin = square + delta
        if not origin & 0x88 and board[origin] == 2 * colour:
            return True
    for delta in KING:
        origin = square + delta
        distance = 1
        while not origin & 0x88:
            piece = board[origin]
            if piece:
                if piece * colour > 0:
                    kind = abs(piece)
                    diagonal = abs(delta) in (15, 17)
                    if kind == 5 or (kind == 6 and distance == 1):
                        return True
                    if (kind == 3 and diagonal) or (kind == 4 and not diagonal):
                        return True
                break
            origin += delta
            distance += 1
    return False


@compiled
def make(board: Array, state: Array, move: int, undo: Array) -> None:
    for i in range(7):
        undo[i] = state[i]
    origin = move & 127
    target = (move >> 7) & 127
    promotion = move >> 14
    piece = board[origin]
    captured_square = target
    if abs(piece) == 1 and target == state[2] and board[target] == 0:
        captured_square = target - 16 * state[0]
    undo[7] = captured_square
    undo[8] = board[captured_square]
    undo[9] = piece
    board[captured_square] = 0
    board[origin] = 0
    board[target] = promotion * state[0] if promotion else piece
    if abs(piece) == 6:
        state[4 if piece > 0 else 5] = target
        state[1] &= 12 if piece > 0 else 3
        if abs(target - origin) == 2:
            rook = origin + 3 if target > origin else origin - 4
            board[(origin + target) // 2] = board[rook]
            board[rook] = 0
    if origin == 0 or target == 0:
        state[1] &= 13
    if origin == 7 or target == 7:
        state[1] &= 14
    if origin == 112 or target == 112:
        state[1] &= 7
    if origin == 119 or target == 119:
        state[1] &= 11
    state[2] = (origin + target) // 2 if abs(piece) == 1 and abs(target - origin) == 32 else -1
    state[3] = 0 if abs(piece) == 1 or undo[8] else state[3] + 1
    state[6] += 1
    state[0] = -state[0]


@compiled
def unmake(board: Array, state: Array, move: int, undo: Array) -> None:
    origin = move & 127
    target = (move >> 7) & 127
    board[origin] = undo[9]
    board[target] = 0
    board[undo[7]] = undo[8]
    if abs(undo[9]) == 6 and abs(target - origin) == 2:
        rook = origin + 3 if target > origin else origin - 4
        board[rook] = board[(origin + target) // 2]
        board[(origin + target) // 2] = 0
    for i in range(7):
        state[i] = undo[i]


@compiled
def append_move(moves: Array, count: int, origin: int, target: int, pawn: bool) -> int:
    if pawn and target // 16 in (0, 7):
        for promotion in (5, 4, 3, 2):
            moves[count] = origin | (target << 7) | (promotion << 14)
            count += 1
    else:
        moves[count] = origin | (target << 7)
        count += 1
    return count


@compiled
def generate(board: Array, state: Array, moves: Array, undo: Array) -> int:
    count = 0
    colour = state[0]
    for origin in range(128):
        if origin & 0x88 or board[origin] * colour <= 0:
            continue
        piece = abs(board[origin])
        if piece == 1:
            target = origin + 16 * colour
            if not target & 0x88 and board[target] == 0:
                count = append_move(moves, count, origin, target, True)
                if origin // 16 == (1 if colour > 0 else 6) and board[target + 16 * colour] == 0:
                    count = append_move(moves, count, origin, target + 16 * colour, True)
            for delta in (15, 17):
                target = origin + delta * colour
                if not target & 0x88 and (board[target] * colour < 0 or target == state[2]):
                    count = append_move(moves, count, origin, target, True)
        else:
            for i in range(8):
                delta = KNIGHT[i] if piece == 2 else KING[i]
                diagonal = abs(delta) in (15, 17)
                if (piece == 3 and not diagonal) or (piece == 4 and diagonal):
                    continue
                target = origin + delta
                while not target & 0x88:
                    occupant = board[target]
                    if occupant * colour > 0:
                        break
                    count = append_move(moves, count, origin, target, False)
                    if occupant or piece in (2, 6):
                        break
                    target += delta
            if piece == 6 and origin == (4 if colour > 0 else 116):
                rights = state[1] if colour > 0 else state[1] >> 2
                if not attacked(board, origin, -colour):
                    if (
                        rights & 1
                        and board[origin + 1] == 0
                        and board[origin + 2] == 0
                        and board[origin + 3] == 4 * colour
                        and not attacked(board, origin + 1, -colour)
                    ):
                        count = append_move(moves, count, origin, origin + 2, False)
                    if (
                        rights & 2
                        and board[origin - 1] == 0
                        and board[origin - 2] == 0
                        and board[origin - 3] == 0
                        and board[origin - 4] == 4 * colour
                        and not attacked(board, origin - 1, -colour)
                    ):
                        count = append_move(moves, count, origin, origin - 2, False)
    legal = 0
    for i in range(count):
        move = moves[i]
        make(board, state, move, undo)
        valid = not attacked(board, state[4 if colour > 0 else 5], -colour)
        unmake(board, state, move, undo)
        if valid:
            moves[legal] = move
            legal += 1
    return legal


@compiled
def position_hash(board: Array, state: Array) -> int:
    # Only legally capturable en-passant squares distinguish repetition positions.
    key = np.int64(0)
    for square in range(128):
        if not square & 0x88 and board[square]:
            key ^= ZOBRIST[board[square] + 6, square]
    key ^= ZOBRIST[13, state[1]]
    if state[0] < 0:
        key ^= ZOBRIST[13, 16]
    ep = state[2]
    if ep >= 0:
        colour = state[0]
        for delta in (15, 17):
            origin = ep - delta * colour
            if not origin & 0x88 and board[origin] == colour:
                board[origin] = 0
                board[ep] = colour
                board[ep - 16 * colour] = 0
                legal = not attacked(board, state[4 if colour > 0 else 5], -colour)
                board[origin] = colour
                board[ep] = 0
                board[ep - 16 * colour] = -colour
                if legal:
                    key ^= ZOBRIST[13, 32 + ep % 16]
                    break
    return int(key)


@compiled
def insufficient(board: Array) -> bool:
    minors = knights = 0
    bishop_colour = -1
    mixed = False
    for square in range(128):
        if square & 0x88:
            continue
        piece = abs(board[square])
        if piece in (1, 4, 5):
            return False
        if piece in (2, 3):
            minors += 1
            if piece == 2:
                knights += 1
            else:
                colour = (square // 16 + square % 16) & 1
                if bishop_colour >= 0 and bishop_colour != colour:
                    mixed = True
                bishop_colour = colour
    return minors <= 1 or (knights == 0 and not mixed)


@compiled
def evaluate(board: Array, state: Array) -> int:
    mg = eg = phase = 0
    counts = np.zeros((2, 8), dtype=np.int64)
    low = np.full((2, 8), 8, dtype=np.int64)
    high = np.full((2, 8), -1, dtype=np.int64)
    bishops = np.zeros(2, dtype=np.int64)
    for square in range(128):
        if square & 0x88 or not board[square]:
            continue
        piece = board[square]
        kind = abs(piece)
        colour = 1 if piece > 0 else 0
        sign = 1 if piece > 0 else -1
        relative = square if piece > 0 else square ^ 112
        mg += sign * PST[kind, relative, 0]
        eg += sign * PST[kind, relative, 1]
        phase += PHASE[kind]
        if kind == 1:
            rank, file = square // 16, square % 16
            counts[colour, file] += 1
            low[colour, file] = min(low[colour, file], rank)
            high[colour, file] = max(high[colour, file], rank)
        if kind == 3:
            bishops[colour] += 1
    for square in range(128):
        if square & 0x88 or not board[square]:
            continue
        piece = board[square]
        kind = abs(piece)
        colour = 1 if piece > 0 else 0
        sign = 1 if piece > 0 else -1
        file = square % 16
        rank = square // 16
        if kind == 1:
            neighbours = (counts[colour, file - 1] if file else 0) + (
                counts[colour, file + 1] if file < 7 else 0
            )
            if neighbours == 0:
                mg -= sign * 12
                eg -= sign * 10
            passed = True
            for f in range(max(0, file - 1), min(8, file + 2)):
                if (sign > 0 and high[1 - colour, f] > rank) or (
                    sign < 0 and low[1 - colour, f] < rank
                ):
                    passed = False
            if passed:
                advance = rank if sign > 0 else 7 - rank
                mg += sign * advance * advance * 2
                eg += sign * advance * advance * 5
        if kind == 4 and counts[colour, file] == 0:
            mg += sign * (12 if counts[1 - colour, file] else 22)
    for colour in range(2):
        sign = 1 if colour else -1
        for file in range(8):
            extra = max(0, counts[colour, file] - 1)
            mg -= sign * 10 * extra
            eg -= sign * 15 * extra
        if bishops[colour] >= 2:
            mg += sign * 25
            eg += sign * 40
        king = state[4 if colour else 5]
        for delta in (15, 16, 17):
            target = king + sign * delta
            if not target & 0x88 and board[target] == sign:
                mg += sign * 12
        enemy_king = state[5 if colour else 4]
        enemy_count = 0
        for square in range(128):
            if not square & 0x88 and board[square] * sign < 0:
                enemy_count += 1
        if enemy_count == 1:
            distance = max(abs(king // 16 - enemy_king // 16), abs(king % 16 - enemy_king % 16))
            eg += sign * (14 - distance) * 5
    phase = min(24, phase)
    total = mg * phase + eg * (24 - phase)
    score = abs(total) // 24 * (1 if total >= 0 else -1)
    return int(score * state[0] + 8)


@compiled
def search(w: Work, depth: int, alpha: int, beta: int, ply: int, deadline: float) -> int:
    w.stats[0] += 1
    # Re-enter Python only for the clock; move generation and recursion stay compiled.
    if w.stats[0] & 255 == 0:
        with objmode(now="float64"):
            now = time.monotonic()
        if now >= deadline:
            w.stats[1] = 1
    if w.stats[1]:
        return 0
    board, state = w.board, w.state
    count = generate(board, state, w.moves[ply], w.undo[ply])
    in_check = attacked(board, state[4 if state[0] > 0 else 5], -state[0])
    if count == 0:
        return -MATE + ply if in_check else 0
    key = position_hash(board, state)
    index = w.stats[3] - 1 + ply
    w.path[index] = key
    repeats = 1
    start = max(0, index - state[3])
    for past in range(index - 2, start - 1, -2):
        if w.path[past] == key:
            repeats += 1
    if repeats >= 3 or state[3] >= 100 or state[6] >= 600 or insufficient(board):
        return 0
    if ply >= LIMIT - 1:
        w.stats[1] = 1
        return 0
    # Ordered reversible history is deliberately conservative for cached score reuse.
    context = np.int64(0)
    for past in range(start, index + 1):
        context = (context ^ w.path[past]) * np.int64(6364136223846793005)
    slot = key & (SIZE - 1)
    hint = 0
    if w.table_keys[slot] == key:
        hint = w.table[slot, 3]
        if (
            depth > 0
            and w.table[slot, 0] >= depth
            and w.table[slot, 4] == state[3]
            and w.table[slot, 5] == state[6]
            and w.table[slot, 6] == context
        ):
            cached = w.table[slot, 1]
            if cached >= MATE - 256:
                cached -= ply
            elif cached <= -MATE + 256:
                cached += ply
            bound = w.table[slot, 2]
            if bound == 0 or (bound == 1 and cached >= beta) or (bound == 2 and cached <= alpha):
                if ply == 0:
                    w.stats[2] = hint
                return int(cached)
    original = alpha
    best = -INF
    if depth <= 0 and not in_check:
        best = int(evaluate(board, state) + residual(w))
        if best >= beta:
            return best
        alpha = max(alpha, best)
    for i in range(count):
        move = w.moves[ply, i]
        origin, target = move & 127, (move >> 7) & 127
        promotion = move >> 14
        captured = abs(board[target])
        if abs(board[origin]) == 1 and target == state[2]:
            captured = 1
        priority = w.history[0 if state[0] > 0 else 1, origin, target]
        if captured or promotion:
            priority = 1000000 + 16 * VALUE[captured] - VALUE[abs(board[origin])] + VALUE[promotion]
        elif move == w.killers[ply, 0] or move == w.killers[ply, 1]:
            priority = 900000
        if move == hint:
            priority = 2000000
        w.order[ply, i] = priority
    best_move = w.moves[ply, 0]
    for i in range(count):
        selected = i
        for j in range(i + 1, count):
            if w.order[ply, j] > w.order[ply, selected]:
                selected = j
        move = w.moves[ply, selected]
        w.moves[ply, selected] = w.moves[ply, i]
        w.moves[ply, i] = move
        w.order[ply, selected] = w.order[ply, i]
        origin, target = move & 127, (move >> 7) & 127
        promotion = move >> 14
        capture = board[target] != 0 or (abs(board[origin]) == 1 and target == state[2])
        if depth <= 0 and not in_check and not capture and not promotion:
            continue
        make(board, state, move, w.undo[ply])
        if i == 0 or depth <= 0:
            score = -search(w, depth - 1, -beta, -alpha, ply + 1, deadline)
        else:
            score = -search(w, depth - 1, -alpha - 1, -alpha, ply + 1, deadline)
            if alpha < score < beta and not w.stats[1]:
                score = -search(w, depth - 1, -beta, -alpha, ply + 1, deadline)
        unmake(board, state, move, w.undo[ply])
        if w.stats[1]:
            return 0
        if score > best:
            best, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            if not capture and not promotion and depth > 0:
                if w.killers[ply, 0] != move:
                    w.killers[ply, 1] = w.killers[ply, 0]
                    w.killers[ply, 0] = move
                c = 0 if state[0] > 0 else 1
                w.history[c, origin, target] = min(
                    800000, w.history[c, origin, target] + depth * depth
                )
            break
    if depth > 0:
        w.table_keys[slot] = key
        w.table[slot, 0] = depth
        w.table[slot, 1] = (
            best + ply if best >= MATE - 256 else best - ply if best <= -MATE + 256 else best
        )
        w.table[slot, 2] = 2 if best <= original else 1 if best >= beta else 0
        w.table[slot, 3] = best_move
        w.table[slot, 4] = state[3]
        w.table[slot, 5] = state[6]
        w.table[slot, 6] = context
        if ply == 0:
            w.stats[2] = best_move
    return best


def encode(position: chess.Board) -> tuple[Array, Array]:
    board = np.zeros(128, dtype=np.int64)
    for square, piece in position.piece_map().items():
        board[(square // 8) * 16 + square % 8] = piece.piece_type * (1 if piece.color else -1)
    rights = (
        int(position.has_kingside_castling_rights(chess.WHITE))
        | int(position.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(position.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(position.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    kings = [position.king(c) for c in (chess.WHITE, chess.BLACK)]
    if kings[0] is None or kings[1] is None:
        raise ValueError("Both kings must be present")
    state = np.array(
        [
            1 if position.turn else -1,
            rights,
            -1
            if position.ep_square is None
            else position.ep_square // 8 * 16 + position.ep_square % 8,
            position.halfmove_clock,
            kings[0] // 8 * 16 + kings[0] % 8,
            kings[1] // 8 * 16 + kings[1] % 8,
            position.ply(),
        ],
        dtype=np.int64,
    )
    return board, state


def workspace(position: chess.Board) -> Work:
    board, state = encode(position)
    return Work(
        board,
        state,
        np.zeros((LIMIT, 10), dtype=np.int64),
        np.zeros((LIMIT, 512), dtype=np.int64),
        np.zeros((LIMIT, 512), dtype=np.int64),
        np.zeros(768, dtype=np.int64),
        np.zeros(SIZE, dtype=np.int64),
        np.zeros((SIZE, 7), dtype=np.int64),
        np.zeros((LIMIT, 2), dtype=np.int64),
        np.zeros((2, 128, 128), dtype=np.int64),
        np.array([0, 0, 0, 1], dtype=np.int64),
        np.zeros(128, dtype=np.int64),
        np.zeros((2, 32), dtype=np.float64),
    )


def decode(move: int) -> chess.Move:
    origin, target = move & 127, (move >> 7) & 127
    return chess.Move(
        origin // 16 * 8 + origin % 16,
        target // 16 * 8 + target % 16,
        promotion=(move >> 14) or None,
    )


class Engine:
    """Keep game history in Python and run the complete search inside Numba."""

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.work = workspace(chess.Board())

    def get_move(self, fen: str, time_left_ms: int) -> str:
        start = time.monotonic()
        incoming = chess.Board(fen)
        legal = list(incoming.legal_moves)
        if not legal:
            raise ValueError("get_move requires a position with a legal move")
        best = legal[0]
        remaining = max(0, time_left_ms / 1000)
        allocation = min(remaining / 35 + 0.2, remaining * 0.08)
        deadline = start + max(0, min(allocation * 2, remaining - min(0.05, remaining * 0.2)))
        board = incoming
        if self.board is not None and incoming.ply() == self.board.ply() + 1:
            for move in self.board.legal_moves:
                if time.monotonic() >= min(deadline, start + allocation * 0.1):
                    break
                self.board.push(move)
                if self.board.fen() == incoming.fen():
                    board = self.board
                    break
                self.board.pop()
        self.work.board[:], self.work.state[:] = encode(board)
        w = self.work
        history = []
        previous = board.copy()
        while True:
            b, s = encode(previous)
            history.append(position_hash(b, s))
            if not previous.move_stack or previous.halfmove_clock == 0:
                break
            previous.pop()
        history.reverse()
        w.path[: len(history)] = history
        w.stats[:] = (0, 0, 0, len(history))
        w.history.fill(0)
        w.killers.fill(0)
        completed = score = 0
        if len(legal) > 1 and time_left_ms > 20:
            for depth in range(1, LIMIT):
                if time.monotonic() >= start + allocation:
                    break
                candidate_score = search(w, depth, -INF, INF, 0, deadline)
                if w.stats[1]:
                    break
                move = decode(int(w.stats[2]))
                if move in legal:
                    best, score, completed = move, candidate_score, depth
                if abs(score) >= MATE - 256:
                    break
        board.push(best)
        self.board = board
        print(
            f"depth={completed} nodes={w.stats[0]} score={score} "
            f"ms={(time.monotonic() - start) * 1000:.0f}"
        )
        return best.uci()


# Compile the same array and scalar signatures used during play before the clock starts.
_started = time.monotonic()
_ENGINE = Engine()
# Exercise the Python clock callback during initialisation too.
_ENGINE.work.stats[0] = 255
search(_ENGINE.work, 2, -INF, INF, 0, time.monotonic() + 60)
print(f"compiled in {time.monotonic() - _started:.2f}s")


def get_move(fen: str, time_left_ms: int) -> str:
    return _ENGINE.get_move(fen, time_left_ms)
