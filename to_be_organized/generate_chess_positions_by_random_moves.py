from cchess import Board
import random

def generate_random_fen_and_moves(max_steps=100, max_capture=0):
    capture_count = 0
    board = Board()
    for i in range(max_steps):
        print(i)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        if board.is_capture(move):
            capture_count+=1
        if capture_count>max_capture:
            while True:
                move = random.choice(legal_moves)
                if not board.is_capture(move):
                    break
        board.push(move)

    fen = board.fen()
    legal_uci_moves = [m.uci() for m in board.legal_moves]
    return fen, legal_uci_moves

# 示例调用
fen, legal_moves = generate_random_fen_and_moves()
print("FEN:", fen)
print("前10个合法走法:", legal_moves)