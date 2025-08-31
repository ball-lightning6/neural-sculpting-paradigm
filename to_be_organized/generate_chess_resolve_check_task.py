import random

# 所有棋子的最大数目
piece_pool = {
    'r': 2, 'n': 2, 'b': 2, 'a': 2, 'k': 1, 'c': 2, 'p': 5,
    'R': 2, 'N': 2, 'B': 2, 'A': 2, 'K': 1, 'C': 2, 'P': 5,
}

# 限制位置定义
restricted_positions = {
    'A': [(9,3), (9,5), (8,4), (7,3), (7,5)],
    'a': [(0,3), (0,5), (1,4), (2,3), (2,5)],
    'B': [(9,2), (9,6), (7,0), (7,4), (7,8), (5,2), (5,6)],
    'b': [(0,2), (0,6), (2,0), (2,4), (2,8), (4,2), (4,6)],
    'K': [(r,c) for r in range(7,10) for c in range(3,6)],
    'k': [(r,c) for r in range(0,3) for c in range(3,6)],
    'P': [(6,0), (6,2), (6,4), (6,6) , (6,8),(5,0), (5,2), (5,4), (5,6) , (5,8)] +[(r,c) for r in range(5) for c in range(9)],
    'p': [(3,0), (3,2), (3,4), (3,6) , (3,8),(4,0), (4,2), (4,4), (4,6) , (4,8)] +[(r,c) for r in range(5,10) for c in range(9)]
}

# 随机棋子数量
def random_piece_counts():
    counts = {}
    for k, v in piece_pool.items():
        if k == 'k' or k == 'K':
            counts[k] = 1  # 强制每方必须有帅/将
        else:
            counts[k] = random.randint(0, v)
    return counts

# 棋盘放置
def place_pieces_with_restriction(piece_counts):
    board = [['.' for _ in range(9)] for _ in range(10)]
    used_positions = set()

    # 先放有限制位置的棋子
    for piece in ['a','A','b','B','k','K','p','P']:
        count = piece_counts.get(piece, 0)
        legal_pos = list(set(restricted_positions[piece]) - used_positions)
        if len(legal_pos) < count:
            return None  # 无法放置，标记非法
        chosen = random.sample(legal_pos, count)
        for r, c in chosen:
            board[r][c] = piece
            used_positions.add((r,c))
        piece_counts[piece] = 0  # 已放置完

    # 剩余棋子随机放在未占用格子
    remaining = [(r,c) for r in range(10) for c in range(9) if (r,c) not in used_positions]
    random.shuffle(remaining)

    for piece, count in piece_counts.items():
        for _ in range(count):
            if not remaining:
                return None
            r,c = remaining.pop()
            board[r][c] = piece
    return board

# 将帅照面检测
def is_general_face_to_face(board):
    red, black = None, None
    for r in range(10):
        for c in range(9):
            if board[r][c] == 'k':
                red = (r,c)
            elif board[r][c] == 'K':
                black = (r,c)
    if red and black and red[1] == black[1]:
        for r in range(min(red[0], black[0]) + 1, max(red[0], black[0])):
            if board[r][red[1]] != '.':
                return False
        return True
    return False

# 转 FEN 表达式
def board_to_fen(board):
    fen_rows = []
    for row in board:
        fen_row = ''
        empty = 0
        for cell in row:
            if cell == '.':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)
from render_cchess_fen import render_chinese_chess_board
# 总生成函数
def generate_restricted_legal_fen():
    for _ in range(100):  # 最多尝试100次
        piece_counts = random_piece_counts()
        board = place_pieces_with_restriction(piece_counts)
        if board is None:
            continue
        if is_general_face_to_face(board):
            continue
        fen = board_to_fen(board)+' '+random.choice(['w', 'b'])
        #print(fen)
        return fen
    return None
import json
with open("move2idx.json", "r") as f:
    move2idx = json.load(f)

import time
s=time.time()
t=0
fen_set=set()
from cchess import Board
def is_check_but_not_checkmate(board):
    return board.is_check() and not board.is_checkmate()

with open('dataset_1048_jiejiang.txt','w') as f, open('legal_moves_dataset_1048_jiejiang.jsonl','w') as g:
    while t<1000000:
        if t%50000==0:
            print(t,time.time() - s)

        fen=generate_restricted_legal_fen()
        if fen is not None:
            if fen not in fen_set:

                b=Board(fen)
                if is_check_but_not_checkmate(b):
                    print(fen)
                    fen_set.add(fen)
                    # import sys
                    # sys.exit()
                    t+=1
                    f.write(fen+'\n')
                    legal_moves = list(b.legal_moves)

                    move_ids = []
                    for move in legal_moves:
                        move_str = move.uci()  # e.g., "a0a2"
                        if move_str in move2idx:
                            #print(move_str)
                            move_ids.append(move2idx[move_str])
                        else:
                            # 可选：记录找不到的 move
                            print(move_str)

                    g.write(json.dumps({
                        "fen": fen,
                        "legal_move_ids": move_ids
                    }) + "\n")


# print(3,5)

print(time.time()-s)

