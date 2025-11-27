#2bak2n1/4a4/2n1b1c2/p1p1p1p1p/3N5/2P3P2/c3P3P/2C2C3/9/1NBAKAB2 b - - 3 13

from cchess import Board

# # 输入你的FEN字符串
# fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
#
# # 初始化棋盘
# board = Board(fen)
#
# # 生成所有合法走法
# legal_moves = list(board.legal_moves)
#
# # 输出所有走法
# print("合法走法总数：", len(legal_moves))
# for move in legal_moves:
#     print(move.uci())  # 输出标准走法字符串，如 e2e3

# import sys
# sys.exit()

import json
from cchess import Board
from tqdm import tqdm

# === 加载 move2idx 映射 ===
with open("move2idx.json", "r") as f:
    move2idx = json.load(f)

# === 输入输出路径 ===
fen_file = "dataset_big.txt"
fen_file = "dataset_1048_5m.txt"#dataset_1048_test.txt

output_file = "legal_moves_dataset_1048_3m_train.jsonl"

# === 处理每一行 FEN ===
with open(fen_file, "r") as fens, open(output_file, "w") as out:
    d = fens.readlines()[:3000000]
    t=0
    for line in tqdm(d, desc="Processing FENs"):
        t+=1
        # print('generating samples: '+ str(t))
        fen = line.strip()
        board = Board(fen)
        legal_moves = list(board.legal_moves)

        move_ids = []
        for move in legal_moves:
            move_str = move.uci()  # e.g., "a0a2"
            if move_str in move2idx:
                move_ids.append(move2idx[move_str])
            else:
                # 可选：记录找不到的 move
                print(move_str)

        out.write(json.dumps({
            "fen": fen,
            "legal_move_ids": move_ids
        }) + "\n")
