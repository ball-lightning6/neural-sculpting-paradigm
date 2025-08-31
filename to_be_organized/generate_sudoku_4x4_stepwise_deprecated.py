import random
import json
import copy

def is_valid(board, row, col, num):
    # 判断 num 是否能合法填入 (row, col)
    for i in range(4):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 2 * (row // 2), 2 * (col // 2)
    for i in range(start_row, start_row + 2):
        for j in range(start_col, start_col + 2):
            if board[i][j] == num:
                return False
    return True

def get_candidates(board, row, col):
    return [n for n in range(1, 5) if is_valid(board, row, col, n)]

def encode_board(board):
    # 将4x4盘面编码为长度16的01字符串列表，每格用4位 one-hot 表示数字1-4或0
    result = []
    for num in sum(board, []):  # flatten board
        vec = [0]*5
        vec[num] = 1
        result.extend(vec)  # 0,1,2,3,4 -> one-hot
    return result

def generate_single_step_data(board, row, col):
    candidates = get_candidates(board, row, col)
    encoded_board = encode_board(board)
    pos_encoding = [0]*16
    pos_encoding[row*4 + col] = 1
    input_vector = encoded_board + pos_encoding
    if len(candidates) == 1:
        return {"input": input_vector, "output": candidates[0]}
    else:
        return {"input": input_vector, "output": 0}

def generate_training_data_from_full_board(full_board):
    data = []
    board = [[0]*4 for _ in range(4)]
    filled = [[False]*4 for _ in range(4)]
    steps = []

    # 将完整数独逐步“倒推”填入，确保每一步可解释（唯一解）
    for _ in range(16):
        for i in range(4):
            for j in range(4):
                if filled[i][j]: continue
                temp_board = copy.deepcopy(board)
                temp_board[i][j] = full_board[i][j]
                if get_candidates(board, i, j) == [full_board[i][j]]:
                    data.append(generate_single_step_data(board, i, j))
                    board[i][j] = full_board[i][j]
                    filled[i][j] = True
                    steps.append((i,j,full_board[i][j]))
                    break
            else:
                continue
            break
        else:
            # 说明某一步无法用唯一候选确定，终止
            break
    return data if sum(filled[i][j] for i in range(4) for j in range(4)) == 16 else []

def generate_many(n=1000, filename="sudoku_4x4_stepwise.jsonl"):
    from itertools import permutations
    from random import shuffle

    # 简化：用合法的全排列构造若干4x4解
    count = 0
    with open(filename, "w") as f:
        while count < n:
            base = [1, 2, 3, 4]
            rows = [base[:], base[:], base[:], base[:]]
            for row in rows:
                shuffle(row)
            valid = True
            for r in range(4):
                for c in range(4):
                    if not is_valid(rows, r, c, rows[r][c]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                dataset = generate_training_data_from_full_board(rows)
                for item in dataset:
                    f.write(json.dumps(item) + "\n")
                count += 1

generate_many(1000)
