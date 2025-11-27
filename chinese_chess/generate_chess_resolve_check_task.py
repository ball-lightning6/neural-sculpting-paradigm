#!/usr/bin/env python3
"""
中国象棋解将任务数据集生成器
专门生成"解将"战术训练数据
"""

import json
import os
import random
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cchess

# ==============================================================================
# 配置参数
# ==============================================================================
NUM_SAMPLES = 1000  # 默认生成样本数
NUM_PROCESSES = cpu_count()
MAX_ATTEMPTS = 100  # 每个样本的最大尝试次数
OUTPUT_DIR = "chess_resolve_check_dataset"

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

# 走法ID映射（需要外部文件或动态生成）
# 这里使用一个简化的映射，实际应用中应该有完整的move2idx.json
def get_move_id_mapping():
    """获取走法到ID的映射（简化版本）"""
    # 这里应该加载真实的move2idx.json文件
    # 临时使用一个简化的映射逻辑
    basic_moves = [
        'a0a1', 'a0a2', 'a0a3', 'a0a4', 'a0a5', 'a0a6',
        'b0b2', 'b0b4', 'b0b6', 'b0b8',
        'c1c3', 'c1c5', 'c1c7',
        'h0h2', 'h0h4', 'h0h6', 'h0h8',
        'i0i1', 'i0i2', 'i0i3', 'i0i4', 'i0i5', 'i0i6'
    ]
    return {move: i for i, move in enumerate(basic_moves)}

# ==============================================================================
# 核心生成函数
# ==============================================================================
def random_piece_counts():
    """生成随机棋子数量配置"""
    counts = {}
    for k, v in piece_pool.items():
        if k == 'k' or k == 'K':
            counts[k] = 1  # 强制每方必须有帅/将
        else:
            counts[k] = random.randint(0, v)
    return counts

def place_pieces_with_restriction(piece_counts):
    """
    在棋盘上放置棋子，遵循位置约束
    返回: 成功放置的棋盘或None（如果无法合法放置）
    """
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

def is_general_face_to_face(board):
    """检测将帅是否照面"""
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

def board_to_fen(board):
    """将棋盘转换为FEN格式"""
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

def generate_single_check_position():
    """生成单个被将军但非将杀的局面"""
    for _ in range(100):  # 最多尝试100次
        piece_counts = random_piece_counts()
        board = place_pieces_with_restriction(piece_counts)
        if board is None:
            continue
        if is_general_face_to_face(board):
            continue
        
        # 生成完整的FEN（包括走子方）
        fen = board_to_fen(board) + ' ' + random.choice(['w', 'b'])
        
        # 检查是否是被将军但非将杀
        try:
            b = cchess.Board(fen)
            if b.is_check() and not b.is_checkmate():
                return fen
        except:
            continue
    
    return None

def get_legal_move_ids(board):
    """获取所有合法走法的ID列表"""
    legal_moves = list(board.legal_moves)
    
    # 这里应该使用真实的move2idx映射
    # 临时使用简单的映射逻辑
    move_ids = []
    for i, move in enumerate(legal_moves):
        move_str = move.uci()
        # 简单的ID映射：基于走法字符串的哈希
        move_id = hash(move_str) % 1000  # 临时方案，应该有真实的move2idx
        move_ids.append(move_id)
    
    return move_ids

# ==============================================================================
# 单样本生成任务（用于多进程）
# ==============================================================================
def generate_sample_task(args):
    """多进程任务函数"""
    idx, max_attempts = args
    try:
        fen = generate_single_check_position()
        if fen is not None:
            # 构建标准输出格式
            board = cchess.Board(fen)
            move_ids = get_legal_move_ids(board)
            
            sample = {
                "input": fen,  # FEN格式局面
                "output": {
                    "fen": fen,
                    "legal_move_ids": move_ids,
                    "num_legal_moves": len(move_ids),
                    "is_check": True,
                    "is_checkmate": False
                },
                "metadata": {
                    "generation_method": "resolve_check_task",
                    "tactical_type": "resolve_check",
                    "constraint_type": "position_and_count_restricted"
                }
            }
            return sample
    except Exception as e:
        print(f"生成样本 {idx} 失败: {e}")
        return None

# ==============================================================================
# 主生成函数
# ==============================================================================
def generate_dataset(num_samples=NUM_SAMPLES, max_attempts=100, output_dir=OUTPUT_DIR):
    """
    生成完整的数据集
    """
    print("=" * 60)
    print("中国象棋解将任务数据集生成器")
    print(f"参数: num_samples={num_samples}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备多进程任务
    tasks = [(i, max_attempts) for i in range(num_samples)]
    
    print(f"开始生成 {num_samples} 个解将局面，使用 {NUM_PROCESSES} 个进程...")
    print("正在生成被将军但非将杀的局面...")
    
    # 多进程生成
    dataset = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_task, tasks), 
                          total=num_samples, desc="生成进度"):
            if result is not None:
                dataset.append(result)
    
    print(f"成功生成 {len(dataset)} 个解将局面")
    
    # 保存训练集 (90%)
    train_size = int(len(dataset) * 0.9)
    train_data = dataset[:train_size]
    
    # 保存评估集 (10%)
    eval_data = dataset[train_size:]
    
    # 写入文件
    train_file = os.path.join(output_dir, f"chess_resolve_check_train.jsonl")
    eval_file = os.path.join(output_dir, f"chess_resolve_check_eval.jsonl")
    
    print(f"写入训练集: {len(train_data)} 条记录 -> {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"写入评估集: {len(eval_data)} 条记录 -> {eval_file}")
    with open(eval_file, 'w', encoding='utf-8') as f:
        for record in eval_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 生成元数据
    metadata = {
        "dataset_name": "中国象棋解将任务数据集",
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "generation_method": "resolve_check_task",
        "tactical_type": "resolve_check",
        "constraint_type": "position_and_count_restricted",
        "output_format": "jsonl",
        "train_file": train_file,
        "eval_file": eval_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ 解将任务数据集生成完成!")
    print(f"总计: {len(dataset)} 个解将局面")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"评估集: {len(eval_data)} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ==============================================================================
# 主执行部分
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中国象棋解将任务数据集生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, 
                       help=f"生成样本数量 (默认: {NUM_SAMPLES})")
    parser.add_argument("--max_attempts", type=int, default=100,
                       help=f"每个样本最大尝试次数 (默认: 100)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES,
                       help=f"进程数 (默认: {NUM_PROCESSES})")
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes > 0:
        NUM_PROCESSES = args.num_processes
    
    # 生成数据集
    generate_dataset(
        num_samples=args.num_samples,
        max_attempts=args.max_attempts,
        output_dir=args.output_dir
    )