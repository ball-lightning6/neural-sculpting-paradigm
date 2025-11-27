#!/usr/bin/env python3
"""
中国象棋引擎自对弈局面生成器
通过专业象棋引擎自对弈生成高质量局面
"""

import json
import os
import random
import subprocess
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cchess

# ==============================================================================
# 配置参数
# ==============================================================================
NUM_SAMPLES = 100  # 默认生成样本数（每盘对局生成多个局面）
NUM_PROCESSES = cpu_count()
MAX_GAMES = 20      # 默认对局数量
MAX_STEPS = 60      # 每局最大步数
SAMPLE_RANGE = (8, 25)  # 抽取局面的步数范围
DEPTH = 2           # 引擎搜索深度
OUTPUT_DIR = "chess_engine_selfplay_dataset"

# 引擎路径（可以通过环境变量或参数指定）
DEFAULT_ENGINE_PATH = os.environ.get('PIKAFISH_PATH', 'pikafish.exe')

# ==============================================================================
# 象棋引擎接口
# ==============================================================================
class PikaFishEngine:
    """PikaFish引擎UCI协议接口"""
    
    def __init__(self, engine_path=DEFAULT_ENGINE_PATH):
        try:
            self.process = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            self._send("uci")
            self._wait_for("uciok")
            self._send("isready")
            self._wait_for("readyok")
        except Exception as e:
            raise RuntimeError(f"无法启动象棋引擎 {engine_path}: {e}")
    
    def _send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
    
    def _wait_for(self, keyword):
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            lines.append(line)
            if keyword in line:
                break
        return lines
    
    def get_best_move(self, fen, depth=DEPTH):
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        lines = self._wait_for("bestmove")
        for line in lines:
            if line.startswith("bestmove"):
                return line.split()[1]
        return None
    
    def close(self):
        try:
            self._send("quit")
            self.process.terminate()
        except:
            pass

# ==============================================================================
# 象棋对弈模拟
# ==============================================================================
class SimulatedGame:
    """象棋引擎自对弈模拟器"""
    
    def __init__(self, engine_path=DEFAULT_ENGINE_PATH, max_steps=MAX_STEPS, depth=DEPTH):
        self.engine = PikaFishEngine(engine_path)
        self.depth = depth
        self.max_steps = max_steps
        self.history = []  # 保存局面序列
        self.start_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    
    def simulate_single_game(self):
        """模拟一盘完整的自对弈，返回局面序列"""
        try:
            board = cchess.Board()
            self.history = [board.fen()]  # 起始局面
            
            for step in range(self.max_steps):
                current_fen = board.fen()
                move_str = self.engine.get_best_move(current_fen, self.depth)
                
                if not move_str or move_str == "(none)":
                    break  # 无合法走法，对局结束
                
                move = cchess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    break  # 非法走法，跳过
                
                board.push(move)
                new_fen = board.fen()
                self.history.append(new_fen)
            
            return self.history
        except Exception as e:
            print(f"对局模拟失败: {e}")
            return []
    
    def close(self):
        self.engine.close()

# ==============================================================================
# 单样本生成任务（用于多进程）
# ==============================================================================
def generate_sample_task(args):
    """多进程任务函数"""
    idx, engine_path, max_steps, depth, sample_range = args
    try:
        game = SimulatedGame(engine_path, max_steps, depth)
        history = game.simulate_single_game()
        game.close()
        
        if not history:
            return None
        
        # 抽取指定范围的步数作为样本
        start_step, end_step = sample_range
        if len(history) > start_step:
            selected_fens = history[start_step:min(end_step, len(history))]
            
            samples = []
            for fen in selected_fens:
                sample = {
                    "input": fen,  # FEN格式局面
                    "output": {
                        "fen": fen,
                        "generation_method": "engine_selfplay",
                        "engine_depth": depth,
                        "game_quality": "high"
                    },
                    "metadata": {
                        "generation_method": "engine_selfplay",
                        "engine_type": "pikafish",
                        "depth": depth,
                        "quality_level": "high"
                    }
                }
                samples.append(sample)
            
            return samples
    except Exception as e:
        print(f"生成样本 {idx} 失败: {e}")
        return None

# ==============================================================================
# 主生成函数
# ==============================================================================
def generate_dataset(num_samples=NUM_SAMPLES, max_games=MAX_GAMES, max_steps=MAX_STEPS, 
                    sample_range=SAMPLE_RANGE, depth=DEPTH, engine_path=DEFAULT_ENGINE_PATH,
                    output_dir=OUTPUT_DIR):
    """
    生成完整的数据集
    """
    print("=" * 60)
    print("中国象棋引擎自对弈局面数据集生成器")
    print(f"参数: num_games={max_games}, max_steps={max_steps}, depth={depth}")
    print(f"引擎路径: {engine_path}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证引擎可用性
    try:
        test_engine = PikaFishEngine(engine_path)
        test_engine.close()
        print("✅ 象棋引擎测试成功")
    except Exception as e:
        print(f"❌ 象棋引擎测试失败: {e}")
        print("请确保PikaFish引擎已安装，或设置PIKAFISH_PATH环境变量")
        return
    
    # 准备多进程任务
    tasks = [(i, engine_path, max_steps, depth, sample_range) for i in range(max_games)]
    
    print(f"开始生成 {max_games} 盘对局，使用 {NUM_PROCESSES} 个进程...")
    print(f"每盘抽取步数范围: {sample_range[0]}-{sample_range[1]}")
    
    # 多进程生成
    all_samples = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_task, tasks), 
                          total=max_games, desc="对局进度"):
            if result is not None:
                all_samples.extend(result)
    
    print(f"成功生成 {len(all_samples)} 个局面样本")
    
    # 保存训练集 (90%)
    train_size = int(len(all_samples) * 0.9)
    train_data = all_samples[:train_size]
    
    # 保存评估集 (10%)
    eval_data = all_samples[train_size:]
    
    # 写入文件
    train_file = os.path.join(output_dir, f"chess_engine_selfplay_train.jsonl")
    eval_file = os.path.join(output_dir, f"chess_engine_selfplay_eval.jsonl")
    
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
        "dataset_name": "中国象棋引擎自对弈局面数据集",
        "total_samples": len(all_samples),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "num_games": max_games,
        "max_steps_per_game": max_steps,
        "sample_range": sample_range,
        "engine_depth": depth,
        "engine_type": "pikafish",
        "generation_method": "engine_selfplay",
        "quality_level": "high",
        "output_format": "jsonl",
        "train_file": train_file,
        "eval_file": eval_file
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ 数据集生成完成!")
    print(f"总计: {len(all_samples)} 个局面样本")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"评估集: {len(eval_data)} 个样本")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

# ==============================================================================
# 主执行部分
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中国象棋引擎自对弈局面数据集生成器")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, 
                       help=f"目标样本数量 (默认: {NUM_SAMPLES})")
    parser.add_argument("--max_games", type=int, default=MAX_GAMES,
                       help=f"对局数量 (默认: {MAX_GAMES})")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                       help=f"每局最大步数 (默认: {MAX_STEPS})")
    parser.add_argument("--depth", type=int, default=DEPTH,
                       help=f"引擎搜索深度 (默认: {DEPTH})")
    parser.add_argument("--sample_range", type=int, nargs=2, default=SAMPLE_RANGE,
                       help=f"抽取步数范围 (默认: {SAMPLE_RANGE[0]} {SAMPLE_RANGE[1]})")
    parser.add_argument("--engine_path", type=str, default=DEFAULT_ENGINE_PATH,
                       help=f"引擎路径 (默认: {DEFAULT_ENGINE_PATH})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES,
                       help=f"进程数 (默认: {NUM_PROCESSES})")
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes > 0:
        NUM_PROCESSES = args.num_processes
    
    # 检查引擎路径
    if not os.path.exists(args.engine_path):
        print(f"警告: 引擎路径 {args.engine_path} 不存在")
        print(f"请设置 PIKAFISH_PATH 环境变量或提供正确的 --engine_path")
        print(f"尝试默认路径: {DEFAULT_ENGINE_PATH}")
    
    # 生成数据集
    generate_dataset(
        num_samples=args.num_samples,
        max_games=args.max_games,
        max_steps=args.max_steps,
        depth=args.depth,
        sample_range=tuple(args.sample_range),
        engine_path=args.engine_path,
        output_dir=args.output_dir
    )