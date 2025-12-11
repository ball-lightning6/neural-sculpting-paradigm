import subprocess
import os
import time
import json
import threading
from queue import Queue, Empty

# 确保这些库已安装: pip install cchess torch
import cchess
import torch
import torch.nn.functional as F

class PikaFishEngineFinal:
    """
    最终修正版引擎: 正确地在初始化时设置一次性选项。
    """
    def __init__(self, engine_path, worker_id=0):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"引擎未找到: {engine_path}")

        command = [engine_path]
        if os.name != 'nt':
            try:
                cpu_to_use = worker_id % os.cpu_count()
                command = ["taskset", "-c", str(cpu_to_use)] + command
            except:
                pass # taskset might fail or os.cpu_count might be None

        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, bufsize=1,
        )

        self.output_queue = Queue()
        self.reader_thread = threading.Thread(target=self._reader_thread_func, daemon=True)
        self.reader_thread.start()

        # --- 初始化阶段 ---
        self._send_and_wait("uci", "uciok")
        # 1. 设置哈希值以打破确定性
        hash_size = 16 + worker_id % 16
        self._send(f"setoption name Hash value {hash_size}")
        # 2. 设置线程数作为物理隔离的补充
        self._send("setoption name Threads value 1")
        # 3. 用 isready 确认所有设置已生效
        self._send_and_wait("isready", "readyok")

    # --- 非阻塞读取线程 (保持不变) ---
    def _reader_thread_func(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                self.output_queue.put(line.strip())
        except ValueError:
            pass

    # --- 通信函数 (保持不变) ---
    def _send(self, command):
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except OSError:
            pass

    def _wait_for(self, keyword, timeout=15):
        lines = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                lines.append(line)
                if keyword in line: return lines
            except Empty: continue
        # raise TimeoutError(f"等待 '{keyword}' 超时")
        return lines # Return what we have instead of crashing, to be more robust

    def _send_and_wait(self, command, keyword, timeout=15):
        self._send(command)
        return self._wait_for(keyword, timeout)

    # --- 核心方法 (已修正) ---
    def get_best_move(self, fen, depth):
        # ！！！删除了这里错误的 setoption 调用！！！
        self._send(f"position fen {fen}")
        lines = self._send_and_wait(f"go depth {depth}", "bestmove")
        for line in reversed(lines):
            if line.startswith("bestmove"):
                return line.split()[1]
        return None

    def get_multipv_scores(self, fen, depth, multipv):
        # ！！！删除了这里错误的 setoption 调用！！！
        self._send(f"position fen {fen}")
        self._send(f"setoption name MultiPV value {multipv}")
        lines = self._send_and_wait(f"go depth {depth}", "bestmove", timeout=40)
        
        move_scores, player_to_move = [], fen.split()[1]
        for line in lines:
            if line.startswith("info") and "score cp" in line and " pv " in line:
                try:
                    parts = line.split()
                    try:
                        pv_index = parts.index("pv")
                        move = parts[pv_index + 1]
                    except ValueError:
                        continue 
                        
                    try:
                        score_index = parts.index("score")
                        score_type = parts[score_index + 1] # cp or mate
                        score_val = int(parts[score_index + 2])
                        
                        if score_type == "mate":
                            # Convert mate score to large cp score
                            score = 10000 if score_val > 0 else -10000
                        else:
                            score = score_val
                    except ValueError:
                        continue

                    if player_to_move == 'b': score = -score
                    move_scores.append({"move": move, "score": score})
                except (ValueError, IndexError):
                    continue
        return move_scores

    def close(self):
        try:
            self._send("quit")
            self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            if self.process: self.process.kill()

# --- Worker 函数 (保持不变) ---
# 它现在调用的是修正后的引擎类，所以它的行为也会被修正
def worker_fen_generation(args):
    worker_id, num_games, max_steps, depth, engine_path, temp_output_file = args
    with open(temp_output_file, "w", encoding="utf-8") as f_out:
        try:
            engine = PikaFishEngineFinal(engine_path, worker_id=worker_id)
            for _ in range(num_games):
                board, fen_set_in_game = cchess.Board(), set()
                for _ in range(max_steps):
                    fen = board.fen()
                    move_str = engine.get_best_move(fen, depth)
                    if not move_str or move_str == "(none)": break
                    try:
                        move = cchess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                            # 为避免单局内循环，也在这里加一个去重
                            key = " ".join(board.fen().split()[:2])
                            if key not in fen_set_in_game:
                                fen_set_in_game.add(key)
                                f_out.write(board.fen() + "\n")
                            else: # 如果出现重复局面，通常意味着对局结束或无意义
                                break
                        else: break
                    except Exception: break
            engine.close()
        except Exception as e:
            print(f"Worker {worker_id} failed: {e}")

def worker_label_generation(args):
    """
    (阶段二 Worker) 负责处理一个FEN数据块，并将软标签结果写入自己的临时文件。
    """
    worker_id, fen_chunk, move_map, config, counter = args
    # 每个 worker 创建自己独立的、绑定到特定核心的引擎实例
    try:
        engine = PikaFishEngineFinal(config['pikafish_engine_path'], worker_id=worker_id)
    except Exception as e:
        print(f"Worker {worker_id} failed to start engine: {e}")
        return
    
    # 定义临时输出文件路径
    temp_output_file = os.path.join(config['temp_dir'], f"labels_{worker_id}.jsonl")

    with open(temp_output_file, "w", encoding="utf-8") as f_out:
        for fen in fen_chunk:
            # 获取 MultiPV 分数
            try:
                move_scores = engine.get_multipv_scores(
                    fen, config['engine_depth'], config['multipv_count']
                )
            except Exception:
                try:
                    engine.close()
                except: pass
                
                try:
                    engine = PikaFishEngineFinal(config['pikafish_engine_path'], worker_id=worker_id)
                    move_scores = None # Skip this one or retry? logic here says skip effectively by setting None then checking loop
                except:
                    move_scores = None
            
            if not move_scores:
                # 即使没有分数，也更新计数器，表示这个FEN处理过了
                if counter is not None:
                    try:
                        counter.value += 1
                    except: pass
                continue

            # --- Softmax 逻辑 ---
            valid_moves, scores = [], []
            for item in move_scores:
                if item["move"] in move_map:
                    valid_moves.append(item["move"])
                    scores.append(item["score"])
            
            if not scores:
                if counter is not None:
                    try:
                        counter.value += 1
                    except: pass
                continue

            try:
                probabilities = F.softmax(torch.tensor(scores, dtype=torch.float32) / config['temperature'], dim=0)
                soft_label_vector = torch.zeros(len(move_map), dtype=torch.float32)
                
                for move_str, prob in zip(valid_moves, probabilities):
                    soft_label_vector[move_map[move_str]] = prob.item()
                
                # 将FEN和计算出的软标签写入临时文件
                f_out.write(json.dumps({"fen": fen, "label": soft_label_vector.tolist()}) + '\n')
            except Exception as e:
                print(f"Error processing probabilities: {e}")
            
            # 每处理完一个FEN，就给共享计数器加一
            if counter is not None: 
                try:
                    counter.value += 1
                except: pass
            
    try:
        engine.close()
    except: pass
