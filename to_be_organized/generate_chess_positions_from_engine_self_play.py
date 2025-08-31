import subprocess
from copy import deepcopy
import cchess
import json

class PikaFishEngine:
    def __init__(self, engine_path="pikafish.exe"):
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

    def get_best_move(self, fen, depth=3):
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        lines = self._wait_for("bestmove")
        for line in lines:
            if line.startswith("bestmove"):
                return line.split()[1]
        return None

    def close(self):
        self._send("quit")
        self.process.terminate()

class SimulatedGame:
    def __init__(self, engine, max_steps=60, depth=2):
        self.engine = engine
        self.depth = depth
        self.max_steps = max_steps
        self.history = []  # 保存局面列表
        self.fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

    def apply_move(self, fen, move):
        # ❗这里是占位逻辑，你需要用真正的象棋库来实现 FEN + move → 新FEN
        # 暂时先重复当前FEN占位（后续我们加合法模拟器）
        return fen

    def simulate(self):
        current_fen = self.fen
        self.history.append(current_fen)

        for step in range(self.max_steps):
            move = self.engine.get_best_move(current_fen, depth=self.depth)
            if not move or move == "(none)":
                print(f"⛔ 博弈终止：step {step+1}")
                break

            print(f"🔁 第 {step+1:02d} 步：{move}")
            next_fen = self.apply_move(current_fen, move)

            # 添加下一步局面
            self.history.append(next_fen)
            current_fen = next_fen

        return deepcopy(self.history)


# engine_path = r"F:\data\皮卡鱼引擎 鲨鱼界面\皮卡鱼引擎+鲨鱼界面\皮卡鱼-Pikafish\pikafish-avx2.exe" #中文路径竟然可以
# engine = PikaFishEngine(engine_path=engine_path)
#
# fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
# move = engine.get_best_move(fen, depth=2)
# print(f"Best move: {move}")
#
# engine.close()

# engine = PikaFishEngine(engine_path=engine_path)

def print_board(board):
    """打印当前棋盘（中文可视化）"""
    symbols = {
        "R": "车", "N": "马", "B": "相", "A": "仕", "K": "帅", "C": "炮", "P": "兵",
        "r": "车", "n": "马", "b": "象", "a": "士", "k": "将", "c": "炮", "p": "卒",
        ".": "．"
    }
    board_str = str(board).split("\n")
    print("  ａ ｂ ｃ ｄ ｅ ｆ ｇ ｈ ｉ")
    for idx, line in enumerate(board_str):
        row = [symbols.get(ch, ch) for ch in line]
        print(f"{9 - idx} {' '.join(row)} {9 - idx}")
    print("  ａ ｂ ｃ ｄ ｅ ｆ ｇ ｈ ｉ")


def simulate_game(max_steps=60, depth=2):
    engine_path = r"F:\data\皮卡鱼引擎 鲨鱼界面\皮卡鱼引擎+鲨鱼界面\皮卡鱼-Pikafish\pikafish-avx2.exe"
    engine = PikaFishEngine(engine_path=engine_path)
    board = cchess.Board()
    history = []

    print("🎯 起始局面：")
    print_board(board)
    print("")

    for step in range(max_steps):
        fen = board.fen()
        move_str = engine.get_best_move(fen, depth)
        if not move_str or move_str == "(none)":
            print(f"⛔ 第 {step+1} 步无合法走法，博弈终止")
            break

        move = cchess.Move.from_uci(move_str)
        if move in board.legal_moves:
            board.push(move)
            new_fen = board.fen()
            history.append({"fen": fen, "best_move": move_str, "next_fen": new_fen})

            print(f"🔁 第 {step+1:02d} 步：{move_str}")
            print_board(board)
            print("")
        else:
            print(f"❌ 非法走法 {move_str}，跳过")
            break

    engine.close()
    return history
unique_fens = set()
# simulate_game()
def generate_fens_from_simulations(num_games=20, max_steps=40, sample_range=(8, 25), depth=2):
    engine_path = r"F:\data\皮卡鱼引擎 鲨鱼界面\皮卡鱼引擎+鲨鱼界面\皮卡鱼-Pikafish\pikafish-avx2.exe"
    engine = PikaFishEngine(engine_path=engine_path)
    all_fens = []
    with open('dataset_huge.txt', 'w') as f:

        for g in range(num_games):
            board = cchess.Board()
            fens = [board.fen()]
            print(f"🎮 模拟第 {g+1} 盘")

            for step in range(max_steps):
                fen = board.fen()
                move_str = engine.get_best_move(fen, depth)
                if not move_str or move_str == "(none)":
                    break
                move = cchess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                    fen = board.fen()
                    key = fen.split()[0] + " " + fen.split()[1]
                    if key not in unique_fens:
                        unique_fens.add(key)
                        f.write(fen+'\n')
                else:
                    break

            # 抽取中盘/残局样本
            selected = fens[sample_range[0]:sample_range[1]]
            all_fens.extend(selected)

        engine.close()
        print(f"✅ 已采样局面数：{len(unique_fens)}")
        return all_fens

# all_fens = list(set(
#     generate_fens_from_simulations(num_games=40000, max_steps=50,
#     sample_range=(0, 50),depth=4) +  # 低
#     generate_fens_from_simulations(num_games=20000, max_steps=50,
#     sample_range=(0, 50),depth=6) +  # 中
#     generate_fens_from_simulations(num_games=10000, max_steps=50,
#     sample_range=(0, 50),depth=8)    # 高
# ))
# print(len(all_fens))
# print(all_fens[:10])
#
# unique_fens = {}
# for fen in all_fens:
#     key = fen.split()[0] + " " + fen.split()[1]
#     if key not in unique_fens:
#         unique_fens[key] = fen  # 保留原始完整 FEN（保守做法）
#
# with open('dataset_huge.txt','w') as f:
#     f.write(json.dumps(unique_fens,indent=4))

generate_fens_from_simulations(num_games=50000, max_steps=50,
    sample_range=(0, 50),depth=4)   # 低
generate_fens_from_simulations(num_games=25000, max_steps=50,
sample_range=(0, 50),depth=6)   # 中
generate_fens_from_simulations(num_games=12500, max_steps=50,
sample_range=(0, 50),depth=8)    # 高

# 4——1783.2432432432432432432432432432
# 6——1192.7814379833858493268404468634
# 8——530.57142857142857142857142857143
# print(len(all_fens))
# print(all_fens[:10])
#
#
# for fen in all_fens:
#     key = fen.split()[0] + " " + fen.split()[1]
#     if key not in unique_fens:
#         unique_fens[key] = fen  # 保留原始完整 FEN（保守做法）

# with open('dataset_huge.txt','w') as f:
#     f.write(json.dumps(unique_fens,indent=4))