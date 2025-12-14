# 中国象棋 (Chinese Chess)

## 总体思路 (Overall Architecture)

本项目构建了一个完整的中国象棋 AI 训练闭环。核心理念是通过**“专家迭代（Expert Iteration）”**和**“软标签蒸馏（Soft Label Distillation）”**技术，将强大的传统引擎（如 PikaFish）的知识迁移到神经网络（Policy Transformer）中。

项目流程主要包含三个阶段：
1.  **基础训练**：利用随机合法走法 (`generate_chess_positions_by_random_moves.py`) 让模型学会基本的象棋规则和走法空间。
2.  **强化数据生成**：利用 `generate_chess_positions_from_engine_self_play.py` 调用 UCI 引擎进行自对弈，生成包含高质量走法和评估分数的训练数据。
3.  **专家策略蒸馏**：通过 `generate_soft_labels.py` 生成概率分布标签（Soft Labels），训练策略网络（Policy Network）模仿引擎的搜索结果。

此外，还包含 `play_with_ai.py` 用于验证模型实战能力，以及 `generate_chess_resolve_check_task.py` 等针对特定战术场景（如解将）的专项训练。

---


## 1. **chinese_chess/generate_chess_positions_by_random_moves.py**

- **用途:** 通过模拟一个完全随机的玩家下棋的过程，快速生成大量看起来合理的、合法的中国象棋局面。
    
- **逻辑:** 脚本从标准的中国象棋起始局面开始。在一个循环中，它会获取当前局面下所有合法的走法，然后随机选择其中一步并执行。这个过程会重复max_steps次，最终得到一个随机但合法的局面。
    
- **I/O格式:**
    
    - 输出: FEN格式的局面字符串。
        
- **主要参数:** max_steps, max_capture。

---

## 2. **chinese_chess/generate_chess_positions_by_random_placement.py**

- **用途:** 通过在棋盘上随机放置棋子（而非模拟下棋）来生成大量非典型的、但大部分合法的中国象棋局面，用于对模型的鲁棒性进行压力测试。
    
- **逻辑:** 该脚本不是通过下棋来生成局面，而是直接在棋盘上随机地、遵循棋子位置约束和将帅不照面规则地放置棋子，从而创造出大量在真实对局中极少出现但语法合法的局面。
    
- **I/O格式:**
    
    - 输出: FEN格式的局面字符串。
        
- **主要参数:** num_fens。

---

## 3. **chinese_chess/generate_chess_positions_from_engine_self_play.py**

- **用途:** 生成大量高质量、符合实战逻辑的中国象棋局面（FEN格式），作为训练棋类AI的基础数据源。
    
- **逻辑:** 通过子进程调用一个强大的第三方象棋引擎（PikaFish），模拟数万盘高水平的自对弈棋局。在模拟过程中，记录下棋局每一步的FEN表示，从而构建一个庞大且真实的局面数据库。
    
- **I/O格式:**
    
    - 输出: 一个.txt文件，每行包含一个完整的FEN字符串。
        
- **主要参数:** num_games, max_steps, depth。

---

## 4. **chinese_chess/generate_preprocess_legal_moves.py**

- **用途:** 这是一个数据预处理脚本，用于将FEN格式的局面数据集转换为模型可以直接学习的“合法走法预测”任务。
    
- **逻辑:** 读取一个FEN文件，对于每一个局面，使用cchess库解析并生成所有合法走法。然后，根据一个全局的映射文件，将每个具体的走法（如 'h2e2'）转换成一个唯一的整数ID。
    
- **I/O格式:**
    
    - 输入: .txt文件，每行一个FEN。
        
    - 输出: .jsonl文件，每个JSON对象包含fen和其对应的legal_move_ids列表。
        
- **主要参数:** fen_file, output_file。

---

## 5. **chinese_chess/generate_chess_resolve_check_task.py**

- **用途:** 生成一个专门针对中国象棋中“解将”（Resolving a Check）这一特定战术场景的数据集。这个任务要求模型在处于被将军的状态下，找出所有能够合法解除将军的走法。
    
- **逻辑:** 脚本首先从一个庞大的随机局面库中进行筛选，只保留那些满足“正被将军，但并非无子可走（非将死）”条件的局面。然后，对于每一个筛选出的局面，它会计算所有能解除将军的合法走法，并将这些走法的ID保存下来。
    
- **I/O格式:**
    
    - 输出: 一个.jsonl文件。每个JSON对象包含fen（局面）和legal_move_ids（一个整数列表，代表所有合法的解将走法）。
        
- **主要参数:** fen_file, output_file。

---

## 6. **chinese_chess/worker_logic.py**

- **用途:** 这是一个核心的**引擎交互与任务调度模块**，支撑了从引擎到数据的桥梁建设。它既用于生成FEN局面，也用于计算走法的软标签。

- **逻辑:** 
    - **封装 Pikafish 引擎:** `PikaFishEngineFinal` 类封装了与 Pikafish 国际水平象棋引擎的 UCI 通信协议。它通过多进程安全的管道交互，发送 `go depth` 等指令，并健壮地解析引擎返回的 `bestmove` 和 `score` 信息。
    - **确定性打破:** 为了生成多样化的数据，引擎初始化时会动态设置 Hash 大小，打破确定性。
    - **多进程 Worker:** 
        - `worker_label_generation`: 专为 `generate_soft_labels.py` 设计的 Worker 函数。它接收分配到的 FEN 块，启动独立的引擎进程，计算每个局面的 MultiPV（多候选）走法及其评分，生成软标签，并写入临时文件。

- **关键特性:** 高效的多进程支持、健壮的 UCI 协议处理、支持 MultiPV 分数获取。

---

## 7. **chinese_chess/generate_soft_labels.py**

- **用途:** 这是一个**知识蒸馏（Knowledge Distillation）**的数据生成脚本。它的目的是将强大的传统象棋引擎（Teacher）的知识，转移到一个适合神经网络（Student）学习的格式（软标签）。

- **逻辑:**
    - **大规模并行:** 使用 `multiprocessing.Pool` 将数百万个 FEN 局面分配给多个 CPU 核心。
    - **软标签计算:** 对于每个局面，调用引擎计算 Top-K（例如 MultiPV=5）最佳走法的评分（CP Score）。
    - **Softmax 归一化:** 将引擎的绝对分数（CP score）通过带温度（Temperature）的 Softmax 函数转换为概率分布。温度参数 `temperature` 控制分布的平滑程度：高温使分布更平滑（保留更多次优走法信息），低温使分布更尖锐（仅关注最佳走法）。
    - **映射与保存:** 将走法字符串映射为模型词表的索引，最终保存为 JSONL 格式。

- **I/O格式:**
    - 输入: FEN 文件（每行一个局面） + `move2idx.json` (词表)。
    - 输出: JSONL 文件，每行包含 `{"fen": "...", "label": [0.01, 0.95, ...]}`。

- **主要参数:** `pikafish_engine_path`, `input_fen_file`, `engine_depth` (思考深度), `multipv_count` (候选数), `temperature`。

---

## 8. **chinese_chess/play_with_ai.py**

- **用途:** 这是一个**人机对弈终端客户端**，用于测试和评估训练好的策略网络。

- **逻辑:**
    - **加载模型:** 自动加载最新的训练检查点。
    - **局面编码:** 将当前棋盘状态（FEN）实时编码为模型输入张量。
    - **策略采样:** 模型输出下一步所有走法的概率分布。脚本根据 `sampling_temperature` 对概率进行调整，并使用多项式采样（Multinomial Sampling）选择走法，从而实现多样化且高水平的下棋风格。
    - **交互循环:** 提供简单的命令行界面，处理用户输入（UCI格式，如 `h2e2`）和 AI 走法的显示。

- **交互方式:** 命令行输入。
- **主要配置:** `model_dir`, `sampling_temperature`。