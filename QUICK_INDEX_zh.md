## A. 符号规则学习 (Symbolic Rule Learning)

- generate_conditional_add_subtract.py: 此脚本用于探究模型处理“规则冲突”或“条件逻辑”的能力。它旨在测试当一个数据集内隐式混合了多种规则时，模型是否会学习失败；以及当提供一个明确的指示符时，模型是否能成功学习。
    
- generate_add_binary_modulo.py: 这是一个早期的基础算术实验，用于测试模型学习模加法（或称“截断加法”）的能力，这种运算常见于计算机硬件的定宽整数运算。
    
- generate_multiply_binary.py: 作为二进制算术能力的一个基准测试，生成N-bit整数的乘法数据集。
    
- generate_multiply_binary_no_carry_phase1.py: 这是乘法“解耦”实验的第一阶段。旨在测试模型是否能学会乘法的第一步：无进位的按位相乘和错位相加，将一个复杂的乘法问题分解为一个更简单的计数问题。
    
- generate_multiply_binary_from_counts_phase2.py: 这是乘法“解耦”实验的第二阶段。旨在验证一个独立的模型能否学会处理复杂的进位逻辑，即从一个“无进位计数向量”中计算出最终的二进制乘积。
    
- generate_add_hexadecimal.py: 对比模型在不同符号系统下的学习能力。此脚本旨在验证模型学习的是加法这一抽象数学概念，还是仅仅是特定于二进制符号的模式。
    
- generate_multiply_decimal.py: 测试模型处理非二进制符号输入（0-9字符），并执行算术运算（乘法）的能力。
    
- generate_add_n_base_with_shuffle.py: 这是我们研究中一项**关键的决定性实验**，旨在彻底分离模型的“表面模式匹配”能力和“抽象结构学习”能力。
    
- generate_add_binary_with_position_shuffle.py: 这是“语义洗牌”系列实验中的“位置洗牌”部分。它旨在验证模型是否依赖于输入的固定空间结构，还是能学习到与位置无关的抽象关系。
    
- generate_add_hidden_constant.py: 测试模型在没有任何直接线索的情况下，从大量样本中**推断出隐藏规则或参数**的能力。这类似于一个简化的系统辨识（System Identification）问题。
    
- generate_multitask_alu.py: 此脚本旨在构建一个模拟算术逻辑单元 (ALU) 的多任务学习场景。它测试模型能否在一次前向传播中，对同一份输入并行执行多种不同的、定义明确的计算任务。
    
- generate_modulo_operation.py: 探究模型学习模运算（Modulo Operation）的能力，这是一个在数论和计算机科学中至关重要但具有“循环”性质的运算。
    
- generate_rsa_encryption.py: 测试模型学习高度非线性的、在计算上被认为是“困难”的确定性规则的能力。RSA加密是一个典型的例子。
    
- generate_cellular_automata_1d.py: 用于生成一维元胞自动机（CA）的演化数据集，以测试模型学习和执行局部、确定性规则的能力。
    
- generate_game_of_life_2d.py: 生成二维元胞自动机——Conway's Game of Life的数据集。此任务比一维CA更复杂，需要模型理解二维空间中的邻域关系。
    
- generate_cellular_automata_1d_multistate.py: 作为一维元胞自动机实验的扩展，测试模型处理非二进制状态空间的能力。
    
- generate_cellular_automata_programmable.py: 测试模型的“可编程性”或“元学习”能力。模型不仅要学会CA的演化过程，还要能根据每次输入中给出的不同规则来执行演化。
    
- generate_deduction_chain_text.py: 生成多步逻辑推理任务，测试模型执行符号演绎（deduction）的能力，类似于一个简化的定理证明器。
    
- generate_deduction_multirule_text.py: 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。
    
- generate_deduction_multirule_text_v2.py: 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。
    
- generate_deduction_multirule_binary.py: 这是对多规则推理任务的**格式优化**版本，旨在测试紧凑的二进制编码是否比稀疏的文本格式更有利于模型学习。
    
- generate_deduction_fixed_depth.py: 测试模型在有明确结构、固定深度的符号演绎任务中的多步推理能力。
    
- generate_function_composition.py: 测试模型学习函数组合（Function Composition）的能力。这要求模型像解释器一样，按顺序解析指令并对数据进行变换。
    
- generate_cellular_automata_inverse_rule90.py: 测试模型解决“逆问题”（Inverse Problem）的能力。给定一个确定性系统的输出，模型需要反向推断出满足特定约束（最稀疏且唯一）的可能输入。
    
- generate_count_set_bits.py: 测试模型执行全局聚合操作的能力。与局部规则不同，计数需要模型综合整个输入序列的信息。
    
- generate_sum_pattern_positions.py: 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。
    
- generate_sum_pattern_positions_v2.py: 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。
    
- generate_sum_pairwise_hamming_distance.py: 测试模型执行一个需要两层嵌套聚合操作的复杂任务。模型需要先在**每个比特位**上进行全局统计，然后再将**所有比特位**的结果累加起来。
    
- generate_circular_shift.py: 测试模型学习位移操作的能力，特别是循环位移（circular shift），这是密码学和底层编程中的常见操作。
    
- generate_multiply_matrix_3x3.py: 测试模型学习结构化代数运算（矩阵乘法）的能力，这比简单的标量运算需要更复杂的“数据路由”和“乘积累加”能力。
    
- generate_evaluate_boolean_expression_text.py: 测试模型解析一个简单的领域特定语言（DSL）并执行求值的能力，这比前面固定结构的表达式求值更进了一步。
    
- generate_evaluate_arithmetic_expression.py: 训练模型执行符号表达式的求值任务，这要求模型理解运算符优先级、变量替换和算术运算。
    
- generate_evaluate_arithmetic_expression_no_multiply.py: 这是对表达式求值任务的简化版本，旨在通过移除乘法运算来降低学习难度。
    
- generate_evaluate_arithmetic_expression_no_multiply_small_range.py: 这是在前一个“无乘法”版本基础上的进一步简化，通过缩小数值范围来进一步降低学习难度。
    
- generate_check_boolean_equivalence.py: 测试模型对布尔代数逻辑等价性的判断能力。这是一个抽象的符号推理任务，要求模型理解表达式的结构和布尔运算法则。
    
- generate_polynomial_shift_coefficients.py: 测试模型学习一个抽象的代数变换规则的能力，该任务需要模型理解多项式展开的内在结构。
    
- generate_convolution_2d.py: 测试模型学习二维卷积（Conv2D）这一基本图像处理操作的能力，并探究其是否能从输入输出对中推断出隐藏的固定规则（即卷积核本身）。
    
- generate_simple_block_cipher.py: 测试模型“破解”或学习一个简单但非平凡的自定义加密算法的能力。该任务代表了一类复杂的、具有高度混沌和雪崩效应的符号变换规则。
    
- generate_sin_function_float32.py: 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，使用标准的32位浮点数格式进行输入和输出。
    
- generate_sin_function_float64_to_int12_deprecated.py: 这是对sin函数拟合任务的另一种编码尝试，旨在探索使用更高精度的浮点输入和更低精度的量化二进制输出对学习效果的影响。
    
- generate_sin_function_float32_to_quantized_int.py: 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，并探索不同输入/输出编码方案对学习效果的影响。
    
- generate_multiply_binary_modulo.py: 作为基础算术实验的一部分，测试模型对截断乘法（或称模乘法）的掌握能力。
    
- generate_explainable_two_step_calculation.py: 测试模型输出计算“中间步骤”或“思维链”的能力，是“功能性可解释性”的一个直接验证。
    
- generate_chess_positions_by_random_moves.py: 通过模拟一个完全随机的玩家下棋的过程，快速生成大量看起来合理的、合法的中国象棋局面。
    
- generate_chess_positions_by_random_placement.py: 通过在棋盘上随机放置棋子（而非模拟下棋）来生成大量非典型的、但大部分合法的中国象棋局面，用于对模型的鲁棒性进行压力测试。
    
- generate_chess_positions_from_engine_self_play.py: 生成大量高质量、符合实战逻辑的中国象棋局面（FEN格式），作为训练棋类AI的基础数据源。
    
- generate_preprocess_legal_moves.py: 这是一个数据预处理脚本，用于将FEN格式的局面数据集转换为模型可以直接学习的“合法走法预测”任务。
    
- generate_chess_resolve_check_task.py: 生成一个专门针对中国象棋中“解将”（Resolving a Check）这一特定战术场景的数据集。这个任务要求模型在处于被将军的状态下，找出所有能够合法解除将军的走法。
    

## B. 算法学习 (Algorithm Learning)

- generate_sort_integers.py: 测试模型执行基本排序算法的能力，这是一个非局部的、需要对输入元素进行比较和重排的经典算法任务。
    
- generate_edit_distance.py: 测试模型学习解决动态规划问题的能力。编辑距离是一个典型的DP问题，需要模型在概念上构建一个二维的求解矩阵。
    
- generate_edit_distance_explainable.py: 这是“功能性可解释性”的一个核心实验。它要求模型不仅给出最终答案（编辑距离），还要输出达成答案的完整“思维链”（编辑过程）。
    
- generate_maze_random_walls.py: 测试模型在随机生成的“多孔”迷宫中的基础寻路能力。
    
- generate_maze_dense.py: 测试模型在复杂的、类似人类设计的“稠密”迷宮中进行路径规划的能力，这比随机墙壁迷宫更具挑战性。
    
- generate_blocks_world_arbitrary_goal.py: 解决经典的“积木世界”（Blocks World）规划问题，这是AI规划领域的基准任务。此版本允许指定任意的初始状态和终止状态。
    
- generate_blocks_world_fixed_goal.py: 这是对“积木世界”任务的简化，通过固定目标状态，旨在测试模型在目标明确、状态空间更结构化的情况下的学习能力。
    
- generate_blocks_world_fixed_goal_multilabel.py: 进一步改进“积木世界”任务，通过允许多个最优解，测试模型处理多标签分类问题的能力，更真实地反映了规划问题中可能存在的等效最优路径。
    
- generate_blocks_world_fixed_goal_multilabel_fixed_format.py: 这是“积木世界”任务的最终优化版本，通过改进输入表示法，旨在为模型提供一个更清晰、更结构化的学习目标。
    
- generate_checkers_jump_1d.py: 解决一个在一维空间中移动棋子的规划问题，该问题源自苹果公司的一篇著名论文，用于测试大语言模型的推理瓶颈。
    
- generate_river_crossing_puzzle.py: 解决一个经典的约束满足和状态空间搜索问题——“N对伴侣过河”。该任务源自苹果公司的一篇论文，用于揭示大型语言模型在某些类型推理任务上的局限性。
    
- generate_trapping_rain_water_aggregate.py: 这是解决“接雨水”算法问题的初步尝试，旨在测试模型学习一个聚合输出（而非解耦输出）的能力。实验结果表明，要求模型直接输出总和值比输出每个位置的详细信息要困难得多。
    
- generate_trapping_rain_water_decoupled.py: 解决经典的“接雨水”算法问题（LeetCode Hard）。这个任务的成功展示了模型学习需要全局信息的复杂算法的能力，并通过问题解耦的思想，证明了输出格式设计对模型学习效率的巨大影响。
    
- generate_trapping_rain_water_2d.py: 作为一维“接雨水”问题的扩展，解决二维版本的“接雨水”问题。该任务要求模型理解二维空间中的“包围”和“边界”概念，是一个更复杂的全局信息处理挑战。
    
- generate_skyline_max_height_aggregate.py: 这是解决“天际线”问题的初步尝试，要求模型从所有建筑的最终高度中，只预测出那个最高的高度值。此任务用于对比聚合输出和解耦输出的学习难度。
    
- generate_skyline_all_heights_decoupled.py: 测试模型解决一个带有一维空间约束的全局优化问题的能力。问题原型是LeetCode "Max-Height Skyline"。通过解耦输出，要求模型预测每一栋建筑的高度，而非仅仅是最大值。
    
- generate_hanoi_tower_path_strategy_sep_format.py: 这是汉诺塔问题的早期实验脚本，旨在测试模型能否学习最优路径上的策略。它采用了分隔符式的输入格式，并将动作预测为一个6分类问题。
    
- generate_hanoi_tower_global_strategy_fixed_format.py: 作为对早期汉诺塔实验的改进，此脚本采用了对模型更友好的固定槽位输入格式，旨在验证输入表示对学习效率的影响。
    
- generate_hanoi_tower_compare_formats.py: 这是一个对比实验脚本，它为同一个汉诺塔问题生成两种不同的输入格式（分隔符 vs. 固定槽位），用于系统性地评估不同数据表示法对模型学习递归策略的影响。
    
- generate_hanoi_tower_compare_formats_and_strategies.py: 这是一个更全面的汉诺塔对比实验脚本。它不仅生成两种输入格式，还生成两种不同的数据集：一种只包含最优路径上的状态（“路径策略”），另一种包含所有可达状态（“全局策略”），用于探究模型在学习局部最优路径和全局最优策略上的能力差异。
    
- generate_hanoi_tower_build_full_state_graph.py: 这是一个“汉诺塔问题”研究的集大成者，旨在通过多种不同的数据表示和采样策略，深度剖析模型对递归结构的理解能力。它是一个自给自足的数据工厂。
    
- generate_hanoi_tower_sample_from_state_graph.py: 这是一个后处理和采样脚本，它利用generate_hanoi_tower_build_full_state_graph.py生成的完整知识库，来精确地提取特定类型的训练数据子集，例如“扭曲路径”或“最难部分”，用于进行更精细的消融实验。
    
- generate_sokoban_planning_astar.py: 解决经典的“推箱子”（Sokoban）规划问题。
    
- generate_sokoban_planning_full.py: 解决经典的“推箱子”（Sokoban）规划问题。这是一个高难度的AI任务，因为它涉及到在一个巨大的状态空间中进行搜索，并且动作会改变环境的状态。
    
- generate_sokoban_planning_claude_deprecated.py: 这是一个早期的、逻辑更复杂的尝试，但未能稳定地生成高质量数据集。 (已弃用)
    
- generate_min_swaps_for_checkerboard.py: 解决一个高度约束的矩阵重排问题：通过任意交换行和列，将一个0/1矩阵变为“棋盘”模式（相邻元素不同）所需的最少交换次数。
    
- generate_min_flips_for_alternating_binary.py: 测试模型解决一个基于位翻转的字符串优化问题，该问题可以被巧妙地映射为一个滑动窗口问题来求解。
    
- generate_min_swaps_for_checkerboard_v2.py: 解决一个高度约束的矩阵重排问题：通过任意交换行和列，将一个0/1矩阵变为“棋盘”模式（相邻元素不同）所需的最少交换次数。
    
- generate_matrix_flip_strategy.py: 解决一个矩阵优化的经典问题（最大化1的数量）。此版本旨在测试模型能否学习到一个“策略”而非最终结果。
    
- generate_matrix_flip_max_score.py: 测试模型学习一个矩阵优化问题的能力，该问题需要通过两步贪心策略（先行翻转，后列翻转）来达到全局最优。该版本要求模型直接输出最终的聚合结果（分数）。
    
- generate_min_prefix_flips.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。
    
- generate_min_k_bit_flips.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力，并且测试其能否将输入的一部分（k）作为“参数”来指导对另一部分（nums）的处理。
    
- generate_min_k_bit_flips_fixed_k.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。此版本中，环境参数（k=2）是固定的、隐藏的，模型必须从数据中隐式学习。
    
- generate_special_binary_string_recursion.py: 测试模型学习一个递归定义的字符串变换规则的能力。该问题（LeetCode Hard "Special Binary String"）要求对输入进行递归分解和重组。
    
- generate_min_flips_for_chunked_binary.py: 测试模型学习一个基于局部块（chunk）的字符串变换优化问题的能力。
    
- generate_count_connected_components.py: 测试模型对图结构的基本理解，特别是“连通性”这一核心概念。
    
- generate_check_graph_connectivity.py: 这是对模型图论基础能力的又一个核心测试，任务是判断图中任意两点之间是否存在一条路径。
    
- generate_minimize_malware_spread.py: 解决一个基于图论的病毒传播优化问题（LeetCode Hard "Minimize Malware Spread"）。模型需要理解图的连通性，并评估移除不同节点对全局传播的影响。
    
- generate_count_islands_1d.py: 测试模型在一维序列上进行模式识别和计数的能力。
    
- generate_largest_island_by_adding_one_cell.py: 解决一个涉及图遍历和全局优化的算法问题(LeetCode 827)。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。
    
- generate_largest_island_by_adding_one_cell_v2.py: 解决一个涉及图遍历和全局优化的算法问题(LeetCode 827)。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。
    
- generate_find_articulation_points.py: 测试模型识别图的“割点”（Articulation Point）或“桥”（Bridge）的能力，这是一个图论中的重要概念。
    
- generate_nim_game_zeckendorf.py: 这个实验旨在测试我的范式能否学习一个基于复杂数论（齐肯多夫表示法）的非直观博弈论问题。它脱离了简单的模式匹配，需要模型理解更深层次的数学结构。
    
- generate_longest_subsequence_constrained.py: 测试模型处理一个混合了序列操作和数值约束的复杂优化问题的能力。
    
- generate_treasure_hunt_tsp.py: 解决一个复杂的状态空间搜索问题，它结合了图的遍历（BFS）和组合优化（状态压缩DP），是算法竞赛中的经典难题。
    
- generate_freedom_trail_dp.py: 测试模型学习解决一个需要动态规划和路径回溯的复杂优化问题的能力。
    
- generate_sum_of_subset_with_mask.py: 测试模型根据一个二进制掩码从一个集合中选择元素并执行聚合操作（求和）的能力。
    
- generate_sudoku_6x6.py: 测试模型在处理有强约束满足问题（Constraint Satisfaction Problem）——数独——上的能力。
    
- generate_valid_parentheses_path_random_deprecated.py: 这是解决“合法括号路径”问题的早期尝试。 (早期探索/已弃用)
    
- generate_valid_parentheses_path_balanced.py: 解决一个二维网格上的路径查找问题，但路径的合法性受到栈式结构（括号匹配）的约束。
    
- generate_sat_solver_text.py: 测试模型解决一个标志性的NP完全问题——布尔可满足性（SAT）问题的能力。
    
- generate_sat_solver_compact_text.py: 这是对 generate_sat_solver_text.py 的一个变种，采用了不同的输入编码格式来解决同样的3-SAT问题。
    
- generate_point_in_polygon.py: 测试模型学习一个计算几何中的经典算法——射线法（Ray Casting Algorithm）——的能力。
    
- generate_shortest_path_in_matrix_bfs.py: 测试模型在一个二维网格中，基于经典的广度优先搜索（BFS）算法寻找最短路径的能力。
    
- generate_sudoku_4x4_stepwise_deprecated.py: 旨在测试模型进行“步进式”（stepwise）推理的能力。 (已弃用)
    
- generate_tiling_problem_deprecated.py: 旨在测试模型解决一个经典的平铺覆盖优化问题的能力，这是一个NP-hard问题。 (已弃用)
    
- generate_hanoi_tower_twisted_path_deprecated.py: 此脚本意图生成一个汉诺塔问题的“扭曲路径”数据集。 (已弃用)
    
- generate_checkers_jump_1d_v2.py: 解决一维空间中的棋子交换规划问题，该问题被用于揭示大型语言模型在某些类型推理任务上的局限性。
    

## C. 图像输出符号 (Image to Symbol)

- generate_checkerboard_to_binary.py: 这是一个基础的视觉到符号转换任务，用于测试模型从原始像素数据中解码结构化信息的能力。
    
- generate_line_angle_to_vector.py: 测试模型从图像中提取精确几何信息（角度）的能力，这是一个比简单识别棋盘格更高级的视觉推理任务。
    
- generate_count_shapes_from_image.py: 测试模型同时进行物体识别（形状）、属性识别（颜色）和计数（聚合）的多重视觉任务能力。
    
- generate_maze_symbolic_to_image.py: 将符号化的迷宫路径规划数据集转换为图像格式，以测试视觉模型（如CNN、ViT）直接从像素进行路径规划的能力。
    
- generate_sokoban_symbolic_to_image_no_labels.py: 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）仅转换为图像格式，用于纯视觉任务或作为更复杂数据处理的中间步骤。
    
- generate_sokoban_symbolic_to_image_with_labels.py: 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）转换为一个完整的图像分类数据集，以供计算机视觉模型（如ViT, Swin Transformer）进行训练。
     
- generate_cellular_automata_image_and_label.py: 通用数据集生成器。为元胞自动机（CA）任务同时生成图像格式（Img2Img）和符号格式（Img2Label）的数据，支持多进程加速。
     
- generate_trapping_rain_water_image_to_symbol.py: 专用数据集生成器。为“接雨水”问题生成图像格式的数据，输入是柱子高度的网格图，输出是雨水量的符号标签。

## D. 图像推理 (Image to Image)

- generate_triangle_to_incircle.py: 这是展示“用梯度下降雕刻精确规则”的一个标志性实验。它测试模型能否学习到一个纯粹的、非平凡的几何构造规则（三角形内切圆）。
    
- generate_polygon_to_symmetry_axis.py: 测试模型从一个完整的对称图形中反向推断出其隐含的对称轴的能力。
    
- generate_triangle_to_centroid.py: 测试模型学习另一个基础几何概念——重心的能力。
    
- generate_triangle_to_tessellation.py: 这是我们范式能力的一个标志性展示。它测试模型能否学习一种无限的、基于晶格的生成规则。由于镶嵌图案的全局关联性和细节的精确性，它有力地排除了模型仅仅是靠“插值”或“记忆”来解决问题的可能性。
    
- generate_game_of_life_image_to_image.py: 这是二维元胞自动机的image-to-image版本，测试模型能否直接在像素空间中执行基于局部规则的演化。
    
- generate_projectile_motion_simulation.py: 测试模型学习一个简单的动态物理过程的能力。这要求模型从初始条件（位置和速度向量）推断出整个时空轨迹。
    
- generate_snell_refraction_simulation.py: 测试模型学习基础物理定律（斯涅尔折射定律）的能力。
    
- generate_snell_refraction_with_contextual_index.py: 测试模型学习基础物理定律（斯涅尔折射定律）的能力，并且要求模型能从图像的上下文信息（背景颜色）中推断出物理参数（折射率）。
    
- generate_cellular_automata_spatial_conditional.py: 测试模型在单一模态（图像）内部分区和解析“指令”与“数据”的能力，是一种“伪多模态”或“空间条件化”的实验。
    
- generate_trapping_rain_water_visualizer.py: 这是一个**数据转换与可视化**脚本。它的作用是将已经生成的、符号化的“接雨水”数据集转换为一个image-to-image格式的数据集，以便用视觉模型来解决同一个问题。
    
- generate_shortest_path_in_tree_deprecated.py: 这是一个早期的实验，旨在测试模型从图像中寻找图上最短路径的能力。 (早期探索/已弃用)
    
- generate_shortest_distance_between_triangles.py: 测试模型在包含多个对象的情况下，进行全局几何关系（最短距离）推理的能力。
    
- generate_reaction_diffusion_deprecated.py: 该脚本用于模拟一个反应-扩散系统，以生成复杂的、类似分形的“雪花”图案。 (探索性/已弃用)
    
- generate_cellular_automata_multimodal_deprecated.py: 生成一个真正的多模态数据集，用于训练能够同时理解图像输入和文本指令的模型。 (已弃用)
    
- generate_cellular_automata_1d_to_grid_image_interp.py: 该脚本旨在设计一个“逻辑/感知混合”任务，用以证明神经网络的规则学习能力和内插能力并非互斥，而是可以一体化地在单个任务中得到体现。它迫使模型必须同时“看穿”输入的连续灰度值以执行离散的逻辑推理，并记住这些灰度值以完成最终的连续值映射。
    

## E. 文字输出图像 (Text to Image)

- generate_coords_to_triangle.py: 这是一个基础的符号到几何的渲染任务，测试模型将抽象的坐标信息转换为具体像素形状的能力。
    
- generate_cellular_automata_1d_to_grid_image.py: 测试模型能否直接将一维的符号计算结果“渲染”成结构化的二维图像。
    
- generate_triangle_coords_to_tessellation.py: 这是一个高级的、混合了符号指令和几何生成规则的推理任务。
    
- generate_cube_rotation_matplotlib_deprecated.py: 旨在测试模型从抽象的姿态参数（旋转角度）推理并渲染出三维物体正确视图的能力。 (早期探索版本)
    
- generate_cube_rotation_pillow_v1.py: 旨在测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，采用了更底层的、渲染效果更精确的技术路线。 (技术升级版本)
    
- generate_cube_rotation_pillow_with_anchor.py: 测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，并通过引入“视觉锚点”来辅助模型学习。 (论文中使用的最终版本)
    
- generate_cube_rotation_pillow_wireframe.py: 测试模型在更稀疏的视觉输入下，能否仅通过线框和锚点信息来学习3D旋转。 (变体实验版本)
    

## F. 物理模拟 (Physics Simulation - Image Paradigm)

- generate_catenary_curve_simulation_deprecated.py: 这是我早期探索悬链线问题的脚本，旨在测试模型学习由物理定律确定的非线性曲线的能力。
    
- generate_catenary_curve_from_points.py: 测试模型学习由物理定律（最小势能原理）唯一确定的非线性曲线（悬链线）的能力。
    
- generate_orbital_path_from_initial_state.py: 测试模型学习更复杂物理定律（开普勒定律/万有引力定律）的能力。
    

## G. ARC-AGI 探索 (ARC-AGI Exploration)

- generate_arc_contextual_color_swap.py: 测试模型从图像的局部“上下文”或“示例”中学习规则，并将其应用到同一图像的全局数据的能力。这直接模仿了ARC-AGI测试的核心理念。
    
- generate_arc_find_cross_pattern.py: 测试模型在包含大量噪音的情况下进行视觉模式识别（或可称作“目标检测”）的能力。
    
- generate_arc_find_odd_one_out.py: 测试模型执行一个复杂的“异类发现”（Find the Odd One Out）元推理任务。模型需要逐行进行模式比较，找出特例，并将其重新组合到输出中。
    
- generate_arc_connect_colored_pairs.py: 测试模型在同一图像中识别多个独立“连接任务”并理解一种隐含的“图层”或“绘制优先级”规则的能力。
    
- generate_arc_conditional_perpendicular_lines.py: 测试模型根据物体的**属性（颜色）**和**全局参照物（边界线、图像边缘）**来执行不同几何操作的能力。
    
- generate_arc_column_projection.py: 测试模型识别复杂的上下文关系（“在...下方且在...范围内”）并执行条件性列操作的能力。
    
- generate_arc_procedural_spiral.py: 测试模型执行一个迭代的、程序性的生成算法的能力。模型需要理解指令、跟踪状态（当前位置、方向、长度）并循环执行。
    
- generate_arc_fractal_stamping.py: 测试模型理解和执行递归或分形生成规则的能力。模型需要将输入图案本身作为一个“笔刷”，根据输入图案中的“指令”进行重复绘制。
    
- generate_arc_flood_fill.py: 测试模型执行经典的“洪水填充”（Flood Fill）或“油漆桶”算法的能力。
    
- generate_arc_layered_fill.py: 测试模型理解一个程序性极强的、依赖于拓扑距离和条件判断的复杂填充算法。
    
- generate_arc_fluid_simulation.py: 测试模型在图像空间中学习和模拟一个具有特定规则的流体动态过程的能力。
    
- generate_arc_periodic_conditional_fill.py: 测试模型学习一个复杂的、带有周期性和特殊case的条件格式化规则的能力。
    
- generate_arc_fill_square_holes.py: 测试模型进行多步视觉推理的能力：首先识别“空洞”，然后判断其几何属性（是否为正方形），最后根据判断结果进行操作。
    
- generate_arc_conditional_recoloring.py: 测试模型理解视觉图层和进行条件性对象属性修改的能力。
    
- generate_arc_sort_by_length_remap_position.py: 测试模型执行一个“属性-位置解耦与重映射”的复杂排序任务。
    
- generate_arc_jigsaw_puzzle_simple.py: 测试模型解决一个视觉匹配与变换问题的能力（早期版本），其中拼图块的尺寸是唯一的，可作为匹配捷径。
    
- generate_arc_jigsaw_puzzle_advanced.py: 测试模型解决一个复杂的视觉匹配与变换问题的能力，该版本要求模型必须真正地根据形状（而非尺寸）来进行匹配。
    
- generate_arc_connect_path_by_sequence.py: 测试模型解析外部指令序列，并据此在图像中执行多步、有状态的路径连接任务的能力。
    
- generate_arc_reflection_simulation_deprecated.py: 旨在测试模型理解复杂的基于物理光学的规则，包括射线发射、碰撞检测、角度反射和颜色变换。 (已废弃)
    

## H. 逆推规则 (Inverse Rule Inference)

- generate_cellular_automata_inverse_rule.py: 这个实验是检验模型**逆向推理**（Inverse Reasoning）能力的第一个尝试。我的问题是：如果模型能从规则正向推出结果，那么它能否从“输入-输出”对中反向推断出其背后的规则？
    
- generate_cellular_automata_inverse_rule_and_steps.py: 这是在实现“唯一解”版本之前的一个早期版本，它同样旨在让模型学习预测规则和迭代次数。
    
- generate_cellular_automata_inverse_rule_and_steps_unique.py: 这是对逆向推理任务的一次重大升级。我不仅要求模型推断出**什么**规则被应用了，还要推断出它被**应用了多少次**。
    

## I. 鲁棒性测试（Robustness Analysis）

- generate_cellular_automata_1d_perturbed.py：该脚本旨在系统性地测试“神经雕刻”范式在面对不完美数据时的鲁棒性。通过向输入（模拟观测噪声）和输出（模拟标签噪声）中引入可控的随机扰动，它探索了模型性能从理想规则世界向嘈杂现实世界过渡的连续谱。
    

## J. 训练脚本（Training Script）

- train_tiny_transformer.py: 用于符号到符号任务，训练一个从零开始构建的、小型的Transformer模型。
    
- train_swin_image2text.py: 用于图像到符号任务，全量微调一个Swin Transformer进行多标签分类。
    
- train_unet.py: 用于纯粹的图像到图像任务，训练一个标准的U-Net模型。
    
- train_text2image.py: 用于符号到图像任务，训练一个从零构建的Transformer编码器和一个U-Net解码器的组合。
    
- train_qwen2_text2image.py: 用于符号到图像任务，使用LoRA高效微调Qwen2作为编码器，结合U-Net解码器，并由Hugging Face Trainer管理。
    
- train_mlp.py: 符号到符号任务，使用一个巨大的MLP作为性能基准。
    
- train_lstm.py: 符号到符号任务，使用LSTM来测试其时序记忆能力。
    
- train_convnext.py: 图像到符号任务，使用先进的CNN架构ConvNeXt作为视觉特征提取器。
    
- train_diffusion.py: 图像到图像任务，使用条件Diffusion模型进行生成式推理。
    
- train_image2image.py: 核心图像到图像任务脚本，使用Swin Transformer + U-Net解码器（即Swin-Unet）。
    

## K. 工具脚本（Utility Script）

- eval_hanoi.py: 实验验证工具。用于验证大语言模型生成的汉诺塔问题解法字符串是否符合游戏规则并最终达成目标。
    
