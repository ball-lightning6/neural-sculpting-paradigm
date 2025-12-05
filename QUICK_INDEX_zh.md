## A: 符号数学逻辑 (Symbolic Math Logic)

- symbolic_math_logic/generate_conditional_add_subtract.py: 脚本生成两个N-bit整数的加法或减法（取绝对值）问题。它包含两种模式：

- symbolic_math_logic/generate_add_binary_modulo.py: 这是一个早期的基础算术实验，用于测试模型学习模加法（或称“截断加法”）的能力，这种运算常见于计算机硬件的定宽整数运算。

- symbolic_math_logic/generate_multiply_binary.py: 作为二进制算术能力的一个基准测试，生成N-bit整数的乘法数据集。

- symbolic_math_logic/generate_multiply_binary_no_carry_phase1.py: 这是乘法“解耦”实验的第一阶段。旨在测试模型是否能学会乘法的第一步：无进位的按位相乘和错位相加，将一个复杂的乘法问题分解为一个更简单的计数问题。

- symbolic_math_logic/generate_multiply_binary_from_counts_phase2.py: 这是乘法“解耦”实验的第二阶段。旨在验证一个独立的模型能否学会处理复杂的进位逻辑，即从一个“无进位计数向量”中计算出最终的二进制乘积。

- symbolic_math_logic/generate_add_hexadecimal.py: 对比模型在不同符号系统下的学习能力。此脚本旨在验证模型学习的是加法这一抽象数学概念，还是仅仅是特定于二进制符号的模式。

- symbolic_math_logic/generate_multiply_decimal.py: 测试模型处理非二进制符号输入（0-9字符），并执行算术运算（乘法）的能力。

- symbolic_math_logic/generate_add_binary_with_position_shuffle.py: 这是“语义洗牌”系列实验中的“位置洗牌”部分。它旨在验证模型是否依赖于输入的固定空间结构，还是能学习到与位置无关的抽象关系。

- symbolic_math_logic/generate_symbol_add_shuffle_dataset.py: 这是我们研究中一项**关键的决定性实验**，旨在彻底分离模型的"表面模式匹配"能力和"抽象结构学习"能力。

- symbolic_math_logic/generate_add_hidden_constant.py: 测试模型在没有任何直接线索的情况下，从大量样本中**推断出隐藏规则或参数**的能力。这类似于一个简化的系统辨识（System Identification）问题。

- symbolic_math_logic/generate_multitask_alu.py: 此脚本旨在构建一个模拟**算术逻辑单元 (ALU)** 的多任务学习场景。它测试模型能否在一次前向传播中，对同一份输入并行执行多种不同的、定义明确的计算任务。

- symbolic_math_logic/generate_modulo_operation.py: 探究模型学习模运算（Modulo Operation）的能力，这是一个在数论和计算机科学中至关重要但具有“循环”性质的运算。

- symbolic_math_logic/generate_rsa_encryption.py: 测试模型学习高度非线性的、在计算上被认为是“困难”的确定性规则的能力。RSA加密是一个典型的例子。

- symbolic_math_logic/generate_deduction_chain_text.py: 生成多步逻辑推理任务，测试模型执行符号演绎（deduction）的能力，类似于一个简化的定理证明器。

- symbolic_math_logic/generate_deduction_multirule_text.py: 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。

- symbolic_math_logic/generate_deduction_multirule_text_v2.py: 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。

- symbolic_math_logic/generate_deduction_multirule_binary.py: 这是对多规则推理任务的**格式优化**版本，旨在测试紧凑的二进制编码是否比稀疏的文本格式更有利于模型学习。

- symbolic_math_logic/generate_deduction_fixed_depth.py: 测试模型在有明确结构、固定深度的符号演绎任务中的多步推理能力。

- symbolic_math_logic/generate_function_composition.py: 测试模型学习函数组合（Function Composition）的能力。这要求模型像解释器一样，按顺序解析指令并对数据进行变换。

- symbolic_math_logic/generate_count_set_bits.py: 测试模型执行全局聚合操作的能力。与局部规则不同，计数需要模型综合整个输入序列的信息。

- symbolic_math_logic/generate_sum_pattern_positions.py: 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。

- symbolic_math_logic/generate_sum_pattern_positions_v2.py: 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。

- symbolic_math_logic/generate_sum_pairwise_hamming_distance.py: 测试模型执行一个需要两层嵌套聚合操作的复杂任务。模型需要先在**每个比特位**上进行全局统计，然后再将**所有比特位**的结果累加起来。

- symbolic_math_logic/generate_circular_shift.py: 测试模型学习位移操作的能力，特别是循环位移（circular shift），这是密码学和底层编程中的常见操作。

- symbolic_math_logic/generate_multiply_matrix_3x3.py: 测试模型学习结构化代数运算（矩阵乘法）的能力，这比简单的标量运算需要更复杂的“数据路由”和“乘积累加”能力。

- symbolic_math_logic/generate_evaluate_boolean_expression_text.py: 测试模型解析一个简单的领域特定语言（DSL）并执行求值的能力，这比前面固定结构的表达式求值更进了一步。

- symbolic_math_logic/generate_evaluate_arithmetic_expression.py: 训练模型执行符号表达式的求值任务，这要求模型理解运算符优先级（通过树状结构隐式表达）、变量替换和算术运算。

- symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply.py: 这是对generate_evaluate_arithmetic_expression.py的简化版本，旨在通过移除乘法运算来降低学习难度，以测试模型在更基础的算术表达式求值上的能力。

- symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply_small_range.py: 这是在前一个“无乘法”版本基础上的进一步简化，通过缩小数值范围来进一步降低学习难度，用于精确诊断模型在最简单表达式求值任务上的性能瓶颈。

- symbolic_math_logic/generate_check_boolean_equivalence.py: 测试模型对布尔代数逻辑等价性的判断能力。这是一个抽象的符号推理任务，要求模型理解表达式的结构和布尔运算法则。

- symbolic_math_logic/generate_polynomial_shift_coefficients.py: 测试模型学习一个抽象的代数变换规则的能力。这个任务需要模型理解多项式展开的内在结构。

- symbolic_math_logic/generate_convolution_2d.py: 测试模型学习二维卷积（Conv2D）这一基本图像处理操作的能力，并探究其是否能从输入输出对中推断出隐藏的固定规则（即卷积核本身）。

- symbolic_math_logic/generate_simple_block_cipher.py: 测试模型“破解”或学习一个简单但非平凡的自定义加密算法的能力。该任务代表了一类复杂的、具有高度混沌和雪崩效应的符号变换规则。

- symbolic_math_logic/generate_sin_function_float32.py: 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，使用标准的32位浮点数格式进行输入和输出。

- symbolic_math_logic/generate_sin_function_float64_to_int12_deprecated.py: 这是对sin函数拟合任务的另一种编码尝试，旨在探索使用更高精度的浮点输入和更低精度的量化二进制输出对学习效果的影响。

- symbolic_math_logic/generate_sin_function_float32_to_quantized_int.py: 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，并探索不同输入/输出编码方案对学习效果的影响。

- symbolic_math_logic/generate_multiply_binary_modulo.py: 作为基础算术实验的一部分，测试模型对截断乘法（或称模乘法）的掌握能力。

- symbolic_math_logic/generate_explainable_two_step_calculation.py: 测试模型输出计算“中间步骤”或“思维链”的能力，是“功能性可解释性”的一个直接验证。

- symbolic_math_logic/generate_min_swaps_for_checkerboard.py: 解决LeetCode 782题"变为棋盘"([https://leetcode.cn/problems/transform-to-chessboard/](https://leetcode.cn/problems/transform-to-chessboard/)) - 通过任意交换行和列，将一个0/1矩阵变为"棋盘"模式（相邻元素不同）所需的最少交换次数。

- symbolic_math_logic/generate_min_flips_for_alternating_binary.py: 测试模型解决一个基于位翻转的字符串优化问题，该问题可以被巧妙地映射为一个滑动窗口问题来求解。

- symbolic_math_logic/generate_min_swaps_for_checkerboard_v2.py: 解决LeetCode 1536题"排布二进制网格的最少交换次数"([https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/](https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/)) - 通过交换相邻行，将一个二进制网格变为上三角形式（主对角线以上全为0）所需的最少交换次数。

- symbolic_math_logic/generate_min_prefix_flips.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。

- symbolic_math_logic/generate_min_flips_for_chunked_binary.py: 测试模型学习一个基于局部块（chunk）的字符串变换优化问题的能力。

- symbolic_math_logic/generate_largest_island_by_adding_one_cell.py: 解决一个涉及图遍历和全局优化的算法问题([LeetCode 827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/))。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。

- symbolic_math_logic/generate_largest_island_by_adding_one_cell_v2.py: 解决一个涉及图遍历和全局优化的算法问题([LeetCode 827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/))。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。

- symbolic_math_logic/generate_sat_solver_text.py: 测试模型解决一个标志性的NP完全问题——布尔可满足性（SAT）问题的能力。

- symbolic_math_logic/generate_sat_solver_compact_text.py: 这是对 symbolic_math_logic/generate_sat_solver_text.py 的一个变种，采用了不同的输入编码格式来解决同样的3-SAT问题。


## B: 算法学习 (Algorithm Learning)

- algorithms/generate_sort_integers.py: 测试模型执行基本排序算法的能力，这是一个非局部的、需要对输入元素进行比较和重排的经典算法任务。

- algorithms/generate_edit_distance.py: 测试模型学习解决动态规划问题的能力。编辑距离是一个典型的DP问题，需要模型在概念上构建一个二维的求解矩阵。 [LeetCode 72. Edit Distance](https://leetcode.com/problems/edit-distance/description/)

- algorithms/generate_edit_distance_explainable.py: 这是“功能性可解释性”的一个核心实验。它要求模型不仅给出最终答案（编辑距离），还要输出达成答案的完整“思维链”（编辑过程）。 [LeetCode 72. Edit Distance (Explainable / Path Construction Version)](https://leetcode.com/problems/edit-distance/description/)

- algorithms/generate_maze_random_walls.py: 测试模型在随机生成的“多孔”迷宫中的基础寻路能力。

- algorithms/generate_maze_dense.py: 测试模型在复杂的、类似人类设计的“稠密”迷宫中进行路径规划的能力，这比随机墙壁迷宫更具挑战性。

- algorithms/generate_blocks_world_arbitrary_goal.py: 解决经典的“积木世界”（Blocks World）规划问题。该问题作为衡量大语言模型推理能力的标准任务之一，在苹果公司发布的著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中被重点讨论，作为评估模型在状态空间搜索和规划能力上的基准测试。该研究通过积木问题、过河问题、汉诺塔等一系列经典规划任务，系统性地揭示了当前大语言模型在精确符号推理和状态空间规划上的根本局限性。本脚本精确实现了论文中的"积木世界"通用版本，允许指定任意的初始状态和终止状态，作为验证神经网络在复杂规划问题上推理能力的核心对照实验。

- algorithms/generate_blocks_world_fixed_goal.py: 这是对“积木世界”任务的简化版本，通过固定目标状态（所有积木有序地堆叠在第一个柱子上），旨在测试模型在目标明确、状态空间更结构化的情况下的学习能力。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究指出，即使对于这样目标状态明确的规划问题，大语言模型在状态空间搜索和最优策略学习方面仍然存在显著困难。本脚本实现了论文中的积木世界简化版本，通过固定目标状态来降低任务复杂度，作为研究目标明确性对模型推理性能影响的对照实验。

- algorithms/generate_blocks_world_fixed_goal_multilabel.py: 进一步改进“积木世界”任务，通过允许多个最优解，测试模型处理多标签分类问题的能力，更真实地反映了规划问题中可能存在的等效最优路径。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究强调，现实世界中的规划问题往往存在多个等价最优解，这对模型的多义性推理能力提出了更高要求。本脚本在固定目标版本基础上进一步改进，为每个状态找到所有能导向最优路径的动作，生成多热（multi-hot）编码输出，作为研究神经网络处理多最优解规划问题的对照实验。

- algorithms/generate_blocks_world_fixed_goal_multilabel_fixed_format.py: 这是“积木世界”任务的最终优化版本，通过改进输入表示法，旨在为模型提供一个更清晰、更结构化的学习目标。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究指出，输入表示法对模型的学习效率和最终性能有着至关重要的影响。本脚本在多标签版本基础上进一步优化，采用固定槽位（fixed-slot）表示法替代可变长度输入，消除了序列化带来的复杂性，为Transformer等架构提供了更友好的结构化输入。这使其成为研究输入表示法对神经符号推理性能影响的对照实验。

- algorithms/generate_checkers_jump_1d.py: 解决一维空间中的棋子交换规划问题。该问题作为衡量大语言模型推理能力的标准任务之一，在苹果公司发布的著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中被重点讨论，作为评估模型在状态空间搜索和规划能力上的基准测试。该研究通过积木问题、过河问题、汉诺塔等一系列经典规划任务，系统性地揭示了当前大语言模型在精确符号推理和状态空间规划上的根本局限性。本脚本精确实现了论文中的"跳棋交换"通用版本，作为验证神经网络在复杂规划问题上推理能力的核心对照实验。

- algorithms/generate_river_crossing_puzzle.py: 解决一个经典的约束满足和状态空间搜索问题——“N对伴侣过河”。该问题要求在满足“任何女性不能在没有其伴侣在场的情况下，与其他男性共处”的约束下，将所有人运到对岸。该任务源自苹果公司的一篇著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity"，该研究通过过河问题、汉诺塔、跳棋等基准测试，揭示了大型语言模型在某些类型推理任务上的根本局限性。本脚本精确复现了论文中的"N对伴侣过河"问题，作为验证神经网络符号推理能力的对照实验。

- algorithms/generate_trapping_rain_water_aggregate.py: 这是解决“接雨水”算法问题的初步尝试，旨在测试模型学习一个聚合输出（而非解耦输出）的能力。实验结果表明，要求模型直接输出总和值（一个单一的聚合数字）比输出每个位置的详细信息要困难得多。**这成为一个关键的对比实验，证明了输出格式设计对模型学习效率的系统性影响**。对应 LeetCode 题目：[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)。该脚本精确实现了LeetCode原题的聚合输出格式（只输出总量），与`generate_trapping_rain_water_decoupled.py`（解耦输出，输出每个位置的水量）形成鲜明对照，用于研究输出表示对神经网络学习难度的影响，验证了论文中"解耦加速收敛"的核心发现。

- algorithms/generate_trapping_rain_water_decoupled.py: 解决经典的“接雨水”算法问题（LeetCode Hard [#42](https://leetcode.com/problems/trapping-rain-water/)）。这个任务的成功展示了模型学习需要全局信息（如全局最高点）的复杂算法的能力，并通过**问题解耦**的思想，证明了输出格式设计对模型学习效率的巨大影响。

- algorithms/generate_trapping_rain_water_2d.py: 作为一维“接雨水”问题的扩展，解决二维版本的“接雨水”问题。该任务要求模型理解二维空间中的“包围”和“边界”概念，是一个更复杂的全局信息处理挑战。对应 LeetCode 题目：[407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)（困难题）。

- algorithms/generate_skyline_max_height_aggregate.py: 这是解决“天际线”问题的初步尝试，要求模型从所有建筑的最终高度中，只预测出那个最高的高度值。此任务用于对比聚合输出和解耦输出的学习难度。对应 LeetCode 题目：[1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/)（聚合输出版本）。

- algorithms/generate_skyline_all_heights_decoupled.py: 测试模型解决一个带有一维空间约束的全局优化问题的能力。问题原型是LeetCode [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/)（解耦输出版本）。该题目要求：给定N个建筑的位置和限高，相邻建筑高度差不超过1，求每个建筑能达到的最大高度。通过解耦输出，要求模型预测每一栋建筑的高度，而非仅仅是最大值，作为研究输出格式设计对模型学习效率影响的对照实验。

- algorithms/generate_hanoi_tower_path_strategy_sep_format.py: 作为对比组A，此脚本用于生成汉诺塔最优路径策略数据集，采用**分隔符+二进制编码**的输入格式。

- algorithms/generate_hanoi_tower_path_strategy_fixed_format.py: 作为对比组B，此脚本同样生成汉诺塔最优路径策略数据集，但采用**结构化的固定槽位**输入格式（重命名自原来的`global`脚本以反映真实逻辑），用于验证数据表示对学习效率的影响。

- algorithms/generate_hanoi_tower_compare_formats.py: 这是一个对比实验脚本，它为同一个汉诺塔问题生成两种不同的输入格式（分隔符 vs. 固定槽位），用于系统性地评估不同数据表示法对模型学习递归策略的影响。

- algorithms/generate_hanoi_tower_compare_formats_and_strategies.py: 这是一个更全面的汉诺塔对比实验脚本。它不仅生成两种输入格式，还生成两种不同的数据集：一种只包含最优路径上的状态（“路径策略”），另一种包含所有可达状态（“全局策略”），用于探究模型在学习局部最优路径和全局最优策略上的能力差异。

- algorithms/generate_hanoi_tower_build_full_state_graph.py: 这是一个“汉诺塔问题”研究的集大成者，旨在通过多种不同的数据表示和采样策略，深度剖析模型对递归结构的理解能力。它是一个自给自足的数据工厂。该研究对应于论文中引用的 Apple Research 相关工作，重点在于通过完整状态图分析模型对递归结构的掌握。

- algorithms/generate_hanoi_tower_sample_from_state_graph.py: 这是一个后处理和采样脚本，它利用generate_hanoi_tower_build_full_state_graph.py生成的完整知识库，来精确地提取特定类型的训练数据子集，例如“扭曲路径”（twisted path）或“最难部分”，用于进行更精细的消融实验。

- algorithms/generate_sokoban_planning_astar.py: 解决经典的“推箱子”（Sokoban）规划问题。这个任务比简单的路径规划更难，因为它涉及到改变环境状态（箱子位置），状态空间巨大。

- algorithms/generate_sokoban_planning_full.py: 解决经典的“推箱子”（Sokoban）规划问题。这是一个高难度的AI任务，因为它涉及到在一个巨大的状态空间中进行搜索，并且动作会改变环境的状态。

- algorithms/generate_sokoban_planning_claude_deprecated.py: (已弃用) 这是一个早期的、逻辑更复杂的尝试，但未能稳定地生成高质量数据集，已被更可靠的generate_sokoban_planning_full.py取代。

- algorithms/generate_matrix_flip_strategy.py: 解决一个矩阵优化的经典问题（Score After Flipping Matrix）。此版本旨在测试模型能否学习到一个"策略"而非最终结果。对应 LeetCode 题目：[861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- algorithms/generate_matrix_flip_max_score.py: 测试模型学习一个矩阵优化问题的能力，该问题需要通过两步贪心策略（先行翻转，后列翻转）来达到全局最优。该版本要求模型直接输出最终的聚合结果（分数）。对应 LeetCode 题目：[861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- algorithms/generate_min_k_bit_flips.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力，并且测试其能否将输入的一部分（k）作为“参数”来指导对另一部分（nums）的处理。对应 LeetCode 题目：[995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)

- algorithms/generate_min_k_bit_flips_fixed_k.py: 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。此版本中，环境参数（k=2）是固定的、隐藏的，模型必须从数据中隐式学习。这是一个对照实验，用于对比k可变的版本。对应 LeetCode 题目：[995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)（k固定版）

- algorithms/generate_special_binary_string_recursion.py: 测试模型学习一个递归定义的字符串变换规则的能力。该问题对应 LeetCode 题目：[761. Special Binary String](https://leetcode.com/problems/special-binary-string/)（困难题）。题目要求：一个特殊的二进制序列具有相等的0和1数量，且任何前缀中1不少于0；可以通过交换相邻的特殊子串来得到字典序最大的结果。本脚本精确实现了该问题的递归分解和字典序排序算法，作为测试神经网络学习复杂递归规则能力的基准实验。

- algorithms/generate_count_connected_components.py: 测试模型对图结构的基本理解，特别是“连通性”这一核心概念 (对应 LeetCode 323. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/)，注：此为会员题)。

- algorithms/generate_check_graph_connectivity.py: 这是对模型图论基础能力的又一个核心测试，任务是判断图中任意两点之间是否存在一条路径 (对应 LeetCode 1971. [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/))。

- algorithms/generate_minimize_malware_spread.py: 解决一个基于图论的病毒传播优化问题（LeetCode Hard "Minimize Malware Spread"）。模型需要理解图的连通性，并评估移除不同节点对全局传播的影响。本脚本提供两种输出格式，用于对比哪种更容易学习。对应 LeetCode 题目：[924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)

- algorithms/generate_count_islands_1d.py: 测试模型在一维序列上进行模式识别和计数的能力。

- algorithms/generate_find_articulation_points.py: 测试模型识别图的“割点”（Articulation Point）或“桥”（Bridge）的能力，这是一个图论中的重要概念。 [LeetCode 1568. Minimum Number of Days to Disconnect Island](https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/)

- algorithms/generate_nim_game_zeckendorf.py: 这个实验旨在测试我的范式能否学习一个基于复杂数论（齐肯多夫表示法）的非直观博弈论问题。它脱离了简单的模式匹配，需要模型理解更深层次的数学结构。

- algorithms/generate_longest_subsequence_constrained.py: 测试模型处理一个混合了序列操作和数值约束的复杂优化问题的能力。对应 LeetCode 题目：[2311. Longest Binary Subsequence Less Than or Equal to K](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/)

- algorithms/generate_treasure_hunt_tsp.py: 解决一个复杂的状态空间搜索问题，它结合了图的遍历（BFS）和组合优化（状态压缩DP），是算法竞赛中的经典难题。

- algorithms/generate_freedom_trail_dp.py: [LeetCode 514. Freedom Trail](https://leetcode.com/problems/freedom-trail/description/) 测试模型学习解决一个需要动态规划和路径回溯的复杂优化问题的能力。

- algorithms/generate_sum_of_subset_with_mask.py: 测试模型根据一个二进制掩码从一个集合中选择元素并执行聚合操作（求和）的能力。这是一个创新性的对照实验，旨在验证**任务结构本身**（而非模型容量）对神经网络可学习性的关键影响。

- algorithms/generate_sudoku_6x6.py: 测试模型在处理有强约束满足问题（Constraint Satisfaction Problem）——数独——上的能力。

- algorithms/generate_valid_parentheses_path_random_deprecated.py: (早期探索/已弃用) 这是解决“合法括号路径”问题的早期尝试。

- algorithms/generate_valid_parentheses_path_balanced.py: 解决一个二维网格上的路径查找问题，但路径的合法性受到栈式结构（括号匹配）的约束。这是一个算法和逻辑约束结合的复杂任务 (LeetCode Hard [#2267](https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/) "Check if There Is a Valid Parentheses String Path")。

- algorithms/generate_point_in_polygon.py: 测试模型学习一个计算几何中的经典算法——射线法（Ray Casting Algorithm）——的能力。

- algorithms/generate_shortest_path_in_matrix_bfs.py: 测试模型在一个二维网格中，基于经典的广度优先搜索（BFS）算法寻找最短路径的能力。

- algorithms/generate_sudoku_4x4_stepwise_deprecated.py: (废弃) 旨在测试模型进行“步进式”（stepwise）推理的能力，即在每个状态下只预测下一步的最优动作，而不是一次性输出完整解。

- algorithms/generate_tiling_problem_deprecated.py: (已弃用) 旨在测试模型解决一个经典的平铺覆盖优化问题的能力，这是一个NP-hard问题。

- algorithms/generate_hanoi_tower_twisted_path_deprecated.py: (废弃) 此脚本意图生成一个汉诺塔问题的“扭曲路径”数据集，即从一个非标准但困难的起始状态到标准终点的最优路径。

- algorithms/generate_checkers_jump_1d_v2.py: 这是一个针对跳棋交换问题的**序列学习对比实验**脚本。与 V1 版本关注单步最优策略不同，本脚本旨在生成多种类型的完整路径数据集（包括最优路径、子路径、非最优路径等），用于系统性地研究模型在不同数据分布下学习长序列规划的能力。该任务同样源自苹果公司论文 [15] 中的跳棋交换问题，但侧重于探究序列级泛化与路径多样性的影响。

- algorithms/generate_maze_symbolic_to_image.py: 将符号化的迷宫路径规划数据集转换为图像格式，以测试视觉模型（如CNN、ViT）直接从像素进行路径规划的能力。

- algorithms/generate_trapping_rain_water_visualizer.py: 这是一个**数据转换与可视化**脚本。它的作用是将已经生成的、符号化的“接雨水”数据集转换为一个image-to-image格式的数据集，以便用视觉模型来解决同一个问题。

- algorithms/generate_shortest_path_in_tree_deprecated.py: (早期探索/已弃用) 这是一个早期的实验，旨在测试模型从图像中寻找图上最短路径的能力。


## C: 视觉推理 (Visual Reasoning)

- visual_reasoning/generate_checkerboard_to_binary.py: 这是一个基础的视觉到符号转换任务，用于测试模型从原始像素数据中解码结构化信息的能力。

- visual_reasoning/generate_line_angle_to_vector.py: 测试模型从图像中提取精确几何信息（角度）的能力，这是一个比简单识别棋盘格更高级的视觉推理任务。

- visual_reasoning/generate_count_shapes_from_image.py: 测试模型同时进行物体识别（形状）、属性识别（颜色）和计数（聚合）的多重视觉任务能力。

- visual_reasoning/generate_sokoban_symbolic_to_image_no_labels.py: 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）仅转换为图像格式，用于纯视觉任务或作为更复杂数据处理的中间步骤。

- visual_reasoning/generate_sokoban_symbolic_to_image_with_labels.py: 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）转换为一个完整的图像分类数据集，以供计算机视觉模型（如ViT, Swin Transformer）进行训练。

- visual_reasoning/generate_triangle_to_incircle.py: 这是展示“用梯度下降雕刻精确规则”的一个标志性实验。它测试模型能否学习到一个纯粹的、非平凡的几何构造规则（三角形内切圆）。

- visual_reasoning/generate_polygon_to_symmetry_axis.py: 测试模型从一个完整的对称图形中反向推断出其隐含的对称轴的能力。

- visual_reasoning/generate_triangle_to_centroid.py: 测试模型学习另一个基础几何概念——重心的能力。

- visual_reasoning/generate_triangle_to_tessellation.py: 这是我们范式能力的一个标志性展示。它测试模型能否学习一种无限的、基于晶格的生成规则。由于镶嵌图案的全局关联性和细节的精确性，它有力地排除了模型仅仅是靠“插值”或“记忆”来解决问题的可能性。

- visual_reasoning/generate_shortest_distance_between_triangles.py: 测试模型在包含多个对象的情况下，进行全局几何关系（最短距离）推理的能力。

- visual_reasoning/generate_coords_to_triangle.py: 这是一个基础的符号到几何的渲染任务，测试模型将抽象的坐标信息转换为具体像素形状的能力。

- visual_reasoning/generate_triangle_coords_to_tessellation.py: 这是一个高级的、混合了符号指令和几何生成规则的推理任务。


## D: 元胞自动机 (Cellular Automata)

- cellular_automata/generate_cellular_automata_1d.py: 用于生成一维元胞自动机（CA）的演化数据集，以测试模型学习和执行局部、确定性规则的能力。

- cellular_automata/generate_game_of_life_2d.py: 生成二维元胞自动机——Conway's Game of Life的数据集。此任务比一维CA更复杂，需要模型理解二维空间中的邻域关系。

- cellular_automata/generate_cellular_automata_1d_multistate.py: 作为一维元胞自动机实验的扩展，测试模型处理非二进制状态空间的能力。

- cellular_automata/generate_cellular_automata_programmable.py: 测试模型的“可编程性”或“元学习”能力。模型不仅要学会CA的演化过程，还要能根据每次输入中给出的不同规则来执行演化。

- cellular_automata/generate_cellular_automata_inverse_rule90.py: 测试模型解决“逆问题”（Inverse Problem）的能力。给定一个确定性系统的输出，模型需要反向推断出满足特定约束（最稀疏且唯一）的可能输入。

- cellular_automata/generate_game_of_life_image_to_image.py: 这是二维元胞自动机的image-to-image版本，测试模型能否直接在像素空间中执行基于局部规则的演化。

- cellular_automata/generate_cellular_automata_spatial_conditional.py: 测试模型在单一模态（图像）内部分区和解析“指令”与“数据”的能力，是一种“伪多模态”或“空间条件化”的实验。

- cellular_automata/generate_cellular_automata_multimodal_deprecated.py: (已弃用) 生成一个真正的多模态数据集，用于训练能够同时理解图像输入和文本指令的模型。

- generate_cellular_automata_1d_to_grid_image_interp.py: 该脚本旨在设计一个“逻辑/感知混合”任务，用以证明神经网络的规则学习能力和内插能力并非互斥，而是可以一体化地在单个任务中得到体现。它迫使模型必须同时“看穿”输入的连续灰度值以执行离散的逻辑推理，并记住这些灰度值以完成最终的连续值映射。

- cellular_automata/generate_cellular_automata_1d_to_grid_image.py: 测试模型能否直接将一维的符号计算结果“渲染”成结构化的二维图像。

- cellular_automata/generate_cellular_automata_inverse_rule.py: 这个实验是检验模型**逆向推理**（Inverse Reasoning）能力的第一个尝试。我的问题是：如果模型能从规则正向推出结果，那么它能否从“输入-输出”对中反向推断出其背后的规则？

- cellular_automata/generate_cellular_automata_inverse_rule_and_steps.py: 这是在实现“唯一解”版本之前的一个早期版本，它同样旨在让模型学习预测规则和迭代次数。

- cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py: 这是对逆向推理任务的一次重大升级。我不仅要求模型推断出**什么**规则被应用了，还要推断出它被**应用了多少次**。

- generate_cellular_automata_1d_perturbed.py: 该脚本旨在系统性地测试“神经雕刻”范式在面对不完美数据时的鲁棒性。通过向输入（模拟观测噪声）和输出（模拟标签噪声）中引入可控的随机扰动，它探索了模型性能从理想规则世界向嘈杂现实世界过渡的连续谱。


## E: 物理模拟 (Physics Simulation)

- physics_simulation/generate_projectile_motion_simulation.py: 测试模型学习一个简单的动态物理过程的能力。这要求模型从初始条件（位置和速度向量）推断出整个时空轨迹。

- physics_simulation/generate_snell_refraction_simulation.py: 测试模型学习基础物理定律（斯涅尔折射定律）的能力。

- physics_simulation/generate_snell_refraction_with_contextual_index.py: 测试模型学习基础物理定律（斯涅尔折射定律）的能力，并且要求模型能从图像的上下文信息（背景颜色）中推断出物理参数（折射率）。

- physics_simulation/generate_reaction_diffusion_deprecated.py: (探索性/已弃用) 该脚本用于模拟基于 **Gray-Scott 模型** 的反应-扩散系统，以生成复杂的、类似分形的“雪花”图案。它本质上属于连续物理场的模拟（涉及偏微分方程的离散化和浮点运算），而非简单的离散状态元胞自动机。

- physics_simulation/generate_cube_rotation_matplotlib_deprecated.py: (早期探索版本) 旨在测试模型从抽象的姿态参数（旋转角度）推理并渲染出三维物体正确视图的能力。

- physics_simulation/generate_cube_rotation_pillow_v1.py: (技术升级版本) 旨在测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，采用了更底层的、渲染效果更精确的技术路线。

- physics_simulation/generate_cube_rotation_pillow_with_anchor.py: (论文中使用的最终版本) 测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，并通过引入“视觉锚点”来辅助模型学习。

- physics_simulation/generate_cube_rotation_pillow_wireframe.py: (变体实验版本) 测试模型在更稀疏的视觉输入下，能否仅通过线框和锚点信息来学习3D旋转。

- physics_simulation/generate_catenary_curve_simulation_deprecated.py: 这是我早期探索悬链线问题的脚本，旨在测试模型学习由物理定律确定的非线性曲线的能力。

- physics_simulation/generate_catenary_curve_from_points.py: 测试模型学习由物理定律（最小势能原理）唯一确定的非线性曲线（悬链线）的能力。

- physics_simulation/generate_orbital_path_from_initial_state.py: 测试模型学习更复杂物理定律（开普勒定律/万有引力定律）的能力。


## F: ARC-AGI探索 (ARC-AGI Exploration)

- arc_agi/generate_arc_contextual_color_swap.py: 测试模型从图像的局部“上下文”或“示例”中学习规则，并将其应用到同一图像的全局数据的能力。这直接模仿了ARC-AGI测试的核心理念。

- arc_agi/generate_arc_find_cross_pattern.py: 测试模型在包含大量噪音的情况下进行视觉模式识别（或可称作“目标检测”）的能力。

- arc_agi/generate_arc_find_odd_one_out.py: 测试模型执行一个复杂的“异类发现”（Find the Odd One Out）元推理任务。模型需要逐行进行模式比较，找出特例，并将其重新组合到输出中。

- arc_agi/generate_arc_connect_colored_pairs.py: 测试模型在同一图像中识别多个独立“连接任务”并理解一种隐含的“图层”或“绘制优先级”规则的能力。

- arc_agi/generate_arc_conditional_perpendicular_lines.py: 测试模型根据物体的**属性（颜色）**和**全局参照物（边界线、图像边缘）**来执行不同几何操作的能力。

- arc_agi/generate_arc_column_projection.py: 测试模型识别复杂的上下文关系（“在...下方且在...范围内”）并执行条件性列操作的能力。

- arc_agi/generate_arc_procedural_spiral.py: 测试模型执行一个迭代的、程序性的生成算法的能力。模型需要理解指令、跟踪状态（当前位置、方向、长度）并循环执行。

- arc_agi/generate_arc_fractal_stamping.py: 测试模型理解和执行递归或分形生成规则的能力。模型需要将输入图案本身作为一个“笔刷”，根据输入图案中的“指令”进行重复绘制。

- arc_agi/generate_arc_flood_fill.py: 测试模型执行经典的“洪水填充”（Flood Fill）或“油漆桶”算法的能力。

- arc_agi/generate_arc_layered_fill.py: 测试模型理解一个程序性极强的、依赖于拓扑距离和条件判断的复杂填充算法。

- arc_agi/generate_arc_fluid_simulation.py: 测试模型在图像空间中学习和模拟一个具有特定规则的流体动态过程的能力。

- arc_agi/generate_arc_periodic_conditional_fill.py: 这个实验旨在测试模型学习一个复杂的、带有周期性和特殊case的条件格式化规则的能力。

- arc_agi/generate_arc_fill_square_holes.py: 这个实验用于测试模型进行多步视觉推理的能力：首先需要识别出复杂的“背景中的前景”（即矩形中的空洞），然后对识别出的对象进行几何属性判断（是否为正方形），最后根据判断结果进行着色。

- arc_agi/generate_arc_conditional_recoloring.py: 测试模型理解视觉图层和进行条件性对象属性修改的能力。

- arc_agi/generate_arc_sort_by_length_remap_position.py: 测试模型执行一个“属性-位置解耦与重映射”的复杂排序任务。

- arc_agi/generate_arc_jigsaw_puzzle_simple.py: 测试模型解决一个视觉匹配与变换问题的能力（早期版本）。

- arc_agi/generate_arc_jigsaw_puzzle_advanced.py: 测试模型解决一个复杂的**视觉匹配与变换**问题的能力。

- arc_agi/generate_arc_connect_path_by_sequence.py: 测试模型解析外部指令序列，并据此在图像中执行多步、有状态的路径连接任务的能力。

- arc_agi/generate_arc_reflection_simulation_deprecated.py: (已废弃) 旨在测试模型理解复杂的基于物理光学的规则，包括射线发射、碰撞检测、角度反射和颜色变换。


## G: 中国象棋 (Chinese Chess)

- chinese_chess/generate_chess_positions_by_random_moves.py: 通过模拟一个完全随机的玩家下棋的过程，快速生成大量看起来合理的、合法的中国象棋局面。

- chinese_chess/generate_chess_positions_by_random_placement.py: 通过在棋盘上随机放置棋子（而非模拟下棋）来生成大量非典型的、但大部分合法的中国象棋局面，用于对模型的鲁棒性进行压力测试。

- chinese_chess/generate_chess_positions_from_engine_self_play.py: 生成大量高质量、符合实战逻辑的中国象棋局面（FEN格式），作为训练棋类AI的基础数据源。

- chinese_chess/generate_preprocess_legal_moves.py: 这是一个数据预处理脚本，用于将FEN格式的局面数据集转换为模型可以直接学习的“合法走法预测”任务。

- chinese_chess/generate_chess_resolve_check_task.py: 生成一个专门针对中国象棋中“解将”（Resolving a Check）这一特定战术场景的数据集。这个任务要求模型在处于被将军的状态下，找出所有能够合法解除将军的走法。


## 训练脚本和工具脚本 (Training Scripts and Tools)

