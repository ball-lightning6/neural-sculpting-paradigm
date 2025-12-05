## A: Symbolic Math Logic

- symbolic_math_logic/generate_conditional_add_subtract.py: The script generates addition or subtraction (absolute value) problems for two N-bit integers. It includes two modes:

- symbolic_math_logic/generate_add_binary_modulo.py: This is an early basic arithmetic experiment to test the model's ability to learn modulo addition (or "truncated addition"), a common operation in fixed-width integer arithmetic in computer hardware.

- symbolic_math_logic/generate_multiply_binary.py: As a benchmark for binary arithmetic capabilities, generates multiplication datasets for N-bit integers.

- symbolic_math_logic/generate_multiply_binary_no_carry_phase1.py: This is the first phase of the multiplication "decoupling" experiment. It aims to test whether the model can learn the first step of multiplication: bitwise multiplication without carry and staggered addition, decomposing a complex multiplication problem into a simpler counting problem.

- symbolic_math_logic/generate_multiply_binary_from_counts_phase2.py: This is the second phase of the multiplication "decoupling" experiment. It aims to verify whether an independent model can learn to handle complex carry logic, i.e., calculate the final binary product from a "no-carry count vector."

- symbolic_math_logic/generate_add_hexadecimal.py: Compares the model's learning ability across different symbolic systems. This script aims to verify whether the model learns the abstract mathematical concept of addition or merely binary-specific patterns.

- symbolic_math_logic/generate_multiply_decimal.py: Tests the model's ability to handle non-binary symbolic inputs (0-9 characters) and perform arithmetic operations (multiplication).

- symbolic_math_logic/generate_add_binary_with_position_shuffle.py: This is the "position shuffle" part of the "semantic shuffle" experiment series. It aims to verify whether the model relies on fixed spatial structure of inputs or can learn position-independent abstract relationships.

- symbolic_math_logic/generate_symbol_add_shuffle_dataset.py: This is a **critical decisive experiment** in our research, aiming to completely separate the model's "surface pattern matching" ability from its "abstract structure learning" ability.

- symbolic_math_logic/generate_add_hidden_constant.py: Tests the model's ability to **infer hidden rules or parameters** from large amounts of samples without any direct clues. This is similar to a simplified System Identification problem.

- symbolic_math_logic/generate_multitask_alu.py: This script aims to build a multi-task learning scenario simulating an **Arithmetic Logic Unit (ALU)**. It tests whether the model can perform multiple different, well-defined computational tasks in parallel on the same input in a single forward pass.

- symbolic_math_logic/generate_modulo_operation.py: Explores the model's ability to learn modulo operation, a crucial but "cyclic" operation in number theory and computer science.

- symbolic_math_logic/generate_rsa_encryption.py: Tests the model's ability to learn highly nonlinear, computationally "hard" deterministic rules. RSA encryption is a typical example.

- symbolic_math_logic/generate_deduction_chain_text.py: Generates multi-step logical reasoning tasks to test the model's ability to perform symbolic deduction, similar to a simplified theorem prover.

- symbolic_math_logic/generate_deduction_multirule_text.py: Tests whether the model can correctly "route" to the appropriate rule and make judgments based on the Query when facing multiple independent, unrelated rules.

- symbolic_math_logic/generate_deduction_multirule_text_v2.py: Tests whether the model can correctly "route" to the appropriate rule and make judgments based on the Query when facing multiple independent, unrelated rules.

- symbolic_math_logic/generate_deduction_multirule_binary.py: This is a **format optimization** version of the multi-rule reasoning task, aiming to test whether compact binary encoding is more conducive to model learning than sparse text format.

- symbolic_math_logic/generate_deduction_fixed_depth.py: Tests the model's multi-step reasoning ability in symbolic deduction tasks with clear structure and fixed depth.

- symbolic_math_logic/generate_function_composition.py: Tests the model's ability to learn function composition. This requires the model to act like an interpreter, parsing instructions sequentially and transforming data.

- symbolic_math_logic/generate_count_set_bits.py: Tests the model's ability to perform global aggregation operations. Unlike local rules, counting requires the model to synthesize information across the entire input sequence.

- symbolic_math_logic/generate_sum_pattern_positions.py: Tests the model's ability to perform more complex, grouped parallel aggregation tasks. The model needs to first split the input, then classify each split pattern, and finally accumulate the **position information** of patterns belonging to the same class.

- symbolic_math_logic/generate_sum_pattern_positions_v2.py: Tests the model's ability to perform more complex, grouped parallel aggregation tasks. The model needs to first split the input, then classify each split pattern, and finally accumulate the **position information** of patterns belonging to the same class.

- symbolic_math_logic/generate_sum_pairwise_hamming_distance.py: Tests the model's ability to perform a complex task requiring two layers of nested aggregation operations. The model needs to first perform global statistics **at each bit position**, then accumulate results **across all bit positions**.

- symbolic_math_logic/generate_circular_shift.py: Tests the model's ability to learn shift operations, particularly circular shift, a common operation in cryptography and low-level programming.

- symbolic_math_logic/generate_multiply_matrix_3x3.py: Tests the model's ability to learn structured algebraic operations (matrix multiplication), which requires more complex "data routing" and "multiply-accumulate" capabilities than simple scalar operations.

- symbolic_math_logic/generate_evaluate_boolean_expression_text.py: Tests the model's ability to parse a simple domain-specific language (DSL) and perform evaluation, a step further than previous fixed-structure expression evaluation.

- symbolic_math_logic/generate_evaluate_arithmetic_expression.py: Trains the model to perform symbolic expression evaluation tasks, requiring the model to understand operator precedence (implicitly expressed through tree structure), variable substitution, and arithmetic operations.

- symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply.py: This is a simplified version of generate_evaluate_arithmetic_expression.py, aiming to reduce learning difficulty by removing multiplication operations to test the model's ability in more basic arithmetic expression evaluation.

- symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply_small_range.py: This is a further simplification based on the previous "no multiplication" version, further reducing learning difficulty by narrowing the numeric range to precisely diagnose the model's performance bottleneck in the simplest expression evaluation tasks.

- symbolic_math_logic/generate_check_boolean_equivalence.py: Tests the model's ability to judge logical equivalence in boolean algebra. This is an abstract symbolic reasoning task requiring the model to understand expression structure and boolean operation rules.

- symbolic_math_logic/generate_polynomial_shift_coefficients.py: Tests the model's ability to learn an abstract algebraic transformation rule. This task requires the model to understand the internal structure of polynomial expansion.

- symbolic_math_logic/generate_convolution_2d.py: Tests the model's ability to learn 2D convolution (Conv2D), a basic image processing operation, and explores whether it can infer the hidden fixed rule (i.e., the convolution kernel itself) from input-output pairs.

- symbolic_math_logic/generate_simple_block_cipher.py: Tests the model's ability to "crack" or learn a simple but nontrivial custom encryption algorithm. This task represents a class of complex symbolic transformation rules with high chaos and avalanche effects.

- symbolic_math_logic/generate_sin_function_float32.py: Tests the model's ability to fit continuous, periodic, nonlinear functions (sin(x)) using standard 32-bit floating-point format for input and output.

- symbolic_math_logic/generate_sin_function_float64_to_int12_deprecated.py: This is another encoding attempt for the sin function fitting task, aiming to explore the impact of using higher-precision floating-point input and lower-precision quantized binary output on learning effectiveness.

- symbolic_math_logic/generate_sin_function_float32_to_quantized_int.py: Tests the model's ability to fit continuous, periodic, nonlinear functions (sin(x)) and explores the impact of different input/output encoding schemes on learning effectiveness.

- symbolic_math_logic/generate_multiply_binary_modulo.py: As part of basic arithmetic experiments, tests the model's mastery of truncated multiplication (or modulo multiplication).

- symbolic_math_logic/generate_explainable_two_step_calculation.py: Tests the model's ability to output "intermediate steps" or "chain of thought" of calculations, a direct verification of "functional interpretability."

- symbolic_math_logic/generate_min_swaps_for_checkerboard.py: Solves LeetCode problem 782 "Transform to Chessboard" ([https://leetcode.cn/problems/transform-to-chessboard/](https://leetcode.cn/problems/transform-to-chessboard/)) - calculates the minimum number of row and column swaps needed to transform a 0/1 matrix into "checkerboard" pattern (adjacent elements differ) through arbitrary row and column exchanges.

- symbolic_math_logic/generate_min_flips_for_alternating_binary.py: Tests the model's ability to solve a string optimization problem based on bit flips, which can be cleverly mapped to a sliding window problem.

- symbolic_math_logic/generate_min_swaps_for_checkerboard_v2.py: Solves LeetCode problem 1536 "Minimum Swaps to Arrange a Binary Grid" ([https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/](https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/)) - calculates the minimum number of adjacent row swaps needed to transform a binary grid into upper triangular form (all zeros above main diagonal).

- symbolic_math_logic/generate_min_prefix_flips.py: Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing.

- symbolic_math_logic/generate_min_flips_for_chunked_binary.py: Tests the model's ability to learn a string transformation optimization problem based on local chunks.

- symbolic_math_logic/generate_largest_island_by_adding_one_cell.py: Solves an algorithm problem involving graph traversal and global optimization ([LeetCode 827. Making a Large Island](https://leetcode.cn/problems/making-a-large-island/)). The model needs to evaluate all possible "land reclamation" positions and select the one that can maximize the merged island area.

- symbolic_math_logic/generate_largest_island_by_adding_one_cell_v2.py: Solves an algorithm problem involving graph traversal and global optimization ([LeetCode 827. Making a Large Island](https://leetcode.cn/problems/making-a-large-island/)). The model needs to evaluate all possible "land reclamation" positions and select the one that can maximize the merged island area.

- symbolic_math_logic/generate_sat_solver_text.py: Tests the model's ability to solve the iconic NP-complete problem - Boolean Satisfiability (SAT) problem.

- symbolic_math_logic/generate_sat_solver_compact_text.py: This is a variant of symbolic_math_logic/generate_sat_solver_text.py, using a different input encoding format to solve the same 3-SAT problem.


## B: Algorithm Learning

- algorithms/generate_sort_integers.py: Tests the model's ability to perform basic sorting algorithms, a non-local classic algorithm task requiring comparison and rearrangement of input elements.

- algorithms/generate_edit_distance.py: Tests the model's ability to learn to solve dynamic programming problems. Edit distance is a typical DP problem requiring the model to conceptually build a 2D solution matrix. [LeetCode 72. Edit Distance](https://leetcode.com/problems/edit-distance/description/)

- algorithms/generate_edit_distance_explainable.py: This is a core experiment in "functional interpretability." It requires the model to not only give the final answer (edit distance) but also output the complete "chain of thought" (edit process) to reach the answer. [LeetCode 72. Edit Distance (Explainable / Path Construction Version)](https://leetcode.com/problems/edit-distance/description/)

- algorithms/generate_maze_random_walls.py: Tests the model's basic pathfinding ability in randomly generated "porous" mazes.

- algorithms/generate_maze_dense.py: Tests the model's ability to perform path planning in complex, human-designed "dense" mazes, which is more challenging than random wall mazes.

- algorithms/generate_blocks_world_arbitrary_goal.py: Solves the classic "Blocks World" planning problem. This problem, as a standard task for measuring large language model reasoning ability, is discussed in detail in Apple's famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity," as a benchmark test for evaluating model capabilities in state space search and planning. The study systematically reveals the fundamental limitations of current large language models in precise symbolic reasoning and state space planning through classic planning tasks like blocks problems, river crossing problems, and Tower of Hanoi. This script precisely implements the general version of "Blocks World" from the paper, allowing specification of arbitrary initial and goal states, as a core control experiment for verifying neural network reasoning capabilities on complex planning problems.

- algorithms/generate_blocks_world_fixed_goal.py: This is a simplified version of the "Blocks World" task. By fixing the goal state (all blocks orderly stacked on the first pillar), it aims to test the model's learning ability in situations with clear goals and more structured state space. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study points out that even for such goal-defined planning problems, large language models still have significant difficulties in state space search and optimal strategy learning. This script implements the simplified version of Blocks World from the paper, reducing task complexity by fixing the goal state, as a control experiment for studying the impact of goal clarity on model reasoning performance.

- algorithms/generate_blocks_world_fixed_goal_multilabel.py: Further improves the "Blocks World" task by allowing multiple optimal solutions, testing the model's ability to handle multi-label classification problems, more realistically reflecting possible equivalent optimal paths in planning problems. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study emphasizes that real-world planning problems often have multiple equivalent optimal solutions, posing higher requirements for the model's ambiguous reasoning capabilities. This script further improves upon the fixed-goal version, finding all actions that lead to optimal paths for each state and generating multi-hot encoded outputs, as a control experiment for studying neural networks' ability to handle planning problems with multiple optimal solutions.

- algorithms/generate_blocks_world_fixed_goal_multilabel_fixed_format.py: This is the final optimized version of the "Blocks World" task, aiming to provide the model with a clearer, more structured learning objective by improving input representation. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study points out that input representation has a crucial impact on model learning efficiency and final performance. This script further optimizes upon the multi-label version, using fixed-slot representation to replace variable-length input, eliminating complexity brought by serialization, providing more friendly structured input for Transformer and other architectures. This makes it a control experiment for studying the impact of input representation on neuro-symbolic reasoning performance.

- algorithms/generate_checkers_jump_1d.py: Solves the checker exchange planning problem in one-dimensional space. This problem, as a standard task for measuring large language model reasoning ability, is discussed in detail in Apple's famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity," as a benchmark test for evaluating model capabilities in state space search and planning. The study systematically reveals the fundamental limitations of current large language models in precise symbolic reasoning and state space planning through classic planning tasks like blocks problems, river crossing problems, and Tower of Hanoi. This script precisely implements the general version of "checker exchange" from the paper, as a core control experiment for verifying neural network reasoning capabilities on complex planning problems.

- algorithms/generate_river_crossing_puzzle.py: Solves the classic constraint satisfaction and state space search problem - "N couples crossing the river." This problem requires transporting everyone to the other side under the constraint that "no woman can be with other men without her partner present." This task originates from a famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" by Apple, which reveals fundamental limitations of large language models in certain types of reasoning tasks through benchmark tests like river crossing, Tower of Hanoi, and checkers. This script precisely reproduces the "N couples crossing the river" problem from the paper, as a control experiment for verifying neural network symbolic reasoning capabilities.

- algorithms/generate_trapping_rain_water_aggregate.py: This is an initial attempt to solve the "trapping rain water" algorithm problem, aiming to test the model's ability to learn an aggregated output (rather than decoupled output). Experimental results show that requiring the model to directly output a sum value (a single aggregated number) is much more difficult than outputting detailed information for each position. **This becomes a key comparative experiment, proving the systematic impact of output format design on model learning efficiency.** Corresponding LeetCode problem: [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/). This script precisely implements LeetCode's original aggregated output format (only outputting total amount), forming a sharp contrast with `generate_trapping_rain_water_decoupled.py` (decoupled output, outputting water amount at each position), used to study the impact of output representation on neural network learning difficulty, verifying the core finding of "decoupling accelerates convergence" in the paper.

- algorithms/generate_trapping_rain_water_decoupled.py: Solves the classic "trapping rain water" algorithm problem (LeetCode Hard [#42](https://leetcode.com/problems/trapping-rain-water/)). The success of this task demonstrates the model's ability to learn complex algorithms requiring global information (like global maximum points), and through the idea of **problem decoupling**, proves the huge impact of output format design on model learning efficiency.

- algorithms/generate_trapping_rain_water_2d.py: As an extension of the one-dimensional "trapping rain water" problem, solves the two-dimensional version. This task requires the model to understand "enclosure" and "boundary" concepts in two-dimensional space, representing a more complex global information processing challenge. Corresponding LeetCode problem: [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/) (Hard difficulty).

- algorithms/generate_skyline_max_height_aggregate.py: This is an initial attempt to solve the "skyline" problem, requiring the model to predict only the highest height value from all buildings' final heights. This task is used to compare learning difficulty between aggregated and decoupled outputs. Corresponding LeetCode problem: [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/) (aggregated output version).

- algorithms/generate_skyline_all_heights_decoupled.py: Tests the model's ability to solve a global optimization problem with one-dimensional spatial constraints. The problem prototype is LeetCode [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/) (decoupled output version). The problem requires: given N buildings' positions and height limits, adjacent building height differences cannot exceed 1, find the maximum height each building can achieve. Through decoupled output, the model is required to predict the height of each building, not just the maximum value, as a control experiment for studying the impact of output format design on model learning efficiency.

- algorithms/generate_hanoi_tower_path_strategy_sep_format.py: As comparison group A, this script generates Tower of Hanoi optimal path strategy datasets using **separator + binary encoding** input format.

- algorithms/generate_hanoi_tower_path_strategy_fixed_format.py: As comparison group B, this script also generates Tower of Hanoi optimal path strategy datasets but uses **structured fixed-slot** input format (renamed from original `global` script to reflect true logic), used to verify the impact of data representation on learning efficiency.

- algorithms/generate_hanoi_tower_compare_formats.py: This is a comparative experimental script that generates two different input formats (separator vs. fixed slot) for the same Tower of Hanoi problem, used to systematically evaluate the impact of different data representations on the model's ability to learn recursive strategies.

- algorithms/generate_hanoi_tower_compare_formats_and_strategies.py: This is a more comprehensive Tower of Hanoi comparative experimental script. It not only generates two input formats but also generates two different datasets: one containing only states on the optimal path ("path strategy"), another containing all reachable states ("global strategy"), used to explore differences in the model's ability to learn local optimal paths and global optimal strategies.

- algorithms/generate_hanoi_tower_build_full_state_graph.py: This is the culmination of "Tower of Hanoi problem" research, aiming to deeply analyze the model's understanding of recursive structures through multiple different data representations and sampling strategies. It is a self-contained data factory. This research corresponds to the Apple Research related work cited in the paper, focusing on analyzing the model's mastery of recursive structures through complete state graphs.

- algorithms/generate_hanoi_tower_sample_from_state_graph.py: This is a post-processing and sampling script that uses the complete knowledge base generated by generate_hanoi_tower_build_full_state_graph.py to precisely extract specific types of training data subsets, such as "twisted path" or "hardest part," for more refined ablation experiments.

- algorithms/generate_sokoban_planning_astar.py: Solves the classic "Sokoban" planning problem. This task is harder than simple path planning because it involves changing environmental states (box positions) and has a huge state space.

- algorithms/generate_sokoban_planning_full.py: Solves the classic "Sokoban" planning problem. This is a high-difficulty AI task because it involves searching in a huge state space and actions change environmental states.

- algorithms/generate_sokoban_planning_claude_deprecated.py: (Deprecated) This is an early attempt with more complex logic, but it failed to stably generate high-quality datasets and has been replaced by the more reliable generate_sokoban_planning_full.py.

- algorithms/generate_matrix_flip_strategy.py: Solves a classic matrix optimization problem (Score After Flipping Matrix). This version aims to test whether the model can learn a "strategy" rather than the final result. Corresponding LeetCode problem: [861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- algorithms/generate_matrix_flip_max_score.py: Tests the model's ability to learn a matrix optimization problem that requires a two-step greedy strategy (row flips first, then column flips) to achieve global optimum. This version requires the model to directly output the final aggregated result (score). Corresponding LeetCode problem: [861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- algorithms/generate_min_k_bit_flips.py: Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing, and tests whether it can use part of the input (k) as a "parameter" to guide processing of another part (nums). Corresponding LeetCode problem: [995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)

- algorithms/generate_min_k_bit_flips_fixed_k.py: Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing. In this version, the environmental parameter (k=2) is fixed and hidden, the model must implicitly learn it from data. This is a control experiment for comparing with the variable k version. Corresponding LeetCode problem: [995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/) (fixed k version)

- algorithms/generate_special_binary_string_recursion.py: Tests the model's ability to learn a recursively defined string transformation rule. This problem corresponds to LeetCode problem: [761. Special Binary String](https://leetcode.com/problems/special-binary-string/) (Hard difficulty). Problem requirement: A special binary sequence has equal numbers of 0s and 1s, and any prefix has no fewer 1s than 0s; you can swap adjacent special substrings to get the lexicographically largest result. This script precisely implements the recursive decomposition and lexicographic sorting algorithm for this problem, as a benchmark experiment for testing neural network's ability to learn complex recursive rules.

- algorithms/generate_count_connected_components.py: Tests the model's basic understanding of graph structure, particularly the core concept of "connectivity" (corresponding to LeetCode 323. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/), note: this is a member problem).

- algorithms/generate_check_graph_connectivity.py: This is another core test of the model's graph theory foundation, the task is to determine whether there exists a path between any two points in a graph (corresponding to LeetCode 1971. [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/)).

- algorithms/generate_minimize_malware_spread.py: Solves a graph theory-based virus spread optimization problem (LeetCode Hard "Minimize Malware Spread"). The model needs to understand graph connectivity and evaluate the impact of removing different nodes on global spread. This script provides two output formats for comparison to see which is easier to learn. Corresponding LeetCode problem: [924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)

- algorithms/generate_count_islands_1d.py: Tests the model's ability to perform pattern recognition and counting on one-dimensional sequences.

- algorithms/generate_find_articulation_points.py: Tests the model's ability to identify graph "articulation points" or "bridges," an important concept in graph theory. [LeetCode 1568. Minimum Number of Days to Disconnect Island](https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/)

- algorithms/generate_nim_game_zeckendorf.py: This experiment aims to test whether my paradigm can learn a non-intuitive game theory problem based on complex number theory (Zeckendorf representation). It moves beyond simple pattern matching and requires the model to understand deeper mathematical structures.

- algorithms/generate_longest_subsequence_constrained.py: Tests the model's ability to handle a complex optimization problem mixing sequence operations and numeric constraints. Corresponding LeetCode problem: [2311. Longest Binary Subsequence Less Than or Equal to K](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/)

- algorithms/generate_treasure_hunt_tsp.py: Solves a complex state space search problem that combines graph traversal (BFS) and combinatorial optimization (state compression DP), a classic difficult problem in algorithm competitions.

- algorithms/generate_freedom_trail_dp.py: [LeetCode 514. Freedom Trail](https://leetcode.com/problems/freedom-trail/description/) Tests the model's ability to learn to solve a complex optimization problem requiring dynamic programming and path backtracking.

- algorithms/generate_sum_of_subset_with_mask.py: Tests the model's ability to select elements from a set based on a binary mask and perform aggregation operations (summing). This is an innovative control experiment aiming to verify that **task structure itself** (rather than model capacity) is the key factor affecting neural network learnability.

- algorithms/generate_sudoku_6x6.py: Tests the model's ability to handle strongly constrained satisfaction problems (CSP) - Sudoku.

- algorithms/generate_valid_parentheses_path_random_deprecated.py: (Early exploration/Deprecated) This is an early attempt to solve the "valid parentheses path" problem.

- algorithms/generate_valid_parentheses_path_balanced.py: Solves a pathfinding problem on a two-dimensional grid, but path validity is constrained by stack structure (parentheses matching). This is a complex task combining algorithms and logical constraints (LeetCode Hard [#2267](https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/) "Check if There Is a Valid Parentheses String Path").

- algorithms/generate_point_in_polygon.py: Tests the model's ability to learn a classic algorithm in computational geometry - the Ray Casting Algorithm.

- algorithms/generate_shortest_path_in_matrix_bfs.py: Tests the model's ability to find the shortest path in a two-dimensional grid based on the classic Breadth-First Search (BFS) algorithm.

- algorithms/generate_sudoku_4x4_stepwise_deprecated.py: (Deprecated) Aims to test the model's ability to perform "stepwise" reasoning, i.e., only predicting the next optimal action at each state, rather than outputting the complete solution at once.

- algorithms/generate_tiling_problem_deprecated.py: (Deprecated) Aims to test the model's ability to solve a classic tiling coverage optimization problem, which is an NP-hard problem.

- algorithms/generate_hanoi_tower_twisted_path_deprecated.py: (Deprecated) This script intended to generate a "twisted path" dataset for the Tower of Hanoi problem, i.e., the optimal path from a non-standard but difficult start state to the standard goal.

- algorithms/generate_checkers_jump_1d_v2.py: This is a **sequence learning comparative experiment** script for the checkers exchange problem. Unlike V1 version focusing on single-step optimal strategy, this script aims to generate various types of complete path datasets (including optimal paths, subpaths, non-optimal paths, etc.) for systematically studying the model's ability to learn long sequence planning under different data distributions. This task also originates from the checkers exchange problem in Apple's paper [15], but focuses on exploring the impact of sequence-level generalization and path diversity.

- algorithms/generate_maze_symbolic_to_image.py: Converts symbolic maze path planning datasets to image format to test the ability of visual models (like CNN, ViT) to perform path planning directly from pixels.

- algorithms/generate_trapping_rain_water_visualizer.py: This is a **data conversion and visualization** script. Its role is to convert the already generated, symbolic "trapping rain water" dataset into an image-to-image format dataset, so that visual models can solve the same problem.

- algorithms/generate_shortest_path_in_tree_deprecated.py: (Early exploration/Deprecated) This is an early experiment aiming to test the model's ability to find the shortest path on a graph from images.


## C: Visual Reasoning

- visual_reasoning/generate_checkerboard_to_binary.py: This is a basic vision-to-symbol conversion task for testing the model's ability to decode structured information from raw pixel data.

- visual_reasoning/generate_line_angle_to_vector.py: Tests the model's ability to extract precise geometric information (angles) from images, a more advanced visual reasoning task than simple checkerboard recognition.

- visual_reasoning/generate_count_shapes_from_image.py: Tests the model's ability to simultaneously perform multiple visual tasks: object recognition (shape), attribute recognition (color), and counting (aggregation).

- visual_reasoning/generate_sokoban_symbolic_to_image_no_labels.py: This is a data conversion script for converting symbolic Sokoban datasets (.jsonl format) to image format only, used for pure vision tasks or as an intermediate step for more complex data processing.

- visual_reasoning/generate_sokoban_symbolic_to_image_with_labels.py: This is a data conversion script for converting symbolic Sokoban datasets (.jsonl format) into a complete image classification dataset for training computer vision models (like ViT, Swin Transformer).

- visual_reasoning/generate_triangle_to_incircle.py: This is a landmark experiment demonstrating "carving precise rules with gradient descent." It tests whether the model can learn a pure, nontrivial geometric construction rule (triangle incircle).

- visual_reasoning/generate_polygon_to_symmetry_axis.py: Tests the model's ability to reverse infer the hidden symmetry axis from a complete symmetric figure.

- visual_reasoning/generate_triangle_to_centroid.py: Tests the model's ability to learn another basic geometric concept - centroid.

- visual_reasoning/generate_triangle_to_tessellation.py: This is a landmark demonstration of our paradigm's capabilities. It tests whether the model can learn an infinite, lattice-based generation rule. Due to the global correlation and precise details of tessellation patterns, it strongly rules out the possibility that the model solves problems merely through "interpolation" or "memorization."

- visual_reasoning/generate_shortest_distance_between_triangles.py: Tests the model's ability to perform global geometric relationship (shortest distance) reasoning when containing multiple objects.

- visual_reasoning/generate_coords_to_triangle.py: This is a basic symbol-to-geometry rendering task, testing the model's ability to convert abstract coordinate information into concrete pixel shapes.

- visual_reasoning/generate_triangle_coords_to_tessellation.py: This is an advanced reasoning task mixing symbolic instructions and geometric generation rules.


## D: Cellular Automata

- cellular_automata/generate_cellular_automata_1d.py: Used to generate one-dimensional cellular automaton (CA) evolution datasets to test the model's ability to learn and execute local, deterministic rules.

- cellular_automata/generate_game_of_life_2d.py: Generates datasets for two-dimensional cellular automaton - Conway's Game of Life. This task is more complex than 1D CA, requiring the model to understand neighborhood relationships in two-dimensional space.

- cellular_automata/generate_cellular_automata_1d_multistate.py: As an extension of one-dimensional cellular automaton experiments, tests the model's ability to handle non-binary state spaces.

- cellular_automata/generate_cellular_automata_programmable.py: Tests the model's "programmability" or "meta-learning" ability. The model must not only learn CA evolution but also be able to perform evolution according to different rules given in each input.

- cellular_automata/generate_cellular_automata_inverse_rule90.py: Tests the model's ability to solve "inverse problems." Given the output of a deterministic system, the model needs to reverse infer possible inputs that satisfy specific constraints (sparsest and unique).

- cellular_automata/generate_game_of_life_image_to_image.py: This is the image-to-image version of two-dimensional cellular automaton, testing whether the model can directly perform local rule-based evolution in pixel space.

- cellular_automata/generate_cellular_automata_spatial_conditional.py: Tests the model's ability to partition and parse "instructions" and "data" within a single modality (image), a "pseudo-multimodal" or "spatial conditioning" experiment.

- cellular_automata/generate_cellular_automata_multimodal_deprecated.py: (Deprecated) Generates a true multimodal dataset for training models that can simultaneously understand image input and text instructions.

- generate_cellular_automata_1d_to_grid_image_interp.py: This script aims to design a "logic/perception hybrid" task to prove that neural network's rule learning ability and interpolation ability are not mutually exclusive, but can be integrated and demonstrated in a single task. It forces the model to simultaneously "see through" the continuous grayscale values of the input to perform discrete logical reasoning, and remember these grayscale values to complete the final continuous value mapping.

- cellular_automata/generate_cellular_automata_1d_to_grid_image.py: Tests whether the model can directly "render" one-dimensional symbolic computation results into structured two-dimensional images.

- cellular_automata/generate_cellular_automata_inverse_rule.py: This experiment is the first attempt to test the model's **inverse reasoning** ability. My question is: if the model can forward derive results from rules, can it reverse infer the underlying rule from "input-output" pairs?

- cellular_automata/generate_cellular_automata_inverse_rule_and_steps.py: This is an early version before implementing the "unique solution" version, also aiming to have the model learn to predict rules and iteration steps.

- cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py: This is a major upgrade to the inverse reasoning task. I not only require the model to infer **what** rule was applied, but also **how many times** it was applied.

- generate_cellular_automata_1d_perturbed.py: This script aims to systematically test the robustness of the "neural sculpting" paradigm when facing imperfect data. By introducing controllable random perturbations to input (simulating observation noise) and output (simulating label noise), it explores the continuous spectrum of model performance transition from ideal rule world to noisy real world.


## E: Physics Simulation

- physics_simulation/generate_projectile_motion_simulation.py: Tests the model's ability to learn a simple dynamic physical process. This requires the model to infer the entire spatiotemporal trajectory from initial conditions (position and velocity vectors).

- physics_simulation/generate_snell_refraction_simulation.py: Tests the model's ability to learn basic physical laws (Snell's law of refraction).

- physics_simulation/generate_snell_refraction_with_contextual_index.py: Tests the model's ability to learn basic physical laws (Snell's law of refraction), and requires the model to infer physical parameters (refractive index) from contextual information (background color) in the image.

- physics_simulation/generate_reaction_diffusion_deprecated.py: (Exploratory/Deprecated) This script simulates a reaction-diffusion system based on the **Gray-Scott model** to generate complex, fractal-like "snowflake" patterns. It essentially belongs to continuous physical field simulation (involving discretization of partial differential equations and floating-point operations), rather than simple discrete state cellular automata.

- physics_simulation/generate_cube_rotation_matplotlib_deprecated.py: (Early exploration version) Aims to test the model's ability to infer and render correct views of 3D objects from abstract pose parameters (rotation angles).

- physics_simulation/generate_cube_rotation_pillow_v1.py: (Technical upgrade version) Aims to test the model's ability to infer and render correct views of 3D objects from abstract pose parameters, using a more low-level, precise technical route.

- physics_simulation/generate_cube_rotation_pillow_with_anchor.py: (Final version used in the paper) Tests the model's ability to infer and render correct views of 3D objects from abstract pose parameters, and assists model learning by introducing "visual anchors."

- physics_simulation/generate_cube_rotation_pillow_wireframe.py: (Variant experimental version) Tests whether the model can learn 3D rotation using only wireframe and anchor point information under sparser visual input.

- physics_simulation/generate_catenary_curve_simulation_deprecated.py: This is my early script for exploring the catenary problem, aiming to test the model's ability to learn nonlinear curves determined by physical laws.

- physics_simulation/generate_catenary_curve_from_points.py: Tests the model's ability to learn nonlinear curves (catenary) uniquely determined by physical laws (minimum potential energy principle).

- physics_simulation/generate_orbital_path_from_initial_state.py: Tests the model's ability to learn more complex physical laws (Kepler's laws / law of universal gravitation).


## F: ARC-AGI Exploration

- arc_agi/generate_arc_contextual_color_swap.py: Tests the model's ability to learn rules from local "context" or "examples" in the image and apply them to global data in the same image. This directly mimics the core concept of ARC-AGI tests.

- arc_agi/generate_arc_find_cross_pattern.py: Tests the model's ability to perform visual pattern recognition (or "object detection") in the presence of large amounts of noise.

- arc_agi/generate_arc_find_odd_one_out.py: Tests the model's ability to perform a complex "Find the Odd One Out" meta-reasoning task. The model needs to compare patterns row by row, find special cases, and recombine them into the output.

- arc_agi/generate_arc_connect_colored_pairs.py: Tests the model's ability to identify multiple independent "connection tasks" in the same image and understand an implicit "layer" or "drawing priority" rule.

- arc_agi/generate_arc_conditional_perpendicular_lines.py: Tests the model's ability to perform different geometric operations based on objects' **attributes (color)** and **global references (boundary lines, image edges).**

- arc_agi/generate_arc_column_projection.py: Tests the model's ability to recognize complex contextual relationships ("below... and within...") and perform conditional column operations.

- arc_agi/generate_arc_procedural_spiral.py: Tests the model's ability to execute an iterative, procedural generation algorithm. The model needs to understand instructions, track state (current position, direction, length), and execute loops.

- arc_agi/generate_arc_fractal_stamping.py: Tests the model's ability to understand and execute recursive or fractal generation rules. The model needs to use the input pattern itself as a "brush" and repeat drawing according to "instructions" in the input pattern.

- arc_agi/generate_arc_flood_fill.py: Tests the model's ability to execute the classic "flood fill" or "paint bucket" algorithm.

- arc_agi/generate_arc_layered_fill.py: Tests the model's ability to understand a highly procedural, topology-distance-dependent and conditional-judgment-based complex filling algorithm.

- arc_agi/generate_arc_fluid_simulation.py: Tests the model's ability to learn and simulate a fluid dynamic process with specific rules in image space.

- arc_agi/generate_arc_periodic_conditional_fill.py: This experiment aims to test the model's ability to learn a complex conditional formatting rule with periodicity and special cases.

- arc_agi/generate_arc_fill_square_holes.py: This experiment tests the model's ability to perform multi-step visual reasoning: first needs to identify complex "foreground in background" (i.e., holes in rectangles), then perform geometric attribute judgment (whether it's a square) on identified objects, and finally color based on judgment results.

- arc_agi/generate_arc_conditional_recoloring.py: Tests the model's ability to understand visual layers and perform conditional object attribute modification.

- arc_agi/generate_arc_sort_by_length_remap_position.py: Tests the model's ability to perform a complex sorting task of "attribute-position decoupling and remapping."

- arc_agi/generate_arc_jigsaw_puzzle_simple.py: Tests the model's ability to solve a visual matching and transformation problem (early version).

- arc_agi/generate_arc_jigsaw_puzzle_advanced.py: Tests the model's ability to solve a complex **visual matching and transformation** problem.

- arc_agi/generate_arc_connect_path_by_sequence.py: Tests the model's ability to parse external instruction sequences and perform multi-step, stateful path connection tasks in the image accordingly.

- arc_agi/generate_arc_reflection_simulation_deprecated.py: (Deprecated) Aims to test the model's ability to understand complex physical optics-based rules, including ray emission, collision detection, angle reflection, and color transformation.


## G: Chinese Chess

- chinese_chess/generate_chess_positions_by_random_moves.py: Quickly generates a large number of plausible, legal Chinese chess positions by simulating a completely random player making moves.

- chinese_chess/generate_chess_positions_by_random_placement.py: Generates a large number of atypical but mostly legal Chinese chess positions by randomly placing pieces on the board (rather than simulating moves), used for stress testing the model's robustness.

- chinese_chess/generate_chess_positions_from_engine_self_play.py: Generates a large number of high-quality, combat-logic-compliant Chinese chess positions (FEN format) as basic data source for training chess AI.

- chinese_chess/generate_preprocess_legal_moves.py: This is a data preprocessing script for converting FEN format position datasets into a "legal move prediction" task that models can directly learn.

- chinese_chess/generate_chess_resolve_check_task.py: Generates a dataset specifically targeting the "resolving a check" tactical scenario in Chinese chess. This task requires the model to find all legal moves that can resolve the check when in a checked state.


## Training Scripts and Tools

