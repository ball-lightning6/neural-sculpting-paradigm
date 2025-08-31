## A. Symbolic Rule Learning

- generate_conditional_add_subtract.py: This script is used to explore the model's ability to handle "rule conflicts" or "conditional logic." It aims to test whether the model fails to learn when multiple rules are implicitly mixed within a dataset, and whether it can succeed when an explicit indicator is provided.
    
- generate_add_binary_modulo.py: This is an early, basic arithmetic experiment to test the model's ability to learn modulo addition (or "truncated addition"), an operation common in fixed-width integer arithmetic in computer hardware.
    
- generate_multiply_binary.py: As a benchmark for binary arithmetic capabilities, this generates a dataset for the multiplication of N-bit integers.
    
- generate_multiply_binary_no_carry_phase1.py: This is the first phase of the multiplication "decoupling" experiment. It aims to test if the model can learn the first step of multiplication: digit-wise multiplication without carry and addition with shifts, decomposing a complex multiplication problem into a simpler counting problem.
    
- generate_multiply_binary_from_counts_phase2.py: This is the second phase of the multiplication "decoupling" experiment. It aims to verify if a separate model can learn to handle complex carry logic, i.e., calculating the final binary product from a "no-carry count vector."
    
- generate_add_hexadecimal.py: Compares the model's learning ability across different symbolic systems. This script aims to verify whether the model is learning the abstract mathematical concept of addition or just patterns specific to binary symbols.
    
- generate_multiply_decimal.py: Tests the model's ability to process non-binary symbolic input (characters 0-9) and perform arithmetic operations (multiplication).
    
- generate_add_n_base_with_shuffle.py: This is a **key, decisive experiment** in our research, designed to completely separate the model's "superficial pattern matching" ability from its "abstract structure learning" ability.
    
- generate_add_binary_with_position_shuffle.py: This is the "position shuffle" part of the "semantic shuffle" series of experiments. It aims to verify whether the model relies on the fixed spatial structure of the input or can learn position-independent abstract relationships.
    
- generate_add_hidden_constant.py: Tests the model's ability to **infer hidden rules or parameters** from a large number of samples without any direct clues. This is similar to a simplified System Identification problem.
    
- generate_multitask_alu.py: This script aims to build a multi-task learning scenario simulating an Arithmetic Logic Unit (ALU). It tests whether the model can perform multiple different, well-defined computational tasks in parallel on the same input in a single forward pass.
    
- generate_modulo_operation.py: Explores the model's ability to learn the Modulo Operation, an operation that is crucial in number theory and computer science but has a "cyclical" nature.
    
- generate_rsa_encryption.py: Tests the model's ability to learn highly non-linear, deterministic rules that are considered computationally "hard." RSA encryption is a typical example.
    
- generate_cellular_automata_1d.py: Used to generate evolution datasets for one-dimensional cellular automata (CA) to test the model's ability to learn and execute local, deterministic rules.
    
- generate_game_of_life_2d.py: Generates datasets for a two-dimensional cellular automaton—Conway's Game of Life. This task is more complex than 1D CA, requiring the model to understand neighborhood relationships in a 2D space.
    
- generate_cellular_automata_1d_multistate.py: As an extension of the 1D cellular automata experiment, this tests the model's ability to handle non-binary state spaces.
    
- generate_cellular_automata_programmable.py: Tests the model's "programmability" or "meta-learning" ability. The model must not only learn the CA evolution process but also execute the evolution according to different rules given in each input.
    
- generate_deduction_chain_text.py: Generates multi-step logical reasoning tasks to test the model's ability to perform symbolic deduction, similar to a simplified theorem prover.
    
- generate_deduction_multirule_text.py: Tests whether the model can correctly "route" to the appropriate rule and make a judgment based on a Query when faced with multiple independent, unrelated rules.
    
- generate_deduction_multirule_text_v2.py: Tests whether the model can correctly "route" to the appropriate rule and make a judgment based on a Query when faced with multiple independent, unrelated rules.
    
- generate_deduction_multirule_binary.py: This is a **format-optimized** version of the multi-rule reasoning task, aiming to test whether a compact binary encoding is more conducive to model learning than a sparse text format.
    
- generate_deduction_fixed_depth.py: Tests the model's multi-step reasoning ability in symbolic deduction tasks with a clear structure and fixed depth.
    
- generate_function_composition.py: Tests the model's ability to learn Function Composition. This requires the model to act like an interpreter, parsing instructions sequentially and transforming data.
    
- generate_cellular_automata_inverse_rule90.py: Tests the model's ability to solve an "Inverse Problem." Given the output of a deterministic system, the model needs to reverse-engineer a possible input that satisfies specific constraints (sparsest and unique).
    
- generate_count_set_bits.py: Tests the model's ability to perform a global aggregation operation. Unlike local rules, counting requires the model to synthesize information from the entire input sequence.
    
- generate_sum_pattern_positions.py: Tests the model's ability to perform a more complex, group-wise parallel aggregation task. The model needs to first segment the input, then classify each segmented pattern, and finally accumulate the **positional information** of patterns belonging to the same class.
    
- generate_sum_pattern_positions_v2.py: Tests the model's ability to perform a more complex, group-wise parallel aggregation task. The model needs to first segment the input, then classify each segmented pattern, and finally accumulate the **positional information** of patterns belonging to the same class.
    
- generate_sum_pairwise_hamming_distance.py: Tests the model's ability to perform a complex task requiring two levels of nested aggregation. The model needs to first perform a global statistic on **each bit position** and then accumulate the results from **all bit positions**.
    
- generate_circular_shift.py: Tests the model's ability to learn shift operations, particularly the circular shift, which is a common operation in cryptography and low-level programming.
    
- generate_multiply_matrix_3x3.py: Tests the model's ability to learn structured algebraic operations (matrix multiplication), which requires more complex "data routing" and "multiply-accumulate" capabilities than simple scalar operations.
    
- generate_evaluate_boolean_expression_text.py: Tests the model's ability to parse and evaluate a simple Domain-Specific Language (DSL), which is a step beyond evaluating fixed-structure expressions.
    
- generate_evaluate_arithmetic_expression.py: Trains the model to perform symbolic expression evaluation, which requires understanding operator precedence, variable substitution, and arithmetic operations.
    
- generate_evaluate_arithmetic_expression_no_multiply.py: This is a simplified version of the expression evaluation task, designed to reduce learning difficulty by removing the multiplication operation.
    
- generate_evaluate_arithmetic_expression_no_multiply_small_range.py: This is a further simplification of the previous "no-multiplication" version, designed to further reduce learning difficulty by narrowing the numerical range.
    
- generate_check_boolean_equivalence.py: Tests the model's ability to judge logical equivalence in Boolean algebra. This is an abstract symbolic reasoning task that requires the model to understand the structure of expressions and the laws of Boolean algebra.
    
- generate_polynomial_shift_coefficients.py: Tests the model's ability to learn an abstract algebraic transformation rule. This task requires the model to understand the internal structure of polynomial expansion.
    
- generate_convolution_2d.py: Tests the model's ability to learn 2D convolution (Conv2D), a fundamental image processing operation, and explores whether it can infer the hidden, fixed rule (i.e., the convolution kernel itself) from input-output pairs.
    
- generate_simple_block_cipher.py: Tests the model's ability to "crack" or learn a simple but non-trivial custom encryption algorithm. This task represents a class of complex symbolic transformation rules with high chaos and avalanche effects.
    
- generate_sin_function_float32.py: Tests the model's ability to fit a continuous, periodic, non-linear function (sin(x)), using the standard 32-bit floating-point format for input and output.
    
- generate_sin_function_float64_to_int12_deprecated.py: This is another encoding attempt for the sin function fitting task, aimed at exploring the effect of using higher-precision floating-point input and lower-precision quantized binary output on learning performance.
    
- generate_sin_function_float32_to_quantized_int.py: Tests the model's ability to fit a continuous, periodic, non-linear function (sin(x)) and explores the impact of different input/output encoding schemes on learning performance.
    
- generate_multiply_binary_modulo.py: As part of the basic arithmetic experiments, this tests the model's mastery of truncated multiplication (or modulo multiplication).
    
- generate_explainable_two_step_calculation.py: Tests the model's ability to output "intermediate steps" or a "chain of thought" for calculations, serving as a direct validation of "functional explainability."
    
- generate_chess_positions_by_random_moves.py: Rapidly generates a large number of plausible and legal Chinese Chess positions by simulating a completely random player.
    
- generate_chess_positions_by_random_placement.py: Generates a large number of atypical but mostly legal Chinese Chess positions by randomly placing pieces on the board (rather than simulating moves), used for stress-testing the model's robustness.
    
- generate_chess_positions_from_engine_self_play.py: Generates a large number of high-quality, strategically sound Chinese Chess positions (in FEN format), serving as a foundational dataset for training a chess AI.
    
- generate_preprocess_legal_moves.py: This is a data preprocessing script used to convert a dataset of positions in FEN format into a "legal move prediction" task that the model can directly learn from.
    
- generate_chess_resolve_check_task.py: Generates a dataset specifically for the "Resolving a Check" tactical scenario in Chinese Chess. This task requires the model, when in a state of check, to find all moves that can legally resolve the check.
    

## B. Algorithm Learning

- generate_sort_integers.py: Tests the model's ability to perform a basic sorting algorithm, a classic non-local algorithmic task that requires comparison and rearrangement of input elements.
    
- generate_edit_distance.py: Tests the model's ability to learn to solve dynamic programming problems. Edit distance is a typical DP problem that conceptually requires the model to construct a 2D solution matrix.
    
- generate_edit_distance_explainable.py: This is a core experiment for "functional explainability." It requires the model not only to give the final answer (the edit distance) but also to output the complete "chain of thought" (the editing process) to reach that answer.
    
- generate_maze_random_walls.py: Tests the model's basic pathfinding ability in randomly generated "porous" mazes.
    
- generate_maze_dense.py: Tests the model's ability to perform path planning in complex, "dense" mazes similar to those designed by humans, which is more challenging than random-wall mazes.
    
- generate_blocks_world_arbitrary_goal.py: Solves the classic "Blocks World" planning problem, a benchmark task in the field of AI planning. This version allows for specifying arbitrary initial and goal states.
    
- generate_blocks_world_fixed_goal.py: This is a simplification of the "Blocks World" task. By fixing the goal state, it aims to test the model's learning ability in a situation with a clear objective and a more structured state space.
    
- generate_blocks_world_fixed_goal_multilabel.py: Further improves the "Blocks World" task. By allowing for multiple optimal solutions, it tests the model's ability to handle multi-label classification problems, more realistically reflecting the possibility of equivalent optimal paths in planning problems.
    
- generate_blocks_world_fixed_goal_multilabel_fixed_format.py: This is the final optimized version of the "Blocks World" task. By improving the input representation, it aims to provide the model with a clearer and more structured learning target.
    
- generate_checkers_jump_1d.py: Solves a planning problem of moving checkers in a one-dimensional space, originating from a well-known paper by Apple, used to test the reasoning bottlenecks of large language models.
    
- generate_river_crossing_puzzle.py: Solves a classic constraint satisfaction and state-space search problem—"N couples crossing a river." This task originates from a paper by Apple, used to reveal the limitations of large language models on certain types of reasoning tasks.
    
- generate_trapping_rain_water_aggregate.py: This is a preliminary attempt to solve the "trapping rain water" algorithm problem, designed to test the model's ability to learn an aggregated output (rather than a decoupled output). Experimental results showed that requiring the model to directly output the total sum is much harder than outputting detailed information for each position.
    
- generate_trapping_rain_water_decoupled.py: Solves the classic "trapping rain water" algorithm problem (LeetCode Hard). The success of this task demonstrates the model's ability to learn complex algorithms that require global information and proves, through the idea of problem decoupling, the significant impact of output format design on model learning efficiency.
    
- generate_trapping_rain_water_2d.py: As an extension of the 1D "trapping rain water" problem, this solves the 2D version. This task requires the model to understand the concepts of "enclosure" and "boundary" in a 2D space, presenting a more complex global information processing challenge.
    
- generate_skyline_max_height_aggregate.py: This is a preliminary attempt to solve the "skyline" problem, requiring the model to predict only the maximum height value from the final heights of all buildings. This task is used to compare the learning difficulty of aggregated versus decoupled outputs.
    
- generate_skyline_all_heights_decoupled.py: Tests the model's ability to solve a global optimization problem with 1D spatial constraints. The problem is based on LeetCode's "Max-Height Skyline." By decoupling the output, it requires the model to predict the height of every single building, rather than just the maximum value.
    
- generate_hanoi_tower_path_strategy_sep_format.py: This is an early experimental script for the Tower of Hanoi problem, designed to test if the model can learn the strategy along the optimal path. It uses a separator-style input format and predicts the action as a 6-class classification problem.
    
- generate_hanoi_tower_global_strategy_fixed_format.py: As an improvement on the early Tower of Hanoi experiments, this script adopts a more model-friendly fixed-slot input format to verify the impact of input representation on learning efficiency.
    
- generate_hanoi_tower_compare_formats.py: This is a comparative experiment script that generates two different input formats (separator vs. fixed-slot) for the same Tower of Hanoi problem, used to systematically evaluate the impact of different data representations on the model's ability to learn a recursive strategy.
    
- generate_hanoi_tower_compare_formats_and_strategies.py: This is a more comprehensive comparative experiment script for the Tower of Hanoi. It not only generates two input formats but also two different datasets: one containing only states on the optimal path ("path strategy") and another containing all reachable states ("global strategy"), used to explore the difference in the model's ability to learn local optimal paths versus a global optimal strategy.
    
- generate_hanoi_tower_build_full_state_graph.py: This is a magnum opus for the "Tower of Hanoi problem" research, aiming to deeply analyze the model's understanding of recursive structures through various data representations and sampling strategies. It is a self-contained data factory.
    
- generate_hanoi_tower_sample_from_state_graph.py: This is a post-processing and sampling script that utilizes the complete knowledge base generated by generate_hanoi_tower_build_full_state_graph.py to precisely extract specific types of training data subsets, such as "twisted paths" or the "hardest parts," for more fine-grained ablation studies.
    
- generate_sokoban_planning_astar.py: Solves the classic "Sokoban" planning problem.
    
- generate_sokoban_planning_full.py: Solves the classic "Sokoban" planning problem. This is a highly difficult AI task as it involves searching in a vast state space where actions change the state of the environment.
    
- generate_sokoban_planning_claude_deprecated.py: This was an earlier, more logically complex attempt that failed to consistently generate high-quality datasets. (Deprecated)
    
- generate_min_swaps_for_checkerboard.py: Solves a highly constrained matrix rearrangement problem: finding the minimum number of swaps (by swapping any rows and any columns) required to turn a 0/1 matrix into a "checkerboard" pattern (adjacent elements are different).
    
- generate_min_flips_for_alternating_binary.py: Tests the model's ability to solve a string optimization problem based on bit flips, which can be cleverly mapped to and solved as a sliding window problem.
    
- generate_min_swaps_for_checkerboard_v2.py: Solves a highly constrained matrix rearrangement problem: finding the minimum number of swaps (by swapping any rows and any columns) required to turn a 0/1 matrix into a "checkerboard" pattern (adjacent elements are different).
    
- generate_matrix_flip_strategy.py: Solves a classic matrix optimization problem (maximizing the number of 1s). This version aims to test if the model can learn a "strategy" rather than just the final result.
    
- generate_matrix_flip_max_score.py: Tests the model's ability to learn a matrix optimization problem that requires a two-step greedy strategy (flip rows first, then columns) to achieve a global optimum. This version requires the model to directly output the final aggregated result (the score).
    
- generate_min_prefix_flips.py: Tests the model's ability to learn a sequential, history-dependent greedy algorithm.
    
- generate_min_k_bit_flips.py: Tests the model's ability to learn a sequential, history-dependent greedy algorithm, and also tests if it can use one part of the input (k) as a "parameter" to guide the processing of another part (nums).
    
- generate_min_k_bit_flips_fixed_k.py: Tests the model's ability to learn a sequential, history-dependent greedy algorithm. In this version, the environmental parameter (k=2) is fixed and hidden; the model must learn it implicitly from the data.
    
- generate_special_binary_string_recursion.py: Tests the model's ability to learn a recursively defined string transformation rule. This problem (LeetCode Hard "Special Binary String") requires recursive decomposition and reassembly of the input.
    
- generate_min_flips_for_chunked_binary.py: Tests the model's ability to learn a string transformation optimization problem based on local chunks.
    
- generate_count_connected_components.py: Tests the model's basic understanding of graph structures, particularly the core concept of "connectivity."
    
- generate_check_graph_connectivity.py: This is another core test of the model's foundational graph theory capabilities. The task is to determine if a path exists between any two points in a graph.
    
- generate_minimize_malware_spread.py: Solves a graph-based virus spread optimization problem (LeetCode Hard "Minimize Malware Spread"). The model needs to understand graph connectivity and evaluate the impact of removing different nodes on the global spread.
    
- generate_count_islands_1d.py: Tests the model's ability to perform pattern recognition and counting on a 1D sequence.
    
- generate_largest_island_by_adding_one_cell.py: Solves an algorithmic problem involving graph traversal and global optimization (LeetCode 827). The model needs to evaluate all possible "land reclamation" positions and select the one that results in the largest merged island area.
    
- generate_largest_island_by_adding_one_cell_v2.py: Solves an algorithmic problem involving graph traversal and global optimization (LeetCode 827). The model needs to evaluate all possible "land reclamation" positions and select the one that results in the largest merged island area.
    
- generate_find_articulation_points.py: Tests the model's ability to identify "Articulation Points" or "Bridges" in a graph, an important concept in graph theory.
    
- generate_nim_game_zeckendorf.py: This experiment aims to test if my paradigm can learn a non-intuitive game theory problem based on complex number theory (Zeckendorf's representation). It moves beyond simple pattern matching and requires the model to understand deeper mathematical structures.
    
- generate_longest_subsequence_constrained.py: Tests the model's ability to handle a complex optimization problem that mixes sequence operations and numerical constraints.
    
- generate_treasure_hunt_tsp.py: Solves a complex state-space search problem that combines graph traversal (BFS) and combinatorial optimization (state compression DP), a classic difficult problem in competitive programming.
    
- generate_freedom_trail_dp.py: Tests the model's ability to learn to solve a complex optimization problem that requires dynamic programming and path backtracking.
    
- generate_sum_of_subset_with_mask.py: Tests the model's ability to select elements from a set based on a binary mask and perform an aggregation operation (summation).
    
- generate_sudoku_6x6.py: Tests the model's ability to handle a strong Constraint Satisfaction Problem—Sudoku.
    
- generate_valid_parentheses_path_random_deprecated.py: This was an early attempt to solve the "valid parentheses path" problem. (Early exploration/Deprecated)
    
- generate_valid_parentheses_path_balanced.py: Solves a pathfinding problem on a 2D grid where the validity of the path is constrained by a stack-like structure (parentheses matching).
    
- generate_sat_solver_text.py: Tests the model's ability to solve a landmark NP-complete problem—Boolean Satisfiability (SAT).
    
- generate_sat_solver_compact_text.py: This is a variant of generate_sat_solver_text.py that uses a different input encoding format to solve the same 3-SAT problem.
    
- generate_point_in_polygon.py: Tests the model's ability to learn a classic algorithm in computational geometry—the Ray Casting Algorithm.
    
- generate_shortest_path_in_matrix_bfs.py: Tests the model's ability to find the shortest path in a 2D grid based on the classic Breadth-First Search (BFS) algorithm.
    
- generate_sudoku_4x4_stepwise_deprecated.py: Aimed to test the model's ability to perform "stepwise" reasoning. (Deprecated)
    
- generate_tiling_problem_deprecated.py: Aimed to test the model's ability to solve a classic tiling optimization problem, which is NP-hard. (Deprecated)
    
- generate_hanoi_tower_twisted_path_deprecated.py: This script was intended to generate a "twisted path" dataset for the Tower of Hanoi problem. (Deprecated)
    
- generate_checkers_jump_1d_v2.py: Solves a checker-swapping planning problem in a 1D space, which has been used to reveal the limitations of large language models on certain types of reasoning tasks.
    
## C. Image to Symbol

- generate_checkerboard_to_binary.py: This is a basic vision-to-symbol conversion task, used to test the model's ability to decode structured information from raw pixel data.
    
- generate_line_angle_to_vector.py: Tests the model's ability to extract precise geometric information (angles) from an image, which is a more advanced visual reasoning task than simply recognizing a checkerboard pattern.
    
- generate_count_shapes_from_image.py: Tests the model's ability to perform multiple visual tasks simultaneously: object recognition (shape), attribute recognition (color), and counting (aggregation).
    
- generate_maze_symbolic_to_image.py: Converts a symbolic maze pathfinding dataset into an image format to test the ability of visual models (like CNNs, ViTs) to perform path planning directly from pixels.
    
- generate_sokoban_symbolic_to_image_no_labels.py: This is a data conversion script that converts a symbolic Sokoban dataset (.jsonl format) into only image format, for use in pure vision tasks or as an intermediate step in more complex data processing.
    
- generate_sokoban_symbolic_to_image_with_labels.py: This is a data conversion script that converts a symbolic Sokoban dataset (.jsonl format) into a complete image classification dataset for training computer vision models (like ViT, Swin Transformer).
    
- generate_cellular_automata_image_and_label.py: A general-purpose dataset generator. It generates data for cellular automata (CA) tasks in both image format (Img2Img) and symbolic format (Img2Label), with support for multiprocessing acceleration.
    
- generate_trapping_rain_water_image_to_symbol.py: A specialized dataset generator. It generates image-format data for the "trapping rain water" problem, where the input is a grid image of pillar heights and the output is a symbolic label of the water amount.
    

## D. Image to Image

- generate_triangle_to_incircle.py: This is a landmark experiment demonstrating "carving out precise rules with gradient descent." It tests whether the model can learn a purely non-trivial geometric construction rule (the incircle of a triangle).
    
- generate_polygon_to_symmetry_axis.py: Tests the model's ability to infer the implicit axis of symmetry from a complete symmetrical figure.
    
- generate_triangle_to_centroid.py: Tests the model's ability to learn another fundamental geometric concept—the centroid.
    
- generate_triangle_to_tessellation.py: This is a landmark demonstration of our paradigm's capability. It tests whether the model can learn an infinite, lattice-based generation rule. Due to the global correlations and precise details of the tessellation pattern, it strongly rules out the possibility that the model is merely solving the problem through "interpolation" or "memorization."
    
- generate_game_of_life_image_to_image.py: This is the image-to-image version of the 2D cellular automaton, testing whether the model can perform local rule-based evolution directly in pixel space.
    
- generate_projectile_motion_simulation.py: Tests the model's ability to learn a simple dynamic physical process. This requires the model to infer the entire spatio-temporal trajectory from initial conditions (position and velocity vectors).
    
- generate_snell_refraction_simulation.py: Tests the model's ability to learn a fundamental physical law (Snell's Law of refraction).
    
- generate_snell_refraction_with_contextual_index.py: Tests the model's ability to learn a fundamental physical law (Snell's Law of refraction) and requires the model to infer physical parameters (refractive index) from the image's contextual information (background color).
    
- generate_cellular_automata_spatial_conditional.py: Tests the model's ability to partition and parse "instructions" and "data" within a single modality (image), serving as a "pseudo-multimodal" or "spatially-conditioned" experiment.
    
- generate_trapping_rain_water_visualizer.py: This is a **data conversion and visualization** script. Its purpose is to convert an already generated, symbolic "trapping rain water" dataset into an image-to-image format dataset, so the same problem can be solved with a visual model.
    
- generate_shortest_path_in_tree_deprecated.py: This was an early experiment aimed at testing the model's ability to find the shortest path in a graph from an image. (Early exploration/Deprecated)
    
- generate_shortest_distance_between_triangles.py: Tests the model's ability to reason about global geometric relationships (shortest distance) in a scene containing multiple objects.
    
- generate_reaction_diffusion_deprecated.py: This script was used to simulate a reaction-diffusion system to generate complex, fractal-like "snowflake" patterns. (Exploratory/Deprecated)
    
- generate_cellular_automata_multimodal_deprecated.py: Generates a truly multimodal dataset for training a model that can understand both image input and text instructions simultaneously. (Deprecated)
    
- generate_cellular_automata_1d_to_grid_image_interp.py: This script is designed to create a "logic/perception hybrid" task to demonstrate that a neural network's rule-learning and interpolation abilities are not mutually exclusive but can be integrated within a single task. It forces the model to simultaneously "see through" the continuous grayscale values of the input to perform discrete logical reasoning, and to remember these grayscale values to complete the final continuous value mapping.
    

## E. Text to Image

- generate_coords_to_triangle.py: This is a basic symbol-to-geometry rendering task, testing the model's ability to convert abstract coordinate information into concrete pixel shapes.
    
- generate_cellular_automata_1d_to_grid_image.py: Tests whether the model can directly "render" 1D symbolic computation results into a structured 2D image.
    
- generate_triangle_coords_to_tessellation.py: This is an advanced reasoning task that mixes symbolic instructions with geometric generation rules.
    
- generate_cube_rotation_matplotlib_deprecated.py: Aimed to test the model's ability to infer and render the correct view of a 3D object from abstract pose parameters (rotation angles). (Early exploration version)
    
- generate_cube_rotation_pillow_v1.py: Aimed to test the model's ability to infer and render the correct view of a 3D object from abstract pose parameters, using a more low-level and precise rendering technique. (Technical upgrade version)
    
- generate_cube_rotation_pillow_with_anchor.py: Tests the model's ability to infer and render the correct view of a 3D object from abstract pose parameters, using "visual anchors" to aid the model's learning. (Final version used in the paper)
    
- generate_cube_rotation_pillow_wireframe.py: Tests whether the model can learn 3D rotation from sparser visual input, using only wireframes and anchor points. (Variant experiment version)
    

## F. Physics Simulation - Image Paradigm

- generate_catenary_curve_simulation_deprecated.py: This is my early script for exploring the catenary problem, aiming to test the model's ability to learn non-linear curves determined by physical laws.
    
- generate_catenary_curve_from_points.py: Tests the model's ability to learn a non-linear curve (catenary) uniquely determined by physical laws (the principle of minimum potential energy).
    
- generate_orbital_path_from_initial_state.py: Tests the model's ability to learn more complex physical laws (Kepler's Laws / Law of Universal Gravitation).
    

## G. ARC-AGI Exploration

- generate_arc_contextual_color_swap.py: Tests the model's ability to learn a rule from a local "context" or "example" within an image and apply it to the global data of the same image. This directly mimics the core philosophy of the ARC-AGI test.
    
- generate_arc_find_cross_pattern.py: Tests the model's ability to perform visual pattern recognition (or "object detection") in the presence of significant noise.
    
- generate_arc_find_odd_one_out.py: Tests the model's ability to perform a complex "Find the Odd One Out" meta-reasoning task. The model needs to compare patterns row by row, identify the exception, and reassemble it in the output.
    
- generate_arc_connect_colored_pairs.py: Tests the model's ability to identify multiple independent "connection tasks" within the same image and understand an implicit rule of "layering" or "drawing priority."
    
- generate_arc_conditional_perpendicular_lines.py: Tests the model's ability to perform different geometric operations based on an object's **attributes (color)** and **global references (boundary lines, image edges)**.
    
- generate_arc_column_projection.py: Tests the model's ability to recognize complex contextual relationships ("below..." and "within the range of...") and perform conditional column operations.
    
- generate_arc_procedural_spiral.py: Tests the model's ability to execute an iterative, procedural generation algorithm. The model needs to understand instructions, track state (current position, direction, length), and execute in a loop.
    
- generate_arc_fractal_stamping.py: Tests the model's ability to understand and execute recursive or fractal generation rules. The model needs to use the input pattern itself as a "brush" and repeatedly draw it according to "instructions" within the input pattern.
    
- generate_arc_flood_fill.py: Tests the model's ability to execute the classic "Flood Fill" or "Paint Bucket" algorithm.
    
- generate_arc_layered_fill.py: Tests the model's ability to understand a highly procedural and complex filling algorithm that depends on topological distance and conditional logic.
    
- generate_arc_fluid_simulation.py: Tests the model's ability to learn and simulate a fluid dynamics process with specific rules in the image space.
    
- generate_arc_periodic_conditional_fill.py: Tests the model's ability to learn a complex conditional formatting rule with periodicity and special cases.
    
- generate_arc_fill_square_holes.py: Tests the model's ability to perform multi-step visual reasoning: first, identify "holes," then determine their geometric properties (whether they are squares), and finally, act based on that determination.
    
- generate_arc_conditional_recoloring.py: Tests the model's ability to understand visual layers and perform conditional modification of object properties.
    
- generate_arc_sort_by_length_remap_position.py: Tests the model's ability to perform a complex sorting task involving "attribute-position decoupling and remapping."
    
- generate_arc_jigsaw_puzzle_simple.py: Tests the model's ability to solve a visual matching and transformation problem (early version), where the puzzle pieces have unique sizes that can be used as a shortcut for matching.
    
- generate_arc_jigsaw_puzzle_advanced.py: Tests the model's ability to solve a complex visual matching and transformation problem. This version requires the model to genuinely match based on shape, not just size.
    
- generate_arc_connect_path_by_sequence.py: Tests the model's ability to parse an external sequence of instructions and execute a multi-step, stateful path connection task within an image accordingly.
    
- generate_arc_reflection_simulation_deprecated.py: Aimed to test the model's understanding of complex physics-based optics rules, including ray emission, collision detection, angle reflection, and color transformation. (Deprecated)
    

## H. Inverse Rule Inference

- generate_cellular_automata_inverse_rule.py: This experiment is the first attempt to test the model's **Inverse Reasoning** ability. The question is: if the model can forward-propagate from a rule to a result, can it reverse-engineer the underlying rule from "input-output" pairs?
    
- generate_cellular_automata_inverse_rule_and_steps.py: This is an earlier version before the "unique solution" version was implemented. It also aimed to have the model learn to predict the rule and the number of iterations.
    
- generate_cellular_automata_inverse_rule_and_steps_unique.py: This is a major upgrade to the inverse reasoning task. I not only require the model to infer **what** rule was applied, but also **how many times** it was applied.
    

## I. Robustness Analysis

- generate_cellular_automata_1d_perturbed.py: This script aims to systematically test the robustness of the "neural carving" paradigm when faced with imperfect data. By introducing controllable random perturbations to the input (simulating observation noise) and the output (simulating label noise), it explores the continuous spectrum of model performance transitioning from an ideal rule-based world to a noisy real world.
    

## J. Training Script

- train_tiny_transformer.py: For symbol-to-symbol tasks, trains a small, from-scratch Transformer model.
    
- train_swin_image2text.py: For image-to-symbol tasks, fully fine-tunes a Swin Transformer for multi-label classification.
    
- train_unet.py: For pure image-to-image tasks, trains a standard U-Net model.
    
- train_text2image.py: For symbol-to-image tasks, trains a combination of a from-scratch Transformer encoder and a U-Net decoder.
    
- train_qwen2_text2image.py: For symbol-to-image tasks, uses LoRA to efficiently fine-tune Qwen2 as the encoder, combined with a U-Net decoder, and managed by the Hugging Face Trainer.
    
- train_mlp.py: For symbol-to-symbol tasks, uses a very large MLP as a performance baseline.
    
- train_lstm.py: For symbol-to-symbol tasks, uses an LSTM to test its sequential memory capabilities.
    
- train_convnext.py: For image-to-symbol tasks, uses the advanced CNN architecture ConvNeXt as the visual feature extractor.
    
- train_diffusion.py: For image-to-image tasks, uses a conditional Diffusion model for generative inference.
    
- train_image2image.py: The core image-to-image task script, using a Swin Transformer + U-Net decoder (i.e., Swin-Unet).
    

## K. Utility Script

- eval_hanoi.py: An experiment validation tool. Used to verify whether the Tower of Hanoi solution strings generated by a large language model adhere to the game rules and ultimately reach the goal state.