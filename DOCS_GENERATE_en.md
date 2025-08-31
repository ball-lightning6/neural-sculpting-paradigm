# A: Symbolic Rule Learning

## 1. **generate_conditional_add_subtract.py**

- **Logic:** The script generates addition or subtraction (absolute value) problems for two N-bit integers. It includes two modes:
    1. **INDICATOR_BIT Mode:** An extra bit is prepended to the input (0 for addition, 1 for subtraction) as an explicit instruction.
    2. **PROBABILITY_MIX Mode:** No instruction is provided, but addition and subtraction samples are mixed at a certain probability during data generation. This is to simulate a "rule-impure" environment.
- **I/O Format:**
    - Input: (INDICATOR_BIT Mode) 1 (indicator bit) + 2N (operands) bits; (PROBABILITY_MIX Mode) 2N bits.
    - Output: An N+1 bit binary multi-label vector representing the calculation result.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`, `EXPERIMENT_MODE`, `PROBABILITY_ADD`.

---

## 2. **generate_add_binary_modulo.py**

- **Purpose:** An early, fundamental arithmetic experiment to test the model's ability to learn modular addition (or "truncated addition"), an operation common in fixed-width integer arithmetic in computer hardware.
- **Logic:** Takes two N-bit integers, `a` and `b`, as input, calculates their sum, and then performs a modulo 2^N operation on the result, effectively discarding any overflow carry (e.g., the N+1st bit).
- **I/O Format:**
    - Input: A binary string of length `bit_length` * 2.
    - Output: A binary multi-label vector of length `bit_length`.
- **Main Parameters:** `n_samples`, `bit_length`.

---

## 3. **generate_multiply_binary.py**

- **Purpose:** Serves as a benchmark for binary arithmetic capabilities by generating a dataset for N-bit integer multiplication.
- **Logic:** Randomly generates two NUM_BITS integers, concatenates their binary strings as input, and uses the binary representation of their product as the output.
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS` * 2.
    - Output: A binary multi-label vector of length `NUM_BITS` * 2.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 4. **generate_multiply_binary_no_carry_phase1.py**

- **Purpose:** This is the first phase of the multiplication "decoupling" experiment. It aims to test whether the model can learn the first step of multiplication: bitwise multiplication and shifted addition without carrying, breaking down a complex multiplication problem into a simpler counting problem.
- **Logic:** Simulates the process of long multiplication by hand. The input is two N-bit numbers, `a` and `b`. The output is not the final product, but a counter vector of length 2*N, where the i-th counter records how many '1's the i-th bit of the final product should have before carrying.
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS` * 2.
    - Output: A binary multi-label vector of length (`NUM_BITS` * 2) * `BITS_PER_COUNTER`.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 5. **generate_multiply_binary_from_counts_phase2.py**

- **Purpose:** This is the second phase of the multiplication "decoupling" experiment. It aims to verify whether a separate model can learn to handle complex carry logic, i.e., calculating the final binary product from a "no-carry count vector."
- **Logic:** The script's input is the output format of the previous "no-carry multiplication" task (a counter vector). It calculates the value represented by this vector (i.e., the original a*b) and outputs its standard binary representation.
- **I/O Format:**
    - Input: A binary string of length 2N * `BITS_PER_COUNTER`, representing the no-carry count values.
    - Output: A binary multi-label vector of length 2*N, representing the final product.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 6. **generate_add_hexadecimal.py**

- **Purpose:** To compare the model's learning ability across different symbolic systems. This script aims to verify whether the model is learning the abstract mathematical concept of addition or merely patterns specific to binary symbols.
- **Logic:** Randomly selects two 16-bit integers. The script generates two independent datasets: one with the binary representation of these numbers as input, and the other with their hexadecimal string representation. The output for both datasets is identical: the 17-bit binary representation of the sum.
- **I/O Format:**
    - Input (Binary): 32 ('0'/'1') | Input (Hexadecimal): 8 ('0'-'9','A'-'F').
    - Output: A 17-bit binary multi-label vector.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 7. **generate_multiply_decimal.py**

- **Purpose:** To test the model's ability to process non-binary symbolic input (characters '0'-'9') and perform arithmetic operations (multiplication).
- **Logic:** Generates two N-digit decimal numbers and concatenates their strings as input. It calculates their product and converts the result to binary as the output.
- **I/O Format:**
    - Input: A string of length `NUM_DIGITS` * 2, composed of '0'-'9'.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`.
- **Main Parameters:** `NUM_DIGITS`, `DATASET_SIZE`.

---

## 8. **generate_add_n_base_with_shuffle.py (Actual filename: generate_symbol_add_shuffle_dataset.py)**

- **Purpose:** This is a **key decisive experiment** in our research, designed to completely separate the model's "surface pattern matching" ability from its "abstract structure learning" ability.
- **Logic:** The script can be configured to perform two types of "shuffling":
    1. **Semantic Shuffle:** Randomly maps the symbols representing N-base digits (e.g., '0'-'F') to arbitrary printable characters. This severs the connection between the symbols and their inherent numerical meaning.
    2. **Positional Shuffle:** Rearranges the position of each character in the input string according to a fixed random mapping. This destroys all local, spatial statistical regularities.
- **I/O Format:**
    - Input: A string of length 2 * `NUM_BITS` (character set is variable).
    - Output: A binary multi-label vector of the sum.
- **Main Parameters:** `NUM_BITS`, `BASE`, `SHUFFLE_SEMANTICS`, `SHUFFLE_POSITIONS`.

---

## 9. **generate_add_binary_with_position_shuffle.py**

- **Purpose:** This is the "positional shuffle" part of the "semantic shuffle" series of experiments. It aims to verify whether the model relies on the fixed spatial structure of the input or can learn position-independent abstract relationships.
- **Logic:** The script generates two datasets for a binary addition task. One set's input is two N-bit numbers concatenated in standard order (a+b). The other set's input is created by rearranging each bit of the standard input according to a predefined, fixed random mapping. The output is the same for both datasets.
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS` * 2.
    - Output: A binary multi-label vector of length `NUM_BITS` + 1.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 10. **generate_add_hidden_constant.py**

- **Purpose:** To test the model's ability to **infer a hidden rule or parameter** from a large number of samples without any direct clues. This is similar to a simplified System Identification problem.
- **Logic:** The script defines a fixed "hidden constant" C internally. For each sample, it only takes a random number `x` as input and provides the result of `x+C` as output. The model must learn the commonality across all samples to encode the effect of the constant C into its weights.
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS` (representing x).
    - Output: A binary multi-label vector of length `NUM_BITS`+1 (representing x+C).
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 11. **generate_multitask_alu.py**

- **Purpose:** This script aims to construct a multi-task learning scenario simulating an **Arithmetic Logic Unit (ALU)**. It tests whether the model can, in a single forward pass, perform multiple different, well-defined computational tasks in parallel on the same input.
- **Logic:** The input consists of two N-bit binary numbers. The output is a long binary vector, divided into multiple "address segments," with each segment corresponding to the result of a specific operation (add, subtract, AND, OR, XOR, compare). This forces the model to internally fork its computational graph and route the results to the specified output locations.
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS` * 2.
    - Output: A binary multi-label vector of length `TOTAL_OUTPUT_BITS`, formed by concatenating the results of all tasks.
- **Main Parameters:** `NUM_BITS`, `DATASET_SIZE`.

---

## 12. **generate_modulo_operation.py**

- **Purpose:** To investigate the model's ability to learn the Modulo Operation, an operation that is crucial in number theory and computer science but has a "cyclical" nature.
- **Logic:** The script generates an A_BITS integer `a` and an N_BITS integer `n`, with the task being to compute `a % n`. In subsequent explorations, we fixed `n` to 3 to delve deeper into why the model struggled to learn this seemingly simple rule.
- **I/O Format:**
    - Input: A binary string of length `A_BITS` (current version only inputs `a`).
    - Output: A binary multi-label vector of length `N_BITS`.
- **Main Parameters:** `A_BITS`, `N_BITS` (though currently fixed at 3), `DATASET_SIZE`.

---

## 13. **generate_rsa_encryption.py**

- **Purpose:** To test the model's ability to learn a highly non-linear, deterministic rule that is considered computationally "hard." RSA encryption is a classic example.
- **Logic:** The script encrypts all possible messages (`m`) from 0 to n-1 under a fixed public key (e, n), generating the corresponding ciphertexts (`c`).
- **I/O Format:**
    - Input: A binary string of `bits` length, representing message `m`.
    - Output: A binary string of `bits` length, representing ciphertext `c`.
- **Main Parameters:** `e`, `n` (public key parameters), `bits` (encoding bits), `output_file`.

---

## 14. **generate_cellular_automata_1d.py**

- **Purpose:** To generate evolution datasets for one-dimensional cellular automata (CA) to test the model's ability to learn and execute local, deterministic rules.
- **Logic:** Given a random binary initial state, the script iterates for a specified number of layers (steps) according to a specified evolution rule (currently Rule 110) and generates the final state.
- **I/O Format:**
    - Input: A binary string of `length` bits, representing the initial state.
    - Output: A binary multi-label vector of `length` bits, representing the final state.
- **Main Parameters:** `num_samples`, `length`, `l` (evolution layers).

---

## 15. **generate_game_of_life_2d.py**

- **Purpose:** To generate datasets for a two-dimensional cellular automaton—Conway's Game of Life. This task is more complex than 1D CA, requiring the model to understand neighborhood relationships in a 2D space.
- **Logic:** Based on a random n*n initial board, it evolves for `d` time steps according to the standard Game of Life rules (B3/S23) and records the final board state.
- **I/O Format:**
    - Input: A flattened binary string of `n*n` bits, representing the initial board.
    - Output: A binary multi-label vector of `n*n` bits, representing the final board.
- **Main Parameters:** `num_samples`, `n` (grid side length), `d` (evolution steps).

---

## 16. **generate_cellular_automata_1d_multistate.py**

- **Purpose:** An extension of the 1D cellular automaton experiment, testing the model's ability to handle non-binary state spaces.
- **Logic:** The evolution rule for the CA is: a cell's next state is its current state + 1 (modulo n_states) if and only if its left or right neighbor's state equals this target state. The script generates the input (initial state) and output (final state) of this evolution process.
- **I/O Format:**
    - Input: A binary string of length `n_cells` * 2 (because n_states=4).
    - Output: A binary multi-label vector of length `n_cells` * 2.
- **Main Parameters:** `n_cells`, `n_states`, `n_samples`, `steps`.

---

## 17. **generate_cellular_automata_programmable.py**

- **Purpose:** To test the model's "programmability" or "meta-learning" ability. The model must not only learn the CA evolution process but also be able to execute the evolution according to a different rule provided in each input.
- **Logic:** Each sample's input consists of an 8-bit rule number and an initial state. The script evolves the state according to that rule to generate the output. This requires the model to understand one part of the input as "program" and the other part as "data."
- **I/O Format:**
    - Input: An 8 (rule) + `CA_WIDTH` (state) bit binary string.
    - Output: A multi-label binary vector of `CA_WIDTH` bits.
- **Main Parameters:** `TARGET_NUM_SAMPLES`, `CA_WIDTH`, `EVOLUTION_STEPS`, `RULES_TO_INCLUDE` (specifies which rules are included in the dataset).

---

## 18. **generate_deduction_chain_text.py**

- **Purpose:** To generate a multi-step logical reasoning task to test the model's ability to perform symbolic deduction, similar to a simplified theorem prover.
- **Logic:** The script defines a series of implicit inference rules (e.g., (A, B) -> C). It first constructs a multi-step reasoning chain (e.g., 5 steps), then determines all the initial "facts" required to derive the final conclusion. A positive sample's input contains all these necessary facts (and possibly some irrelevant "noise" facts), while a negative sample will deliberately lack one or more key facts. The model's task is to determine if a given Query can be deduced from the given facts and rules.
- **I/O Format:**
    - Input: A text string in the format "Facts: ...\nRules: ...\nQuery: ...".
    - Output: '1' (can be deduced) or '0' (cannot be deduced).
- **Main Parameters:** `num_samples`, `attr_range`, `depth`.

---

## 19. **generate_deduction_multirule_text.py**

- **Purpose:** To test whether the model can correctly "route" to the appropriate rule and make a judgment based on the Query when faced with multiple independent, unrelated rules.
- **Logic:** The script defines two independent sets of "fact -> conclusion" rules. When generating each sample, it first determines the target to be queried (5 or 6), then checks the prerequisites for inferring that target. Positive samples will provide all necessary conditions (plus some noise facts), while negative samples will deliberately lack at least one necessary condition.
- **I/O Format:**
    - Input: A text string in the format "Facts: ..., Query: ...".
    - Output: A single character '1' (can be deduced) or '0' (cannot be deduced).
- **Main Parameters:** `n_samples`.

---

## 20. **generate_deduction_multirule_text_v2.py**

- **Purpose:** To test whether the model can correctly "route" to the appropriate rule and make a judgment based on the Query when faced with multiple independent, unrelated rules.
- **Logic:** The script defines two independent sets of "fact -> conclusion" rules. When generating each sample, it first determines the target to be queried (5 or 6), then checks the prerequisites for inferring that target. Positive samples will provide all necessary conditions (plus some noise facts), while negative samples will deliberately lack at least one necessary condition.
- **I/O Format:**
    - Input: A text string in the format "Facts: ..., Query: ...".
    - Output: A single character '1' (can be deduced) or '0' (cannot be deduced).
- **Main Parameters:** `n_samples`.

---

## 21. **generate_deduction_multirule_binary.py**

- **Purpose:** This is a **format-optimized** version of the multi-rule reasoning task, designed to test whether a compact binary encoding is more conducive to model learning than a sparse text format.
- **Logic:** The core logic is consistent with the text version, but the input/output representation is changed:
    1. All 8 possible facts are represented as an 8-bit binary mask.
    2. The query target (5 or 6) is represented as a single binary bit.
    3. These two parts are concatenated into a 9-bit input string.
- **I/O Format:**
    - Input: 8 (fact mask) + 1 (query target encoding) = 9-bit binary string.
    - Output: A single character '1' or '0'.
- **Main Parameters:** `n_samples`.

---

## 22. **generate_deduction_fixed_depth.py**

- **Purpose:** To test the model's multi-step reasoning ability in a clearly structured, fixed-depth symbolic deduction task.
- **Logic:** The script first internally generates a random 5-step reasoning chain (e.g., A+B->X, C+D->Y, ..., X+Y->Z). Then, using a "backchaining" method, it traces back from the final conclusion (Z) to find all initial facts that must be true.
    - **Positive Sample:** Input contains the mask of all necessary facts, the query target is Z, and the label is '1'.
    - **Negative Sample:** Input contains the same fact mask, but the query target is a "noise" attribute that cannot be inferred from these facts, and the label is '0'.
- **I/O Format:**
    - Input: 16 (fact mask) + 4 (query target encoding) = 20-bit binary string.
    - Output: A single character '1' (can be deduced) or '0' (cannot be deduced).
- **Main Parameters:** `depth`, `num_attrs`, `num_samples`.

---

## 23. **generate_function_composition.py**

- **Purpose:** To test the model's ability to learn Function Composition. This requires the model to act like an interpreter, parsing instructions sequentially and transforming data accordingly.
- **Logic:** The script defines four basic functions (double, increment, square, decrement). Each sample's input consists of two parts: an instruction string representing a sequence of 4 function calls (each function encoded with 2 bits), and a 16-bit initial integer. The script applies these 4 functions in order, ensuring that the intermediate result at each step remains within the range, and outputs the final result.
- **I/O Format:**
    - Input: (4 * 2) (function instructions) + 16 (initial value) = 24-bit binary string.
    - Output: A 16-bit binary string.
- **Main Parameters:** `num_samples`.

---

## 24. **generate_cellular_automata_inverse_rule90.py**

- **Purpose:** To test the model's ability to solve an "Inverse Problem." Given the output of a deterministic system, the model needs to infer a possible input that satisfies a specific constraint (the sparsest and unique one).
- **Logic:** The input is the state of a 1D cellular automaton (Rule 90) after one step of evolution. The task is to find the "previous state" among all possibilities that has the minimum number of '1's (is the sparsest). To ensure a unique solution, the script uses a brute-force search to only keep samples where there is exactly one "sparsest solution."
- **I/O Format:**
    - Input: A binary string of `length` bits (post-evolution state).
    - Output: A binary string of `length`+2 bits (pre-evolution state).
- **Main Parameters:** `num_samples`, `length`.

---

## 25. **generate_count_set_bits.py**

- **Purpose:** To test the model's ability to perform a global aggregation operation. Unlike local rules, counting requires the model to synthesize information from the entire input sequence.
- **Logic:** The script generates a random binary string and counts the number of '1's in it. The `balanced` mode can ensure that the number of samples for each count value in the dataset is roughly equal.
- **I/O Format:**
    - Input: A binary string of length `input_bits`.
    - Output: A binary multi-label vector of length `output_bits`, representing the total count of '1's.
- **Main Parameters:** `num_samples`, `input_bits`, `output_bits`, `balanced`.

---

## 26. **generate_sum_pattern_positions.py**

- **Purpose:** To test the model's ability to perform a more complex, group-wise parallel aggregation task. The model needs to first segment the input, then classify each segmented pattern, and finally aggregate the **positional information** of patterns belonging to the same class.
- **Logic:** The script splits a long binary string into `q` (`NUM_PATTERNS`) consecutive sub-patterns of `p` bits (`BITS_PER_PATTERN`). Then, for each possible sub-pattern (2^p in total), it calculates the sum of the indices (from 1 to q) of all occurrences of that pattern in the input.
- **I/O Format:**
    - Input: A binary string of length `p` * `q`.
    - Output: A binary multi-label vector of length (2^p) * `BITS_PER_SUM`, representing the sum of positions for each pattern type.
- **Main Parameters:** `BITS_PER_PATTERN`, `NUM_PATTERNS`, `DATASET_SIZE`.

---

## 27. **generate_sum_pattern_positions_v2.py**

- **Purpose:** To test the model's ability to perform a more complex, group-wise parallel aggregation task. The model needs to first segment the input, then classify each segmented pattern, and finally aggregate the **positional information** of patterns belonging to the same class.
- **Logic:** The script splits a long binary string into `q` (`NUM_PATTERNS`) consecutive sub-patterns of `p` bits (`BITS_PER_PATTERN`). Then, for each possible sub-pattern (2^p in total), it calculates the sum of the indices (from 1 to q) of all occurrences of that pattern in the input.
- **I/O Format:**
    - Input: A binary string of length `p` * `q`.
    - Output: A binary multi-label vector of length (2^p) * `BITS_PER_SUM`, representing the sum of positions for each pattern type.
- **Main Parameters:** `BITS_PER_PATTERN`, `NUM_PATTERNS`, `DATASET_SIZE`.

---

## 28. **generate_sum_pairwise_hamming_distance.py**

- **Purpose:** To test the model's ability to perform a complex task requiring two levels of nested aggregation. The model needs to first perform a global statistic on **each bit position** and then aggregate the results from **all bit positions**.
- **Logic:** The input is a string formed by concatenating N M-bit binary numbers. The task is to calculate the sum of Hamming distances for all pairs of these N numbers. For example, for [A, B, C], it needs to compute dist(A,B) + dist(A,C) + dist(B,C). The script uses an efficient O(N*M) algorithm to compute this value.
- **I/O Format:**
    - Input: A binary string of length `NUM_ITEMS` * `BITS_PER_ITEM`.
    - Output: A binary multi-label vector of `OUTPUT_BITS` bits, representing the total Hamming distance.
- **Main Parameters:** `NUM_ITEMS`, `BITS_PER_ITEM`, `DATASET_SIZE`.

---

## 29. **generate_circular_shift.py**

- **Purpose:** To test the model's ability to learn bit shift operations, particularly circular shifts, which are common operations in cryptography and low-level programming.
- **Logic:** The input consists of two concatenated parts: a binary data string of length `NUM_DATA_BITS`, and a binary number of length `NUM_SHIFT_BITS` (representing the number of bits `k` to shift right circularly). The output is the result of circularly shifting the data string right by `k` bits.
- **I/O Format:**
    - Input: A binary string of length `NUM_DATA_BITS` + `NUM_SHIFT_BITS`.
    - Output: A binary multi-label vector of length `NUM_DATA_BITS`.
- **Main Parameters:** `NUM_DATA_BITS`, `NUM_SHIFT_BITS`, `DATASET_SIZE`.

---

## 30. **generate_multiply_matrix_3x3.py**

- **Purpose:** To test the model's ability to learn structured algebraic operations (matrix multiplication), which requires more complex "data routing" and "multiply-accumulate" capabilities than simple scalar operations.
- **Logic:** The input consists of two 3x3 binary matrices, flattened and concatenated into an 18-bit binary string. The output is the resulting 3x3 matrix from their multiplication (with elements ranging from 0-3), also flattened and encoded into a binary multi-label vector.
- **I/O Format:**
    - Input: An 18-bit binary string.
    - Output: An 18-bit binary multi-label vector (9 elements * 2 bits/element).
- **Main Parameters:** `num_samples`.

---

## 31. **generate_evaluate_boolean_expression_text.py**

- **Purpose:** To test the model's ability to parse and evaluate a simple domain-specific language (DSL), which is a step beyond evaluating fixed-structure expressions.
- **Logic:** The script randomly generates a boolean expression, such as `(x0 | x1) & (x2)`. Simultaneously, it assigns a random boolean value (0 or 1) to all variables involved in the expression. The input is formed by concatenating the expression string and the variable assignment string, and the output is the final evaluation result of the expression.
- **I/O Format:**
    - Input: A string in the format "x=...;expr=(...)".
    - Output: A single character '1' (True) or '0' (False).
- **Main Parameters:** `num_samples`, `num_vars`.

---

## 32. **generate_evaluate_arithmetic_expression.py**

- **Purpose:** To train the model to perform evaluation of symbolic expressions, which requires understanding operator precedence (implicitly expressed through a tree structure), variable substitution, and arithmetic operations.
- **Logic:** The script first randomly generates an expression tree containing addition, subtraction, multiplication, numerical constants, and the variable 'x'. It then flattens the tree structure into a prefix token sequence and encodes it in binary. Finally, it randomly generates a value for 'x', concatenates the expression and the value of 'x' as input, and provides the final computed result as output.
- **I/O Format:**
    - Input: A binary string of `(TOKEN_LEN * N) + X_BITS` bits, representing the expression and the value of x.
    - Output: A binary string of `OUTPUT_BITS` bits, representing the evaluation result.
- **Main Parameters:** `VAL_RANGE`, `X_VAL_RANGE`, `DATASET_SIZE`.

---

## 33. **generate_evaluate_arithmetic_expression_no_multiply.py**

- **Purpose:** This is a simplified version of `generate_evaluate_arithmetic_expression.py`, designed to reduce the learning difficulty by removing the multiplication operation, in order to test the model's ability on more basic arithmetic expression evaluation.
- **Logic:** The logic is similar to the previous script, but the randomly generated expression tree only contains addition and subtraction operations, completely excluding multiplication. This makes the numerical range of the expression more controllable and reduces the learning burden on the model.
- **I/O Format:**
    - Input: A binary string of `(TOKEN_LEN * N) + X_BITS` bits, representing the expression and the value of x.
    - Output: A binary string of `OUTPUT_BITS` bits, representing the evaluation result.
- **Main Parameters:** `VAL_RANGE`, `X_VAL_RANGE`, `DATASET_SIZE`.

---

## 34. **generate_evaluate_arithmetic_expression_no_multiply_small_range.py**

- **Purpose:** This is a further simplification based on the previous "no multiplication" version, further reducing learning difficulty by narrowing the numerical range, used for precisely diagnosing the model's performance bottlenecks on the simplest expression evaluation tasks.
- **Logic:** The logic is the same as the `..._no_multiply.py` script, but the `VAL_RANGE` and `X_VAL_RANGE` parameters are set to smaller values. This ensures that all intermediate and final values during computation remain within a small range, making it the easiest version of the expression evaluation task.
- **I/O Format:**
    - Input: A binary string of `(TOKEN_LEN * N) + X_BITS` bits, representing the expression and the value of x.
    - Output: A binary string of `OUTPUT_BITS` bits, representing the evaluation result.
- **Main Parameters:** `VAL_RANGE`, `X_VAL_RANGE`, `DATASET_SIZE`.

---

## 35. **generate_check_boolean_equivalence.py**

- **Purpose:** To test the model's ability to judge the logical equivalence of boolean algebra expressions. This is an abstract symbolic reasoning task that requires the model to understand the structure of expressions and the rules of boolean algebra.
- **Logic:** The script randomly generates two expressions containing variables ('a','b','c','d') and boolean operators ('&', '|', '~'). It determines whether these two expressions are equivalent in all cases by using the **truth table** method, i.e., iterating through all possible combinations of variable assignments.
- **I/O Format:**
    - Input: A string in the format "expr1=...;expr2=...".
    - Output: A single character '1' (equivalent) or '0' (not equivalent).
- **Main Parameters:** `n` (number of samples), `vars` (set of variables).

---

## 36. **generate_polynomial_shift_coefficients.py**

- **Purpose:** To test the model's ability to learn an abstract algebraic transformation rule. This task requires the model to understand the internal structure of polynomial expansion.
- **Logic:** The input is 6 integers (representing the coefficients of a 5th-degree polynomial a5*x^5 + ... + a0), with each coefficient represented by a 3-bit binary number. The output is the 6 coefficients of the new polynomial after the variable substitution x -> x+1, with each new coefficient represented by an 8-bit binary number. The core of the script is the `poly_eval_at_shifted` function, which correctly uses the binomial theorem to calculate the coefficients of the new polynomial.
- **I/O Format:**
    - Input: An 18-bit binary string (6 * 3).
    - Output: A 48-bit binary string (6 * 8).
- **Main Parameters:** `max_samples`.

---

## 37. **generate_convolution_2d.py**

- **Purpose:** To test the model's ability to learn the basic image processing operation of 2D convolution (Conv2D), and to explore whether it can infer the hidden, fixed rule (i.e., the convolutional kernel itself) from input-output pairs.
- **Logic:** The script fixes a hidden 3x3 binary convolutional kernel. It generates two types of datasets: one where the input includes both the feature map and the kernel, testing the model's ability to perform the operation directly; another where the input only includes the feature map, requiring the model to parameterize the hidden kernel into its own weights by learning from a large number of samples.
- **I/O Format:**
    - Input (Visible): A (`MAP_SIZE`^2 + `KERNEL_SIZE`^2)-bit binary string. | Input (Hidden): A `MAP_SIZE`^2-bit binary string.
    - Output: A binary multi-label vector of length `MAP_SIZE`^2 * `BITS_PER_OUTPUT_ELEMENT`, representing the convolution result (the accumulated value for each pixel).
- **Main Parameters:** `MAP_SIZE`, `KERNEL_SIZE`, `DATASET_SIZE`.

---

## 38. **generate_simple_block_cipher.py**

- **Purpose:** To test the model's ability to "crack" or learn a simple but non-trivial custom encryption algorithm. This task represents a class of complex symbolic transformation rules with high chaos and avalanche effects.
- **Logic:** The script defines a fixed, hidden round key (`HIDDEN_KEY`) and a simple block cipher algorithm called T-Cipher. It generates training data pairs by encrypting random plaintexts for N rounds to produce ciphertexts.
- **I/O Format:**
    - Input: A plaintext binary string of length `INPUT_BITS`.
    - Output: A ciphertext binary multi-label vector of length `INPUT_BITS`.
- **Main Parameters:** `INPUT_BITS`, `NUM_ROUNDS`, `DATASET_SIZE`.

---

## 39. **generate_sin_function_float32.py**

- **Purpose:** To test the model's ability to fit a continuous, periodic, non-linear function (sin(x)), using the standard 32-bit floating-point format for both input and output.
- **Logic:** The script's input is a floating-point number `x`, using its standard IEEE 754 32-bit binary representation. The output is the result of sin(x), also using its 32-bit binary representation.
- **I/O Format:**
    - Input: A 32-bit binary multi-label vector.
    - Output: A 32-bit binary multi-label vector.
- **Main Parameters:** `N` (number of samples), `x_range`.

---

## 40. **generate_sin_function_float64_to_int12_deprecated.py**

- **Purpose:** This is another encoding attempt for the sin function fitting task, aimed at exploring the effect of using higher-precision floating-point input and lower-precision quantized binary output on learning performance.
- **Logic:** The script's input is a floating-point number `x`, using its 64-bit (double-precision) binary representation. The output is the result of sin(x), but linearly mapped and quantized to a 12-bit signed integer space.
- **I/O Format:**
    - Input: A 64-bit binary multi-label vector.
    - Output: A 12-bit binary multi-label vector.
- **Status:** (Deprecated) This is an early, problematic version that has been replaced by the more successful `generate_sin_function_float32_to_quantized_int.py`.

---

## 41. **generate_sin_function_float32_to_quantized_int.py**

- **Purpose:** To test the model's ability to fit a continuous, periodic, non-linear function (sin(x)), and to explore the impact of different input/output encoding schemes on learning performance.
- **Logic:** This script adopts an effective encoding strategy:
    1. **Input:** A floating-point number `x`, using its standard IEEE 754 32-bit binary representation.
    2. **Output:** It calculates y = sin(x) (range [-1, 1]), then linearly maps and quantizes it to a 24-bit signed integer space. This discretized representation is more suitable for classification models to learn.
- **I/O Format:**
    - Input: A 32-bit binary multi-label vector.
    - Output: A 24-bit binary multi-label vector.
- **Main Parameters:** `N` (number of samples), `x_range`.

---

## 42. **generate_multiply_binary_modulo.py**

- **Purpose:** As part of the basic arithmetic experiments, to test the model's mastery of truncated multiplication (or modular multiplication).
- **Logic:** Multiplies two N-bit integers and then performs a modulo 2^N operation on the result to ensure the output has the same number of bits as the input operands.
- **I/O Format:**
    - Input: A binary string of length `bits` * 2.
    - Output: A binary multi-label vector of length `bits`.
- **Main Parameters:** `num_samples`, `bits`.

---

## 43. **generate_explainable_two_step_calculation.py**

- **Purpose:** To test the model's ability to output intermediate calculation steps or a "chain of thought," serving as a direct validation of "functional explainability."
- **Logic:** The input consists of three 8-bit binary numbers and two operators. The model is required to output a concatenated vector where the first part is the intermediate result of the first operation, and the second part is the final result. This forces the model not only to calculate the answer but also to "backtrack" and present a key state from its calculation process.
- **I/O Format:**
    - Input: A string of length 8*3 (operands) + 2 (operators).
    - Output: A binary string of length 8 (intermediate result) + 8 (final result).
- **Main Parameters:** `count`.

---

## 44. **generate_chess_positions_by_random_moves.py**

- **Purpose:** To quickly generate a large number of plausible and legal Chinese chess positions by simulating a completely random player.
- **Logic:** The script starts from the standard initial position of Chinese chess. In a loop, it gets all legal moves for the current position, then randomly selects and executes one of them. This process is repeated `max_steps` times to arrive at a random but legal position.
- **I/O Format:**
    - Output: A position string in FEN format.
- **Main Parameters:** `max_steps`, `max_capture`.

---

## 45. **generate_chess_positions_by_random_placement.py**

- **Purpose:** To generate a large number of atypical, yet mostly legal, Chinese chess positions by randomly placing pieces on the board (rather than simulating moves), for stress-testing the model's robustness.
- **Logic:** Instead of generating positions through gameplay, this script directly places pieces on the board randomly, following piece placement constraints and the rule that generals cannot face each other, thereby creating a large number of positions that are syntactically legal but rarely seen in actual games.
- **I/O Format:**
    - Output: A position string in FEN format.
- **Main Parameters:** `num_fens`.

---

## 46. **generate_chess_positions_from_engine_self_play.py**

- **Purpose:** To generate a large number of high-quality, logically sound Chinese chess positions (in FEN format) to serve as a foundational data source for training a chess AI.
- **Logic:** It calls a powerful third-party chess engine (PikaFish) via a subprocess to simulate tens of thousands of high-level self-play games. During the simulation, it records the FEN representation of each move in the game, thereby building a vast and realistic position database.
- **I/O Format:**
    - Output: A .txt file, with each line containing a complete FEN string.
- **Main Parameters:** `num_games`, `max_steps`, `depth`.

---

## 47. **generate_preprocess_legal_moves.py**

- **Purpose:** This is a data preprocessing script used to convert a FEN-formatted position dataset into a "legal move prediction" task that the model can directly learn from.
- **Logic:** It reads a FEN file, and for each position, it uses the `cchess` library to parse and generate all legal moves. Then, based on a global mapping file, it converts each specific move (e.g., 'h2e2') into a unique integer ID.
- **I/O Format:**
    - Input: A .txt file, one FEN per line.
    - Output: A .jsonl file, where each JSON object contains a `fen` and its corresponding list of `legal_move_ids`.
- **Main Parameters:** `fen_file`, `output_file`.

---

## 48. **generate_chess_resolve_check_task.py**

- **Purpose:** To generate a dataset specifically for the tactical scenario of "Resolving a Check" in Chinese chess. This task requires the model, while in a state of being checked, to find all legal moves that can resolve the check.
- **Logic:** The script first filters a large library of random positions, keeping only those that satisfy the condition of "being in check, but not checkmated (has legal moves)." Then, for each filtered position, it calculates all legal moves that resolve the check and saves their IDs.
- **I/O Format:**
    - Output: A .jsonl file. Each JSON object contains a `fen` (position) and `legal_move_ids` (a list of integers representing all legal moves to resolve the check).
- **Main Parameters:** `fen_file`, `output_file`.

---

# B: Algorithm Learning

## 1. **generate_sort_integers.py**

- **Purpose:** To test the model's ability to execute a basic sorting algorithm, a classic non-local algorithmic task that requires comparison and rearrangement of input elements.
- **Logic:** The input is a binary string formed by concatenating `NUM_ITEMS` unordered integers, each of `NUM_BITS_PER_ITEM` bits. The output is a binary string formed by re-concatenating these numbers after sorting them in ascending order. The script ensures that all numbers in the input are unique.
- **I/O Format:**
    - Input: A binary string of length `NUM_ITEMS` * `NUM_BITS_PER_ITEM`.
    - Output: A binary multi-label vector of length `NUM_ITEMS` * `NUM_BITS_PER_ITEM`.
- **Main Parameters:** `NUM_ITEMS`, `NUM_BITS_PER_ITEM`, `DATASET_SIZE`.

---

## 2. **generate_edit_distance.py**

- **Purpose:** To test the model's ability to learn to solve dynamic programming problems. Edit distance is a typical DP problem that conceptually requires the model to construct a 2D solution matrix.
- **Logic:** The input is the concatenation of two equal-length binary strings, `s1` and `s2`. The output is the binary representation of their minimum edit distance (allowing insertion, deletion, and substitution operations).
- **I/O Format:**
    - Input: A binary string of length `NUM_BITS_PER_STRING` * 2.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`.
- **Main Parameters:** `NUM_BITS_PER_STRING`, `DATASET_SIZE`.

---

## 3. **generate_edit_distance_explainable.py**

- **Purpose:** This is a core experiment for "functional explainability." It requires the model not only to give the final answer (edit distance) but also to output the complete "chain of thought" (the editing process).
- **Logic:** The input is two strings, `s1` and `s2`. The output is a long vector composed of `max_steps` "state frames." Each state frame contains two parts: the binary representation of an intermediate string during the editing process, and a mask to indicate the valid length of that string. This forces the model to learn to simulate the step-by-step transformation from `s1` to `s2`. The script uses a clever mechanism to retain only samples where the optimal edit path is unique, ensuring the labels are unambiguous.
- **I/O Format:**
    - Input: A binary string of length `str_len` * 2.
    - Output: A binary multi-label vector of length `max_edits` * `str_len` * 2.
- **Main Parameters:** `num_samples`, `str_len`, `max_edits`.

---

## 4. **generate_maze_random_walls.py**

- **Purpose:** To test the model's basic pathfinding ability in randomly generated "porous" mazes.
- **Logic:** The script generates mazes by randomly placing walls on a grid. This method typically produces mazes with shorter paths, high connectivity, and relatively simple structures. Then, for all traversable points, it uses a reverse BFS from a fixed endpoint to calculate the shortest path to that point. The model's task is, given a maze layout with a start and end point, to predict the optimal first step from the starting point.
- **I/O Format:**
    - Input: A string of length H * W, representing the maze layout.
    - Output: A 4-class categorical label (Up/Down/Left/Right).
- **Main Parameters:** `MAZE_HEIGHT`, `MAZE_WIDTH`, `TARGET_NUM_SAMPLES`.

---

## 5. **generate_maze_dense.py**

- **Purpose:** To test the model's path-planning ability in complex, "dense" mazes similar to human designs, which are more challenging than random-wall mazes.
- **Logic:** The script first uses a specialized maze generation algorithm (like recursive division) to create a challenging, connected, dense maze characterized by long, winding passages. Then, similar to the previous script, it uses reverse BFS to compute the optimal policy for all reachable points.
- **I/O Format:**
    - Input: A string of length H * W, representing the maze layout.
    - Output: A 4-class categorical label.
- **Main Parameters:** `MAZE_HEIGHT`, `MAZE_WIDTH`, `TARGET_NUM_SAMPLES`.

---

## 6. **generate_blocks_world_arbitrary_goal.py**

- **Purpose:** To solve the classic "Blocks World" planning problem, a benchmark task in the field of AI planning. This version allows for arbitrary initial and goal states.
- **Logic:** The script randomly generates an initial state and a goal state for each sample. It then uses Breadth-First Search (BFS) to find the shortest sequence of actions from the initial state to the goal state. The model's task is to predict the first optimal action in this sequence.
- **I/O Format:**
    - Input: Binary encoding of the initial and goal states.
    - Output: A 6-class categorical label, representing the optimal action.
- **Main Parameters:** `BLOCKS_N` (number of blocks).

---

## 7. **generate_blocks_world_fixed_goal.py**

- **Purpose:** A simplification of the "Blocks World" task. By fixing the goal state, it aims to test the model's learning ability in a situation with a clear objective and a more structured state space.
- **Logic:** The script sets a fixed goal state (all blocks stacked in order on the first peg). It then performs a **reverse** Breadth-First Search (BFS) starting from the goal state to efficiently traverse all reachable states and calculate the optimal policy for each state to reach the goal.
- **I/O Format:**
    - Input: Encoding of the initial state of the blocks.
    - Output: A 6-class categorical label, representing the optimal action.
- **Main Parameters:** `BLOCKS_N`.

---

## 8. **generate_blocks_world_fixed_goal_multilabel.py**

- **Purpose:** Further improving the "Blocks World" task by allowing for multiple optimal solutions. It tests the model's ability to handle multi-label classification problems, which more realistically reflects the possibility of equivalent optimal paths in planning problems.
- **Logic:** Inherits the logic of a fixed goal and reverse search from the previous script. The key improvement is in the output format: for each state, the script finds **all** optimal actions that bring it one step closer to the goal and generates a multi-hot encoded output vector.
- **I/O Format:**
    - Input: Encoding of the initial state of the blocks.
    - Output: A binary multi-label vector of length `NUM_ACTIONS`.
- **Main Parameters:** `BLOCKS_N`.

---

## 9. **generate_blocks_world_fixed_goal_multilabel_fixed_format.py**

- **Purpose:** This is the final optimized version of the "Blocks World" task. By improving the input representation, it aims to provide the model with a clearer, more structured learning target.
- **Logic:** The core logic is the same as the previous script (multi-label output, fixed goal, reverse search). The key improvement is in the input format: instead of using separators, it allocates a fixed number of "slots" for each peg to represent the state. This fixed-length representation eliminates the complexity of variable-length inputs and is more friendly to models like Transformers.
- **I/O Format:**
    - Input: A string of length `NUM_BLOCKS` * `NUM_STACKS`, where 0 represents empty and 1-N represent blocks.
    - Output: A binary multi-label vector of length `NUM_ACTIONS`.
- **Main Parameters:** `BLOCKS_N`, `NUM_STACKS`.

---

## 10. **generate_checkers_jump_1d.py**

- **Purpose:** To solve a planning problem involving moving checkers in a 1D space, which originated from a famous Apple paper used to test the reasoning bottlenecks of large language models.
- **Logic:** The script simulates the process of two types of checkers ('R' and 'B') moving past each other on a 1D board. It uses an efficient reverse Breadth-First Search (BFS) starting from the goal state, traversing the entire state space backwards, to compute the unique optimal next move for every reachable state.
- **I/O Format:**
    - Input: An integer sequence of length 2*N+1, representing the board state.
    - Output: A single integer, representing the **position index** of the checker to be moved.
- **Main Parameters:** `CHECKERS_N` (number of checkers of each color).

---

## 11. **generate_river_crossing_puzzle.py**

- **Purpose:** To solve a classic constraint satisfaction and state-space search problem—"N couples crossing a river." The problem requires transporting everyone to the other side under the constraint that "no woman can be in the presence of other men unless her partner is present." This task originates from an Apple paper used to reveal the limitations of large language models on certain types of reasoning tasks.
- **Logic:** The script defines each state by the set of people on the left bank and the position of the boat. It uses an efficient reverse Breadth-First Search (BFS) starting from the goal state (everyone on the right bank) to construct an optimal policy graph covering all reachable states. The output is a multi-label vector indicating which people should get on the boat together to perform the optimal move in the current state.
- **I/O Format:**
    - Input: A binary string of length 2*N+1 (N clients C, N agents A, 1 boat position).
    - Output: A multi-label binary vector of length 2*N, representing whether each person boards the boat.
- **Main Parameters:** `PAIRS_N` (number of couples), `BOAT_CAPACITY_K`.

---

## 12. **generate_trapping_rain_water_aggregate.py**

- **Purpose:** An initial attempt to solve the "Trapping Rain Water" algorithm problem, aimed at testing the model's ability to learn an aggregated output (rather than a decoupled one). The results showed that requiring the model to directly output the total sum (a single aggregated number) is much harder than outputting detailed information for each position.
- **Logic:** The input is a 1D height map. The script calculates the total amount of rainwater that can be trapped on this height map and provides this single integer value as the output.
- **I/O Format:**
    - Input: A binary string of N * K bits, representing the heights of N columns.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the total rainwater amount.
- **Main Parameters:** `NUM_COLUMNS_N`, `BITS_PER_HEIGHT`.

---

## 13. **generate_trapping_rain_water_decoupled.py**

- **Purpose:** To solve the classic "Trapping Rain Water" algorithm problem (LeetCode Hard). The success of this task demonstrates the model's ability to learn complex algorithms that require global information (like the global maximum height) and proves, through the idea of **problem decoupling**, the significant impact of output format design on model learning efficiency.
- **Logic:** The script designs the output based on a key problem-decoupling insight. Instead of having the model predict a single aggregated value (total rainwater), it requires the model to predict a sequence with a structure isomorphic to the input, where each element represents the amount of rainwater trapped on the corresponding column. This change greatly simplifies the learning task, allowing the model to converge successfully.
- **I/O Format:**
    - Input: A binary string of N * K bits, representing the heights of N columns.
    - Output: A binary multi-label vector of N * K bits, representing the amount of water trapped on each of the N columns.
- **Main Parameters:** `NUM_COLUMNS_N`, `BITS_PER_HEIGHT`.

---

## 14. **generate_trapping_rain_water_2d.py**

- **Purpose:** An extension of the 1D "Trapping Rain Water" problem, solving the 2D version. This task requires the model to understand concepts of "enclosure" and "boundary" in a 2D space, presenting a more complex challenge of global information processing.
- **Logic:** It also adopts the idea of problem decoupling. The input is a 2D height map (matrix), and the output is a matrix of the same size, where the value of each cell represents the amount of rainwater that can be trapped at that location. The solver determines the water level at each point by performing a BFS-like "filling" operation from the boundaries inward.
- **I/O Format:**
    - Input: A binary string of N*M*K bits, representing the heights of an N*M grid.
    - Output: A binary multi-label vector of N*M*K bits, representing the amount of water trapped in each cell.
- **Main Parameters:** `GRID_N`, `GRID_M`, `BITS_PER_HEIGHT`.

---

## 15. **generate_skyline_max_height_aggregate.py**

- **Purpose:** An initial attempt to solve the "Skyline" problem, requiring the model to predict only the maximum height among all the final building heights. This task is used to compare the learning difficulty of aggregated versus decoupled outputs.
- **Logic:** The input is a series of height restrictions for buildings. Under the constraint that the height difference between adjacent buildings does not exceed 1, the script uses dynamic programming to calculate the maximum possible height for each building and then finds the maximum value among all buildings as the output.
- **I/O Format:**
    - Input: A binary string of length `n` * `bit_count`, representing the height restrictions for each building.
    - Output: A binary multi-label vector of length `bit_count`, representing the global maximum height.
- **Main Parameters:** `NUM_SAMPLES`, `FIXED_N` (number of buildings), `MAX_HEIGHT`.

---

## 16. **generate_skyline_all_heights_decoupled.py**

- **Purpose:** To test the model's ability to solve a global optimization problem with 1D spatial constraints. The problem prototype is LeetCode's "Max-Height Skyline." By decoupling the output, it requires the model to predict the height of each building, rather than just the maximum value.
- **Logic:** The input is a series of height restrictions for buildings. The rule is that, while satisfying all restrictions, the height difference between adjacent buildings cannot exceed 1. The script uses an efficient bidirectional dynamic programming algorithm to solve for the maximum possible height of each building under these constraints. The output is a sequence composed of the final heights of all buildings.
- **I/O Format:**
    - Input: A binary string of length `n` * `bit_count`, representing the initial height restrictions for each building.
    - Output: A binary multi-label vector of length `n` * `bit_count`, representing the final height of each building.
- **Main Parameters:** `NUM_SAMPLES`, `FIXED_N` (number of buildings), `MAX_HEIGHT`.

---

## 17. **generate_hanoi_tower_path_strategy_sep_format.py**

- **Purpose:** An early experimental script for the Tower of Hanoi problem, aimed at testing whether the model can learn the strategy along the optimal path. It uses a separator-style input format and predicts the action as a 6-class classification problem.
- **Logic:** The script generates all states along the optimal path of the Tower of Hanoi using a standard recursive solver. For each state, it calculates the next optimal move (e.g., from peg 1 to peg 3). The input is the `sep` format encoding of the state, and the output is a categorical label for the 6 possible moves.
- **I/O Format:**
    - Input: A `sep` format string, e.g., `123|4|56`.
    - Output: A 6-class categorical label.
- **Main Parameters:** `HANOI_N`.

---

## 18. **generate_hanoi_tower_global_strategy_fixed_format.py**

- **Purpose:** As an improvement on earlier Tower of Hanoi experiments, this script adopts a fixed-slot input format that is more friendly to the model, aiming to verify the impact of input representation on learning efficiency.
- **Logic:** The script generates all possible states of the Tower of Hanoi problem. The input uses a fixed-length slot representation for the disks on each peg. The output is still the prediction of the optimal next move (6-class classification).
- **I/O Format:**
    - Input: A fixed-length string, e.g., `123000400056000`.
    - Output: A 6-class categorical label.
- **Main Parameters:** `HANOI_N`, `dataset_size`.

---

## 19. **generate_hanoi_tower_compare_formats.py**

- **Purpose:** This is a comparative experiment script that generates two different input formats (separator vs. fixed-slot) for the same Tower of Hanoi problem, to systematically evaluate the effect of different data representations on the model's learning of the recursive strategy.
- **Logic:** The script simultaneously generates two datasets, one using the `sep` format and the other using the fixed-slot format. Both datasets only contain states on the optimal path and require the model to predict the next optimal move.
- **I/O Format:**
    - Input: `sep` format or fixed-slot format.
    - Output: A 6-class categorical label.
- **Main Parameters:** `HANOI_N`, `DATASET_SIZE`.

---

## 20. **generate_hanoi_tower_compare_formats_and_strategies.py**

- **Purpose:** A more comprehensive comparative experiment script for the Tower of Hanoi. It not only generates two input formats but also two different types of datasets: one containing only states on the optimal path ("path strategy") and another containing all reachable states ("global strategy"). This is to explore the difference in the model's ability to learn a local optimal path versus a global optimal policy.
- **Logic:** The script generates a total of four datasets (2 formats x 2 strategies). The results showed that the model could easily learn the "path strategy" but encountered difficulties in learning the "global strategy," revealing potential limitations of the model in handling recursion and state-space explosion problems.
- **I/O Format:**
    - Input: `sep` format or fixed-slot format.
    - Output: A 6-class categorical label.
- **Main Parameters:** `HANOI_N`, `DATASET_SIZE`.

---

## 21. **generate_hanoi_tower_build_full_state_graph.py**

- **Purpose:** This is a culmination of the "Tower of Hanoi" research, aiming to deeply analyze the model's understanding of recursive structures through various data representations and sampling strategies. It is a self-contained data factory.
- **Logic:** The core of this script is a remarkable implementation: instead of using a traditional recursive solver, it directly constructs the complete "state-action" graph of the Tower of Hanoi problem with its 3^N states in memory, based on an elegant mathematical structure involving the fractal and self-similar nature of the Hanoi graph. This allows for arbitrary and efficient data sampling from this complete knowledge base later on.
- **I/O Format:**
    - Output: A .pkl file storing the complete knowledge base containing all states and their optimal actions.
- **Main Parameters:** `HANOI_N`.

---

## 22. **generate_hanoi_tower_sample_from_state_graph.py**

- **Purpose:** This is a post-processing and sampling script that utilizes the complete knowledge base generated by `generate_hanoi_tower_build_full_state_graph.py` to precisely extract specific types of training data subsets, such as "twisted paths" or the "hardest parts," for more fine-grained ablation studies.
- **Logic:** The script first loads the complete state graph generated by the `_mine` script. Then, based on user-specified start and end states, it can accurately extract the complete optimal path connecting these two points and save it as a trainable .jsonl file.
- **I/O Format:**
    - Input: `all_states_and_moves_N.pkl` file.
    - Output: Training data in .jsonl format.
- **Main Parameters:** `HANOI_N`, `start_idx`, `end_idx`.

---

## 23. **generate_sokoban_planning_astar.py**

- **Purpose:** To solve the classic "Sokoban" planning problem. This task is harder than simple pathfinding because it involves changing the state of the environment (box positions), resulting in a huge state space.
- **Logic:** The script first randomly generates a maze layout containing walls, a player, a single box, and a single goal. It then uses an efficient A* search algorithm to find the optimal sequence of actions to push the box to the goal location. The model's task is, given a board state, to predict the player's next optimal move (Up/Down/Left/Right), which could be just moving the player or pushing the box.
- **I/O Format:**
    - Input: A string of length M * N, representing the Sokoban layout.
    - Output: A 4-class categorical label.
- **Main Parameters:** `M_DIMENSION`, `N_DIMENSION`, `NUM_SAMPLES`.

---

## 24. **generate_sokoban_planning_full.py**

- **Purpose:** To solve the classic "Sokoban" planning problem. This is a highly difficult AI task as it involves searching in a massive state space where actions alter the environment's state.
- **Logic:** This is a very mature dataset generator.
    1. **Intelligent Generation:** It generates puzzles that are both random and potentially interesting by randomly placing walls and performing a reverse random walk from the goal.
    2. **A* Solver:** It uses an efficient A* search algorithm with Manhattan distance as the heuristic to calculate the shortest box-pushing path from the initial state to the goal.
    3. **Optimal Policy Extraction:** For all states on the optimal path, it calculates all actions that lead to the next optimal state and generates a multi-label (multi-hot) output.
    4. **Quality Control:** It includes difficulty filters (only keeping solutions within a specific step range) and performs global deduplication and shuffling.
- **I/O Format:**
    - Input: A string of length (M-2)*(N-2), representing the Sokoban layout with boundary walls removed.
    - Output: A 4-bit multi-label binary vector, representing whether Up/Down/Left/Right are optimal actions.
- **Main Parameters:** `M_DIMENSION`, `N_DIMENSION`, `NUM_SAMPLES`, `MIN/MAX_DIFFICULTY`.

---

## 25. **generate_sokoban_planning_claude_deprecated.py**

- **Purpose:** (Deprecated) This was an earlier attempt with more complex logic, but it failed to consistently generate high-quality datasets and has been replaced by the more reliable `generate_sokoban_planning_full.py`.
- **Logic:** An early version attempting to solve the Sokoban problem.
- **Status:** **Deprecated**.

---

## 26. **generate_min_swaps_for_checkerboard.py**

- **Purpose:** To solve a highly constrained matrix rearrangement problem: find the minimum number of swaps (of any rows or columns) required to transform a 0/1 matrix into a "checkerboard" pattern (adjacent elements differ).
- **Logic:** The script first intelligently generates a guaranteed "solvable" input matrix by randomly swapping rows and columns of a perfect checkerboard. Then, it uses a complex algorithm based on bitwise operations and combinatorial analysis to precisely calculate the minimum total number of row and column swaps required to restore the checkerboard pattern. If it cannot be restored, it returns -1.
- **I/O Format:**
    - Input: A binary string of length N*N.
    - Output: A binary multi-label vector of length `OUTPUT_BITS` (-1 is mapped to 0, k moves are mapped to k+1).
- **Main Parameters:** `MATRIX_SIZE_N`, `DATASET_SIZE`.

---

## 27. **generate_min_flips_for_alternating_binary.py**

- **Purpose:** To test the model's ability to solve a string optimization problem based on bit flips, which can be cleverly mapped to a sliding window problem for a solution.
- **Logic:** A "beautiful string" is defined as an alternating sequence of 0s and 1s (e.g., '0101...' or '1010...'). The input is an arbitrary binary string, and the task is to calculate the minimum number of flips required to make it "beautiful."
- **I/O Format:**
    - Input: A binary string of length `STRING_LENGTH_N`.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the minimum number of flips.
- **Main Parameters:** `STRING_LENGTH_N`, `DATASET_SIZE`.

---

## 28. **generate_min_swaps_for_checkerboard_v2.py**

- **Purpose:** To solve a highly constrained matrix rearrangement problem: find the minimum number of swaps (of any rows or columns) required to transform a 0/1 matrix into a "checkerboard" pattern (adjacent elements differ).
- **Logic:** The script first intelligently generates a guaranteed "solvable" input matrix by randomly swapping rows and columns of a perfect checkerboard. Then, it uses a complex algorithm based on bitwise operations and combinatorial analysis to precisely calculate the minimum total number of row and column swaps required to restore the checkerboard pattern. If it cannot be restored, it returns -1.
- **I/O Format:**
    - Input: A binary string of length N*N.
    - Output: A binary multi-label vector of length `OUTPUT_BITS` (-1 is mapped to 0, k moves are mapped to k+1).
- **Main Parameters:** `MATRIX_SIZE_N`, `DATASET_SIZE`.

---

## 29. **generate_matrix_flip_strategy.py**

- **Purpose:** To solve a classic matrix optimization problem (maximizing the number of 1s). This version aims to test if the model can learn a "policy" rather than the final result.
- **Logic:** For a given M x N binary matrix, its content can be changed by flipping any row or any column. The task is to find a flipping strategy that maximizes the number of '1's in the final matrix. The script uses an efficient greedy algorithm to find the optimal strategy: first, fix the first column to be all 1s (by flipping necessary rows), then iterate through the remaining columns, flipping a column if it has more 0s than 1s.
- **I/O Format:**
    - Input: A binary string of length M*N.
    - Output: A binary multi-label vector of length M+N, representing the row and column flip masks.
- **Main Parameters:** `MATRIX_M`, `MATRIX_N`, `DATASET_SIZE`.

---

## 30. **generate_matrix_flip_max_score.py**

- **Purpose:** To test the model's ability to learn a matrix optimization problem that requires a two-step greedy strategy (first flip rows, then columns) to achieve a global optimum. This version requires the model to directly output the final aggregated result (the score).
- **Logic:** The script implements an efficient greedy algorithm to find the maximum score. Step 1: Iterate through each row; if the most significant bit (leftmost) of the row is 0, flip the row. Step 2: Iterate through each column; if the number of 0s in the column is greater than the number of 1s, flip the column. Finally, the resulting matrix is summed with binary weighting to get the maximum score.
- **I/O Format:**
    - Input: A binary string of length `MATRIX_M` * `MATRIX_N`.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the maximum score.
- **Main Parameters:** `MATRIX_M`, `MATRIX_N`, `DATASET_SIZE`.

---

## 31. **generate_min_prefix_flips.py**

- **Purpose:** To test the model's ability to learn a sequential processing greedy algorithm that depends on historical states.
- **Logic:** This is a classic "prefix flip" or "light bulb" problem. Traverse the sequence from left to right. If the current position is still '0' after considering the cumulative effect of all previous flips, you must "pull" the switch at the current position (which flips all bits from the current position to the end) and count one operation.
- **I/O Format:**
    - Input: A binary string of length `STRING_LENGTH_N`.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the minimum number of flips.
- **Main Parameters:** `STRING_LENGTH_N`, `DATASET_SIZE`.

---

## 32. **generate_min_k_bit_flips.py**

- **Purpose:** To test the model's ability to learn a sequential processing greedy algorithm that depends on historical states, and to test if it can use one part of the input (`k`) as a "parameter" to guide the processing of another part (`nums`).
- **Logic:** Solves the classic "Minimum Number of K Consecutive Bit Flips" problem (LeetCode 995). It traverses the array from left to right, using an efficient difference array technique to track the cumulative effect of previous flips. If the value at the current position is still '0' under the cumulative effect, a new flip must be performed, and the difference array is updated.
- **I/O Format:**
    - Input: A binary string of length `NUMS_LENGTH_N` + `K_BITS` (data + parameter k).
    - Output: A binary multi-label vector of length `OUTPUT_BITS` (representing the minimum number of flips, 0 for no solution).
- **Main Parameters:** `NUMS_LENGTH_N`, `K_MAX_N`, `DATASET_SIZE`.

---

## 33. **generate_min_k_bit_flips_fixed_k.py**

- **Purpose:** To test the model's ability to learn a sequential processing greedy algorithm that depends on historical states. In this version, the environmental parameter (k=2) is fixed and hidden, and the model must implicitly learn it from the data.
- **Logic:** The logic is the same as `generate_min_k_bit_flips.py`, but the length of the flipping window `k` is fixed to 2 and is not provided in the input.
- **I/O Format:**
    - Input: A binary string of length `NUMS_LENGTH_N` (data only).
    - Output: A binary multi-label vector of length `OUTPUT_BITS` (representing the minimum number of flips, 0 for no solution).
- **Main Parameters:** `NUMS_LENGTH_N`, `DATASET_SIZE`.

---

## 34. **generate_special_binary_string_recursion.py**

- **Purpose:** To test the model's ability to learn a recursively defined string transformation rule. This problem (LeetCode Hard "Special Binary String") requires recursive decomposition and reassembly of the input.
- **Logic:** The properties of a "special binary string" are similar to those of a valid parenthesis sequence (1 represents '(', 0 represents ')'). The core idea of the algorithm is that any special string can be decomposed into the form `1 + A + 0 + B`, where A and B are also (possibly empty) special strings. The script recursively finds all outermost special substrings, optimally transforms each of them, and then concatenates the results in descending lexicographical order to get the final answer.
- **I/O Format:**
    - Input: A special binary string of length `STRING_LENGTH_N`.
    - Output: A binary multi-label vector of length `STRING_LENGTH_N`, representing the lexicographically largest result.
- **Main Parameters:** `STRING_LENGTH_N`, `DATASET_SIZE`.

---

## 35. **generate_min_flips_for_chunked_binary.py**

- **Purpose:** To test the model's ability to learn a string transformation optimization problem based on local chunks.
- **Logic:** Takes an even-length binary string as input. It is split into 2-bit chunks. For each 2-bit chunk, if the two bits are different (e.g., '01' or '10'), one flip operation is needed to make it "beautiful" (change it to '00' or '11'). The task is to calculate the total minimum number of flips required.
- **I/O Format:**
    - Input: A binary string of length `INPUT_BITS`.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the total number of flips.
- **Main Parameters:** `INPUT_BITS`, `DATASET_SIZE`.

---

## 36. **generate_count_connected_components.py**

- **Purpose:** To test the model's basic understanding of graph structures, especially the core concept of "connectivity."
- **Logic:** The script randomly generates an N x N adjacency matrix to represent an undirected graph. It then uses Breadth-First Search (BFS) or Depth-First Search (DFS) to traverse the graph and count the total number of independent connected components.
- **I/O Format:**
    - Input: A binary string of length N*N (the adjacency matrix).
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the number of connected components.
- **Main Parameters:** `GRAPH_SIZE_N`, `EDGE_PROBABILITY` (controls graph sparsity), `DATASET_SIZE`.

---

## 37. **generate_check_graph_connectivity.py**

- **Purpose:** Another core test of the model's foundational graph theory abilities. The task is to determine if a path exists between any two given nodes in a graph.
- **Logic:** The script randomly generates a graph's adjacency matrix and randomly selects two nodes as the start and end points. It utilizes a standard graph algorithm library to determine if these two points are in the same connected component.
- **I/O Format:**
    - Input: A string in the format `size*size (adjacency matrix) + ; + start_node_char + end_node_char`.
    - Output: `[1]` (connected) or `[0]` (not connected).
- **Main Parameters:** `num_samples`, `size` (number of nodes in the graph).

---

## 38. **generate_minimize_malware_spread.py**

- **Purpose:** To solve a graph-theory-based virus spread optimization problem (LeetCode Hard "Minimize Malware Spread"). The model needs to understand graph connectivity and evaluate the impact of removing different nodes on the global spread.
- **Logic:** The input is a graph's adjacency matrix and a set of initially infected nodes. The task is to remove only **one** of the initial infected nodes to minimize the total number of nodes ultimately infected by the virus. The script finds the optimal node to remove by brute-force simulating the spread after removing each initial node and comparing the results.
- **I/O Format:**
    - Input: A binary string of length (N*N) (adjacency matrix) + N (initial infected node mask).
    - Output: A 1-bit binary label (current implementation), indicating if the first initial infected node is one of the optimal solutions.
- **Main Parameters:** `GRAPH_SIZE_N`, `NUM_INITIAL`, `DATASET_SIZE`.

---

## 39. **generate_count_islands_1d.py**

- **Purpose:** To test the model's ability for pattern recognition and counting on a 1D sequence.
- **Logic:** The input is a 1D binary string. The task is to count the number of contiguous blocks of '1's (islands) separated by '0's. For example, in `0110100111`, there are 3 islands.
- **I/O Format:**
    - Input: A binary string of length `NUM_INPUT_BITS`.
    - Output: A binary multi-label vector of length `NUM_OUTPUT_BITS`, representing the number of islands.
- **Main Parameters:** `NUM_INPUT_BITS`, `DATASET_SIZE`.

---

## 40. **generate_largest_island_by_adding_one_cell.py**

- **Purpose:** To solve an algorithm problem involving graph traversal and global optimization (LeetCode 827). The model needs to evaluate all possible "land reclamation" positions and choose the one that results in the largest merged island area.
- **Logic:** The script first uses DFS or BFS to traverse the input grid, labeling all existing islands and calculating their areas. Then, it iterates through all water cells ('0'), calculates which adjacent islands could be connected if that cell were turned into land, and thereby computes the new total area. Finally, it finds the optimal position that yields the maximum area.
- **I/O Format:**
    - Input: A binary string of length N*N.
    - Output: A JSON object containing `output_class` (categorical label of the best position) and `output_area` (binary string of the max area).
- **Main Parameters:** `NUM_SAMPLES`, `GRID_SIZE`.

---

## 41. **generate_largest_island_by_adding_one_cell_v2.py**

- **Purpose:** To solve an algorithm problem involving graph traversal and global optimization (LeetCode 827). The model needs to evaluate all possible "land reclamation" positions and choose the one that results in the largest merged island area.
- **Logic:** The script first uses DFS or BFS to traverse the input grid, labeling all existing islands and calculating their areas. Then, it iterates through all water cells ('0'), calculates which adjacent islands could be connected if that cell were turned into land, and thereby computes the new total area. Finally, it finds the optimal position that yields the maximum area.
- **I/O Format:**
    - Input: A binary string of length N*N.
    - Output: A JSON object containing `output_class` (categorical label of the best position) and `output_area` (binary string of the max area).
- **Main Parameters:** `NUM_SAMPLES`, `GRID_SIZE`.

---

## 42. **generate_find_articulation_points.py**

- **Purpose:** To test the model's ability to identify "Articulation Points" or "Bridges" in a graph, an important concept in graph theory.
- **Logic:** The input is a 2D grid composed of '1's (land) and '0's (water). The task is essentially to find the minimum number of '1's to remove to disconnect the original single connected component (island). The script finds the solution by brute-force trials (removing 1 point, removing 2 points). The output is designed as a heatmap of the finally removed points, rather than the number of days.
- **I/O Format:**
    - Input: A binary string of length M*N.
    - Output: A binary multi-label vector of length M*N, marking the removed points.
- **Main Parameters:** `NUM_SAMPLES`, `GRID_M`, `GRID_N`.

---

## 43. **generate_nim_game_zeckendorf.py**

- **Purpose:** This experiment aims to test if my paradigm can learn a non-intuitive game theory problem based on complex number theory (Zeckendorf's representation). It moves beyond simple pattern matching and requires the model to understand a deeper mathematical structure.
- **Logic:** I implemented a solver for a classic stone game variant (similar to Wythoff's game), whose solution is closely related to the Fibonacci sequence and Zeckendorf's representation. To make it easier for the model to learn, I simplified the task: the original problem might be to count how many winning positions are in the interval [1, n], I changed it to just determining if a given `n` itself is a winning position. This creates a more direct causal relationship for each input and output.
- **I/O Format:**
    - Input: A binary string of length `N_BITS`, representing the total number of stones `n`.
    - Output: A 1-bit binary multi-label (`[1]` for a winning position, `[0]` for a losing one).
- **Main Parameters:** `N_BITS`, `DATASET_SIZE`.

---

## 44. **generate_longest_subsequence_constrained.py**

- **Purpose:** To test the model's ability to handle a complex optimization problem that mixes sequence operations and numerical constraints.
- **Logic:** The input is a binary string `s` and an integer `k` (also represented in binary). The task is to find a subsequence of `s` (which can be non-contiguous) such that the binary number formed by the subsequence has a value less than or equal to `k`, and its length is maximized. The output is the length of this longest subsequence.
- **I/O Format:**
    - Input: A binary string of length `STRING_LENGTH_N` + `K_BITS`.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the length.
- **Main Parameters:** `STRING_LENGTH_N`, `K_BITS`, `DATASET_SIZE`.

---

## 45. **generate_treasure_hunt_tsp.py**

- **Purpose:** To solve a complex state-space search problem that combines graph traversal (BFS) and combinatorial optimization (state compression DP), a classic difficult problem in competitive programming.
- **Logic:** In a given maze, the player must start from 'S', trigger all mechanisms 'M', and finally reach the destination 'T'. Stones 'O' can be used along the way to instantly trigger any mechanism. The script calculates the shortest distances between all key points (S, T, M, O) using a series of BFS runs, and then uses state compression dynamic programming to find the shortest total path length to traverse all mechanisms and reach the destination.
- **I/O Format:**
    - Input: A string of length N*M, representing the maze layout.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the shortest path length (-1 is mapped to 0).
- **Main Parameters:** `MAZE_N`, `MAZE_M`, `DATASET_SIZE`.

---

## 46. **generate_freedom_trail_dp.py**

- **Purpose:** To test the model's ability to learn to solve a complex optimization problem that requires dynamic programming and path backtracking.
- **Logic:** The input is a `ring` string representing characters on a circular dial and a target `key` string. The script uses dynamic programming to calculate the minimum number of rotational steps required to spell out the `key`. A key modification is that the script not only calculates the total steps but also backtracks the DP table to reconstruct the optimal operation for each step (clockwise or counter-clockwise, and the specific number of steps), and outputs this sequence of operations.
- **I/O Format:**
    - Input: A string in the format `ring|key`.
    - Output: A binary multi-label vector of length `KEY_LENGTH` * `move_bits`, encoding the operation for each step.
- **Main Parameters:** `RING_LENGTH`, `KEY_LENGTH`, `NUM_SAMPLES`.

---

## 47. **generate_sum_of_subset_with_mask.py**

- **Purpose:** To test the model's ability to select elements from a set based on a binary mask and perform an aggregation operation (summation).
- **Logic:** The script first generates a set of `n_items` unique integers. Then, it finds subsets whose sums satisfy a uniqueness condition by enumerating all possible subsets. The input is a concatenation of the **set of numbers** and a **binary mask indicating which subset is selected**. The output is the **sum of the elements** of the selected subset.
- **I/O Format:**
    - Input: A binary string of length (`n_items` * 4) (set of numbers) + `n_items` (mask).
    - Output: A 6-bit binary string, representing the sum.
- **Main Parameters:** `n_items`, `value_range` (number range), `num_samples`.

---

## 48. **generate_sudoku_6x6.py**

- **Purpose:** To test the model's ability on a strong Constraint Satisfaction Problem—Sudoku.
- **Logic:** The script implements a solver with backtracking to generate complete 6x6 Sudoku solutions and creates puzzles with unique solutions by "digging holes."
- **I/O Format:**
    - Input: A 36-character string, using `_` for empty spaces.
    - Output: A 108-bit binary multi-label vector (36 numbers * 3 bits per number).
- **Main Parameters:** `num_puzzles`, `difficulty` (hole digging ratio).

---

## 49. **generate_valid_parentheses_path_random_deprecated.py**

- **Purpose:** (Early exploration/Deprecated) An early attempt to solve the "Valid Parentheses Path" problem.
- **Status:** **Deprecated**. This script created datasets by randomly generating parenthesis grids, but this led to a severe data imbalance problem (the vast majority of random grids have no valid path), which is not conducive to model training. It has been replaced by `generate_valid_parentheses_path_balanced.py`.
- **Logic:** Randomly generates M x N parenthesis grids and calls a solver to determine if a valid path exists.
- **Main Parameters:** `MAZE_M`, `MAZE_N`, `DATASET_SIZE`.

---

## 50. **generate_valid_parentheses_path_balanced.py**

- **Purpose:** To solve a pathfinding problem on a 2D grid where path validity is constrained by a stack-like structure (parenthesis matching). This is a complex task combining algorithms and logical constraints (LeetCode Hard "Check if There Is a Valid Parentheses Path").
- **Logic:** The script generates data in two ways to ensure balance:
    1. **Positive Samples:** First, a path is determined on the grid, then a valid parenthesis sequence is generated along this path, and the cells outside the path are filled randomly.
    2. **Negative Samples:** Randomly generates grids and verifies with a solver that they indeed have no valid path.
The model's task is to determine if a path exists from (0,0) to (M-1,N-1) in a given parenthesis grid such that the sequence of parentheses along the path is valid.
- **I/O Format:**
    - Input: A binary string of length M*N ('('->0, ')'->1).
    - Output: `[1]` (exists) or `[0]` (does not exist).
- **Main Parameters:** `MAZE_M`, `MAZE_N`, `DATASET_SIZE`.

---

## 51. **generate_sat_solver_text.py**

- **Purpose:** To test the model's ability to solve a landmark NP-complete problem—Boolean Satisfiability (SAT).
- **Logic:** Randomly generates a CNF (Conjunctive Normal Form) formula composed of multiple clauses. The input is a string encoding of this formula. The script then calls an external solver (pycosat) to determine if there exists a variable assignment that makes the formula true. The script strives to ensure a 1:1 ratio of satisfiable and unsatisfiable samples.
- **I/O Format:**
    - Input: A string representing the entire formula.
    - Output: '1' (satisfiable) or '0' (unsatisfiable).
- **Main Parameters:** `num_vars`, `num_clauses`, `num_samples_per_class`.

---

## 52. **generate_sat_solver_compact_text.py**

- **Purpose:** A variant of `generate_sat_solver_text.py` that uses a different input encoding format to solve the same 3-SAT problem.
- **Logic:** The core logic is the same as the previous script, generating labels via an external solver (Z3) and ensuring data balance. The main difference is the input format: this version uses uppercase letters to represent the negation of a variable (e.g., `a` for x1, `A` for ~x1), which is a more compact representation.
- **I/O Format:**
    - Input: A string of length `NUM_CLAUSES` * 3, representing the entire formula.
    - Output: '1' (satisfiable) or '0' (unsatisfiable).
- **Main Parameters:** `VAR_COUNT`, `NUM_CLAUSES`, `NUM_SAMPLES_PER_CLASS`.

---

## 53. **generate_point_in_polygon.py**

- **Purpose:** To test the model's ability to learn a classic algorithm in computational geometry—the Ray Casting Algorithm.
- **Logic:** The script first randomly generates N vertices of a non-self-intersecting polygon, then randomly generates a test point. The input is a string formed by concatenating the binary encodings of all vertex and test point coordinates. The output is a single bit indicating whether the test point is inside the polygon. To ensure the dataset is balanced, the script tries to have roughly equal numbers of inside and outside samples.
- **I/O Format:**
    - Input: A binary string of length (`NUM_VERTICES_N` + 1) * 2 * `BITS_PER_COORD`.
    - Output: `[1]` (inside) or `[0]` (outside).
- **Main Parameters:** `NUM_VERTICES_N`, `BITS_PER_COORD`, `DATASET_SIZE`.

---

## 54. **generate_shortest_path_in_matrix_bfs.py**

- **Purpose:** To test the model's ability to find the shortest path in a 2D grid based on the classic Breadth-First Search (BFS) algorithm.
- **Logic:** The input is an N x N binary matrix where '0' represents a path and '1' represents a wall. The task is to calculate the shortest path length from the top-left corner (0,0) to the bottom-right corner (N-1, N-1) (allowing eight-directional movement). The script uses the BFS algorithm to find the optimal solution. If the two points are not connected, the path length is 0.
- **I/O Format:**
    - Input: An N*N-bit binary string or an (N*N)/4-digit hexadecimal string.
    - Output: A binary multi-label vector of length `OUTPUT_BITS`, representing the path length.
- **Main Parameters:** `MATRIX_SIZE_N`, `INPUT_FORMAT`, `DATASET_SIZE`.

---

## 55. **generate_sudoku_4x4_stepwise_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's ability for "stepwise" reasoning, i.e., predicting only the next optimal action at each state, rather than outputting the complete solution at once.
- **Status:** **Deprecated**. The script attempted to generate stepwise solutions for 4x4 Sudoku using a complex backtracking logic, but its core algorithm was unreliable and could not guarantee the correctness and validity of the generated data. It has been replaced by more robust scripts like `generate_sudoku_6x6.py`.
- **Logic:** (Problematic) Tried to deduce the optimal move at each step by "digging holes" from a complete 4x4 Sudoku solution and checking for uniqueness.

---

## 56. **generate_tiling_problem_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's ability to solve a classic tiling coverage optimization problem, which is NP-hard.
- **Status:** **Deprecated**. The core solver used backtracking search with pruning, but this is an exponential-time algorithm. For matrices larger than about 13x13, the computation time becomes impractical, making it impossible to efficiently generate large-scale datasets.
- **Logic:** Uses a backtracking search method to solve the problem of "tiling an m*n rectangle with the minimum number of squares."

---

## 57. **generate_hanoi_tower_twisted_path_deprecated.py**

- **Purpose:** (Deprecated) This script intended to generate a "twisted path" dataset for the Tower of Hanoi problem, i.e., the optimal path from a non-standard but difficult starting state to the standard end state.
- **Status:** **Deprecated**. The core move logic (`apply_move`) was flawed and did not correctly simulate the "large disk below, small disk above" rule of Tower of Hanoi, resulting in generated paths that were not valid solutions. It was later superseded by more robust scripts like `generate_hanoi_tower_build_full_state_graph.py`.

---

## 58. **generate_checkers_jump_1d_v2.py (Refer to generate_checkers_jump_1d.py)**

- **Purpose:** To solve the 1D checker-swapping planning problem, which has been used to reveal the limitations of large language models on certain types of reasoning tasks.
- **Logic:** The script simulates the process of two colors of checkers ('R' and 'B') moving past each other on a 1D board to reach the other's starting positions. It uses an efficient reverse Breadth-First Search (BFS), starting from the goal state and traversing the entire state space backwards, to compute the unique optimal next move for every reachable state.
- **I/O Format:**
    - Input: An integer sequence of length 2*N+1, representing the board state.
    - Output: A single integer, representing the **position index** of the checker to be moved, which is a classification problem.
- **Main Parameters:** `CHECKERS_N` (number of checkers of each color).

---

# C: Image to Symbol

## 1. **generate_checkerboard_to_binary.py**

- **Purpose:** This is a basic vision-to-symbol conversion task, used to test the model's ability to decode structured information from raw pixel data.
- **Logic:** For each sample, the script generates a random N x N binary grid and renders it as an `IMAGE_SIZE` x `IMAGE_SIZE` black and white checkerboard image. The input is this image, and the output is the corresponding N*N-bit flattened binary string.
- **I/O Format:**
    - Input: An `IMAGE_SIZE` x `IMAGE_SIZE` grayscale image.
    - Output: A binary multi-label vector of length `GRID_DIM` * `GRID_DIM`.
- **Main Parameters:** `NUM_SAMPLES`, `IMAGE_SIZE`, `GRID_DIM`.

---

## 2. **generate_line_angle_to_vector.py**

- **Purpose:** To test the model's ability to extract precise geometric information (angles) from an image, a more advanced visual reasoning task than simple checkerboard recognition.
- **Logic:** The script generates an image resembling a radar scan or a clock face. From the center of the image, it draws several lines with random colors, widths, and angles. The entire 360 degrees are divided into `num_angle_bins` sectors. The model's task is to output a multi-hot encoded vector, indicating which angular bins contain a line segment.
- **I/O Format:**
    - Input: An `image_size` x `image_size` RGB image.
    - Output: A binary multi-label vector of length `num_angle_bins`.
- **Main Parameters:** `image_size`, `num_angle_bins`, `min_lines`, `max_lines`.

---

## 3. **generate_count_shapes_from_image.py**

- **Purpose:** To test the model's ability to perform multiple visual tasks simultaneously: object recognition (shape), attribute recognition (color), and counting (aggregation).
- **Logic:** The script randomly places objects of different shapes (squares, circles, triangles) and colors (red, green, blue) on a white canvas, ensuring they do not overlap. The model's task is to output a 12-bit vector that encodes the total count for each shape and each color. (Note: Since colors are assigned randomly, there might be a slight imbalance in color counts).
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image.
    - Output: A 12-bit binary multi-label vector, encoding the counts for 6 categories respectively.
- **Main Parameters:** `TOTAL_SAMPLES`, `MAX_COUNT_PER_CATEGORY` (only effective for shapes).

---

## 4. **generate_maze_symbolic_to_image.py**

- **Purpose:** To convert a symbolic maze path-planning dataset into an image format to test the ability of visual models (like CNNs, ViTs) to perform path planning directly from pixels.
- **Logic:** The script reads a .jsonl file containing maze layout strings and their corresponding optimal actions. For each line, it renders the maze layout (including walls, paths, start, and end points) into a high-contrast color image. Finally, it generates an image folder and a `labels.csv` file, associating the image filenames with the optimal action labels (a classification of 0-3).
- **I/O Format:**
    - Input: A .jsonl file.
    - Output: JPG images in an `images/` directory and a `labels.csv` metadata file.
- **Main Parameters:** `INPUT_JSONL_FILE`, `OUTPUT_IMAGE_DIR`, `IMAGE_SIZE`, `GRID_DIM`.

---

## 5. **generate_sokoban_symbolic_to_image_no_labels.py**

- **Purpose:** This is a data conversion script used to convert a symbolic Sokoban dataset (.jsonl format) into only image format, for purely visual tasks or as an intermediate step for more complex data processing.
- **Logic:** The script reads a .jsonl file line by line, where each line contains a Sokoban layout string. For each line, it renders the layout string into a 224x224 color image with a specified visual style and saves it. This version **does not generate** a corresponding labels file.
- **I/O Format:**
    - Input: `sokoban_optimized_dataset.jsonl` file.
    - Output: PNG images in an `images/` directory.
- **Main Parameters:** `INPUT_JSONL_PATH`, `OUTPUT_DIR`, `GRID_SIZE`, `CELL_PIXELS`.

---

## 6. **generate_sokoban_symbolic_to_image_with_labels.py**

- **Purpose:** This is a data conversion script used to transform a symbolic Sokoban dataset (.jsonl format) into a complete image classification dataset for training computer vision models (like ViT, Swin Transformer).
- **Logic:** The script reads a .jsonl file line by line, where each line contains a Sokoban layout string and its corresponding optimal action. For each line, it renders the layout string into a 224x224 color image with a specified visual style and writes the image's filename along with the original optimal action label into a `labels.csv` metadata file.
- **I/O Format:**
    - Input: `sokoban_optimized_dataset.jsonl` file.
    - Output: PNG images in an `images/` directory and a `labels.csv` file.
- **Main Parameters:** `INPUT_JSONL_PATH`, `OUTPUT_DIR`, `GRID_SIZE`, `CELL_PIXELS`.

---

# D: Image to Image

## 1. **generate_triangle_to_incircle.py**

- **Purpose:** A landmark experiment demonstrating "carving precise rules with gradient descent." It tests whether the model can learn a purely non-trivial geometric construction rule (the incircle of a triangle).
- **Logic:** For each sample, the script generates a random green triangle as the input image. It then precisely calculates the unique incircle of that triangle and draws this red incircle on the original triangle to create the output image.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing a green triangle).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (triangle + red incircle).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`, `MIN_TRIANGLE_AREA`.

---

## 2. **generate_polygon_to_symmetry_axis.py**

- **Purpose:** To test the model's ability to infer the implicit axis of symmetry from a complete symmetrical figure.
- **Logic:** The script first defines a random axis of symmetry. Then, it randomly generates a set of vertices on one side of the axis and mirrors these vertices to the other side, thus forming a perfectly symmetrical polygon. The input image contains only this polygon, while the output image additionally draws the hidden axis of symmetry on top of it.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing a symmetrical figure).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (symmetrical figure + axis of symmetry).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`, `MIN_POLYGON_VERTICES_HALF`.

---

## 3. **generate_triangle_to_centroid.py**

- **Purpose:** To test the model's ability to learn another fundamental geometric concept—the centroid.
- **Logic:** The script generates a random green triangle as the input image. It then calculates the centroid (center of mass) of the triangle and draws a fixed-size red dot at that location to create the output image.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing a green triangle).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (triangle + red centroid circle).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`, `MIN_TRIANGLE_AREA`.

---

## 4. **generate_triangle_to_tessellation.py**

- **Purpose:** This is a landmark demonstration of our paradigm's capability. It tests whether the model can learn an infinite, lattice-based generative rule. Due to the global correlations and precise details of the tessellation pattern, it strongly rules out the possibility that the model is merely solving the problem through "interpolation" or "memorization."
- **Logic:** The input image contains only a single randomly generated and oriented green triangle. The script uses this triangle as the basis for a "unit cell" and tiles the entire canvas by translating it along two non-collinear basis vectors, alternating with green and red triangles to form a perfect planar tessellation pattern. The output image is this complete tessellation.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing a green triangle).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (the complete tessellation pattern).
- **Main Parameters:** `NUM_SAMPLES`, `IMG_SIZE`.

---

## 5. **generate_game_of_life_image_to_image.py**

- **Purpose:** This is the image-to-image version of the 2D cellular automaton, testing whether the model can execute local rule-based evolution directly in pixel space.
- **Logic:** The script generates a random `GRID_SIZE` x `GRID_SIZE` initial state and renders it as a black and white image for input. It then calculates the next state according to the Game of Life rules and renders it as another image for output.
- **I/O Format:**
    - Input: An `IMAGE_SIZE` x `IMAGE_SIZE` grayscale image (initial state).
    - Output: An `IMAGE_SIZE` x `IMAGE_SIZE` grayscale image (state after one step of evolution).
- **Main Parameters:** `GRID_SIZE`, `IMAGE_SIZE`, `NUM_SAMPLES`.

---

## 6. **generate_projectile_motion_simulation.py**

- **Purpose:** To test the model's ability to learn a simple dynamic physical process. This requires the model to infer the entire spatio-temporal trajectory from initial conditions (position and velocity vector).
- **Logic:** The input image encodes the initial position and velocity vector of a ball using a starting point and a directed line segment (the segment's direction represents velocity direction, and its color represents velocity magnitude). The internal physics engine simulates the ball's parabolic bouncing trajectory in a gravitational field based on these initial conditions. The output image draws the complete trajectory.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing the initial state).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing the full trajectory).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`, `GRAVITY`, `ELASTICITY_FACTOR`.

---

## 7. **generate_snell_refraction_simulation.py**

- **Purpose:** To test the model's ability to learn a fundamental law of physics (Snell's Law of Refraction).
- **Logic:** (Represented by `zheshe2.py`) The input image contains two media of different colors and an incident light ray directed at their interface. The script accurately calculates the path of the refracted light ray based on Snell's Law (n1*sin(θ1) = n2*sin(θ2)). The model's task is to predict the correct refracted light ray based on the input image.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing two media and the incident ray).
    - Output: Same as input, but with the red refracted ray additionally drawn.
- **Main Parameters:** `IMG_SIZE`, `NUM_SAMPLES_TRAIN`.

---

## 8. **generate_snell_refraction_with_contextual_index.py**

- **Purpose:** To test the model's ability to learn a fundamental law of physics (Snell's Law of Refraction) and its capacity to infer a physical parameter (refractive index) from contextual information in the image (background color).
- **Logic:** The input image contains two media of different colors and an incident light ray directed at their interface. The script accurately calculates the path of the refracted light ray based on Snell's Law. In this version, the color of one of the media is functionally related to its refractive index n2. The model's task is to predict the correct refracted light ray based on the input image.
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (containing two media and the incident ray).
    - Output: Same as input, but with the red refracted ray additionally drawn.
- **Main Parameters:** `IMG_SIZE`, `NUM_SAMPLES_TRAIN`.

---

## 9. **generate_cellular_automata_spatial_conditional.py**

- **Purpose:** To test the model's ability to partition and parse "instructions" and "data" within a single modality (image), a kind of "pseudo-multimodal" or "spatially conditioned" experiment.
- **Logic:** The script encodes a 36-bit cellular automaton problem into one image. A narrow strip at the top of the image encodes an 8-bit evolution rule using specific colors (red/green), while the large 6x6 area below encodes the 36-bit initial state with black and white blocks. The output image is the final state after evolving for 3 steps under that rule. The model must learn to "read" the rule from the top and apply it to the state below.
- **I/O Format:**
    - Input: An `IMG_WIDTH` x `IMG_HEIGHT` RGB image (top for rule encoding, bottom for initial state).
    - Output: An `IMG_WIDTH` x `IMG_HEIGHT` RGB image (the final state).
- **Main Parameters:** `NUM_INITIAL_STATES`, `ITERATIONS`, `GRID_DIM`.

---

## 10. **generate_trapping_rain_water_visualizer.py**

- **Purpose:** This is a **data conversion and visualization** script. Its role is to convert an already generated, symbolic "trapping rain water" dataset into an image-to-image format, so that the same problem can be solved with a visual model.
- **Logic:** The script reads a .jsonl file line by line, parsing the column heights and corresponding trapped water amounts for each sample. It first renders the column heights into a black and white image as input. Then, it renders the corresponding rainwater amount in blue above the columns to generate the output image.
- **I/O Format:** Input: .jsonl file -> Output: PNG image pairs in an `images/` directory.
- **Main Parameters:** `input_file`, `output_dir`, `image_size`.

---

## 11. **generate_shortest_path_in_tree_deprecated.py**

- **Purpose:** (Early exploration/Deprecated) An early experiment designed to test the model's ability to find the shortest path in a graph from an image.
- **Status:** **Deprecated**. The script ensured graph planarity by generating random trees, but this inadvertently simplified the problem: the path between any two nodes in a tree is unique, so the model didn't need to learn the concept of "shortest." This task was later replaced by the more challenging dense maze path-planning task.
- **Logic:** The script generates a random tree graph and draws it on an image. The input image highlights a start and an end node. The output image highlights the unique path connecting the start and end nodes on top of the input.
- **Main Parameters:** `MIN_NODES`, `MAX_NODES`.

---

## 12. **generate_shortest_distance_between_triangles.py**

- **Purpose:** To test the model's ability to perform global geometric reasoning (shortest distance) in a scene with multiple objects.
- **Logic:** The script randomly generates two non-overlapping green triangles on a canvas. It then uses a professional computational geometry library, `shapely`, to accurately calculate the shortest distance line segment between these two triangles. The input image contains only the two triangles, while the output image additionally draws this red shortest connecting line.
- **I/O Format:**
    - Input: An `image_size` x `image_size` RGB image (containing two green triangles).
    - Output: An `image_size` x `image_size` RGB image (triangles + red shortest connecting line).
- **Status:** Logic is correct, but depends on the external library `shapely`, which might have environment configuration issues.

---

## 13. **generate_reaction_diffusion_deprecated.py**

- **Purpose:** (Exploratory/Deprecated) This script was used to simulate a reaction-diffusion system to generate complex, fractal-like "snowflake" patterns.
- **Status:** **Deprecated**. The output of this task is the result of a dynamic evolution and is highly sensitive to initial conditions, which does not fit the "mapping from a clear input to a unique deterministic output" that is studied in the paradigm. Therefore, it was deemed "unsuitable" and abandoned.
- **Logic:** The script starts with one or several "nuclei" and gradually generates complex solid structures by iteratively simulating the diffusion and reaction of two fields: Nutrient and Matter.

---

## 14. **generate_cellular_automata_multimodal_deprecated.py**

- **Purpose:** (Deprecated) To generate a truly multimodal dataset for training a model that can understand both image inputs and text instructions simultaneously.
- **Status:** **Deprecated**. Due to the lack of a suitable, easy-to-train multimodal model (within the experimental framework), and since `generate_cellular_automata_spatial_conditional.py` provided a cleaner alternative, the training set generated by this script was not used.
- **Logic:** For each sample, the script generates an image representing the initial state of a cellular automaton and a text string representing the evolution rule. The output is the image of the evolved state.
- **Main Parameters:** `NUM_SAMPLES`, `GRID_DIM`, `ITERATIONS`.

---

### 15. **generate_cellular_automata_1d_to_grid_image_interp.py**

- **Purpose:** This script is designed to create a "logic/perception hybrid" task to demonstrate that a neural network's rule-learning and interpolation abilities are not mutually exclusive but can be integrated within a single task. It forces the model to simultaneously "see through" the continuous grayscale values of the input to perform discrete logical reasoning, and to remember these grayscale values to complete the final continuous value mapping.
- **Logic:** The script first generates a 36-bit logical initial state for a cellular automaton. When generating the input image, cells representing logical '0' are assigned a random dark gray value (e.g., 0-63), while cells representing logical '1' are assigned a random light gray value (e.g., 192-255). Then, the script calculates the final logical output state based on the cellular automaton rule. When generating the output image, it follows a hybrid rule: if a cell's final logical state is '1', its grayscale value remains the same as the corresponding cell in the input image; if the final logical state is '0', its grayscale value becomes the inverse of the input grayscale value (255 - input value).
- **I/O Format:**
    - Input: An `IMG_SIZE` x `IMG_SIZE` RGB image (a 6x6 checkerboard where each cell is a random dark or light gray).
    - Output: An `IMG_SIZE` x `IMG_SIZE` RGB image (a 6x6 checkerboard transformed according to the logical rule and input colors).
- **Main Parameters:** `NUM_SAMPLES`, `IMG_SIZE`, `RULE_NUMBER`, `ITERATIONS`, `ENABLE_INTERPOLATION_MODE`.

---

# E: Text to Image

## 1. **generate_coords_to_triangle.py**

- **Purpose:** This is a basic symbol-to-geometry rendering task, testing the model's ability to convert abstract coordinate information into concrete pixel shapes.
- **Logic:** The script's input is a 48-bit binary string that encodes the (x, y) coordinates of a triangle's three vertices (8 bits per coordinate). The output is an image of a green solid triangle drawn according to these coordinates.
- **I/O Format:**
    - Input: A 48-bit binary string.
    - Output: A 256x256 RGB image.
- **Main Parameters:** `NUM_SAMPLES`, `IMAGE_SIZE`.

---

## 2. **generate_cellular_automata_1d_to_grid_image.py**

- **Purpose:** To test whether the model can directly "render" the results of a 1D symbolic computation into a structured 2D image.
- **Logic:** The input is a 36-bit binary string, representing the initial state of a 1D cellular automaton. The script first internally evolves it for 3 steps according to Rule 110 to get a 36-bit final state. Then, it renders this 1D final state into a 6x6 black and white checkerboard image as the output.
- **I/O Format:**
    - Input: A 36-bit binary string.
    - Output: A 240x240 RGB image (black and white checkerboard).
- **Main Parameters:** `CA_WIDTH`, `RULE_NUMBER`, `ITERATIONS`, `GRID_DIM`.

---

## 3. **generate_triangle_coords_to_tessellation.py**

- **Purpose:** This is an advanced reasoning task that combines symbolic instructions with geometric generation rules.
- **Logic:** Same as `generate_coords_to_triangle.py`, the input is a 48-bit binary string defining a base triangle. Same as `generate_triangle_to_tessellation.py`, the output is a perfect planar tessellation pattern based on this base triangle. The **key modification** is that in the output tessellation pattern, the base triangle directly defined by the input is colored specially (e.g., black), while the other triangles remain green and red. This provides the necessary "grounding" information for the model.
- **I/O Format:**
    - Input: A 48-bit binary string.
    - Output: A 224x224 RGB image (tessellation pattern).
- **Main Parameters:** `NUM_SAMPLES`, `IMG_SIZE`.

---

## 4. **generate_cube_rotation_matplotlib_deprecated.py**

- **Purpose:** (Early exploratory version) Aims to test the model's ability to infer and render the correct view of a 3D object from abstract pose parameters (rotation angles).
- **Logic:** This script uses matplotlib's 3D plotting engine to render the cube. It builds and rotates the object directly in 3D space. While functionally feasible, the complex control over rendering layers in matplotlib could lead to unexpected visual effects in the occlusion relationship between wireframes and filled faces at certain angles.
- **Status:** **Deprecated**. Replaced by subsequent versions based on Pillow, which offer more controllable rendering effects.

---

## 5. **generate_cube_rotation_pillow_v1.py**

- **Purpose:** (Technical upgrade version) Aims to test the model's ability to infer and render the correct view of a 3D object from abstract pose parameters, using a more low-level and precise rendering technique.
- **Logic:** This script represents a major technical refactoring of this task. It abandons the high-level matplotlib library in favor of the more fundamental Pillow library. The script manually implements the complete 3D-to-2D projection transformation, back-face culling based on vector cross products, and depth sorting based on the average depth of faces. This approach provides full control over the rendering pipeline, ensuring that the occlusion relationships and layer order are physically correct at all angles.
- **Status:** This was a key step towards the final successful version, but it lacked the important auxiliary strategy of "highlighting a vertex."

---

## 6. **generate_cube_rotation_pillow_with_anchor.py**

- **Purpose:** (Final version used in the paper) To test the model's ability to infer and render the correct view of a 3D object from abstract pose parameters, and to aid the model's learning by introducing a "visual anchor."
- **Logic:** This script inherits the entire precise, Pillow-based manual rendering pipeline from `generate_cube_rotation_pillow_v1.py`. On top of that, it introduces a **key innovation**: after all regular rendering steps are completed, it always draws a conspicuous highlight marker (an orange dot) on a fixed, special vertex (e.g., the (1,1,1) corner), regardless of whether this vertex is occluded in the current view. This "visual anchor" provides the model with a constant reference, greatly helping it to resolve the inherent symmetries and ambiguities of rotation, thus leading to successful convergence.
- **I/O Format:**
    - Input: A 24-bit binary string (3 angles * 8 bits/angle).
    - Output: A 256x256 RGB image.
- **Main Parameters:** `NUM_SAMPLES`, `IMAGE_SIZE_PX`, `SPECIAL_VERTEX_INDEX`.

---

## 7. **generate_cube_rotation_pillow_wireframe.py**

- **Purpose:** (Variant experiment version) To test whether the model can learn 3D rotation from sparser visual input, using only wireframe and anchor point information.
- **Logic:** This script is an **ablation version** of `generate_cube_rotation_pillow_with_anchor.py`. It retains all core logic, including the precise wireframe drawing and the highlighting of the special vertex, but **removes all face color fills**. This creates a "wireframe mode" dataset, designed to explore whether the model can still understand and reconstruct the 3D structure in the absence of surface information.
- **Status:** This is a variant experiment for in-depth analysis.

---

### 8. generate_cellular_automata_image_and_label.py

- **Purpose:** This is a general dataset generator that creates data for cellular automata tasks in both **Image-to-Image (Img2Img)** and **Image-to-Symbol (Img2Label)** formats.
- **Logic:** The script first generates a random 36-bit binary string as the initial state and renders it as a 6x6 grid image (**input image**). Next, it evolves the initial state internally according to a specified rule (e.g., Rule 110) to obtain the final state. Finally, it renders the final state as a 6x6 grid image (**target image**) and saves its 36-bit binary string (**target label**).
- **I/O Format:**
    - Output 1 (Img2Img): 240x240 Input Image -> 240x240 Target Image.
    - Output 2 (Img2Label): 240x240 Input Image -> 36-bit binary label.
- **Main Parameters:** `CA_WIDTH`, `RULE_NUMBER`, `ITERATIONS`, `NUM_SAMPLES`, `GRID_DIM`.

---

### 9. generate_trapping_rain_water_image_to_symbol.py

- **Purpose:** To generate a dataset for the classic "Trapping Rain Water" algorithm problem in the **Image-to-Vector (Image-to-Symbol)** format.
- **Logic:** The script first generates a random array of column heights (e.g., 12 columns, each height represented by 3 bits). It then renders this total 36-bit input information into a 6x6 black and white grid image. Internally, the script uses a two-pointer algorithm to accurately calculate the amount of rainwater that can be trapped on each column and uses this result, also a 36-bit string, as the output label.
- **I/O Format:**
    - Input: A 240x240 RGB image (a black and white grid representing column heights).
    - Output: A 36-bit binary string (representing the rainwater amount at each position).
- **Main Parameters:** `NUM_COLUMNS_N`, `BITS_PER_HEIGHT`, `DATASET_SIZE`, `GRID_SIZE`.

---

# F: Physics Simulation (Image Paradigm)

## 1. **generate_catenary_curve_simulation_deprecated.py**

- **Purpose:** This was my early exploration of the catenary problem, aiming to test the model's ability to learn non-linear curves determined by physical laws.
- **Logic:** This version was a preliminary attempt at the catenary problem, possibly using a numerical solver to reverse-engineer the catenary equation from given parameters like two endpoints and curve length. This method might have had numerical stability issues and was an early exploratory work, later replaced by the more robust method in `generate_catenary_curve_from_points.py`.
- **I/O Format:**
    - **Input:** An `IMG_SIZE` x `IMG_SIZE` RGB image (containing two endpoints and other information).
    - **Output:** An `IMG_SIZE` x `IMG_SIZE` RGB image (containing the generated catenary curve).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`.

---

## 2. **generate_catenary_curve_from_points.py**

- **Purpose:** To test the model's ability to learn a non-linear curve (catenary) uniquely determined by physical laws (the principle of minimum potential energy).
- **Logic:** The script employs an efficient "forward construction" method: first, it randomly defines the mathematical parameters a, b, c of a catenary; then, it randomly samples three points on this perfect curve (two anchor points P1, P2, and a pass-through point P3). The input image contains only these three points, while the output image draws the complete catenary segment connecting P1 and P2 and passing through P3. This method avoids the complex and unstable process of reverse-engineering parameters from points.
- **I/O Format:**
    - **Input:** An `IMG_SIZE` x `IMG_SIZE` RGB image (containing three points).
    - **Output:** An `IMG_SIZE` x `IMG_SIZE` RGB image (three points + the catenary curve).
- **Main Parameters:** `NUM_SAMPLES_TRAIN`, `IMG_SIZE`.

---

## 3. **generate_orbital_path_from_initial_state.py**

- **Purpose:** To test the model's ability to learn more complex physical laws (Kepler's Laws / Law of Universal Gravitation).
- **Logic:** The script first mathematically defines a random, stable elliptical orbit. Then, it randomly selects a point on the orbit as the planet's initial position and calculates the velocity vector at that point. The input image encodes the star's position, planet's position, planet's velocity direction, and velocity magnitude using different colored points and line segments. The output image, on top of this, draws the complete elliptical orbit.
- **I/O Format:**
    - **Input:** An `IMG_SIZE` x `IMG_SIZE` RGB image (encoding the initial state).
    - **Output:** An `IMG_SIZE` x `IMG_SIZE` RGB image (initial state + complete orbit).
- **Main Parameters:** `NUM_SAMPLES`, `IMG_SIZE`, `G` (gravitational constant).

---

# G: ARC-AGI Exploration

## 1. **generate_arc_contextual_color_swap.py**

- **Purpose:** To test the model's ability to learn a rule from a local "context" or "example" within an image and apply it to the global data in the same image. This directly mimics the core philosophy of the ARC-AGI test.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. First, the logic of an ARC puzzle is manually analyzed and then programmed. The top-left corner of each input image has four color blocks that define two pairs of color swap rules (e.g., the color at (0,0) and the color at (0,1) are swapped). The rest of the image is randomly scattered with dots of these four colors. The model's task is to generate an output image where the colors of all scattered dots have been swapped according to the rule in the top-left corner.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 2. **generate_arc_find_cross_pattern.py**

- **Purpose:** To test the model's ability for visual pattern recognition (or what could be called "object detection") in the presence of a large amount of noise.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image is a red background with a large number of yellow scattered dots. Among them, some yellow dots are carefully arranged into a 3x3 cross pattern, while others are randomly distributed noise. The model's task is to "separate the wheat from the chaff," ignoring all noise dots, accurately finding all hidden cross patterns, and highlighting them in blue in the output image.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 3. **generate_arc_find_odd_one_out.py**

- **Purpose:** To test the model's ability to perform a complex "Find the Odd One Out" meta-reasoning task. The model needs to perform row-wise pattern comparison, identify the exception, and reassemble it in the output.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input is a large grid divided into 4 rows. Each row contains four similar 3x3 small patterns, three of which are identical "normal" patterns, and one is a "special" pattern. The model's task is to identify the "special" pattern in each row and rearrange these four special patterns found from different rows into a 2x2 output grid.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM_IN`, `GRID_DIM_OUT`, `NUM_SAMPLES`.

---

## 4. **generate_arc_connect_colored_pairs.py**

- **Purpose:** To test the model's ability to identify multiple independent "connection tasks" within the same image and understand an implicit "layering" or "drawing priority" rule.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image has several pairs of same-colored dots scattered around. The model's task is to find each pair of same-colored dots and connect them with a line of the corresponding color. An additional, hidden rule is that if a horizontal line and a vertical line intersect, the vertical line is always drawn on top of the horizontal line.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 5. **generate_arc_conditional_perpendicular_lines.py**

- **Purpose:** To test the model's ability to perform different geometric operations based on an object's **attributes (color)** and **global references (boundary lines, image edges)**.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image contains a horizontal gray baseline and some red and blue scattered dots. The model's task is: for each **red** dot, draw a perpendicular line from that dot to the **gray baseline**; for each **blue** dot, draw a perpendicular line from that dot to the **nearest horizontal image edge** (top or bottom).
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 6. **generate_arc_column_projection.py**

- **Purpose:** To test the model's ability to recognize complex contextual relationships ("below... and within the range of...") and perform conditional column operations.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image contains a large, downward-pointing arrow of a specific color and some scattered dots of the same color. The model's task is to find all scattered dots that are directly below the body of the arrow. Then, for each **vertical column** that contains such a "qualified" dot, all pixels in that column of the output image, from the bottom of the arrow to the bottom of the image, are painted with a projection color.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 7. **generate_arc_procedural_spiral.py**

- **Purpose:** To test the model's ability to execute an iterative, procedural generation algorithm. The model needs to understand instructions, track state (current position, direction, length), and execute in a loop.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image is very concise: two color blocks (Color A and Color B) in the top-left corner as instructions, and a blue dot as the "starting point" for drawing. The model's task is to generate an outwardly expanding spiral starting from this blue dot. The drawing of the spiral follows strict rules: the first line segment (length 2) is to the left, with Color A; the second (length 2) is downward, with Color B; the third (length 3) is to the right, with Color A, and so on.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 8. **generate_arc_fractal_stamping.py**

- **Purpose:** To test the model's ability to understand and execute recursive or fractal generation rules. The model needs to use the input pattern itself as a "brush" and repeatedly draw based on "instructions" within the input pattern.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input is a 4x4 pattern. The output is a larger 16x16 canvas. The rule is: iterate through each cell of the input 4x4 pattern. If the cell at position (r, c) is **red**, then the entire 4x4 input pattern is copied ("stamped") onto the output canvas with its top-left corner at (r*4, c*4).
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM_IN`, `GRID_DIM_OUT`, `NUM_SAMPLES`.

---

## 9. **generate_arc_flood_fill.py**

- **Purpose:** To test the model's ability to execute the classic "Flood Fill" or "Paint Bucket" algorithm.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The script first programmatically generates a closed area enclosed by green "walls" on a black background, guaranteed to be connected. The input image is this map with the green enclosure. The output image is the input image but with the black area enclosed by the green walls completely filled with yellow.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 10. **generate_arc_layered_fill.py**

- **Purpose:** To test the model's understanding of a highly procedural and complex filling algorithm that depends on topological distance and conditional logic.
- **Logic:** Adopts a "logic programming-assisted learning" strategy. The input image is divided into multiple regions by line segments. Each region contains one or two "color instruction points." If there is only one color point (A), the entire region is filled with color A. If there are two color points (A and B), the region is filled in "layers": the layer closest to the region's boundary is painted with color A, the next closest layer with color B, and so on, forming a "contour line" pattern of alternating colors.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 11. **generate_arc_fluid_simulation.py**

- **Purpose:** To test the model's ability to learn and simulate a fluid dynamics process with specific rules in the image space.
- **Logic:** The input image contains several red horizontal "baffles" and one or two purple "faucets" at the top. The model's task is to simulate the purple liquid flowing from the faucets. When the liquid encounters a baffle, it splits to the left and right. At edges without baffle support, it continues to drip downwards until it reaches the bottom of the image.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 12. **generate_arc_periodic_conditional_fill.py**

- **Purpose:** This experiment aims to test the model's ability to learn a complex conditional formatting rule with periodicity and special cases.
- **Logic:** The bottom of the input image has a yellow line segment that defines an "operation area." The model needs to check row by row from bottom to top. Based on the distance `d` from the current row to the second-to-last row, a periodic rule modulo 6 is applied:
    - `d % 6` is 1 or 5: Fill the operation area with yellow.
    - `d % 6` is 0, 2, 4: Fill the operation area with the background color.
    - `d % 6` is 3 (special rule): Not only fill the operation area with **green**, but also change **all** original scattered dots in that row to **green**.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 13. **generate_arc_fill_square_holes.py**

- **Purpose:** This experiment is to test the model's multi-step visual reasoning ability: first, it needs to identify complex "foreground in the background" (i.e., holes in rectangles), then judge the geometric properties of the identified objects (whether they are squares), and finally color them based on the judgment.
- **Logic:** The input image contains multiple gray rectangles with black holes. The model's task is to identify all holes, determine the shape of each hole, and if a hole is a **square**, fill it with red in the output image.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 14. **generate_arc_conditional_recoloring.py**

- **Purpose:** To test the model's understanding of visual layers and its ability to perform conditional object attribute modification.
- **Logic:** The input image contains a "bottom layer" composed of dark blue scattered dots and a black background, and a light blue rectangle as a "marker layer." This marker layer is superimposed on the bottom layer but only covers the black background, without altering the original dark blue dots. The model's task is to identify the area of this light blue rectangle and find all the dark blue dots from the **bottom layer** that are within this area, and then change the color of these dots to green in the output image.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 15. **generate_arc_sort_by_length_remap_position.py**

- **Purpose:** To test the model's ability to perform a complex sorting task involving "attribute-position decoupling and remapping."
- **Logic:** The input image contains a series of colored vertical bars of different colors, lengths, and positions. The model's task is to:
    1.  Conceptually, extract the **length** attribute of all bars and sort them.
    2.  Simultaneously, maintain the original horizontal **position** and **color** attributes of all bars.
    3.  In the output image, draw the **shortest** bar at the position of the **leftmost** original bar, using that position's original color; draw the **second shortest** bar at the position of the **second leftmost** original bar, using that position's original color, and so on.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 16. **generate_arc_jigsaw_puzzle_simple.py**

- **Purpose:** To test the model's ability to solve a visual matching and transformation problem (early version).
- **Logic:** The left side of the input image is a template with several puzzle pieces missing. On the right are the corresponding pieces, enlarged and randomly rotated/mirrored. The **key compromise** in this version is that, to simplify the matching problem, each puzzle piece has a unique size (number of squares), which the model can use as a "shortcut" to identify correspondences, rather than relying solely on shape.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`, `num_source_pieces`.

---

## 17. **generate_arc_jigsaw_puzzle_advanced.py**

- **Purpose:** To test the model's ability to solve a complex **visual matching and transformation** problem.
- **Logic:** This is a major improvement on the jigsaw puzzle task. The left side of the input image is a template with several puzzle pieces missing. On the right are the corresponding pieces, scaled up by 2x, randomly rotated and mirrored, along with some noise pieces. The **key improvement** is that this version allows for the generation of multiple puzzle pieces that have the **same size but different shapes**, forcing the model to match **truly based on shape**. The output image shows the result after all pieces are correctly scaled down and placed back into the template.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`, `num_source_pieces`.

---

## 18. **generate_arc_connect_path_by_sequence.py**

- **Purpose:** To test the model's ability to parse an external sequence of instructions and, based on it, perform a multi-step, stateful path connection task within an image.
- **Logic:** The input image contains two parts: (1) several squares with colors inside them, scattered on the canvas; (2) a row of color blocks at the very bottom of the image, which is an "instruction sequence." The model's task is to connect the corresponding squares in the order of the colors in the instruction sequence. For example, if the instruction is [Red, Green, Blue], the model needs to first draw a line from the red square to the green square, and then from the green square to the blue square. An additional rule is that the color of each connecting line is determined by the color of the **previous** square.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** `GRID_DIM`, `NUM_SAMPLES`.

---

## 19. **generate_arc_reflection_simulation_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's understanding of complex rules based on physical optics, including ray emission, collision detection, angular reflection, and color transformation.
- **Logic:** The script attempted to write a generator for this very complex ARC task, but because programmatically generating all possible, physically correct reflection and interaction scenarios unambiguously is extremely difficult, the quality and consistency of the generated data could not be guaranteed. The experiment was ultimately abandoned.
- **I/O Format:** Image-to-Image.
- **Main Parameters:** N/A.

---

# H: Inverse Rule Inference

## 1. **generate_cellular_automata_inverse_rule.py**

- **Purpose:** This experiment was the first attempt to test the model's **Inverse Reasoning** ability. My question was: If the model can forward-propagate a result from a rule, can it then reverse-infer the underlying rule from "input-output" pairs?
- **Logic:** I first randomly select a cellular automaton rule (from 0 to 255). Then, starting from a random initial state, I apply this rule to evolve for a fixed number of steps to get a final state. I concatenate the initial state and the final state as the model's input, and the hidden 8-bit rule as the model's prediction target.
- **I/O Format:**
    - Input: A binary string of length `CA_WIDTH` * 2 (initial_state + final_state).
    - Output: An 8-bit binary multi-label vector (representing the predicted rule).
- **Main Parameters:** `CA_WIDTH`, `NUM_SAMPLES`, `ITERATION_LAYERS`.

---

## 2. **generate_cellular_automata_inverse_rule_and_steps.py**

- **Purpose:** This is an earlier version before the implementation of the "unique solution" version. It also aims to have the model learn to predict both the rule and the number of iterations.
- **Logic:** Like the final `_unique` version, this script also randomly selects a rule and a number of iterations for each sample to generate data. However, it lacks the step to verify the uniqueness of the solution. This means the dataset might contain some ambiguous samples, where multiple (rule, steps) combinations could produce the same input-output pair.
- **I/O Format:**
    - Input: A binary string of length `CA_WIDTH` * 2.
    - Output: A binary multi-label vector of length 8 + `ITERATION_BITS`.
- **Main Parameters:** `CA_WIDTH`, `NUM_SAMPLES`, `MAX_ITERATION_LAYERS`.

---

## 3. **generate_cellular_automata_inverse_rule_and_steps_unique.py**

- **Purpose:** This is a major upgrade to the inverse reasoning task. I not only require the model to infer **what** rule was applied, but also **how many times** it was applied.
- **Logic:** This script inherits the idea from the previous experiment but adds complexity. When generating each sample, I randomly select a rule and a **random number of iterations**. After obtaining the input/output pair, I introduce a crucial **uniqueness verification step**: I brute-force check if any other combination of rule/iteration count could also produce the exact same final state from the same initial state. I only keep samples where the solution is unique, thus providing the model with an unambiguous learning target.
- **I/O Format:**
    - Input: A binary string of length `CA_WIDTH` * 2.
    - Output: A binary multi-label vector of length 8 + `ITERATION_BITS`, concatenating the rule and the iteration count.
- **Main Parameters:** `CA_WIDTH`, `NUM_SAMPLES`, `MAX_ITERATION_LAYERS`.

---

## **I: Robustness Analysis**

### 1. **generate_cellular_automata_1d_perturbed.py**

- **Purpose:** This script aims to systematically test the robustness of the "neural carving" paradigm when faced with imperfect data. By introducing controllable random perturbations to the input (simulating observation noise) and the output (simulating label noise), it explores the continuous spectrum of model performance transitioning from an ideal rule-based world to a noisy real world.
- **Logic:** The script first generates an original initial state for a cellular automaton. Then, it randomly flips bits in this original state according to `INPUT_PERTURBATION_RATE` to get the final "input" sequence. Next, based on the unperturbed original state, it evolves for `EVOLUTION_LAYERS` steps according to the precise cellular automaton rule to get a "correct" output sequence. Finally, it randomly flips bits in this correct output sequence according to `OUTPUT_PERTURBATION_RATE` to get the final "output" label.
- **I/O Format:**
    - Input: A 0/1 string of length `LENGTH`.
    - Output: A list of 0/1 integers of length `LENGTH`.
- **Main Parameters:** `NUM_SAMPLES`, `LENGTH`, `EVOLUTION_LAYERS`, `INPUT_PERTURBATION_RATE`, `OUTPUT_PERTURBATION_RATE`.