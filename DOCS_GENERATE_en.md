# A: Symbolic Math Logic

## 1. **symbolic_math_logic/generate_conditional_add_subtract.py**

- **Logic:** The script generates addition or subtraction (absolute value) problems for two N-bit integers. It includes two modes:
    
    1. **INDICATOR_BIT mode:** Adds a bit before the input (0 for addition, 1 for subtraction) as an explicit instruction.
        
    2. **PROBABILITY_MIX mode:** Provides no instructions, but mixes addition and subtraction samples with a certain probability during data generation. This simulates an environment with "impure rules."
        
- **I/O Format:**
    
    - Input: (INDICATOR_BIT mode) 1 (indicator bit) + 2N (operands) bits; (PROBABILITY_MIX mode) 2N bits.
        
    - Output: N+1 bit binary multi-label vector representing the calculation result.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE, EXPERIMENT_MODE, PROBABILITY_ADD.

---

## 2. **symbolic_math_logic/generate_add_binary_modulo.py**

- **Purpose:** This is an early basic arithmetic experiment to test the model's ability to learn modulo addition (or "truncated addition"), a common operation in fixed-width integer arithmetic in computer hardware.
    
- **Logic:** Input two N-bit integers a and b, calculate their sum, then take modulo 2^N of the result, effectively discarding any overflow carry (e.g., the N+1th bit). Input is encoded as concatenation of two binary strings, output is encoded as multi-label binary classification format.
    
- **I/O Format:**
    
    - Input: N * 2 length binary string (concatenation of two N-bit operands).
        
    - Output: N length multi-label binary classification vector (0/1 list).
        
- **Main Parameters:** n_samples, bit_length.

---

## 3. **symbolic_math_logic/generate_multiply_binary.py**

- **Purpose:** As a benchmark for binary arithmetic capabilities, generates multiplication datasets for N-bit integers.
    
- **Logic:** Randomly generates two NUM_BITS-bit integers, concatenates their binary strings as input, and uses the binary representation of their product as output.
    
- **I/O Format:**
    
    - Input: NUM_BITS * 2 length binary string.
        
    - Output: NUM_BITS * 2 length binary multi-label vector.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 4. **symbolic_math_logic/generate_multiply_binary_no_carry_phase1.py**

- **Purpose:** This is the first phase of the multiplication "decoupling" experiment. It aims to test whether the model can learn the first step of multiplication: bitwise multiplication without carry and staggered addition, decomposing a complex multiplication problem into a simpler counting problem.
    
- **Logic:** Simulates the process of manual multiplication. Input is two N-bit numbers a and b. Output is no longer the final product, but a length 2*N counter vector where the ith counter records how many '1's the ith bit of the final product should have before carry.
    
- **I/O Format:**
    
    - Input: NUM_BITS * 2 length binary string.
        
    - Output: (NUM_BITS * 2) * BITS_PER_COUNTER length binary multi-label vector.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 5. **symbolic_math_logic/generate_multiply_binary_from_counts_phase2.py**

- **Purpose:** This is the second phase of the multiplication "decoupling" experiment. It aims to verify whether an independent model can learn to handle complex carry logic, i.e., calculate the final binary product from a "no-carry count vector."
    
- **Logic:** The script's input is the output format of the previous "no-carry multiplication" task (a counter vector). It calculates the value represented by this vector (i.e., the original a*b) and uses its standard binary representation as output.
    
- **I/O Format:**
    
    - Input: Binary string of length 2N*BITS_PER_COUNTER representing no-carry count values.
        
    - Output: Binary multi-label vector of length 2*N representing the final product.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 6. **symbolic_math_logic/generate_add_hexadecimal.py**

- **Purpose:** Compares the model's learning ability across different symbolic systems. This script aims to verify whether the model learns the abstract mathematical concept of addition or merely binary-specific patterns.
    
- **Logic:** Randomly selects two 16-bit integers. The script generates two independent datasets: one with inputs as binary representation of these numbers, another with inputs as hexadecimal string representation. Both datasets have identical outputs: 17-bit binary representation of the sum.
    
- **I/O Format:**
    
    - Input (binary): 32 ('0'/'1') | Input (hexadecimal): 8 ('0'-'9','A'-'F').
        
    - Output: 17-bit binary multi-label vector.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 7. **symbolic_math_logic/generate_multiply_decimal.py**

- **Purpose:** Tests the model's ability to handle non-binary symbolic inputs (0-9 characters) and perform arithmetic operations (multiplication).
    
- **Logic:** Generates two N-digit decimal numbers, concatenates their strings as input. Calculates their product and converts the result to binary as output.
    
- **I/O Format:**
    
    - Input: NUM_DIGITS * 2 length string composed of '0'-'9'.
        
    - Output: OUTPUT_BITS length binary multi-label vector.
        
- **Main Parameters:** NUM_DIGITS, DATASET_SIZE.

---

## 8. **symbolic_math_logic/generate_add_binary_with_position_shuffle.py**

- **Purpose:** This is the "position shuffle" part of the "semantic shuffle" experiment series. It aims to verify whether the model relies on fixed spatial structure of inputs or can learn position-independent abstract relationships.
    
- **Logic:** The script generates two datasets for binary addition tasks. One has inputs as standard concatenation of two N-bit numbers (a+b). The other has inputs where each bit of the standard input is repositioned according to a predefined, fixed random mapping. Both datasets have identical outputs.
    
- **I/O Format:**
    
    - Input: NUM_BITS * 2 length binary string.
        
    - Output: NUM_BITS + 1 length binary multi-label vector.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 9. **symbolic_math_logic/generate_symbol_add_shuffle_dataset.py**

- **Purpose:** This is a **critical decisive experiment** in our research, aiming to completely separate the model's "surface pattern matching" ability from its "abstract structure learning" ability.

- **Logic:** The script can be configured to perform two types of "shuffling":

    1. **Semantic Shuffle:** Randomly maps symbols representing N-ary digits (e.g., '0'-'F') to any printable characters. This severs the connection between symbols and their inherent numerical meanings.

    2. **Positional Shuffle:** Reposition each character in the input string according to a fixed random mapping. This destroys all local, spatial statistical patterns.

- **I/O Format:**

    - Input: 2 * NUM_BITS length string (character set variable).

    - Output: Binary multi-label vector of the sum.

- **Main Parameters:** NUM_BITS, BASE, SHUFFLE_SEMANTICS, SHUFFLE_POSITIONS.

---

## 10. **symbolic_math_logic/generate_add_hidden_constant.py**

- **Purpose:** Tests the model's ability to **infer hidden rules or parameters** from large amounts of samples without any direct clues. This is similar to a simplified System Identification problem.
    
- **Logic:** The script internally defines a fixed "hidden constant" C. For each sample, it only uses a random number x as input and uses x+C as output. The model must learn the effect of constant C from the commonality of all samples and encode it into its weights.
    
- **I/O Format:**
    
    - Input: NUM_BITS length binary string (representing x).
        
    - Output: NUM_BITS+1 length binary multi-label vector (representing x+C).
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 11. **symbolic_math_logic/generate_multitask_alu.py**

- **Purpose:** This script aims to build a multi-task learning scenario simulating an **Arithmetic Logic Unit (ALU)**. It tests whether the model can perform multiple different, well-defined computational tasks in parallel on the same input in a single forward pass.
    
- **Logic:** Input is two N-bit binary numbers. Output is a long binary vector divided into multiple "address segments," each corresponding to the result of a specific operation (add, subtract, AND, OR, XOR, compare). This forces the model to internally fork the computational flow graph and route results to specified output positions.
    
- **I/O Format:**
    
    - Input: NUM_BITS * 2 length binary string.
        
    - Output: TOTAL_OUTPUT_BITS length binary multi-label vector, concatenated from all task results.
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 12. **symbolic_math_logic/generate_modulo_operation.py**

- **Purpose:** Explores the model's ability to learn modulo operation, a crucial but "cyclic" operation in number theory and computer science.
    
- **Logic:** The script generates an A_BITS-bit integer a and an N_BITS-bit integer n, task is to calculate a % n. In subsequent explorations, we fix n to 3 to deeply investigate why the model struggles to learn this seemingly simple rule.
    
- **I/O Format:**
    
    - Input: A_BITS length binary string (current version only inputs a).
        
    - Output: N_BITS length binary multi-label vector.
        
- **Main Parameters:** A_BITS, N_BITS (though currently fixed to 3), DATASET_SIZE.

---

## 13. **symbolic_math_logic/generate_rsa_encryption.py**

- **Purpose:** Tests the model's ability to learn highly nonlinear, computationally "hard" deterministic rules. RSA encryption is a typical example.
    
- **Logic:** The script encrypts all possible messages (m) from 0 to n-1 under a fixed public key (e, n), generating corresponding ciphertexts (c).
    
- **I/O Format:**
    
    - Input: bits-bit binary string representing message m.
        
    - Output: bits-bit binary string representing ciphertext c.
        
- **Main Parameters:** e, n (public key parameters), bits (encoding bits), output_file.

---

## 14. **symbolic_math_logic/generate_deduction_chain_text.py**

- **Purpose:** Generates multi-step logical reasoning tasks to test the model's ability to perform symbolic deduction, similar to a simplified theorem prover.
    
- **Logic:** The script defines a series of implicit inference rules (e.g., (A, B) -> C). It first builds a multi-step inference chain (e.g., 5 steps), then determines all initial "facts" needed to derive the final conclusion. Positive samples' inputs contain all necessary facts (possibly with some irrelevant "noise" facts), negative samples deliberately miss one or more key facts. The model's task is to determine whether a given Query can be deduced from the given facts and rules.
    
- **I/O Format:**
    
    - Input: Text string in format "Facts: ...\nRules: ...\nQuery: ...".
        
    - Output: '1' (can deduce) or '0' (cannot deduce).
        
- **Main Parameters:** num_samples, attr_range, depth.

---

## 15. **symbolic_math_logic/generate_deduction_multirule_text.py**

- **Purpose:** Tests whether the model can correctly "route" to the appropriate rule and make judgments based on the Query when facing multiple independent, unrelated rules.
    
- **Logic:** The script defines two independent "facts->conclusion" rules. When generating each sample, it first determines the target of this query (5 or 6), then checks the preconditions needed to infer that target. Positive samples provide all necessary conditions (plus some noise facts), negative samples deliberately miss at least one necessary condition.
    
- **I/O Format:**
    
    - Input: Text string in format "Facts: ..., Query: ...".
        
    - Output: Single character '1' (can deduce) or '0' (cannot deduce).
        
- **Main Parameters:** n_samples.

---

## 16. **symbolic_math_logic/generate_deduction_multirule_text_v2.py**

- **Purpose:** Tests whether the model can correctly "route" to the appropriate rule and make judgments based on the Query when facing multiple independent, unrelated rules.
    
- **Logic:** The script defines two independent "facts->conclusion" rules. When generating each sample, it first determines the target of this query (5 or 6), then checks the preconditions needed to infer that target. Positive samples provide all necessary conditions (plus some noise facts), negative samples deliberately miss at least one necessary condition.
    
- **I/O Format:**
    
    - Input: Text string in format "Facts: ..., Query: ...".
        
    - Output: Single character '1' (can deduce) or '0' (cannot deduce).
        
- **Main Parameters:** n_samples.

---

## 17. **symbolic_math_logic/generate_deduction_multirule_binary.py**

- **Purpose:** This is a **format optimization** version of the multi-rule reasoning task, aiming to test whether compact binary encoding is more conducive to model learning than sparse text format.
    
- **Logic:** Core logic is consistent with the text version, but input/output representation is changed:
    
    1. All 8 possible facts are represented as an 8-bit binary mask.
        
    2. Query target (5 or 6) is represented as a single binary bit.
        
    3. These two parts are concatenated into a 9-bit input string.
        
- **I/O Format:**
    
    - Input: 8 (fact mask) + 1 (query target encoding) = 9-bit binary string.
        
    - Output: Single character '1' or '0'.
        
- **Main Parameters:** n_samples.

---

## 18. **symbolic_math_logic/generate_deduction_fixed_depth.py**

- **Purpose:** Tests the model's multi-step reasoning ability in symbolic deduction tasks with clear structure and fixed depth.
    
- **Logic:** The script first randomly generates a 5-step inference chain internally (e.g., A+B->X, C+D->Y, ..., X+Y->Z). Then it uses "backchaining" method, starting from the final conclusion (Z) and backtracking to find all initial facts that must be true.
    
    - **Positive samples:** Input contains mask of all necessary facts, query target is Z, label is '1'.
        
    - **Negative samples:** Input contains same fact mask, but query target is a "noise" attribute that cannot be inferred from these facts, label is '0'.
        
- **I/O Format:**
    
    - Input: 16 (fact mask) + 4 (query target encoding) = 20-bit binary string.
        
    - Output: Single character '1' (can deduce) or '0' (cannot deduce).
        
- **Main Parameters:** depth, num_attrs, num_samples.

---

## 19. **symbolic_math_logic/generate_function_composition.py**

- **Purpose:** Tests the model's ability to learn function composition. This requires the model to act like an interpreter, parsing instructions sequentially and transforming data.
    
- **Logic:** The script defines four basic functions (double, increment, square, decrement). Each sample's input consists of two parts: an instruction string representing a sequence of 4 function calls (each function encoded with 2 bits), and a 16-bit initial integer. The script applies these 4 functions in sequence, ensuring each intermediate result stays within [0, 65535], using the final result as output.
    
- **I/O Format:**
    
    - Input: (4 * 2) (function instructions) + 16 (initial value) = 24-bit binary string.
        
    - Output: 16-bit binary string.
        
- **Main Parameters:** num_samples.

---

## 20. **symbolic_math_logic/generate_count_set_bits.py**

- **Purpose:** Tests the model's ability to perform global aggregation operations. Unlike local rules, counting requires the model to synthesize information across the entire input sequence.
    
- **Logic:** The script generates a random binary string and counts the number of '1's in it. Balanced mode ensures roughly equal sample counts for each count value in the dataset.
    
- **I/O Format:**
    
    - Input: input_bits length binary string.
        
    - Output: output_bits length binary multi-label vector representing the total number of '1's.
        
- **Main Parameters:** num_samples, input_bits, output_bits, balanced.

---

## 21. **symbolic_math_logic/generate_sum_pattern_positions.py**

- **Purpose:** Tests the model's ability to perform more complex, grouped parallel aggregation tasks. The model needs to first split the input, then classify each split pattern, and finally accumulate the **position information** of patterns belonging to the same class.
    
- **Logic:** The script splits a long binary string into q (NUM_PATTERNS) consecutive sub-patterns by p bits (BITS_PER_PATTERN). Then, for each possible sub-pattern (2^p total), it calculates the sum of indices (1 to q) of all positions where that pattern appears in the input.
    
- **I/O Format:**
    
    - Input: p * q length binary string.
        
    - Output: (2^p) * BITS_PER_SUM length binary multi-label vector representing the position sum for each pattern.
        
- **Main Parameters:** BITS_PER_PATTERN, NUM_PATTERNS, DATASET_SIZE.

---

## 22. **symbolic_math_logic/generate_sum_pattern_positions_v2.py**

- **Purpose:** Tests the model's ability to perform more complex, grouped parallel aggregation tasks. The model needs to first split the input, then classify each split pattern, and finally accumulate the **position information** of patterns belonging to the same class.
    
- **Logic:** The script splits a long binary string into q (NUM_PATTERNS) consecutive sub-patterns by p bits (BITS_PER_PATTERN). Then, for each possible sub-pattern (2^p total), it calculates the sum of indices (1 to q) of all positions where that pattern appears in the input.
    
- **I/O Format:**
    
    - Input: p * q length binary string.
        
    - Output: (2^p) * BITS_PER_SUM length binary multi-label vector representing the position sum for each pattern.
        
- **Main Parameters:** BITS_PER_PATTERN, NUM_PATTERNS, DATASET_SIZE.

---

## 23. **symbolic_math_logic/generate_sum_pairwise_hamming_distance.py**

- **Purpose:** Tests the model's ability to perform a complex task requiring two layers of nested aggregation operations. The model needs to first perform global statistics **at each bit position**, then accumulate results **across all bit positions**.
    
- **Logic:** Input is a string concatenated from N M-bit binary numbers. Task is to calculate the sum of pairwise Hamming distances of these N numbers. For example, for [A, B, C], need to calculate dist(A,B) + dist(A,C) + dist(B,C). The script uses an O(N*M) algorithm to efficiently compute this value.
    
- **I/O Format:**
    
    - Input: NUM_ITEMS * BITS_PER_ITEM length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the total Hamming distance sum.
        
- **Main Parameters:** NUM_ITEMS, BITS_PER_ITEM, DATASET_SIZE.

---

## 24. **symbolic_math_logic/generate_circular_shift.py**

- **Purpose:** Tests the model's ability to learn shift operations, particularly circular shift, a common operation in cryptography and low-level programming.
    
- **Logic:** Input consists of two concatenated parts: a NUM_DATA_BITS length binary data string, and a NUM_SHIFT_BITS length binary number (representing the number of bits k to circularly shift right). Output is the result of circularly shifting the data string right by k bits.
    
- **I/O Format:**
    
    - Input: NUM_DATA_BITS + NUM_SHIFT_BITS length binary string.
        
    - Output: NUM_DATA_BITS length binary multi-label vector.
        
- **Main Parameters:** NUM_DATA_BITS, NUM_SHIFT_BITS, DATASET_SIZE.

---

## 25. **symbolic_math_logic/generate_multiply_matrix_3x3.py**

- **Purpose:** Tests the model's ability to learn structured algebraic operations (matrix multiplication), which requires more complex "data routing" and "multiply-accumulate" capabilities than simple scalar operations.
    
- **Logic:** Input is two 3x3 binary matrices, flattened and concatenated into an 18-bit binary string. Output is the 3x3 result matrix (with elements ranging 0-3) after multiplying these two matrices, also flattened and encoded as a binary multi-label vector.
    
- **I/O Format:**
    
    - Input: 18-bit binary string.
        
    - Output: 18-bit binary multi-label vector (9 elements * 2 bits/element).
        
- **Main Parameters:** num_samples.

---

## 26. **symbolic_math_logic/generate_evaluate_boolean_expression_text.py**

- **Purpose:** Tests the model's ability to parse a simple domain-specific language (DSL) and perform evaluation, a step further than previous fixed-structure expression evaluation.
    
- **Logic:** The script randomly generates a boolean expression, e.g., (x0 | x1) & (x2). Simultaneously, it randomly assigns a boolean value (0 or 1) to all variables involved in the expression. Input is concatenation of expression string and variable assignment string, output is the final evaluation result of the expression.
    
- **I/O Format:**
    
    - Input: String in format "x=...;expr=(...)".
        
    - Output: Single character '1' (True) or '0' (False).
        
- **Main Parameters:** num_samples, num_vars.

---

## 27. **symbolic_math_logic/generate_evaluate_arithmetic_expression.py**

- **Purpose:** Trains the model to perform symbolic expression evaluation tasks, requiring the model to understand operator precedence (implicitly expressed through tree structure), variable substitution, and arithmetic operations.
    
- **Logic:** The script first randomly generates an expression tree containing addition, subtraction, multiplication, numeric constants, and variable 'x'. Then it flattens the tree structure into prefix token sequence and performs binary encoding. Finally, it randomly generates a value for 'x', concatenates the expression and x's value as input, and uses the final calculation result as output.
    
- **I/O Format:**
    
    - Input: (TOKEN_LEN * N) + X_BITS bit binary string representing expression and x's value.
        
    - Output: OUTPUT_BITS bit binary string representing evaluation result.
        
- **Main Parameters:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE.

---

## 28. **symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply.py**

- **Purpose:** This is a simplified version of generate_evaluate_arithmetic_expression.py, aiming to reduce learning difficulty by removing multiplication operations to test the model's ability in more basic arithmetic expression evaluation.
    
- **Logic:** Logic is similar to the previous script, but the randomly generated expression tree only contains addition and subtraction operations, completely excluding multiplication. This makes the expression's numeric range more controllable and reduces the model's learning burden.
    
- **I/O Format:**
    
    - Input: (TOKEN_LEN * N) + X_BITS bit binary string representing expression and x's value.
        
    - Output: OUTPUT_BITS bit binary string representing evaluation result.
        
- **Main Parameters:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE.

---

## 29. **symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply_small_range.py**

- **Purpose:** This is a further simplification based on the previous generate_evaluate_arithmetic_expression_no_multiply.py version, further reducing learning difficulty by narrowing the numeric range to precisely diagnose the model's performance bottleneck in the simplest expression evaluation tasks.
    
- **Logic:** Logic is the same as ...nomul_dataset.py script, but VAL_RANGE and X_VAL_RANGE parameters are set to smaller values. This ensures all intermediate values and final results during calculation stay within a small range, making it the lowest difficulty expression evaluation version.
    
- **I/O Format:**
    
    - Input: (TOKEN_LEN * N) + X_BITS bit binary string representing expression and x's value.
        
    - Output: OUTPUT_BITS bit binary string representing evaluation result.
        
- **Main Parameters:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE.

---

## 30. **symbolic_math_logic/generate_check_boolean_equivalence.py**

- **Purpose:** Tests the model's ability to judge logical equivalence in boolean algebra. This is an abstract symbolic reasoning task requiring the model to understand expression structure and boolean operation rules.
    
- **Logic:** The script randomly generates two expressions containing variables ('a','b','c','d') and boolean operators ('&', '|', '~'). It determines whether these two expressions produce the same result in all cases through **truth table** method, i.e., traversing all possible variable assignment combinations.
    
- **I/O Format:**
    
    - Input: String in format "expr1=...;expr2=...".
        
    - Output: Single character '1' (equivalent) or '0' (not equivalent).
        
- **Main Parameters:** n (number of samples), vars (variable set).

---

## 31. **symbolic_math_logic/generate_polynomial_shift_coefficients.py**

- **Purpose:** Tests the model's ability to learn an abstract algebraic transformation rule. This task requires the model to understand the internal structure of polynomial expansion.
    
- **Logic:** Input is 6 integers (representing coefficients of a 5th-degree polynomial a5*x^5 + ... + a0), each coefficient represented with 3-bit binary. Output is the 6 coefficients of the new polynomial after variable substitution x -> x+1, each new coefficient represented with 8-bit binary. The script's core is the poly_eval_at_shifted function, which correctly uses binomial theorem to calculate coefficients of the new polynomial.
    
- **I/O Format:**
    
    - Input: 6 * 3 = 18-bit binary string.
        
    - Output: 6 * 8 = 48-bit binary string.
        
- **Main Parameters:** max_samples.

---

## 32. **symbolic_math_logic/generate_convolution_2d.py**

- **Purpose:** Tests the model's ability to learn 2D convolution (Conv2D), a basic image processing operation, and explores whether it can infer the hidden fixed rule (i.e., the convolution kernel itself) from input-output pairs.
    
- **Logic:** The script fixes a hidden 3x3 binary convolution kernel. It generates two types of datasets: one where inputs contain feature map and convolution kernel, testing the model's ability to directly perform operations; another where inputs only contain feature map, requiring the model to parameterize the hidden convolution kernel into its own weights through learning large amounts of samples.
    
- **I/O Format:**
    
    - Input (visible): (MAP_SIZE^2 + KERNEL_SIZE^2) bit binary string. | Input (hidden): MAP_SIZE^2 bit binary string.
        
    - Output: MAP_SIZE^2 * BITS_PER_OUTPUT_ELEMENT length binary multi-label vector representing convolution result (accumulated value at each pixel).
        
- **Main Parameters:** MAP_SIZE, KERNEL_SIZE, DATASET_SIZE.

---

## 33. **symbolic_math_logic/generate_simple_block_cipher.py**

- **Purpose:** Tests the model's ability to "crack" or learn a simple but nontrivial custom encryption algorithm. This task represents a class of complex symbolic transformation rules with high chaos and avalanche effects.
    
- **Logic:** The script defines a fixed, hidden round key (HIDDEN_KEY) and a simple block cipher algorithm called T-Cipher. It generates ciphertext by performing N rounds of encryption on random plaintext, building training data pairs.
    
- **I/O Format:**
    
    - Input: INPUT_BITS length plaintext binary string.
        
    - Output: INPUT_BITS length ciphertext binary multi-label vector.
        
- **Main Parameters:** INPUT_BITS, NUM_ROUNDS, DATASET_SIZE.

---

## 34. **symbolic_math_logic/generate_sin_function_float32.py**

- **Purpose:** Tests the model's ability to fit continuous, periodic, nonlinear functions (sin(x)) using standard 32-bit floating-point format for input and output.
    
- **Logic:** The script's input is a floating-point number x, using its standard IEEE 754 32-bit binary representation. Output is the calculation result of sin(x), also using its 32-bit binary representation.
    
- **I/O Format:**
    
    - Input: 32-bit binary multi-label vector.
        
    - Output: 32-bit binary multi-label vector.
        
- **Main Parameters:** N (number of samples), x_range.

---

## 35. **symbolic_math_logic/generate_sin_function_float64_to_int12_deprecated.py**

- **Purpose:** This is another encoding attempt for the sin function fitting task, aiming to explore the impact of using higher-precision floating-point input and lower-precision quantized binary output on learning effectiveness.
    
- **Logic:** The script's input is a floating-point number x, using its 64-bit (double precision) binary representation. Output is the result of sin(x), but linearly mapped and quantized to a 12-bit signed integer space.
    
- **I/O Format:**
    
    - Input: 64-bit binary multi-label vector.
        
    - Output: 12-bit binary multi-label vector.
        
- **Status:** (Deprecated) This is an early, problematic version that has been replaced by the more successful generate_sin_function_float32_to_quantized_int.py.

---

## 36. **symbolic_math_logic/generate_sin_function_float32_to_quantized_int.py**

- **Purpose:** Tests the model's ability to fit continuous, periodic, nonlinear functions (sin(x)) and explores the impact of different input/output encoding schemes on learning effectiveness.
    
- **Logic:** This script adopts an effective encoding strategy:
    
    1. **Input:** A floating-point number x, using its standard IEEE 754 32-bit binary representation.
        
    2. **Output:** Calculate y = sin(x) (range [-1, 1]), then linearly map and quantize it to a 24-bit signed integer space. This discretized representation is more suitable for classification models to learn.
        
- **I/O Format:**
    
    - Input: 32-bit binary multi-label vector.
        
    - Output: 24-bit binary multi-label vector.
        
- **Main Parameters:** N (number of samples), x_range.

---

## 37. **symbolic_math_logic/generate_multiply_binary_modulo.py**

- **Purpose:** As part of basic arithmetic experiments, tests the model's mastery of truncated multiplication (or modulo multiplication).
    
- **Logic:** Multiplies two N-bit integers, then performs modulo 2^N operation on the result to ensure output has the same bit width as input operands.
    
- **I/O Format:**
    
    - Input: bits * 2 length binary string.
        
    - Output: bits length binary multi-label vector.
        
- **Main Parameters:** num_samples, bits.

---

## 38. **symbolic_math_logic/generate_explainable_two_step_calculation.py**

- **Purpose:** Tests the model's ability to output "intermediate steps" or "chain of thought" of calculations, a direct verification of "functional interpretability."
    
- **Logic:** Input is three 8-bit binary numbers and two operators. The model is required to output a concatenated vector where the first part is the intermediate result of the first operation and the second part is the final result. This forces the model to not only calculate the answer but also "trace back" and present a key state in its computation process.
    
- **I/O Format:**
    
    - Input: 8*3 (operands) + 2 (operators) length string.
        
    - Output: 8 (intermediate result) + 8 (final result) length binary string.
        
- **Main Parameters:** count.

---

## 39. **symbolic_math_logic/generate_min_swaps_for_checkerboard.py**

- **Purpose:** Solves LeetCode problem 782 "Transform to Chessboard" ([https://leetcode.cn/problems/transform-to-chessboard/](https://leetcode.cn/problems/transform-to-chessboard/)) - calculates the minimum number of row and column swaps needed to transform a 0/1 matrix into "checkerboard" pattern (adjacent elements differ) through arbitrary row and column exchanges.
    
- **Logic:** The script first intelligently generates a guaranteed "solvable" input matrix by performing random row and column swaps on a perfect checkerboard. Then it uses a complex algorithm based on bit operations and combinatorial analysis to precisely calculate the total minimum number of row and column swaps needed to restore the checkerboard pattern. Returns -1 if restoration is impossible.
    
- **I/O Format:**
    
    - Input: N*N length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector (-1 mapped to 0, k moves mapped to k+1).
        
- **Main Parameters:** MATRIX_SIZE_N, DATASET_SIZE.

---

## 40. **symbolic_math_logic/generate_min_flips_for_alternating_binary.py**

- **Purpose:** Tests the model's ability to solve a string optimization problem based on bit flips, which can be cleverly mapped to a sliding window problem.
    
- **Logic:** A "beautiful string" is defined as an alternating 01 sequence (like '0101...' or '1010...'). Input is an arbitrary binary string, task is to calculate the minimum number of flips needed to make it "beautiful."
    
- **I/O Format:**
    
    - Input: STRING_LENGTH_N length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the minimum number of flips.
        
- **Main Parameters:** STRING_LENGTH_N, DATASET_SIZE.

---

## 41. **symbolic_math_logic/generate_min_swaps_for_checkerboard_v2.py**

- **Purpose:** Solves LeetCode problem 1536 "Minimum Swaps to Arrange a Binary Grid" ([https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/](https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/)) - calculates the minimum number of adjacent row swaps needed to transform a binary grid into upper triangular form (all zeros above main diagonal).
    
- **Logic:** The script first randomly generates a binary matrix. Then it uses BFS to search the permutation space, finding the minimum number of adjacent row swaps to transform the matrix into upper triangular form. Returns -1 if transformation is impossible.
    
- **I/O Format:**
    
    - Input: N*N length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector (-1 mapped to 0, k moves mapped to k+1).
        
- **Main Parameters:** MATRIX_SIZE_N, DATASET_SIZE.

---

## 42. **symbolic_math_logic/generate_min_prefix_flips.py**

- **Purpose:** Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing.
    
- **Logic:** This is a classic "prefix flip" or "light bulb" problem. Traverse the sequence from left to right, if the current position is still '0' after considering the cumulative effect of all previous flips, you must "pull" the switch at the current position (which flips all bits from current position to the end) and count one operation.
    
- **I/O Format:**
    
    - Input: STRING_LENGTH_N length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the minimum number of flips.
        
- **Main Parameters:** STRING_LENGTH_N, DATASET_SIZE.

---

## 43. **symbolic_math_logic/generate_min_flips_for_chunked_binary.py**

- **Purpose:** Tests the model's ability to learn a string transformation optimization problem based on local chunks.
    
- **Logic:** Input is an even-length binary string. Split it by every two bits, for each 2-bit chunk, if the two bits are different (like '01' or '10'), one flip operation is needed to make it "beautiful" (become '00' or '11'). Task is to calculate the total minimum number of flips needed.
    
- **I/O Format:**
    
    - Input: INPUT_BITS length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the total number of flips.
        
- **Main Parameters:** INPUT_BITS, DATASET_SIZE.

---

## 44. **symbolic_math_logic/generate_largest_island_by_adding_one_cell.py**

- **Purpose:** Solves an algorithm problem involving graph traversal and global optimization ([LeetCode 827. Making a Large Island](https://leetcode.cn/problems/making-a-large-island/)). The model needs to evaluate all possible "land reclamation" positions and select the one that can maximize the merged island area.
    
- **Logic:** The script first uses DFS or BFS to traverse the input grid, marking all existing islands and calculating their areas. Then it traverses all water grid points ('0'), calculating which adjacent islands can be connected if that point becomes land, and thereby calculating the new total area formed. Finally, it finds the best position that can produce the maximum area.
    
- **I/O Format:**
    
    - Input: N*N length binary string.
        
    - Output: A JSON object containing output_class (category label of best position) and output_area (binary string of maximum area).
        
- **Main Parameters:** NUM_SAMPLES, GRID_SIZE.

---

## 45. **symbolic_math_logic/generate_largest_island_by_adding_one_cell_v2.py**

- **Purpose:** Solves an algorithm problem involving graph traversal and global optimization ([LeetCode 827. Making a Large Island](https://leetcode.cn/problems/making-a-large-island/)). The model needs to evaluate all possible "land reclamation" positions and select the one that can maximize the merged island area.
    
- **Logic:** The script first uses DFS or BFS to traverse the input grid, marking all existing islands and calculating their areas. Then it traverses all water grid points ('0'), calculating which adjacent islands can be connected if that point becomes land, and thereby calculating the new total area formed. Finally, it finds the best position that can produce the maximum area.
    
- **I/O Format:**
    
    - Input: N*N length binary string.
        
    - Output: A JSON object containing output_class (category label of best position) and output_area (binary string of maximum area).
        
- **Main Parameters:** NUM_SAMPLES, GRID_SIZE.

---

## 46. **symbolic_math_logic/generate_sat_solver_text.py**

- **Purpose:** Tests the model's ability to solve the iconic NP-complete problem - Boolean Satisfiability (SAT) problem.
    
- **Logic:** Randomly generates a CNF (Conjunctive Normal Form) formula consisting of multiple clauses. Input is a string encoding of this formula. Then the script calls an external solver (pycosat) to determine whether there exists a variable assignment that makes the formula true. The script strives to ensure a 1:1 ratio of satisfiable and unsatisfiable samples.
    
- **I/O Format:**
    
    - Input: String representing the entire formula.
        
    - Output: '1' (satisfiable) or '0' (unsatisfiable).
        
- **Main Parameters:** num_vars, num_clauses, num_samples_per_class.

---

## 47. **symbolic_math_logic/generate_sat_solver_compact_text.py**

- **Purpose:** This is a variant of symbolic_math_logic/generate_sat_solver_text.py, using a different input encoding format to solve the same 3-SAT problem.
    
- **Logic:** Core logic is the same as the previous script, both generating labels through external solver (Z3) and ensuring data balance. Main difference is input format: this version uses uppercase letters to represent variable negation (e.g., a represents x1, A represents ~x1), which is a more compact representation.
    
- **I/O Format:**
    
    - Input: NUM_CLAUSES * 3 length string representing the entire formula.
        
    - Output: '1' (satisfiable) or '0' (unsatisfiable).
        
- **Main Parameters:** VAR_COUNT, NUM_CLAUSES, NUM_SAMPLES_PER_CLASS.

---

## 48. **generate_binary_mod3_dfa_explain.py**

- **Purpose:** This is an **interpretability** experiment designed to test whether the model can learn and output the internal state transition process of a Deterministic Finite Automaton (DFA).
    
- **Logic:** The script generates random N-bit binary numbers. It not only calculates the result of the number modulo 3, but also records the state transition trajectory (S0, S1, S2) of the DFA as it processes each bit of input. This forces the model to not only give the final answer, but also demonstrate how it derived it step by step.
    
- **I/O Format:**
    
    - Input: N-bit binary string.
        
    - Output: JSON object containing `final_mod_result` (2 bits) and `dfa_state_trace` (N * 2 bits).
        
- **Main Parameters:** NUM_BITS, DATASET_SIZE.

---

## 49. **symbolic_math_logic/generate_add_binary_explainable.py**

- **Purpose:** Tests whether the model can learn the internal mechanism (carry propagation) of binary addition. This is an interpretability experiment that requires the model to output not only the result but also the carry status of each bit.
    
- **Logic:** Simulates bitwise binary addition. Input is two N-bit binary numbers. Output contains two parts: one is the carry-free sum (XOR result) of each bit, and the other is the carry-in of each bit. This forces the model to explicitly represent the carry chain.
    
- **I/O Format:**
    
    - Input: N * 2 length binary string.
        
    - Output: JSON object containing `output` (2*N bits, first N bits are current bit result, last N bits are carry) and `sum_output` (N+1 bits final sum).
        
- **Main Parameters:** NUM_BITS, NUM_SAMPLES.

---

# B: Algorithm Learning

## 1. **algorithms/generate_sort_integers.py**

- **Purpose:** Tests the model's ability to perform basic sorting algorithms, a non-local classic algorithm task requiring comparison and rearrangement of input elements.
    
- **Logic:** Input is a binary string concatenated from NUM_ITEMS unordered integers of NUM_BITS_PER_ITEM bits. Output is the binary string re-concatenated after sorting these numbers in ascending order. The script ensures all numbers in the input are unique.
    
- **I/O Format:**
    
    - Input: NUM_ITEMS * NUM_BITS_PER_ITEM length binary string.
        
    - Output: NUM_ITEMS * NUM_BITS_PER_ITEM length binary multi-label vector.
        
- **Main Parameters:** NUM_ITEMS, NUM_BITS_PER_ITEM, DATASET_SIZE.

---

## 2. **algorithms/generate_edit_distance.py**

- **Purpose:** Tests the model's ability to learn to solve dynamic programming problems. Edit distance is a typical DP problem requiring the model to conceptually build a 2D solution matrix. [LeetCode 72. Edit Distance](https://leetcode.com/problems/edit-distance/description/)
    
- **Logic:** Input is concatenation of two equal-length binary strings s1 and s2. Output is the binary representation of their minimum edit distance (allowing insert, delete, replace operations).
    
- **I/O Format:**
    
    - Input: NUM_BITS_PER_STRING * 2 length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector.
        
- **Main Parameters:** NUM_BITS_PER_STRING, DATASET_SIZE.

---

## 3. **algorithms/generate_edit_distance_explainable.py**

- **Purpose:** This is a core experiment in "functional interpretability." It requires the model to not only give the final answer (edit distance) but also output the complete "chain of thought" (edit process) to reach the answer. [LeetCode 72. Edit Distance (Explainable / Path Construction Version)](https://leetcode.com/problems/edit-distance/description/)
    
- **Logic:** Input is two strings s1 and s2. Output is a long vector concatenated from max_steps "state frames." Each state frame contains two parts: binary representation of an intermediate string during the edit process, and a mask indicating the valid length of that string. This requires the model to learn to simulate the step-by-step transformation from s1 to s2. The script uses a clever mechanism to only keep samples where the optimal edit path is unique, ensuring unambiguous labels.
    
- **I/O Format:**
    
    - Input: str_len * 2 length binary string.
        
    - Output: max_edits * str_len * 2 length binary multi-label vector.
        
- **Main Parameters:** num_samples, str_len, max_edits.

---

## 4. **algorithms/generate_maze_random_walls.py**

- **Purpose:** Tests the model's basic pathfinding ability in randomly generated "porous" mazes.
    
- **Logic:** The script generates mazes by randomly placing walls on a grid. This method typically produces mazes with short paths, high connectivity, and relatively simple structure. Then, for all passable points, it uses reverse BFS from a fixed endpoint to calculate the shortest path to that point. The model's task is to predict the optimal first-step direction from the start point given a maze layout containing start and end points.
    
- **I/O Format:**
    
    - Input: H * W length string representing maze layout.
        
    - Output: 4-class category label (up/down/left/right).
        
- **Main Parameters:** MAZE_HEIGHT, MAZE_WIDTH, TARGET_NUM_SAMPLES.

---

## 5. **algorithms/generate_maze_dense.py**

- **Purpose:** Tests the model's ability to perform path planning in complex, human-designed "dense" mazes, which is more challenging than random wall mazes.
    
- **Logic:** The script first uses a specialized maze generation algorithm (like recursive division) to create a challenging, connected dense maze characterized by long, winding passages. Then, similar to the previous script, it uses reverse BFS to calculate optimal strategies for all reachable points.
    
- **I/O Format:**
    
    - Input: H * W length string representing maze layout.
        
    - Output: 4-class category label.
        
- **Main Parameters:** MAZE_HEIGHT, MAZE_WIDTH, TARGET_NUM_SAMPLES.

---

## 6. **algorithms/generate_blocks_world_arbitrary_goal.py**

- **Purpose:** Solves the classic "Blocks World" planning problem. This problem, as a standard task for measuring large language model reasoning ability, is discussed in detail in Apple's famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity," as a benchmark test for evaluating model capabilities in state space search and planning. The study systematically reveals the fundamental limitations of current large language models in precise symbolic reasoning and state space planning through classic planning tasks like blocks problems, river crossing problems, and Tower of Hanoi. This script precisely implements the general version of "Blocks World" from the paper, allowing specification of arbitrary initial and goal states, as a core control experiment for verifying neural network reasoning capabilities on complex planning problems.
    
- **Logic:** The script randomly generates an initial state and a goal state for each sample. Then it uses Breadth-First Search (BFS) to find the shortest action sequence from the initial state to the goal state. The model's task is to predict the first optimal action in this sequence.
    
- **I/O Format:**
    
    - Input: Binary encoding of initial state and goal state.
        
    - Output: 6-class category label representing the optimal action.
        
- **Main Parameters:** BLOCKS_N (number of blocks).

---

## 7. **algorithms/generate_blocks_world_fixed_goal.py**

- **Purpose:** This is a simplified version of the "Blocks World" task. By fixing the goal state (all blocks orderly stacked on the first pillar), it aims to test the model's learning ability in situations with clear goals and more structured state space. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study points out that even for such goal-defined planning problems, large language models still have significant difficulties in state space search and optimal strategy learning. This script implements the simplified version of Blocks World from the paper, reducing task complexity by fixing the goal state, as a control experiment for studying the impact of goal clarity on model reasoning performance.
    
- **Logic:** The script sets a fixed goal state (all blocks orderly on the first pillar). Then it efficiently traverses all reachable states and calculates the optimal strategy from each state to the goal by performing **reverse** Breadth-First Search (BFS) from the goal state.
    
- **I/O Format:**
    
    - Input: Encoding of blocks' initial state.
        
    - Output: 6-class category label representing the optimal action.
        
- **Main Parameters:** BLOCKS_N.

---

## 8. **algorithms/generate_blocks_world_fixed_goal_multilabel.py**

- **Purpose:** Further improves the "Blocks World" task by allowing multiple optimal solutions, testing the model's ability to handle multi-label classification problems, more realistically reflecting possible equivalent optimal paths in planning problems. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study emphasizes that real-world planning problems often have multiple equivalent optimal solutions, posing higher requirements for the model's ambiguous reasoning capabilities. This script further improves upon the fixed-goal version, finding all actions that lead to optimal paths for each state and generating multi-hot encoded outputs, as a control experiment for studying neural networks' ability to handle planning problems with multiple optimal solutions.
    
- **Logic:** Inherits the fixed goal and reverse search logic from the previous script. The key improvement is in output format: for each state, the script finds **all** optimal actions that bring it one step closer to the goal and generates a multi-hot encoded output vector.
    
- **I/O Format:**
    
    - Input: Encoding of blocks' initial state.
        
    - Output: NUM_ACTIONS length binary multi-label vector.
        
- **Main Parameters:** BLOCKS_N.

---

## 9. **algorithms/generate_blocks_world_fixed_goal_multilabel_fixed_format.py**

- **Purpose:** This is the final optimized version of the "Blocks World" task, aiming to provide the model with a clearer, more structured learning objective by improving input representation. This problem also originates from the benchmark test tasks in Apple's paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity." The study points out that input representation has a crucial impact on model learning efficiency and final performance. This script further optimizes upon the multi-label version, using fixed-slot representation to replace variable-length input, eliminating complexity brought by serialization, providing more friendly structured input for Transformer and other architectures. This makes it a control experiment for studying the impact of input representation on neuro-symbolic reasoning performance.
    
- **Logic:** Core logic is the same as the previous script (multi-label output, fixed goal, reverse search). The key improvement is in input format: instead of using separators, it allocates a fixed number of "slots" for each pillar to represent the state. This fixed-length representation eliminates the complexity of variable-length input and is more friendly to Transformer and other models.
    
- **I/O Format:**
    
    - Input: NUM_BLOCKS * NUM_STACKS length string, 0 represents empty, 1-N represents blocks.
        
    - Output: NUM_ACTIONS length binary multi-label vector.
        
- **Main Parameters:** BLOCKS_N, NUM_STACKS.

---

## 10. **algorithms/generate_checkers_jump_1d.py**

- **Purpose:** Solves the checker exchange planning problem in one-dimensional space. This problem, as a standard task for measuring large language model reasoning ability, is discussed in detail in Apple's famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity," as a benchmark test for evaluating model capabilities in state space search and planning. The study systematically reveals the fundamental limitations of current large language models in precise symbolic reasoning and state space planning through classic planning tasks like blocks problems, river crossing problems, and Tower of Hanoi. This script precisely implements the general version of "checker exchange" from the paper, as a core control experiment for verifying neural network reasoning capabilities on complex planning problems.
    
- **Logic:** The script simulates the process of two types of checkers ('R' and 'B') crossing each other on a one-dimensional board. It uses efficient reverse Breadth-First Search (BFS) from the goal state, traversing the entire state space in reverse, thereby calculating the unique optimal next step for every reachable state.
    
- **I/O Format:**
    
    - Input: 2*N+1 length integer sequence representing board state.
        
    - Output: Single integer representing the **position index** of the checker to move.
        
- **Main Parameters:** CHECKERS_N (number of checkers of each color).

---

## 11. **algorithms/generate_river_crossing_puzzle.py**

- **Purpose:** Solves the classic constraint satisfaction and state space search problem - "N couples crossing the river." This problem requires transporting everyone to the other side under the constraint that "no woman can be with other men without her partner present." This task originates from a famous paper [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" by Apple, which reveals fundamental limitations of large language models in certain types of reasoning tasks through benchmark tests like river crossing, Tower of Hanoi, and checkers. This script precisely reproduces the "N couples crossing the river" problem from the paper, as a control experiment for verifying neural network symbolic reasoning capabilities.
    
- **Logic:** The script defines each state as "the set of people on the left bank" and "the boat's position." It builds an optimal strategy graph covering all reachable states through efficient reverse Breadth-First Search (BFS) from the goal state (everyone on the right bank). The output is a multi-label vector indicating who should board the boat together to perform the optimal move in the current state.
    
- **I/O Format:**
    
    - Input: 2*N+1 length binary string (N clients C, N agents A, 1 boat position).
        
    - Output: 2*N length multi-label binary vector indicating whether each person boards the boat.
        
- **Main Parameters:** PAIRS_N (number of couples), BOAT_CAPACITY_K.

---

## 12. **algorithms/generate_trapping_rain_water_aggregate.py**

- **Purpose:** This is an initial attempt to solve the "trapping rain water" algorithm problem, aiming to test the model's ability to learn an aggregated output (rather than decoupled output). Experimental results show that requiring the model to directly output a sum value (a single aggregated number) is much more difficult than outputting detailed information for each position. **This becomes a key comparative experiment, proving the systematic impact of output format design on model learning efficiency.** Corresponding LeetCode problem: [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/). This script precisely implements LeetCode's original aggregated output format (only outputting total amount), forming a sharp contrast with `generate_trapping_rain_water_decoupled.py` (decoupled output, outputting water amount at each position), used to study the impact of output representation on neural network learning difficulty, verifying the core finding of "decoupling accelerates convergence" in the paper.
    
- **Logic:** Input is a one-dimensional height map. The script calculates the total amount of rainwater that can be trapped on this height map and uses this single integer value as output.
    
- **I/O Format:**
    
    - Input: N * K bit binary string representing heights of N columns.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing total rainwater amount.
        
- **Main Parameters:** NUM_COLUMNS_N, BITS_PER_HEIGHT.

---

## 13. **algorithms/generate_trapping_rain_water_decoupled.py**

- **Purpose:** Solves the classic "trapping rain water" algorithm problem (LeetCode Hard [#42](https://leetcode.com/problems/trapping-rain-water/)). The success of this task demonstrates the model's ability to learn complex algorithms requiring global information (like global maximum points), and through the idea of **problem decoupling**, proves the huge impact of output format design on model learning efficiency.
    
- **Logic:** The script designs output through a key insight of problem decoupling. Instead of having the model directly predict a single aggregated value (total rainwater amount), it requires the model to predict a sequence isomorphic to the input structure, where each element represents the amount of water trapped on the corresponding column. This change greatly simplifies the learning task, enabling the model to successfully converge.
    
- **I/O Format:**
    
    - Input: N * K bit binary string representing heights of N columns.
        
    - Output: N * K bit binary multi-label vector representing water trapped by each of the N columns.
        
- **Main Parameters:** NUM_COLUMNS_N, BITS_PER_HEIGHT.

---

## 14. **algorithms/generate_trapping_rain_water_2d.py**

- **Purpose:** As an extension of the one-dimensional "trapping rain water" problem, solves the two-dimensional version. This task requires the model to understand "enclosure" and "boundary" concepts in two-dimensional space, representing a more complex global information processing challenge. Corresponding LeetCode problem: [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/) (Hard difficulty).
    
- **Logic:** Also adopts the idea of problem decoupling. Input is a two-dimensional height map (matrix), output is a matrix of the same size where each cell's value represents the amount of water that can be trapped at that position. The solver determines the water level at each point through a BFS-like "flooding" operation from the boundary inward.
    
- **I/O Format:**
    
    - Input: N*M*K bit binary string representing heights of N*M grid.
        
    - Output: N*M*K bit binary multi-label vector representing water trapped by each cell.
        
- **Main Parameters:** GRID_N, GRID_M, BITS_PER_HEIGHT.

---

## 15. **algorithms/generate_skyline_max_height_aggregate.py**

- **Purpose:** This is an initial attempt to solve the "skyline" problem, requiring the model to predict only the highest height value from all buildings' final heights. This task is used to compare learning difficulty between aggregated and decoupled outputs. Corresponding LeetCode problem: [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/) (aggregated output version).
    
- **Logic:** Input is a series of building height limits. Under the constraint that adjacent building height differences cannot exceed 1, the script uses dynamic programming to calculate the maximum possible height for each building, then finds the maximum value among all buildings as output.
    
- **I/O Format:**
    
    - Input: n * bit_count length binary string representing each building's height limit.
        
    - Output: bit_count length binary multi-label vector representing the global maximum height.
        
- **Main Parameters:** NUM_SAMPLES, FIXED_N (number of buildings), MAX_HEIGHT.

---

## 16. **algorithms/generate_skyline_all_heights_decoupled.py**

- **Purpose:** Tests the model's ability to solve a global optimization problem with one-dimensional spatial constraints. The problem prototype is LeetCode [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/) (decoupled output version). The problem requires: given N buildings' positions and height limits, adjacent building height differences cannot exceed 1, find the maximum height each building can achieve. Through decoupled output, the model is required to predict the height of each building, not just the maximum value, as a control experiment for studying the impact of output format design on model learning efficiency.
    
- **Logic:** Input is a series of building height limits. The rule is that adjacent building height differences cannot exceed 1 while satisfying all constraints. This script uses an efficient bidirectional dynamic programming algorithm to solve for the maximum possible height of each building under these constraints. Output is a sequence of final heights of all buildings.
    
- **I/O Format:**
    
    - Input: n * bit_count length binary string representing each building's initial height limit.
        
    - Output: n * bit_count length binary multi-label vector representing each building's final height.
        
- **Main Parameters:** NUM_SAMPLES, FIXED_N (number of buildings), MAX_HEIGHT.

---

## 17. **algorithms/generate_hanoi_tower_path_strategy_sep_format.py**

- **Purpose:** As comparison group A, this script generates Tower of Hanoi optimal path strategy datasets using **separator + binary encoding** input format.
    
- **Logic:** Standard recursive solver generates optimal solution path. Input is token serialization of states (like `[3, 2, 1, SEP, SEP]`), then each token is converted to fixed-width **binary string** and concatenated. This format simulates unstructured linear input.
    
- **I/O Format:**
    
    - Input: Long binary string (encoded from token sequence).
        
    - Output: 6-class action labels (representing source and target pillars).
        
- **Main Parameters:** HANOI_N.

---

## 18. **algorithms/generate_hanoi_tower_path_strategy_fixed_format.py**

- **Purpose:** As comparison group B, this script also generates Tower of Hanoi optimal path strategy datasets but uses **structured fixed-slot** input format (renamed from original `global` script to reflect true logic), used to verify the impact of data representation on learning efficiency.
    
- **Logic:** Also based on recursive solver generating optimal path. The difference is in input encoding: expand 3 pillars into fixed-length arrays, each position corresponds to a physical slot, filled with disk ID or 0 (empty slot). This format preserves spatial structure information.
    
- **I/O Format:**
    
    - Input: Fixed-length numeric string (like `"321000400"`).
        
    - Output: 6-class action labels.
        
- **Main Parameters:** HANOI_N, dataset_size.

---

## 19. **algorithms/generate_hanoi_tower_compare_formats.py**

- **Purpose:** This is a comparative experimental script that generates two different input formats (separator vs. fixed slot) for the same Tower of Hanoi problem, used to systematically evaluate the impact of different data representations on the model's ability to learn recursive strategies.
    
- **Logic:** The script simultaneously generates two datasets, one using sep format and another using fixed slot format. Both datasets only contain states on the optimal path and require the model to predict the next optimal action.
    
- **I/O Format:**
    
    - Input: sep format or fixed slot format.
        
    - Output: 6-class category label.
        
- **Main Parameters:** HANOI_N, DATASET_SIZE.

---

## 20. **algorithms/generate_hanoi_tower_compare_formats_and_strategies.py**

- **Purpose:** This is a more comprehensive Tower of Hanoi comparative experimental script. It not only generates two input formats but also generates two different datasets: one containing only states on the optimal path ("path strategy"), another containing all reachable states ("global strategy"), used to explore differences in the model's ability to learn local optimal paths and global optimal strategies.
    
- **Logic:** The script generates a total of four datasets (2 formats x 2 strategies). Experimental results show that the model can easily learn "path strategy" but struggles with learning "global strategy," revealing potential limitations of the model in handling recursion and state space explosion problems.
    
- **I/O Format:**
    
    - Input: sep format or fixed slot format.
        
    - Output: 6-class category label.
        
- **Main Parameters:** HANOI_N, DATASET_SIZE.

---

## 21. **algorithms/generate_hanoi_tower_build_full_state_graph.py**

- **Purpose:** This is the culmination of "Tower of Hanoi problem" research, aiming to deeply analyze the model's understanding of recursive structures through multiple different data representations and sampling strategies. It is a self-contained data factory. This research corresponds to the Apple Research related work cited in the paper, focusing on analyzing the model's mastery of recursive structures through complete state graphs.
    
- **Logic:** The core of this script is an extraordinary implementation: instead of using traditional recursive solvers, it directly builds the complete "state-action" graph of the Tower of Hanoi problem with 3^N states in memory through an elegant mathematical structure based on the fractal and self-similarity properties of the Tower of Hanoi graph. This enables subsequent arbitrary, efficient data sampling from this complete knowledge base.
    
- **I/O Format:**
    
    - Output: Multiple .jsonl files (like hanoi_n10_path_slots_train_all.jsonl), containing input states (state_to_slots_B format) and output actions (integer labels 0-5).
        
- **Main Parameters:** HANOI_N.

---

## 22. **algorithms/generate_hanoi_tower_sample_from_state_graph.py**

- **Purpose:** This is a post-processing and sampling script that uses the complete knowledge base generated by generate_hanoi_tower_build_full_state_graph.py to precisely extract specific types of training data subsets, such as "twisted path" or "hardest part," for more refined ablation experiments.
    
- **Logic:** The script first loads the complete state graph generated by the _mine script. Then it can precisely extract the complete optimal path connecting two specified points based on user-specified start and end states, and save it as a trainable .jsonl file.
    
- **I/O Format:**
    
    - Input: `all_states_n{N}.json` file (generated by `generate_hanoi_tower_build_full_state_graph.py`).
        
    - Output: .jsonl format training data.
        
- **Main Parameters:** HANOI_N, start_idx, end_idx.

---

## 23. **algorithms/generate_sokoban_planning_astar.py**

- **Purpose:** Solves the classic "Sokoban" planning problem. This task is harder than simple path planning because it involves changing environmental states (box positions) and has a huge state space.
    
- **Logic:** The script first randomly generates a maze layout containing walls, player, single box, and single goal. Then it uses efficient A* search algorithm to find the optimal action sequence to push the box to the goal position. The model's task is to predict the player's next optimal action (up/down/left/right) given a situation, which may only move the player or may push the box.
    
- **I/O Format:**
    
    - Input: M * N length string representing Sokoban layout.
        
    - Output: 4-class category label.
        
- **Main Parameters:** M_DIMENSION, N_DIMENSION, NUM_SAMPLES.

---

## 24. **algorithms/generate_sokoban_planning_full.py**

- **Purpose:** Solves the classic "Sokoban" planning problem. This is a high-difficulty AI task because it involves searching in a huge state space and actions change environmental states.
    
- **Logic:** This is a very mature dataset generator.
    
    1. **Intelligent Generation:** It generates puzzle layouts that are both random and potentially interesting by randomly placing walls and performing random walks backward from the goal.
        
    2. **A* Solver:** Uses efficient A* search algorithm with Manhattan distance as heuristic function to calculate the shortest Sokoban path from initial state to goal state.
        
    3. **Optimal Strategy Extraction:** For all states on the optimal path, it calculates all actions that can lead to the next optimal state and generates a multi-label (multi-hot) output.
        
    4. **Quality Control:** It includes difficulty filters (only keeps solutions within specific step ranges) and performs global deduplication and shuffling.
        
- **I/O Format:**
    
    - Input: (M-2)*(N-2) length string representing Sokoban layout with boundary walls removed.
        
    - Output: 4-bit multi-label binary vector representing whether up/down/left/right directions are optimal actions.
        
- **Main Parameters:** M_DIMENSION, N_DIMENSION, NUM_SAMPLES, MIN/MAX_DIFFICULTY.

---

## 25. **algorithms/generate_sokoban_planning_claude_deprecated.py**

- **Purpose:** (Deprecated) This is an early attempt with more complex logic, but it failed to stably generate high-quality datasets and has been replaced by the more reliable generate_sokoban_planning_full.py.
    
- **Logic:** This script is an early attempt version for solving the Sokoban problem.
    
- **Status:** **Deprecated**.

---

## 26. **algorithms/generate_matrix_flip_strategy.py**

- **Purpose:** Solves a classic matrix optimization problem (Score After Flipping Matrix). This version aims to test whether the model can learn a "strategy" rather than the final result. Corresponding LeetCode problem: [861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- **Logic:** For a given M x N binary matrix, you can change its content by flipping any row or column. The task is to find a flipping strategy that maximizes the sum after interpreting each row as a binary number. The script uses an efficient greedy algorithm to find the optimal strategy: first ensure the first column is all 1s (by flipping rows with 0 as the first bit), then traverse remaining columns, flipping the column if it has more 0s than 1s to maximize each bit's contribution.

- **I/O Format:**
    
    - Input: M*N length binary string.
        
    - Output: M+N length binary multi-label vector representing row and column flip masks.
        
- **Main Parameters:** MATRIX_M, MATRIX_N, DATASET_SIZE.

---

## 27. **algorithms/generate_matrix_flip_max_score.py**

- **Purpose:** Tests the model's ability to learn a matrix optimization problem that requires a two-step greedy strategy (row flips first, then column flips) to achieve global optimum. This version requires the model to directly output the final aggregated result (score). Corresponding LeetCode problem: [861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)
    
- **Logic:** The script implements an efficient greedy algorithm to find the maximum score. Step 1: Traverse each row, flip the row if its highest bit (leftmost) is 0. Step 2: Traverse each column, flip the column if it has more 0s than 1s. Finally, weight the resulting matrix by binary to get the maximum score.
    
- **I/O Format:**
    
    - Input: MATRIX_M * MATRIX_N length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the maximum score.
        
- **Main Parameters:** MATRIX_M, MATRIX_N, DATASET_SIZE.

---

## 28. **algorithms/generate_min_k_bit_flips.py**

- **Purpose:** Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing, and tests whether it can use part of the input (k) as a "parameter" to guide processing of another part (nums). Corresponding LeetCode problem: [995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)
    
- **Logic:** Solves the classic "K consecutive bit flips" problem (LeetCode 995). It traverses the array from left to right, using an efficient difference trick to track the cumulative effect of previous flips. If the current position's value is still '0' under cumulative effect, a new flip must be performed and the difference array updated.
    
- **I/O Format:**
    
    - Input: NUMS_LENGTH_N + K_BITS length binary string (data + parameter k).
        
    - Output: OUTPUT_BITS length binary multi-label vector (representing minimum flips, 0 means no solution).
        
- **Main Parameters:** NUMS_LENGTH_N, K_MAX_N, DATASET_SIZE.

---

## 29. **algorithms/generate_min_k_bit_flips_fixed_k.py**

- **Purpose:** Tests the model's ability to learn a greedy algorithm that depends on historical state and sequential processing. In this version, the environmental parameter (k=2) is fixed and hidden, the model must implicitly learn it from data. This is a control experiment for comparing with the variable k version. Corresponding LeetCode problem: [995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/) (fixed k version)
    
- **Logic:** Same logic as generate_min_k_bit_flips.py, but the flip window length k is **hard-coded as 2** (fixed `k=2` in code) and not provided in input. This design significantly reduces task complexity: the model doesn't need to parse dynamic k parameter, only needs to learn the deterministic algorithm under a single, fixed window size.
    
- **Experimental Design Note:** According to preliminary observations, when k is randomly variable in the range 2~K_MAX_N, simultaneously learning to parse the k parameter and adapt to algorithm behaviors with different k values significantly increases convergence difficulty. After fixing k to a single value, model training speed and stability both improve. This experiment aims to verify this hypothesis. **Note: Trainability verification of the variable k version is not yet complete and requires further experiments.**
    
- **I/O Format:**
    
    - Input: NUMS_LENGTH_N length binary string (data only, no k parameter).
        
    - Output: OUTPUT_BITS length binary multi-label vector (representing minimum flips, 0 means no solution).
        
- **Main Parameters:** NUMS_LENGTH_N, DATASET_SIZE.

---

## 30. **algorithms/generate_special_binary_string_recursion.py**

- **Purpose:** Tests the model's ability to learn a recursively defined string transformation rule. This problem corresponds to LeetCode problem: [761. Special Binary String](https://leetcode.com/problems/special-binary-string/) (Hard difficulty). Problem requirement: A special binary sequence has equal numbers of 0s and 1s, and any prefix has no fewer 1s than 0s; you can swap adjacent special substrings to get the lexicographically largest result. This script precisely implements the recursive decomposition and lexicographic sorting algorithm for this problem, as a benchmark experiment for testing neural network's ability to learn complex recursive rules.
    
- **Logic:** The property of "special binary string" is similar to valid parenthesis sequences (1 represents '(', 0 represents ')'). The core algorithm idea is that any special string can be decomposed into 1 + A + 0 + B form, where A and B are also (possibly empty) special strings. The script recursively finds all outermost special substrings, performs optimal transformation on each, then concatenates the results in descending lexicographic order to get the final answer.
    
- **I/O Format:**
    
    - Input: STRING_LENGTH_N length special binary string.
        
    - Output: STRING_LENGTH_N length binary multi-label vector representing the lexicographically largest result.
        
- **Main Parameters:** STRING_LENGTH_N, DATASET_SIZE.

---

## 31. **algorithms/generate_count_connected_components.py**

- **Purpose:** Tests the model's basic understanding of graph structure, particularly the core concept of "connectivity" (corresponding to LeetCode 323. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/), note: this is a member problem).
    
- **Logic:** The script randomly generates an N x N adjacency matrix to represent an undirected graph. Then it uses Breadth-First Search (BFS) or Depth-First Search (DFS) to traverse the graph and calculate the total number of independent connected components.
    
- **I/O Format:**
    
    - Input: N*N length binary string (adjacency matrix).
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the number of connected components.
        
- **Main Parameters:** GRAPH_SIZE_N, EDGE_PROBABILITY (controls graph sparsity), DATASET_SIZE.

---

## 32. **algorithms/generate_check_graph_connectivity.py**

- **Purpose:** This is another core test of the model's graph theory foundation, the task is to determine whether there exists a path between any two points in a graph (corresponding to LeetCode 1971. [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/)).
    
- **Logic:** The script randomly generates a graph's adjacency matrix and randomly selects two nodes as start and end points. It uses a standard graph algorithm library to determine whether these two points are in the same connected component.
    
- **I/O Format:**
    
    - Input: String in format size*size (adjacency matrix) + ; + start_node_char + end_node_char.
        
    - Output: [1] (connected) or [0] (not connected).
        
- **Main Parameters:** num_samples, size (number of nodes in graph).

---

## 33. **algorithms/generate_minimize_malware_spread.py**

- **Purpose:** Solves a graph theory-based virus spread optimization problem (LeetCode Hard "Minimize Malware Spread"). The model needs to understand graph connectivity and evaluate the impact of removing different nodes on global spread. This script provides two output formats for comparison to see which is easier to learn. Corresponding LeetCode problem: [924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)
    
- **Logic:** Input is a graph's adjacency matrix and a set of initially infected nodes. The task is to remove **only one** initially infected node to minimize the total number of nodes eventually infected by the virus. The script finds the optimal removal target set by brute-force simulation of the spread after removing each initial node and comparing results.
    
- **I/O Format:**
    
    - Input: (N*N) (adjacency matrix) + N (initial infected node mask) length binary string.
        
    - Output (dual format):
        - `output_simple`: 1-bit binary label indicating whether the first initial infected node is one of the optimal solutions (simplified task, reduces learning difficulty).
        - `output_full`: N-bit binary vector indicating whether each node belongs to the optimal removal set (complete task, N-bit multi-label classification).
        
- **Main Parameters:** GRAPH_SIZE_N, NUM_INITIAL, DATASET_SIZE.

---

## 34. **algorithms/generate_count_islands_1d.py**

- **Purpose:** Tests the model's ability to perform pattern recognition and counting on one-dimensional sequences.
    
- **Logic:** Input is a one-dimensional binary string. The task is to calculate the number of continuous '1' blocks (islands) separated by '0's. For example, in 0110100111, there are 3 islands.
    
- **I/O Format:**
    
    - Input: NUM_INPUT_BITS length binary string.
        
    - Output: NUM_OUTPUT_BITS length binary multi-label vector representing the number of islands.
        
- **Main Parameters:** NUM_INPUT_BITS, DATASET_SIZE.

---

## 35. **algorithms/generate_find_articulation_points.py**

- **Purpose:** Tests the model's ability to identify graph "articulation points" or "bridges," an important concept in graph theory. [LeetCode 1568. Minimum Number of Days to Disconnect Island](https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/)
    
- **Logic:** Input is a two-dimensional grid composed of '1's (land) and '0's (water). The task essence is to find the minimum number of '1's that need to be removed to disconnect the original single connected component (island). The script finds the solution through brute-force attempts (removing 1 point, removing 2 points). Output is designed as a heatmap of final removed points, not days.
    
- **I/O Format:**
    
    - Input: M*N length binary string.
        
    - Output: M*N length binary multi-label vector marking the removed points.
        
- **Main Parameters:** NUM_SAMPLES, GRID_M, GRID_N.

---

## 36. **algorithms/generate_nim_game_zeckendorf.py**

- **Purpose:** This experiment aims to test whether my paradigm can learn a non-intuitive game theory problem based on complex number theory (Zeckendorf representation). It moves beyond simple pattern matching and requires the model to understand deeper mathematical structures.
    
- **Logic:** I implemented a solver for a classic stone game variant (similar to Wythoff's game) whose solution is closely related to Fibonacci sequence and Zeckendorf representation. To make it easier for the model to learn, I simplified the task: the original problem might be to calculate how many winning positions exist in the interval [1, n], I modified it to only determine whether the given n itself is a winning position. This gives each input and output a more direct causal relationship.
    
- **I/O Format:**
    
    - Input: N_BITS length binary string representing total number of stones n.
        
    - Output: 1-bit binary multi-label ([1] means winning, [0] means losing).
        
- **Main Parameters:** N_BITS, DATASET_SIZE.

---

## 37. **algorithms/generate_longest_subsequence_constrained.py**

- **Purpose:** Tests the model's ability to handle a complex optimization problem mixing sequence operations and numeric constraints. Corresponding LeetCode problem: [2311. Longest Binary Subsequence Less Than or Equal to K](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/)
    
- **Logic:** Input is a binary string s and an integer k (also represented in binary). The task is to find a subsequence of s (can be non-continuous) whose value as a binary number is less than or equal to k, with the longest possible length. Output is the length of this longest subsequence.
    
- **I/O Format:**
    
    - Input: STRING_LENGTH_N + K_BITS length binary string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing length.
        
- **Main Parameters:** STRING_LENGTH_N, K_BITS, DATASET_SIZE.

---

## 38. **algorithms/generate_treasure_hunt_tsp.py**

- **Purpose:** Solves a complex state space search problem that combines graph traversal (BFS) and combinatorial optimization (state compression DP), a classic difficult problem in algorithm competitions.
    
- **Logic:** In a given maze, the player needs to start from point 'S', trigger all mechanisms 'M', and finally reach endpoint 'T'. Stones 'O' can be used along the way to instantly trigger any mechanism. The script uses a series of BFS to calculate shortest distances between all key points (S, T, M, O), then uses state compression dynamic programming to find the shortest total path length to traverse all mechanisms and reach the endpoint. - **LeetCode Problem:** [LCP 13. Treasure Hunt](https://leetcode.cn/problems/xun-bao/) - This is exactly the core problem implemented by this script
    
- **I/O Format:**
    
    - Input: N*M length string representing maze layout.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing the shortest path length (-1 mapped to 0).
        
- **Main Parameters:** MAZE_N, MAZE_M, DATASET_SIZE.

---

## 39. **algorithms/generate_freedom_trail_dp.py**

- **Purpose:** [LeetCode 514. Freedom Trail](https://leetcode.com/problems/freedom-trail/description/) Tests the model's ability to learn to solve a complex optimization problem requiring dynamic programming and path backtracking.
    
- **Logic:** Input is a ring string representing disk characters and a target key string. The script uses dynamic programming to calculate the minimum rotation steps needed to spell out the key. A key modification is that the script not only calculates the total steps but also backtracks the DP table to reconstruct the optimal operation (clockwise or counterclockwise, and specific steps) for each step, using this operation sequence as output.
    
- **I/O Format:**
    
    - Input: ring|key format string.
        
    - Output: KEY_LENGTH * move_bits length binary multi-label vector encoding the operation at each step.
        
- **Main Parameters:** RING_LENGTH, KEY_LENGTH, NUM_SAMPLES.

---

## 40. **algorithms/generate_sum_of_subset_with_mask.py**

- **Purpose:** Tests the model's ability to select elements from a set based on a binary mask and perform aggregation operations (summing). This is an innovative control experiment aiming to verify that **task structure itself** (rather than model capacity) is the key factor affecting neural network learnability.
    
- **Logic:** This script provides **two problem formats** for comparative research:
    
    1. **Inverse Problem (Combinatorial Optimization):** Input is number set + target value, output is mask (which subset's sum equals target). This is an NP-hard problem requiring search, decision-making, and combinatorial reasoning, with extremely high learning difficulty, used to test the model's ability to learn combinatorial optimization.
    
    2. **Forward Problem (Pure Calculation):** Input is number set + subset mask (already tells you which subset to select), output is sum (the sum of these numbers). This only requires addition operations, with extremely low learning difficulty, used to verify the impact of task structure on learnability.
    
    The script switches between the two modes through a `MODE` configuration parameter. Actual training uses **forward mode**, because inverse mode (original subset sum problem) is almost unlearnable for neural networks, powerfully proving that **task structure itself** is the key factor determining learnability, not insufficient model capacity.
    
- **I/O Format:**
    
    - Inverse mode: Input is numbers(4-bit) + target(6-bit), output is mask(n_items bits)
    - Forward mode: Input is numbers(4-bit) + mask(n_items bits), output is sum(6-bit)
    
- **Main Parameters:** n_items, value_range (number range), num_samples, MODE ("forward" or "reverse").

---

## 41. **algorithms/generate_sudoku_6x6.py**

- **Purpose:** Tests the model's ability to handle strongly constrained satisfaction problems (CSP) - Sudoku.
    
- **Logic:** The script implements a backtracking solver to generate complete 6x6 Sudoku solutions and creates puzzles with unique solutions through "digging holes."
    
- **I/O Format:**
    
    - Input: 36-character string with _ representing empty cells.
        
    - Output: 36 * 3 = 108-bit binary multi-label vector, each digit represented with 3-bit binary.
        
- **Main Parameters:** num_puzzles, difficulty (hole digging ratio).

---

## 42. **algorithms/generate_valid_parentheses_path_random_deprecated.py**

- **Purpose:** (Early exploration/Deprecated) This is an early attempt to solve the "valid parentheses path" problem.
    
- **Status:** **Deprecated**. This script creates datasets by randomly generating parentheses grids, but this leads to severe data imbalance problems (the vast majority of random grids have no valid path), which is not conducive to model training. Has been replaced by algorithms/generate_valid_parentheses_path_balanced.py.
    
- **Logic:** Randomly generates M x N parentheses grids and calls a solver to determine whether a valid path exists.
    
- **Main Parameters:** MAZE_M, MAZE_N, DATASET_SIZE.

---

## 43. **algorithms/generate_valid_parentheses_path_balanced.py**

- **Purpose:** Solves a pathfinding problem on a two-dimensional grid, but path validity is constrained by stack structure (parentheses matching). This is a complex task combining algorithms and logical constraints (LeetCode Hard [#2267](https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/) "Check if There Is a Valid Parentheses String Path").
    
- **Logic:** The script generates data in two ways to ensure balance:
    
    1. **Positive Samples:** First determines a path on the grid, then generates a valid parentheses sequence along this path, and randomly fills cells outside the path.
    
    2. **Negative Samples:** Randomly generates grids and verifies through solver that they indeed have no valid path.
        The model's task is to determine whether there exists a path from (0,0) to (M-1,N-1) in the given parentheses grid such that the parentheses sequence on the path is valid.
        
- **I/O Format:**
    
    - Input: M*N length binary string ('('->0, ')'->1).
        
    - Output: [1] (exists) or [0] (doesn't exist).
        
- **Main Parameters:** MAZE_M, MAZE_N, DATASET_SIZE.

---

## 44. **algorithms/generate_point_in_polygon.py**

- **Purpose:** Tests the model's ability to learn a classic algorithm in computational geometry - the Ray Casting Algorithm.
    
- **Logic:** The script first randomly generates N vertices of a non-self-intersecting polygon, then randomly generates a test point. Input is a string concatenated from binary encodings of all vertices and the test point's coordinates. Output is a bit indicating whether the test point is inside the polygon. To ensure dataset balance, the script strives to make the number of interior and exterior samples roughly equal.
    
- **I/O Format:**
    
    - Input: (NUM_VERTICES_N + 1) * 2 * BITS_PER_COORD length binary string.
        
    - Output: [1] (inside) or [0] (outside).
        
- **Main Parameters:** NUM_VERTICES_N, BITS_PER_COORD, DATASET_SIZE.

---

## 45. **algorithms/generate_shortest_path_in_matrix_bfs.py**

- **Purpose:** Tests the model's ability to find the shortest path in a two-dimensional grid based on the classic Breadth-First Search (BFS) algorithm.
    
- **Logic:** Input is an N x N binary matrix where '0' represents passable paths and '1' represents walls. The task is to calculate the shortest path length from top-left corner (0,0) to bottom-right corner (N-1, N-1) (allowing 8-direction movement). The script uses BFS algorithm to find the optimal solution. Path length is 0 if the two points are not connected.
    
- **I/O Format:**
    
    - Input: NN-bit binary string or (NN)/4-bit hexadecimal string.
        
    - Output: OUTPUT_BITS length binary multi-label vector representing path length.
        
- **Main Parameters:** MATRIX_SIZE_N, INPUT_FORMAT, DATASET_SIZE.

---

## 46. **algorithms/generate_sudoku_4x4_stepwise_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's ability to perform "stepwise" reasoning, i.e., only predicting the next optimal action at each state, rather than outputting the complete solution at once.
    
- **Status:** **Deprecated**. This script attempts to generate stepwise solutions for 4x4 Sudoku through a complex backward logic, but its core algorithm is unreliable and cannot guarantee the correctness and validity of generated data. Has been replaced by generate_sudoku_6x6.py and other more complete scripts.
    
- **Logic:** (Problematic) Attempts to backward derive the optimal solution for each step from a complete 4x4 Sudoku solution by "digging holes" and checking uniqueness.

---

## 47. **algorithms/generate_tiling_problem_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's ability to solve a classic tiling coverage optimization problem, which is an NP-hard problem.
    
- **Status:** **Deprecated**. The core solver uses backtracking search with pruning, but this is an exponential complexity algorithm. For matrices larger than about 13x13, its computation time becomes impractical, making it impossible to efficiently generate large-scale datasets.
    
- **Logic:** Uses backtracking search to solve the problem of "covering an m*n rectangle with the minimum number of squares."

---

## 48. **algorithms/generate_hanoi_tower_twisted_path_deprecated.py**

- **Purpose:** (Deprecated) This script intended to generate a "twisted path" dataset for the Tower of Hanoi problem, i.e., the optimal path from a non-standard but difficult start state to the standard goal.
    
- **Status:** **Deprecated**. The core move logic (apply_move) has errors and fails to correctly simulate the Tower of Hanoi rule of "larger disks below, smaller disks above," resulting in generated paths that are not valid solutions. Has been replaced by generate_hanoi_tower_build_full_state_graph.py and other more complete scripts.

---

## 49. **algorithms/generate_checkers_jump_1d_v2.py**

- **Purpose:** This is a **sequence learning comparative experiment** script for the checkers exchange problem. Unlike V1 version focusing on single-step optimal strategy, this script aims to generate various types of complete path datasets (including optimal paths, subpaths, non-optimal paths, etc.) for systematically studying the model's ability to learn long sequence planning under different data distributions. This task also originates from the checkers exchange problem in Apple's paper [15], but focuses on exploring the impact of sequence-level generalization and path diversity.
    
- **Logic:** The script simulates the process of two colors of checkers ('R' and 'B') crossing each other to reach each other's initial positions on a one-dimensional board. It uses efficient reverse Breadth-First Search (BFS), starting from the goal state and traversing the entire state space in reverse, thereby calculating the unique optimal next step for every reachable state.
    
- **I/O Format:**
    
    - Input: 2*N+1 length integer sequence representing board state.
        
    - Output: Single integer representing the **position index** of the checker to move, which is a classification problem.
        
- **Main Parameters:** CHECKERS_N (number of checkers of each color).

---

## 50. **algorithms/generate_maze_symbolic_to_image.py**

- **Purpose:** Converts symbolic maze path planning datasets to image format to test the ability of visual models (like CNN, ViT) to perform path planning directly from pixels.
    
- **Logic:** This script reads a .jsonl file containing maze layout strings and corresponding optimal actions. For each line, it renders the maze layout (containing walls, paths, start and end points) into a high-contrast color image. Finally, it generates an image folder and a labels.csv file, associating image filenames with optimal action labels (0-3 classification).
    
- **I/O Format:**
    
    - Input: .jsonl file.
        
    - Output: JPG images in images/ directory and labels.csv metadata file.
        
- **Main Parameters:** INPUT_JSONL_FILE, OUTPUT_IMAGE_DIR, IMAGE_SIZE, GRID_DIM.

---

## 51. **algorithms/generate_trapping_rain_water_visualizer.py**

- **Purpose:** This is a **data conversion and visualization** script. Its role is to convert the already generated, symbolic "trapping rain water" dataset into an image-to-image format dataset, so that visual models can solve the same problem.
    
- **Logic:** The script reads the .jsonl file line by line, parsing out the column heights and corresponding trapped water amounts for each sample. It first renders the column heights into a black and white image as input. Then it renders the corresponding water amount in blue above the columns to generate the output image.
    
- **I/O Format:** Input: .jsonl file -> Output: PNG image pairs in images/ directory.
    
- **Main Parameters:** input_file, output_dir, image_size.

---

## 52. **algorithms/generate_shortest_path_in_tree_deprecated.py**

- **Purpose:** (Early exploration/Deprecated) This is an early experiment aiming to test the model's ability to find the shortest path on a graph from images.
    
- **Status:** **Deprecated**. This script ensures graph planarity by generating random trees, but this accidentally simplifies the problem: the path between any two points in a tree is unique, the model doesn't need to learn the concept of "shortest." This task was later replaced by more challenging dense maze path planning tasks.
    
- **Logic:** The script generates a random tree graph and draws it on an image. The input image highlights a start point and an end point. The output image highlights the unique path connecting the start and end points on top of the input.
    
- **Main Parameters:** MIN_NODES, MAX_NODES.

---

## 53. **generate_rain_water_final_showdown.py**

- **Purpose:** This is an **algorithm comparison** experiment designed to study which algorithmic logic better fits the inductive bias of neural networks by providing intermediate processes of three different algorithms (dynamic programming, monotonic stack, two pointers) as auxiliary labels.
    
- **Logic:** The script generates random column heights. Besides calculating the final trapped water amount, it simultaneously generates execution traces for three algorithms:
    1.  **DP:** Records left_max and right_max arrays.
    2.  **Stack:** Records triplets (left_idx, right_idx, top_height) during monotonic stack operations.
    3.  **Two Pointers:** Records max value updates during two pointers movement.
    
- **I/O Format:**
    
    - Input: N * BITS_PER_HEIGHT bit binary string.
        
    - Output: JSON object containing `final_answer` and `explain_dp`, `explain_stack`, `explain_tp`.
        
- **Main Parameters:** NUM_COLUMNS_N, BITS_PER_HEIGHT.

---

# C: Visual Reasoning

## 1. **visual_reasoning/generate_checkerboard_to_binary.py**

- **Purpose:** This is a basic vision-to-symbol conversion task for testing the model's ability to decode structured information from raw pixel data.
    
- **Logic:** The script generates a random N x N binary grid for each sample and renders it into an IMAGE_SIZE x IMAGE_SIZE black and white checkerboard image. Input is this image, output is the corresponding N*N bit flattened binary string behind it.
    
- **I/O Format:**
    
    - Input: IMAGE_SIZE x IMAGE_SIZE grayscale image.
        
    - Output: GRID_DIM * GRID_DIM length binary multi-label vector.
        
- **Main Parameters:** NUM_SAMPLES, IMAGE_SIZE, GRID_DIM.

---

## 2. **visual_reasoning/generate_line_angle_to_vector.py**

- **Purpose:** Tests the model's ability to extract precise geometric information (angles) from images, a more advanced visual reasoning task than simple checkerboard recognition.
    
- **Logic:** The script generates an image similar to radar scan or clock face. It draws several line segments with random colors, widths, and angles from the image center point. The entire 360 degrees is divided into num_angle_bins fan-shaped intervals. The model's task is to output a multi-hot encoded vector marking which angle intervals contain line segments.
    
- **I/O Format:**
    
    - Input: image_size x image_size RGB image.
        
    - Output: num_angle_bins length binary multi-label vector.
        
- **Main Parameters:** image_size, num_angle_bins, min_lines, max_lines.

---

## 3. **visual_reasoning/generate_count_shapes_from_image.py**

- **Purpose:** Tests the model's ability to simultaneously perform multiple visual tasks: object recognition (shape), attribute recognition (color), and counting (aggregation).
    
- **Logic:** The script randomly places objects of different shapes (square, circle, triangle) and colors (red, green, blue) on a white canvas, ensuring they don't overlap. The model's task is to output a 12-bit vector encoding the total count of each shape and each color. (Note: Due to random color assignment, there may be slight imbalance in color counts)
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image.
        
    - Output: 12-bit binary multi-label vector encoding counts for 6 categories.
        
- **Main Parameters:** TOTAL_SAMPLES, MAX_COUNT_PER_CATEGORY (only effective for shapes).

---

## 4. **visual_reasoning/generate_sokoban_symbolic_to_image_no_labels.py**

- **Purpose:** This is a data conversion script for converting symbolic Sokoban datasets (.jsonl format) to image format only, used for pure vision tasks or as an intermediate step for more complex data processing.
    
- **Logic:** The script reads a .jsonl file containing Sokoban layout strings line by line. For each line, it renders the layout string into a colored, visually styled 224x224 image and saves it. This version **does not generate** corresponding label files.
    
- **I/O Format:**
    
    - Input: sokoban_optimized_dataset.jsonl file.
        
    - Output: PNG images in images/ directory.
        
- **Main Parameters:** INPUT_JSONL_PATH, OUTPUT_DIR, GRID_SIZE, CELL_PIXELS.

---

## 5. **visual_reasoning/generate_sokoban_symbolic_to_image_with_labels.py**

- **Purpose:** This is a data conversion script for converting symbolic Sokoban datasets (.jsonl format) into a complete image classification dataset for training computer vision models (like ViT, Swin Transformer).
    
- **Logic:** The script reads a .jsonl file containing Sokoban layout strings and corresponding optimal actions line by line. For each line, it renders the layout string into a colored, visually styled 224x224 image, and writes the image's filename along with the original optimal action label into a labels.csv metadata file.
    
- **I/O Format:**
    
    - Input: sokoban_optimized_dataset.jsonl file.
        
    - Output: PNG images in images/ directory and labels.csv file.
        
- **Main Parameters:** INPUT_JSONL_PATH, OUTPUT_DIR, GRID_SIZE, CELL_PIXELS.

---

## 6. **visual_reasoning/generate_triangle_to_incircle.py**

- **Purpose:** This is a landmark experiment demonstrating "carving precise rules with gradient descent." It tests whether the model can learn a pure, nontrivial geometric construction rule (triangle incircle).
    
- **Logic:** The script generates a random green triangle as input image for each sample. Then it precisely calculates the unique incircle of that triangle and draws this red incircle on the original triangle as the output image.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing a green triangle).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (triangle + red incircle).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_TRIANGLE_AREA.

---

## 7. **visual_reasoning/generate_polygon_to_symmetry_axis.py**

- **Purpose:** Tests the model's ability to reverse infer the hidden symmetry axis from a complete symmetric figure.
    
- **Logic:** The script first defines a random symmetry axis. Then it randomly generates a set of vertices on one side of the axis and mirrors these vertices to the other side, thereby forming a perfect axisymmetric polygon. The input image only contains this polygon, the output image additionally draws the hidden symmetry axis on this basis.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing a symmetric figure).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (symmetric figure + symmetry axis).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_POLYGON_VERTICES_HALF.

---

## 8. **visual_reasoning/generate_triangle_to_centroid.py**

- **Purpose:** Tests the model's ability to learn another basic geometric concept - centroid.
    
- **Logic:** The script generates a random green triangle as input image. Then it calculates the triangle's centroid (center of mass) and draws a fixed-size red dot at that position as the output image.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing a green triangle).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (triangle + red centroid dot).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_TRIANGLE_AREA.

---

## 9. **visual_reasoning/generate_triangle_to_tessellation.py**

- **Purpose:** This is a landmark demonstration of our paradigm's capabilities. It tests whether the model can learn an infinite, lattice-based generation rule. Due to the global correlation and precise details of tessellation patterns, it strongly rules out the possibility that the model solves problems merely through "interpolation" or "memorization."
    
- **Logic:** The input image only contains a randomly generated, randomly placed green triangle. The script uses this triangle as the basis for a "unit cell," and by translating it in two non-collinear basis vector directions, it tiles the entire canvas with alternating green and red triangles, forming a perfect planar tessellation pattern. The output image is this complete tessellation pattern.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing a green triangle).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (complete tessellation pattern).
        
- **Main Parameters:** NUM_SAMPLES, IMG_SIZE.

---

## 10. **visual_reasoning/generate_shortest_distance_between_triangles.py**

- **Purpose:** Tests the model's ability to perform global geometric relationship (shortest distance) reasoning when containing multiple objects.
    
- **Logic:** The script randomly generates two non-overlapping green triangles on a canvas. Then it uses the professional computational geometry library shapely to precisely calculate the shortest distance line segment between these two triangles. The input image only contains the two triangles, the output image additionally draws this red shortest connecting line.
    
- **I/O Format:**
    
    - Input: image_size x image_size RGB image (containing two green triangles).
        
    - Output: image_size x image_size RGB image (triangles + red shortest connecting line).
        
- **Status:** Logic is correct, but depends on external library shapely, may have environment configuration issues.

---

## 11. **visual_reasoning/generate_coords_to_triangle.py**

- **Purpose:** This is a basic symbol-to-geometry rendering task, testing the model's ability to convert abstract coordinate information into concrete pixel shapes.
    
- **Logic:** The script's input is a 48-bit binary string encoding the (x, y) coordinates of a triangle's three vertices (8 bits per coordinate). The output is an image of a green solid triangle drawn based on these coordinates.
    
- **I/O Format:**
    
    - Input: 48-bit binary string.
        
    - Output: 256x256 RGB image.
        
- **Main Parameters:** NUM_SAMPLES, IMAGE_SIZE.

---

## 12. **visual_reasoning/generate_triangle_coords_to_tessellation.py**

- **Purpose:** This is an advanced reasoning task mixing symbolic instructions and geometric generation rules.
    
- **Logic:** Like generate_coords_to_triangle.py, the input is a 48-bit binary string defining a base triangle. Like generate_triangle_to_tessellation.py, the output is a perfect planar tessellation pattern based on this base triangle. **The key modification** is that in the output tessellation pattern, the base triangle directly defined by the input is colored with a special color (like black), while other triangles remain green and red. This provides necessary "grounding" information for the model.
    
- **I/O Format:**
    
    - Input: 48-bit binary string.
        
    - Output: 256x256 RGB image (tessellation pattern).
        
- **Main Parameters:** NUM_SAMPLES, IMG_SIZE.

---

## 13. **generate_trapping_rain_water_image_to_symbol.py**

- **Purpose:** This is an **image-to-symbol** conversion task, used to test whether visual models like CNNs can extract precise physical quantities from images.
    
- **Logic:** The script generates random column heights and renders them into a black and white image as input. Simultaneously, it calculates the amount of water trapped by each column and encodes it as a binary string for the label. This allows us to train a model to calculate rainwater amount by looking at the image.
    
- **I/O Format:**
    
    - Input: 240x240 grayscale image (column height map).
        
    - Output: Binary string label (water amount for each column).
        
- **Main Parameters:** NUM_COLUMNS_N, BITS_PER_HEIGHT, IMAGE_SIZE.

---

# D: Cellular Automata

## 1. **cellular_automata/generate_cellular_automata_1d.py**

- **Purpose:** Used to generate one-dimensional cellular automaton (CA) evolution datasets to test the model's ability to learn and execute local, deterministic rules.
    
- **Logic:** Given a random binary initial state, the script iterates a specified number of layers (steps) according to the specified evolution rule (currently Rule 110) and generates the final state.
    
- **I/O Format:**
    
    - Input: length-bit binary string representing initial state.
        
    - Output: length-bit binary multi-label vector representing final state.
        
- **Main Parameters:** num_samples, length, l (evolution layers).

---

## 2. **cellular_automata/generate_game_of_life_2d.py**

- **Purpose:** Generates datasets for two-dimensional cellular automaton - Conway's Game of Life. This task is more complex than 1D CA, requiring the model to understand neighborhood relationships in two-dimensional space.
    
- **Logic:** Based on a random n*n initial board, evolves d time steps according to the standard Game of Life rules (B3/S23), recording the final board state.
    
- **I/O Format:**
    
    - Input: n*n-bit flattened binary string representing initial board.
        
    - Output: n*n-bit binary multi-label vector representing final board.
        
- **Main Parameters:** num_samples, n (grid side length), d (evolution steps).

---

## 3. **cellular_automata/generate_cellular_automata_1d_multistate.py**

- **Purpose:** As an extension of one-dimensional cellular automaton experiments, tests the model's ability to handle non-binary state spaces.
    
- **Logic:** The CCA evolution rule is: a cell's next state is its own current state +1 (mod n_states), if and only if its left neighbor or right neighbor's state equals this target state. The script generates input (initial state) and output (final state) of this evolution process.
    
- **I/O Format:**
    
    - Input: n_cells * 2 (since n_states=4) length binary string.
        
    - Output: n_cells * 2 length binary multi-label vector.
        
- **Main Parameters:** n_cells, n_states, n_samples, steps.

---

## 4. **cellular_automata/generate_cellular_automata_programmable.py**

- **Purpose:** Tests the model's "programmability" or "meta-learning" ability. The model must not only learn CA evolution but also be able to perform evolution according to different rules given in each input.
    
- **Logic:** Each sample takes an 8-bit rule number and an initial state as input together. The script evolves the state according to that rule to generate output. This requires the model to understand part of the input as "program" and another part as "data."
    
- **I/O Format:**
    
    - Input: 8 (rule) + CA_WIDTH (state) bit binary string.
        
    - Output: CA_WIDTH bit multi-label binary vector.
        
- **Main Parameters:** TARGET_NUM_SAMPLES, CA_WIDTH, EVOLUTION_STEPS, RULES_TO_INCLUDE (specifies which rules to include in dataset).

---

## 5. **cellular_automata/generate_cellular_automata_inverse_rule90.py**

- **Purpose:** Tests the model's ability to solve "inverse problems." Given the output of a deterministic system, the model needs to reverse infer possible inputs that satisfy specific constraints (sparsest and unique).
    
- **Logic:** Input is the state after one-step evolution of one-dimensional cellular automaton Rule 90. The task is to find the one with the fewest '1's (sparsest) among all possible "previous step" states. To make the problem have a unique solution, the script only keeps samples where the "sparsest solution" is exactly one through brute-force search.
    
- **I/O Format:**
    
    - Input: N-bit binary string (evolved state).
        
    - Output: N-bit binary string (pre-evolution state).
        
- **Main Parameters:** num_samples, length.

---

## 6. **cellular_automata/generate_game_of_life_image_to_image.py**

- **Purpose:** This is the image-to-image version of two-dimensional cellular automaton, testing whether the model can directly perform local rule-based evolution in pixel space.
    
- **Logic:** The script generates a random GRID_SIZE x GRID_SIZE initial state and renders it into a black and white image as input. Then it calculates the next state according to Game of Life rules and renders it into another image as output.
    
- **I/O Format:**
    
    - Input: IMAGE_SIZE x IMAGE_SIZE grayscale image (initial state).
        
    - Output: IMAGE_SIZE x IMAGE_SIZE grayscale image (state after one evolution step).
        
- **Main Parameters:** GRID_SIZE, IMAGE_SIZE, NUM_SAMPLES.

---

## 7. **cellular_automata/generate_cellular_automata_spatial_conditional.py**

- **Purpose:** Tests the model's ability to partition and parse "instructions" and "data" within a single modality (image), a "pseudo-multimodal" or "spatial conditioning" experiment.
    
- **Logic:** The script encodes a 36-bit cellular automaton problem in an image. The narrow strip area at the top of the image encodes an 8-bit evolution rule with specific colors (red/green), the large 6x6 area below encodes the 36-bit initial state with black and white blocks. The output image is the final state after 3 evolution steps under that rule. The model must learn to "read" the rule at the top and apply it to the state below.
    
- **I/O Format:**
    
    - Input: IMG_WIDTH x IMG_HEIGHT RGB image (rule encoding at top, initial state at bottom).
        
    - Output: IMG_WIDTH x IMG_HEIGHT RGB image (final state).
        
- **Main Parameters:** NUM_INITIAL_STATES, ITERATIONS, GRID_DIM.

---

## 8. **cellular_automata/generate_cellular_automata_multimodal_deprecated.py**

- **Purpose:** (Deprecated) Generates a true multimodal dataset for training models that can simultaneously understand image input and text instructions.
    
- **Status:** **Deprecated**. Due to lack of suitable, easily trainable multimodal models (within the experimental framework), and generate_cellular_automata_spatial_conditional.py providing a more concise alternative, the training set generated by this script was not used.
    
- **Logic:** The script generates an image representing the cellular automaton initial state and a text string representing the evolution rule for each sample. Output is the evolved state image.
    
- **Main Parameters:** NUM_SAMPLES, GRID_DIM, ITERATIONS.

---

## 9. **generate_cellular_automata_1d_to_grid_image_interp.py**

- **Purpose:** This script aims to design a "logic/perception hybrid" task to prove that neural network's rule learning ability and interpolation ability are not mutually exclusive, but can be integrated and demonstrated in a single task. It forces the model to simultaneously "see through" the continuous grayscale values of the input to perform discrete logical reasoning, and remember these grayscale values to complete the final continuous value mapping.
    
- **Logic:** The script first generates a 36-bit logical initial state of cellular automaton. When generating the input image, cells representing logical "0" are assigned a random dark grayscale value (like 0-63), cells representing logical "1" are assigned a random light grayscale value (like 192-255). Then the script calculates the final logical output state according to cellular automaton rules. When generating the output image, it follows a hybrid rule: if a cell's final logical state is "1", its grayscale value remains the same as the corresponding cell in the input image; if the final logical state is "0", its grayscale value becomes the inverse of the input grayscale value (255 - input value).
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (a 6x6 checkerboard where each cell's color is random dark or light gray).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (6x6 checkerboard transformed according to logical rules and input colors).
        
- **Main Parameters:** NUM_SAMPLES, IMG_SIZE, RULE_NUMBER, ITERATIONS, ENABLE_INTERPOLATION_MODE.

---

## 10. **cellular_automata/generate_cellular_automata_1d_to_grid_image.py**

- **Purpose:** Tests whether the model can directly "render" one-dimensional symbolic computation results into structured two-dimensional images.
    
- **Logic:** Input is a 36-bit binary string representing the initial state of one-dimensional cellular automaton. The script first evolves it 3 steps internally according to Rule 110 to get a 36-bit final state. Then it renders this one-dimensional final state into a 6x6 black and white checkerboard image as output.
    
- **I/O Format:**
    
    - Input: 36-bit binary string.
        
    - Output: 240x240 RGB image (black and white checkerboard).
        
- **Main Parameters:** CA_WIDTH, RULE_NUMBER, ITERATIONS, GRID_DIM.

---

## 11. **cellular_automata/generate_cellular_automata_inverse_rule.py**

- **Purpose:** This experiment is the first attempt to test the model's **inverse reasoning** ability. My question is: if the model can forward derive results from rules, can it reverse infer the underlying rule from "input-output" pairs?
    
- **Logic:** I first randomly select a cellular automaton rule (from 0 to 255). Then, **construct an initial state containing all 8 types of 3-bit neighborhood patterns (using De Bruijn sequence)**, apply this rule to evolve a fixed number of steps to get the final state. This special initial state design ensures the rule can be uniquely inferred. I concatenate the initial state and final state as the model's input, and use the hidden 8-bit rule as the model's prediction target.
    
- **I/O Format:**
    
    - Input: CA_WIDTH * 2 length binary string (initial_state + final_state).
        
    - Output: 8-bit binary multi-label vector (representing the predicted rule).
        
- **Main Parameters:** CA_WIDTH, NUM_SAMPLES, ITERATION_LAYERS.

---

## 12. **cellular_automata/generate_cellular_automata_inverse_rule_and_steps.py**

- **Purpose:** This is an early version before implementing the "unique solution" version, also aiming to have the model learn to predict rules and iteration steps.
    
- **Logic:** Like the final _unique version, this script also randomly selects rules and iteration steps for each sample to generate data. However, it lacks the crucial step of verifying solution uniqueness. This means the dataset may contain some ambiguous samples where multiple (rule, step) combinations could produce the same input-output pair.
    
- **I/O Format:**
    
    - Input: CA_WIDTH * 2 length binary string.
        
    - Output: 8 + ITERATION_BITS length binary multi-label vector.
        
- **Main Parameters:** CA_WIDTH, NUM_SAMPLES, MAX_ITERATION_LAYERS.

---

## 13. **cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py**

- **Purpose:** This is a major upgrade to the inverse reasoning task. I not only require the model to infer **what** rule was applied, but also **how many times** it was applied.
    
- **Logic:** This script inherits the idea from the previous experiment but adds complexity. When generating each sample, I randomly select a rule **and a random number of iteration steps**. After obtaining the input/output pair, I introduce a crucial **uniqueness verification step**: I brute-force check whether any other rule/iteration step combination could also produce the exact same final state from the same initial state. I only keep samples where the solution is unique, thus providing an unambiguous learning objective for the model.
    
- **I/O Format:**
    
    - Input: CA_WIDTH * 2 length binary string.
        
    - Output: 8 + ITERATION_BITS length binary multi-label vector concatenating rule and iteration steps.
        
- **Main Parameters:** CA_WIDTH, NUM_SAMPLES, MAX_ITERATION_LAYERS.

---

## 14. **generate_cellular_automata_1d_perturbed.py**

- **Purpose:** This script aims to systematically test the robustness of the "neural sculpting" paradigm when facing imperfect data. By introducing controllable random perturbations to input (simulating observation noise) and output (simulating label noise), it explores the continuous spectrum of model performance transition from ideal rule world to noisy real world.
    
- **Logic:** The script first generates an original cellular automaton initial state. Then it randomly flips this original state according to INPUT_PERTURBATION_RATE to get the final "input" sequence. Next, based on the unperturbed original state, it evolves EVOLUTION_LAYERS steps according to precise cellular automaton rules to get a "correct" output sequence. Finally, it randomly flips this correct output sequence according to OUTPUT_PERTURBATION_RATE to get the final "output" label.
    
- **I/O Format:**
    
    - Input: 0/1 string of length LENGTH.
        
    - Output: 0/1 integer list of length LENGTH.
        
- **Main Parameters:** NUM_SAMPLES, LENGTH, EVOLUTION_LAYERS, INPUT_PERTURBATION_RATE, OUTPUT_PERTURBATION_RATE.

---

## 15. **generate_ca110_full_trace.py**

- **Purpose:** This is a **deep analysis** experiment designed to generate the complete trajectory of cellular automaton evolution, used to study the relationship between the neural network's internal representation and evolution steps (i.e., "Neural Mind Scanner" experiment).
    
- **Logic:** The script simulates the evolution process of Rule 110, but not only records the final state. It concatenates all intermediate states from step 1 to step N as the output label. This allows us to train a model to predict the entire evolutionary history at once.
    
- **I/O Format:**
    
    - Input: N-bit binary string (initial state).
        
    - Output: TOTAL_LAYERS * N-bit binary multi-label vector (complete trajectory).
        
- **Main Parameters:** NUM_BITS, TOTAL_LAYERS.

---

## 16. **generate_ca_text_format_dataset.py**

- **Purpose:** This is an **autoregressive** experiment designed to test whether standard Decoder-only Transformers (like GPT) can learn cellular automaton rules through text-based Prompts.
    
- **Logic:** The script generates the initial and final states of a cellular automaton and formats them into a text string like "Evolve this: [Input] -> [Output]". This turns the task into a standard text generation task.
    
- **I/O Format:**
    
    - Output: JSON object containing "text" field.
        
- **Main Parameters:** NUM_BITS, TOTAL_LAYERS.

---

## 17. **generate_cellular_automata_image_and_label.py**

- **Purpose:** This is a **universal dataset generator** used to generate both image format (Img2Img) and symbolic format (Img2Label) data for cellular automaton tasks simultaneously, supporting multi-process acceleration.
    
- **Logic:** The script generates random initial states in parallel and renders them as images. Then it performs evolution, renders the final state as an image, and retains its symbolic form. Finally, it generates metadata containing input images, output images, and output labels.
    
- **I/O Format:**
    
    - Output: Image file pairs (initial/final) and metadata.csv.
        
- **Main Parameters:** CA_WIDTH, RULE_NUMBER, ITERATIONS, NUM_SAMPLES.

---

## 18. **generate_mnist_ca_110.py**

- **Purpose:** This is a **perception and reasoning fusion** experiment. It uses MNIST handwritten digits as "cells" of the cellular automaton to test whether the model can perform logical evolution while recognizing digits.
    
- **Logic:** The script generates a cellular automaton state, but instead of using black and white pixels to represent 0 and 1, it randomly selects images of digits '0' and '1' from the MNIST dataset and pastes them at corresponding positions. The model must first recognize these digits, understand the logical state they represent, and then perform evolution.
    
- **I/O Format:**
    
    - Input: Image collaged from MNIST digits.
        
    - Output: Evolved black and white state image.
        
- **Main Parameters:** RULE, STEP_EVOL.

---

## 19. **generate_rulemnist_ca.py**

- **Purpose:** This is a more advanced **perception and reasoning fusion** experiment that introduces **rule control**.
    
- **Logic:** The input image contains a large MNIST digit (0 or 1) as background, and the cellular automaton state overlaid on top. The MNIST digit determines the evolution rule (e.g., 0 represents Rule 30, 1 represents Rule 110). The model must recognize the background digit to select the rule and recognize the foreground state to perform evolution.
    
- **I/O Format:**
    
    - Input: MNIST image with overlaid CA state.
        
    - Output: Evolved state image.
        
- **Main Parameters:** RULE_MAP, RULES.

---

# E: Physics Simulation

## 1. **physics_simulation/generate_projectile_motion_simulation.py**

- **Purpose:** Tests the model's ability to learn a simple dynamic physical process. This requires the model to infer the entire spatiotemporal trajectory from initial conditions (position and velocity vectors).
    
- **Logic:** The input image encodes the ball's initial position and velocity vector through a starting point and a directed line segment (line direction represents velocity direction, color represents velocity magnitude). The script's internal physics engine simulates the ball's parabolic bouncing trajectory in a gravitational field based on these initial conditions. The output image draws the complete trajectory.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing initial state).
        
    - Output: IMG_SIZE x IMG_SIZE RGB image (containing complete trajectory).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE, GRAVITY, ELASTICITY_FACTOR.

---

## 2. **physics_simulation/generate_snell_refraction_simulation.py**

- **Purpose:** Tests the model's ability to learn basic physical laws (Snell's law of refraction).
    
- **Logic:** The input image contains two media of different colors and an incident light ray shooting toward their interface. The script precisely calculates the refracted light ray's path according to Snell's law (n1sin(θ1) = n2sin(θ2)). The model's task is to predict the correct refracted light ray based on the input image.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing two media and incident light).
        
    - Output: Same as input but with additional red refracted light ray drawn.
        
- **Main Parameters:** IMG_SIZE, NUM_SAMPLES_TRAIN.

---

## 3. **physics_simulation/generate_snell_refraction_with_contextual_index.py**

- **Purpose:** Tests the model's ability to learn basic physical laws (Snell's law of refraction), and requires the model to infer physical parameters (refractive index) from contextual information (background color) in the image.
    
- **Logic:** The input image contains two media of different colors and an incident light ray shooting toward their interface. The script precisely calculates the refracted light ray's path according to Snell's law. In this version, the color of one medium has a functional relationship with its refractive index n2. The model's task is to predict the correct refracted light ray based on the input image.
    
- **I/O Format:**
    
    - Input: IMG_SIZE x IMG_SIZE RGB image (containing two media and incident light).
        
    - Output: Same as input but with additional red refracted light ray drawn.
        
- **Main Parameters:** IMG_SIZE, NUM_SAMPLES_TRAIN.

---

## 4. **physics_simulation/generate_reaction_diffusion_deprecated.py**

- **Purpose:** (Exploratory/Deprecated) This script simulates a reaction-diffusion system based on the **Gray-Scott model** to generate complex, fractal-like "snowflake" patterns. It essentially belongs to continuous physical field simulation (involving discretization of partial differential equations and floating-point operations), rather than simple discrete state cellular automata.
    
- **Status:** **Deprecated**. Although the physical simulation logic is correct, this task's output is a dynamically evolving result and is highly sensitive to initial conditions, not conforming to the "mapping from a clear input to a unique deterministic output" studied in the paradigm, therefore judged as "unsuitable" and abandoned. Additionally, because its core logic is based on continuous physical simulation, it has been moved from cellular automata category to physics simulation category.
    
- **Logic:** The script starts from one or several "seeds," and gradually generates complex solid structures by iteratively simulating the diffusion and reaction of two continuous fields of nutrient and matter (using convolution to calculate Laplacian operator).

---

## 5. **physics_simulation/generate_cube_rotation_matplotlib_deprecated.py**

- **Purpose:** (Early exploration version) Aims to test the model's ability to infer and render correct views of 3D objects from abstract pose parameters (rotation angles).
    
- **Logic:** This script uses matplotlib's 3D plotting engine to render cubes. It directly builds and rotates objects in 3D space. While functionally feasible, because matplotlib's control over rendering hierarchy is relatively complex, it may cause unexpected visual effects in occlusion relationships between wireframes and filled faces at certain angles.
    
- **Status:** **Deprecated**. Replaced by subsequent versions based on Pillow with more controllable rendering effects.

---

## 6. **physics_simulation/generate_cube_rotation_pillow_v1.py**

- **Purpose:** (Technical upgrade version) Aims to test the model's ability to infer and render correct views of 3D objects from abstract pose parameters, using a more low-level, precise technical route.
    
- **Logic:** This script is a major technical refactoring of this task. It abandons the high-level matplotlib library and instead uses the more basic Pillow library. The script manually implements complete 3D-to-2D projection transformation, back-face culling based on vector cross product, and depth sorting based on face average depth internally. This approach provides complete control over the rendering pipeline, ensuring physically correct occlusion relationships and layer order at all angles.
    
- **Status:** This is a crucial step toward the final successful version, but lacks the important auxiliary strategy of "highlighting vertices."

---

## 7. **physics_simulation/generate_cube_rotation_pillow_with_anchor.py**

- **Purpose:** (Final version used in the paper) Tests the model's ability to infer and render correct views of 3D objects from abstract pose parameters, and assists model learning by introducing "visual anchors."
    
- **Logic:** This script inherits all the precise, Pillow-based manual rendering pipeline from physics_simulation/generate_cube_rotation_pillow_v1.py. On this basis, it introduces a **key innovation:** after all regular rendering steps are completed, it always draws a conspicuous highlight marker (an orange dot) at a fixed special vertex (like the (1,1,1) corner), regardless of whether that vertex is occluded in the current view. This "visual anchor" provides a constant reference for the model, greatly helping the model solve the inherent symmetry and ambiguity problems of rotation, thus successfully converging.
    
- **I/O Format:**
    
    - Input: 24-bit binary string (3 angles * 8 bits/angle).
        
    - Output: 256x256 RGB image.
        
- **Main Parameters:** NUM_SAMPLES, IMAGE_SIZE_PX, SPECIAL_VERTEX_INDEX.
    
---

## 8. **physics_simulation/generate_cube_rotation_pillow_wireframe.py**

- **Purpose:** (Variant experimental version) Tests whether the model can learn 3D rotation using only wireframe and anchor point information under sparser visual input.
    
- **Logic:** This script is an **ablation version** of physics_simulation/generate_cube_rotation_pillow_with_anchor.py. It retains all core logic, including precise wireframe drawing and special vertex highlighting, but **removes all face color filling**. This creates a "wireframe mode" dataset, aiming to explore whether the model can still understand and reconstruct 3D structure when surface information is missing.
    
- **Status:** This is a variant experiment for in-depth analysis.

---

## 9. **physics_simulation/generate_catenary_curve_simulation_deprecated.py**

- **Purpose:** This is my early script for exploring the catenary problem, aiming to test the model's ability to learn nonlinear curves determined by physical laws.
    
- **Logic:** This version is my initial attempt at the catenary problem, possibly using a numerical solver to reverse solve the catenary equation from given parameters like two endpoints and curve length. This method may have numerical instability issues and is early exploratory work, later replaced by the more robust method of generate_catenary_curve_from_points.py.
    
- **I/O Format:**
    
    - **Input:** IMG_SIZE x IMG_SIZE RGB image (containing two endpoints and other information).
        
    - **Output:** IMG_SIZE x IMG_SIZE RGB image (containing generated catenary).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE.

---

## 10. **physics_simulation/generate_catenary_curve_from_points.py**

- **Purpose:** Tests the model's ability to learn nonlinear curves (catenary) uniquely determined by physical laws (minimum potential energy principle).
    
- **Logic:** The script adopts an efficient "forward construction" method: first, it randomly defines mathematical parameters a, b, c of a catenary curve; then it randomly samples three points on this perfect curve (two anchor points P1, P2 and one passing point P3). The input image only contains these three points, the output image draws the complete catenary segment connecting P1 and P2 and passing through P3. This method avoids the complex and unstable process of solving parameters from points.
    
- **I/O Format:**
    
    - **Input:** IMG_SIZE x IMG_SIZE RGB image (containing three points).
        
    - **Output:** IMG_SIZE x IMG_SIZE RGB image (three points + catenary).
        
- **Main Parameters:** NUM_SAMPLES_TRAIN, IMG_SIZE.

---

## 11. **physics_simulation/generate_orbital_path_from_initial_state.py**

- **Purpose:** Tests the model's ability to learn more complex physical laws (Kepler's laws / law of universal gravitation).
    
- **Logic:** The script first mathematically defines a random, stable elliptical orbit. Then it randomly selects a point on the orbit as the planet's initial position and calculates the velocity vector at that point. The input image encodes the star's position, planet's position, planet's velocity direction, and velocity magnitude through points and line segments of different colors. The output image draws the complete elliptical orbit on this basis.
    
- **I/O Format:**
    
    - **Input:** IMG_SIZE x IMG_SIZE RGB image (encoding initial state).
        
    - **Output:** IMG_SIZE x IMG_SIZE RGB image (initial state + complete orbit).
        
- **Main Parameters:** NUM_SAMPLES, IMG_SIZE, G (gravitational constant).

---

# F: ARC-AGI Exploration

## 1. **arc_agi/generate_arc_contextual_color_swap.py**

- **Purpose:** Tests the model's ability to learn rules from local "context" or "examples" in the image and apply them to global data in the same image. This directly mimics the core concept of ARC-AGI tests.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. First manually analyze the logic of ARC puzzles, then program them. Each input image has four color blocks in the top-left corner defining two pairs of color swap rules (e.g., swap colors at (0,0) and (0,1)). The rest of the image randomly scatters dots of these four colors. The model's task is to generate an output image where all scattered dots' colors have been swapped according to the top-left rules.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 2. **arc_agi/generate_arc_find_cross_pattern.py**

- **Purpose:** Tests the model's ability to perform visual pattern recognition (or "object detection") in the presence of large amounts of noise.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image is a red background with a large number of yellow scattered dots. Among them, some yellow dots are carefully arranged into 3x3 cross patterns, while others are randomly distributed noise. The model's task is to "remove the coarse and retain the fine," ignoring all noise dots and accurately finding all hidden cross patterns, highlighting them in blue in the output image.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 3. **arc_agi/generate_arc_find_odd_one_out.py**

- **Purpose:** Tests the model's ability to perform a complex "Find the Odd One Out" meta-reasoning task. The model needs to compare patterns row by row, find special cases, and recombine them into the output.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input is a large grid divided into 4 rows. Each row contains four similar 3x3 small patterns, three of which are identical "ordinary" patterns and one is a "special" pattern. The model's task is to identify the "special" pattern in each row, and rearrange these four special patterns found from different rows into a 2x2 output grid.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM_IN, GRID_DIM_OUT, NUM_SAMPLES.

---

## 4. **arc_agi/generate_arc_connect_colored_pairs.py**

- **Purpose:** Tests the model's ability to identify multiple independent "connection tasks" in the same image and understand an implicit "layer" or "drawing priority" rule.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image scatters several pairs of dots with the same color. The model's task is to find each pair of same-colored dots and connect them with straight lines of the corresponding color. An additional hidden rule is that if horizontal and vertical lines cross, vertical lines are always drawn above horizontal lines.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 5. **arc_agi/generate_arc_conditional_perpendicular_lines.py**

- **Purpose:** Tests the model's ability to perform different geometric operations based on objects' **attributes (color)** and **global references (boundary lines, image edges).**
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image contains a horizontal gray baseline and some red and blue scattered dots. The model's task is: for each **red** dot, draw a perpendicular line from that dot to the **gray baseline**; for each **blue** dot, draw a perpendicular line from that dot to the **nearest horizontal edge of the image** (top or bottom).
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 6. **arc_agi/generate_arc_column_projection.py**

- **Purpose:** Tests the model's ability to recognize complex contextual relationships ("below... and within...") and perform conditional column operations.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image contains a large, specific-colored downward arrow and some same-colored scattered dots. The model's task is to find all scattered dots located directly below the arrow's body. Then, for each **vertical column** containing such "qualified" dots, paint all pixels in that column from the arrow's bottom to the image's bottom with the projection color in the output image.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 7. **arc_agi/generate_arc_procedural_spiral.py**

- **Purpose:** Tests the model's ability to execute an iterative, procedural generation algorithm. The model needs to understand instructions, track state (current position, direction, length), and execute loops.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image is very simple: two color blocks (color A and color B) in the top-left corner as instructions, and a blue dot as the drawing "starting point." The model's task is to start from this blue dot and generate an outward expanding spiral. The spiral drawing follows strict rules: the first segment (length 2) goes left, color A; the second segment (length 2) goes down, color B; the third segment (length 3) goes right, color A, and so on.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 8. **arc_agi/generate_arc_fractal_stamping.py**

- **Purpose:** Tests the model's ability to understand and execute recursive or fractal generation rules. The model needs to use the input pattern itself as a "brush" and repeat drawing according to "instructions" in the input pattern.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input is a 4x4 pattern. The output is a larger 16x16 canvas. The rule is: traverse each cell in the 4x4 input pattern, if the cell at position (r, c) is **red**, then copy ("stamp") the entire 4x4 input pattern completely onto the output canvas at position (r*4, c*4) as the top-left corner.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM_IN, GRID_DIM_OUT, NUM_SAMPLES.

---

## 9. **arc_agi/generate_arc_flood_fill.py**

- **Purpose:** Tests the model's ability to execute the classic "flood fill" or "paint bucket" algorithm.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The script first programmatically generates a guaranteed connected enclosed area surrounded by green "walls" on a black background. The input image is this image with green walls. The output image fills the black area surrounded by these green walls completely with yellow on top of the input.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 10. **arc_agi/generate_arc_layered_fill.py**

- **Purpose:** Tests the model's ability to understand a highly procedural, topology-distance-dependent and conditional-judgment-based complex filling algorithm.
    
- **Logic:** Adopts a "logic programming assisted learning" strategy. The input image is divided into multiple areas by line segments. Each area contains one or two "color instruction points." If there's only one color point (A), the entire area is filled with color A. If there are two color points (A and B), the area is "layered" filled: the layer closest to the area boundary is painted with color A, the next closest layer with color B, and so on, forming alternating color "contour line" patterns.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 11. **arc_agi/generate_arc_fluid_simulation.py**

- **Purpose:** Tests the model's ability to learn and simulate a fluid dynamic process with specific rules in image space.
    
- **Logic:** The input image contains several red horizontal "baffles" and one or two purple "faucets" at the top. The model's task is to simulate purple liquid flowing out from the faucets. When the liquid encounters baffles, it splits to left and right sides, and at edges without baffle support, it continues to drip down until the image bottom.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 12. **arc_agi/generate_arc_periodic_conditional_fill.py**

- **Purpose:** This experiment aims to test the model's ability to learn a complex conditional formatting rule with periodicity and special cases.
    
- **Logic:** The input image has a yellow line segment at the very bottom defining an "operation area." The model needs to check row by row from bottom to top. According to the distance d from the current row to the second-to-last row, apply modulo 6 periodic rules:
    
    - d % 6 is 1 or 5: Fill yellow in operation area.
        
    - d % 6 is 0,2,4: Fill background color in operation area.
        
    - d % 6 is 3 (special rule): Not only fill **green** in operation area, but also change **all** original scattered dots in that row to **green**.
        
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 13. **arc_agi/generate_arc_fill_square_holes.py**

- **Purpose:** This experiment tests the model's ability to perform multi-step visual reasoning: first needs to identify complex "foreground in background" (i.e., holes in rectangles), then perform geometric attribute judgment (whether it's a square) on identified objects, and finally color based on judgment results.
    
- **Logic:** The input image contains multiple gray rectangles with black holes. The model's task is to identify all holes, judge the shape of each hole, and if a hole is a **square**, fill it with red in the output image.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 14. **arc_agi/generate_arc_conditional_recoloring.py**

- **Purpose:** Tests the model's ability to understand visual layers and perform conditional object attribute modification.
    
- **Logic:** The input image contains a "bottom layer" composed of dark blue scattered dots and black background, and a light blue rectangle as a "marking layer." This marking layer is overlaid on the bottom layer but only covers the black background without changing the original dark blue scattered dots. The model's task is to identify the area of this light blue rectangle, find all dark blue scattered dots in the **bottom layer** that are located within this area, and change the color of these dots to green in the output image.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 15. **arc_agi/generate_arc_sort_by_length_remap_position.py**

- **Purpose:** Tests the model's ability to perform a complex sorting task of "attribute-position decoupling and remapping."
    
- **Logic:** The input image contains a series of colored vertical pillars with different colors, lengths, and positions. The model's task is:
    
    1. Conceptually, extract the **length** attribute of all pillars and sort them.
        
    2. Simultaneously, keep all pillars' original horizontal **positions** and **color** attributes unchanged.
        
    3. In the output image, draw the **shortest** pillar at the position of the **leftmost** original pillar, using that position's original color; draw the **second shortest** pillar at the position of the **second leftmost** original pillar, using that position's original color, and so on.
        
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 16. **arc_agi/generate_arc_jigsaw_puzzle_simple.py**

- **Purpose:** Tests the model's ability to solve a visual matching and transformation problem (early version).
    
- **Logic:** The left side of the input image is a template with several puzzle pieces dug out, the right side scatters enlarged, randomly rotated/mirrored versions of the corresponding puzzle pieces. The **key compromise** of this version is that to simplify the matching problem, each puzzle piece has a unique size (number of squares), allowing the model to use this "shortcut" to identify correspondences rather than relying entirely on shape.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES, num_source_pieces.

---

## 17. **arc_agi/generate_arc_jigsaw_puzzle_advanced.py**

- **Purpose:** Tests the model's ability to solve a complex **visual matching and transformation** problem.
    
- **Logic:** This is a major improvement to the jigsaw_puzzle task. The left side of the input image is a template with several puzzle pieces dug out, the right side scatters 2x enlarged, randomly rotated and mirrored versions of the corresponding puzzle pieces, plus some noise pieces. **The key improvement** is that this version allows generating multiple **same-sized but differently shaped** puzzle pieces, forcing the model to **truly match based on shape**. The output image shows all puzzle pieces correctly shrunk and placed back into the template.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES, num_source_pieces.

---

## 18. **arc_agi/generate_arc_connect_path_by_sequence.py**

- **Purpose:** Tests the model's ability to parse external instruction sequences and perform multi-step, stateful path connection tasks in the image accordingly.
    
- **Logic:** The input image contains two parts: (1) Multiple squares with colored interiors are scattered on the canvas; (2) A row of color blocks at the bottom of the image is an "instruction sequence." The model's task is to connect corresponding squares in the order of the instruction sequence colors. For example, if the instruction is [red, green, blue], the model needs to first draw a line from the red square to the green square, then draw a line from the green square to the blue square. An additional rule is that the color of each connecting segment is determined by the **previous** square's color.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** GRID_DIM, NUM_SAMPLES.

---

## 19. **arc_agi/generate_arc_reflection_simulation_deprecated.py**

- **Purpose:** (Deprecated) Aims to test the model's ability to understand complex physical optics-based rules, including ray emission, collision detection, angle reflection, and color transformation.
    
- **Logic:** The script attempts to write a generation script for this very complex ARC task, but because programmatically and unambiguously generating all possible physically correct reflection and interaction scenarios is extremely difficult, it cannot guarantee the quality and consistency of generated data, and this experiment was ultimately abandoned.
    
- **I/O Format:** Image-to-Image.
    
- **Main Parameters:** Not applicable.

---

# G: Chinese Chess

## 1. **chinese_chess/generate_chess_positions_by_random_moves.py**

- **Purpose:** Quickly generates a large number of plausible, legal Chinese chess positions by simulating a completely random player making moves.
    
- **Logic:** The script starts from the standard Chinese chess starting position. In a loop, it gets all legal moves in the current position, then randomly selects one to execute. This process repeats max_steps times, finally obtaining a random but legal position.
    
- **I/O Format:**
    
    - Output: FEN format position string.
        
- **Main Parameters:** max_steps, max_capture.

---

## 2. **chinese_chess/generate_chess_positions_by_random_placement.py**

- **Purpose:** Generates a large number of atypical but mostly legal Chinese chess positions by randomly placing pieces on the board (rather than simulating moves), used for stress testing the model's robustness.
    
- **Logic:** Instead of generating positions through moves, this script directly places pieces randomly on the board following piece position constraints and the rule that kings cannot face each other, thereby creating a large number of positions that rarely appear in real games but are syntactically legal.
    
- **I/O Format:**
    
    - Output: FEN format position string.
        
- **Main Parameters:** num_fens.

---

## 3. **chinese_chess/generate_chess_positions_from_engine_self_play.py**

- **Purpose:** Generates a large number of high-quality, combat-logic-compliant Chinese chess positions (FEN format) as basic data source for training chess AI.
    
- **Logic:** Simulates tens of thousands of high-level self-play games by calling a powerful third-party chess engine (PikaFish) through subprocess. During simulation, it records the FEN representation of each move in the game, thereby building a large and realistic position database.
    
- **I/O Format:**
    
    - Output: A .txt file, each line containing a complete FEN string.
        
- **Main Parameters:** num_games, max_steps, depth.

---

## 4. **chinese_chess/generate_preprocess_legal_moves.py**

- **Purpose:** This is a data preprocessing script for converting FEN format position datasets into a "legal move prediction" task that models can directly learn.
    
- **Logic:** Reads a FEN file, for each position uses the cchess library to parse and generate all legal moves. Then, according to a global mapping file, converts each specific move (like 'h2e2') into a unique integer ID.
    
- **I/O Format:**
    
    - Input: .txt file, one FEN per line.
        
    - Output: .jsonl file, each JSON object contains fen and its corresponding legal_move_ids list.
        
- **Main Parameters:** fen_file, output_file.

---

## 5. **chinese_chess/generate_chess_resolve_check_task.py**

- **Purpose:** Generates a dataset specifically targeting the "resolving a check" tactical scenario in Chinese chess. This task requires the model to find all legal moves that can resolve the check when in a checked state.
    
- **Logic:** The script first filters from a large random position library, only keeping positions that satisfy the condition of "currently being checked but not checkmated (not stalemate)." Then, for each filtered position, it calculates all legal moves that can resolve the check and saves the IDs of these moves.
    
- **I/O Format:**
    
    - Output: A .jsonl file. Each JSON object contains fen (position) and legal_move_ids (an integer list representing all legal check-resolving moves).
        
- **Main Parameters:** fen_file, output_file.

---

# Training Scripts and Tools

---

## 1. **utils/eval_hanoi.py**

- **Purpose:** This is a **verification tool**, not a training script. Its function is to receive a Tower of Hanoi solution string (e.g., "1>3;1>2;...") generated by a large language model (or other sources) and strictly simulate it according to Tower of Hanoi game rules to judge whether the solution is correct.

- **Core Features:**
    - **State Simulation:** Internally simulates the state of three pillars and n disks.
    - **Rule Checking:** Automatically checks whether each move is legal (e.g., cannot move from an empty pillar, larger disk cannot be placed on smaller disk).
    - **Final State Verification:** Checks whether all disks are moved to the target pillar in correct order after all moves are completed.
    - **Clear Error Reporting:** If the solution is incorrect, it clearly points out which step has an error and the reason.

- **Usage:**
    1. **As a command-line tool:**
        - Open `eval_hanoi.py` file.
        - At the bottom `if __name__ == "__main__":` section, find `verify_hanoi_solution(n, solution_str)` function call.
        - Change the first parameter `n` to the number of disks you want to verify.
        - Replace the second parameter `solution_str` with the solution string you obtained from the large model.
        - Run script: `python eval_hanoi.py`
        - Console will output ✅ Correct! or ❌ Error: with details.

    2. **Import as a library:**
        ```python
        from eval_hanoi import verify_hanoi_solution
        
        n = 6
        llm_output = "1>2;1>3;..." # Output from your model
        is_correct = verify_hanoi_solution(n, llm_output)
        print(f"Is LLM solution correct: {is_correct}")
        ```

---

## 2. **utils/create_videos.py**

- **Purpose:** This is a **training process visualization video generation tool**, used to convert the sequence of eval images generated during model training into a video, facilitating observation and analysis of model learning dynamics.

- **Core Functions:**
    - **Batch Processing:** Automatically generates independent evolution videos for multiple samples (up to 32).
    - **Flexible Configuration:** Supports custom start step, end step, step interval, and video frame rate.
    - **High-Quality Output:** Uses FFmpeg to generate high-quality MP4 videos, supporting custom encoding parameters.

- **Main Configuration Parameters:**
    - `IMAGE_DIR`: Directory path storing eval images
    - `OUTPUT_DIR`: Video output directory (automatically created)
    - `NUM_SAMPLES`: Number of samples to process (0-31)
    - `START_STEP`: Video start training step
    - `END_STEP`: Video end training step
    - `STEP_INTERVAL`: Sampling interval (e.g., take one image every 20 steps)
    - `VIDEO_FRAMERATE`: Output video frame rate (controls playback speed)
    - `FFMPEG_PATH`: FFmpeg executable file path

- **Usage:**
    1. **Configure Parameters:** Modify relevant parameters in the configuration area at the beginning of the script.
    2. **Prepare Data:** Ensure `IMAGE_DIR` contains eval images saved according to naming rules (e.g., `step_100_sample_0.png`).
    3. **Run Script:** `python utils/create_videos.py`
    4. **View Results:** Find generated MP4 video files in `OUTPUT_DIR`.

- **Typical Application Scenarios:**
    - Observing model learning process on ARC-AGI tasks
    - Analyzing convergence patterns of cellular automaton evolution tasks
    - Creating training dynamic visualization materials for papers or presentations
    - Comparing learning speed differences under different models or hyperparameters

- **Technical Implementation:**
    - Uses FFmpeg's concat protocol to batch process image sequences
    - Automatically generates temporary file lists to avoid command line length limits
    - Supports error handling and progress indication
    - Automatically cleans up temporary files

---

## 3. **utils/analyze_ca_inverse_ambiguity.py**

- **Purpose:** This is a **Monte Carlo simulation analysis tool** for quantitatively estimating the **ambiguity probability** in 1D cellular automata (CA) inverse engineering tasks—the probability that different (rule, evolution layers) combinations produce identical outputs.

- **Research Background:** This tool supports the discussion on the necessity of "uniqueness verification" in the paper. In the `cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py` script, we filter out samples with non-unique solutions. This tool quantitatively estimates the probability of such ambiguity through large-scale sampling, providing theoretical basis for experiment design.

- **Core Algorithm:**
    - For each random initial state, enumerate 256 rules × 4 layer depths = 1024 combinations
    - Count combinations producing identical outputs (i.e., "collisions")
    - Estimate average ambiguity probability through 200,000 samples
    - Uses iterative optimization: only 4 CA evolutions per rule (instead of 16)

- **Usage:**
    ```bash
    # Run with default parameters (200K samples, 30-bit width)
    python utils/analyze_ca_inverse_ambiguity.py
    
    # Custom parameters
    python utils/analyze_ca_inverse_ambiguity.py --samples 100000 --length 36
    
    # Quiet mode (only output final result)
    python utils/analyze_ca_inverse_ambiguity.py --quiet
    ```

- **Output Example:**
    ```
    =================================================================
    1D Cellular Automata Inverse Engineering Ambiguity Simulation
    =================================================================
    Parameters: length=30, samples=200,000
    ...
    **Estimated ambiguity probability**: 0.0000009530 (0.00009530%)
    95% Confidence Interval: [0.0000008912, 0.0000010148]
    ```

- **Main Parameters:**
    - `--samples`: Number of samples (default: 200000)
    - `--length`: CA state width in bits (default: 30)
    - `--quiet`: Quiet mode

---

