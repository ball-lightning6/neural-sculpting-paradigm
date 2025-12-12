# A: 符号数学逻辑 (Symbolic Math Logic)

## 1. **symbolic_math_logic/generate_conditional_add_subtract.py**

- **逻辑:** 脚本生成两个N-bit整数的加法或减法（取绝对值）问题。它包含两种模式：
    
    1. **INDICATOR_BIT模式:** 在输入前添加一个bit（0代表加法，1代表减法）作为明确的指令。
        
    2. **PROBABILITY_MIX模式:** 不提供任何指令，但在生成数据时按一定概率混合加法和减法样本。这是为了模拟一个“规则不纯净”的环境。
        
- **I/O格式:**
    
    - 输入: (INDICATOR_BIT模式) 1 (指示位) + 2N (操作数) bit；(PROBABILITY_MIX模式) 2N bit。
        
    - 输出: N+1 bit的二进制多标签向量，代表计算结果。
        
- **主要参数:** NUM_BITS, DATASET_SIZE, EXPERIMENT_MODE, PROBABILITY_ADD。

---

## 2. **symbolic_math_logic/generate_add_binary_modulo.py**

- **用途:** 这是一个早期的基础算术实验，用于测试模型学习模加法（或称“截断加法”）的能力，这种运算常见于计算机硬件的定宽整数运算。
    
- **逻辑:** 输入两个N-bit整数a和b，计算它们的和，然后将结果对2^N取模，有效地丢弃了任何溢出的进位（例如第N+1位）。输入被编码为两个二进制字符串的拼接，输出被编码为多标签二分类格式。
    
- **I/O格式:**
    
    - 输入: N * 2长度的二进制字符串（两个N-bit操作数拼接）。
        
    - 输出: N长度的多标签二分类向量（0/1列表）。
        
- **主要参数:** n_samples, bit_length。

---

## 3. **symbolic_math_logic/generate_multiply_binary.py**

- **用途:** 作为二进制算术能力的一个基准测试，生成N-bit整数的乘法数据集。
    
- **逻辑:** 随机生成两个NUM_BITS位的整数，将它们的二进制字符串拼接作为输入，将其乘积的二进制表示作为输出。
    
- **I/O格式:**
    
    - 输入: NUM_BITS * 2长度的二进制字符串。
        
    - 输出: NUM_BITS * 2长度的二进制多标签向量。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 4. **symbolic_math_logic/generate_multiply_binary_no_carry_phase1.py**

- **用途:** 这是乘法“解耦”实验的第一阶段。旨在测试模型是否能学会乘法的第一步：无进位的按位相乘和错位相加，将一个复杂的乘法问题分解为一个更简单的计数问题。
    
- **逻辑:** 模拟手算乘法的过程。输入是两个N-bit数a和b。输出不再是最终乘积，而是一个长度为2*N的计数器向量，其中第i个计数器记录了最终乘积的第i位在进位前应该有多少个'1'。
    
- **I/O格式:**
    
    - 输入: NUM_BITS * 2长度的二进制字符串。
        
    - 输出: (NUM_BITS * 2) * BITS_PER_COUNTER长度的二进制多标签向量。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 5. **symbolic_math_logic/generate_multiply_binary_from_counts_phase2.py**

- **用途:** 这是乘法“解耦”实验的第二阶段。旨在验证一个独立的模型能否学会处理复杂的进位逻辑，即从一个“无进位计数向量”中计算出最终的二进制乘积。
    
- **逻辑:** 脚本的输入是上一个“无进位乘法”任务的输出格式（一个计数器向量）。它将这个向量所代表的值（即原始的a*b）计算出来，并将其标准的二进制表示作为输出。
    
- **I/O格式:**
    
    - 输入: 长度为2NBITS_PER_COUNTER的二进制字符串，代表无进位计数值。
        
    - 输出: 长度为2*N的二进制多标签向量，代表最终乘积。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 6. **symbolic_math_logic/generate_add_hexadecimal.py**

- **用途:** 对比模型在不同符号系统下的学习能力。此脚本旨在验证模型学习的是加法这一抽象数学概念，还是仅仅是特定于二进制符号的模式。
    
- **逻辑:** 随机选择两个16-bit整数。脚本会生成两组独立的数据集：一组的输入是这两个数的二进制表示，另一组的输入是它们的十六进制字符串表示。两组数据集的输出完全相同，都是和的17位二进制表示。
    
- **I/O格式:**
    
    - 输入 (二进制): 32 ('0'/'1') | 输入 (十六进制): 8 ('0'-'9','A'-'F')。
        
    - 输出: 17位二进制多标签向量。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 7. **symbolic_math_logic/generate_multiply_decimal.py**

- **用途:** 测试模型处理非二进制符号输入（0-9字符），并执行算术运算（乘法）的能力。
    
- **逻辑:** 生成两个N位十进制数，将其字符串拼接作为输入。计算它们的乘积，并将结果转换为二进制作为输出。
    
- **I/O格式:**
    
    - 输入: NUM_DIGITS * 2长度的字符串，由'0'-'9'组成。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量。
        
- **主要参数:** NUM_DIGITS, DATASET_SIZE。

---

## 8. **symbolic_math_logic/generate_add_binary_with_position_shuffle.py**

- **用途:** 这是“语义洗牌”系列实验中的“位置洗牌”部分。它旨在验证模型是否依赖于输入的固定空间结构，还是能学习到与位置无关的抽象关系。
    
- **逻辑:** 该脚本为二进制加法任务生成两套数据集。一套的输入是两个N-bit数按标准顺序拼接 (a+b)。另一套的输入则是将标准输入中的每个bit按照一个预先定义好的、固定的随机映射进行位置重排。两套数据集的输出是相同的。
    
- **I/O格式:**
    
    - 输入: NUM_BITS * 2长度的二进制字符串。
        
    - 输出: NUM_BITS + 1长度的二进制多标签向量。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 9. **symbolic_math_logic/generate_symbol_add_shuffle_dataset.py**

- **用途:** 这是我们研究中一项**关键的决定性实验**，旨在彻底分离模型的"表面模式匹配"能力和"抽象结构学习"能力。

- **逻辑:** 脚本可配置进行两种"洗牌"：

    1. **语义洗牌 (Semantic Shuffle):** 将表示N进制数字的符号（如'0'-'F'）随机映射到任意可打印字符。这切断了符号与其固有数值含义的联系。

    2. **位置洗牌 (Positional Shuffle):** 将输入字符串中每个字符的位置根据一个固定的随机映射进行重排。这破坏了所有局部的、空间的统计规律。

- **I/O格式:**

    - 输入: 2 * NUM_BITS长度的字符串（字符集可变）。

    - 输出: 和的二进制多标签向量。

- **主要参数:** NUM_BITS, BASE, SHUFFLE_SEMANTICS, SHUFFLE_POSITIONS。

---

## 10. **symbolic_math_logic/generate_add_hidden_constant.py**

- **用途:** 测试模型在没有任何直接线索的情况下，从大量样本中**推断出隐藏规则或参数**的能力。这类似于一个简化的系统辨识（System Identification）问题。
    
- **逻辑:** 脚本在内部定义一个固定的“隐藏常数”C。对于每个样本，它只将随机数x作为输入，而将x+C的结果作为输出。模型必须通过学习所有样本的共性，将常数C的效果编码到其权重中。
    
- **I/O格式:**
    
    - 输入: NUM_BITS长度的二进制字符串 (代表x)。
        
    - 输出: NUM_BITS+1长度的二进制多标签向量 (代表x+C)。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 11. **symbolic_math_logic/generate_multitask_alu.py**

- **用途:** 此脚本旨在构建一个模拟**算术逻辑单元 (ALU)** 的多任务学习场景。它测试模型能否在一次前向传播中，对同一份输入并行执行多种不同的、定义明确的计算任务。
    
- **逻辑:** 输入是两个N-bit的二进制数。输出是一个长的二进制向量，被划分为多个“地址段”，每个段对应一个特定运算（加、减、与、或、异或、比较）的结果。这迫使模型在内部将计算流图分叉，并将结果路由到指定的输出位置。
    
- **I/O格式:**
    
    - 输入: NUM_BITS * 2长度的二进制字符串。
        
    - 输出: TOTAL_OUTPUT_BITS长度的二进制多标签向量，由所有任务结果拼接而成。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 12. **symbolic_math_logic/generate_modulo_operation.py**

- **用途:** 探究模型学习模运算（Modulo Operation）的能力，这是一个在数论和计算机科学中至关重要但具有“循环”性质的运算。
    
- **逻辑:** 脚本生成一个A_BITS位的整数a和一个N_BITS位的整数n，任务是计算a % n。在后续的探索中，我们将n固定为3，以深入研究为何模型难以学习这个看似简单的规则。
    
- **I/O格式:**
    
    - 输入: A_BITS长度的二进制字符串 (当前版本只输入a)。
        
    - 输出: N_BITS长度的二进制多标签向量。
        
- **主要参数:** A_BITS, N_BITS (尽管当前固定为3), DATASET_SIZE。

---

## 13. **symbolic_math_logic/generate_rsa_encryption.py**

- **用途:** 测试模型学习高度非线性的、在计算上被认为是“困难”的确定性规则的能力。RSA加密是一个典型的例子。
    
- **逻辑:** 该脚本在一个固定的公钥(e, n)下，将所有可能的消息(m)从0到n-1进行加密，生成对应的密文(c)。
    
- **I/O格式:**
    
    - 输入: bits位的二进制字符串，代表消息m。
        
    - 输出: bits位的二进制字符串，代表密文c。
        
- **主要参数:** e, n (公钥参数), bits (编码位数), output_file。

---

## 14. **symbolic_math_logic/generate_deduction_chain_text.py**

- **用途:** 生成多步逻辑推理任务，测试模型执行符号演绎（deduction）的能力，类似于一个简化的定理证明器。
    
- **逻辑:** 脚本定义了一系列隐含的推理规则（如 (A, B) -> C）。它首先构建一个多步的推理链（例如5步），然后确定推导出最终结论所需的所有初始“事实”。正样本的输入包含所有这些必要事实（可能还有一些无关的“噪音”事实），负样本则会故意缺少一个或多个关键事实。模型的任务是判断给定的查询（Query）能否从给定的事实和规则中被推导出来。
    
- **I/O格式:**
    
    - 输入: Facts: ...\nRules: ...\nQuery: ... 格式的文本字符串。
        
    - 输出: '1' (可以推出) 或 '0' (无法推出)。
        
- **主要参数:** num_samples, attr_range, depth。

---

## 15. **symbolic_math_logic/generate_deduction_multirule_text.py**

- **用途:** 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。
    
- **逻辑:** 脚本定义了两套独立的“事实->结论”规则。在生成每个样本时，它首先确定本次要查询的目标（5或6），然后检查推断该目标所需的前提条件。正样本会提供所有必要条件（外加一些噪音事实），负样本则会故意缺少至少一个必要条件。
    
- **I/O格式:**
    
    - 输入: Facts: ..., Query: ... 格式的文本字符串。
        
    - 输出: 单个字符 '1' (可以推出) 或 '0' (无法推出)。
        
- **主要参数:** n_samples。

---

## 16. **symbolic_math_logic/generate_deduction_multirule_text_v2.py**

- **用途:** 测试模型在面对多个独立的、互不相干的规则时，能否根据查询（Query）正确地“路由”到相应的规则并进行判断。
    
- **逻辑:** 脚本定义了两套独立的“事实->结论”规则。在生成每个样本时，它首先确定本次要查询的目标（5或6），然后检查推断该目标所需的前提条件。正样本会提供所有必要条件（外加一些噪音事实），负样本则会故意缺少至少一个必要条件。
    
- **I/O格式:**
    
    - 输入: Facts: ..., Query: ... 格式的文本字符串。
        
    - 输出: 单个字符 '1' (可以推出) 或 '0' (无法推出)。
        
- **主要参数:** n_samples。

---

## 17. **symbolic_math_logic/generate_deduction_multirule_binary.py**

- **用途:** 这是对多规则推理任务的**格式优化**版本，旨在测试紧凑的二进制编码是否比稀疏的文本格式更有利于模型学习。
    
- **逻辑:** 核心逻辑与文本版本一致，但输入/输出表示被改变：
    
    1. 所有8个可能的事实被表示为一个8-bit的二进制掩码。
        
    2. 查询目标（5还是6）被表示为单个二进制位。
        
    3. 这两部分拼接成一个9-bit的输入字符串。
        
- **I/O格式:**
    
    - 输入: 8 (事实掩码) + 1 (查询目标编码) = 9位二进制字符串。
        
    - 输出: 单个字符 '1' 或 '0'。
        
- **主要参数:** n_samples。

---

## 18. **symbolic_math_logic/generate_deduction_fixed_depth.py**

- **用途:** 测试模型在有明确结构、固定深度的符号演绎任务中的多步推理能力。
    
- **逻辑:** 脚本首先在内部随机生成一个5步的推理链（例如 A+B->X, C+D->Y, ..., X+Y->Z）。然后，它通过“反向链（backchaining）”方法，从最终结论（Z）开始回溯，找出所有必须为真的初始事实。
    
    - **正样本:** 输入包含所有必要事实的掩码，查询目标为Z，标签为'1'。
        
    - **负样本:** 输入包含同样的事实掩码，但查询目标是一个无法从这些事实推断出的“噪音”属性，标签为'0'。
        
- **I/O格式:**
    
    - 输入: 16 (事实掩码) + 4 (查询目标编码) = 20位二进制字符串。
        
    - 输出: 单个字符 '1' (可以推出) 或 '0' (无法推出)。
        
- **主要参数:** depth, num_attrs, num_samples。

---

## 19. **symbolic_math_logic/generate_function_composition.py**

- **用途:** 测试模型学习函数组合（Function Composition）的能力。这要求模型像解释器一样，按顺序解析指令并对数据进行变换。
    
- **逻辑:** 脚本定义了四个基本函数（double, increment, square, decrement）。每个样本的输入由两部分组成：一个代表4个函数调用序列的指令串（每个函数用2-bit编码），和一个16-bit的初始整数。脚本会按顺序应用这4个函数，并确保每一步的中间结果都在[0, 65535]范围内，将最终结果作为输出。
    
- **I/O格式:**
    
    - 输入: (4 * 2) (函数指令) + 16 (初始值) = 24位二进制字符串。
        
    - 输出: 16位二进制字符串。
        
- **主要参数:** num_samples。

---

## 20. **symbolic_math_logic/generate_count_set_bits.py**

- **用途:** 测试模型执行全局聚合操作的能力。与局部规则不同，计数需要模型综合整个输入序列的信息。
    
- **逻辑:** 脚本生成一个随机的二进制字符串，并计算其中'1'的数量。balanced模式可以确保数据集中每种计数值的样本量大致相等。
    
- **I/O格式:**
    
    - 输入: input_bits长度的二进制字符串。
        
    - 输出: output_bits长度的二进制多标签向量，代表'1'的总数。
        
- **主要参数:** num_samples, input_bits, output_bits, balanced。

---

## 21. **symbolic_math_logic/generate_sum_pattern_positions.py**

- **用途:** 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。
    
- **逻辑:** 脚本将一个长二进制字符串按p位（BITS_PER_PATTERN）切分成q个（NUM_PATTERNS）连续的子模式。然后，它为每一种可能的子模式（共2^p种），计算出该模式在输入中所有出现位置的索引（1到q）之和。
    
- **I/O格式:**
    
    - 输入: p * q长度的二进制字符串。
        
    - 输出: (2^p) * BITS_PER_SUM长度的二进制多标签向量，代表每种模式的位置和。
        
- **主要参数:** BITS_PER_PATTERN, NUM_PATTERNS, DATASET_SIZE。

---

## 22. **symbolic_math_logic/generate_sum_pattern_positions_v2.py**

- **用途:** 测试模型执行更复杂的、分组式的并行聚合任务的能力。模型需要先分割输入，然后对每个分割后的模式进行分类，最后对属于同一类的模式的**位置信息**进行累加。
    
- **逻辑:** 脚本将一个长二进制字符串按p位（BITS_PER_PATTERN）切分成q个（NUM_PATTERNS）连续的子模式。然后，它为每一种可能的子模式（共2^p种），计算出该模式在输入中所有出现位置的索引（1到q）之和。
    
- **I/O格式:**
    
    - 输入: p * q长度的二进制字符串。
        
    - 输出: (2^p) * BITS_PER_SUM长度的二进制多标签向量，代表每种模式的位置和。
        
- **主要参数:** BITS_PER_PATTERN, NUM_PATTERNS, DATASET_SIZE。

---

## 23. **symbolic_math_logic/generate_sum_pairwise_hamming_distance.py**

- **用途:** 测试模型执行一个需要两层嵌套聚合操作的复杂任务。模型需要先在**每个比特位**上进行全局统计，然后再将**所有比特位**的结果累加起来。
    
- **逻辑:** 输入是N个M-bit的二进制数拼接成的字符串。任务是计算这N个数两两配对后的汉明距离总和。例如，对于[A, B, C]，需要计算dist(A,B) + dist(A,C) + dist(B,C)。脚本使用了一个O(N*M)的算法高效计算该值。
    
- **I/O格式:**
    
    - 输入: NUM_ITEMS * BITS_PER_ITEM长度的二进制字符串。
        
    - 输出: OUTPUT_BITS位二进制多标签向量，代表汉明距离总和。
        
- **主要参数:** NUM_ITEMS, BITS_PER_ITEM, DATASET_SIZE。

---

## 24. **symbolic_math_logic/generate_circular_shift.py**

- **用途:** 测试模型学习位移操作的能力，特别是循环位移（circular shift），这是密码学和底层编程中的常见操作。
    
- **逻辑:** 输入由两部分拼接而成：一个NUM_DATA_BITS长度的二进制数据串，和一个NUM_SHIFT_BITS长度的二进制数（代表向右循环移动的位数k）。输出是数据串循环右移k位后的结果。
    
- **I/O格式:**
    
    - 输入: NUM_DATA_BITS + NUM_SHIFT_BITS长度的二进制字符串。
        
    - 输出: NUM_DATA_BITS长度的二进制多标签向量。
        
- **主要参数:** NUM_DATA_BITS, NUM_SHIFT_BITS, DATASET_SIZE。

---

## 25. **symbolic_math_logic/generate_multiply_matrix_3x3.py**

- **用途:** 测试模型学习结构化代数运算（矩阵乘法）的能力，这比简单的标量运算需要更复杂的“数据路由”和“乘积累加”能力。
    
- **逻辑:** 输入是两个3x3的二进制矩阵，被展平并拼接成一个18位的二进制字符串。输出是这两个矩阵相乘后得到的3x3结果矩阵（其元素范围为0-3），同样被展平并编码成二进制多标签向量。
    
- **I/O格式:**
    
    - 输入: 18位的二进制字符串。
        
    - 输出: 18位的二进制多标签向量 (9个元素 * 2 bit/元素)。
        
- **主要参数:** num_samples。

---

## 26. **symbolic_math_logic/generate_evaluate_boolean_expression_text.py**

- **用途:** 测试模型解析一个简单的领域特定语言（DSL）并执行求值的能力，这比前面固定结构的表达式求值更进了一步。
    
- **逻辑:** 脚本随机生成一个布尔表达式，如 (x0 | x1) & (x2)。同时，它为表达式中涉及的所有变量随机赋一个布尔值（0或1）。输入由表达式字符串和变量赋值字符串拼接而成，输出是该表达式最终的求值结果。
    
- **I/O格式:**
    
    - 输入: x=...;expr=(...)格式的字符串。
        
    - 输出: 单个字符 '1' (True) 或 '0' (False)。
        
- **主要参数:** num_samples, num_vars。

---

## 27. **symbolic_math_logic/generate_evaluate_arithmetic_expression.py**

- **用途:** 训练模型执行符号表达式的求值任务，这要求模型理解运算符优先级（通过树状结构隐式表达）、变量替换和算术运算。
    
- **逻辑:** 脚本首先随机生成一个包含加、减、乘法、数字常量和变量'x'的表达式树。然后，它将树结构展平为前缀token序列并进行二进制编码。最后，它随机生成一个'x'的值，将表达式和'x'的值拼接作为输入，将最终计算结果作为输出。
    
- **I/O格式:**
    
    - 输入: (TOKEN_LEN * N) + X_BITS位二进制串，代表表达式及x的值。
        
    - 输出: OUTPUT_BITS位二进制串，代表求值结果。
        
- **主要参数:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE。

---

## 28. **symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply.py**

- **用途:** 这是对generate_evaluate_arithmetic_expression.py的简化版本，旨在通过移除乘法运算来降低学习难度，以测试模型在更基础的算术表达式求值上的能力。
    
- **逻辑:** 逻辑与前一个脚本类似，但随机生成的表达式树中只包含加法和减法运算，完全排除了乘法。这使得表达式的数值范围更可控，降低了模型的学习负担。
    
- **I/O格式:**
    
    - 输入: (TOKEN_LEN * N) + X_BITS位二进制串，代表表达式及x的值。
        
    - 输出: OUTPUT_BITS位二进制串，代表求值结果。
        
- **主要参数:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE。

---

## 29. **symbolic_math_logic/generate_evaluate_arithmetic_expression_no_multiply_small_range.py**

- **用途:** 这是在前一个“无乘法”版本基础上的进一步简化，通过缩小数值范围来进一步降低学习难度，用于精确诊断模型在最简单表达式求值任务上的性能瓶颈。
    
- **逻辑:** 逻辑与generate_evaluate_arithmetic_expression_no_multiply.py脚本相同，但VAL_RANGE和X_VAL_RANGE参数被设置为更小的值。这确保了计算过程中的所有中间值和最终结果都保持在一个较小的范围内，是难度最低的表达式求值版本。
    
- **I/O格式:**
    
    - 输入: (TOKEN_LEN * N) + X_BITS位二进制串，代表表达式及x的值。
        
    - 输出: OUTPUT_BITS位二进制串，代表求值结果。
        
- **主要参数:** VAL_RANGE, X_VAL_RANGE, DATASET_SIZE。

---

## 30. **symbolic_math_logic/generate_check_boolean_equivalence.py**

- **用途:** 测试模型对布尔代数逻辑等价性的判断能力。这是一个抽象的符号推理任务，要求模型理解表达式的结构和布尔运算法则。
    
- **逻辑:** 脚本随机生成两个包含变量('a','b','c','d')和布尔运算符('&', '|', '~')的表达式。它通过**真值表**方法，即遍历所有可能的变量赋值组合，来确定这两个表达式是否在所有情况下都产生相同的结果。
    
- **I/O格式:**
    
    - 输入: expr1=...;expr2=...格式的字符串。
        
    - 输出: 单个字符 '1' (等价) 或 '0' (不等价)。
        
- **主要参数:** n (样本数), vars (变量集)。

---

## 31. **symbolic_math_logic/generate_polynomial_shift_coefficients.py**

- **用途:** 测试模型学习一个抽象的代数变换规则的能力。这个任务需要模型理解多项式展开的内在结构。
    
- **逻辑:** 输入是6个整数（代表一个5次多项式 a5*x^5 + ... + a0 的系数），每个系数用3-bit二进制表示。输出是该多项式进行变量替换 x -> x+1 后的新多项式的6个系数，每个新系数用8-bit二进制表示。脚本的核心是poly_eval_at_shifted函数，它正确地使用了二项式定理来计算新多项式的系数。
    
- **I/O格式:**
    
    - 输入: 6 * 3 = 18位的二进制字符串。
        
    - 输出: 6 * 8 = 48位的二进制字符串。
        
- **主要参数:** max_samples。

---

## 32. **symbolic_math_logic/generate_convolution_2d.py**

- **用途:** 测试模型学习二维卷积（Conv2D）这一基本图像处理操作的能力，并探究其是否能从输入输出对中推断出隐藏的固定规则（即卷积核本身）。
    
- **逻辑:** 脚本固定一个隐藏的3x3二进制卷积核。它生成两类数据集：一类输入包含特征图和卷积核，测试模型直接执行运算的能力；另一类只包含特征图，模型必须通过学习大量样本来将隐藏的卷积核参数化到自己的权重中。
    
- **I/O格式:**
    
    - 输入 (可见): (MAP_SIZE^2 + KERNEL_SIZE^2)位二进制串。 | 输入 (隐藏): MAP_SIZE^2位二进制串。
        
    - 输出: MAP_SIZE^2 * BITS_PER_OUTPUT_ELEMENT长度的二进制多标签向量，代表卷积结果（每个像素点的累加值）。
        
- **主要参数:** MAP_SIZE, KERNEL_SIZE, DATASET_SIZE。

---

## 33. **symbolic_math_logic/generate_simple_block_cipher.py**

- **用途:** 测试模型“破解”或学习一个简单但非平凡的自定义加密算法的能力。该任务代表了一类复杂的、具有高度混沌和雪崩效应的符号变换规则。
    
- **逻辑:** 脚本定义了一个固定的、隐藏的轮密钥(HIDDEN_KEY)和一个名为T-Cipher的简单分组密码算法。它通过对随机明文进行N轮加密来生成密文，构建用于训练的数据对。
    
- **I/O格式:**
    
    - 输入: INPUT_BITS长度的明文二进制字符串。
        
    - 输出: INPUT_BITS长度的密文二进制多标签向量。
        
- **主要参数:** INPUT_BITS, NUM_ROUNDS, DATASET_SIZE。

---

## 34. **symbolic_math_logic/generate_sin_function_float32.py**

- **用途:** 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，使用标准的32位浮点数格式进行输入和输出。
    
- **逻辑:** 脚本的输入是一个浮点数x，使用其标准的IEEE 754 32位二进制表示。输出是sin(x)的计算结果，同样使用其32位二进制表示。
    
- **I/O格式:**
    
    - 输入: 32位二进制多标签向量。
        
    - 输出: 32位二进制多标签向量。
        
- **主要参数:** N (样本数), x_range。

---

## 35. **symbolic_math_logic/generate_sin_function_float64_to_int12_deprecated.py**

- **用途:** 这是对sin函数拟合任务的另一种编码尝试，旨在探索使用更高精度的浮点输入和更低精度的量化二进制输出对学习效果的影响。
    
- **逻辑:** 脚本的输入是一个浮点数x，使用其64位（双精度）的二进制表示。输出是sin(x)的结果，但被线性映射并量化到一个12位的有符号整数空间。
    
- **I/O格式:**
    
    - 输入: 64位二进制多标签向量。
        
    - 输出: 12位二进制多标签向量。
        
- **状态:** (已弃用) 这是一个早期的、有问题的版本，已被更成功的generate_sin_function_float32_to_quantized_int.py取代。

---

## 36. **symbolic_math_logic/generate_sin_function_float32_to_quantized_int.py**

- **用途:** 测试模型拟合连续、周期性、非线性函数（sin(x)）的能力，并探索不同输入/输出编码方案对学习效果的影响。
    
- **逻辑:** 该脚本采用了一种有效的编码策略：
    
    1. **输入:** 一个浮点数x，使用其标准的IEEE 754 32位二进制表示。
        
    2. **输出:** 计算y = sin(x)（值域[-1, 1]），然后将其线性映射并量化到一个24位的有符号整数空间。这种离散化的表示法更适合分类模型学习。
        
- **I/O格式:**
    
    - 输入: 32位二进制多标签向量。
        
    - 输出: 24位二进制多标签向量。
        
- **主要参数:** N (样本数), x_range。

---

## 37. **symbolic_math_logic/generate_multiply_binary_modulo.py**

- **用途:** 作为基础算术实验的一部分，测试模型对截断乘法（或称模乘法）的掌握能力。
    
- **逻辑:** 将两个N-bit整数相乘，然后对结果进行模2^N运算，以确保输出与输入操作数的位数相同。
    
- **I/O格式:**
    
    - 输入: bits * 2长度的二进制字符串。
        
    - 输出: bits长度的二进制多标签向量。
        
- **主要参数:** num_samples, bits。

---

## 38. **symbolic_math_logic/generate_explainable_two_step_calculation.py**

- **用途:** 测试模型输出计算“中间步骤”或“思维链”的能力，是“功能性可解释性”的一个直接验证。
    
- **逻辑:** 输入是三个8-bit二进制数和两个运算符。模型被要求输出一个拼接起来的向量，其中第一部分是第一个运算的中间结果，第二部分是最终结果。这强制模型不仅要算出答案，还要能“回溯”并呈现其计算过程中的一个关键状态。
    
- **I/O格式:**
    
    - 输入: 8*3 (操作数) + 2 (运算符)长度的字符串。
        
    - 输出: 8 (中间结果) + 8 (最终结果)长度的二进制字符串。
        
- **主要参数:** count。

---

## 39. **symbolic_math_logic/generate_min_swaps_for_checkerboard.py**

- **用途:** 解决LeetCode 782题"变为棋盘"([https://leetcode.cn/problems/transform-to-chessboard/](https://leetcode.cn/problems/transform-to-chessboard/)) - 通过任意交换行和列，将一个0/1矩阵变为"棋盘"模式（相邻元素不同）所需的最少交换次数。
    
- **逻辑:** 脚本首先通过对一个完美棋盘进行随机行列交换，来智能地生成一个保证"可解"的输入矩阵。然后，它使用一个复杂的、基于位运算和组合分析的算法来精确计算恢复到棋盘模式所需的最少行列交换总数。如果无法恢复，则返回-1。
    
- **I/O格式:**
    
    - 输入: N*N长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量 (-1映射为0，k次移动映射为k+1)。
        
- **主要参数:** MATRIX_SIZE_N, DATASET_SIZE。

---

## 40. **symbolic_math_logic/generate_min_flips_for_alternating_binary.py**

- **用途:** 测试模型解决一个基于位翻转的字符串优化问题，该问题可以被巧妙地映射为一个滑动窗口问题来求解。
    
- **逻辑:** "美丽字符串"被定义为一个交替的01序列（如'0101...'或'1010...'）。输入是一个任意的二进制字符串，任务是计算最少的翻转次数，使其变为“美丽”。
    
- **I/O格式:**
    
    - 输入: STRING_LENGTH_N长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表最小翻转次数。
        
- **主要参数:** STRING_LENGTH_N, DATASET_SIZE。

---

## 41. **symbolic_math_logic/generate_min_swaps_for_checkerboard_v2.py**

- **用途:** 解决LeetCode 1536题"排布二进制网格的最少交换次数"([https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/](https://leetcode.cn/problems/minimum-swaps-to-arrange-a-binary-grid/)) - 通过交换相邻行，将一个二进制网格变为上三角形式（主对角线以上全为0）所需的最少交换次数。
    
- **逻辑:** 脚本首先随机生成一个二进制矩阵。然后，它使用BFS搜索排列空间，寻找通过相邻行交换将矩阵变为上三角形式的最小交换次数。如果无法完成，则返回-1。
    
- **I/O格式:**
    
    - 输入: N*N长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量 (-1映射为0，k次移动映射为k+1)。
        
- **主要参数:** MATRIX_SIZE_N, DATASET_SIZE。

---

## 42. **symbolic_math_logic/generate_min_prefix_flips.py**

- **用途:** 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。
    
- **逻辑:** 这是一个经典的“前缀翻转”或“灯泡”问题。从左到右遍历该序列，如果当前位置在考虑了之前所有翻转的累积效应后仍然是'0'，则必须“拉动”当前位置的开关（这会翻转从当前位置到末尾的所有位），并计一次操作。
    
- **I/O格式:**
    
    - 输入: STRING_LENGTH_N长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表最小翻转次数。
        
- **主要参数:** STRING_LENGTH_N, DATASET_SIZE。

---

## 43. **symbolic_math_logic/generate_min_flips_for_chunked_binary.py**

- **用途:** 测试模型学习一个基于局部块（chunk）的字符串变换优化问题的能力。
    
- **逻辑:** 输入一个偶数长度的二进制字符串。将其按两位一切分，对于每个2-bit的块，如果两个bit不相同（如'01'或'10'），则需要一次翻转操作才能使其“美丽”（变为'00'或'11'）。任务是计算总共需要的最少翻转次数。
    
- **I/O格式:**
    
    - 输入: INPUT_BITS长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表总翻转次数。
        
- **主要参数:** INPUT_BITS, DATASET_SIZE。

---

## 44. **symbolic_math_logic/generate_largest_island_by_adding_one_cell.py**

- **用途:** 解决一个涉及图遍历和全局优化的算法问题([LeetCode 827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/))。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。
    
- **逻辑:** 脚本首先使用DFS或BFS遍历输入网格，标记所有已存在的岛屿并计算它们的面积。然后，它遍历所有水域格点（'0'），计算如果将该点变为陆地，可以连接哪些相邻岛屿，并由此计算出新形成的总面积。最终，它找到能产生最大面积的最佳位置。
    
- **I/O格式:**
    
    - 输入: N*N长度的二进制字符串。
        
    - 输出: 一个包含output_class（最佳位置的类别标签）和output_area（最大面积的二进制串）的JSON对象。
        
- **主要参数:** NUM_SAMPLES, GRID_SIZE。

---

## 45. **symbolic_math_logic/generate_largest_island_by_adding_one_cell_v2.py**

- **用途:** 解决一个涉及图遍历和全局优化的算法问题([LeetCode 827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/))。模型需要评估所有可能的“填海”位置，并选出能使合并后岛屿面积最大的那一个。
    
- **逻辑:** 脚本首先使用DFS或BFS遍历输入网格，标记所有已存在的岛屿并计算它们的面积。然后，它遍历所有水域格点（'0'），计算如果将该点变为陆地，可以连接哪些相邻岛屿，并由此计算出新形成的总面积。最终，它找到能产生最大面积的最佳位置。
    
- **I/O格式:**
    
    - 输入: N*N长度的二进制字符串。
        
    - 输出: 一个包含output_class（最佳位置的类别标签）和output_area（最大面积的二进制串）的JSON对象。
        
- **主要参数:** NUM_SAMPLES, GRID_SIZE。

---

## 46. **symbolic_math_logic/generate_sat_solver_text.py**

- **用途:** 测试模型解决一个标志性的NP完全问题——布尔可满足性（SAT）问题的能力。
    
- **逻辑:** 随机生成一个由多个子句组成的CNF（合取范式）公式。输入是对这个公式的字符串编码。然后，脚本调用一个外部求解器（pycosat）来判断该公式是否存在一组变量赋值使其为真。脚本会努力确保可满足和不可满足的样本比例为1:1。
    
- **I/O格式:**
    
    - 输入: 代表整个公式的字符串。
        
    - 输出: '1' (可满足) 或 '0' (不可满足)。
        
- **主要参数:** num_vars, num_clauses, num_samples_per_class。

---

## 47. **symbolic_math_logic/generate_sat_solver_compact_text.py**

- **用途:** 这是对 symbolic_math_logic/generate_sat_solver_text.py 的一个变种，采用了不同的输入编码格式来解决同样的3-SAT问题。
    
- **逻辑:** 核心逻辑与前一个脚本相同，都是通过外部求解器（Z3）生成标签并保证数据平衡。主要区别在于输入格式：此版本使用大写字母表示变量的否定（如 a 代表 x1，A 代表 ~x1），这是一种更紧凑的表示法。
    
- **I/O格式:**
    
    - 输入: NUM_CLAUSES * 3长度的字符串，代表整个公式。
        
    - 输出: '1' (可满足) 或 '0' (不可满足)。
        
- **主要参数:** VAR_COUNT, NUM_CLAUSES, NUM_SAMPLES_PER_CLASS。

---

## 48. **generate_binary_mod3_dfa_explain.py**

- **用途:** 这是一个**可解释性**实验，旨在测试模型是否能学会并输出有限状态自动机（DFA）的内部状态转移过程。
    
- **逻辑:** 脚本生成随机的N位二进制数。它不仅计算该数模3的结果，还记录了DFA处理每一位输入时的状态转移轨迹（S0, S1, S2）。这迫使模型不仅要给出最终答案，还要展示它是如何一步步推导出来的。
    
- **I/O格式:**
    
    - 输入: N位二进制字符串。
        
    - 输出: 包含 `final_mod_result` (2 bits) 和 `dfa_state_trace` (N * 2 bits) 的JSON对象。
        
- **主要参数:** NUM_BITS, DATASET_SIZE。

---

## 49. **symbolic_math_logic/generate_add_binary_explainable.py**

- **用途:** 测试模型是否能学会二进制加法的内部机制（进位传播）。这是一个可解释性实验，要求模型不仅输出结果，还要输出每一位的进位状态。
    
- **逻辑:** 模拟逐位二进制加法。输入是两个 N 位二进制数。输出包含两部分：一是每一位的无进位和（XOR结果），二是每一位的进位输入（Carry-in）。这迫使模型显式地表示出进位链。
    
- **I/O格式:**
    
    - 输入: N * 2 长度的二进制字符串。
        
    - 输出: 包含 `output` (2*N位，前N位为当前位结果，后N位为进位) 和 `sum_output` (N+1位最终和) 的JSON对象。
        
- **主要参数:** NUM_BITS, NUM_SAMPLES。

---

## 50. **symbolic_math_logic/generate_add_quaternary_float_encoding.py**

- **用途:** 测试MLP是否有能力解析实数并将其对应成编码的二进制，完成计算。事实证明它可以做到，说明在训练MLP时，可以将间隔一定距离的实数当作纯粹的符号使用。
    
- **逻辑:** 生成两个20位整数的加法任务。输入不直接使用0/1，而是采用“四进制浮点编码”（Quaternary Float Encoding），将每两位二进制（00, 01, 10, 11）映射为特定的浮点数（如 0.0, 5.5, 1.3, 10.0）。模型必须学会将这些连续的浮点数值映射回离散的二进制逻辑，并执行加法运算。
    
- **I/O格式:**
    
    - 输入: NUM_BITS / 2 长度的浮点数列表。
        
    - 输出: NUM_BITS + 1 长度的二进制多标签向量。
        
- **主要参数:** NUM_BITS_PER_ADDEND, DATASET_SIZE, QUATERNARY_MAP。

---

## 51. **symbolic_math_logic/generate_add_binary_masked_output.py**

- **用途:** 测试模型在"条件输出"任务上的能力。模型不仅需要计算加法，还需要根据输入的掩码（Mask）动态决定只输出结果的哪几位。这类似于Attention机制，需要模型学会根据指令（Mask）来聚焦和路由信息。
    
- **逻辑:** 生成两个N位整数A和B，以及一个随机的二进制掩码Mask。模型需要计算 Sum = A + B，但只输出 Mask中为'1'的位置对应的Sum的值。例如，如果Sum是`10110`，Mask是`01010`，则只需输出`0`（第2位）和`1`（第4位）。
    
- **I/O格式:**
    
    - 输入: A(N-bit) + B(N-bit) + Mask(N+1-bit)。
        
    - 输出: 长度等于Mask中'1'的个数的二进制多标签向量。
        
- **主要参数:** BITS_PER_ADDEND, MASK_VISIBLE_BITS, NUM_SAMPLES。

---

## 52. **symbolic_math_logic/generate_multitask_prefixed_ca110.py**

- **用途:** 生成"前置CA演化 + 4个后置任务"的多任务学习数据集，用于测试模型在单次前向传播中执行共享表示提取和多个独立下游任务的能力。这是对 `generate_multitask_alu.py` 的重要扩展，增加了共享前置变换层的概念。

- **实验验证（论文第八幕）:** 该脚本用于验证论文第八幕末尾提出的重要假设：**多任务学习是否会在认知经济学压力下促使中间表示的自发涌现，从而加速训练？** 实验结果证实了这一点——同时训练多个任务比单独训练每一个任务都要快，这几乎可以确定是因为神经网络在多任务压力下更快地接近了可复用的中间表示层，而非各自独立地学习。这为论文中关于"多任务促进抽象表示涌现"的核心论点提供了有力的实证支持。

- **逻辑:** 该脚本采用了一个独特的"共享前置 + 并行后置"架构：
    1. **共享前置任务：** 输入的30位二进制字符串首先被送入元胞自动机（Rule 110）进行4层演化，得到一个30位的中间状态。这个中间状态作为所有后置任务的共享输入。
    2. **并行后置任务：** 中间状态被同时送入4个独立的求解器：
        - **Task A (二进制加法)：** 将30位拆分为两个15位数，计算它们的和（16位输出）。
        - **Task B (接雨水)：** 将30位解释为10个柱子的高度（每柱3位），计算每个柱子的接水量（30位输出）。
        - **Task C (Mod 3 DFA)：** 对30位二进制数计算 mod 3，并输出完整的DFA状态转移轨迹（60位输出）。
        - **Task D (CA Rule 30)：** 对30位状态应用Rule 30，演化3层（30位输出）。
    
    此设计迫使模型必须先学习一个共享的特征提取器（前置CA），然后将学到的表示分发给多个下游任务头。这对于研究多任务学习中的知识迁移、共享表示的形成机制以及不同任务间的相互影响（正迁移 vs 负迁移）具有重要价值。

- **I/O格式:**
    - 输入: 30位二进制字符串（原始输入）。
    - 输出: 一个包含6个字段的JSON对象：
        - `input`: 原始输入字符串。
        - `inter`: 中间状态（前置CA演化后的30位状态，列表格式）。
        - `output_add`: Task A的输出（16位列表）。
        - `output_rain`: Task B的输出（30位列表）。
        - `output_mod3`: Task C的输出（60位列表）。
        - `output_ca30`: Task D的输出（30位列表）。

- **主要参数:** 
    - `INPUT_LEN` (30): 原始输入长度。
    - `PRE_CA_RULE_NUMBER` (110): 前置CA规则。
    - `PRE_CA_EVOLUTION_LAYERS` (4): 前置CA演化层数。
    - `DATASET_SIZE` (500000): 数据集大小。

- **训练建议:** 配合 `train_mlp_final.ipynb`（多任务/多头MLP训练脚本）使用，可用于研究多任务学习的各个方面。训练时可以选择：
    - 单任务模式：只训练某一个后置任务，观察模型如何利用共享前置表示。
    - 多任务模式：同时训练所有4个任务，研究任务间的迁移效应和资源竞争。

---

# B: 算法学习 (Algorithm Learning)

## 1. **algorithms/generate_sort_integers.py**

- **用途:** 测试模型执行基本排序算法的能力，这是一个非局部的、需要对输入元素进行比较和重排的经典算法任务。
    
- **逻辑:** 输入是NUM_ITEMS个NUM_BITS_PER_ITEM-bit的无序整数拼接而成的二进制串。输出是将这些数按升序排列后，重新拼接成的二进制串。脚本确保了输入中的所有数都是唯一的。
    
- **I/O格式:**
    
    - 输入: NUM_ITEMS * NUM_BITS_PER_ITEM长度的二进制字符串。
        
    - 输出: NUM_ITEMS * NUM_BITS_PER_ITEM长度的二进制多标签向量。
        
- **主要参数:** NUM_ITEMS, NUM_BITS_PER_ITEM, DATASET_SIZE。

---

## 2. **algorithms/generate_edit_distance.py**

- **用途:** 测试模型学习解决动态规划问题的能力。编辑距离是一个典型的DP问题，需要模型在概念上构建一个二维的求解矩阵。 [LeetCode 72. Edit Distance](https://leetcode.com/problems/edit-distance/description/)
    
- **逻辑:** 输入是两个等长的二进制字符串s1和s2的拼接。输出是它们的最小编辑距离（允许插入、删除、替换操作）的二进制表示。
    
- **I/O格式:**
    
    - 输入: NUM_BITS_PER_STRING * 2长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量。
        
- **主要参数:** NUM_BITS_PER_STRING, DATASET_SIZE。

---

## 3. **algorithms/generate_edit_distance_explainable.py**

- **用途:** 这是“功能性可解释性”的一个核心实验。它要求模型不仅给出最终答案（编辑距离），还要输出达成答案的完整“思维链”（编辑过程）。 [LeetCode 72. Edit Distance (Explainable / Path Construction Version)](https://leetcode.com/problems/edit-distance/description/)
    
- **逻辑:** 输入是两个字符串s1和s2。输出是一个长向量，它由max_steps个“状态帧”拼接而成。每个状态帧包含两部分：编辑过程中一个中间字符串的二进制表示，以及一个掩码（mask），用于指示该字符串的有效长度。这使得模型必须学会模拟从s1到s2的逐步转换。脚本通过一个巧妙的机制，只保留那些最优编辑路径唯一的样本，以确保标签的无歧义性。
    
- **I/O格式:**
    
    - 输入: str_len * 2长度的二进制字符串。
        
    - 输出: max_edits * str_len * 2长度的二进制多标签向量。
        
- **主要参数:** num_samples, str_len, max_edits。

---

## 4. **algorithms/generate_maze_random_walls.py**

- **用途:** 测试模型在随机生成的“多孔”迷宫中的基础寻路能力。
    
- **逻辑:** 脚本通过在网格上随机放置墙壁来生成迷宫，这种方法生成的迷宫通常路径较短，连通性高，结构相对简单。然后，对于所有可通行的点，它使用反向BFS从固定的终点计算出到达该点的最短路径。模型的任务是，给定一个包含起点和终点的迷宫布局，预测从起点出发的第一步最优方向。
    
- **I/O格式:**
    
    - 输入: H * W长度的字符串，代表迷宫布局。
        
    - 输出: 4分类的类别标签（上/下/左/右）。
        
- **主要参数:** MAZE_HEIGHT, MAZE_WIDTH, TARGET_NUM_SAMPLES。

---

## 5. **algorithms/generate_maze_dense.py**

- **用途:** 测试模型在复杂的、类似人类设计的“稠密”迷宫中进行路径规划的能力，这比随机墙壁迷宫更具挑战性。
    
- **逻辑:** 脚本首先使用一个专门的迷宫生成算法（如递归分割法）创建一个具有挑战性的、连通的稠密迷宫，其特点是通道长而蜿蜒。然后，与前一个脚本类似，它使用反向BFS为所有可达点计算最优策略。
    
- **I/O格式:**
    
    - 输入: H * W长度的字符串，代表迷宫布局。
        
    - 输出: 4分类的类别标签。
        
- **主要参数:** MAZE_HEIGHT, MAZE_WIDTH, TARGET_NUM_SAMPLES。

---

## 6. **algorithms/generate_blocks_world_arbitrary_goal.py**

- **用途:** 解决经典的“积木世界”（Blocks World）规划问题。该问题作为衡量大语言模型推理能力的标准任务之一，在苹果公司发布的著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中被重点讨论，作为评估模型在状态空间搜索和规划能力上的基准测试。该研究通过积木问题、过河问题、汉诺塔等一系列经典规划任务，系统性地揭示了当前大语言模型在精确符号推理和状态空间规划上的根本局限性。本脚本精确实现了论文中的"积木世界"通用版本，允许指定任意的初始状态和终止状态，作为验证神经网络在复杂规划问题上推理能力的核心对照实验。
    
- **逻辑:** 脚本为每个样本随机生成一个初始状态和一个目标状态。然后，它使用广度优先搜索（BFS）来找到从初始状态到目标状态的最短动作序列。模型的任务是预测这个序列中的第一步最优动作。
    
- **I/O格式:**
    
    - 输入: 初始状态和目标状态的二进制编码。
        
    - 输出: 6分类的类别标签，代表最优动作。
        
- **主要参数:** BLOCKS_N (积木数量)。

---

## 7. **algorithms/generate_blocks_world_fixed_goal.py**

- **用途:** 这是对“积木世界”任务的简化版本，通过固定目标状态（所有积木有序地堆叠在第一个柱子上），旨在测试模型在目标明确、状态空间更结构化的情况下的学习能力。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究指出，即使对于这样目标状态明确的规划问题，大语言模型在状态空间搜索和最优策略学习方面仍然存在显著困难。本脚本实现了论文中的积木世界简化版本，通过固定目标状态来降低任务复杂度，作为研究目标明确性对模型推理性能影响的对照实验。
    
- **逻辑:** 脚本设定一个固定的目标状态（所有积木有序地在第一个柱子上）。然后，它通过从目标状态开始进行**反向**广度优先搜索（BFS），高效地遍历所有可达状态，并计算出每个状态到目标的最优策略。
    
- **I/O格式:**
    
    - 输入: 积木的初始状态编码。
        
    - 输出: 6分类的类别标签，代表最优动作。
        
- **主要参数:** BLOCKS_N。

---

## 8. **algorithms/generate_blocks_world_fixed_goal_multilabel.py**

- **用途:** 进一步改进“积木世界”任务，通过允许多个最优解，测试模型处理多标签分类问题的能力，更真实地反映了规划问题中可能存在的等效最优路径。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究强调，现实世界中的规划问题往往存在多个等价最优解，这对模型的多义性推理能力提出了更高要求。本脚本在固定目标版本基础上进一步改进，为每个状态找到所有能导向最优路径的动作，生成多热（multi-hot）编码输出，作为研究神经网络处理多最优解规划问题的对照实验。
    
- **逻辑:** 继承了上一个脚本固定目标和反向搜索的逻辑。关键的改进在于输出格式：对于每个状态，脚本找到**所有**能使其离目标更近一步的最优动作，并生成一个多热（multi-hot）编码的输出向量。
    
- **I/O格式:**
    
    - 输入: 积木的初始状态编码。
        
    - 输出: NUM_ACTIONS长度的二进制多标签向量。
        
- **主要参数:** BLOCKS_N。

---

## 9. **algorithms/generate_blocks_world_fixed_goal_multilabel_fixed_format.py**

- **用途:** 这是“积木世界”任务的最终优化版本，通过改进输入表示法，旨在为模型提供一个更清晰、更结构化的学习目标。该问题同样源自苹果公司论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中的基准测试任务。该研究指出，输入表示法对模型的学习效率和最终性能有着至关重要的影响。本脚本在多标签版本基础上进一步优化，采用固定槽位（fixed-slot）表示法替代可变长度输入，消除了序列化带来的复杂性，为Transformer等架构提供了更友好的结构化输入。这使其成为研究输入表示法对神经符号推理性能影响的对照实验。
    
- **逻辑:** 核心逻辑与上一个脚本（多标签输出、固定目标、反向搜索）相同。关键改进在于输入格式：不再使用分隔符，而是为每个柱子分配固定数量的“槽位”来表示状态。这种固定长度的表示法消除了可变长度输入的复杂性，对Transformer等模型更为友好。
    
- **I/O格式:**
    
    - 输入: NUM_BLOCKS * NUM_STACKS长度的字符串，0代表空，1-N代表积木。
        
    - 输出: NUM_ACTIONS长度的二进制多标签向量。
        
- **主要参数:** BLOCKS_N, NUM_STACKS。

---

## 10. **algorithms/generate_checkers_jump_1d.py**

- **用途:** 解决一维空间中的棋子交换规划问题。该问题作为衡量大语言模型推理能力的标准任务之一，在苹果公司发布的著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity" 中被重点讨论，作为评估模型在状态空间搜索和规划能力上的基准测试。该研究通过积木问题、过河问题、汉诺塔等一系列经典规划任务，系统性地揭示了当前大语言模型在精确符号推理和状态空间规划上的根本局限性。本脚本精确实现了论文中的"跳棋交换"通用版本，作为验证神经网络在复杂规划问题上推理能力的核心对照实验。
    
- **逻辑:** 脚本在一个一维棋盘上，模拟两种棋子（'R'和'B'）互相穿过的过程。它使用高效的反向广度优先搜索（BFS）从目标状态开始，逆向遍历整个状态空间，从而为每一个可达的状态都计算出其唯一的最优下一步。
    
- **I/O格式:**
    
    - 输入: 2*N+1长度的整数序列，代表棋盘状态。
        
    - 输出: 单个整数，代表要移动的棋子的**位置索引**。
        
- **主要参数:** CHECKERS_N (每种颜色棋子数)。

---

## 11. **algorithms/generate_river_crossing_puzzle.py**

- **用途:** 解决一个经典的约束满足和状态空间搜索问题——“N对伴侣过河”。该问题要求在满足“任何女性不能在没有其伴侣在场的情况下，与其他男性共处”的约束下，将所有人运到对岸。该任务源自苹果公司的一篇著名论文 [15] "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity"，该研究通过过河问题、汉诺塔、跳棋等基准测试，揭示了大型语言模型在某些类型推理任务上的根本局限性。本脚本精确复现了论文中的"N对伴侣过河"问题，作为验证神经网络符号推理能力的对照实验。
    
- **逻辑:** 脚本将每个状态定义为“在左岸的人员集合”和“船的位置”。它通过高效的反向广度优先搜索（BFS）从目标状态（所有人都在右岸）开始搜索，构建出覆盖所有可达状态的最优策略图。输出是一个多标签向量，指明在当前状态下，哪些人应该一起上船以执行最优移动。
    
- **I/O格式:**
    
    - 输入: 2*N+1长度的二进制字符串（N个客户C，N个代理人A，1个boat的位置）。
        
    - 输出: 2*N长度的多标签二进制向量，表示每个人是否上船。
        
- **主要参数:** PAIRS_N (伴侣对数), BOAT_CAPACITY_K。

---

## 12. **algorithms/generate_trapping_rain_water_aggregate.py**

- **用途:** 这是解决“接雨水”算法问题的初步尝试，旨在测试模型学习一个聚合输出（而非解耦输出）的能力。实验结果表明，要求模型直接输出总和值（一个单一的聚合数字）比输出每个位置的详细信息要困难得多。**这成为一个关键的对比实验，证明了输出格式设计对模型学习效率的系统性影响**。对应 LeetCode 题目：[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)。该脚本精确实现了LeetCode原题的聚合输出格式（只输出总量），与`generate_trapping_rain_water_decoupled.py`（解耦输出，输出每个位置的水量）形成鲜明对照，用于研究输出表示对神经网络学习难度的影响，验证了论文中"解耦加速收敛"的核心发现。
    
- **逻辑:** 输入是一维的高度图。脚本计算在该高度图上可以接到的总雨水量，并将这个单一的整数值作为输出。
    
- **I/O格式:**
    
    - 输入: N * K位二进制串，代表N个柱子的高度。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表总雨水量。
        
- **主要参数:** NUM_COLUMNS_N, BITS_PER_HEIGHT。

---

## 13. **algorithms/generate_trapping_rain_water_decoupled.py**

- **用途:** 解决经典的“接雨水”算法问题（LeetCode Hard [#42](https://leetcode.com/problems/trapping-rain-water/)）。这个任务的成功展示了模型学习需要全局信息（如全局最高点）的复杂算法的能力，并通过**问题解耦**的思想，证明了输出格式设计对模型学习效率的巨大影响。
    
- **逻辑:** 脚本通过一个关键的问题解耦洞察来设计输出。它没有让模型直接预测一个单一的聚合值（总雨水量），而是要求模型预测一个与输入结构同构的序列，其中每个元素代表对应柱子上所接的雨水量。这一改变极大地简化了学习任务，使得模型能够成功收敛。
    
- **I/O格式:**
    
    - 输入: N * K位二进制串，代表N个柱子的高度。
        
    - 输出: N * K位二进制多标签向量，代表N个柱子各自的接水量。
        
- **主要参数:** NUM_COLUMNS_N, BITS_PER_HEIGHT。

---

## 14. **algorithms/generate_trapping_rain_water_2d.py**

- **用途:** 作为一维“接雨水”问题的扩展，解决二维版本的“接雨水”问题。该任务要求模型理解二维空间中的“包围”和“边界”概念，是一个更复杂的全局信息处理挑战。对应 LeetCode 题目：[407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)（困难题）。
    
- **逻辑:** 同样采用了问题解耦的思想。输入是一个二维的高度图（矩阵），输出是一个同样大小的矩阵，其中每个单元格的值代表该位置可以接到的雨水量。求解器通过从边界向内进行类似BFS的“灌水”操作来确定每个点的水位。
    
- **I/O格式:**
    
    - 输入: NMK位二进制串，代表N*M网格的高度。
        
    - 输出: NMK位二进制多标签向量，代表每个单元格的接水量。
        
- **主要参数:** GRID_N, GRID_M, BITS_PER_HEIGHT。

---

## 15. **algorithms/generate_skyline_max_height_aggregate.py**

- **用途:** 这是解决“天际线”问题的初步尝试，要求模型从所有建筑的最终高度中，只预测出那个最高的高度值。此任务用于对比聚合输出和解耦输出的学习难度。对应 LeetCode 题目：[1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/)（聚合输出版本）。
    
- **逻辑:** 输入是一系列建筑的高度限制。在满足相邻建筑高度差不超过1的约束下，脚本通过动态规划计算出每栋建筑可能达到的最大高度，然后找出所有建筑中的最大值作为输出。
    
- **I/O格式:**
    
    - 输入: n * bit_count长度的二进制字符串，代表每栋楼的高度限制。
        
    - 输出: bit_count长度的二进制多标签向量，代表全局最高高度。
        
- **主要参数:** NUM_SAMPLES, FIXED_N (建筑数量), MAX_HEIGHT。

---

## 16. **algorithms/generate_skyline_all_heights_decoupled.py**

- **用途:** 测试模型解决一个带有一维空间约束的全局优化问题的能力。问题原型是LeetCode [1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/)（解耦输出版本）。该题目要求：给定N个建筑的位置和限高，相邻建筑高度差不超过1，求每个建筑能达到的最大高度。通过解耦输出，要求模型预测每一栋建筑的高度，而非仅仅是最大值，作为研究输出格式设计对模型学习效率影响的对照实验。
    
- **逻辑:** 输入是一系列建筑的高度限制。规则是在满足所有限制的前提下，相邻建筑的高度差不能超过1。该脚本通过一个高效的双向动态规划算法来求解在这些约束下每栋建筑可能达到的最大高度。输出是所有建筑最终高度组成的一个序列。
    
- **I/O格式:**
    
    - 输入: n * bit_count长度的二进制字符串，代表每栋楼的初始高度限制。
        
    - 输出: n * bit_count长度的二进制多标签向量，代表每栋楼的最终高度。
        
- **主要参数:** NUM_SAMPLES, FIXED_N (建筑数量), MAX_HEIGHT。

---

## 17. **algorithms/generate_hanoi_tower_path_strategy_sep_format.py**

- **用途:** 作为对比组A，此脚本用于生成汉诺塔最优路径策略数据集，采用**分隔符+二进制编码**的输入格式。
    
- **逻辑:** 标准递归求解器生成最优解路径。输入是对状态的Token序列化（如 `[3, 2, 1, SEP, SEP]`），随后每个Token被转换为固定位宽的**二进制字符串**并拼接。这种格式模拟了非结构化的线性输入。
    
- **I/O格式:**
    
    - 输入: 长二进制字符串（由Token序列编码而来）。
        
    - 输出: 6分类的动作标签（代表源柱和目标柱）。
        
- **主要参数:** HANOI_N。

---

## 18. **algorithms/generate_hanoi_tower_path_strategy_fixed_format.py**

- **用途:** 作为对比组B，此脚本同样生成汉诺塔最优路径策略数据集，但采用**结构化的固定槽位**输入格式（重命名自原来的`global`脚本以反映真实逻辑），用于验证数据表示对学习效率的影响。
    
- **逻辑:** 同样基于递归求解器生成最优路径。区别在于输入编码：将3个柱子展开为固定长度的数组，每个位置对应一个物理槽位，填入盘子ID或0（空位）。这种格式保留了空间结构信息。
    
- **I/O格式:**
    
    - 输入: 固定长度的数字字符串（如 `"321000400"`）。
        
    - 输出: 6分类的动作标签。
        
- **主要参数:** HANOI_N, dataset_size。

---

## 19. **algorithms/generate_hanoi_tower_compare_formats.py**

- **用途:** 这是一个对比实验脚本，它为同一个汉诺塔问题生成两种不同的输入格式（分隔符 vs. 固定槽位），用于系统性地评估不同数据表示法对模型学习递归策略的影响。
    
- **逻辑:** 脚本同时生成两份数据集，一份采用sep格式，另一份采用固定槽位格式。两份数据集都只包含最优路径上的状态，并要求模型预测下一步的最优动作。
    
- **I/O格式:**
    
    - 输入: sep格式 或 固定槽位格式。
        
    - 输出: 6分类的类别标签。
        
- **主要参数:** HANOI_N, DATASET_SIZE。

---

## 20. **algorithms/generate_hanoi_tower_compare_formats_and_strategies.py**

- **用途:** 这是一个更全面的汉诺塔对比实验脚本。它不仅生成两种输入格式，还生成两种不同的数据集：一种只包含最优路径上的状态（“路径策略”），另一种包含所有可达状态（“全局策略”），用于探究模型在学习局部最优路径和全局最优策略上的能力差异。
    
- **逻辑:** 脚本总共生成四个数据集（2种格式 x 2种策略）。实验结果表明，模型能轻易学会“路径策略”，但在学习“全局策略”时遇到困难，这揭示了模型在处理递归和状态空间爆炸问题上的潜在局限性。
    
- **I/O格式:**
    
    - 输入: sep格式 或 固定槽位格式。
        
    - 输出: 6分类的类别标签。
        
- **主要参数:** HANOI_N, DATASET_SIZE。

---

## 21. **algorithms/generate_hanoi_tower_build_full_state_graph.py**

- **用途:** 这是一个“汉诺塔问题”研究的集大成者，旨在通过多种不同的数据表示和采样策略，深度剖析模型对递归结构的理解能力。它是一个自给自足的数据工厂。该研究对应于论文中引用的 Apple Research 相关工作，重点在于通过完整状态图分析模型对递归结构的掌握。
    
- **逻辑:** 此脚本的核心是一个非凡的实现：它没有使用传统的递归求解器，而是通过一个精巧的、基于汉诺塔图的分形和自相似性的数学结构，直接在内存中构建了汉诺塔问题拥有3^N个状态的完整“状态-动作”图。这使得后续可以从这个完整的知识库中进行任意的、高效的数据采样。
    
- **I/O格式:**
    
    - 输出: 多个.jsonl文件（如hanoi_n10_path_slots_train_all.jsonl），包含输入状态（state_to_slots_B格式）和输出动作（0-5的整数标签）。
        
- **主要参数:** HANOI_N。

---

## 22. **algorithms/generate_hanoi_tower_sample_from_state_graph.py**

- **用途:** 这是一个后处理和采样脚本，它利用generate_hanoi_tower_build_full_state_graph.py生成的完整知识库，来精确地提取特定类型的训练数据子集，例如“扭曲路径”（twisted path）或“最难部分”，用于进行更精细的消融实验。
    
- **逻辑:** 脚本首先加载由_mine脚本生成的完整状态图。然后，它可以根据用户指定的起始状态和结束状态，精确地提取出连接这两点之间的完整最优路径，并将其保存为可供训练的.jsonl文件。
    
- **I/O格式:**
    
    - 输入: `all_states_n{N}.json`文件（由 `generate_hanoi_tower_build_full_state_graph.py` 生成）。
        
    - 输出: .jsonl格式的训练数据。
        
- **主要参数:** HANOI_N, start_idx, end_idx。

---

## 23. **algorithms/generate_sokoban_planning_astar.py**

- **用途:** 解决经典的“推箱子”（Sokoban）规划问题。这个任务比简单的路径规划更难，因为它涉及到改变环境状态（箱子位置），状态空间巨大。
    
- **逻辑:** 脚本首先随机生成一个包含墙壁、玩家、单个箱子和单个目标的迷宫布局。然后，它使用高效的A*搜索算法来找到将箱子推到目标位置的最优动作序列。模型的任务是，给定一个局面，预测玩家的下一个最优动作（上/下/左/右），这个动作可能只是移动玩家，也可能是推动箱子。
    
- **I/O格式:**
    
    - 输入: M * N长度的字符串，代表推箱子布局。
        
    - 输出: 4分类的类别标签。
        
- **主要参数:** M_DIMENSION, N_DIMENSION, NUM_SAMPLES。

---

## 24. **algorithms/generate_sokoban_planning_full.py**

- **用途:** 解决经典的“推箱子”（Sokoban）规划问题。这是一个高难度的AI任务，因为它涉及到在一个巨大的状态空间中进行搜索，并且动作会改变环境的状态。
    
- **逻辑:** 这是一个非常成熟的数据集生成器。
    
    1. **智能生成:** 它通过随机放置墙壁和从目标反向随机游走的方式生成一个既随机又可能有趣的谜题布局。
        
    2. A 求解:* 使用高效的A* 搜索算法，以曼哈顿距离为启发函数，来计算从初始状态到目标状态的最短推箱子路径。
        
    3. **最优策略提取:** 对于所有位于最优路径上的状态，它计算出所有能够导向下一步最优状态的动作，并生成一个多标签（multi-hot）的输出。
        
    4. **质量控制:** 它包含难度过滤器（只保留特定步数范围内的解），并进行全局去重和洗牌。
        
- **I/O格式:**
    
    - 输入: (M-2)*(N-2) 长度的字符串，代表去除了边界墙的推箱子布局。
        
    - 输出: 4位多标签二进制向量，代表上/下/左/右四个方向是否为最优动作。
        
- **主要参数:** M_DIMENSION, N_DIMENSION, NUM_SAMPLES, MIN/MAX_DIFFICULTY。

---

## 25. **algorithms/generate_sokoban_planning_claude_deprecated.py**

- **用途:** (已弃用) 这是一个早期的、逻辑更复杂的尝试，但未能稳定地生成高质量数据集，已被更可靠的generate_sokoban_planning_full.py取代。
    
- **逻辑:** 该脚本是解决推箱子问题的一个早期尝试版本。
    
- **状态:** **已弃用**。

---

## 26. **algorithms/generate_matrix_flip_strategy.py**

- **用途:** 解决一个矩阵优化的经典问题（Score After Flipping Matrix）。此版本旨在测试模型能否学习到一个"策略"而非最终结果。对应 LeetCode 题目：[861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)

- **逻辑:** 对于一个给定的M x N二进制矩阵，可以通过翻转任意行或任意列来改变其内容。任务是找到一个翻转策略，使得将每行解释为二进制数后的总和达到最大。该脚本使用一个高效的贪心算法来找到最优策略：首先确保第一列全为1（通过翻转首位为0的行），然后遍历其余列，如果该列的0多于1，则翻转该列以最大化每一位的贡献。

- **I/O格式:**
    
    - 输入: M*N长度的二进制字符串。
        
    - 输出: M+N长度的二进制多标签向量，代表行和列的翻转掩码。
        
- **主要参数:** MATRIX_M, MATRIX_N, DATASET_SIZE。

---

## 27. **algorithms/generate_matrix_flip_max_score.py**

- **用途:** 测试模型学习一个矩阵优化问题的能力，该问题需要通过两步贪心策略（先行翻转，后列翻转）来达到全局最优。该版本要求模型直接输出最终的聚合结果（分数）。对应 LeetCode 题目：[861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/)
    
- **逻辑:** 脚本实现了一个高效的贪心算法来找到最大分数。第一步：遍历每一行，如果该行的最高位（最左边）是0，则翻转该行。第二步：遍历每一列，如果该列中0的数量多于1，则翻转该列。最终将得到的矩阵按二进制加权求和，得到最大分数。
    
- **I/O格式:**
    
    - 输入: MATRIX_M * MATRIX_N 长度的二进制字符串。
        
    - 输出: OUTPUT_BITS 长度的二进制多标签向量，代表最大得分。
        
- **主要参数:** MATRIX_M, MATRIX_N, DATASET_SIZE。

---

## 28. **algorithms/generate_min_k_bit_flips.py**

- **用途:** 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力，并且测试其能否将输入的一部分（k）作为“参数”来指导对另一部分（nums）的处理。对应 LeetCode 题目：[995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)
    
- **逻辑:** 解决了经典的“K连续位翻转”问题（LeetCode 995）。它从左到右遍历数组，使用一个高效的差分技巧来跟踪之前翻转的累积效应。如果当前位置的值在累积效应下仍为'0'，则必须执行一次新的翻转，并更新差分数组。
    
- **I/O格式:**
    
    - 输入: NUMS_LENGTH_N + K_BITS 长度的二进制字符串 (数据+参数k)。
        
    - 输出: OUTPUT_BITS 长度的二进制多标签向量 (代表最少翻转次数，0代表无解)。
        
- **主要参数:** NUMS_LENGTH_N, K_MAX_N, DATASET_SIZE。

---

## 29. **algorithms/generate_min_k_bit_flips_fixed_k.py**

- **用途:** 测试模型学习一个依赖于历史状态的、顺序处理的贪心算法的能力。此版本中，环境参数（k=2）是固定的、隐藏的，模型必须从数据中隐式学习。这是一个对照实验，用于对比k可变的版本。对应 LeetCode 题目：[995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)（k固定版）
    
- **逻辑:** 与generate_min_k_bit_flips.py逻辑相同，但翻转窗口的长度k被**硬编码为2**（代码中固定`k=2`），并且不在输入中提供。这种设计大幅降低了任务复杂度：模型无需解析动态变化的k参数，只需学习单一、固定的窗口大小下的确定性算法。
        
- **实验设计说明:** 根据初步观察，当k在2~K_MAX_N范围内随机可变时，同时学习解析k参数和适应不同k值的算法行为会显著增加收敛难度。而将k固定为单一值后，模型训练速度和稳定性都得到提升。这个实验旨在验证这一假设。**注：k可变版本的可训练性验证尚未完成，需进一步实验确认。**
    
- **I/O格式:**
    
    - 输入: NUMS_LENGTH_N 长度的二进制字符串 (仅数据，不包含k参数)。
        
    - 输出: OUTPUT_BITS 长度的二进制多标签向量 (代表最少翻转次数，0代表无解)。
        
- **主要参数:** NUMS_LENGTH_N, DATASET_SIZE。

---

## 30. **algorithms/generate_special_binary_string_recursion.py**

- **用途:** 测试模型学习一个递归定义的字符串变换规则的能力。该问题对应 LeetCode 题目：[761. Special Binary String](https://leetcode.com/problems/special-binary-string/)（困难题）。题目要求：一个特殊的二进制序列具有相等的0和1数量，且任何前缀中1不少于0；可以通过交换相邻的特殊子串来得到字典序最大的结果。本脚本精确实现了该问题的递归分解和字典序排序算法，作为测试神经网络学习复杂递归规则能力的基准实验。
    
- **逻辑:** “特殊二进制串”的性质与合法括号序列类似（1代表'('，0代表')'）。算法的核心思想是，任何特殊串都可以被分解为1 + A + 0 + B的形式，其中A和B也是（可能为空的）特殊串。该脚本通过递归地找到所有最外层的特殊子串，对它们各自进行最优变换，然后将得到的结果按字典序降序排列后拼接起来，得到最终答案。
    
- **I/O格式:**
    
    - 输入: STRING_LENGTH_N长度的特殊二进制字符串。
        
    - 输出: STRING_LENGTH_N长度的二进制多标签向量，代表字典序最大的结果。
        
- **主要参数:** STRING_LENGTH_N, DATASET_SIZE。

---

## 31. **algorithms/generate_count_connected_components.py**

- **用途:** 测试模型对图结构的基本理解，特别是“连通性”这一核心概念 (对应 LeetCode 323. [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/)，注：此为会员题)。
    
- **逻辑:** 脚本随机生成一个N x N的邻接矩阵来表示一个无向图。然后，它使用广度优先搜索（BFS）或深度优先搜索（DFS）来遍历图，计算其中独立的连通分量的总数。
    
- **I/O格式:**
    
    - 输入: N*N长度的二进制字符串（邻接矩阵）。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表连通分量数。
        
- **主要参数:** GRAPH_SIZE_N, EDGE_PROBABILITY (控制图的稀疏度), DATASET_SIZE。

---

## 32. **algorithms/generate_check_graph_connectivity.py**

- **用途:** 这是对模型图论基础能力的又一个核心测试，任务是判断图中任意两点之间是否存在一条路径 (对应 LeetCode 1971. [Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/))。
    
- **逻辑:** 脚本随机生成一个图的邻接矩阵，并随机选择两个节点作为起点和终点。它利用一个标准的图算法库来确定这两个点是否位于同一个连通分量中。
    
- **I/O格式:**
    
    - 输入: size*size (邻接矩阵) + ; + start_node_char + end_node_char格式的字符串。
        
    - 输出: [1] (可连通)或 [0] (不连通)。
        
- **主要参数:** num_samples, size (图的节点数)。

---

## 33. **algorithms/generate_minimize_malware_spread.py**

- **用途:** 解决一个基于图论的病毒传播优化问题（LeetCode Hard "Minimize Malware Spread"）。模型需要理解图的连通性，并评估移除不同节点对全局传播的影响。本脚本提供两种输出格式，用于对比哪种更容易学习。对应 LeetCode 题目：[924. Minimize Malware Spread](https://leetcode.com/problems/minimize-malware-spread/)
    
- **逻辑:** 输入是一个图的邻接矩阵和一组初始感染节点。任务是只移除其中**一个**初始感染节点，使得最终被病毒感染的总节点数最少。脚本通过暴力模拟移除每一个初始节点后的传播情况，并比较结果，来找到最优的移除目标集合。
    
- **I/O格式:**
    
    - 输入: (N*N) (邻接矩阵) + N (初始感染节点掩码)长度的二进制字符串。
        
    - 输出（双格式）:
        - `output_simple`: 1位二进制标签，表示第一个初始感染节点是否是最优解之一（简化任务，降低学习难度）。
        - `output_full`: N位二进制向量，表示每个节点是否属于最优移除集合（完整任务，N位多标签分类）。
        
- **主要参数:** GRAPH_SIZE_N, NUM_INITIAL, DATASET_SIZE。

---

## 34. **algorithms/generate_count_islands_1d.py**

- **用途:** 测试模型在一维序列上进行模式识别和计数的能力。
    
- **逻辑:** 输入是一个一维的二进制字符串。任务是计算其中由'0'分隔开的连续'1'的块（岛屿）的数量。例如，在0110100111中，有3个岛屿。
    
- **I/O格式:**
    
    - 输入: NUM_INPUT_BITS长度的二进制字符串。
        
    - 输出: NUM_OUTPUT_BITS长度的二进制多标签向量，代表岛屿数量。
        
- **主要参数:** NUM_INPUT_BITS, DATASET_SIZE。

---

## 35. **algorithms/generate_find_articulation_points.py**

- **用途:** 测试模型识别图的“割点”（Articulation Point）或“桥”（Bridge）的能力，这是一个图论中的重要概念。 [LeetCode 1568. Minimum Number of Days to Disconnect Island](https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/)
    
- **逻辑:** 输入是一个由'1'（陆地）和'0'（水）组成的二维网格。任务本质是找到最少移除多少个'1'能使原来的单一连通块（岛屿）变得不连通。脚本通过暴力尝试（移除1个点、移除2个点）来找到解。输出被设计为最终移除点的热力图，而不是天数。
    
- **I/O格式:**
    
    - 输入: M*N长度的二进制字符串。
        
    - 输出: M*N长度的二进制多标签向量，标记了被移除的点。
        
- **主要参数:** NUM_SAMPLES, GRID_M, GRID_N。

---

## 36. **algorithms/generate_nim_game_zeckendorf.py**

- **用途:** 这个实验旨在测试我的范式能否学习一个基于复杂数论（齐肯多夫表示法）的非直观博弈论问题。它脱离了简单的模式匹配，需要模型理解更深层次的数学结构。
    
- **逻辑:** 我实现了一个经典的石子游戏变种（类似Wythoff博弈）的求解器，其解法与斐波那契数列和齐肯多夫表示法紧密相关。为了让模型能更好地学习，我简化了任务：原始问题可能是计算[1, n]区间内有多少个必胜局面，我将其修改为只判断给定的n本身是否为必胜局面。这让每个输入和输出都有了更直接的因果关系。
   
- **I/O格式:**
    
    - 输入: N_BITS长度的二进制字符串，代表石子总数n。
        
    - 输出: 1位二进制多标签（[1]代表必胜, [0]代表必败）。
        
- **主要参数:** N_BITS, DATASET_SIZE。

---

## 37. **algorithms/generate_longest_subsequence_constrained.py**

- **用途:** 测试模型处理一个混合了序列操作和数值约束的复杂优化问题的能力。对应 LeetCode 题目：[2311. Longest Binary Subsequence Less Than or Equal to K](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/)
    
- **逻辑:** 输入是一个二进制字符串s和一个整数k（也由二进制表示）。任务是找到s的一个子序列（可以不连续），该子序列组成的二进制数的值小于等于k，且长度最长。输出是这个最长子序列的长度。
    
- **I/O格式:**
    
    - 输入: STRING_LENGTH_N + K_BITS长度的二进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表长度。
        
- **主要参数:** STRING_LENGTH_N, K_BITS, DATASET_SIZE。

---

## 38. **algorithms/generate_treasure_hunt_tsp.py**

- **用途:** 解决一个复杂的状态空间搜索问题，它结合了图的遍历（BFS）和组合优化（状态压缩DP），是算法竞赛中的经典难题。
    
- **逻辑:** 在一个给定的迷宫中，玩家需从起点'S'出发，触发所有机关'M'，最终到达终点'T'。途中可以利用石头'O'来瞬间触发任意一个机关。脚本通过一系列BFS计算出所有关键点（S, T, M, O）之间的最短距离，然后使用状态压缩动态规划来找到遍历所有机关并到达终点的最短总路径长度。- **LeetCode题目:** [LCP 13. 寻宝](https://leetcode.cn/problems/xun-bao/) - 这正是本脚本实现的核心问题
    
- **I/O格式:**
    
    - 输入: N*M长度的字符串，代表迷宫布局。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表最短路径长度（-1映射为0）。
        
- **主要参数:** MAZE_N, MAZE_M, DATASET_SIZE。

---

## 39. **algorithms/generate_freedom_trail_dp.py**

- **用途:** [LeetCode 514. Freedom Trail](https://leetcode.com/problems/freedom-trail/description/) 测试模型学习解决一个需要动态规划和路径回溯的复杂优化问题的能力。
    
- **逻辑:** 输入是一个代表圆盘字符的ring字符串和一个目标key字符串。该脚本使用动态规划来计算拼写出key所需的最小旋转步数。一个关键的修改是，脚本不仅计算总步数，还回溯DP表以重建每一步的最优操作（顺时针还是逆时-针，以及具体步数），并将这个操作序列作为输出。
    
- **I/O格式:**
    
    - 输入: ring|key格式的字符串。
        
    - 输出: KEY_LENGTH * move_bits长度的二进制多标签向量，编码了每一步的操作。
        
- **主要参数:** RING_LENGTH, KEY_LENGTH, NUM_SAMPLES。

---

## 40. **algorithms/generate_sum_of_subset_with_mask.py**

- **用途:** 测试模型根据一个二进制掩码从一个集合中选择元素并执行聚合操作（求和）的能力。这是一个创新性的对照实验，旨在验证**任务结构本身**（而非模型容量）对神经网络可学习性的关键影响。
    
- **逻辑:** 本脚本提供**两种问题格式**，用于对比研究：
        
    1. **逆向问题（组合优化）**: 输入是数字集合 + 目标值，输出是mask（哪个子集的和等于target）。这是一个需要搜索、决策、组合推理的NP-hard问题，学习难度极高，用于测试模型学习组合优化的能力。
        
    2. **顺向问题（纯计算）**: 输入是数字集合 + 子集掩码（已经告诉你选哪个子集），输出是sum（这些数的和）。这只需要加法运算，学习难度极低，用于验证任务结构对可学习性的影响。
    
    脚本通过`MODE`配置参数切换两种模式。实际训练采用的是**顺向模式**，因为逆向模式（原始子集和问题）对神经网络而言几乎不可学习，这有力证明了**任务结构本身**是决定可学习性的关键因素，而非模型容量不足。
    
- **I/O格式:**
    
    - 逆向模式: 输入为numbers(4-bit) + target(6-bit)，输出为mask(n_items位)
    - 顺向模式: 输入为numbers(4-bit) + mask(n_items位)，输出为sum(6-bit)
    
- **主要参数:** n_items, value_range (数字范围), num_samples, MODE ("forward"或"reverse")。

---

## 41. **algorithms/generate_sudoku_6x6.py**

- **用途:** 测试模型在处理有强约束满足问题（Constraint Satisfaction Problem）——数独——上的能力。
    
- **逻辑:** 脚本实现了一个带回溯的求解器来生成完整的6x6数独解，并通过“挖洞”的方式创建有唯一解的谜题。
    
- **I/O格式:**
    
    - 输入: 36个字符的字符串，用_表示空格。
        
    - 输出: 36 * 3 = 108位二进制多标签向量，每个数字用3位二进制表示。
        
- **主要参数:** num_puzzles, difficulty (挖洞比例)。

---

## 42. **algorithms/generate_valid_parentheses_path_random_deprecated.py**

- **用途:** (早期探索/已弃用) 这是解决“合法括号路径”问题的早期尝试。
    
- **状态:** **已弃用**。此脚本通过随机生成括号网格来创建数据集，但这导致了严重的数据不平衡问题（绝大多数随机网格都没有合法路径），不利于模型训练。已被 algorithms/generate_valid_parentheses_path_balanced.py 取代。
    
- **逻辑:** 随机生成M x N的括号网格，并调用求解器判断是否存在合法路径。
    
- **主要参数:** MAZE_M, MAZE_N, DATASET_SIZE。

---

## 43. **algorithms/generate_valid_parentheses_path_balanced.py**

- **用途:** 解决一个二维网格上的路径查找问题，但路径的合法性受到栈式结构（括号匹配）的约束。这是一个算法和逻辑约束结合的复杂任务 (LeetCode Hard [#2267](https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/) "Check if There Is a Valid Parentheses String Path")。
    
- **逻辑:** 脚本通过两种方式生成数据以确保平衡：
    
    1. **正样本:** 首先在网格上确定一条路径，然后沿着这条路径生成一个合法的括号序列，再随机填充路径外的格子。
        
    2. **负样本:** 随机生成网格，并通过求解器验证其确实没有合法路径。  
        模型的任务是判断给定的括号网格中，是否存在一条从(0,0)到(M-1,N-1)的路径，使得路径上的括号序列是合法的。
        
- **I/O格式:**
    
    - 输入: M*N长度的二进制字符串（'('->0, ')'->1）。
        
    - 输出: [1] (存在) 或 [0] (不存在)。
        
- **主要参数:** MAZE_M, MAZE_N, DATASET_SIZE。

---

## 44. **algorithms/generate_point_in_polygon.py**

- **用途:** 测试模型学习一个计算几何中的经典算法——射线法（Ray Casting Algorithm）——的能力。
    
- **逻辑:** 脚本首先随机生成一个不自交的多边形的N个顶点，然后随机生成一个测试点。输入是将所有顶点和测试点的坐标进行二进制编码后拼接而成的字符串。输出是一个bit，表示该测试点是否位于多边形内部。为了确保数据集平衡，脚本会努力让内部和外部的样本数量大致相等。
    
- **I/O格式:**
    
    - 输入: (NUM_VERTICES_N + 1) * 2 * BITS_PER_COORD长度的二进制字符串。
        
    - 输出: [1] (在内部)或 [0] (在外部)。
        
- **主要参数:** NUM_VERTICES_N, BITS_PER_COORD, DATASET_SIZE。

---

## 45. **algorithms/generate_shortest_path_in_matrix_bfs.py**

- **用途:** 测试模型在一个二维网格中，基于经典的广度优先搜索（BFS）算法寻找最短路径的能力。
    
- **逻辑:** 输入是一个N x N的二进制矩阵，其中'0'代表通路，'1'代表墙壁。任务是计算从左上角(0,0)到右下角(N-1, N-1)的最短路径长度（允许八向移动）。脚本使用BFS算法来找到最优解。如果两点不连通，则路径长度为0。
    
- **I/O格式:**
    
    - 输入: NN位二进制字符串 或 (NN)/4位十六进制字符串。
        
    - 输出: OUTPUT_BITS长度的二进制多标签向量，代表路径长度。
        
- **主要参数:** MATRIX_SIZE_N, INPUT_FORMAT, DATASET_SIZE。

---

## 46. **algorithms/generate_sudoku_4x4_stepwise_deprecated.py**

- **用途:** (废弃) 旨在测试模型进行“步进式”（stepwise）推理的能力，即在每个状态下只预测下一步的最优动作，而不是一次性输出完整解。
    
- **状态:** **已弃用**。该脚本试图通过一种复杂的倒推逻辑来为4x4数独生成步进式解法，但其核心算法不可靠，无法保证生成的数据的正确性和有效性。已被generate_sudoku_6x6.py等更完善的脚本替代。
    
- **逻辑:** (存在问题) 尝试从一个完整的4x4数独解中，通过“挖洞”并检查唯一性的方式倒推出每一步的最优解。

---

## 47. **algorithms/generate_tiling_problem_deprecated.py**

- **用途:** (已弃用) 旨在测试模型解决一个经典的平铺覆盖优化问题的能力，这是一个NP-hard问题。
    
- **状态:** **已弃用**。核心求解器使用了带剪枝的回溯搜索，但这是一种指数级复杂度的算法。对于大于约 13x13 的矩阵，其计算时间会变得不切实际，因此无法高效地生成大规模数据集。
    
- **逻辑:** 使用回溯搜索法解决“用最少的正方形铺满一个m*n矩形”的问题。

---

## 48. **algorithms/generate_hanoi_tower_twisted_path_deprecated.py**

- **用途:** (废弃) 此脚本意图生成一个汉诺塔问题的“扭曲路径”数据集，即从一个非标准但困难的起始状态到标准终点的最优路径。
    
- **状态:** **已弃用**。核心的移动逻辑（apply_move）存在错误，未能正确模拟汉诺塔的“大盘在下，小盘在上”的规则，导致生成的路径并非有效的解。后续已被generate_hanoi_tower_build_full_state_graph.py等更完善的脚本取代。

---

## 49. **algorithms/generate_checkers_jump_1d_v2.py**

- **用途:** 这是一个针对跳棋交换问题的**序列学习对比实验**脚本。与 V1 版本关注单步最优策略不同，本脚本旨在生成多种类型的完整路径数据集（包括最优路径、子路径、非最优路径等），用于系统性地研究模型在不同数据分布下学习长序列规划的能力。该任务同样源自苹果公司论文 [15] 中的跳棋交换问题，但侧重于探究序列级泛化与路径多样性的影响。
    
- **逻辑:** 脚本在一个一维棋盘上模拟两种颜色的棋子（'R' 和 'B'）互相穿过到达对方初始位置的过程。它使用高效的反向广度优先搜索（BFS），从目标状态开始，逆向遍历整个状态空间，从而为每一个可达的状态都计算出其唯一的最优下一步。
    
- **I/O格式:**
    
    - 输入: 2*N+1长度的整数序列，代表棋盘状态。
        
    - 输出: 单个整数，代表要移动的棋子的**位置索引**，这是一个分类问题。
        
- **主要参数:** CHECKERS_N (每种颜色棋子数)。

---

## 50. **algorithms/generate_maze_symbolic_to_image.py**

- **用途:** 将符号化的迷宫路径规划数据集转换为图像格式，以测试视觉模型（如CNN、ViT）直接从像素进行路径规划的能力。
    
- **逻辑:** 该脚本读取一个包含迷宫布局字符串和对应最优动作的.jsonl文件。对于每一行，它将迷宫布局（包含墙壁、通路、起点和终点）渲染成一张高对比度的彩色图像。最终，它会生成一个图像文件夹和一个labels.csv文件，将图像文件名与最优动作标签（0-3的分类）关联起来。
    
- **I/O格式:**
    
    - 输入: .jsonl文件。
        
    - 输出: images/目录下的JPG图像 和 labels.csv元数据文件。
        
- **主要参数:** INPUT_JSONL_FILE, OUTPUT_IMAGE_DIR, IMAGE_SIZE, GRID_DIM。

---

## 51. **algorithms/generate_trapping_rain_water_visualizer.py**

- **用途:** 这是一个**数据转换与可视化**脚本。它的作用是将已经生成的、符号化的“接雨水”数据集转换为一个image-to-image格式的数据集，以便用视觉模型来解决同一个问题。
    
- **逻辑:** 脚本逐行读取.jsonl文件，解析出每个样本的柱子高度和对应的接水量。它首先将柱子的高度渲染成一张黑白的图像作为输入。然后，它在柱子上方将对应的雨水量用蓝色渲染出来，生成输出图像。
    
- **I/O格式:** 输入: .jsonl文件 -> 输出: images/目录下的PNG图像对。
    
- **主要参数:** input_file, output_dir, image_size。

---

## 52. **algorithms/generate_shortest_path_in_tree_deprecated.py**

- **用途:** (早期探索/已弃用) 这是一个早期的实验，旨在测试模型从图像中寻找图上最短路径的能力。
    
- **状态:** **已弃用**。该脚本通过生成随机树来确保图的平面性，但这意外地将问题简化了：树中任意两点间的路径是唯一的，模型无需学习“最短”这一概念。该任务后来被更具挑战性的稠密迷宫路径规划任务所取代。
    
- **逻辑:** 脚本生成一个随机树图，并将其绘制在图像上。输入图像会高亮一个起点和一个终点。输出图像则会在输入的基础上，将连接起点和终点的唯一路径高亮出来。
    
- **主要参数:** MIN_NODES, MAX_NODES。

---

## 53. **generate_rain_water_final_showdown.py**

- **用途:** 这是一个**算法对比**实验，旨在通过提供三种不同算法（动态规划、单调栈、双指针）的中间过程作为辅助标签，来研究哪种算法逻辑更符合神经网络的归纳偏置。
    
- **逻辑:** 脚本生成随机的柱子高度。除了计算最终的接水量外，它还同时生成了三种算法的执行轨迹：
    1.  **DP:** 记录left_max和right_max数组。
    2.  **Stack:** 记录单调栈操作过程中的三元组(left_idx, right_idx, top_height)。
    3.  **Two Pointers:** 记录双指针移动过程中的max值更新。
    
- **I/O格式:**
    
    - 输入: N * BITS_PER_HEIGHT位二进制串。
        
    - 输出: 包含 `final_answer` 和 `explain_dp`, `explain_stack`, `explain_tp` 的JSON对象。
        
- **主要参数:** NUM_COLUMNS_N, BITS_PER_HEIGHT。

---

## 53. **algorithms/generate_bricks_falling.py**

- **用途:** 解决 LeetCode 困难题 [803. Bricks Falling When Hit](https://leetcode.cn/problems/bricks-falling-when-hit/description/)。测试模型对物理稳定性和连通性的理解。砖块只有在与顶部直接相连或间接通过其他稳定砖块相连时才是稳定的，否则会掉落。
    
- **逻辑:** 
    1.  随机生成一个初始网格并计算其稳定状态（移除悬空砖块）。
    2.  在该稳定局面上选择一个砖块进行“击打”（移除）。
    3.  使用并查集（Union-Find）重新计算击打后的稳定局面，所有不再与顶部连通的砖块都会消失。
    4.  模型的任务是预测击打后的最终稳定局面。
    
- **I/O格式:**
    
    - 输入: 初始稳定网格(Flattened) + 击打位置(Row index binary) + 击打位置(Col index binary)。
        
    - 输出: 最终稳定网格(Flattened二进制向量)。
        
- **主要参数:** ROWS, COLS, NUM_SAMPLES。

---

## 54. **algorithms/generate_maximal_rectangle_coords.py**

- **用途:** 解决 LeetCode 困难题 [85. Maximal Rectangle](https://leetcode.cn/problems/maximal-rectangle/)。
- **逻辑:** 给定一个包含0和1的二维矩阵，找出只包含1的最大矩形，并返回其精确坐标 `(r1, c1, r2, c2)`。
- **I/O格式:**
    - 输入: 此脚本针对固定 8x8 矩阵，输入为 Flatten 后的 64 位二进制串。
    - 输出: 4个坐标值的二进制编码拼接。
- **主要参数:** ROWS, COLS, NUM_SAMPLES。

---

## 55. **algorithms/generate_median_island_median_finder.py**

- **用途:** 这是一个复杂的符号推理任务，要求神经网络在二进制序列中执行多层逻辑操作，结合了序列分割、中位数计算和定位操作。

- **逻辑:**
    1. 扫描输入字符串，识别所有连续'1'孤岛的起止位置
    2. 计算孤岛总数，找到中位孤岛（按顺序排列的中间孤岛）
    3. 在该中位孤岛中，找到中间位置的'1'
    4. 输出该中位'1'的绝对位置索引

- **算法步骤（"七重炼狱"算法）:**
    1. 孤岛分割与计数
    2. 中位孤岛的定位
    3. 中位'1'的内部序号计算
    4. 最终绝对位置计算

- **I/O格式:**
    - 输入: 30位二进制字符串
    - 输出: 5位二进制编码（表示0-30的位置索引，特殊值30表示无孤岛）

- **主要参数:** INPUT_WIDTH = 30, NUM_SAMPLES = 500_000。

- **任务特点:**
    - 结合序列分割、中位数计算和定位操作
    - 要求理解序列结构和相对位置关系
    - 对神经网络的空间推理能力提出挑战
    - 包含边界情况处理（无孤岛时返回-1，编码为30）

---

## 56. **algorithms/generate_rain_water_summation.py**

- **用途:** “接雨水”问题的子任务变体（Task B）。输入不是地形高度，而是已经计算好的**每格雨水量**（中间状态）。任务是计算这些雨水量的**总和**。这用于测试模型单纯进行“求和”这一数学聚合操作的能力，与完整的接雨水任务（分析地形+求和）形成对比。

- **逻辑:** 随机生成N个格子的雨水量数据。将它们相加得到总和。

- **I/O格式:**
    - 输入: NUM_COLUMNS_N * BITS_PER_CELL 长度的二进制字符串（每格雨水量的拼接）。
    - 输出: TOTAL_OUTPUT_BITS 长度的二进制多标签向量（总雨水量的二进制表示）。

- **主要参数:** NUM_COLUMNS_N, BITS_PER_CELL, DATASET_SIZE.

---

## 57. **algorithms/generate_rain_water_then_cellular_automata.py**

- **用途:** 这是一个**跨领域链式推理**任务（Task D）。测试模型能否在一次前向传播中，连续执行两个逻辑上完全无关的复杂步骤：首先解决“接雨水”问题计算每格水量，然后立即将该水量数据作为初始状态，执行“元胞自动机”演化。

- **逻辑:**
    1. **阶段一（接雨水）:** 生成随机地形，计算每格的接水量。
    2. **阶段二（CA演化）:** 将阶段一计算出的“每格水量”序列直接作为一维元胞自动机（Rule 110）的初始状态。
    3. **演化:** 对该状态进行多步（CA_EVOLUTION_LAYERS）演化。
    4. **输出:** 最终的CA状态。

- **I/O格式:**
    - 输入: 地形高度数据的二进制编码。
    - 输出: CA演化后的最终状态（二进制多标签）。

- **主要参数:** NUM_COLUMNS_N, BITS_PER_HEIGHT, CA_RULE_NUMBER (110), CA_EVOLUTION_LAYERS.

---

# C: 视觉推理 (Visual Reasoning)

## 1. **visual_reasoning/generate_checkerboard_to_binary.py**

- **用途:** 这是一个基础的视觉到符号转换任务，用于测试模型从原始像素数据中解码结构化信息的能力。
    
- **逻辑:** 脚本为每个样本生成一个随机的N x N二进制网格，并将其渲染成一个IMAGE_SIZE x IMAGE_SIZE的黑白棋盘格图像。输入是这张图像，输出是其背后对应的N*N位扁平化二进制字符串。
    
- **I/O格式:**
    
    - 输入: IMAGE_SIZE x IMAGE_SIZE的灰度图像。
        
    - 输出: GRID_DIM * GRID_DIM长度的二进制多标签向量。
        
- **主要参数:** NUM_SAMPLES, IMAGE_SIZE, GRID_DIM。

---

## 2. **visual_reasoning/generate_line_angle_to_vector.py**

- **用途:** 测试模型从图像中提取精确几何信息（角度）的能力，这是一个比简单识别棋盘格更高级的视觉推理任务。
    
- **逻辑:** 脚本生成一个类似雷达扫描或钟表表盘的图像。从图像中心点出发，绘制若干条颜色、宽度、角度都随机的线段。整个360度被划分为num_angle_bins个扇形区间。模型的任务是输出一个多热编码向量，标记出哪些角度区间内存在线段。
    
- **I/O格式:**
    
    - 输入: image_size x image_size的RGB图像。
        
    - 输出: num_angle_bins长度的二进制多标签向量。
        
- **主要参数:** image_size, num_angle_bins, min_lines, max_lines。

---

## 3. **visual_reasoning/generate_count_shapes_from_image.py**

- **用途:** 测试模型同时进行物体识别（形状）、属性识别（颜色）和计数（聚合）的多重视觉任务能力。
    
- **逻辑:** 脚本在一个白色画布上随机放置不同形状（正方形、圆形、三角形）和颜色（红、绿、蓝）的物体，并确保它们互不重叠。模型的任务是输出一个12-bit的向量，该向量编码了每种形状和每种颜色的总数。 (注意: 由于颜色是随机分配的，可能存在颜色计数的轻微不平衡)
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像。
        
    - 输出: 12位二进制多标签向量，分别编码6个类别的计数。
        
- **主要参数:** TOTAL_SAMPLES, MAX_COUNT_PER_CATEGORY (仅对形状生效)。

---

## 4. **visual_reasoning/generate_sokoban_symbolic_to_image_no_labels.py**

- **用途:** 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）仅转换为图像格式，用于纯视觉任务或作为更复杂数据处理的中间步骤。
    
- **逻辑:** 脚本逐行读取一个包含推箱子布局字符串的.jsonl文件。对于每一行，它将布局字符串渲染成一张彩色的、具有指定视觉风格的224x224图像并保存。此版本**不生成**对应的标签文件。
    
- **I/O格式:**
    
    - 输入: sokoban_optimized_dataset.jsonl文件。
        
    - 输出: images/目录下的PNG图像。
        
- **主要参数:** INPUT_JSONL_PATH, OUTPUT_DIR, GRID_SIZE, CELL_PIXELS。

---

## 5. **visual_reasoning/generate_sokoban_symbolic_to_image_with_labels.py**

- **用途:** 这是一个数据转换脚本，用于将符号化的推箱子数据集（.jsonl格式）转换为一个完整的图像分类数据集，以供计算机视觉模型（如ViT, Swin Transformer）进行训练。
    
- **逻辑:** 脚本逐行读取一个包含推箱子布局字符串和对应最优动作的.jsonl文件。对于每一行，它将布局字符串渲染成一张彩色的、具有指定视觉风格的224x224图像，并将该图像的文件名与原始的最优动作标签一起，写入一个labels.csv元数据文件中。
    
- **I/O格式:**
    
    - 输入: sokoban_optimized_dataset.jsonl文件。
        
    - 输出: images/目录下的PNG图像 和 labels.csv文件。
        
- **主要参数:** INPUT_JSONL_PATH, OUTPUT_DIR, GRID_SIZE, CELL_PIXELS。

---

## 6. **visual_reasoning/generate_triangle_to_incircle.py**

- **用途:** 这是展示“用梯度下降雕刻精确规则”的一个标志性实验。它测试模型能否学习到一个纯粹的、非平凡的几何构造规则（三角形内切圆）。
    
- **逻辑:** 脚本为每个样本生成一个随机的绿色三角形作为输入图像。然后，它精确地计算出该三角形唯一的内切圆，并将这个红色的内切圆绘制在原始三角形上，作为输出图像。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含一个绿色三角形)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (三角形+红色内切圆)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_TRIANGLE_AREA。

---

## 7. **visual_reasoning/generate_polygon_to_symmetry_axis.py**

- **用途:** 测试模型从一个完整的对称图形中反向推断出其隐含的对称轴的能力。
    
- **逻辑:** 脚本首先定义一条随机的对称轴。然后，它在轴的一侧随机生成一组顶点，并将这些顶点镜像到另一侧，从而构成一个完美的轴对称多边形。输入图像只包含这个多边形，输出图像则在此基础上额外绘制出了那条隐藏的对称轴。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含一个对称图形)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (对称图形+对称轴)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_POLYGON_VERTICES_HALF。

---

## 8. **visual_reasoning/generate_triangle_to_centroid.py**

- **用途:** 测试模型学习另一个基础几何概念——重心的能力。
    
- **逻辑:** 脚本生成一个随机的绿色三角形作为输入图像。然后，它计算出该三角形的质心（重心），并在该位置绘制一个固定大小的红色圆点作为输出图像。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含一个绿色三角形)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (三角形+红色重心圆)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE, MIN_TRIANGLE_AREA。

---

## 9. **visual_reasoning/generate_triangle_to_tessellation.py**

- **用途:** 这是我们范式能力的一个标志性展示。它测试模型能否学习一种无限的、基于晶格的生成规则。由于镶嵌图案的全局关联性和细节的精确性，它有力地排除了模型仅仅是靠“插值”或“记忆”来解决问题的可能性。
    
- **逻辑:** 输入图像仅包含一个随机生成、随机摆放的绿色三角形。脚本将这个三角形作为“晶胞”的基础，通过在两个不共线的基向量方向上进行平移，用绿色和红色的三角形交替铺满整个画布，形成完美的平面镶嵌图案。输出图像就是这个完整的镶嵌图案。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含一个绿色三角形)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (完整的镶嵌图案)。
        
- **主要参数:** NUM_SAMPLES, IMG_SIZE。

---

## 10. **visual_reasoning/generate_shortest_distance_between_triangles.py**

- **用途:** 测试模型在包含多个对象的情况下，进行全局几何关系（最短距离）推理的能力。
    
- **逻辑:** 脚本在一个画布上随机生成两个不重叠的绿色三角形。然后，它使用专业的计算几何库shapely来精确计算出这两个三角形之间的最短距离线段。输入图像只包含两个三角形，输出图像则额外绘制了这条红色的最短连线。
    
- **I/O格式:**
    
    - 输入: image_size x image_size的RGB图像 (包含两个绿色三角形)。
        
    - 输出: image_size x image_size的RGB图像 (三角形+红色最短连线)。
        
- **状态:** 逻辑正确，但依赖外部库shapely，可能存在环境配置问题。

---

## 11. **visual_reasoning/generate_coords_to_triangle.py**

- **用途:** 这是一个基础的符号到几何的渲染任务，测试模型将抽象的坐标信息转换为具体像素形状的能力。
    
- **逻辑:** 脚本的输入是一个48-bit的二进制字符串，它编码了一个三角形三个顶点的(x, y)坐标（每个坐标8-bit）。输出是根据这些坐标绘制出的一个绿色实心三角形的图像。
    
- **I/O格式:**
    
    - 输入: 48位二进制字符串。
        
    - 输出: 256x256的RGB图像。
        
- **主要参数:** NUM_SAMPLES, IMAGE_SIZE。

---

## 12. **visual_reasoning/generate_triangle_coords_to_tessellation.py**

- **用途:** 这是一个高级的、混合了符号指令和几何生成规则的推理任务。
    
- **逻辑:** 和generate_coords_to_triangle.py一样，输入是一个48-bit的二进制串，定义了一个基准三角形。和generate_triangle_to_tessellation.py一样，输出是基于这个基准三角形的完美平面镶嵌图案。**关键的修改**是，在输出的镶嵌图案中，由输入直接定义的那个基准三角形会被染成特殊的颜色（如黑色），而其他的三角形则依然是绿色和红色。这为模型提供了必要的“接地”（grounding）信息。
    
- **I/O格式:**
    
    - 输入: 48位二进制字符串。
        
    - 输出: 256x256的RGB图像 (镶嵌图案)。
        
- **主要参数:** NUM_SAMPLES, IMG_SIZE。

---

## 13. **generate_trapping_rain_water_image_to_symbol.py**

- **用途:** 这是一个**图像到符号**的转换任务，用于测试CNN等视觉模型是否能从图像中提取精确的物理量。
    
- **逻辑:** 脚本生成随机的柱子高度，并将其渲染成黑白图像作为输入。同时，它计算每个柱子的接水量，并将其编码为二进制字符串作为标签。这使得我们可以训练一个模型，看图计算雨水量。
    
- **I/O格式:**
    
    - 输入: 240x240的灰度图像 (柱子高度图)。
        
    - 输出: 二进制字符串标签 (每个柱子的接水量)。
        
- **主要参数:** NUM_COLUMNS_N, BITS_PER_HEIGHT, IMAGE_SIZE。

---

# D: 元胞自动机 (Cellular Automata)

## 1. **cellular_automata/generate_cellular_automata_1d.py**

- **用途:** 用于生成一维元胞自动机（CA）的演化数据集，以测试模型学习和执行局部、确定性规则的能力。
    
- **逻辑:** 给定一个随机的二进制初始状态，脚本会根据指定的演化规则（当前为Rule 110），迭代指定的层数（步数），并生成最终状态。
    
- **I/O格式:**
    
    - 输入: length位的二进制字符串，代表初始状态。
        
    - 输出: length位的二进制多标签向量，代表最终状态。
        
- **主要参数:** num_samples, length, l (演化层数)。

---

## 2. **cellular_automata/generate_game_of_life_2d.py**

- **用途:** 生成二维元胞自动机——Conway's Game of Life的数据集。此任务比一维CA更复杂，需要模型理解二维空间中的邻域关系。
    
- **逻辑:** 基于一个随机的n*n初始棋盘，根据生命游戏标准规则（B3/S23）演化d个时间步，记录最终的棋盘状态。
    
- **I/O格式:**
    
    - 输入: n*n位的扁平化二进制字符串，代表初始棋盘。
        
    - 输出: n*n位的二进制多标签向量，代表最终棋盘。
        
- **主要参数:** num_samples, n (网格边长), d (演化步数)。

---

## 3. **cellular_automata/generate_cellular_automata_1d_multistate.py**

- **用途:** 作为一维元胞自动机实验的扩展，测试模型处理非二进制状态空间的能力。
    
- **逻辑:** CCA的演化规则是：一个元胞的下一个状态，是它自己当前状态+1（模n_states），当且仅当它的左邻居或右邻居的状态等于这个目标状态。该脚本生成这种演化过程的输入（初始状态）和输出（最终状态）。
    
- **I/O格式:**
    
    - 输入: n_cells * 2 (因为n_states=4) 长度的二进制字符串。
        
    - 输出: n_cells * 2 长度的二进制多标签向量。
        
- **主要参数:** n_cells, n_states, n_samples, steps。

---

## 4. **cellular_automata/generate_cellular_automata_programmable.py**

- **用途:** 测试模型的“可编程性”或“元学习”能力。模型不仅要学会CA的演化过程，还要能根据每次输入中给出的不同规则来执行演化。
    
- **逻辑:** 每个样本将一个8-bit的规则编号和一个初始状态一同作为输入。脚本根据该规则对状态进行演化，生成输出。这要求模型将输入的一部分理解为“程序”，另一部分理解为“数据”。
    
- **I/O格式:**
    
    - 输入: 8 (规则) + CA_WIDTH (状态)位的二进制字符串。
        
    - 输出: CA_WIDTH位的多标签二进制向量。
        
- **主要参数:** TARGET_NUM_SAMPLES, CA_WIDTH, EVOLUTION_STEPS, RULES_TO_INCLUDE (指定数据集中包含哪些规则)。

---

## 5. **cellular_automata/generate_cellular_automata_inverse_rule90.py**

- **用途:** 测试模型解决“逆问题”（Inverse Problem）的能力。给定一个确定性系统的输出，模型需要反向推断出满足特定约束（最稀疏且唯一）的可能输入。
    
- **逻辑:** 输入是一维元胞自动机Rule 90演化一步后的状态。任务是找到所有可能的“前一步”状态中，'1'的数量最少（最稀疏）的那一个。为了让问题有唯一解，脚本通过暴力搜索的方式，只保留那些“最稀疏解”恰好只有一个的样本。
    
- **I/O格式:**
    
    - 输入: N位二进制字符串 (演化后状态)。
        
    - 输出: N位二进制字符串 (演化前状态)。
        
- **主要参数:** num_samples, length。

---

## 6. **cellular_automata/generate_game_of_life_image_to_image.py**

- **用途:** 这是二维元胞自动机的image-to-image版本，测试模型能否直接在像素空间中执行基于局部规则的演化。
    
- **逻辑:** 脚本生成一个随机的GRID_SIZE x GRID_SIZE的初始状态，并将其渲染成一张黑白图像作为输入。然后，它根据生命游戏规则计算出下一步的状态，并将其渲染成另一张图像作为输出。
    
- **I/O格式:**
    
    - 输入: IMAGE_SIZE x IMAGE_SIZE的灰度图像 (初始状态)。
        
    - 输出: IMAGE_SIZE x IMAGE_SIZE的灰度图像 (演化一步后的状态)。
        
- **主要参数:** GRID_SIZE, IMAGE_SIZE, NUM_SAMPLES。

---

## 7. **cellular_automata/generate_cellular_automata_spatial_conditional.py**

- **用途:** 测试模型在单一模态（图像）内部分区和解析“指令”与“数据”的能力，是一种“伪多模态”或“空间条件化”的实验。
    
- **逻辑:** 脚本将一个36位的元胞自动机问题编码在一张图像中。图像顶部的窄条区域用特定颜色（红/绿）编码一个8-bit的演化规则，图像下方的6x6大区域则用黑白色块编码36-bit的初始状态。输出图像则是在该规则下演化3步后的最终状态。模型必须学会“读取”顶部的规则，并将其应用到下方的状态上。
    
- **I/O格式:**
    
    - 输入: IMG_WIDTH x IMG_HEIGHT的RGB图像 (顶部为规则编码，下部为初始状态)。
        
    - 输出: IMG_WIDTH x IMG_HEIGHT的RGB图像 (最终状态)。
        
- **主要参数:** NUM_INITIAL_STATES, ITERATIONS, GRID_DIM。

---

## 8. **cellular_automata/generate_cellular_automata_multimodal_deprecated.py**

- **用途:** (已弃用) 生成一个真正的多模态数据集，用于训练能够同时理解图像输入和文本指令的模型。
    
- **状态:** **已弃用**。由于缺乏合适的、易于训练的多模态模型（在实验框架内），以及generate_cellular_automata_spatial_conditional.py提供了一个更简洁的替代方案，该脚本生成的训练集未被使用。
    
- **逻辑:** 脚本为每个样本生成一张代表元胞自动机初始状态的图像，和一条代表演化规则的文本字符串。输出是演化后的状态图像。
    
- **主要参数:** NUM_SAMPLES, GRID_DIM, ITERATIONS。

---

## 9. **generate_cellular_automata_1d_to_grid_image_interp.py**

- **用途:** 该脚本旨在设计一个“逻辑/感知混合”任务，用以证明神经网络的规则学习能力和内插能力并非互斥，而是可以一体化地在单个任务中得到体现。它迫使模型必须同时“看穿”输入的连续灰度值以执行离散的逻辑推理，并记住这些灰度值以完成最终的连续值映射。
    
- **逻辑:** 脚本首先生成一个36位的元胞自动机逻辑初始状态。在生成输入图像时，代表逻辑“0”的格子被赋予一个随机的深灰度值（如0-63），代表逻辑“1”的格子则被赋予一个随机的浅灰度值（如192-255）。然后，脚本根据元胞自动机规则计算出最终的逻辑输出状态。在生成输出图像时，它遵循一个混合规则：如果一个格子的最终逻辑状态是“1”，则其灰度值保持与输入图像中对应格子的灰度值相同；如果最终逻辑状态是“0”，则其灰度值变为输入灰度值的反色（255 - 输入值）。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (一个6x6的棋盘格，每个格子的颜色是随机的深灰或浅灰)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (根据逻辑规则和输入颜色变换后的6x6棋盘格)。
        
- **主要参数:** NUM_SAMPLES, IMG_SIZE, RULE_NUMBER, ITERATIONS, ENABLE_INTERPOLATION_MODE。

---

## 10. **cellular_automata/generate_cellular_automata_1d_to_grid_image.py**

- **用途:** 测试模型能否直接将一维的符号计算结果“渲染”成结构化的二维图像。
    
- **逻辑:** 输入是一个36位的二进制字符串，代表一维元胞自动机的初始状态。脚本首先在内部根据Rule 110对其进行3步演化，得到一个36位的最终状态。然后，它将这个一维的最终状态渲染成一个6x6的黑白棋盘格图像作为输出。
    
- **I/O格式:**
    
    - 输入: 36位二进制字符串。
        
    - 输出: 240x240的RGB图像（黑白棋盘格）。
        
- **主要参数:** CA_WIDTH, RULE_NUMBER, ITERATIONS, GRID_DIM。

---

## 11. **cellular_automata/generate_cellular_automata_inverse_rule.py**

- **用途:** 这个实验是检验模型**逆向推理**（Inverse Reasoning）能力的第一个尝试。我的问题是：如果模型能从规则正向推出结果，那么它能否从“输入-输出”对中反向推断出其背后的规则？
    
- **逻辑:** 我首先随机选择一个元胞自动机规则（从0到255）。然后，**构造一个包含所有8种3位邻域模式的初始状态（使用De Bruijn序列）**，应用这个规则演化固定的几步，得到最终状态。这种特殊的初始状态设计确保了规则可以被唯一推断。我将初始状态和最终状态拼接起来作为模型的输入，而将那个被隐藏的8-bit规则作为模型的预测目标。
    
- **I/O格式:**
    
    - 输入: CA_WIDTH * 2长度的二进制字符串 (initial_state + final_state)。
        
    - 输出: 8位二进制多标签向量 (代表被预测的规则)。
        
- **主要参数:** CA_WIDTH, NUM_SAMPLES, ITERATION_LAYERS。

---

## 12. **cellular_automata/generate_cellular_automata_inverse_rule_and_steps.py**

- **用途:** 这是在实现“唯一解”版本之前的一个早期版本，它同样旨在让模型学习预测规则和迭代次数。
    
- **逻辑:** 和最终的_unique版本一样，这个脚本也为每个样本随机选择规则和迭代次数来生成数据。然而，它缺少了对解的唯一性进行验证的步骤。这意味着数据集中可能存在一些模糊的样本，即多个（规则，步数）组合可能产生相同的输入-输出对。
    
- **I/O格式:**
    
    - 输入: CA_WIDTH * 2长度的二进制字符串。
        
    - 输出: 8 + ITERATION_BITS长度的二进制多标签向量。
        
- **主要参数:** CA_WIDTH, NUM_SAMPLES, MAX_ITERATION_LAYERS。

---

## 13. **cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py**

- **用途:** 这是对逆向推理任务的一次重大升级。我不仅要求模型推断出**什么**规则被应用了，还要推断出它被**应用了多少次**。
    
- **逻辑:** 这个脚本继承了上一个实验的思想，但增加了复杂性。在生成每个样本时，我随机选择一个规则和**一个随机的迭代次数**。在得到输入/输出对后，我引入了一个至关重要的**唯一性验证步骤**：我会暴力检查是否存在任何其他的规则/迭代次数组合，也能从同一个初始状态得到完全相同的最终状态。我只保留那些解是唯一的样本，从而为模型提供了一个无歧义的学习目标。
    
- **I/O格式:**
    
    - 输入: CA_WIDTH * 2长度的二进制字符串。
        
    - 输出: 8 + ITERATION_BITS长度的二进制多标签向量，拼接了规则和迭代次数。
        
- **主要参数:** CA_WIDTH, NUM_SAMPLES, MAX_ITERATION_LAYERS。

---

## 14. **generate_cellular_automata_1d_perturbed.py**

- **用途:** 该脚本旨在系统性地测试“神经雕刻”范式在面对不完美数据时的鲁棒性。通过向输入（模拟观测噪声）和输出（模拟标签噪声）中引入可控的随机扰动，它探索了模型性能从理想规则世界向嘈杂现实世界过渡的连续谱。
    
- **逻辑:** 脚本首先生成一个原始的元胞自动机初始状态。然后，它根据INPUT_PERTURBATION_RATE对这个原始状态进行随机翻转，得到最终的“输入”序列。接着，它基于未受扰动的原始状态，根据精确的元胞自动机规则演化EVOLUTION_LAYERS步，得到一个“正确”的输出序列。最后，它根据OUTPUT_PERTURBATION_RATE对这个正确的输出序列进行随机翻转，得到最终的“输出”标签。
    
- **I/O格式:**
    
    - 输入: 长度为LENGTH的0/1字符串。
        
    - 输出: 长度为LENGTH的0/1整数列表。
        
- **主要参数:** NUM_SAMPLES, LENGTH, EVOLUTION_LAYERS, INPUT_PERTURBATION_RATE, OUTPUT_PERTURBATION_RATE。

---

## 15. **generate_ca110_full_trace.py**

- **用途:** 这是一个**深度分析**实验，旨在生成元胞自动机演化的完整轨迹，用于研究神经网络内部表示与演化步骤之间的关系（即“神经心智扫描仪”实验）。
    
- **逻辑:** 脚本模拟Rule 110的演化过程，但不仅仅记录最终状态。它将从第1步到第N步的所有中间状态都拼接起来作为输出标签。这使得我们可以训练一个模型来一次性预测整个演化历史。
    
- **I/O格式:**
    
    - 输入: N位二进制字符串 (初始状态)。
        
    - 输出: TOTAL_LAYERS * N位二进制多标签向量 (完整轨迹)。
        
- **主要参数:** NUM_BITS, TOTAL_LAYERS。

---

## 16. **generate_ca_text_format_dataset.py**

- **用途:** 这是一个**自回归**实验，旨在测试标准的Decoder-only Transformer（如GPT）是否能通过文本形式的Prompt来学习元胞自动机规则。
    
- **逻辑:** 脚本生成元胞自动机的初始状态和最终状态，并将它们格式化为类似 "Evolve this: [Input] -> [Output]" 的文本字符串。这使得任务变成了标准的文本生成任务。
    
- **I/O格式:**
    
    - 输出: 包含 "text" 字段的JSON对象。
        
- **主要参数:** NUM_BITS, TOTAL_LAYERS。

---

## 17. **generate_cellular_automata_image_and_label.py**

- **用途:** 这是一个**通用数据集生成器**，用于为元胞自动机任务同时生成图像格式（Img2Img）和符号格式（Img2Label）的数据，支持多进程加速。
    
- **逻辑:** 脚本并行生成随机初始状态，将其渲染为图像。然后进行演化，将最终状态也渲染为图像，并保留其符号形式。最终生成包含输入图像、输出图像和输出标签的元数据。
    
- **I/O格式:**
    
    - 输出: 图像文件对 (initial/final) 和 metadata.csv。
        
- **主要参数:** CA_WIDTH, RULE_NUMBER, ITERATIONS, NUM_SAMPLES。

---

## 18. **generate_mnist_ca_110.py**

- **用途:** 这是一个**感知与推理融合**的实验。它将MNIST手写数字作为元胞自动机的“细胞”，测试模型是否能在识别数字的同时执行逻辑演化。
    
- **逻辑:** 脚本生成一个元胞自动机状态，但不是用黑白像素表示0和1，而是从MNIST数据集中随机选取数字'0'和'1'的图像贴在对应位置。模型必须先识别出这些数字，理解其代表的逻辑状态，然后进行演化。
    
- **I/O格式:**
    
    - 输入: 由MNIST数字拼贴而成的图像。
        
    - 输出: 演化后的黑白状态图像。
        
- **主要参数:** RULE, STEP_EVOL。

---

## 19. **generate_rulemnist_ca.py**

- **用途:** 这是一个更高级的**感知与推理融合**实验，引入了**规则控制**。
    
- **逻辑:** 输入图像包含一个大的MNIST数字（0或1）作为背景，以及叠加在上面的元胞自动机状态。MNIST数字决定了演化规则（例如0代表Rule 30，1代表Rule 110）。模型必须识别背景数字来选择规则，识别前景状态来执行演化。
    
- **I/O格式:**
    
    - 输入: 叠加了CA状态的MNIST图像。
        
    - 输出: 演化后的状态图像。
        
- **主要参数:** RULE_MAP, RULES。

---

## 20. **cellular_automata/generate_ca_continuous_transform.py**

- **用途:** 这是一个**连续-离散-连续**混合的元胞自动机实验，专注于测试 MLP 等模型处理"模糊逻辑"输入的能力。

- **逻辑:** 
    1.  **模糊输入:** 随机生成30个实数，[0, 0.25] 代表逻辑 '0'，[0.75, 1.0] 代表逻辑 '1'。
    2.  **符号演化:** 将输入的逻辑值按 Rule 110 演化5层，得到输出逻辑值。
    3.  **条件变换:** 根据演化结果决定输出实数的值：如果某位逻辑值未变，则输出等于输入；如果逻辑值变了，则输出等于 `1.0 - 输入`。
    4.  这迫使模型必须同时学会"去模糊化"(解码连续值)、"逻辑推理"(CA演化)和"数值变换"(内插)。

- **I/O格式:**
    
    - 输入: 30个浮点数列表。
        
    - 输出: 30个浮点数列表。
        
- **主要参数:** CA_WIDTH, CA_LAYERS, NUM_SAMPLES。
    
- **注意:** 该数据集包含连续实数，请使用MLP或支持浮点输入输出的模型进行训练。

---

## 21. **cellular_automata/generate_ca_nested_masked.py**

- **用途:** 生成"俄罗斯套娃"式（嵌套掩码）的元胞自动机数据集，测试模型在动态掩码条件下进行多阶段隐式推理的能力。
- **逻辑:** 过程分为三阶段：1. 初始演化：生成随机初始状态 $S_0$，按Rule 110演化L1步得到 $S_1$。 2. 动态掩码：生成随机掩码 $M$，提取 $S_1$ 中的可见部分。 3. 二次演化：将提取的部分作为新初始状态，再次演化L2步得到最终输出。模型必须学会隐式地推演第一阶段并处理掩码，才能正确预测第二阶段的结果。
- **I/O格式:**
    - 输入: CA1_WIDTH位初始状态 + CA1_WIDTH位掩码 (拼接字符串)。
    - 输出: MASK_VISIBLE_BITS位最终状态 (二进制列表)。
- **主要参数:** CA1_WIDTH, CA1_LAYERS, MASK_VISIBLE_BITS, CA2_LAYERS。

---

## 22. **cellular_automata/generate_ca_masked_output.py**

- **用途:** 生成"部分可观测"的元胞自动机数据集，测试模型根据动态掩码选择性输出计算结果的能力。
- **逻辑:** 1. 生成随机初始状态 $S_0$。 2. 按Rule 110演化3层得到 $S_3$。 3. 生成随机掩码 $M$。 4. 输出 $S_3$ 中被 $M$ 选中的部分（Masked Output）。模型不需要输出完整的演化结果，只需要从隐藏的完整状态中"提取"出被查询的部分。
- **I/O格式:**
    - 输入: CA1_WIDTH位初始状态 + CA1_WIDTH位掩码 (拼接字符串)。
    - 输出: MASK_VISIBLE_BITS位最终状态 (二进制列表)。
- **主要参数:** CA1_WIDTH, CA_LAYERS, MASK_VISIBLE_BITS。

---

## 23. **cellular_automata/generate_ca_to_abstract_real.py**

- **用途:** 生成"CA演化 → 抽象实数符号"的数据集，测试模型能否将离散演化结果编码为任意实数符号表示，验证神经网络对符号抽象性的理解。

- **逻辑:** 
    1. **输入：** 30位二进制字符串（元胞自动机初始状态）。
    2. **演化：** 按Rule 110演化5层，得到30位最终状态。
    3. **编码转换：** 将最终的30位二进制状态按每2位一组分割（共15组），根据预定义的任意实数映射表，将每组转换为一个实数。映射关系：
        - `(0,0) → 7.6`
        - `(0,1) → 1.3`
        - `(1,0) → 5.9`
        - `(1,1) → 3.0`
    4. **输出：** 15个实数的列表。

- **实验意义:** 该任务强制模型必须：
    - 首先学会元胞自动机的演化规则。
    - 然后学会将离散的二进制模式映射到完全任意的、无语义关联的实数符号上。
    - 这验证了模型学习的是抽象关系而非具体符号，与论文中的"语义洗牌"实验形成呼应。

- **I/O格式:**
    - 输入: 30位二进制字符串。
    - 输出: 15个浮点数的列表（JSON数组）。

- **主要参数:** 
    - `CA_WIDTH` (30): 元胞自动机宽度。
    - `CA_LAYERS` (5): 演化层数。
    - `REAL_SYMBOL_MAP`: 2位二进制到实数的映射表。
    - `NUM_SAMPLES` (500000): 数据集大小。

- **训练建议:** 使用支持浮点输出的模型（如MLP），配合`train_mlp.py`训练。需要特别关注模型能否收敛，因为输出不再是离散的0/1，而是连续的任意实数。

---

# E: 物理模拟 (Physics Simulation)

## 1. **physics_simulation/generate_projectile_motion_simulation.py**

- **用途:** 测试模型学习一个简单的动态物理过程的能力。这要求模型从初始条件（位置和速度向量）推断出整个时空轨迹。
    
- **逻辑:** 输入图像通过一个起始点和一条有向线段来编码小球的初始位置和速度向量（线段方向代表速度方向，颜色代表速度大小）。脚本内部的物理引擎会根据这些初始条件模拟出小球在重力场中的抛物线弹跳轨迹。输出图像则绘制了完整的轨迹。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含初始状态)。
        
    - 输出: IMG_SIZE x IMG_SIZE的RGB图像 (包含完整轨迹)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE, GRAVITY, ELASTICITY_FACTOR。

---

## 2. **physics_simulation/generate_snell_refraction_simulation.py**

- **用途:** 测试模型学习基础物理定律（斯涅尔折射定律）的能力。
    
- **逻辑:** 输入图像包含两种不同颜色的介质，以及一条射向它们分界面的入射光线。脚本根据斯涅尔定律（n1sin(θ1) = n2sin(θ2)）精确计算出折射光线的路径。模型的任务是根据输入图像，预测出正确的折射光线。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含两种介质和入射光)。
        
    - 输出: 与输入相同，但额外绘制了红色的折射光线。
        
- **主要参数:** IMG_SIZE, NUM_SAMPLES_TRAIN。

---

## 3. **physics_simulation/generate_snell_refraction_with_contextual_index.py**

- **用途:** 测试模型学习基础物理定律（斯涅尔折射定律）的能力，并且要求模型能从图像的上下文信息（背景颜色）中推断出物理参数（折射率）。
    
- **逻辑:** 输入图像包含两种不同颜色的介质，以及一条射向它们分界面的入射光线。脚本根据斯涅尔定律精确计算出折射光线的路径。在此版本中，其中一种介质的颜色与它的折射率n2是函数关系。模型的任务是根据输入图像，预测出正确的折射光线。
    
- **I/O格式:**
    
    - 输入: IMG_SIZE x IMG_SIZE的RGB图像 (包含两种介质和入射光)。
        
    - 输出: 与输入相同，但额外绘制了红色的折射光线。
        
- **主要参数:** IMG_SIZE, NUM_SAMPLES_TRAIN。

---

## 4. **physics_simulation/generate_reaction_diffusion_deprecated.py**

- **用途:** (探索性/已弃用) 该脚本用于模拟基于 **Gray-Scott 模型** 的反应-扩散系统，以生成复杂的、类似分形的“雪花”图案。它本质上属于连续物理场的模拟（涉及偏微分方程的离散化和浮点运算），而非简单的离散状态元胞自动机。
    
- **状态:** **已弃用**。虽然物理模拟逻辑正确，但此任务的输出是动态演化的结果，且对初始条件高度敏感，不符合范式中研究的“从一个明确输入到唯一确定性输出”的映射关系，因此被判定为“不合适”并放弃。此外，因其核心逻辑基于连续物理模拟，现已从元胞自动机分类移至物理模拟分类。
    
- **逻辑:** 脚本从一个或几个“晶核”开始，通过迭代模拟养分（Nutrient）和物质（Matter）两个连续场的扩散和反应（使用卷积计算拉普拉斯算子），逐步生成复杂的固态结构。

---

## 5. **physics_simulation/generate_cube_rotation_matplotlib_deprecated.py**

- **用途:** (早期探索版本) 旨在测试模型从抽象的姿态参数（旋转角度）推理并渲染出三维物体正确视图的能力。
    
- **逻辑:** 此脚本使用 matplotlib 的3D绘图引擎来渲染立方体。它直接在3D空间中构建和旋转对象。虽然功能上可行，但由于 matplotlib 对渲染层次的控制较为复杂，可能导致在某些角度下线框和填充面的遮挡关系出现非预期的视觉效果。
    
- **状态:** **已弃用**。被基于 Pillow 的、渲染效果更可控的后续版本所取代。

---

## 6. **physics_simulation/generate_cube_rotation_pillow_v1.py**

- **用途:** (技术升级版本) 旨在测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，采用了更底层的、渲染效果更精确的技术路线。
    
- **逻辑:** 此脚本是对此任务的一次重大技术重构。它放弃了高级的 matplotlib 库，转而使用更基础的 Pillow 库。脚本在内部手动实现了完整的3D到2D投影变换、基于向量叉乘的背面剔除（back-face culling）以及基于面平均深度的深度排序。这种方式提供了对渲染管线的完全控制，确保了所有角度下的遮挡关系和图层顺序都是物理正确的。
    
- **状态:** 这是通往最终成功版本的关键一步，但缺少了“高亮顶点”这一重要的辅助策略。

---

## 7. **physics_simulation/generate_cube_rotation_pillow_with_anchor.py**

- **用途:** (论文中使用的最终版本) 测试模型从抽象的姿态参数推理并渲染出三维物体正确视图的能力，并通过引入“视觉锚点”来辅助模型学习。
    
- **逻辑:** 该脚本继承了 physics_simulation/generate_cube_rotation_pillow_v1.py 中所有精确的、基于 Pillow 的手动渲染管线。在此基础上，它引入了一个**关键的创新**：在所有常规渲染步骤完成后，总是在一个固定的特殊顶点（如(1,1,1)角点）上绘制一个醒目的高亮标记（一个橙色圆点），无论该顶点在当前视角下是否被遮挡。这个“视觉锚点”为模型提供了一个恒定的参照，极大地帮助了模型解决旋转的内在对称性和模糊性问题，从而成功收敛。
    
- **I/O格式:**
    
    - 输入: 24位二进制字符串（3个角度 * 8位/角）。
        
    - 输出: 256x256的RGB图像。
        
- **主要参数:** NUM_SAMPLES, IMAGE_SIZE_PX, SPECIAL_VERTEX_INDEX。

---

## 8. **physics_simulation/generate_cube_rotation_pillow_wireframe.py**

- **用途:** (变体实验版本) 测试模型在更稀疏的视觉输入下，能否仅通过线框和锚点信息来学习3D旋转。
    
- **逻辑:** 该脚本是 physics_simulation/generate_cube_rotation_pillow_with_anchor.py 的一个**消融（ablation）版本**。它保留了所有核心逻辑，包括精确的线框绘制和特殊顶点的高亮，但**移除了所有面的颜色填充**。这创建了一个“线框模式”的数据集，旨在探索模型在缺少表面信息的情况下，是否依然能够理解和重建三维结构。
    
- **状态:** 这是一个用于深入分析的变体实验。

---

## 9. **physics_simulation/generate_catenary_curve_simulation_deprecated.py**

- **用途:** 这是我早期探索悬链线问题的脚本，旨在测试模型学习由物理定律确定的非线性曲线的能力。
    
- **逻辑:** 该版本是我对悬链线问题的一个初步尝试，可能通过数值求解器，从给定的两个端点和曲线长度等参数反向求解悬链线的方程。这种方法可能存在数值不稳定的问题，是早期探索性的工作，后被generate_catenary_curve_from_points.py的更稳健方法所取代。
    
- **I/O格式:**
    
    - **输入:** IMG_SIZE x IMG_SIZE的RGB图像 (包含两个端点等信息)。
        
    - **输出:** IMG_SIZE x IMG_SIZE的RGB图像 (包含生成的悬链线)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE。

---

## 10. **physics_simulation/generate_catenary_curve_from_points.py**

- **用途:** 测试模型学习由物理定律（最小势能原理）唯一确定的非线性曲线（悬链线）的能力。
    
- **逻辑:** 脚本采用了高效的“正向构造”方法：首先，它随机定义一条悬链线的数学参数a, b, c；然后，它在这条完美的曲线上随机采样三个点（两个锚点P1, P2和一个经过点P3）。输入图像只包含这三个点，输出图像则绘制了连接P1和P2并经过P3的完整悬链线段。这种方法避免了从点反解参数的复杂和不稳定过程。
    
- **I/O格式:**
    
    - **输入:** IMG_SIZE x IMG_SIZE的RGB图像 (包含三个点)。
        
    - **输出:** IMG_SIZE x IMG_SIZE的RGB图像 (三个点+悬链线)。
        
- **主要参数:** NUM_SAMPLES_TRAIN, IMG_SIZE。

---

## 11. **physics_simulation/generate_orbital_path_from_initial_state.py**

- **用途:** 测试模型学习更复杂物理定律（开普勒定律/万有引力定律）的能力。
    
- **逻辑:** 脚本首先在数学上定义一个随机的、稳定的椭圆轨道。然后，它在轨道上随机选择一个点作为行星的初始位置，并计算该点的速度向量。输入图像通过不同颜色的点和线段编码了恒星位置、行星位置、行星速度方向和速度大小。输出图像则是在此基础上，绘制出完整的椭圆轨道。
    
- **I/O格式:**
    
    - **输入:** IMG_SIZE x IMG_SIZE的RGB图像 (编码了初始状态)。
        
    - **输出:** IMG_SIZE x IMG_SIZE的RGB图像 (初始状态+完整轨道)。
        
- **主要参数:** NUM_SAMPLES, IMG_SIZE, G (引力常数)。

---

# F: ARC-AGI探索 (ARC-AGI Exploration)

## 1. **arc_agi/generate_arc_contextual_color_swap.py**

- **用途:** 测试模型从图像的局部“上下文”或“示例”中学习规则，并将其应用到同一图像的全局数据的能力。这直接模仿了ARC-AGI测试的核心理念。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。首先人工分析ARC谜题的逻辑，然后将其编程。每个输入图像的左上角都有四个色块，定义了两对颜色交换规则（例如，(0,0)的颜色和(0,1)的颜色互换）。图像的其余部分随机散布着这四种颜色的点。模型的任务是生成一张输出图像，其中所有散点的颜色都已根据左上角的规则进行了交换。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 2. **arc_agi/generate_arc_find_cross_pattern.py**

- **用途:** 测试模型在包含大量噪音的情况下进行视觉模式识别（或可称作“目标检测”）的能力。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像是一个带有大量黄色散点的红色背景。其中，一些黄点被精心布置成了3x3的十字形图案，而另一些则是随机分布的噪音。模型的任务是“去粗存精”，忽略所有噪音点，准确地找出所有隐藏的十字形图案，并在输出图像中将它们用蓝色高亮出来。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 3. **arc_agi/generate_arc_find_odd_one_out.py**

- **用途:** 测试模型执行一个复杂的“异类发现”（Find the Odd One Out）元推理任务。模型需要逐行进行模式比较，找出特例，并将其重新组合到输出中。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入是一个被划分为4行的大网格。每一行都包含四个相似的3x3小图案，其中三个是完全相同的“普通”图案，一个是“特殊”图案。模型的任务是识别出每一行中的那个“特殊”图案，并将这四个从不同行找到的特殊图案，重新排列到一个2x2的输出网格中。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM_IN, GRID_DIM_OUT, NUM_SAMPLES。

---

## 4. **arc_agi/generate_arc_connect_colored_pairs.py**

- **用途:** 测试模型在同一图像中识别多个独立“连接任务”并理解一种隐含的“图层”或“绘制优先级”规则的能力。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像中散布着若干对颜色相同的点。模型的任务是找到每一对颜色相同的点，并将它们用对应颜色的直线连接起来。一个附加的、隐藏的规则是，如果水平线和垂直线交叉，垂直线总是会绘制在水平线的上方。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 5. **arc_agi/generate_arc_conditional_perpendicular_lines.py**

- **用途:** 测试模型根据物体的**属性（颜色）**和**全局参照物（边界线、图像边缘）**来执行不同几何操作的能力。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像中有一条水平的灰色基准线，以及一些红色和蓝色的散点。模型的任务是：对于每一个**红色**的点，从该点向**灰色基准线**画一条垂线；对于每一个**蓝色**的点，从该点向**离它最近的图像水平边缘**（顶部或底部）画一条垂线。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 6. **arc_agi/generate_arc_column_projection.py**

- **用途:** 测试模型识别复杂的上下文关系（“在...下方且在...范围内”）并执行条件性列操作的能力。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像中有一个大的、特定颜色的向下箭头和一些同色的散点。模型的任务是找到所有位于箭头主体正下方的散点。然后，对于每一个包含这种“合格”散点的**垂直列**，将输出图像中该列从箭头底部到图像底部的所有像素都涂上投影颜色。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 7. **arc_agi/generate_arc_procedural_spiral.py**

- **用途:** 测试模型执行一个迭代的、程序性的生成算法的能力。模型需要理解指令、跟踪状态（当前位置、方向、长度）并循环执行。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像非常简洁：左上角有两个颜色块（颜色A和颜色B）作为指令，以及一个蓝色的点作为绘图的“起点”。模型的任务是从这个蓝点开始，生成一个不断向外扩展的螺旋。螺旋的绘制遵循严格的规则：第一段线（长度2）向左，颜色为A；第二段（长度2）向下，颜色为B；第三段（长度3）向右，颜色为A，以此类推。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 8. **arc_agi/generate_arc_fractal_stamping.py**

- **用途:** 测试模型理解和执行递归或分形生成规则的能力。模型需要将输入图案本身作为一个“笔刷”，根据输入图案中的“指令”进行重复绘制。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入是一个4x4的图案。输出是一个16x16的更大画布。规则是：遍历输入4x4图案中的每一个格子，如果(r, c)位置的格子是**红色**的，那么就把整个4x4的输入图案完整地复制（“盖章”）到输出画布上以(r4, c4)为左上角的位置。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM_IN, GRID_DIM_OUT, NUM_SAMPLES。

---

## 9. **arc_agi/generate_arc_flood_fill.py**

- **用途:** 测试模型执行经典的“洪水填充”（Flood Fill）或“油漆桶”算法的能力。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。脚本首先在黑色背景上程序化地生成一个由绿色“墙壁”围成的、保证连通的封闭区域。输入图像就是这张带有绿色围墙的图。输出图像则是在输入的基础上，将这个绿色围墙所包围的黑色区域完全填充为黄色。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 10. **arc_agi/generate_arc_layered_fill.py**

- **用途:** 测试模型理解一个程序性极强的、依赖于拓扑距离和条件判断的复杂填充算法。
    
- **逻辑:** 采用“逻辑编程辅助学习”的策略。输入图像被线段分割成多个区域。每个区域内有一到两个“颜色指令点”。如果只有一个颜色点（A），则整个区域填充为颜色A。如果有两个颜色点（A和B），则对该区域进行“分层”填充：离区域边界最近的一层涂上颜色A，次近的一层涂上颜色B，以此类推，形成交替色的“等高线”图案。
    
- **I/O格式:** 图像到图像 (Image-to-Image)。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 11. **arc_agi/generate_arc_fluid_simulation.py**

- **用途:** 测试模型在图像空间中学习和模拟一个具有特定规则的流体动态过程的能力。
    
- **逻辑:** 输入图像中包含若干红色的水平“挡板”，以及顶部的一或两个紫色“水龙头”。模型的任务是模拟从水龙头流出的紫色液体，当液体遇到挡板时会向左右两边分流，在没有挡板支撑的边缘则会继续向下滴落，直到图像底部。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 12. **arc_agi/generate_arc_periodic_conditional_fill.py**

- **用途:** 这个实验旨在测试模型学习一个复杂的、带有周期性和特殊case的条件格式化规则的能力。
    
- **逻辑:** 输入图像最底部有一条黄色线段，它定义了一个“操作区域”。模型需要从下到上逐行检查。根据当前行到倒数第二行的距离d，应用模6的周期性规则：
    
    - d % 6为1或5: 在操作区域内填充黄色。
        
    - d % 6为0,2,4: 在操作区域内填充背景色。
        
    - d % 6为3 (特殊规则): 不仅在操作区域内填充**绿色**，还要将该行**所有**原始的散点都变为**绿色**。
        
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 13. **arc_agi/generate_arc_fill_square_holes.py**

- **用途:** 这个实验用于测试模型进行多步视觉推理的能力：首先需要识别出复杂的“背景中的前景”（即矩形中的空洞），然后对识别出的对象进行几何属性判断（是否为正方形），最后根据判断结果进行着色。
    
- **逻辑:** 输入图像包含多个灰色的、带有黑色空洞的矩形。模型的任务是识别出所有空洞，判断每个空洞的形状，如果某个空洞是**正方形**，则在输出图像中将其用红色填充。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 14. **arc_agi/generate_arc_conditional_recoloring.py**

- **用途:** 测试模型理解视觉图层和进行条件性对象属性修改的能力。
    
- **逻辑:** 输入图像包含一个由深蓝色散点和黑色背景构成的“底层”，以及一个浅蓝色矩形作为“标记层”。这个标记层叠加在底层之上，但只覆盖了黑色的背景，并未改变原始的深蓝色散点。模型的任务是，识别出这个浅蓝色矩形的区域，并找到所有位于该区域内的、属于**底层**的深蓝色散点，然后在输出图像中将这些散点的颜色变为绿色。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 15. **arc_agi/generate_arc_sort_by_length_remap_position.py**

- **用途:** 测试模型执行一个“属性-位置解耦与重映射”的复杂排序任务。
    
- **逻辑:** 输入图像中有一系列颜色、长度、位置都不同的彩色垂直柱子。模型的任务是：
    
    1. 在概念上，提取所有柱子的**长度**属性并排序。
        
    2. 同时，保持所有柱子原始的水平**位置**和**颜色**属性不变。
        
    3. 在输出图像中，将**最短**的柱子画在**最左边**原始柱子的位置上，用该位置原始的颜色；将**第二短**的柱子画在**第二左**原始柱子的位置上，用该位置原始的颜色，以此类推。
        
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 16. **arc_agi/generate_arc_jigsaw_puzzle_simple.py**

- **用途:** 测试模型解决一个视觉匹配与变换问题的能力（早期版本）。
    
- **逻辑:** 输入图像左侧是一个被挖掉了几个拼图块的模板，右侧散落着对应拼图块的放大、随机旋转/镜像后的版本。此版本的**关键妥协**是，为了简化匹配问题，每个拼图块都有独一无二的尺寸（方格数），模型可以利用这个“捷径”来识别对应关系，而非完全依赖形状。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES, num_source_pieces。

---

## 17. **arc_agi/generate_arc_jigsaw_puzzle_advanced.py**

- **用途:** 测试模型解决一个复杂的**视觉匹配与变换**问题的能力。
    
- **逻辑:** 这是对jigsaw_puzzle任务的重大改进。输入图像左侧是一个被挖掉了几个拼图块的模板，右侧散落着对应拼图块的2倍放大、随机旋转和镜像后的版本，以及一些噪音块。**关键改进**是，此版本允许生成多个**尺寸相同但形状不同**的拼图块，这迫使模型必须**真正地根据形状**来进行匹配。输出图像是将所有拼图块正确地缩小并放回模板后的样子。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES, num_source_pieces。

---

## 18. **arc_agi/generate_arc_connect_path_by_sequence.py**

- **用途:** 测试模型解析外部指令序列，并据此在图像中执行多步、有状态的路径连接任务的能力。
    
- **逻辑:** 输入图像包含两部分：(1) 画布上散落着多个内部有颜色的正方形；(2) 图像最下方有一排颜色块，这是一个“指令序列”。模型的任务是按照指令序列的颜色顺序，依次连接对应的正方形。例如，如果指令是[红, 绿, 蓝]，模型需要先从红色的方块画一条线到绿色的方块，再从绿色的方块画一条线到蓝色的方块。一个附加规则是，每段连接线的颜色，都由**前一个**方块的颜色决定。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** GRID_DIM, NUM_SAMPLES。

---

## 19. **arc_agi/generate_arc_reflection_simulation_deprecated.py**

- **用途:** (已废弃) 旨在测试模型理解复杂的基于物理光学的规则，包括射线发射、碰撞检测、角度反射和颜色变换。
    
- **逻辑:** 该脚本尝试为这个非常复杂的ARC任务编写生成脚本，但由于以程序方式无歧义地生成所有可能的、物理正确的反射和交互场景是极其困难的，无法保证生成数据的质量和一致性，最终放弃了这个实验。
    
- **I/O格式:** 图像到图像。
    
- **主要参数:** 不适用。

---

# G: 中国象棋 (Chinese Chess)

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


# 训练脚本和工具脚本 (Training Scripts and Tools)

---

## 1. **utils/eval_hanoi.py**

- **用途:** 这是一个**验证工具**，而非训练脚本。它的功能是接收一个大语言模型（或其他来源）生成的汉诺塔问题解法字符串（如 "1>3;1>2;..."），并严格按照汉诺塔的游戏规则进行模拟，以判断该解法是否正确。

- **核心特性:**
    - **状态模拟:** 在内部模拟三个柱子和n个盘子的状态。
    - **规则检查:** 自动检查每一步移动是否合法（如：不能从空柱子移出，大盘不能放在小盘上）。
    - **最终状态验证:** 检查所有移动完成后，是否所有盘子都按正确顺序移动到了目标柱子。
    - **清晰的错误提示:** 如果解法有误，会明确指出错误在哪一步以及原因。

- **使用方法:**
    1. **作为命令行工具:**
        - 打开eval_hanoi.py文件。
        - 在文件底部的if __name__ == "__main__":部分，找到verify_hanoi_solution(n, solution_str)函数调用。
        - 将第一个参数n修改为你想要验证的盘子数量。
        - 将第二个参数solution_str替换为你从大模型获取的解法字符串。
        - 运行脚本：`python eval_hanoi.py`
        - 控制台会输出✅ 正确!或❌ 错误:以及详细信息。

    2. **作为库导入:**
        ```python
        from eval_hanoi import verify_hanoi_solution
        
        n = 6
        llm_output = "1>2;1>3;..." # 从你的模型获取输出
        is_correct = verify_hanoi_solution(n, llm_output)
        print(f"LLM的解法是否正确: {is_correct}")
        ```

---

## 2. **utils/create_videos.py**

- **用途:** 这是一个**训练过程可视化视频生成工具**，用于将模型训练过程中生成的eval image序列转换为视频，方便观察和分析模型的学习动态。

- **核心功能:**
    - **批量处理:** 自动为多个样本（最多32个）生成独立的演化视频。
    - **灵活配置:** 支持自定义起始step、结束step、步长间隔和视频帧率。
    - **高质量输出:** 使用FFmpeg生成高质量的MP4视频，支持自定义编码参数。

- **主要配置参数:**
    - `IMAGE_DIR`: 存放eval image的目录路径
    - `OUTPUT_DIR`: 视频输出目录（自动创建）
    - `NUM_SAMPLES`: 要处理的样本数量（0-31）
    - `START_STEP`: 视频起始训练步数
    - `END_STEP`: 视频结束训练步数
    - `STEP_INTERVAL`: 采样间隔（如每20步取一张图）
    - `VIDEO_FRAMERATE`: 输出视频帧率（控制播放速度）
    - `FFMPEG_PATH`: FFmpeg可执行文件路径

- **使用方法:**
    1. **配置参数:** 在脚本开头的配置区修改相关参数
    2. **准备数据:** 确保IMAGE_DIR中有按命名规则保存的eval image（如step_100_sample_0.png）
    3. **运行脚本:** `python utils/create_videos.py`
    4. **查看结果:** 在OUTPUT_DIR中找到生成的MP4视频文件

- **典型应用场景:**
    - 观察模型在ARC-AGI任务上的学习过程
    - 分析元胞自动机演化任务的收敛模式
    - 制作论文或演示所需的训练动态可视化素材
    - 对比不同模型或超参数下的学习速度差异

- **技术实现:**
    - 使用FFmpeg的concat协议批量处理图像序列
    - 自动生成临时文件列表，避免命令行长度限制
    - 支持错误处理和进度提示
    - 自动清理临时文件

---

## 3. **utils/analyze_ca_inverse_ambiguity.py**

- **用途:** 这是一个**蒙特卡洛模拟分析工具**，用于定量估算一维元胞自动机（CA）逆向工程任务中的**歧义概率**——即不同的(规则, 演化层数)组合产生相同输出的概率。

- **研究背景:** 该工具支撑了论文中关于"唯一性验证"必要性的论述。在 `cellular_automata/generate_cellular_automata_inverse_rule_and_steps_unique.py` 脚本中，我们会过滤掉那些解不唯一的样本。本工具通过大规模采样定量估算了这种歧义发生的概率，为实验设计提供理论依据。

- **核心算法:**
    - 对每个随机初始状态，遍历256种规则 × 4层深度 = 1024种组合
    - 统计产生相同输出的组合数（即"碰撞"）
    - 通过20万次采样估算平均歧义概率
    - 使用迭代优化技巧，每个规则只需4次CA演化（而非16次）

- **使用方法:**
    ```bash
    # 默认参数运行（20万采样，30位宽度）
    python utils/analyze_ca_inverse_ambiguity.py
    
    # 自定义参数
    python utils/analyze_ca_inverse_ambiguity.py --samples 100000 --length 36
    
    # 静默模式（仅输出最终结果）
    python utils/analyze_ca_inverse_ambiguity.py --quiet
    ```

- **输出示例:**
    ```
    =================================================================
    1D Cellular Automata Inverse Engineering Ambiguity Simulation
    =================================================================
    Parameters: length=30, samples=200,000
    ...
    **Estimated ambiguity probability**: 0.0000009530 (0.00009530%)
    95% Confidence Interval: [0.0000008912, 0.0000010148]
    ```

- **主要参数:**
    - `--samples`: 采样数量（默认200000）
    - `--length`: CA状态宽度（默认30位）
    - `--quiet`: 静默模式

---

## 4. **chinese_chess/worker_logic.py**

- **用途:** 这是一个核心的**引擎交互与任务调度模块**，支撑了从引擎到数据的桥梁建设。它既用于生成FEN局面，也用于计算走法的软标签。

- **逻辑:** 
    - **封装 Pikafish 引擎:** `PikaFishEngineFinal` 类封装了与 Pikafish 国际水平象棋引擎的 UCI 通信协议。它通过多进程安全的管道交互，发送 `go depth` 等指令，并健壮地解析引擎返回的 `bestmove` 和 `score` 信息。
    - **确定性打破:** 为了生成多样化的数据，引擎初始化时会动态设置 Hash 大小，打破确定性。
    - **多进程 Worker:** 
        - `worker_label_generation`: 专为 `generate_soft_labels.py` 设计的 Worker 函数。它接收分配到的 FEN 块，启动独立的引擎进程，计算每个局面的 MultiPV（多候选）走法及其评分，生成软标签，并写入临时文件。

- **关键特性:** 高效的多进程支持、健壮的 UCI 协议处理、支持 MultiPV 分数获取。

---

## 5. **chinese_chess/generate_soft_labels.py**

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

## 6. **training_scripts/train_chess_policy.py**

- **用途:** 这是**策略网络（Policy Network）**的训练脚本。它训练一个 Transformer 模型，使其能够根据盘面（FEN）预测下一步的最佳走法概率分布。

- **逻辑:**
    - **Policy Transformer:** 定义了一个基于 Encoder-only Transformer（类似 BERT）的神经网络架构。
        - **输入:** 将 FEN 字符串 Token 化。
        - **输出:** 通过一个线性头（Policy Head）输出所有合法走法（Vocab 大小）的 Logits。
    - **高效数据流:** 实现了 `NpzChunkDataset`，支持按块懒加载大规模 NPZ 格式的训练数据，大大降低内存占用。
    - **训练目标:** 使用 Soft CrossEntropy（即 KL 散度）逼近引擎生成的软标签概率分布。这比单纯学习 One-hot 的“最佳走法”包含更多信息量。

- **I/O格式:**
    - 输入: 包含 `fens` 和 `labels` 数组的 .npz 文件目录。
    - 输出: 训练好的模型权重 (`pytorch_model.bin`) 和配置。

- **主要参数:** `train_data_dir`, `num_epochs`, `learning_rate`, `model_config`。

---

## 7. **chinese_chess/play_with_ai.py**

- **用途:** 这是一个**人机对弈终端客户端**，用于测试和评估训练好的策略网络。

- **逻辑:**
    - **加载模型:** 自动加载最新的训练检查点。
    - **局面编码:** 将当前棋盘状态（FEN）实时编码为模型输入张量。
    - **策略采样:** 模型输出下一步所有走法的概率分布。脚本根据 `sampling_temperature` 对概率进行调整，并使用多项式采样（Multinomial Sampling）选择走法，从而实现多样化且高水平的下棋风格。
    - **交互循环:** 提供简单的命令行界面，处理用户输入（UCI格式，如 `h2e2`）和 AI 走法的显示。

- **交互方式:** 命令行输入。
- **主要配置:** `model_dir`, `sampling_temperature`。
