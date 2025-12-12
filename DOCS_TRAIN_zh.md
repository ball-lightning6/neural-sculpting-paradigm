### 1. train_tiny_transformer.py

**▶︎ 简要说明**  
该脚本用于训练一个自定义的、从零开始构建的TinyTransformer模型，专注于解决**符号到符号 (Sequence-to-Vector)** 的任务，如符号规则学习、算法拟合等。它接收一个字符序列输入，并输出一个固定长度的多标签二分类结果。

**▶︎ 核心架构**

- **Encoder**: TinyTransformerForCausalLM，一个为自回归任务设计的轻量级Transformer。
    
- **Output Head**: 模型的lm_head被替换为一个nn.Linear层，以匹配任务的num_labels。
    
- **Loss**: BCEWithLogitsLoss，用于多标签二分类。
    
- **Tokenization**: 采用简单的字符级编码 (ord(char))，无需预训练的分词器。
    

**▶︎ 如何配置和使用**

1. **修改超参数**: 在脚本中找到 hyperparams 字典，按需调整 num_epochs, train_batch_size, learning_rate 等。
    
2. **指定数据集**: 修改 dataset = LightsOutDataset("your_dataset.jsonl", tokenizer) 这一行，指向你的.jsonl数据集文件。文件格式应为每行一个JSON对象，包含"input"和"output"键。
    
3. **设置输出维度**: 修改 num_labels = N，其中N是你任务输出的二进制向量长度。
    
4. **运行训练**:
    
    codeBash
    
    ```
    python train_tiny_transformer.py
    ```
    
5. **产出**: 训练日志会打印在控制台，模型检查点会保存在output_dir指定的目录中。
    

---

### 2. train_swin_image2text.py

**▶︎ 简要说明**  
此脚本用于**图像到符号 (Image-to-Vector)** 的任务，通过全量微调（Full Fine-tuning）一个预训练的Swin Transformer，使其能够完成多标签图像分类。

**▶︎ 核心架构**

- **Model**: Hugging Face AutoModelForImageClassification，加载microsoft/swin-*系列模型。
    
- **Processor**: Hugging Face AutoImageProcessor，自动处理Swin模型所需的图像预处理。
    
- **Loss**: BCEWithLogitsLoss，用于多标签分类。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在脚本顶部的配置区域修改以下变量：
    
    - MODEL_NAME: 指定要使用的Swin模型，如 microsoft/swin-tiny-patch4-window7-224。
        
    - NUM_LABELS: 任务输出的二进制向量长度。
        
    - IMAGE_DIR, LABEL_DIR/METADATA_PATH: 指向你的数据集路径。脚本内置了两种Dataset，根据你的数据格式注释掉不用的那个。
        
    - BATCH_SIZE, LEARNING_RATE 等训练参数。
        
2. **准备数据**: 确保你的数据集目录结构符合所选Dataset类的要求。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_swin_image2text.py
    ```
    
4. **产出**: 检查点（best_model.pth 和 final_model.pth）和日志文件会保存在OUTPUT_DIR。
    

---

### 3. train_unet.py

**▶︎ 简要说明**  
该脚本用于纯粹的**图像到图像 (Image-to-Image)** 任务，训练一个标准的U-Net模型。它的一大特点是在验证阶段会自动生成（输入 | 目标 | 预测）的三联图，方便直观地评估模型性能。

**▶︎ 核心架构**

- **Model**: 一个从零实现的经典UNet。
    
- **Loss**: MSELoss (或可切换为 BCEWithLogitsLoss 等)。
    
- **可视化**: 验证循环中集成了save_image功能，用于生成对比图。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 所有配置项都在Config类中。
    
    - DATASET_DIR: 指向你的数据集根目录。
        
    - OUTPUT_DIR: 指定所有训练产出的保存位置。
        
    - EPOCHS, BATCH_SIZE, LEARNING_RATE 等。
        
2. **准备数据**: DATASET_DIR下应包含initial_images/、final_images/子目录和一个metadata.csv文件。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_unet.py
    ```
    
4. **产出**: OUTPUT_DIR中会包含日志文件 (training_log_unet.log) 和一个eval_images目录，里面存放着每个评估步骤生成的三联对比图。
    

---

### 4. train_text2image.py

**▶︎ 简要说明**  
该脚本用于**符号到图像 (Text-to-Image)** 的任务，训练一个完全从零构建的模型。该模型由一个轻量级TinyTransformer作为文本编码器和一个U-Net风格的解码器组成。

**▶︎ 核心架构**

- **Text Encoder**: TinyTransformerForCausalLM，与train_tiny_transformer.py中的模型相同。
    
- **Image Decoder**: ImageDecoder，一个U-Net风格的解码器，它通过线性投影将文本特征向量“注入”到解码过程的各个阶段作为跳跃连接。
    
- **Tokenization**: 同样使用简单的字符级编码 (ord(char))。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在脚本顶部的配置区域修改：
    
    - TINY_TRANSFORMER_CONFIG: 配置Transformer的结构。
        
    - IMAGE_SIZE, OUTPUT_CHANNELS: 配置输出图像的属性。
        
    - DATASET_DIR: 指向数据集根目录。
        
    - OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE 等。
        
2. **准备数据**: DATASET_DIR下应包含images/子目录和一个metadata.csv文件。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_text2image.py
    ```
    
4. **产出**: 检查点目录OUTPUT_DIR下会包含日志文件和eval_predictions目录，其中存放着（目标图像 | 生成图像）的二联对比图。
    

---

### 5. train_qwen2_text2image.py

**▶︎ 简要说明**  
这是一个更强大版本的**符号到图像 (Text-to-Image)** 脚本。它使用预训练的Qwen2大语言模型作为文本编码器，通过PEFT LoRA进行高效微调，并结合了与前者相同的U-Net风格解码器。整个训练流程由Hugging Face Trainer管理。

**▶︎ 核心架构**

- **Text Encoder**: Hugging Face AutoModelForCausalLM 加载Qwen2，并使用peft库应用LoRA。
    
- **Image Decoder**: ImageDecoder，与train_text2image.py中的解码器相同。
    
- **Trainer**: 使用自定义的ImageGenTrainer，重写了compute_loss方法以处理图像重建损失，并由Hugging Face Trainer API驱动。
    
- **Callbacks**: 包含一个SaveImagePredictionCallback回调，用于在评估时自动保存对比图。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在脚本顶部的配置区域修改：
    
    - TEXT_MODEL_NAME: 指定要使用的Qwen2模型版本，如qwen2-0.5b。
        
    - LORA_R, LORA_ALPHA: LoRA配置。
        
    - IMAGE_SIZE, DATASET_DIR, OUTPUT_DIR 等。
        
2. **登录Hugging Face (如果需要)**: 如果模型是私有的，需要先通过huggingface-cli login登录。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_qwen2_text2image.py
    ```
    
1. **产出**: OUTPUT_DIR中会生成标准的Hugging Face训练输出，包括checkpoint-*目录、日志，以及由回调生成的eval_predictions对比图目录。LoRA适配器和解码器权重会在训练结束后单独保存。


---

### 6. train_mlp.py

**▶︎ 简要说明**  
此脚本用于训练一个“巨型”MLP（多层感知机），解决**符号到符号**的任务。其主要目的是为更复杂的架构（如Transformer, RNN）提供一个无结构偏置的性能基准（baseline）。

**▶︎ 核心架构**

- **Model**: 一个深度和宽度都较大的MLP，包含Linear, GELU, LayerNorm, Dropout层。
    
- **Data**: 从.jsonl文件加载符号数据，每行包含'input'和'output'字符串。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在Config类中调整：
    
    - DATASET_PATH: 指向你的.jsonl数据集文件。
        
    - BITS: 输入和输出的向量维度。
        
    - HIDDEN_SIZE, NUM_HIDDEN_LAYERS: 调整MLP的大小以匹配其他模型的参数量。
        
    - BATCH_SIZE, LEARNING_RATE等训练参数。
        
2. **运行训练**:
    
    codeBash
    
    ```
    python train_mlp.py
    ```
    
3. **产出**: 控制台输出训练进度，日志保存在training_log_mlp.log。
    

---

### 7. train_lstm.py

**▶︎ 简要说明**  
该脚本使用LSTM（或可切换为GRU/RNN）来解决**符号到符号**的任务。其设计巧妙地测试了RNN的**时序演化和记忆能力**：模型接收一次性输入，然后在内部进行EVOLUTION_STEPS次迭代，最后输出结果。

**▶︎ 核心架构**

- **Model**: RNNModel，包含一个输入编码器、一个RNN核心（LSTM/GRU/RNN）和一个输出解码器。
    
- **Forward Pass**: 模型在前向传播中被强制进行多步“空输入”的自演化，依赖其隐藏状态来计算。
    
- **Data**: 与train_mlp.py使用相同格式的.jsonl数据集。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在Config类中调整：
    
    - DATASET_PATH: 指向数据集文件。
        
    - EVOLUTION_STEPS: **关键参数**，必须与数据集中的演化步数或你希望模型模拟的步数一致。
        
    - RNN_TYPE: 可选 'LSTM', 'GRU', 或 'RNN'。
        
    - HIDDEN_SIZE, NUM_LAYERS: 配置RNN的规模。
        
2. **运行训练**:
    
    codeBash
    
    ```
    python train_lstm.py
    ```
    
3. **产出**: 日志文件training_log_rnn.log记录了详细的训练和验证过程。
    

---

### 8. train_convnext.py

**▶︎ 简要说明**  
此脚本用于**图像到符号**的任务，使用预训练的ConvNeXt模型。它将图像作为输入，输出一个固定长度的符号序列（二进制向量）。旨在测试先进的CNN架构在您的范式下的推理能力。

**▶︎ 核心架构**

- **Model**: torchvision.models.convnext_tiny，加载了ImageNet预训练权重，并替换了最后的分类头以匹配任务输出维度。
    
- **Data**: 从一个包含图像和元数据的目录中加载数据。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在Config类中调整：
    
    - DATASET_DIR: 指向数据集根目录。
        
    - BITS: 输出的二进制向量长度。
        
    - BATCH_SIZE, LEARNING_RATE等。
        
2. **准备数据**: DATASET_DIR下应包含initial_images/子目录和一个metadata.csv文件。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_convnext.py
    ```
    
4. **产出**: 训练日志保存在ca_image_log.txt中，包含详细的位准确率和完全匹配率。
    

---

### 9. train_diffusion.py

**▶︎ 简要说明**  
此脚本用于**图像到图像**的任务，但采用的是**条件Diffusion模型**。它将初始状态图像作为条件，学习生成演化后的目标图像。这是对生成模型能否学习确定性规则的严格测试。

**▶︎ 核心架构**

- **Model**: 来自diffusers库的UNet2DModel，输入通道被修改为6（3通道噪声图像 + 3通道条件图像）。
    
- **Scheduler**: DDPMScheduler，管理扩散过程的加噪和去噪步长。
    
- **Training**: 使用accelerate库进行分布式训练和混合精度管理。
    
- **可视化**: 验证阶段会生成（条件 | 目标 | 生成）的三联图。
    

**▶︎ 如何配置和使用**

1. **安装依赖**: 确保已安装diffusers, accelerate, transformers。
    
2. **修改配置**: 在TrainingConfig类中调整：
    
    - dataset_dir: 指向数据集根目录。
        
    - output_dir: 指定所有产出（日志、样本、检查点）的保存位置。
        
    - train_batch_size通常需要设置得较小，因为Diffusion模型显存占用大。
        
3. **运行训练**:
    
    codeBash
    
    ```
    accelerate launch train_diffusion.py
    ```
    
4. **产出**: output_dir中会包含logs/（TensorBoard日志）、samples/（三联对比图）和模型检查点。
    

---

### 10. train_image2image.py

**▶︎ 简要说明**  
这是您的核心**图像到图像**任务训练脚本，实现了一个**Swin-Unet**架构。它使用预训练的Swin Transformer作为编码器，一个U-Net风格的解码器来重建输出图像。

**▶︎ 核心架构**

- **Encoder**: timm.create_model加载Swin Transformer并设置为features_only模式，以提取多尺度特征。
    
- **Decoder**: U-Net解码器，通过UpsampleBlock和跳跃连接逐步恢复图像分辨率。
    
- **可视化**: 训练过程中会定期保存（输入 | 预测 | 目标）的三联对比图。
    

**▶︎ 如何配置和使用**

1. **修改配置**: 在脚本顶部配置区域修改：
    
    - MODEL_NAME: 使用的Swin Transformer模型名称。
        
    - DATASET_DIR: **关键参数**，通过注释/取消注释来选择不同的任务数据集。
        
    - IMAGE_SIZE: 确保与数据集的图像尺寸匹配。
        
    - PRETRAINED_MODEL_PATH: **重要**，指向你本地的pytorch_model.bin权重文件路径，以实现离线加载。
        
    - EVAL_EVERY_N_STEPS, SAVE_IMAGE_EVERY_N_STEPS: 控制评估和可视化的频率。
        
2. **准备数据**: DATASET_DIR下应包含input/和output/两个子目录。
    
3. **运行训练**:
    
    codeBash
    
    ```
    python train_image2image.py
    ```
    
1. **产出**: OUTPUT_DIR中会包含日志文件、best_model.pth检查点，以及定期生成的三联对比图。
    

---

### 11. train_ar_transformer.py

**▶︎ 简要说明**  
该脚本用于训练一个**自回归Transformer模型（GPT-2结构）**，以完成**符号到符号（Text-to-Text）**的生成任务，例如元胞自动机演化、算法步骤预测等。模型通过“提示 → 答案”格式进行训练，具备生成能力。

**▶︎ 核心架构**

- **Model**: 使用 `GPT2LMHeadModel`，从头初始化，非预训练权重。
- **Tokenizer**: 基于 `qwen2_0.5b` 的分词器，支持中文与符号混合输入。
- **Data Collator**: 自定义 `CausalInferenceDataCollator`，支持因果语言建模格式，屏蔽提示部分损失。
- **Loss**: 内部为 `CrossEntropyLoss`，适用于 next-token prediction。
- **Trainer**: 继承自 `Trainer`，支持生成式验证（可采样输出查看效果）。

**▶︎ 如何配置和使用**

1. **修改数据集路径**: 将 `DATASET_PATH` 指向你的 `.jsonl` 文件，格式为每行包含 `"text"` 字段，如 `"1010 -> 1101"`。
2. **调整模型规模**: 可修改 `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_HEADS` 控制模型大小。
3. **设置训练参数**: 如 `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` 等。
4. **运行训练**:

   ```bash
   python train_ar_transformer.py
   ```

5. **产出**: 模型保存在 `OUTPUT_DIR`，支持最终生成测试，打印输入提示与模型生成结果对比。

---

### 12. train_mlp_ctscan.py

**▶︎ 简要说明**  
该脚本用于**“CT扫描”式探测神经网络隐藏层**，揭示模型在解决元胞自动机任务时，**各层是否编码了中间演化状态（S₁→S₈）**。这是对你论文中 **“神经网络是否逐层模拟演化”** 假设的直接验证。

**▶︎ 核心架构**

- **主模型（Body）**: 一个可提取中间层的 MLP，逐层输出隐藏状态。
- **探针（Probe）**: 独立的小 MLP，接收某层隐藏状态作为输入，预测某一步演化结果（如 S₃）。
- **训练方式**: 主模型只训练最终输出（S₈），探针独立训练，不反向传播至主模型。
- **评估方式**: 每层 × 每步演化状态 → 一个 Exact Match 热力图。

**▶︎ 如何配置和使用**

1. **准备数据集**: 使用 `ca_rule110_n30_l8_full_trace.jsonl`，每行包含 `input` 和 `output`（完整轨迹 S₁→S₈）。
2. **设置任务参数**: 如 `TOTAL_LAYERS = 8` 表示演化步数，`NUM_BITS = 30` 表示状态长度。
3. **运行脚本**:

   ```bash
   python train_mlp_ctscan.py
   ```

4. **产出**: 控制台打印信息热力图，显示每层隐藏状态对不同演化步的解码能力。

---

### 13. train_mlp_fulltrace.py

**▶︎ 简要说明**  
该脚本用于验证：**一个只训练最终输出（S₈）的神经网络，其最终隐藏层是否完整编码了中间演化轨迹（S₁→S₆）**。这是对你论文中 **“最终层是否包含完整思维链”** 的强验证。

**▶︎ 核心架构**

- **主模型**: 一个标准 MLP，只训练预测最终状态 S₈。
- **探针模型**: 一个更强的 MLP，接收最终隐藏层，尝试还原完整轨迹（S₁→S₆）。
- **数据集**: 使用 `ca_rule110_n30_l4_full_trace.jsonl`，包含完整演化链。
- **Loss**: `BCEWithLogitsLoss`，逐位回归轨迹。

**▶︎ 如何配置和使用**

1. **设置路径**: 修改 `DATASET_PATH` 指向你的轨迹数据集。
2. **调整探针容量**: 如 `PROBE_HEAD_HIDDEN_SIZE` 和 `PROBE_HEAD_NUM_HIDDEN_LAYERS`。
3. **运行脚本**:

   ```bash
   python train_mlp_fulltrace.py
   ```

4. **产出**: 打印探针解码完整轨迹的 Exact Match，若 >95%，说明最终层几乎“记得”全过程。

---

### 14. train_mlp_prefer.py

**▶︎ 简要说明**  
该脚本用于探测：在解决“接雨水”问题时，神经网络的隐藏层更偏向哪种算法结构（DP、单调栈、双指针）。这是对你论文中 **“模型是否内化了某种算法风格”** 的实证分析。

**▶︎ 核心架构**

- **主模型**: 训练预测最终接雨水结果（每根柱子接水量）。
- **探针模型**: 分别预测三种算法的中间变量（如 leftMax、rightMax、栈状态等）。
- **数据集**: 使用 `rain_water_10n_4b_final_showdown.jsonl`，包含三种算法的解释标签。
- **评估方式**: 比较三种探针的 Exact Match，最高者即为模型“偏好”的算法风格。

**▶︎ 如何配置和使用**

1. **设置数据集路径**: 修改 `DATASET_PATH` 指向你的最终版数据集。
2. **设置探针任务**: 修改 `probe_tasks` 字典，添加或删除算法标签。
3. **运行脚本**:

   ```bash
   python train_mlp_prefer.py
   ```

4. **产出**: 打印三种算法探针的准确率，最高者即为模型最“偏好”的内在算法结构。

---

### 15. train_mlp_visualize.py

**▶︎ 简要说明**  
该脚本用于**逐比特观察神经网络在学习元胞自动机规则时的收敛过程**，可视化每一位的准确率变化，揭示模型是否“从低位到高位”或“同步”学习。

**▶︎ 核心架构**

- **模型**: 标准 MLP，输入为初始状态，输出为最终状态。
- **数据集**: 使用 `ca_rule110_layer6_30.jsonl`，每行包含输入与输出二进制串。
- **验证方式**: 每若干步打印每位准确率，观察学习顺序与收敛节奏。
- **Loss**: `BCEWithLogitsLoss`，逐位计算。

**▶︎ 如何配置和使用**

1. **设置数据集路径**: 修改 `DATASET_PATH` 指向你的二进制任务数据。
2. **调整输出维度**: 修改 `OUTPUT_SIZE = NUM_BITS` 以匹配任务输出长度。
3. **运行脚本**:

   ```bash
   python train_mlp_visualize.py
   ```

4. **产出**: 控制台打印每位比特的准确率变化，可用于分析学习顺序（如是否从低位开始收敛）。


---

### 16. training_scripts/train_mlp_dual_head_supervision.py

**▶︎ 简要说明**  
该脚本实现了一种**早期的双分支监督解耦方法**。它将MLP网络分为Part1和Part2两部分，Part1的输出同时连接到两个分支：一个预测中间解释（无进位计数器），另一个继续送入Part2预测最终答案（乘积）。通过混合损失函数同时监督两个输出，强制中间层学习可解释的表示。

**⚠️ 局限性说明**  
这是论文第八幕"解耦辅助训练"的一个早期探索方法，但实践证明**不太好用**：
- **浪费网络容量：** 需要人工设计Part1/Part2的分界点，难以充分利用网络参数。
- **手动选择层级：** 必须预先决定在哪一层输出中间解释，缺乏灵活性。
- **现在通用的更好方法：** 直接在输出层拼接解耦信息（如`generate_rain_water_final_showdown.py`），让网络自由分配内部表示，无需人为切分网络结构。

本脚本作为历史记录保留，展示了解耦思想的演化过程。

**▶︎ 核心架构**

- **Part1**: 前4层MLP，输出连接到`intermediate_head`预测中间解释。
- **Part2**: 后4层MLP，从Part1的输出继续处理，连接到`final_head`预测最终答案。
- **混合损失**: `loss = α × loss_intermediate + β × loss_final`，两个分支同时监督。

**▶︎ 如何配置和使用**

1. **准备数据集**: 需要使用包含拼接标签的数据集（如Untitled7生成的乘法解耦数据）。
2. **修改配置**: 在`Config`类中调整：
    - `DATASET_PATH`: 数据集路径。
    - `NUM_LAYERS_PART1`, `NUM_LAYERS_PART2`: 网络分段点（需要手动调整）。
    - `LOSS_WEIGHT_INTERMEDIATE`, `LOSS_WEIGHT_FINAL`: 损失权重（默认各0.5）。
3. **运行训练**:
    ```bash
    python training_scripts/train_mlp_dual_head_supervision.py
    ```
4. **产出**: 训练日志保存在`training_log_dual_head_supervision.log`。

---

### 17. training_scripts/train_probe_control_baseline.py

**▶︎ 简要说明**  
这是一个**探针实验的对照组（Control Group）基准测试脚本**。它的目的是验证一个关键问题：**探针成功解码中间表征，是因为主模型内部真的形成了可解释表征，还是仅仅因为探针头本身足够强大，能直接从原始输入学会到解释的映射？**

**▶︎ 实验设计**

- **对照方法：** 创建一个独立的浅层MLP（**不使用**主模型的隐藏层表征）。
- **训练目标：** 直接学习"原始输入 → 解释标签"的映射。
- **判断标准：** 
    - 如果对照组性能**远低于**探针组 → 主模型确实学会了可解释表征
    - 如果对照组性能**接近**探针组 → 探针成功可能只是因为头部足够强大

**▶︎ 使用流程**

1. **第一步：** 先运行主探针实验（如`train_mlp_probe_add_binary.py`）获得探针性能基准。
2. **第二步：** 运行本脚本作为对照实验。
3. **第三步：** 比较两者的Exact Match性能。

**▶︎ 核心架构**

- **ProbeHeadOnly模型：** 一个浅层MLP（1层隐藏层），直接从原始输入预测解释标签。
- **参数量：** 与主探针实验中Probe Head的参数量保持一致，确保公平对比。
- **训练轮数：** 与探针阶段相同（如5000个epoch），确保充分训练。

**▶︎ 如何配置和使用**

1. **修改配置**: 在`ControlConfig`类中调整：
    - `DATASET_PATH`: 使用与探针实验相同的数据集。
    - `NUM_HIDDEN_LAYERS`: 控制模型复杂度（0=纯线性，1=浅MLP）。
    - `EPOCHS`, `BATCH_SIZE`: 与探针实验保持一致。
2. **准备数据**: 使用 `symbolic_math_logic/generate_add_binary_explainable.py` 生成的数据集。
3. **运行脚本**:
    ```bash
    python training_scripts/train_probe_control_baseline.py
    ```
4. **产出**: 日志文件`control_experiment_log.log`会给出最终的Exact Match，用于与探针实验对比。

---

### 18. training_scripts/train_mlp_probe_add_binary.py

**▶︎ 简要说明**
该脚本用于**针对二进制加法任务的线性探针（Linear Probe）实验**。它验证了模型在学会输出最终和之后，其隐藏层是否自发编码了中间的进位信息。这是对论文中“对已经收敛的神经网络的研究”的补充实验。

**▶︎ 核心架构**

- **MLPBody**: 核心隐藏层部分，负责提取特征。
- **ExplainableMLP**: 完整的MLP模型，包含Body和Head。
- **多阶段训练**:
    - **train_sum**: 阶段一，只训练模型输出最终的加法结果。
    - **train_probe**: 阶段二，冻结Body权重，只训练一个新的Head来输出可解释性标签（每一位的进位和结果）。

**▶︎ 如何配置和使用**

1. **准备数据**: 使用 `symbolic_math_logic/generate_add_binary_explainable.py` 生成数据集。
2. **修改配置**: 在 `Config` 类中调整 `DATASET_PATH`, `NUM_BITS`, `TASK_MODE` 等参数。
3. **运行脚本**:
    ```bash
    python training_scripts/train_mlp_probe_add_binary.py
    ```
4. **产出**: 训练日志和模型权重保存在 `OUTPUT_DIR` 中。

---

### 1. eval_hanoi.py

**▶︎ 简要说明**  
这是一个**验证工具**，而非训练脚本。它的功能是接收一个大语言模型（或其他来源）生成的汉诺塔问题解法字符串（如 "1>3;1>2;..."），并严格按照汉诺塔的游戏规则进行模拟，以判断该解法是否正确。

**▶︎ 核心特性**

- **状态模拟**: 在内部模拟三个柱子和n个盘子的状态。
    
- **规则检查**: 自动检查每一步移动是否合法（如：不能从空柱子移出，大盘不能放在小盘上）。
    
- **最终状态验证**: 检查所有移动完成后，是否所有盘子都按正确顺序移动到了目标柱子。
    
- **清晰的错误提示**: 如果解法有误，会明确指出错误在哪一步以及原因。
    

**▶︎ 如何配置和使用**

1. **作为命令行工具**:
    
    - 打开eval_hanoi.py文件。
        
    - 在文件底部的if __name__ == "__main__":部分，找到verify_hanoi_solution(n, solution_str)函数调用。
        
    - 将第一个参数n修改为你想要验证的盘子数量。
        
    - 将第二个参数solution_str替换为你从大模型获取的解法字符串。
        
    - 运行脚本：
        
        codeBash
        
        ```
        python eval_hanoi.py
        ```
        
    - 控制台会输出✅ 正确!或❌ 错误:以及详细信息。
        
2. **作为库导入**:  
    你也可以在其他Python脚本中导入并使用verify_hanoi_solution函数：
    
    codePython
    
    ```
    from eval_hanoi import verify_hanoi_solution
    
    n = 6
    llm_output = "1>2;1>3;..." # 从你的模型获取输出
    is_correct = verify_hanoi_solution(n, llm_output)
    print(f"LLM的解法是否正确: {is_correct}")
    ```
