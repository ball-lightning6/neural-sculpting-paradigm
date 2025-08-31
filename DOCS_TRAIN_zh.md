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

### 1. eval_hanoi.py

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