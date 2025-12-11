### 1. train_tiny_transformer.py

**▶︎ Brief Description**
This script is used to train a custom TinyTransformer model built from scratch, focusing on **Sequence-to-Vector** tasks, such as learning symbolic rules or fitting algorithms. It takes a character sequence as input and outputs a fixed-length, multi-label binary classification result.

**▶︎ Core Architecture**

- **Encoder**: `TinyTransformerForCausalLM`, a lightweight Transformer designed for autoregressive tasks.
- **Output Head**: The model's `lm_head` is replaced with an `nn.Linear` layer to match the `num_labels` of the task.
- **Loss**: `BCEWithLogitsLoss`, used for multi-label binary classification.
- **Tokenization**: Employs simple character-level encoding (`ord(char)`), requiring no pre-trained tokenizer.

**▶︎ How to Configure and Use**

1.  **Modify Hyperparameters**: Find the `hyperparams` dictionary in the script and adjust `num_epochs`, `train_batch_size`, `learning_rate`, etc., as needed.
2.  **Specify Dataset**: Modify the line `dataset = LightsOutDataset("your_dataset.jsonl", tokenizer)` to point to your `.jsonl` dataset file. The file format should have one JSON object per line, containing "input" and "output" keys.
3.  **Set Output Dimension**: Modify `num_labels = N`, where N is the length of the binary vector your task outputs.
4.  **Run Training**:
    ```bash
    python train_tiny_transformer.py
    ```
5.  **Output**: Training logs will be printed to the console, and model checkpoints will be saved in the directory specified by `output_dir`.

---

### 2. train_swin_image2text.py

**▶︎ Brief Description**
This script is for **Image-to-Vector** tasks, performing full fine-tuning on a pre-trained Swin Transformer to enable it for multi-label image classification.

**▶︎ Core Architecture**

- **Model**: Hugging Face `AutoModelForImageClassification`, loading models from the `microsoft/swin-*` series.
- **Processor**: Hugging Face `AutoImageProcessor`, which automatically handles the image preprocessing required for Swin models.
- **Loss**: `BCEWithLogitsLoss`, used for multi-label classification.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the configuration section at the top of the script, modify the following variables:
    -   `MODEL_NAME`: Specify the Swin model to use, e.g., `microsoft/swin-tiny-patch4-window7-224`.
    -   `NUM_LABELS`: The length of the binary vector for the task's output.
    -   `IMAGE_DIR`, `LABEL_DIR`/`METADATA_PATH`: Point to your dataset paths. The script has two built-in `Dataset` classes; comment out the one you are not using based on your data format.
    -   Training parameters like `BATCH_SIZE`, `LEARNING_RATE`, etc.
2.  **Prepare Data**: Ensure your dataset's directory structure matches the requirements of the selected `Dataset` class.
3.  **Run Training**:
    ```bash
    python train_swin_image2text.py
    ```
4.  **Output**: Checkpoints (`best_model.pth` and `final_model.pth`) and a log file will be saved in `OUTPUT_DIR`.

---

### 3. train_unet.py

**▶︎ Brief Description**
This script is for pure **Image-to-Image** tasks, training a standard U-Net model. A key feature is that it automatically generates triplet images (input | ground truth | prediction) during the validation phase, allowing for intuitive evaluation of model performance.

**▶︎ Core Architecture**

- **Model**: A classic UNet implemented from scratch.
- **Loss**: `MSELoss` (or can be switched to `BCEWithLogitsLoss`, etc.).
- **Visualization**: The validation loop integrates `save_image` functionality to generate comparison images.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: All configuration options are in the `Config` class.
    -   `DATASET_DIR`: Point to your dataset's root directory.
    -   `OUTPUT_DIR`: Specify the save location for all training outputs.
    -   `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, etc.
2.  **Prepare Data**: The `DATASET_DIR` should contain `initial_images/` and `final_images/` subdirectories and a `metadata.csv` file.
3.  **Run Training**:
    ```bash
    python train_unet.py
    ```
4.  **Output**: The `OUTPUT_DIR` will contain a log file (`training_log_unet.log`) and an `eval_images` directory, which stores the triplet comparison images generated at each evaluation step.

---

### 4. train_text2image.py

**▶︎ Brief Description**
This script is for **Text-to-Image** tasks, training a model built entirely from scratch. The model consists of a lightweight TinyTransformer as the text encoder and a U-Net style decoder.

**▶︎ Core Architecture**

- **Text Encoder**: `TinyTransformerForCausalLM`, the same model as in `train_tiny_transformer.py`.
- **Image Decoder**: `ImageDecoder`, a U-Net style decoder that "injects" the text feature vector into various stages of the decoding process as skip connections via a linear projection.
- **Tokenization**: Also uses simple character-level encoding (`ord(char)`).

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the configuration section at the top of the script, modify:
    -   `TINY_TRANSFORMER_CONFIG`: Configure the Transformer's architecture.
    -   `IMAGE_SIZE`, `OUTPUT_CHANNELS`: Configure the output image's properties.
    -   `DATASET_DIR`: Point to the dataset root directory.
    -   `OUTPUT_DIR`, `BATCH_SIZE`, `LEARNING_RATE`, etc.
2.  **Prepare Data**: The `DATASET_DIR` should contain an `images/` subdirectory and a `metadata.csv` file.
3.  **Run Training**:
    ```bash
    python train_text2image.py
    ```
4.  **Output**: The checkpoint directory `OUTPUT_DIR` will contain a log file and an `eval_predictions` directory, which stores doublet comparison images (ground truth image | generated image).

---

### 5. train_qwen2_text2image.py

**▶︎ Brief Description**
This is a more powerful version of the **Text-to-Image** script. It uses the pre-trained Qwen2 large language model as the text encoder, fine-tuned efficiently with PEFT LoRA, and combined with the same U-Net style decoder as the previous script. The entire training process is managed by the Hugging Face `Trainer`.

**▶︎ Core Architecture**

- **Text Encoder**: Hugging Face `AutoModelForCausalLM` loads Qwen2, and the `peft` library is used to apply LoRA.
- **Image Decoder**: `ImageDecoder`, identical to the decoder in `train_text2image.py`.
- **Trainer**: Uses a custom `ImageGenTrainer`, which overrides the `compute_loss` method to handle image reconstruction loss, and is driven by the Hugging Face `Trainer` API.
- **Callbacks**: Includes a `SaveImagePredictionCallback` to automatically save comparison images during evaluation.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the configuration section at the top of the script, modify:
    -   `TEXT_MODEL_NAME`: Specify the Qwen2 model version to use, e.g., `qwen2-0.5b`.
    -   `LORA_R`, `LORA_ALPHA`: LoRA configuration.
    -   `IMAGE_SIZE`, `DATASET_DIR`, `OUTPUT_DIR`, etc.
2.  **Log in to Hugging Face (if required)**: If the model is private, you need to log in first via `huggingface-cli login`.
3.  **Run Training**:
    ```bash
    python train_qwen2_text2image.py
    ```
4.  **Output**: The `OUTPUT_DIR` will contain standard Hugging Face training outputs, including `checkpoint-*` directories, logs, and the `eval_predictions` comparison image directory generated by the callback. The LoRA adapter and decoder weights will be saved separately after training is complete.

---

### 6. train_mlp.py

**▶︎ Brief Description**
This script is used to train a "giant" MLP (Multi-Layer Perceptron) to solve **Sequence-to-Sequence** tasks. Its main purpose is to provide a performance baseline with no structural bias for more complex architectures (like Transformers, RNNs).

**▶︎ Core Architecture**

- **Model**: A deep and wide MLP, containing `Linear`, `GELU`, `LayerNorm`, and `Dropout` layers.
- **Data**: Loads sequence data from a `.jsonl` file, where each line contains 'input' and 'output' strings.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the `Config` class, adjust:
    -   `DATASET_PATH`: Point to your `.jsonl` dataset file.
    -   `BITS`: The vector dimensions for input and output.
    -   `HIDDEN_SIZE`, `NUM_HIDDEN_LAYERS`: Adjust the size of the MLP to match the parameter count of other models.
    -   Training parameters like `BATCH_SIZE`, `LEARNING_RATE`.
2.  **Run Training**:
    ```bash
    python train_mlp.py
    ```
3.  **Output**: Training progress is printed to the console, and logs are saved in `training_log_mlp.log`.

---

### 7. train_lstm.py

**▶︎ Brief Description**
This script uses an LSTM (or can be switched to GRU/RNN) to solve **Sequence-to-Sequence** tasks. Its design cleverly tests the **temporal evolution and memory capabilities** of an RNN: the model receives a one-time input, then iterates internally for `EVOLUTION_STEPS`, and finally outputs the result.

**▶︎ Core Architecture**

- **Model**: `RNNModel`, which includes an input encoder, an RNN core (LSTM/GRU/RNN), and an output decoder.
- **Forward Pass**: The model is forced to self-evolve with "empty inputs" for multiple steps during the forward pass, relying on its hidden state for computation.
- **Data**: Uses the same format `.jsonl` dataset as `train_mlp.py`.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the `Config` class, adjust:
    -   `DATASET_PATH`: Point to the dataset file.
    -   `EVOLUTION_STEPS`: **Key parameter**, must match the number of evolution steps in the dataset or the number of steps you want the model to simulate.
    -   `RNN_TYPE`: Options are `'LSTM'`, `'GRU'`, or `'RNN'`.
    -   `HIDDEN_SIZE`, `NUM_LAYERS`: Configure the scale of the RNN.
2.  **Run Training**:
    ```bash
    python train_lstm.py
    ```
3.  **Output**: The log file `training_log_rnn.log` records the detailed training and validation process.

---

### 8. train_convnext.py

**▶︎ Brief Description**
This script is for **Image-to-Sequence** tasks, using a pre-trained ConvNeXt model. It takes an image as input and outputs a fixed-length sequence of symbols (a binary vector). It aims to test the reasoning capabilities of advanced CNN architectures under your paradigm.

**▶︎ Core Architecture**

- **Model**: `torchvision.models.convnext_tiny`, loaded with ImageNet pre-trained weights, with the final classification head replaced to match the task's output dimension.
- **Data**: Loads data from a directory containing images and metadata.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the `Config` class, adjust:
    -   `DATASET_DIR`: Point to the dataset root directory.
    -   `BITS`: The length of the output binary vector.
    -   `BATCH_SIZE`, `LEARNING_RATE`, etc.
2.  **Prepare Data**: The `DATASET_DIR` should contain an `initial_images/` subdirectory and a `metadata.csv` file.
3.  **Run Training**:
    ```bash
    python train_convnext.py
    ```
4.  **Output**: Training logs are saved in `ca_image_log.txt`, including detailed bit accuracy and exact match accuracy.

---

### 9. train_diffusion.py

**▶︎ Brief Description**
This script is for **Image-to-Image** tasks but employs a **conditional Diffusion model**. It takes an initial state image as a condition and learns to generate the evolved target image. This is a rigorous test of whether a generative model can learn deterministic rules.

**▶︎ Core Architecture**

- **Model**: `UNet2DModel` from the `diffusers` library, with input channels modified to 6 (3 channels for the noisy image + 3 channels for the conditional image).
- **Scheduler**: `DDPMScheduler`, which manages the noising and denoising steps of the diffusion process.
- **Training**: Uses the `accelerate` library for distributed training and mixed-precision management.
- **Visualization**: Generates (condition | ground truth | generated) triplet images during the validation phase.

**▶︎ How to Configure and Use**

1.  **Install Dependencies**: Make sure `diffusers`, `accelerate`, and `transformers` are installed.
2.  **Modify Configuration**: In the `TrainingConfig` class, adjust:
    -   `dataset_dir`: Point to the dataset root directory.
    -   `output_dir`: Specify the save location for all outputs (logs, samples, checkpoints).
    -   `train_batch_size` usually needs to be set low, as Diffusion models are memory-intensive.
3.  **Run Training**:
    ```bash
    accelerate launch train_diffusion.py
    ```
4.  **Output**: The `output_dir` will contain `logs/` (TensorBoard logs), `samples/` (triplet comparison images), and model checkpoints.

---

### 10. train_image2image.py

**▶︎ Brief Description**
This is your core **Image-to-Image** task training script, implementing a **Swin-Unet** architecture. It uses a pre-trained Swin Transformer as the encoder and a U-Net style decoder to reconstruct the output image.

**▶︎ Core Architecture**

- **Encoder**: `timm.create_model` loads a Swin Transformer and sets it to `features_only` mode to extract multi-scale features.
- **Decoder**: A U-Net decoder that progressively restores image resolution using `UpsampleBlock` and skip connections.
- **Visualization**: Periodically saves (input | prediction | ground truth) triplet comparison images during training.

**▶︎ How to Configure and Use**

1.  **Modify Configuration**: In the configuration section at the top of the script, modify:
    -   `MODEL_NAME`: The name of the Swin Transformer model to use.
    -   `DATASET_DIR`: **Key parameter**, select different task datasets by commenting/uncommenting lines.
    -   `IMAGE_SIZE`: Ensure it matches the image dimensions of the dataset.
    -   `PRETRAINED_MODEL_PATH`: **Important**, point to your local `pytorch_model.bin` weight file path to enable offline loading.
    -   `EVAL_EVERY_N_STEPS`, `SAVE_IMAGE_EVERY_N_STEPS`: Control the frequency of evaluation and visualization.
2.  **Prepare Data**: The `DATASET_DIR` should contain `input/` and `output/` subdirectories.
3.  **Run Training**:
    ```bash
    python train_image2image.py
    ```
4.  **Output**: The `OUTPUT_DIR` will contain a log file, the `best_model.pth` checkpoint, and periodically generated triplet comparison images.

---

### 11. train_ar_transformer.py

**▶︎ Brief Description**  
This script trains a **custom autoregressive Transformer model (GPT-2 architecture)** for **symbol-to-symbol (Text-to-Text)** generation tasks, such as cellular automata evolution or algorithm step prediction. The model is trained in a “prompt → answer” format and supports generation.

**▶︎ Core Architecture**

- **Model**: `GPT2LMHeadModel`, initialized from scratch (no pretrained weights).
- **Tokenizer**: Based on the `qwen2_0.5b` tokenizer, supporting mixed Chinese and symbolic input.
- **Data Collator**: Custom `CausalInferenceDataCollator` for causal language modeling; masks loss on prompt tokens.
- **Loss**: Internally uses `CrossEntropyLoss` for next-token prediction.
- **Trainer**: Extends Hugging Face `Trainer`, supports generative evaluation (sampling outputs for inspection).

**▶︎ How to Configure and Use**

1. **Modify dataset path**: Set `DATASET_PATH` to your `.jsonl` file. Each line should contain a `"text"` field like `"1010 -> 1101"`.
2. **Adjust model size**: Modify `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_HEADS` to control model capacity.
3. **Set training params**: E.g., `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`.
4. **Run training**:

   ```bash
   python train_ar_transformer.py
   ```

5. **Output**: Model saved in `OUTPUT_DIR`. Final test prints prompt vs. generated result for comparison.

---

### 12. train_mlp_ctscan.py

**▶︎ Brief Description**  
This script performs a **“CT-scan” style probing of neural network hidden layers**, revealing whether each layer encodes intermediate evolution states (S₁→S₈) during a cellular automaton task. It directly tests the hypothesis: *“Does the network simulate step-by-step evolution internally?”*

**▶︎ Core Architecture**

- **Main Model (Body)**: An MLP that outputs hidden states layer-by-layer.
- **Probe Model**: A small MLP that takes a hidden state and predicts an intermediate evolution step (e.g., S₃).
- **Training**: Main model is trained only to predict final output (S₈); probes are trained independently without backpropagating into the main model.
- **Evaluation**: Exact Match heatmap across layers × evolution steps.

**▶︎ How to Configure and Use**

1. **Prepare dataset**: Use `ca_rule110_n30_l8_full_trace.jsonl`, each line contains `input` and `output` (full trace S₁→S₈).
2. **Set task params**: E.g., `TOTAL_LAYERS = 8` for evolution steps, `NUM_BITS = 30` for state length.
3. **Run script**:

   ```bash
   python train_mlp_ctscan.py
   ```

4. **Output**: Console prints an information heatmap showing how well each hidden layer decodes each evolution step.

---

### 13. train_mlp_fulltrace.py

**▶︎ Brief Description**  
This script tests whether a **neural network trained only on the final output (S₈)** still **encodes the full intermediate trace (S₁→S₆)** in its final hidden layer. It’s a strong validation of your claim: *“The final layer remembers the full thought process.”*

**▶︎ Core Architecture**

- **Main Model**: Standard MLP trained only to predict final state S₈.
- **Probe Model**: A stronger MLP that takes the final hidden state and tries to reconstruct the full trace (S₁→S₆).
- **Dataset**: Uses `ca_rule110_n30_l4_full_trace.jsonl` with complete evolution chains.
- **Loss**: `BCEWithLogitsLoss`, bit-wise regression over the trace.

**▶︎ How to Configure and Use**

1. **Set path**: Modify `DATASET_PATH` to point to your trace dataset.
2. **Adjust probe capacity**: E.g., `PROBE_HEAD_HIDDEN_SIZE`, `PROBE_HEAD_NUM_HIDDEN_LAYERS`.
3. **Run script**:

   ```bash
   python train_mlp_fulltrace.py
   ```

4. **Output**: Prints probe’s Exact Match on full trace. >95% implies final layer nearly “remembers” the entire process.

---

### 14. train_mlp_prefer.py

**▶︎ Brief Description**  
This script probes **which algorithmic structure (DP, monotonic stack, two-pointer) a neural network prefers internally** when solving the “Trapping Rain Water” problem. It empirically tests your hypothesis: *“The model internalizes a specific algorithmic style.”*

**▶︎ Core Architecture**

- **Main Model**: Trained to predict final water amount per column.
- **Probe Models**: Three independent probes predicting intermediate variables of each algorithm (e.g., leftMax, stack state).
- **Dataset**: Uses `rain_water_10n_4b_final_showdown.jsonl`, containing labels for all three algorithmic explanations.
- **Evaluation**: Compare Exact Match of three probes; highest = model’s “preferred” style.

**▶︎ How to Configure and Use**

1. **Set dataset path**: Modify `DATASET_PATH` to your final dataset.
2. **Set probe tasks**: Edit `probe_tasks` dict to add/remove algorithm labels.
3. **Run script**:

   ```bash
   python train_mlp_prefer.py
   ```

4. **Output**: Prints accuracy of all three algorithm probes; highest = model’s intrinsic cognitive bias.

---

### 15. train_mlp_visualize.py

**▶︎ Brief Description**  
This script **visualizes the per-bit learning dynamics** of a neural network trained on a cellular automaton rule. It reveals whether the model learns **bit-by-bit** or **all-at-once**, offering insight into its internal learning order.

**▶︎ Core Architecture**

- **Model**: Standard MLP, input = initial state, output = final state.
- **Dataset**: Uses `ca_rule110_layer6_30.jsonl`, each line contains input/output binary strings.
- **Validation**: Prints per-bit accuracy every N steps to observe convergence order.
- **Loss**: `BCEWithLogitsLoss`, computed per bit.

**▶︎ How to Configure and Use**

1. **Set dataset path**: Modify `DATASET_PATH` to your binary task data.
2. **Adjust output size**: Set `OUTPUT_SIZE = NUM_BITS` to match task output length.
3. **Run script**:

   ```bash
   python train_mlp_visualize.py
   ```

4. **Output**: Console prints per-bit accuracy over time, useful for analyzing learning order (e.g., does it converge from LSB to MSB?).


---

### 16. training_scripts/train_mlp_probe_add_binary.py

**▶︎ Brief Description**
This script is used for a **Linear Probe experiment on the binary addition task**. It verifies whether the model's hidden layers spontaneously encode intermediate carry information after learning to output the final sum. This is a supplementary experiment to the "Research on Converged Neural Networks" section in the paper.

**▶︎ Core Architecture**

- **MLPBody**: The core hidden layer part, responsible for feature extraction.
- **ExplainableMLP**: The complete MLP model, containing Body and Head.
- **Multi-stage Training**:
    - **train_sum**: Phase 1, train the model to output only the final addition result.
    - **train_probe**: Phase 2, freeze Body weights, train only a new Head to output interpretability labels (carry and result of each bit).

**▶︎ How to Configure and Use**

1. **Prepare Data**: Use `symbolic_math_logic/generate_add_binary_explainable.py` to generate the dataset.
2. **Modify Configuration**: Adjust parameters like `DATASET_PATH`, `NUM_BITS`, `TASK_MODE` in the `Config` class.
3. **Run Script**:
    ```bash
    python training_scripts/train_mlp_probe_add_binary.py
    ```
4. **Output**: Training logs and model weights are saved in `OUTPUT_DIR`.

---

### 1. eval_hanoi.py

**▶︎ Brief Description**
This is a **validation tool**, not a training script. Its function is to take a solution string for the Tower of Hanoi problem (e.g., `"1>3;1>2;..."`) generated by a large language model (or another source) and simulate it strictly according to the game's rules to determine if the solution is correct.

**▶︎ Core Features**

- **State Simulation**: Internally simulates the state of three pegs and n disks.
- **Rule Checking**: Automatically checks if each move is legal (e.g., cannot move from an empty peg, a larger disk cannot be placed on a smaller one).
- **Final State Validation**: After all moves are completed, it checks if all disks have been moved to the target peg in the correct order.
- **Clear Error Messages**: If the solution is incorrect, it will clearly state which step was wrong and why.

**▶︎ How to Configure and Use**

1.  **As a Command-Line Tool**:
    -   Open the `eval_hanoi.py` file.
    -   At the bottom of the file, in the `if __name__ == "__main__":` block, find the `verify_hanoi_solution(n, solution_str)` function call.
    -   Change the first argument `n` to the number of disks you want to verify.
    -   Replace the second argument `solution_str` with the solution string you obtained from the large model.
    -   Run the script:
        ```bash
        python eval_hanoi.py
        ```
    -   The console will output ✅ Correct! or ❌ Error: with detailed information.

2.  **As a Library Import**:
    You can also import and use the `verify_hanoi_solution` function in other Python scripts:
    ```python
    from eval_hanoi import verify_hanoi_solution

    n = 6
    llm_output = "1>2;1>3;..." # Get output from your model
    is_correct = verify_hanoi_solution(n, llm_output)
    print(f"Is the LLM's solution correct: {is_correct}")
    ```