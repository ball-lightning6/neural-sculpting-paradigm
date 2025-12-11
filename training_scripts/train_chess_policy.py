import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import glob
import random
import numpy as np
from tqdm import tqdm

# ==============================================================================
# --- Configuration ---
# ==============================================================================
CONFIG = {
    # --- Data Paths ---
    "train_data_dir": "./data/chess_policy_data",  # Directory containing .npz files
    "move_to_idx_file": "move2idx.json",
    "output_dir": "./models/chess_policy_model_from_npz",

    # --- FEN Vocab ---
    "fen_vocab": {
        'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5, 'r': 6, 'R': 7,
        'c': 8, 'C': 9, 'a': 10, 'A': 11, 'k': 12, 'K': 13,
        '1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20,
        '8': 21, '9': 22, '/': 23, '[PAD]': 24, '[CLS]': 25,
    },

    # --- Model Architecture ---
    "model_config": {
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "max_position_embeddings": 128,
        "dropout_prob": 0.1,
    },

    # --- Training Hyperparameters ---
    "training_params": {
        "num_epochs": 3000, 
        "train_batch_size": 256, 
        "eval_batch_size": 512,
        "learning_rate": 5e-4, 
        "warmup_steps_ratio": 0.1, 
        "validation_split_ratio": 0.005,
    },

    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "seed": 42
}

# ==============================================================================
# --- Model Definition ---
# ==============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            dropout=config["dropout_prob"],
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config["hidden_size"], config["intermediate_size"]),
            nn.GELU(),
            nn.Linear(config["intermediate_size"], config["hidden_size"]),
            nn.Dropout(config["dropout_prob"])
        )
        self.ln_1 = nn.LayerNorm(config["hidden_size"])
        self.ln_2 = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout_prob"])

    def forward(self, x, attention_mask=None):
        residual = x
        x_norm = self.ln_1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=attention_mask, need_weights=False)
        x = residual + self.dropout(attn_output)
        residual = x
        x_norm = self.ln_2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)
        return x


class PolicyTransformer(nn.Module):
    def __init__(self, vocab_size, num_policy_labels, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, config["hidden_size"])
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])])
        self.final_layernorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout_prob"])
        # Policy head outputs logits for all possible moves
        self.policy_head = nn.Linear(config["hidden_size"], num_policy_labels, bias=False)

    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)

        token_embeds = self.embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + pos_embeds)

        # attention_mask: 1 for valid token, 0 for padding.
        # MultiheadAttention key_padding_mask: True for padding.
        key_padding_mask = (attention_mask==0) if attention_mask is not None else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=key_padding_mask)

        hidden_states = self.final_layernorm(hidden_states)

        # Use CLS token output for classification
        cls_token_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        policy_logits = self.policy_head(cls_token_output)  # [batch_size, num_policy_labels]

        return policy_logits

    def save_pretrained(self, save_directory, optimizer=None, scheduler=None):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        if optimizer:
            torch.save(optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))
        if scheduler:
            torch.save(scheduler.state_dict(), os.path.join(save_directory, "scheduler.pt"))
        with open(os.path.join(save_directory, "model_config.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"✅ Model saved to {save_directory}")

# ==============================================================================
# --- Dataset and Dataloader ---
# ==============================================================================

class NpzChunkDataset(Dataset):
    """Dataset wrapper for a single .npz file chunk."""
    def __init__(self, npz_file_path, fen_vocab, cls_token_id):
        self.npz_file_path = npz_file_path
        self.fen_vocab = fen_vocab
        self.cls_token_id = cls_token_id
        
        # Lazy loading
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.load(self.npz_file_path, allow_pickle=True)
            self.fens = self._data['fens']
            self.labels = self._data['labels']
            self.tokenized_fens = self._tokenize_all_fens(self.fens)
        return self._data

    def _tokenize_all_fens(self, fens):
        tokenized_list = []
        for fen in fens:
            # Only use the board part of FEN for now
            fen_board_part = fen.split()[0]
            token_ids = [self.fen_vocab.get(char, self.fen_vocab['[PAD]']) for char in fen_board_part]
            tokenized_list.append([self.cls_token_id] + token_ids)
        return tokenized_list

    def __len__(self):
        if self._data is not None:
             return len(self.fens)
        # Assuming filename contains length info or pre-calculate metadata would be better for massive datasets
        return len(self.data['fens'])

    def __getitem__(self, idx):
        _ = self.data
        input_ids = self.tokenized_fens[idx]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"input_ids": input_ids, "labels": label_tensor}

class PolicyDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        input_ids_list = [feature["input_ids"] for feature in features]

        # Padding
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []
        for ids in input_ids_list:
            padding_len = max_len - len(ids)
            padded_ids = ids + [self.pad_token_id] * padding_len
            mask = [1] * len(ids) + [0] * padding_len
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.stack(labels)
        }


def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
    model.train()
    if len(dataloader) > 0:
        return total_loss / len(dataloader)
    return 0.0


# ==============================================================================
# --- Main Training Loop ---
# ==============================================================================
def run_training_from_npz():
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"--- Device: {device} ---")

    # 1. Load config and scan data
    print("--- 1. Loading configuration and data files ---")
    try:
        with open(CONFIG["move_to_idx_file"], 'r') as f:
            num_policy_labels = len(json.load(f))
        
        data_dir = CONFIG["train_data_dir"]
        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not npz_files:
            # Fallback to current directory for backward compatibility or simple tests
            npz_files = sorted(glob.glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in '{data_dir}' or current directory.")
        
        print(f"Found {len(npz_files)} data chunks.")
        
        all_chunk_datasets = [
            NpzChunkDataset(
                f, 
                CONFIG["fen_vocab"], 
                CONFIG["fen_vocab"]['[CLS]']
            ) for f in npz_files
        ]
        
        full_dataset = ConcatDataset(all_chunk_datasets)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Split dataset
    val_ratio = CONFIG["training_params"]["validation_split_ratio"]
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset initialized. Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
    
    # 3. Create DataLoaders
    collator = PolicyDataCollator(pad_token_id=CONFIG["fen_vocab"]["[PAD]"])
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["training_params"]["train_batch_size"],
        collate_fn=collator, shuffle=True, pin_memory=True, num_workers=2
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=CONFIG["training_params"]["eval_batch_size"],
        collate_fn=collator, pin_memory=True, num_workers=2
    )

    # 4. Initialize Model, Optimizer, Loss
    print("--- 2. Initializing Model and Optimizer ---")
    model = PolicyTransformer(
        vocab_size=len(CONFIG["fen_vocab"]),
        num_policy_labels=num_policy_labels,
        config=CONFIG["model_config"]
    ).to(device)

    # Use CrossEntropyLoss. For soft labels (probability distributions), 
    # PyTorch's CrossEntropyLoss supports probability targets directly.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CONFIG["training_params"]["learning_rate"])

    # Learning Rate Scheduler
    total_training_steps = len(train_loader) * CONFIG["training_params"]["num_epochs"]
    warmup_steps = int(total_training_steps * CONFIG["training_params"]["warmup_steps_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps)

    # Resume from checkpoint if exists
    if os.path.exists(os.path.join(CONFIG['output_dir'], "pytorch_model.bin")):
        print("Resuming from checkpoint...")
        model.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], "pytorch_model.bin")))
        if os.path.exists(os.path.join(CONFIG['output_dir'], "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], "optimizer.pt")))
        if os.path.exists(os.path.join(CONFIG['output_dir'], "scheduler.pt")):
            scheduler.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], "scheduler.pt")))

    # 5. Training Loop
    print("--- 3. Starting Training ---")
    model.train()
    for epoch in range(CONFIG["training_params"]["num_epochs"]):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['training_params']['num_epochs']}")
        for idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx > 0 and idx % 3000 == 0:
                eval_loss = evaluate_model(model, eval_loader, loss_fn, device)
                print(f"Epoch {epoch + 1} step {idx} eval loss: {eval_loss}")
            
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        # Save model after each epoch (or periodically)
        model.save_pretrained(CONFIG["output_dir"], optimizer, scheduler)
        print(f'Model saved at {CONFIG["output_dir"]}')

    # 6. Final Evaluation
    print("--- 4. Training Complete. Final Evaluation ---")
    final_eval_loss = evaluate_model(model, eval_loader, loss_fn, device)
    print(f"✅ Final Validation Loss: {final_eval_loss:.6f}")

    model.save_pretrained(CONFIG["output_dir"])
    # Save Vocab
    with open(os.path.join(CONFIG["output_dir"], "fen_vocab.json"), 'w') as f:
        json.dump(CONFIG["fen_vocab"], f, indent=4)

    print(f"🎉 Process Finished! Model saved to '{CONFIG['output_dir']}'.")

if __name__ == "__main__":
    run_training_from_npz()
