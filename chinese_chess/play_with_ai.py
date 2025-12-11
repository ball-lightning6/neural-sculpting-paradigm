import os
import torch
import torch.nn as nn
import json
import random
import numpy as np
import cchess
import glob
import re

# ==============================================================================
# --- Configuration ---
# ==============================================================================
CONFIG = {
    # --- Paths ---
    "model_dir": "./models/chess_policy_model_from_npz",
    "move_to_idx_file": "move2idx.json",

    # --- Fen Vocab (Must match training) ---
    "fen_vocab": {
        'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5, 'r': 6, 'R': 7, 'c': 8, 
        'C': 9, 'a': 10, 'A': 11, 'k': 12, 'K': 13, '1': 14, '2': 15, '3': 16, 
        '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, '9': 22, '/': 23, 
        '[PAD]': 24, '[CLS]': 25,
    },

    # --- Model Config (Must match training) ---
    "model_config": {
        "hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6,
        "intermediate_size": 1536, "max_position_embeddings": 128, "dropout_prob": 0.1,
    },

    # --- Inference Config ---
    "play_config": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sampling_temperature": 0.8,
    }
}

# ==============================================================================
# --- Model Definition (Must match training) ---
# ==============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__();self.self_attn=nn.MultiheadAttention(embed_dim=config["hidden_size"],num_heads=config["num_attention_heads"],dropout=config["dropout_prob"],batch_first=True);self.ffn=nn.Sequential(nn.Linear(config["hidden_size"],config["intermediate_size"]),nn.GELU(),nn.Linear(config["intermediate_size"],config["hidden_size"]),nn.Dropout(config["dropout_prob"]));self.ln_1=nn.LayerNorm(config["hidden_size"]);self.ln_2=nn.LayerNorm(config["hidden_size"]);self.dropout=nn.Dropout(config["dropout_prob"])
    def forward(self,x,attention_mask=None):
        residual=x;x_norm=self.ln_1(x);attn_output,_=self.self_attn(x_norm,x_norm,x_norm,key_padding_mask=attention_mask,need_weights=False);x=residual+self.dropout(attn_output);residual=x;x_norm=self.ln_2(x);ffn_output=self.ffn(x_norm);x=residual+self.dropout(ffn_output);return x

class PolicyTransformer(nn.Module):
    def __init__(self,vocab_size,num_policy_labels,config):
        super().__init__();self.config=config;self.embeddings=nn.Embedding(vocab_size,config["hidden_size"]);self.position_embeddings=nn.Embedding(config["max_position_embeddings"],config["hidden_size"]);self.layers=nn.ModuleList([TransformerBlock(config) for _ in range(config["num_hidden_layers"])]);self.final_layernorm=nn.LayerNorm(config["hidden_size"]);self.dropout=nn.Dropout(config["dropout_prob"]);self.policy_head=nn.Linear(config["hidden_size"],num_policy_labels,bias=False)
    def forward(self,input_ids,attention_mask=None):
        seq_length=input_ids.shape[1];position_ids=torch.arange(0,seq_length,device=input_ids.device).unsqueeze(0);token_embeds=self.embeddings(input_ids);pos_embeds=self.position_embeddings(position_ids);hidden_states=self.dropout(token_embeds+pos_embeds);key_padding_mask=(attention_mask==0) if attention_mask is not None else None
        for l in self.layers: hidden_states=l(hidden_states,attention_mask=key_padding_mask)
        hidden_states=self.final_layernorm(hidden_states);cls_token_output=hidden_states[:,0];policy_logits=self.policy_head(cls_token_output);return policy_logits

# ==============================================================================
# --- Inference Engine ---
# ==============================================================================
def print_board(board):
    symbols={'R':'车','N':'马','B':'相','A':'仕','K':'帅','C':'炮','P':'兵','r':'车','n':'马','b':'象','a':'士','k':'将','c':'炮','p':'卒','.':'．'}
    board_str = str(board)
    for i in range(1, 10): board_str = board_str.replace(str(i), '.' * i)
    board_str = board_str.split("/")
    print("\n  ａ ｂ ｃ ｄ ｅ ｆ ｇ ｈ ｉ")
    print(" --------------------------")
    for i, row in enumerate(board_str): print(f"{9-i}| {' '.join([symbols.get(c,c) for c in row])} |{9-i}")
    print(" --------------------------")
    print("  ａ ｂ ｃ ｄ ｅ ｆ ｇ ｈ ｉ\n")

class PolicyEngine:
    def __init__(self, model, config):
        self.model = model; self.config = config; self.device = torch.device(config['play_config']['device']); self.model.to(self.device).eval()
        with open(config["move_to_idx_file"],'r') as f: self.move_to_idx = json.load(f)
    def _fen_to_input(self, fen):
        fen_board=fen.split()[0]; token_ids=[self.config['fen_vocab'].get(c,self.config['fen_vocab']['[PAD]']) for c in fen_board]; input_ids=[self.config['fen_vocab']['[CLS]']]+token_ids
        input_tensor=torch.tensor([input_ids],dtype=torch.long).to(self.device); attention_mask=torch.ones_like(input_tensor); return input_tensor, attention_mask
    def get_ai_move(self, board: cchess.Board, by_prob=True):
        legal_uci_moves = [str(m) for m in board.legal_moves]
        if not legal_uci_moves: return None
        with torch.no_grad():
            input_tensor, attention_mask = self._fen_to_input(board.fen())
            logits = self.model(input_tensor, attention_mask).squeeze(0)
        filtered_moves, filtered_indices = [], []
        for move_uci in legal_uci_moves:
            idx = self.move_to_idx.get(move_uci)
            if idx is not None: filtered_moves.append(move_uci); filtered_indices.append(idx)
        if not filtered_moves: return random.choice(legal_uci_moves)
        legal_logits = logits[filtered_indices]
        if by_prob:
            probs=torch.nn.functional.softmax(legal_logits / self.config['play_config']['sampling_temperature'], dim=0)
            chosen_index=torch.multinomial(probs, 1).item()
        else:
            chosen_index=torch.argmax(legal_logits).item()
        return filtered_moves[chosen_index]

def main_play_loop():
    print("--- 🤖 Chinese Chess AI (Policy Network) ---")
    print("--- 1. Loading Model... ---")
    try:
        with open(CONFIG["move_to_idx_file"],'r') as f: num_policy_labels=len(json.load(f))
        model_dir=CONFIG["model_dir"]
        checkpoints=glob.glob(os.path.join(model_dir,"checkpoint-*"))
        if checkpoints:
            latest_checkpoint=max(checkpoints,key=lambda p: int(re.search(r'checkpoint-(\d+)',p).group(1)))
            print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
            model_path=os.path.join(latest_checkpoint,"pytorch_model.bin")
        else:
            print(f"Warning: No checkpoints found, using base directory {model_dir}")
            model_path=os.path.join(model_dir,"pytorch_model.bin")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model weights not found: {model_path}")
        model=PolicyTransformer(len(CONFIG["fen_vocab"]),num_policy_labels,CONFIG["model_config"])
        model.load_state_dict(torch.load(model_path,map_location=CONFIG['play_config']['device']))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return
    
    ai_engine = PolicyEngine(model,CONFIG)
    board = cchess.Board()
    player_choice=""
    while player_choice not in['r','b']: player_choice=input("Choose side - Red (r) or Black (b): ").lower()
    human_is_red=(player_choice=='r')
    
    while not board.is_game_over():
        print_board(board)
        current_player_is_red=(board.turn == cchess.RED)
        if current_player_is_red==human_is_red:
            move_uci=None
            while not move_uci:
                move_str_input = input(f"[{'Red' if human_is_red else 'Black'}] Your move (UCI, e.g., h2e2): ")
                try:
                    # Basic UCI validation attempt
                    move = cchess.Move.from_uci(move_str_input)
                    if move in board.legal_moves:
                        move_uci = move_str_input
                    else:
                        print("Illegal move, please try again.")
                except:
                    print("Invalid format, please use UCI format (e.g., h2e2).")

            board.push(cchess.Move.from_uci(move_uci))
            print(f"\n> You moved: {move_uci}")
        else:
            print(f"[{'AI(Black)' if human_is_red else 'AI(Red)'}] Thinking...")
            ai_move_uci = ai_engine.get_ai_move(board, by_prob=True)
            if ai_move_uci:
                board.push(cchess.Move.from_uci(ai_move_uci))
                print(f"\n> AI moved: {ai_move_uci}")
            else:
                print("AI resigned (no moves)."); break
    
    print("\n" + "="*20 + " Game Over " + "="*20)
    print_board(board)
    print(f"Final FEN: {board.fen()}")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    main_play_loop()
