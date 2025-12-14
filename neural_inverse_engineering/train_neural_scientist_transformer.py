import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np

# ================= 1. 配置区域 =================
class Config:
    # --- 基础配置 ---
    SEQ_LEN = 30
    RULE_LAYERS = 4
    RULE_BITS = 8
    
    INPUT_DIM = SEQ_LEN * 2  # 60 (30 Input + 30 Output)
    OUTPUT_DIM = RULE_LAYERS * RULE_BITS # 9
    
    # --- 核心架构参数 ---
    EMBED_DIM = 512         # 隐状态维度 (MLP输出/Attention输入)
    
    # MLP Encoder 参数
    MLP_HIDDEN = 4096#2048
    MLP_LAYERS = 4#3
    
    # Aggregator (Transformer) 参数
    ATTN_LAYERS = 2
    ATTN_HEADS = 8
    
    # --- 训练参数 ---
    TRAIN_SAMPLES = 100000   # 训练用的"规则数" (每个规则会生成N个样本)
    VAL_SAMPLES = 1000
    
    BATCH_SIZE = 64          # 这是一个Batch里有多少个"规则任务"
    MIN_OBSERVE = 24          # 每个规则最少看几个样本
    MAX_OBSERVE = 48         # 每个规则最多看几个样本
    
    EPOCHS = 5000
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BEST_MODEL_PATH = "neural_scientist_best.pth"

# ================= 2. 规则定义 (复用之前的逻辑) =================
def int_to_bits(n, length): return [int(x) for x in f"{n:0{length}b}"]
def bits_to_int(bits): return int("".join(map(str, bits)), 2)
def cyclic_shift(bits, k): k = k % len(bits); return bits[k:] + bits[:k]
def get_neighbors(bits, i): return bits[i-1], bits[i], bits[(i+1)%len(bits)]

def apply_ca(bits, rule_number):
    rule_map = int_to_bits(rule_number, 8)
    return [rule_map[7 - (get_neighbors(bits, i)[0]*4 + get_neighbors(bits, i)[1]*2 + get_neighbors(bits, i)[2])] for i in range(len(bits))]

def apply_add_shift(bits):
    val = (bits_to_int(bits) + bits_to_int(cyclic_shift(bits, 15))) % (2**30)
    return int_to_bits(val, 30)

def apply_long_xor(bits):
    shifted = cyclic_shift(bits, 10)
    return [b ^ s for b, s in zip(bits, shifted)]
    
def apply_long_or(bits):
    shifted = cyclic_shift(bits, 10)
    return [b | s for b, s in zip(bits, shifted)]

def apply_dynamic_shift(bits):
    k = bits_to_int(bits[:5])
    return cyclic_shift(bits, k)

def apply_majority(bits):
    new_bits = []
    for i in range(len(bits)):
        l, c, r = get_neighbors(bits, i)
        if (l + c + r) >= 2: new_bits.append(1)
        else: new_bits.append(0)
    return new_bits
def apply_shift_part_reverse(bits):
    return bits[1::3]+[1 - x for x in bits[2::3]]+bits[0::3][::-1]

RULES_FUNC = {
    0: lambda b: apply_ca(b, 30),
    1: lambda b: apply_ca(b, 110),
    2: lambda b: apply_ca(b, 167),
    3: lambda b: apply_ca(b, 184),
    4: apply_majority,
    5: apply_add_shift,
    6: apply_dynamic_shift,
    7: apply_shift_part_reverse
}

# ================= 3. 动态数据生成器 (Online Generation) =================
class DynamicRuleDataset(Dataset):
    def __init__(self, num_samples, min_obs, max_obs):
        self.num_samples = num_samples
        self.min_obs = min_obs
        self.max_obs = max_obs
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 1. 随机生成一个规则序列
        rule_indices = [random.randint(0, 7) for _ in range(Config.RULE_LAYERS)]
        
        # 2. 构造 Label
        label_bits = []
        for r in rule_indices:
            #label_bits.extend(int_to_bits(r, Config.RULE_BITS))
            rule_list=[0]*Config.RULE_BITS
            rule_list[r]=1
            label_bits.extend(rule_list)
            
        # 3. 随机决定这一轮看多少个样本 (N)
        n_obs = random.randint(self.min_obs, self.max_obs)
        
        # 4. 生成 N 个 (Input, Output) 对
        obs_data = []
        for _ in range(n_obs):
            x_bits = [random.randint(0, 1) for _ in range(Config.SEQ_LEN)]
            current = x_bits[:]
            for r_idx in rule_indices:
                current = RULES_FUNC[r_idx](current)
            y_bits = current
            obs_data.append(x_bits + y_bits)
            
        # 5. Padding (为了 Batch 训练，需要 Pad 到 MAX_OBS)
        # 使用全0或者特定值Pad，并且生成 mask
        real_data = torch.tensor(obs_data, dtype=torch.float32) # [N, 60]
        padded_data = torch.zeros((self.max_obs, Config.INPUT_DIM), dtype=torch.float32)
        padded_data[:n_obs] = real_data
        
        # Mask: 1 表示真实数据，0 表示 Padding
        mask = torch.zeros(self.max_obs, dtype=torch.bool) # True/False
        mask[:n_obs] = True 
        # 注意 Transformer 的 mask 通常是 True 表示 Padding (忽略)，False 表示真实
        # 这里我们生成 src_key_padding_mask, True for padded elements
        padding_mask = torch.ones(self.max_obs, dtype=torch.bool)
        padding_mask[:n_obs] = False
        
        return padded_data, torch.tensor(label_bits, dtype=torch.float32), padding_mask

# ================= 4. 神经科学家模型 (Neural Scientist) =================
class NeuralScientist(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 1. Observer (MLP Encoder) ---
        # 负责把单个 (X, Y) 变成隐状态
        self.encoder = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.MLP_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(config.MLP_HIDDEN),
            nn.Linear(config.MLP_HIDDEN, config.MLP_HIDDEN),
            nn.GELU(),
            nn.LayerNorm(config.MLP_HIDDEN),
            nn.Linear(config.MLP_HIDDEN, config.EMBED_DIM) # 投影到 Attention 维度
        )
        
        # --- 2. Reasoner (Attention Aggregator) ---
        # 负责综合多个样本的隐状态
        # 使用 Transformer Encoder，无位置编码 (Permutation Invariant)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.ATTN_HEADS,
            dim_feedforward=config.MLP_HIDDEN,
            dropout=0.1,
            batch_first=True
        )
        self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=config.ATTN_LAYERS)
        
        # --- 3. Decider (Decoder Head) ---
        # 从综合特征解码规则
        self.decoder = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(config.MLP_HIDDEN, config.OUTPUT_DIM)
        )

    def forward(self, x, padding_mask):
        """
        x: [Batch, Max_Obs, 60]
        padding_mask: [Batch, Max_Obs] (True for padded)
        """
        B, N, D = x.shape
        
        # 1. Encode: 每个样本独立通过 MLP
        # Flatten: [B*N, 60]
        x_flat = x.view(-1, D)
        embeddings = self.encoder(x_flat) # [B*N, Embed_Dim]
        
        # Reshape back: [B, N, Embed_Dim]
        embeddings = embeddings.view(B, N, -1)
        
        # 2. Aggregate: 通过 Self-Attention 交互
        # 注意：Transformer 需要 src_key_padding_mask
        attn_out = self.aggregator(embeddings, src_key_padding_mask=padding_mask) # [B, N, Embed_Dim]
        
        # 3. Pooling: 取有效样本的平均值作为 Global Context
        # 需要排除 Padding 的影响
        mask_float = (~padding_mask).float().unsqueeze(-1) # [B, N, 1], 1 for real
        sum_embeddings = (attn_out * mask_float).sum(dim=1) # [B, Embed_Dim]
        count = mask_float.sum(dim=1) # [B, 1]
        global_context = sum_embeddings / (count + 1e-9) # Avoid div 0
        
        # 4. Decode
        logits = self.decoder(global_context) # [B, 9]
        
        return logits

# ================= 5. 训练循环 =================
def train():
    config = Config()
    device = config.DEVICE
    print(f"Running on {device}")
    
    model = NeuralScientist(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 动态生成数据，不需要预先生成文件
    train_ds = DynamicRuleDataset(config.TRAIN_SAMPLES, config.MIN_OBSERVE, config.MAX_OBSERVE)
    val_ds = DynamicRuleDataset(config.VAL_SAMPLES, config.MAX_OBSERVE*5, config.MAX_OBSERVE*5) # 验证时用最大样本数
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    best_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y, mask in pbar:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        correct_rules = 0
        total_rules = 0
        
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x, mask)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                # 检查每一位是否都对 (Exact Match)
                is_correct = torch.all(preds == y, dim=1)
                correct_rules += is_correct.sum().item()
                total_rules += y.size(0)
                
        acc = 100 * correct_rules / total_rules
        print(f"Epoch {epoch+1} Val Accuracy (Exact Match): {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! Acc: {best_acc:.2f}%")
            
            if acc >= 100.0:
                print("已达到完美收敛！停止训练。")
                break

if __name__ == "__main__":
    train()
