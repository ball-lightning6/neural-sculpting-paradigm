"""发布入口配置；默认只做已有证据检查，不启动训练或 SMC。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
WORK_DIR = ROOT / "_runs"
FIGURE_DIR = ROOT.parents[2] / "assets" / "figures"

TASK = "review"
ALLOW_TRAINING = False
ALLOW_SMC = False
ALLOW_FULL_ENUMERATION = False
DEVICE = "cuda:0"
CUDA_VISIBLE_DEVICES = "1"
RUN_HOURS = 12.0

# 只有显式配置完整目录时才运行 min-max；本包不附带旧 SMC 完整输出。
CATALOGUE_ROOT = None
# 可选逐样本 loss 配对；None 表示跳过，绝不伪造为零相关。
LOSS_PAIRING_DIR = None

