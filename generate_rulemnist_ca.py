# generate_rulemnist_ca.py
import os, gzip, types, sys
import numpy as np
from PIL import Image

# 先屏蔽 onnx（若你仍用屏蔽法）
# sys.modules['torch.onnx'] = types.ModuleType('torch.onnx')

RULE_MAP = {0: 30, 1: 110}   # 后续扩 2-9
RULES = {30: [0,0,0,1,1,1,1,0],   # Rule30
         110:[0,1,1,0,1,1,1,0]}   # Rule110

def rule_step(state, rule_bits):
    """通用 1-D CA 一步"""
    left = np.roll(state, 1); right = np.roll(state, -1)
    triple = left*4 + state*2 + right
    return np.array([rule_bits[t] for t in triple], dtype=np.uint8)

def mnist_img_by_label(images, labels, target_label, rng):
    idx_pool = np.where(labels == target_label)[0]
    i = rng.choice(idx_pool)
    return Image.fromarray(images[i], mode='L')

def state_mask_to_img(base_img: Image.Image, state_36, inv=False):
    """将 36 位状态映射到 6×6 反色遮罩"""
    canvas = np.array(base_img, dtype=np.uint8)
    h, w = canvas.shape
    cell_h, cell_w = h//6, w//6
    # mask = np.zeros_like(canvas,dtype=np.float32)
    for idx in range(36):
        r, c = divmod(idx, 6)
        y0, x0 = r*cell_h, c*cell_w
        if state_36[idx]==1:
            canvas[y0:y0+cell_h, x0:x0+cell_w]=255-canvas[y0:y0+cell_h, x0:x0+cell_w]
        # mask[y0:y0+cell_h, x0:x0+cell_w] = 255 if (state_36[idx] ^ inv) else 0
    # 反色叠加：0 区域保持原图，255 区域反色
    # canvas = canvas*(1-mask)+(255-canvas)*
    return Image.fromarray(canvas, mode='L')

def build_set(images, labels, N, split_name, rng, steps=3):
    out_dir = f'./autodl-tmp/rulemnist_ca_{steps}step/{split_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir,'input'), exist_ok=True)
    os.makedirs(os.path.join(out_dir,'output'), exist_ok=True)

    for i in range(N):
        # 1. 随机选 0/1 图 & 随机初始状态
        label = rng.choice([0, 1])
        base_img = mnist_img_by_label(images, labels, label, rng)
        base_img = base_img.resize((240,240))
        base_img.save('111.png')
        # print(base_img.shape,base_img.dtype)
        s0 = rng.randint(0, 2, 36, dtype=np.uint8)

        # 2. 演化
        s = s0.copy()
        rule_bits = RULES[RULE_MAP[label]]
        for _ in range(steps):
            s = rule_step(s, rule_bits)

        # 3. 生成输入/输出图像
        img_in  = state_mask_to_img(base_img, s0, inv=False)   # 1 保持，0 反色
        img_out = state_mask_to_img(base_img, s,  inv=False)   # 同上，但用演化后状态
        img_in.save(f'{out_dir}/input/{i:06d}.png')
        img_out.save(f'{out_dir}/output/{i:06d}.png')

        if i % 5000 == 0 and i:
            print(f'  {split_name} {i}/{N} done')

def main():
    np.random.seed(42)
    rng = np.random
    # 读取 MNIST
    def read_im(f): return np.frombuffer(gzip.open(f,'rb').read()[16:], dtype=np.uint8).reshape(-1,28,28)
    def read_lb(f): return np.frombuffer(gzip.open(f,'rb').read()[8:], dtype=np.uint8)
    train_im = read_im('mnist_dataset/train-images-idx3-ubyte.gz')
    train_lb = read_lb('mnist_dataset/train-labels-idx1-ubyte.gz')
    test_im  = read_im('mnist_dataset/t10k-images-idx3-ubyte.gz')
    test_lb  = read_lb('mnist_dataset/t10k-labels-idx1-ubyte.gz')

    print('Building train...')
    build_set(train_im, train_lb, 200000, 'train', rng, steps=3)
    print('Building val...')
    build_set(test_im, test_lb, 3000, 'val', rng, steps=3)
    print('All done ->', os.path.abspath('./rulemnist_ca_3step'))

if __name__ == '__main__':
    main()