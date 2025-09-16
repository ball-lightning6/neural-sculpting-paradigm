#!/usr/bin/env python3
# generate_mnist_ca_110.py
import os
import gzip
import numpy as np
from PIL import Image

# ---------- 参数 ----------
OUT_DIR = './autodl-tmp/mnist_ca110_240x240'
CELL_W = 40
GRID = 6
PIX_W = GRID * CELL_W          # 240
STEP_EVOL = 3
RULE = 110
# ---------------------------

def read_idx3_ubyte(fname):
    """返回 (N,28,28) 的 uint8 数组"""
    with gzip.open(fname, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_img = int.from_bytes(f.read(4), 'big')
        n_row = int.from_bytes(f.read(4), 'big')
        n_col = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_img, n_row, n_col)

def read_idx1_ubyte(fname):
    with gzip.open(fname, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_lab = int.from_bytes(f.read(4), 'big')
        return np.frombuffer(f.read(), dtype=np.uint8)

def rule110_step(state):
    """state: length-36 0/1 数组 -> 下一步长度-36 0/1 数组"""
    # 循环边界
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    triple = left * 4 + state * 2 + right   # 0-7
    # Rule110: 0,1,1,0,1,1,1,0  (位0是111, 位7是000)
    table = np.array([0,1,1,0,1,1,1,0], dtype=np.uint8)
    return table[triple]

def state_to_img(state, use_mnist, mnist_img=None):
    """
    将 36 位 0/1 数组 -> 240x240 PIL 图像
    use_mnist=True 时，用 mnist_img（28x28）填充对应格点
    """
    canvas = np.zeros((PIX_W, PIX_W), dtype=np.uint8)
    for idx in range(36):
        r, c = divmod(idx, GRID)
        y0, x0 = r * CELL_W, c * CELL_W
        if use_mnist:
            # 取对应 MNIST 图像块（0 或 1）
            label = state[idx]
            # 从 mnist_img 中随机裁剪一块（这里简单居中缩放）
            pil28 = Image.fromarray(mnist_img, mode='L')
            pil40 = pil28.resize((CELL_W, CELL_W), Image.LANCZOS)
            tile = np.asarray(pil40)
            canvas[y0:y0+CELL_W, x0:x0+CELL_W] = tile
        else:
            # 黑白棋盘格：1->黑(0), 0->白(255)
            col = 0 if state[idx] else 255
            canvas[y0:y0+CELL_W, x0:x0+CELL_W] = col
    return Image.fromarray(canvas, mode='L')

def sample_state(rng):
    return rng.randint(0, 2, size=36, dtype=np.uint8)

def build_split(rng, N, mnist_images, mnist_labels, split_name):
    base = os.path.join(OUT_DIR, split_name)
    dirs = [os.path.join(base, x) for x in
            ['input_mnist', 'input_bin', 'target_bin']]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 预过滤出 0 和 1 的 MNIST 图片索引
    avail_0 = np.where(mnist_labels == 0)[0]
    avail_1 = np.where(mnist_labels == 1)[0]
    rng.shuffle(avail_0)
    rng.shuffle(avail_1)
    ptr0 = ptr1 = 0

    def pop_mnist(label):
        nonlocal ptr0, ptr1
        if label == 0:
            idx = avail_0[ptr0 % len(avail_0)]
            ptr0 += 1
        else:
            idx = avail_1[ptr1 % len(avail_1)]
            ptr1 += 1
        return mnist_images[idx]

    for i in range(N):
        s0 = sample_state(rng)
        # 演化 3 步
        s = s0.copy()
        for _ in range(STEP_EVOL):
            s = rule110_step(s)
        # 取 MNIST 图像块（按 s0 的 0/1 选图）
        mnist_patches = [pop_mnist(int(x)) for x in s0]
        # 生成三张图
        img_in_mnist = state_to_img(s0, True,
                                    np.zeros((28,28), dtype=np.uint8))  # 占位，后面逐格替换
        # 实际逐格贴图
        canvas = np.full((PIX_W, PIX_W), 255, dtype=np.uint8)
        for idx in range(36):
            r, c = divmod(idx, GRID)
            y0, x0 = r * CELL_W, c * CELL_W
            tile = Image.fromarray(mnist_patches[idx], mode='L').resize(
                (CELL_W, CELL_W), Image.LANCZOS)
            canvas[y0:y0+CELL_W, x0:x0+CELL_W] = np.asarray(tile)
        Image.fromarray(canvas, mode='L').save(
            os.path.join(dirs[0], f'{i:06d}.png'))

        # 黑白输入
        img_in_bin = state_to_img(s0, False, None)
        img_in_bin.save(os.path.join(dirs[1], f'{i:06d}.png'))
        # 黑白目标
        img_tgt_bin = state_to_img(s, False, None)
        img_tgt_bin.save(os.path.join(dirs[2], f'{i:06d}.png'))

        if i % 5000 == 0 and i:
            print(f'  {split_name} {i}/{N} done')

def main():
    np.random.seed(42)
    rng= np.random
    print('Loading MNIST...')
    train_im = read_idx3_ubyte('mnist_dataset/train-images-idx3-ubyte.gz')
    train_lb = read_idx1_ubyte('mnist_dataset/train-labels-idx1-ubyte.gz')
    test_im  = read_idx3_ubyte('mnist_dataset/t10k-images-idx3-ubyte.gz')
    test_lb  = read_idx1_ubyte('mnist_dataset/t10k-labels-idx1-ubyte.gz')

    print('Building train...')
    build_split(rng, 150000, train_im, train_lb, 'train')
    print('Building val...')
    build_split(rng, 3000, test_im, test_lb, 'val')
    print('All done. ->', os.path.abspath(OUT_DIR))

if __name__ == '__main__':
    main()