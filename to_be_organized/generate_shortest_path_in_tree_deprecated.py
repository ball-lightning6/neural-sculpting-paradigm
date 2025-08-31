import os
import numpy as np
import networkx as nx
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil


class Config:
    """数据集生成配置 (V2 - 无交叉版)"""
    # 1. 目录和样本数量
    DATA_DIR = "shortest_path_dataset_planar"  # 新建一个目录以示区别
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    EVAL_DIR = os.path.join(DATA_DIR, "eval")
    NUM_SAMPLES_TRAIN = 1000
    NUM_SAMPLES_EVAL = 200

    # 2. 图像属性
    IMG_SIZE = 256
    BG_COLOR = (0, 0, 0)

    # 3. 绘图元素样式
    NODE_RADIUS = 4
    LINE_THICKNESS = 2
    PATH_THICKNESS = 3

    NODE_COLOR = (128, 128, 128)
    START_COLOR = (0, 255, 0)
    END_COLOR = (255, 0, 0)
    PATH_COLOR = (0, 192, 255)

    # 4. "幼儿园" 级别图属性 (使用树来保证无交叉)
    MIN_NODES = 6
    MAX_NODES = 15

    # 5. 模糊性检查阈值
    MIN_AMBIGUITY_RATIO = 0.15


def scale_pos(pos, size, padding=0.1):
    """将任意范围的坐标缩放到图像尺寸"""
    # (代码与上一版相同，无需改动)
    min_coord = np.min(list(pos.values()), axis=0)
    max_coord = np.max(list(pos.values()), axis=0)
    coord_range = max_coord - min_coord
    coord_range[coord_range==0] = 1
    scaled_pos = {}
    for node, (x, y) in pos.items():
        norm_x = (x - min_coord[0]) / coord_range[0]
        norm_y = (y - min_coord[1]) / coord_range[1]
        img_x = norm_x * (size * (1 - 2 * padding)) + (size * padding)
        img_y = norm_y * (size * (1 - 2 * padding)) + (size * padding)
        scaled_pos[node] = (img_x, img_y)
    return scaled_pos


def draw_graph(G, pos, size, config, start_node=None, end_node=None, highlight_path=None):
    """在PIL图像上绘制图形"""
    # (代码与上一版相同，无需改动)
    img = Image.new('RGB', (size, size), config.BG_COLOR)
    draw = ImageDraw.Draw(img)
    scaled_pos = scale_pos(pos, size)
    # 1. 绘制所有普通边
    for u, v in G.edges():
        p1 = scaled_pos[u]
        p2 = scaled_pos[v]
        draw.line([p1, p2], fill=config.NODE_COLOR, width=config.LINE_THICKNESS)
    # 2. 绘制高亮路径
    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        for u, v in path_edges:
            p1 = scaled_pos[u]
            p2 = scaled_pos[v]
            draw.line([p1, p2], fill=config.PATH_COLOR, width=config.PATH_THICKNESS)
    # 3. 绘制所有普通节点
    for node, p in scaled_pos.items():
        draw.ellipse([p[0] - config.NODE_RADIUS, p[1] - config.NODE_RADIUS, p[0] + config.NODE_RADIUS,
                      p[1] + config.NODE_RADIUS], fill=config.NODE_COLOR)
    # 4. 绘制起点和终点
    if start_node is not None:
        p = scaled_pos[start_node]
        draw.ellipse([p[0] - config.NODE_RADIUS, p[1] - config.NODE_RADIUS, p[0] + config.NODE_RADIUS,
                      p[1] + config.NODE_RADIUS], fill=config.START_COLOR)
    if end_node is not None:
        p = scaled_pos[end_node]
        draw.ellipse([p[0] - config.NODE_RADIUS, p[1] - config.NODE_RADIUS, p[0] + config.NODE_RADIUS,
                      p[1] + config.NODE_RADIUS], fill=config.END_COLOR)
    return img


def create_single_sample(config):
    """生成一个有效的、无交叉的样本"""
    while True:
        # 1. 【核心修改】生成一个随机树，树保证了图的平面性（无交叉）
        num_nodes = np.random.randint(config.MIN_NODES, config.MAX_NODES + 1)
        # G = nx.random_tree(n=num_nodes)
        prufer_seq = np.random.randint(0, num_nodes, size=num_nodes - 2)
        G = nx.from_prufer_sequence(prufer_seq)

        # 2. 【核心修改】为树生成一个美观的、无交叉的布局
        # spring_layout 是一种力导向算法，能很好地展开图，避免交叉
        pos = nx.spring_layout(G, iterations=200)  # 增加迭代次数以获得更好布局

        # 3. 为边赋予权重 (欧几里得距离)
        for u, v in G.edges():
            p1 = np.array(pos[u])
            p2 = np.array(pos[v])
            G.edges[u, v]['weight'] = np.linalg.norm(p1 - p2)

        # 4. 随机选择起点和终点
        nodes = list(G.nodes())
        start_node, end_node = np.random.choice(nodes, 2, replace=False)

        # 5. 计算最短路径 (在树中，路径是唯一的)
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')

        # 对于树来说，任意两点间的最短路径是唯一的，所以不需要做复杂的模糊性检查
        # 我们可以直接生成图像
        input_img = draw_graph(G, pos, config.IMG_SIZE, config, start_node, end_node)
        output_img = draw_graph(G, pos, config.IMG_SIZE, config, start_node, end_node, highlight_path=shortest_path)
        return input_img, output_img


def generate_dataset(num_samples, output_dir, config):
    # (代码与上一版相同，无需改动)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"开始生成 {num_samples} 个样本到 '{output_dir}'...")
    for i in tqdm(range(num_samples)):
        input_img, output_img = create_single_sample(config)
        input_path = os.path.join(output_dir, f"{i}_input.png")
        output_path = os.path.join(output_dir, f"{i}_output.png")
        input_img.save(input_path)
        output_img.save(output_path)


if __name__=="__main__":
    cfg = Config()
    if os.path.exists(cfg.DATA_DIR):
        print(f"发现旧数据目录 '{cfg.DATA_DIR}', 正在删除...")
        shutil.rmtree(cfg.DATA_DIR)
    generate_dataset(cfg.NUM_SAMPLES_TRAIN, cfg.TRAIN_DIR, cfg)
    generate_dataset(cfg.NUM_SAMPLES_EVAL, cfg.EVAL_DIR, cfg)
    print("\n数据集生成完毕！")
    print(f"训练集: {cfg.NUM_SAMPLES_TRAIN} 个样本在 '{cfg.TRAIN_DIR}'")
    print(f"验证集: {cfg.NUM_SAMPLES_EVAL} 个样本在 '{cfg.EVAL_DIR}'")

