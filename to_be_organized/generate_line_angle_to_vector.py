import os
import random
import numpy as np
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
import time


class DatasetGenerator:
    """
    一个健壮的、用于生成“神经雷达”任务数据集的类。

    核心特性:
    - 保证每个角度区间最多一条线段。
    - 保证生成的标签组合在数据集中是唯一的。
    - 可配置的参数，如图片大小、区间数量等。
    - 清晰的进度反馈。
    """

    # --- 1. 配置参数 ---
    def __init__(self,
                 image_size=(224, 224),
                 num_angle_bins=36,
                 min_lines=1,
                 max_lines=10,
                 line_length_ratio=0.33,
                 min_line_width=1,
                 max_line_width=8):

        self.image_size = image_size
        self.num_angle_bins = num_angle_bins
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.line_length = int(image_size[0] * line_length_ratio)
        self.min_width = min_line_width
        self.max_width = max_line_width

        self.angle_per_bin = 360 / self.num_angle_bins
        self.center = (self.image_size[0] // 2, self.image_size[1] // 2)

    def _angle_to_bin(self, angle):
        return int(angle // self.angle_per_bin)

    def _generate_single_sample(self, num_lines, chosen_bins):
        """生成单个图片和标签的核心逻辑"""
        image = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(image)
        label = np.zeros(self.num_angle_bins, dtype=int)

        for bin_id in chosen_bins:
            angle_start = bin_id * self.angle_per_bin
            angle_end = (bin_id + 1) * self.angle_per_bin
            angle_deg = random.uniform(angle_start, angle_end)
            angle_rad = np.deg2rad(angle_deg)

            color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
            width = random.randint(self.min_width, self.max_width)

            end_x = self.center[0] + self.line_length * np.cos(angle_rad)
            end_y = self.center[1] + self.line_length * np.sin(angle_rad)
            draw.line([self.center, (end_x, end_y)], fill=color, width=width)

            label[bin_id] = 1

        return image, label.tolist()

    def generate_dataset(self, num_images, folder_name, ensure_unique_labels=True):
        """
        生成完整的数据集。

        Args:
            num_images (int): 目标图片数量。
            folder_name (str): 保存数据集的文件夹名 (e.g., 'train', 'val')。
            ensure_unique_labels (bool): 是否确保标签的唯一性。
                                         对于验证集，可以设为False。
        """
        print(f"\nGenerating dataset for '{folder_name}'...")

        image_folder = os.path.join(folder_name, 'images')
        os.makedirs(image_folder, exist_ok=True)

        metadata = {}
        # 使用集合来跟踪已生成的标签，以保证唯一性
        generated_labels = set()

        # 使用tqdm来显示进度条
        pbar = tqdm(total=num_images, desc=f"Creating '{folder_name}'")

        i = 0
        attempts = 0  # 记录为了找到唯一标签而尝试的次数
        max_attempts = num_images * 20  # 设置一个上限，防止死循环

        while i < num_images and attempts < max_attempts:
            attempts += 1
            num_lines = random.randint(self.min_lines, self.max_lines)

            # 1. 随机选择不重复的区间
            available_bins = list(range(self.num_angle_bins))
            chosen_bins = random.sample(available_bins, num_lines)

            # 将标签列表转换为元组，因为列表不可哈希
            label_tuple = tuple(sorted(chosen_bins))

            # 2. 检查标签是否已经生成过 (如果需要)
            if ensure_unique_labels and label_tuple in generated_labels:
                continue  # 如果重复，则跳过此次循环，重新生成

            generated_labels.add(label_tuple)

            # 3. 生成图片和最终的one-hot标签
            # 我们需要从chosen_bins重新生成完整的0/1向量
            final_label_vector = np.zeros(self.num_angle_bins, dtype=int)
            for bin_id in chosen_bins:
                final_label_vector[bin_id] = 1

            image, _ = self._generate_single_sample(num_lines, chosen_bins)

            filename = f"{i}.png"
            image.save(os.path.join(image_folder, filename))
            metadata[filename] = final_label_vector.tolist()

            i += 1
            pbar.update(1)

        pbar.close()

        if i < num_images:
            print(f"\nWarning: Could not generate {num_images} unique labels. "
                  f"Generated {i} samples after {max_attempts} attempts.")

        with open(os.path.join(folder_name, 'labels.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Successfully generated {i} images in '{folder_name}'.")


# --- 执行区 ---
if __name__=="__main__":
    # 初始化生成器
    generator = DatasetGenerator()

    # --- 生成训练集 ---
    # 我们希望训练集的标签尽可能多样，所以开启唯一性检查
    NUM_TRAIN_IMAGES = 10000
    generator.generate_dataset(NUM_TRAIN_IMAGES, 'line_angle', ensure_unique_labels=True)

    # --- 生成验证集 ---
    # 对于验证集，我们可以不强制要求标签唯一，因为它的主要目的是评估模型泛化能力
    # 当然，开启也无妨，这里为了演示设为False
    # NUM_VAL_IMAGES = 2000
    # generator.generate_dataset(NUM_VAL_IMAGES, 'val', ensure_unique_labels=False)

    print("\n--------------------")
    print("Dataset generation complete!")
    print("--------------------")

