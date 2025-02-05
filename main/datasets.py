import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BodyFatDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        print(f"正在加载 CSV 文件: {csv_file}")
        with open(csv_file, 'r') as f:
            self.data = [json.loads(line) for line in f]

        print(f"总数据条目: {len(self.data)}")
        print("数据示例:")
        for i, item in enumerate(self.data[:3]):
            print(f"条目 {i}: {item}")

        self.root_dir = root_dir
        print(f"根目录: {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            img_filename = item.get('image', '')
            
            if img_filename.startswith("._"):
                logger.warning(f"跳过隐藏文件: {img_filename}")
                return self.__getitem__((idx + 1) % len(self.data))  # 跳到下一个样本

            if not img_filename:
                raise ValueError(f"索引 {idx} 缺少图像文件名")

            category_map = {
                '1-4': 0, '5-7': 1, '8-10': 2, '11-12': 3,
                '13-15': 4, '16-19': 5, '20-24': 6,
                '25-30': 7, '35-40': 8
            }
            class_label = category_map.get(item['category'])
            if class_label is None:
                raise ValueError(f"无效的类别: {item['category']}")

            img_path = os.path.join(self.root_dir, img_filename)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"文件 {img_path} 不存在")

            img = Image.open(img_path).convert('RGB')

            # 打印调试信息
            logger.debug(f"图像: {img_path}, 模式: {img.mode}, 尺寸: {img.size}")

            if self.transform:
                img = self.transform(img)

            logger.debug(f"转换后的张量形状: {img.shape}")

            class_label = torch.tensor(class_label, dtype=torch.long)
            return img, class_label

        except Exception as e:
            logger.error(f"加载图像时发生错误: {e}")
            traceback.print_exc()
            raise