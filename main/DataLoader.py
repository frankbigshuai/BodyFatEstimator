import random
from torch.utils.data import DataLoader, Subset, random_split
from datasets import BodyFatDataset
from torchvision import transforms
import torch
import os

def create_dataloaders(csv_file, root_dir, batch_size, train_split=0.7, val_split=0.2, num_workers=4, pin_memory=True):
    """
    创建训练集、验证集和测试集的 DataLoader。
    Args:
        csv_file (str): CSV 文件路径。
        root_dir (str): 图片存放的根目录。
        batch_size (int): 批量大小。
        train_split (float): 训练集比例。
        val_split (float): 验证集比例。
        num_workers (int): 加载数据的线程数。
        pin_memory (bool): 是否固定内存，加速 GPU 数据传输。
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 检查参数
    assert 0 < train_split < 1, "train_split must be between 0 and 1."
    assert 0 < val_split < 1, "val_split must be between 0 and 1."
    assert train_split + val_split < 1, "train_split + val_split must be less than 1 (test_split will be 1 - train_split - val_split)."

    # 定义训练集的增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机缩放和裁剪到 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # 随机擦除
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 定义验证集和测试集的预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),  # 先将短边调整为 256
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 调整图片大小为 224x224
    #     transforms.ToTensor(),  # 转换为张量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    # ])

    # 加载完整数据集
    full_dataset = BodyFatDataset(csv_file=csv_file, root_dir=root_dir, transform=None)

    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # 分别设置 transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # 动态设置 pin_memory 和 num_workers
    pin_memory = pin_memory if torch.cuda.is_available() else False
    num_workers = min(num_workers, os.cpu_count() - 1)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader