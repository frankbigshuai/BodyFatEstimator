import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.multiprocessing import freeze_support
from resnet import resnet50
from DataLoader import create_dataloaders

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"测试批次 {batch_idx}: 图像形状: {images.shape}, 标签: {labels}")
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            print(f"模型输出形状: {outputs.shape}, 标签形状: {labels.shape}")

            # 打印模型输出的概率分布
            print(f"模型输出 (softmax): {outputs.softmax(dim=1)}")

            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    print(f"预训练模型测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")

def test_image_loading(csv_file, root_dir):
    with open(csv_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    for item in data[:10]:
        img_filename = item.get('image', '')
        if not img_filename:
            print("警告：找不到图像文件名")
            continue
        
        img_path = os.path.join(root_dir, img_filename)
        
        print(f"测试图像: {img_path}")
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
            print(f"成功加载 {img_path}")
            print(f"图像模式: {img.mode}")
            print(f"图像尺寸: {img.size}")
        except Exception as e:
            print(f"加载 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    # 配置参数
    csv_file = '/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/label/cleaned_data_male.json'
    root_dir = '/Volumes/TOSHIBA/bodypicsdataset/zipevery1000/photos_batch_1/male'
    pretrained_path = "/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/main/resnet50.pth"

    # 首先测试图像加载
    test_image_loading(csv_file, root_dir)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 DataLoader
    batch_size = 32
    train_split = 0.01  # 不使用训练集
    val_split = 0.01    # 50% 验证集
    num_workers = min(4, os.cpu_count() - 1)  # 动态设置 num_workers
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers,
        pin_memory=True
    )

    # 打印数据加载器信息
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    if len(test_loader) == 0:
        print("警告：测试集为空，请检查数据划分或数据加载逻辑。")

    # 创建模型
    model = resnet50(num_classes=9).to(device)  # 9 类分类任务

    # 加载预训练权重
    checkpoint = torch.load(pretrained_path, map_location=device)
    print("预训练权重加载成功，键值如下：")
    for key in checkpoint.keys():
        print(key)

    # 过滤掉 fc 层的权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if "fc" not in k}
    
    # 加载过滤后的权重
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # 替换最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9).to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    print(f"损失函数: {criterion}")

    # 测试模型性能
    test_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()