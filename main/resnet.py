import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 Bottleneck Block（相比普通 Residual Block，增加了 1x1 卷积来进行通道压缩与扩展）
class Bottleneck(nn.Module):
    expansion = 4  # 定义扩展倍数，Bottleneck 会将输出通道扩展 4 倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 卷积层，压缩通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积层，提取特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积层，扩展通道数到 4 倍
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于调整维度以便进行残差连接

    def forward(self, x):
        identity = x  # 保存输入以进行跳跃连接
        if self.downsample is not None:  # 如果需要下采样或调整维度，则对输入进行 downsample
            identity = self.downsample(x)

        # 第一层卷积 + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积 + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三层卷积 + BN（这里不使用 ReLU，因为下一步是进行残差连接）
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接：将输入（identity）与卷积结果相加
        out += identity
        out = self.relu(out)  # 对相加结果进行 ReLU 激活

        return out


# 定义 ResNet 类
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始输入通道数为 64

        # 第一层：7x7 卷积，步幅为 2，输出通道数为 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建四个残差模块，每个模块包含多个 Bottleneck Block
        # layer1：不需要下采样，stride=1
        self.layer1 = self._make_layer(block, 64, layers[0])
        # layer2 至 layer4：步幅为 2，进行下采样
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化，将特征图降维为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，输出为 num_classes（类别数）
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        创建包含多个 Bottleneck Block 的残差模块
        Args:
            block (nn.Module): 残差块类型（这里是 Bottleneck）
            out_channels (int): 输出通道数
            blocks (int): 残差块的数量
            stride (int): 步幅
        """
        downsample = None
        # 如果 stride 不为 1 或输入通道数与输出通道数不匹配，则需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion  # 更新当前通道数

        # 后续残差块不需要下采样
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 创建 ResNet-50 模型实例
def resnet50(num_classes=1000):
    """
    创建 ResNet-50 模型
    Args:
        num_classes (int): 输出的类别数
    Returns:
        ResNet: ResNet-50 模型
    """
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":
    # 创建 ResNet-50 模型实例，输出类别数为 10
    model = resnet50(num_classes=9)
    # 随机生成一个输入张量（1 个样本，3 通道，224x224 尺寸）
    x = torch.randn(1, 3, 224, 224)
    # 前向传播，获取输出
    y = model(x)
    print(f"输出大小: {y.size()}")
