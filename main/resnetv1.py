import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 Bottleneck 模块（用于 ResNet-50、ResNet-101、ResNet-152）
class Bottleneck(nn.Module):
    expansion = 4  # 扩展因子，输出通道数为输入通道数的 4 倍

    def __init__(self, in_planes, planes, stride=1):
        """
        Args:
            in_planes (int): 输入通道数
            planes (int): 输出通道数（未扩展）
            stride (int): 卷积步幅，控制特征图的降采样
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)  # 1x1 卷积，改变通道数
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  # 3x3 卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)  # 1x1 卷积，扩展通道数
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # 如果步幅不为 1 或通道数不匹配，则使用 1x1 卷积调整
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 处理跳跃连接
        out = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积 + BN + ReLU
        out = F.relu(self.bn2(self.conv2(out)))  # 第二层卷积 + BN + ReLU
        out = self.bn3(self.conv3(out))  # 第三层卷积 + BN（不使用 ReLU）
        out += identity  # 跳跃连接：将输入与卷积输出相加
        out = F.relu(out)  # 对相加结果进行 ReLU 激活
        return out

# 定义 ResNet 主网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        """
        Args:
            block (class): 残差模块类型（Bottleneck 或 BasicBlock）
            num_blocks (list): 每个 Residual Block 中 Bottleneck 的数量
            num_classes (int): 输出类别数
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初始卷积层和 BN 层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 7x7 卷积，步幅为 2
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3x3 最大池化，步幅为 2

        # 四个 Residual Block
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x

        # 全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        创建一个 Residual Block 层。
        Args:
            block (class): 残差模块类型
            planes (int): 输出通道数（未扩展）
            num_blocks (int): Bottleneck 的数量
            stride (int): 步幅
        Returns:
            nn.Sequential: 包含多个 Bottleneck 的层
        """
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个 Bottleneck 的步幅为 stride，其余为 1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 初始卷积层 + BN + ReLU
        x = self.maxpool(x)  # 最大池化层
        x = self.layer1(x)  # 第一个 Residual Block
        x = self.layer2(x)  # 第二个 Residual Block
        x = self.layer3(x)  # 第三个 Residual Block
        x = self.layer4(x)  # 第四个 Residual Block
        x = self.avgpool(x)  # 全局平均池化层
        x = torch.flatten(x, 1)  # 展平
        out = self.linear(x)  # 全连接层
        return out

# 创建 ResNet-50 模型
def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
