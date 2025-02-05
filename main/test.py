from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

# 加载图片
img_path = "/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/bodyfatstandard/fit.webp"
image = Image.open(img_path).convert("RGB")

# 定义正确的数据增强管道
transform = transforms.Compose([
    transforms.Resize((256, 256)),                
    transforms.ToTensor(),                        # 先转换为张量
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 应用数据增强
transformed_img = transform(image)

# 将张量转换回可视化格式
def show_tensor_image(tensor):
    tensor = tensor.clone().detach().numpy().transpose(1, 2, 0)
    tensor = tensor * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    tensor = tensor.clip(0, 1)
    plt.imshow(tensor)
    plt.axis('off')
    plt.show()

# 显示增强后的图像
show_tensor_image(transformed_img)
