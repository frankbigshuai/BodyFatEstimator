from PIL import Image
import numpy as np
import os

def generate_transparent_images(original_folder, saliency_folder, output_folder):
    """
    使用显著性图批量生成透明背景的图片
    :param original_folder: 原始图片文件夹路径
    :param saliency_folder: 显著性图文件夹路径
    :param output_folder: 生成的透明背景图片文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 遍历显著性图文件夹中的所有文件
    for saliency_image_name in os.listdir(saliency_folder):
        saliency_image_path = os.path.join(saliency_folder, saliency_image_name)

        # 对应的原始图片路径
        original_image_path = os.path.join(original_folder, saliency_image_name)

        # 检查文件是否存在于原始图片文件夹中
        if not os.path.exists(original_image_path):
            print(f"原始图片缺失: {original_image_path}")
            continue

        # 打开原始图片和显著性图
        original_image = Image.open(original_image_path).convert("RGBA")
        saliency_image = Image.open(saliency_image_path).convert("L")  # 显著性图以灰度图加载

        # 调整显著性图大小与原图一致
        saliency_image = saliency_image.resize(original_image.size, resample=Image.BILINEAR)

        # 转换图片为 NumPy 数组
        original_image_np = np.array(original_image)
        saliency_np = np.array(saliency_image) / 255.0  # 将显著性图归一化到 [0, 1]

        # 修改 Alpha 通道：保留显著性区域，移除背景
        original_image_np[..., 3] = (original_image_np[..., 3] * saliency_np).astype(np.uint8)

        # 保存生成的透明背景图片
        transparent_image = Image.fromarray(original_image_np)
        output_image_path = os.path.join(output_folder, saliency_image_name)  # 保存的文件名与显著性图一致
        transparent_image.save(output_image_path)

        print(f"生成透明背景图片: {output_image_path}")

# 示例代码
original_folder = "test_data/test_images/"  # 原始图片文件夹路径
saliency_folder = "test_data/u2net_results/"  # 显著性图文件夹路径
output_folder = "test_data/output/"  # 输出透明背景图片文件夹路径

# 批量生成透明背景图片
generate_transparent_images(original_folder, saliency_folder, output_folder)
print("所有透明背景图片已处理完成。")
