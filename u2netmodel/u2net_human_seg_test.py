import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET  # full-size version


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


# save transparent background image directly
def save_transparent_output(original_image_path, pred, output_path):
    # Load original image
    original_image = Image.open(original_image_path).convert("RGBA")

    # Normalize saliency map and resize to match the original image
    pred_np = pred.squeeze().cpu().data.numpy()
    pred_image = Image.fromarray((pred_np * 255).astype(np.uint8)).convert("L")
    pred_image = pred_image.resize(original_image.size, resample=Image.BILINEAR)

    # Apply saliency map to alpha channel
    original_np = np.array(original_image)
    saliency_np = np.array(pred_image) / 255.0
    original_np[..., 3] = (original_np[..., 3] * saliency_np).astype(np.uint8)

    # Save transparent image
    transparent_image = Image.fromarray(original_np)
    transparent_image.save(output_path)

    print(f"生成透明背景图片: {output_path}")


def main():
    # --------- 1. Define paths ---------
    model_name = 'u2net'
    dataset_name = 'images_from_Physiquecritique'
    image_dir = os.path.join('/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/' +  dataset_name)
    output_dir = os.path.join(os.getcwd(), 'test_data',dataset_name + '_output')  # Transparent images
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(f"找到原始图片: {img_name_list}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --------- 2. Load dataset ---------
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. Load model ---------
    print("加载模型...")
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. Process each image ---------
    print("开始处理图片...")
    for i_test, data_test in enumerate(test_salobj_dataloader):
        original_image_path = img_name_list[i_test]
        output_image_path = os.path.join(output_dir, os.path.basename(original_image_path))

        print(f"正在处理: {original_image_path}")

        # Prepare input tensor
        inputs_test = data_test['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # Run inference
        d1, _, _, _, _, _, _ = net(inputs_test)
        pred = normPRED(d1[:, 0, :, :])

        # Save transparent background image
        save_transparent_output(original_image_path, pred, output_image_path)

        del d1

    print("所有图片处理完成，生成透明背景图片。")


if __name__ == "__main__":
    main()
