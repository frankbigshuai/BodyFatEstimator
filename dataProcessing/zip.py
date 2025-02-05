import os
import zipfile
from tqdm import tqdm  # 引入 tqdm 库

# 原始照片文件夹路径
source_folder = "/Volumes/TOSHIBA/bodypicsdataset/withoutbackground/images_from_Physiquecritique_output"
# 输出 ZIP 文件存放路径
output_folder = "/Volumes/TOSHIBA/bodypicsdataset/zipevery1000"
# 每个 ZIP 文件包含的照片数量
photos_per_zip = 1000

# 获取所有照片文件（按文件名排序，确保顺序一致）
photos = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
photos.sort()

# 计算需要生成多少个 ZIP 文件
num_zips = (len(photos) + photos_per_zip - 1) // photos_per_zip

for i in range(num_zips):
    # 每个 ZIP 文件的名称
    zip_filename = os.path.join(output_folder, f"photos_batch_{i + 1}.zip")
    
    # 创建 ZIP 文件并添加进度条
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        start = i * photos_per_zip
        end = min(start + photos_per_zip, len(photos))  # 确保最后一批不超过总数
        for photo in tqdm(photos[start:end], desc=f"正在创建 {zip_filename}", unit="file"):
            photo_path = os.path.join(source_folder, photo)
            zipf.write(photo_path, photo)  # 第二个参数是文件在 ZIP 中的路径
    
    print(f"已创建：{zip_filename}，包含 {end - start} 张照片")

print("所有照片已分批打包完成！")
