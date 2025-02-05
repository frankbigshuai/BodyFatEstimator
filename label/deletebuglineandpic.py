import json
import os
import shutil

# 指定 JSON 文件和图片文件夹路径
json_file_path = '/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/label/photos_batch_1.json'
image_folder_path = '/Volumes/TOSHIBA/bodypicsdataset/zipevery1000/photos_batch_1'  # 图片文件夹路径
output_json_male = 'cleaned_data_male.json'
output_json_female = 'cleaned_data_female.json'

# 创建性别分类文件夹
male_image_folder = os.path.join(image_folder_path, 'male')
female_image_folder = os.path.join(image_folder_path, 'female')

# 确保目标文件夹存在
os.makedirs(male_image_folder, exist_ok=True)
os.makedirs(female_image_folder, exist_ok=True)

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 过滤数据并按性别分类存放
filtered_male_data = []
filtered_female_data = []

for item in data:
    image_path = os.path.join(image_folder_path, item["image"])
    
    if item["category"] == "bug":
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
    else:
        if item["gender"] == "男":
            filtered_male_data.append(item)
            target_path = os.path.join(male_image_folder, item["image"])
            shutil.move(image_path, target_path)
            print(f"Moved male image: {target_path}")
        elif item["gender"] == "女":
            filtered_female_data.append(item)
            target_path = os.path.join(female_image_folder, item["image"])
            shutil.move(image_path, target_path)
            print(f"Moved female image: {target_path}")

# 将过滤后的数据写入新的 JSON 文件
with open(output_json_male, 'w', encoding='utf-8') as f:
    for item in filtered_male_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(output_json_female, 'w', encoding='utf-8') as f:
    for item in filtered_female_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("数据清理和分类完成，已生成 cleaned_data_male.json 和 cleaned_data_female.json，并移动对应图片")
