import json
import pandas as pd

# 假设多人生成的标注文件分别为 labels1.json, labels2.json, labels3.json
file_list = ["labels1.json", "labels2.json", "labels3.json"]

# 读取并合并所有 JSON 文件
all_data = []
for file in file_list:
    with open(file, "r") as f:
        for line in f:
            all_data.append(json.loads(line))  # 按行读取 JSON 数据

# 去重（基于整个对象的内容）
all_data = pd.DataFrame(all_data).drop_duplicates().to_dict(orient="records")

# 保存合并结果为新的 JSON 文件
with open("merged_labels.json", "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")

print("所有标注文件已合并为 merged_labels.json")
