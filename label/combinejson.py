import json
import glob

# 定义要合并的 JSON 文件路径（例如所有 .json 文件）
input_files = glob.glob('path/to/json_files/*.json')  # 修改路径到你的JSON文件目录
combined_data = []

# 逐个读取 JSON 文件并添加到列表
for file in input_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            combined_data.append(json.loads(line))

# 将合并后的数据写入到一个新的 JSON 文件
output_file = 'combined.json'
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in combined_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"所有 JSON 文件已合并到 {output_file}")
