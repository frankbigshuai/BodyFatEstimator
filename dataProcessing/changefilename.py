import os

def rename_files_with_offset(folder_path, offset):
    """
    从小到大顺序重命名文件，使用临时文件名避免冲突。
    
    :param folder_path: 文件夹路径
    :param offset: 偏移量（整数）
    """
    # 1. 收集并排序文件（从小到大）
    files_to_rename = []
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)
            if name.isdigit():
                files_to_rename.append((int(name), ext, old_path))
    
    # 从小到大排序
    files_to_rename.sort(key=lambda x: x[0])
    
    # 2. 首先将所有文件重命名为临时名称
    print("第1步：创建临时文件名...")
    for num, ext, old_path in files_to_rename:
        temp_name = f"temp_{num}{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)
        print(f"临时重命名: {num}{ext} -> {temp_name}")
    
    # 3. 然后将临时文件重命名为目标名称
    print("\n第2步：重命名为目标文件名...")
    success_count = 0
    for num, ext, _ in files_to_rename:
        temp_name = f"temp_{num}{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        new_name = f"{num + offset}{ext}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(temp_path, new_path)
        print(f"最终重命名: {temp_name} -> {new_name}")
        success_count += 1
    
    print(f"\n总结: 成功重命名 {success_count} 个文件")
    return success_count

# 使用示例
folder_path = "/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/images_from_Physiquecritique"

success = rename_files_with_offset(folder_path,5447)
# 运行诊断
