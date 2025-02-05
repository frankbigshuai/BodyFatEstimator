import os
import shutil
import hashlib

def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def safe_move_files(source_folder, destination_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹不存在: {source_folder}")
        return
    
    # 检查是否有足够的磁盘空间
    _, _, source_free = shutil.disk_usage(source_folder)
    _, _, dest_free = shutil.disk_usage(destination_folder)
    
    # 计算源文件夹大小
    total_size = sum(os.path.getsize(os.path.join(source_folder, f)) 
                    for f in os.listdir(source_folder) 
                    if os.path.isfile(os.path.join(source_folder, f)))
    
    if dest_free < total_size:
        print(f"错误: 目标磁盘空间不足。需要: {total_size/1024/1024:.2f}MB, 可用: {dest_free/1024/1024:.2f}MB")
        return

    os.makedirs(destination_folder, exist_ok=True)
    
    successful_moves = []
    failed_moves = []
    skipped_files = []
    
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        
        if not os.path.isfile(source_path):
            print(f"跳过: {source_path} 不是文件")
            skipped_files.append((source_path, "不是文件"))
            continue
            
        if os.path.exists(destination_path):
            print(f"警告: 目标路径已存在文件: {destination_path}")
            skipped_files.append((source_path, "目标文件已存在"))
            continue
            
        try:
            # 计算源文件哈希值
            source_hash = get_file_hash(source_path)
            
            # 复制文件
            shutil.copy2(source_path, destination_path)
            
            # 验证复制后文件的哈希值
            dest_hash = get_file_hash(destination_path)
            
            if source_hash == dest_hash:
                try:
                    os.remove(source_path)
                    successful_moves.append(source_path)
                    print(f"成功移动: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"复制成功但删除源文件失败: {source_path}")
                    failed_moves.append((source_path, f"删除源文件失败: {str(e)}"))
            else:
                print(f"文件验证失败: {source_path}")
                # 如果复制的文件与源文件不一致，删除目标文件
                os.remove(destination_path)
                failed_moves.append((source_path, "文件验证失败"))
                
        except Exception as e:
            print(f"移动文件时出错 {source_path}: {str(e)}")
            failed_moves.append((source_path, str(e)))
    
    print("\n移动操作完成!")
    print(f"成功: {len(successful_moves)} 个文件")
    print(f"失败: {len(failed_moves)} 个文件")
    print(f"跳过: {len(skipped_files)} 个文件")
    
    if failed_moves:
        print("\n失败的文件:")
        for path, reason in failed_moves:
            print(f"文件: {path}")
            print(f"原因: {reason}")
    
    if skipped_files:
        print("\n跳过的文件:")
        for path, reason in skipped_files:
            print(f"文件: {path}")
            print(f"原因: {reason}")

    return successful_moves, failed_moves, skipped_files

# 使用方法
source_folder = "/Volumes/TOSHIBA/bodypicsdataset/withoutbackground/images_from_bulkorcut_output"
destination_folder = "/Volumes/TOSHIBA/bodypicsdataset/withoutbackground/images_from_Physiquecritique_output"

# 在移动前询问用户
print(f"将要从 {source_folder} 移动文件到 {destination_folder}")
response = input("是否继续? (y/n): ")

if response.lower() == 'y':
    successful_moves, failed_moves, skipped_files = safe_move_files(source_folder, destination_folder)
else:
    print("操作已取消")