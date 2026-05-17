import os

# 文件夹路径
folder_path = r'C:\Users\78456\Desktop\data\Annotations'  # 替换为你的目标文件夹路径

for filename in os.listdir(folder_path):
    name, ext = os.path.splitext(filename)

    if 'd' in name:
        # 找到第一个字母 'd' 出现的位置
        index = name.find('d')
        new_name = name[:index] + ext

        # 重命名文件
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)

        print(f"已重命名: {filename} -> {new_name}")

print("全部重命名完成。")
