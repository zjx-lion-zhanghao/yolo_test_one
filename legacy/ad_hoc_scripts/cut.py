import os
import shutil

# 源文件夹路径（替换为你的路径）
source_folder = r'C:\path\to\source'

# 目标文件夹路径（替换为你的路径）
target_folder = r'C:\path\to\target'

# 如果目标文件夹不存在就创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 支持的图片扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    # 判断是否为文件且扩展名是图片类型
    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
        # 构造目标路径
        target_path = os.path.join(target_folder, filename)
        # 执行剪切操作
        shutil.move(file_path, target_path)
        print(f'已剪切：{filename}')

print("图片文件全部剪切完成！")
