import os

# 设置你的路径
image_folder = r'C:\Users\78456\Desktop\data\JPEGImages'  # 替换成你的图片文件夹路径
label_folder = r'C:\Users\78456\Desktop\data\Annotations'  # 替换成你的标签文件夹路径

# 获取标签文件名（不含扩展名）
label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f)))

# 遍历图片文件夹
for image_file in os.listdir(image_folder):
    image_name, ext = os.path.splitext(image_file)
    image_path = os.path.join(image_folder, image_file)

    # 如果图片文件名不在标签文件名中，就删除
    if image_name not in label_files:
        os.remove(image_path)
        print(f"已删除未标注图片: {image_file}")

print("处理完成。")
