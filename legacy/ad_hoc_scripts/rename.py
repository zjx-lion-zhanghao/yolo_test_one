import os

# 指定目标文件夹路径
folder_path = r"C:\Users\CQU-996\Desktop\新建文件夹"

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        name, ext = os.path.splitext(filename)
        base_name = f"{name}d"
        new_filename = base_name + ext
        new_path = os.path.join(folder_path, new_filename)

        counter = 1
        # 避免命名冲突
        while os.path.exists(new_path):
            new_filename = f"{base_name}{counter}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            counter += 1

        os.rename(old_path, new_path)
        print(f"重命名: {filename} → {new_filename}")
