import os
import shutil


def organize_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在!")
        return
    xml_folder = os.path.join(folder_path, 'xml')
    jpg_folder = os.path.join(folder_path, 'jpg')
    os.makedirs(xml_folder, exist_ok=True)
    os.makedirs(jpg_folder, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                if filename.endswith('.xml'):  # 文件后缀为txt则修改为txt
                    try:
                        shutil.move(file_path, os.path.join(xml_folder, filename))
                        print(f"已将 {filename} 移动到 xml 文件夹")
                    except Exception as e:
                        print(f"无法移动文件 {filename}: {e}")
                elif filename.endswith(('.jpg', 'png')):
                    try:
                        shutil.move(file_path, os.path.join(jpg_folder, filename))
                        print(f"已将 {filename} 移动到 jpg 文件夹")
                    except Exception as e:
                        print(f"无法移动文件 {filename}: {e}")


if __name__ == '__main__':
    folder = r'C:\Users\HP\Downloads\11111'  # 修改为解压缩后的文件目录
    organize_files(folder)