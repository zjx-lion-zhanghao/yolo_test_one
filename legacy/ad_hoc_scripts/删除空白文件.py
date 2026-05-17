import os


def delete_empty_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在!")
        return
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.getsize(file_path) == 0:
                try:
                    os.remove(file_path)
                    print(f"已删除空文件: {file_path}")
                except Exception as e:
                    print(f"无法删除文件 {file_path}: {e}")


if __name__ == '__main__':
    folder = r'C:\Users\78456\Desktop\data\labels'
    delete_empty_files(folder)