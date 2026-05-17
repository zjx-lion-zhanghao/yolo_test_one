import os
import shutil


def copy_file_multiple_times(source_file, destination_folder, copies=100):
    """
    将源文件复制多份到目标文件夹，文件名添加递增数字

    参数:
        source_file (str): 源文件路径（如 "C:/Users/xxx/file.jpg"）
        destination_folder (str): 目标文件夹路径（如 "C:/Users/xxx/folder"）
        copies (int): 要创建的副本数量，默认为100
    """
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"错误：源文件 '{source_file}' 不存在")
        return

    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 获取文件名和扩展名
    filename, extension = os.path.splitext(os.path.basename(source_file))

    # 复制文件
    for i in range(1, copies + 1):
        new_filename = f"{filename}_{i}{extension}"
        destination_path = os.path.join(destination_folder, new_filename)
        shutil.copy2(source_file, destination_path)
        print(f"已创建副本: {new_filename}")

    print(f"\n成功创建 {copies} 个副本在文件夹 '{destination_folder}' 中")


if __name__ == "__main__":
    # 示例用法（直接修改这里的路径）
    source_file = r"C:\Users\CQU-996\Desktop\frame_38_6w.txt"  # 替换成你的文件路径
    destination_folder = r"C:\Users\CQU-996\Desktop\10"  # 替换成你的目标文件夹

    # 调用函数复制文件（这里设置 copies=200）
    copy_file_multiple_times(source_file, destination_folder, copies=200)