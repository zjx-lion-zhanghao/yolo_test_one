import os


def find_and_delete_files(folder_path_1, folder_path_2):
    files_1 = os.listdir(folder_path_1)
    files_2 = os.listdir(folder_path_2)
    for file_1 in files_1:
        if os.path.isfile(os.path.join(folder_path_1, file_1)):
            file_name_1 = os.path.splitext(file_1)[0]
            found_match = False
            for file_2 in files_2:
                # 获取第二个文件夹中每个文件的文件名（不含扩展名）
                file_name_2 = os.path.splitext(file_2)[0]
                if file_name_1 == file_name_2:
                    found_match = True
                    break
            if not found_match:
                file_to_delete = os.path.join(folder_path_1, file_1)
                os.remove(file_to_delete)
                print(f"Deleted: {file_1} from {folder_path_1}")
            else:
                print(f"Found: {file_1} in both folders")
        else:
            print(f"Skipping non-file: {file_1}")


folder_path_1 = r"C:\Users\HP\Downloads\11111\jpg"  # 第1个文件夹路径
folder_path_2 = r"C:\Users\HP\Downloads\11111\labels"  # 第2个文件夹路径
# 根据2删除1,  1文件夹被删除
find_and_delete_files(folder_path_1, folder_path_2)