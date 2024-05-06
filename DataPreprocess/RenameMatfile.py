import os
import csv
# from scipy.io import loadmat

# 指定路径
path_to_main_dict = "./Processed raw data_240313/03_RPBD_data"
label = os.path.basename(path_to_main_dict).split("_")[1]
folder_list = os.listdir(path_to_main_dict)     # 5 bands

for folder in folder_list:
    csv_output_file = os.path.join(path_to_main_dict,folder,f"renaming_info_HC_{folder}.csv")
    path_to_mat_files = os.path.join(path_to_main_dict,folder)
    # csv_output_file = os.path.join(path_to_mat_files, "renaming_info.csv")

    # 打开CSV文件进行写入
    with open(csv_output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # 写入CSV文件的标题行
        csv_writer.writerow(['Original Name', 'New Name'])

        # 遍历指定路径下的所有文件
        for i, filename in enumerate(os.listdir(path_to_mat_files)):
            if filename.endswith(".mat"):
                # 构建完整的文件路径
                mat_file_path = os.path.join(path_to_mat_files, filename)

                # # 读取MAT文件内容
                # mat_data = loadmat(mat_file_path)

                # 假设MAT文件中有一个名为'data'的变量
                # 根据实际情况调整变量名
                # data_variable = mat_data.get('data', None)

                # 构建新的文件名
                new_filename = f"{label}-{i}"

                # 重命名文件
                new_file_path = os.path.join(path_to_mat_files, new_filename + ".mat")
                os.rename(mat_file_path, new_file_path)
                print("{}->{}".format(mat_file_path, new_file_path))
                # 将信息写入CSV文件
                csv_writer.writerow([filename, new_filename])

    print("文件重命名和信息记录完成。")
    path_to_mat_files = ""
