import os
import h5py
import numpy as np

dir_path = 'D:\\PycharmProjects\\DeepLearning\\Graduation Project\\Data\\Unprocessed raw data'
folder_list = os.listdir(dir_path)

print(folder_list)  # '01_HC_data', '02_PBD_data', '03_RPBD_data'

for i in range(0, len(folder_list)):
    # 指定包含MAT文件的文件夹路径
    folder_path = os.path.join(dir_path, folder_list[i])
    subfolder_bands = os.listdir(folder_path)     # 5 bands: '1-4', '1-45', '13-30', '30-45', '4-8', '8-13'
    # print(subfolder)

    for band in subfolder_bands:
        print("正在处理{}Hz".format(band))
        # 获取文件夹中所有文件的列表
        file_path = os.path.join(folder_path, band)
        file_list = os.listdir(file_path)

        # 筛选出所有以'.mat'为后缀的文件
        mat_files = [file for file in file_list if file.endswith('.mat')]
        num_mat_files = len(mat_files) # mat文件数
        print(folder_list[i][:-5], num_mat_files)

        for mat_file_name in mat_files:
            # 指定MAT文件的路径
            mat_file_path = os.path.join(file_path, mat_file_name)

            # 使用h5py库打开MAT文件
            with h5py.File(mat_file_path, 'r') as mat_file:
                # 获取MAT文件中的变量名列表
                variable_names = list(mat_file.keys())

                # 打印变量名
                for variable_name in variable_names:    # ['Matrix1']
                    print(f"Variable Name: {variable_name}")
                    # 获取变量的值
                    # variable_value = mat_file[variable_name]
                    variable_value = np.array(mat_file[variable_name])
                    print(f"Variable Value:\n{variable_value}")
                    print(variable_value.shape)
                    data_processed = [variable_value[i, :, :] for i in range(0, variable_value.shape[0], 50)]  # list

                    data_processed = np.stack(data_processed, axis=0).swapaxes(0, 2)  # list->ndarray
                    print(data_processed.shape)
                    from scipy.io import savemat

                    save_path = os.path.join("./Processed raw data",folder_list[i], band)
                    new_mat_file_path = os.path.join(save_path, mat_file_name)

                    if not os.path.exists(save_path):
                        # 如果不存在，创建文件夹
                        os.makedirs(save_path)
                        print(f"文件夹 '{save_path}' 创建成功")
                    else:
                        print(f"文件夹 '{save_path}' 已存在")

                    # 将 NumPy 数组放入字典，并保存到新的 MAT 文件中
                    processed_data_dict = {'T_time': data_processed}
                    savemat(new_mat_file_path, processed_data_dict, do_compression=True)
                    print(f"处理后的数据已保存到：{new_mat_file_path}")