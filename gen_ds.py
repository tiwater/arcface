import os

def generate_dataset_txt(root_dir, output_file):
    # 打开文件，准备写入
    with open(output_file, 'w') as f:
        # 遍历根目录及其子目录
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                # 检查文件是否为图像文件（可以根据需要扩展文件类型）
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    # 获取文件的完整路径
                    file_path = os.path.join(subdir, file)
                    # 将文件路径写入dataset.txt
                    f.write(f"{file_path}\n")
    print(f"dataset.txt 文件已生成，路径为: {output_file}")

# 指定根目录路径
root_directory = '/rknn-workspace/dataset/lfw'
# 输出文件路径
output_file_path = 'dataset.txt'

# 生成 dataset.txt 文件
generate_dataset_txt(root_directory, output_file_path)