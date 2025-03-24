import os
import zipfile

def zip_png_files(folder_path, zip_name):
    # 创建一个新的 ZIP 文件
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹中的所有文件
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.png'):
                    # 获取完整文件路径
                    file_path = os.path.join(foldername, filename)
                    # 将文件添加到 ZIP 文件中
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
                    print(f'Adding {file_path} to {zip_name}')
    print(f'All PNG files have been zipped into {zip_name}')

# 示例用法
folder_path = 'output_images'  # 替换为你的文件夹路径
zip_name = 'output_images/images.zip'  # 你希望生成的 ZIP 文件名
zip_png_files(folder_path, zip_name)
