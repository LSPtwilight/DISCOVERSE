import cv2
import os
import glob

# 设置图片文件夹路径和视频输出路径
image_folder = 'output_images'  # 替换为你的图片所在文件夹
output_video = 'output_images/output_video.mp4'  # 生成的视频文件名
fps = 20  # 每秒帧数

# 获取所有 PNG 图片，并按数字顺序排序（假设文件名包含数字）
images = sorted(glob.glob(os.path.join(image_folder, '*.png')), key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

# 确保至少有一张图片
if not images:
    print("No images found!")
    exit()

# 读取第一张图片以获取宽度和高度
frame = cv2.imread(images[0])
h, w, _ = frame.shape

# 定义视频编码器并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# 逐帧写入视频
for img_path in images:
    img = cv2.imread(img_path)
    video_writer.write(img)

# 释放资源
video_writer.release()
cv2.destroyAllWindows()

print(f"Video successfully saved as {output_video}")
