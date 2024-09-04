import os
from PIL import Image
from ultralytics import YOLO
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 指定文件夹路径
    folder_path = r"C:\Users\56274\Desktop\frame3500-end\image"
    # 加载之前训练完的模型
    model = YOLO(r"D:\JetBrains\cv_project\ultralytics-main\runs\pose\train7\weights\best.pt")
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 判断文件是否为图片文件
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 拼接文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 打开图片文件
            # image = Image.open(file_path)
            # 预测并生成标签
            results = model(file_path,save=True)
            # 进行相关操作，例如显示图片、处理图片等
            # image.show()