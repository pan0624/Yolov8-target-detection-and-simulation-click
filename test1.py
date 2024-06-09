from ultralytics import YOLO
import cv2
import numpy as np

# 加载YOLOv8模型
model = YOLO('runs/detect/train/weights/best.pt')

# 打开视频文件
cap = cv2.VideoCapture('test/test2.mp4')

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()
    if not success:
        break

    # 在帧上运行YOLOv8检测
    results = model.predict(frame)

    # 检查是否有检测结果
    if results:
        # 获取框和类别信息
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 修改为获取xyxy格式的边界框，并转换为numpy数组
        classes = results[0].boxes.cls.cpu().numpy()  # 获取类别索引，并转换为numpy数组

        # 在帧上展示结果
        annotated_frame = results[0].plot()  # 绘制检测结果

        # 展示带注释的帧
        annotated_frame = cv2.resize(annotated_frame, (640, 480))
        cv2.imshow('YOLOv8 Detection', annotated_frame)
    else:
        # 如果没有检测结果，直接展示原始帧
        cv2.imshow('YOLOv8 Detection', frame)

    # 如果按下'q'则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()