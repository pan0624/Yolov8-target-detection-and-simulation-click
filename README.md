# Yolov8-target-detection-and-simulation-click
YOLOv8算法在实时视频流目标检测中的应用，训练了自己的检测模型，并以此为基础，实现《植物大战僵尸》游戏中的自动模拟点击锤僵尸功能。
---
date: 星期日, 六月 9日 2024, 9:33:28 上午
lastmod: 星期日, 六月 9日 2024, 12:25:06 中午
---
# 环境部署
## cuda和cuDNN环境配置
### 检查自己的英伟达驱动版本

![image.png](https://s2.loli.net/2024/06/09/REqZN8ICamTzdxU.png)

### 根据对应表选择合适的cuda版本

![image.png](https://s2.loli.net/2024/06/09/n8H3NA1IxyRdMQV.png)

建议选择cuda11.8即可

### 下载并安装cuda和cuDNN
https://developer.nvidia.com/cuda-toolkit-archive 
cuda下载链接，下载选择11.8
https://developer.nvidia.com/cudnn-downloads
cuDNN下载链接，需要微软开发者账户，建议直接注册一个

![image.png](https://s2.loli.net/2024/06/09/p5gSdh4j9wPkOzs.png)

如图下载本地安装包，避免网络问题
安装过程不再赘述，建议解压和安装路径都用默认且避免中文路径，选择自定义安装，不要勾选visual studio即可，是否覆盖安装显卡驱动请随意

命令行输入nvcc --version 如果返回如下信息证明安装成功

![image.png](https://s2.loli.net/2024/06/09/asZ79Trbe5Gvzl8.png)

接下来进行cuDNN配置
下载选项如图

![image.png](https://s2.loli.net/2024/06/09/pG5wQJCNiXe3cPF.png)

目前暂时没有win11专用版本，使用win10版本没影响
进入路径
```js
 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```
将下载下来的cuDNN压缩包内bin,include,lib文件夹复制到前面给的路径内即可
![image.png](https://s2.loli.net/2024/06/09/CDpKfOgLsmdAqy3.png)

## python虚拟环境配置
conda安装使用请自行参考教程，可能碰到的问题链接
https://blog.csdn.net/u010393510/article/details/130715238

首先使用conda创建环境
```js
conda create  -p  D:\yolov8\yolo python=3.9
```

需要注意这里的参数 -p为指定路径在D:\yolov8\路径下创建一个名为yolo的虚拟python环境，并且指定python版本为3.9，这样做的好处是python环境直接下载到当前项目内，不需要做链接，项目转接给别人也能快速上手

![image.png](https://s2.loli.net/2024/06/09/8fLyDBaGNRctiIC.png)

这里可以看到对应路径下环境已经存在，需要注意yolo文件夹下放的是虚拟环境相关文件
这里可以直接cd到D:\yolov8路径下然后conda activate D:\yolov8\yolo激活虚拟环境

![image.png](https://s2.loli.net/2024/06/09/bGDfamwO9tEUIK2.png)

但是在对于包管理以及后续脚本运行不是很方便这里我们使用pycharm来管理整个项目并导入这个已经存在的conda环境

pycharm打开整个项目文件夹
添加已经存在的conda解解释器
具体操作如图
![IMAGE 2024-06-09 10:21:55.jpg](https://s2.loli.net/2024/06/09/QABxaM1oUr2LZvK.jpg)

![IMAGE 2024-06-09 10:22:12.jpg](https://s2.loli.net/2024/06/09/voRljSbhZUdk2er.jpg)

![IMAGE 2024-06-09 10:22:26.jpg](https://s2.loli.net/2024/06/09/q7QWSVw4nfErsCt.jpg)

打开pycharm自带的终端

![image.png](https://s2.loli.net/2024/06/09/6UCEaMPA7JSpTF5.png)

powershell前面括号如图显示证明配置正确

## 依赖下载
首先不要进行 pip install -r requirements.txt
因为默认下载的pytorch是cpu版本，需要自己先下载对应版本的pytorch，我的设备是英伟达的显卡，所以下载cuda11.8 对应的torch
 https://pytorch.org/get-started/locally/
 
 ![image.png](https://s2.loli.net/2024/06/09/25bQEU84YFhXZkW.png)
如图选择，官方就已经给出了要执行的命令，在pycharm的终端里面执行即可
这里给出一个用于测试cuda是否可用的小脚本
```python
import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 如果CUDA可用，列出CUDA设备
if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
```

![image.png](https://s2.loli.net/2024/06/09/umzLXnC3N6xUMWk.png)

随后执行
```python
pip install -r requirements.txt
pip install ultralytics
# ultralytics中包含了yolov8，不需要额外pip install yolo，这里的yolo下载下来居然是一个管理amp的包，会造成yolo命令冲突而失效
```
验证yolo环境
```js
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```
在路径runs/detect/predict下可以看见一张标注出红框的图片即表示安装成功
# 模型训练
## 配置文件
本项目的目的只是为了对对象进行识别，不需要对轮廓进行分割等操作，所以使用ultralytics训练好的预训练模型yolov8n.pt，首先在官网查看coco8的配置文件进行参考
```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO8 dataset (first 8 images from COCO train2017) by Ultralytics
# Documentation: https://docs.ultralytics.com/datasets/detect/coco8/
# Example usage: yolo train data=coco8.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco8  ← downloads here (1 MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8 # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: https://ultralytics.com/assets/coco8.zip
```
大致结构如下
```yaml

path: ../datasets/coco8 # dataset root dir 训练的数据集在的根目录
train: images/train # train images (relative to 'path') 4 images 训练图片
val: images/val # val images (relative to 'path') 4 images 验证图片
test: # test images (optional) 测试图片

# Classes 人工标注的框的种类
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
```
进行修改
```yaml
path: D:\yolov8\datasets\pvz  # dataset root dir
train: images  # train images (relative to 'path') 128 images
val: images  # val images (relative to 'path') 128 images

names:
  0: common
  1: hat
  2: cat
  3: iron
  4: sun
```
这里对于僵尸头上带的物品，及僵尸头本身，和阳光进行了标注
保存为pvztrain.html这里我是直接放在了D:\yolov8\datasets\pvz下面
## 文件路径
```
datasets
	|--coco8
	|__pvz
		|--images
		|--labels
		|__pvztrain.yaml
```
这里需要注意的是images和labels下面还有train和val（验证）对应数据和文件夹

![image.png](https://s2.loli.net/2024/06/09/QwqbcDp9EFTCsmH.png)

## 数据标注
先把图片分成train和val两部分塞入images下两个对应文件夹，再批量重命名为序号.jpg
然后使用labelimg进行标注，具体操作不在多说，自行参考网上教程，需要注意的是保存的文件夹设置为labels下对应文件夹，格式选择yolo格式，只需要矩形框标注即可。
本项目提供标注好的数据集并且已经放在了对应位置可以直接使用

## 开始训练
参考一下官网提供的示例脚本
![image.png](https://s2.loli.net/2024/06/09/8UF5EHgA361xoaZ.png)

我们的数据集并不大只有不到200张图片，所以不需要特意设置onnx格式提高速度并降低精度，同样对于使用模型检测一张图片也不需要，后续通过另外的脚本直接截取视频流进行检测
```python
from ultralytics import YOLO

def main():
   # Load model
   model = YOLO("yolov8n.pt")

   # Train
   model.train(data="datasets/pvz/pvztrain.yaml", epochs=250,patience=150)

   # Validate
   model.val()

if __name__ == "__main__":
   main()
#这里设置轮数为250轮以提高精度，但其实100轮后提升效果就已经区别不大，设置patience=150，即在150轮后检测如果已经无提升则直接结束训练
```
执行这个脚本，会在run路径下生成对应文件和模型

![image.png](https://s2.loli.net/2024/06/09/FsQENkwr1fK7dDM.png)

如图，train下面就是模型本身，train2下面就是对于这个模型进行的各种数学评估的图片
# 测试效果
这里对于测试模型效果，因为训练过程中就能看到拟合程度已经很高，所以直接下载了一个敲僵尸的游戏视频进行目标检测
```python
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
```

![image.png](https://s2.loli.net/2024/06/09/5qwlXti8JkSYIHB.png)

随意截取一帧，能看到识别精度很高

![image.png](https://s2.loli.net/2024/06/09/gyMLbUh2eBcJDoK.png)

对于复杂情况，抗干扰能力也很强
# 模拟点击
对于模拟点击的部分，直接使用的pyautogui库模拟的点击，使用win32gui来抓取的窗口
详细代码如下
```python
import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import torch

from ultralytics import YOLO

import pyautogui
from PIL import ImageGrab
import win32gui, win32con, win32com.client
import numpy as np
import time


def cilck_init():
    hwnd = win32gui.FindWindow(None, '植物大战僵尸中文版')
    print(hwnd)
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    win32gui.SetForegroundWindow(hwnd)
    window_x, window_y, right, bottom = win32gui.GetWindowRect(hwnd)
    box = (window_x, window_y, right, bottom)
    print(box)
    return box


def run(weights='runs/detect/train/weights/best.pt', source='self_data/pvz', imgsz=640, conf_thres=0.25,
        iou_thres=0.45):
    # Load model
    model = YOLO(weights)

    # Initialize click function
    box = cilck_init()

    num_pic = 1
    while num_pic:
        # Grab screenshot
        background_bgr = np.array(ImageGrab.grab(box))
        background = background_bgr[:, :, [2, 1, 0]]  # Convert BGR to RGB
        img_path = 'datasets/data/pvz/test.jpg'
        cv2.imwrite(img_path, background)

        # Perform inference
        results = model.predict(img_path, imgsz=imgsz, conf=conf_thres, iou=iou_thres)

        # Process results
        for result in results:
            for det in result.boxes.data:
                xyxy = det[:4].cpu().numpy().astype(int)
                conf = det[4].cpu().numpy()
                cls = int(det[5].cpu().numpy())
                print(xyxy, conf, cls)

                # Use pyautogui to click on detected coordinates
                pyautogui.click(box[0] + xyxy[0], box[1] + xyxy[1] + 20)

                # Additional processing or saving results can be added here

        num_pic += 1
        #time.sleep(1)  # Add a delay to avoid excessive clicking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='self_data/pvz', help='source')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()

    run(opt.weights, opt.source, opt.imgsz, opt.conf_thres, opt.iou_thres)


if __name__ == "__main__":
    main()

```

# 成果展示
https://mp4.ziyuan.wang/view.php/e88b97cc1951a8f05773b902dd9a004f.mp4
