from ultralytics import YOLO

if __name__ == '__main__':
    # 从头开始创建一个新的YOLO模型
    model = YOLO('yolov8n.yaml')

    # 加载预训练的YOLO模型（推荐用于训练）
    model = YOLO('yolov8n-seg.pt')

    # 使用数据集训练模型epochs个周期
    results = model.train(data='datasets/pvz/pvztrain.yaml', epochs=100, batch=4)

    # 评估模型在验证集上的性能
    results = model.val()

    # 使用模型对图片进行目标检测
    results = model('test/test.jpg')

    # 将模型导出为ONNX格式
    success = model.export(format='onnx')