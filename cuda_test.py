import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 如果CUDA可用，列出CUDA设备
if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")