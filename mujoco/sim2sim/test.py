import torch
from torchinfo import summary

# state = torch.load('runs/CartpoleCmd.pth')
# print(type(state))

# print(state.keys())

# model = torch.load('runs/CartpoleCmd.pth')

pth_path = "runs/CartpoleCmd.pth"
model_data = torch.load(pth_path, map_location='cpu')
print(model_data)

# print("类型：", type(model_data))

# if isinstance(model_data, dict):
#     print("字典键名预览：")
#     print(list(model_data.keys())[:10])  # 只打印前10个键名
# else:
#     print("模型结构：")
#     print(model_data)
