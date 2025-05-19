import torch

print(torch.cuda.device_count())  # 有多少张GPU
for i in range(torch.cuda.device_count()):
    print(f"{i}: {torch.cuda.get_device_name(i)}")
