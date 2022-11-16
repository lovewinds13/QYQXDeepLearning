"""
测试GPU是否支持
"""

import torch


print(torch.cuda.is_available())
print(torch.cuda_version)
