# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:17:30 2025

@author: 95811
"""

import torch
 
# 检查PyTorch版本及CPU支持
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())  # 应输出 False
 
# 基础张量运算测试
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print("张量相加结果:", c)
 
# 验证torchvision和torchaudio是否安装（不报错即可）
try:
    import torchvision
    import torchaudio
    print("\n验证通过！所有包均已正确安装。")
except ImportError as e:
    print("\n安装异常:", e)
