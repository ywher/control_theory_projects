# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:36:01 2025

@author: 95811
"""

import importlib
deps = {
    "opencv-python": ["cv2", "4.11.0"],
    "numpy": ["numpy", "1.24.3"],
    "pandas": ["pandas", "2.0.3"],
    "requests": ["requests", "2.32.3"],
    "matplotlib": ["matplotlib", "3.7.5"],
    "scipy": ["scipy", "1.10.1"],
    "oauthlib": ["oauthlib", "3.0.0"],
    "tensorboard": ["tensorboard", "1.15.0"],
    "tqdm": ["tqdm", "4.67.1"],
    "pyyaml": ["yaml", "6.0.2"],
    "thop": ["thop", None]
}
def chk_ver(actual, expected): return actual.split('+')[0] == expected
results = []
for lib, (imp, exp) in deps.items():
    try:
        mod = importlib.import_module(imp)
        ver = getattr(mod, "__version__", "未知版本")
        stat = "✓" if (exp is None or chk_ver(ver, exp)) else "✗"
    except Exception as e:
        ver, stat = f"未安装({e})", "✗"
    results.append((lib, ver, stat))
print("\n验证结果：")
print(f"{'库名':<15} | {'实际版本':<15} | {'状态':<5}")
print("-"*40)
for lib, ver, stat in results:
    color = "\033[92m" if stat=="✓" else "\033[91m"
    print(f"{lib:<15} | {ver:<15} | {color}{stat}\033[0m")
if all(s=="✓" for _,_,s in results):
    print("\n所有依赖库验证通过！")
else:
    print("\n警告：部分依赖库未正确安装！")
