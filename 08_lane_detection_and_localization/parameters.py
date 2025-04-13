# parameters.py
# 参数配置文件，存放全局参数

import numpy as np

##### 建议不调整的参数 #####
# 白色像素阈值
WHITE_PIXEL_THRESHOLD_L = 80    # 低阈值
WHITE_PIXEL_THRESHOLD_H = 200   # 高阈值

# 图像处理参数
ERODE_SIZE = 3                  # 腐蚀核大小
GAUSSIAN_KERNEL_SIZE = 3        # 高斯模糊核大小

# Canny 边缘检测参数
CANNY_THRESHOLD1 = 80           # canny算法低阈值
CANNY_THRESHOLD2 = 200          # canny算法高阈值


##### 需要调整的参数 #####
# Hough变换参数
RHO = 5                         # Hough变换的距离分辨率
THETA = 5 * np.pi / 180         # Hough变换的角度分辨率
HOUGH_THRESHOLD = 30            # Hough变换的阈值
MIN_LINE_LENGTH = 30            # 最小线段长度
MAX_LINE_GAP = 30               # 最大线段间隔

# 车道线斜率阈值
SLOPE_THRESHOLD = 0.1           # 斜率阈值
