# utils.py
# 工具函数集合，包含图像预处理、边缘检测、直线拟合等功能

import os
import cv2
import numpy as np
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_mask(mask_path):
    """
    读取掩膜图像
    :param mask_path: 掩膜图像路径
    :return: 掩膜图像（灰度图）
    """
    # 检查掩膜路径是否存在
    if not os.path.exists(mask_path):
        logging.error(f"Mask image path does not exist: {mask_path}")
        return None
    # 读取掩膜图像为灰度图
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logging.error(f"Failed to read mask image from {mask_path}")
    return mask

def grayscale(image):
    """
    图像灰度化处理
    :param image: BGR彩色图像
    :return: 灰度图像
    """
    # 使用OpenCV将BGR图像转换为灰度图像
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image, low_threshold, high_threshold):
    """
    使用Canny算法进行边缘检测
    :param image: 灰度图像
    :param low_threshold: 较低的阈值
    :param high_threshold: 较高的阈值
    :return: 边缘图
    """
    # 使用Canny算法检测图像边缘
    return cv2.Canny(image, low_threshold, high_threshold)

def gaussian_blur(image, kernel_size):
    """
    对图像进行高斯模糊降噪
    :param image: 输入图像
    :param kernel_size: 核大小（必须为奇数）
    :return: 模糊后的图像
    """
    # 使用高斯模糊减少图像噪声
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def region_of_interest(image, vertices):
    """
    提取图像中的感兴趣区域
    :param image: 输入图像
    :param vertices: 多边形顶点数组
    :return: 截取感兴趣区域后的图像
    """
    # 创建与输入图像大小相同的全零掩膜
    mask = np.zeros_like(image)
    # 根据图像通道数设置掩膜颜色
    ignore_mask_color = (255,) * image.shape[2] if len(image.shape) == 3 else 255
    # 在掩膜上填充多边形区域
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # 使用掩膜提取感兴趣区域
    return cv2.bitwise_and(image, mask)

def filter_white(image, car_mask_path, white_low, white_high):
    """
    过滤图像中白色区域，选取车道线的白色部分
    :param image: 输入图像（BGR）
    :param car_mask_path: 车辆遮罩路径，用于排除车辆区域
    :param white_low: 白色下限阈值
    :param white_high: 白色上限阈值
    :return: 过滤后的图像
    """
    # 定义白色像素的上下限
    lower_white = np.array([white_low] * 3)
    upper_white = np.array([white_high] * 3)
    # 创建白色区域的掩膜
    mask = cv2.inRange(image, lower_white, upper_white)
    # 如果提供了车辆遮罩路径，则读取遮罩并排除车辆区域
    if car_mask_path:
        car_mask = read_mask(car_mask_path)
        if car_mask is not None:
            car_mask = cv2.bitwise_not(car_mask)
            mask = cv2.bitwise_and(mask, car_mask)
    # 使用掩膜过滤图像，仅保留白色区域
    return cv2.bitwise_and(image, image, mask=mask)

def erode_dilate(img, erode_size):
    """
    先腐蚀后膨胀，去除噪点
    :param img: 输入图像
    :param erode_size: 腐蚀核大小
    :return: 处理后的图像
    """
    # 创建腐蚀和膨胀的核
    kernel = np.ones((erode_size, erode_size), np.uint8)
    # 对图像进行腐蚀操作
    eroded = cv2.erode(img, kernel, iterations=1)
    # 对图像进行膨胀操作
    return cv2.dilate(eroded, kernel, iterations=1)

def fit_line(line_list):
    """
    利用线性回归拟合直线，计算斜率、截距以及平均线长
    :param line_list: 直线列表（每条直线为[[x1, y1, x2, y2]]）
    :return: (斜率, 截距, 平均线长)
    """
    # 初始化x和y坐标列表，以及总线段长度
    xs, ys = [], []
    total_length = 0
    # 遍历每条直线，提取坐标并计算线段长度
    for line in line_list:
        x1, y1, x2, y2 = line[0]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        total_length += np.hypot(x2 - x1, y2 - y1)
    # 如果存在有效的x坐标，则进行线性回归拟合
    if xs:
        m, b = np.polyfit(xs, ys, 1)
        # 如果所有x值相同，则认为是垂直直线
        if np.allclose(xs, xs[0]):
            m = 999.0
            b = xs[0]
        # 计算平均线段长度
        avg_length = total_length / len(line_list)
        return m, b, avg_length
    # 如果没有有效直线，返回默认值
    return None, None, 0.0

def draw_lines(img, lines, slope_threshold, trap_height):
    """
    绘制车道线：区分左右车道线并拟合成一条直线，计算中心线
    :param img: 用于绘制直线的图像
    :param lines: 检测到的直线列表
    :param slope_threshold: 斜率阈值，过滤无效直线
    :param trap_height: 感兴趣区域高度比例
    :return: (所有直线图, 有效直线图, 左车道线图, 右车道线图, 中心线)
    """
    black_img = np.zeros(img.shape, dtype=np.uint8)
    # 如果没有检测到直线，返回空结果
    if lines is None or len(lines) == 0:
        return black_img, black_img, black_img, black_img, None

    # 绘制所有检测到的直线
    all_lines_img = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(all_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 初始化有效直线列表和斜率列表
    slopes = []
    valid_lines = []
    # 遍历每条直线，计算斜率并过滤无效直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = 999.0 if x2 == x1 else (y2 - y1) / (x2 - x1)
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            valid_lines.append(line)
    logging.info(f"有效直线/总直线数: {len(valid_lines)}/{len(lines)}")

    # 绘制有效直线
    valid_lines_img = img.copy()
    for line in valid_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(valid_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 如果没有有效直线，返回空结果
    if not valid_lines:
        logging.warning("未检测到有效直线")
        return all_lines_img, valid_lines_img, black_img, black_img, None

    # 根据图像中心区分左右车道线
    img_center = img.shape[1] / 2
    left_lines, right_lines = [], []
    for i, line in enumerate(valid_lines):
        x1, y1, x2, y2 = line[0]
        if slopes[i] > 0 and (x1 >= img_center or x2 >= img_center):
            right_lines.append(line)
        elif slopes[i] <= 0 and (x1 <= img_center or x2 <= img_center):
            left_lines.append(line)
        elif slopes[i] == 999.0:
            (left_lines if x1 <= img_center else right_lines).append(line)

    # 绘制左右车道线
    left_lines_img = img.copy()
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(left_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    right_lines_img = img.copy()
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(right_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 拟合左右车道线
    left_m, left_b, left_avg_len = fit_line(left_lines)
    right_m, right_b, right_avg_len = fit_line(right_lines)

    # 定义感兴趣区域的上下边界
    y1_coord = img.shape[0]
    y2_coord = int(img.shape[0] * (1 - trap_height))
    # 绘制拟合的左右车道线
    if right_m is not None:
        if right_m != 999.0:
            right_x1 = int((y1_coord - right_b) / right_m)
            right_x2 = int((y2_coord - right_b) / right_m)
        else:
            right_x1 = right_x2 = int(right_b)
        cv2.line(img, (right_x1, y1_coord), (right_x2, y2_coord), (255, 0, 0), 10)
    if left_m is not None:
        if left_m != 999.0:
            left_x1 = int((y1_coord - left_b) / left_m)
            left_x2 = int((y2_coord - left_b) / left_m)
        else:
            left_x1 = left_x2 = int(left_b)
        cv2.line(img, (left_x1, y1_coord), (left_x2, y2_coord), (255, 0, 0), 10)

    # 计算中心线
    if left_m is not None and right_m is not None:
        center_line = (left_x1, y1_coord, left_x2, y2_coord) if left_avg_len < right_avg_len else (right_x1, y1_coord, right_x2, y2_coord)
    elif left_m is not None:
        center_line = (left_x1, y1_coord, left_x2, y2_coord)
    elif right_m is not None:
        center_line = (right_x1, y1_coord, right_x2, y2_coord)
    else:
        center_line = None

    # 绘制中心线
    if center_line is not None:
        cv2.line(img, (center_line[0], y1_coord), (center_line[2], y2_coord), (0, 0, 255), 10)

    return all_lines_img, valid_lines_img, left_lines_img, right_lines_img, center_line

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, slope_threshold, trap_height):
    """
    通过Hough变换检测直线，并绘制左右车道线和中心线
    :param img: 边缘图（例如Canny输出）
    :param rho: 距离分辨率（以像素为单位）
    :param theta: 角度分辨率（以弧度为单位）
    :param threshold: 累加器阈值，只有超过该值的直线才会被检测到
    :param min_line_len: 最小直线长度（像素）
    :param max_line_gap: 最大间隙（像素），允许将两条线段连接成一条直线
    :param slope_threshold: 斜率过滤阈值，用于区分有效直线
    :param trap_height: 感兴趣区域的高度比例
    :return: (原始直线图, 各类直线图, 中心线)
    """
    # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # 创建一个空白图像，用于绘制检测到的直线
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    # 调用 draw_lines 函数，绘制所有直线、有效直线、左右车道线和中心线
    all_lines_img, valid_lines_img, left_lines_img, right_lines_img, center_line = draw_lines(
        line_img, lines, slope_threshold, trap_height
    )
    
    # 返回绘制的图像和中心线信息
    return line_img, all_lines_img, valid_lines_img, left_lines_img, right_lines_img, center_line


def weighted_img(img, initial_img, alpha=0.5, beta=0.5, gamma=0.0):
    """
    图像加权叠加
    :param img: 要叠加的图像（通常是标注后的图像）
    :param initial_img: 原始图像
    :param alpha: 原始图像的权重
    :param beta: 叠加图像的权重
    :param gamma: 叠加结果的亮度调整值
    :return: 叠加后的图像
    """
    # 使用 OpenCV 的 addWeighted 函数进行图像加权叠加
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def cal_point_point_distance(point1, point2):
    """
    计算两点间的欧式距离
    :param point1: 第一个点的坐标 (x1, y1)
    :param point2: 第二个点的坐标 (x2, y2)
    :return: 两点间的欧式距离
    """
    # 使用 NumPy 的 hypot 函数计算欧式距离
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


def cal_point_line_intersection(point, line):
    """
    计算点与直线的交点
    :param point: 点的坐标 (x0, y0)
    :param line: 直线的两个端点坐标 (x1, y1, x2, y2)
    :return: 点与直线的交点坐标 (x, y)
    """
    # 解构直线的两个端点坐标
    x1, y1, x2, y2 = line
    # 解构点的坐标
    x0, y0 = point
    
    # 如果直线是垂直线（x1 == x2），交点的 x 坐标为直线的 x 坐标，y 坐标为点的 y 坐标
    if x1 == x2:
        return int(x1), int(y0)
    
    # 如果直线是水平线（y1 == y2），交点的 y 坐标为直线的 y 坐标，x 坐标为点的 x 坐标
    if y1 == y2:
        return int(x0), int(y1)
    
    # 计算直线的斜率 k
    k = (y2 - y1) / (x2 - x1)
    
    # 使用直线方程和点的坐标计算交点的 x 和 y 坐标
    x = (x0 + k * y0 - k * y1 + k**2 * x1) / (k**2 + 1)
    y = (k**2 * y0 + k * x0 + y1 - k * x1) / (k**2 + 1)
    
    # 返回交点的整数坐标
    return int(x), int(y)

def concat_png_2_pdf(image_folder, suffix='png', file_sort=True):
    """
    将指定文件夹中的PNG图像按照文件名排序后纵向拼接成PDF文件, 在每张图的左下角标注图像名称
    :param image_folder: 图像文件夹路径
    :param suffix: 图像文件后缀名（默认为png）
    :param file_sort: 是否按文件名排序（默认为True）
    :return: None
    """
    image_files = [a for a in os.listdir(image_folder) if a.endswith(suffix)]
    if file_sort:
        image_files.sort()

    # 读取图像并添加名称水印
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to read image: {image_path}")
            continue
        # 在图像左下角添加名称水印
        cv2.putText(img, image_file, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        images.append(img)
        
    if len(images) == 0:
        logging.error("No valid images found to create PDF.")
        return

    # 拼接图像, 之间留一些空隙
    if len(images) > 1:
        images = [np.vstack([images[i], np.ones((50, images[i].shape[1], 3), dtype=np.uint8)*255]) for i in range(len(images) - 1)] + [images[-1]]
    else:
        images = [images[0]]
    

    # 纵向拼接图像
    pdf_image = np.vstack(images)

    # 将拼接后的图像转换为 RGB 格式以兼容 Pillow
    pdf_image_rgb = cv2.cvtColor(pdf_image, cv2.COLOR_BGR2RGB)

    # 使用 Pillow 保存为 PDF 文件
    pdf_path = os.path.join(image_folder, '..', 'output.pdf')
    pdf_image_pil = Image.fromarray(pdf_image_rgb)
    pdf_image_pil.save(pdf_path, "PDF", resolution=100.0)
    # logging.info(f"PDF saved to {pdf_path}")
