# lane_detector.py
# 车道线检测类，实现整体车道线检测流程

import os
import cv2
import numpy as np
import logging
from utils import (read_mask, grayscale, canny, gaussian_blur, erode_dilate,
                   filter_white, hough_lines, weighted_img, cal_point_point_distance,
                   cal_point_line_intersection, concat_png_2_pdf)
import parameters as params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LaneDetector:
    def __init__(self, car_mask=None, roi_mask=None):
        """
        初始化车道检测器
        :param car_mask: 车辆遮罩图像路径（用于排除车辆区域）
        :param roi_mask: 感兴趣区域遮罩图像路径（仅保留道路区域）
        """
        self.car_mask = car_mask
        self.roi_mask = roi_mask

    def add_text_to_image(self, image, text, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
        """
        在图像上添加文本
        :param image: 输入图像
        :param text: 要添加的文本
        :param position: 文本位置（x, y）
        :param font_scale: 字体缩放比例
        :param color: 文本颜色（BGR）
        :param thickness: 字体粗细
        :return: 添加文本后的图像
        """
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return image
    
    def process_image(self, image):
        """
        对单张图像进行车道线检测，并返回标注后的图像和拟合的中心线
        :param image: 输入图像（BGR）
        :param output_path: 输出图像保存路径（可选）
        :return: (标注后的图像, 中心线坐标元组)
        """
        # 1. 过滤图像，仅保留白色像素（车道线颜色）
        filtered = filter_white(image, self.car_mask, params.WHITE_PIXEL_THRESHOLD_L, params.WHITE_PIXEL_THRESHOLD_H)
        # 2. 灰度化
        gray = grayscale(filtered)
        # 3. 腐蚀和膨胀去噪
        eroded = erode_dilate(gray, params.ERODE_SIZE)
        # 4. 高斯模糊
        blur_gray = gaussian_blur(eroded, params.GAUSSIAN_KERNEL_SIZE)
        blur_gray = blur_gray.astype(np.uint8)
        # 5. Canny边缘检测
        edges = canny(blur_gray, params.CANNY_THRESHOLD1, params.CANNY_THRESHOLD2)

        # 6. 应用感兴趣区域掩膜（如果提供）
        if self.roi_mask and os.path.exists(self.roi_mask):
            roi = read_mask(self.roi_mask)
            masked_edges = cv2.bitwise_and(edges, edges, mask=roi)
        else:
            masked_edges = edges

        # 7. 利用Hough变换检测直线，并绘制车道线及中心线
        line_img, all_lines_img, valid_lines_img, left_lines_img, right_lines_img, center_line = hough_lines(masked_edges,
                                                         params.RHO, params.THETA, params.HOUGH_THRESHOLD,
                                                         params.MIN_LINE_LENGTH, params.MAX_LINE_GAP,
                                                         params.SLOPE_THRESHOLD, params.TRAP_HEIGHT)
        
        # 8. 图像叠加，将检测结果叠加到原图上
        annotated_image = weighted_img(line_img, image)
        
        concat_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # 水平拼接图像, 原图、过滤后的图、灰度图、腐蚀膨胀图、高斯模糊图、Canny 边缘检测图
        h_concat_image1 = cv2.hconcat([
                                    self.add_text_to_image(image, 'Image'),
                                    self.add_text_to_image(filtered, 'White Filtered'),
                                    self.add_text_to_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'Gray Image'),
                                    self.add_text_to_image(cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR), 'Erode Image'),
                                    self.add_text_to_image(cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2BGR), 'Gaussian Blur'),
                                    self.add_text_to_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 'Canny Edges')
                                    ])
        # 水平拼接图像, mask后的Canny 边缘检测图、所有直线图、有效直线图、所有左车道线图、所有右车道线图、车道中心线标注图
        h_concat_image2 = cv2.hconcat([
                                    self.add_text_to_image(cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR), 'Masked Edges'),
                                    self.add_text_to_image(all_lines_img, 'All Lines'),
                                    self.add_text_to_image(valid_lines_img, 'Valid Lines'),
                                    self.add_text_to_image(left_lines_img, 'Left Lane Lines'),
                                    self.add_text_to_image(right_lines_img, 'Right Lane Lines'),
                                    self.add_text_to_image(annotated_image, 'Center Line')
                                        ])
        concat_image = cv2.vconcat([h_concat_image1, h_concat_image2])
        
        
        
        return center_line, concat_image


    def transform_with_matrix(self, image, matrix, output_size, center_line, crop=False, margin=0):
        """
        使用透视变换对图像进行变换，并计算车中心与投影直线交点的距离
        :param image: 原始图像
        :param matrix: 透视变换矩阵
        :param output_size: 输出图像尺寸（宽, 高）
        :param center_line: 拟合的车道中心线
        :param crop: 是否裁剪图像
        :param margin: 裁剪时的边缘余量
        :return: (变换后的图像, 距离值)
        """
        img = image.copy()
        warped = cv2.warpPerspective(img, matrix, output_size)
        # 这里假设车辆中心点的像素坐标固定为 (320, 345)，根据实际情况修改
        car_center = np.array([[[320, 345]]], dtype=np.float32)
        car_center_warped = cv2.perspectiveTransform(car_center, matrix)[0][0]
        car_center_warped = tuple(map(int, car_center_warped))
        distance = 0.0
        if center_line is not None:
            x1, y1, x2, y2 = center_line
            line_arr = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
            line_warped = cv2.perspectiveTransform(line_arr, matrix).astype(np.int32)
            pt1, pt2 = tuple(line_warped[0][0]), tuple(line_warped[0][1])
            cv2.line(warped, pt1, pt2, (0, 0, 255), 3)
            intersection = cal_point_line_intersection(car_center_warped, (pt1[0], pt1[1], pt2[0], pt2[1]))
            cv2.line(warped, car_center_warped, intersection, (255, 255, 255), 3)
            cv2.circle(warped, intersection, 8, (255, 0, 0), -1)
            distance = cal_point_point_distance(car_center_warped, intersection)
            logging.info(f"投影车中心: {car_center_warped}, 直线: {pt1, pt2}, 交点: {intersection}, 像素距离: {distance:.1f}")
        if crop:
            crop_y = min(car_center_warped[1] + margin, warped.shape[0])
            warped = warped[:crop_y, :]
        # 在变换后的图像上绘制车中心点
        cv2.circle(warped, car_center_warped, 8, (0, 255, 0), -1)
        return warped, distance

    def detect_folder(self, input_folder, loc_gt_folder, output_folder, ipm_matrix_path, out_wh_path, resolution=50):
        """
        遍历文件夹中所有图像，对车道线进行标注与定位，并保存结果
        :param input_folder: 输入图像文件夹路径
        :param loc_gt_folder: 定位真值文件夹路径
        :param output_folder: 输出结果文件夹路径
        :param ipm_matrix_path: 透视变换矩阵路径（.npy文件）
        :param out_wh_path: 输出图像尺寸（宽高）的.npy文件路径
        :param resolution: 每米对应的像素数量（默认值为50）
        """
        # 定义支持的图像文件扩展名
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        # 如果输出文件夹不存在，则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_image_folder = os.path.join(output_folder, 'images')
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        
        
        # 定义定位结果输出文件路径
        loc_output_file = os.path.join(output_folder, 'loc_output.txt')
        
        # 打开定位结果文件以写入
        with open(loc_output_file, 'w') as loc_output:
            # 加载透视变换矩阵和输出图像尺寸
            M = np.load(ipm_matrix_path)
            output_size = tuple(np.load(out_wh_path))
            
            # 用于存储所有图像的定位误差
            all_errors = []
            
            # 遍历输入文件夹中的所有文件
            for filename in sorted(os.listdir(input_folder)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:  # 检查文件扩展名是否有效
                    # 获取对应的定位真值文件路径
                    gt_path = os.path.join(loc_gt_folder, filename.replace(ext, '.txt'))
                    loc_gt = None
                    
                    # 如果真值文件存在，则读取真值
                    if os.path.exists(gt_path):
                        with open(gt_path, 'r') as f:
                            line = f.readline().split(' ')[0]
                            try:
                                loc_gt = float(line)
                            except ValueError:
                                logging.warning(f"无法解析真值: {line} in {gt_path}")
                    
                    # 读取输入图像
                    in_path = os.path.join(input_folder, filename)
                    logging.info("----------------")
                    logging.info(f"处理图像: {in_path} ...")
                    image = cv2.imread(in_path)
                    if image is None:
                        logging.warning(f"无法读取 {in_path}")
                        continue
                    
                    # 对图像进行车道线检测和标注，并获取中心线（重要功能）
                    center_line, concat_image = self.process_image(image)
                    
                    # 使用透视变换对图像进行变换，并计算距离
                    warped, distance = self.transform_with_matrix(image, M, output_size, center_line)
                    warped = self.add_text_to_image(warped, 'Warped Image')
                    distance_m = distance / resolution  # 将像素距离转换为实际距离（米）
                    
                    # 保存透视变换后的图像
                    # 将warped图像保持长宽比的前提下高度设置为concat_image的高度
                    warped_height = concat_image.shape[0]
                    warped_width = int(warped.shape[1] * (warped_height / warped.shape[0]))
                    warped = cv2.resize(warped, (warped_width, warped_height))
                    out_warp = os.path.join(output_image_folder, filename)
                    cv2.imwrite(out_warp, cv2.hconcat([concat_image, warped]))
                    
                    # 判断中心线相对于车中心的位置, -1表示右侧，1表示左侧
                    center_line_side = None
                    if center_line is not None:
                        x1, y1, x2, y2 = center_line
                        bot_x = x1 if y1 >= y2 else x2
                        car_center_x = image.shape[1] // 2
                        center_line_side = 1 if bot_x > car_center_x else -1
                    
                    # 如果真值和中心线信息均存在，则计算定位误差
                    if loc_gt is not None and center_line_side is not None:
                        loc_error = abs(loc_gt - center_line_side * distance_m)
                        logging.info(f"{filename:<12} 真实位置: {loc_gt:>5.2f} 米, 预测位置: {center_line_side * distance_m:>5.2f} 米, 定位误差: {loc_error:>5.2f} 米")
                        loc_output.write(f"{filename:<12} 真实位置: {loc_gt:>5.2f} 米, 预测位置: {center_line_side * distance_m:>5.2f} 米, 定位误差: {loc_error:>5.2f} 米\n")
                        all_errors.append(loc_error)
                    else:
                        # 缺少中心线，定位误差设置为100，并提示缺少中心线
                        loc_error = 100.0
                        logging.warning("缺少中心线信息，定位误差设置为100.0 米")
                        loc_output.write(f"{filename:<12} 真实位置: {loc_gt:>5.2f} 米, 缺少中心线信息，定位误差设置为100.0 米\n")
                        all_errors.append(loc_error)
                    
            # 计算平均定位误差
            avg_error = np.mean(all_errors) if all_errors else 0.0
            logging.info("--------------------\n")
            logging.info(f"平均定位误差: {avg_error:.2f} 米")
            loc_output.write("--------------------\n")
            loc_output.write(f"平均定位误差: {avg_error:.2f} 米")

        # 生成PDF文件
        concat_png_2_pdf(output_image_folder, '.png', file_sort=True)