# main.py
# 主程序入口，解析参数并调用LaneDetector进行车道线检测与定位

import argparse
import os
import logging
import numpy as np
from lane_detector import LaneDetector
import parameters as params
from utils import read_mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="车道线检测与定位")
    parser.add_argument("-i", "--input_file", type=str, default="./data/image", help="输入图像文件或文件夹路径")
    parser.add_argument("-o", "--output_file", type=str, default="./data/output", help="输出图像文件或文件夹路径")
    parser.add_argument("--loc_gt_folder", type=str, default="./data/gt", help="横向定位真值文件夹路径")
    
    parser.add_argument("--car_mask", type=str, default='data/mask/car_mask.png', help="车辆遮罩图像路径")
    parser.add_argument("--roi_mask", type=str, default='data/mask/roi_mask.png', help="感兴趣区域遮罩图像路径")
    
    parser.add_argument("--ipm_matrix_path", type=str, default='data/ipm/ipm_matrix.npy', help="透视变换矩阵路径")
    parser.add_argument("--out_wh_path", type=str, default='data/ipm/ipm_size.npy', help="IPM输出图像尺寸文件路径")
    parser.add_argument("--resolution", type=float, default=50, help="每米对应的像素数")
    args = parser.parse_args()

    # 如果存在ROI遮罩，根据其范围更新TRAP_HEIGHT，用于限制直线在图像中的上下端点位置
    if args.roi_mask and os.path.exists(args.roi_mask):
        roi = read_mask(args.roi_mask)
        if roi is not None:
            y_coords = np.where(roi == 255)[0]
            if y_coords.size > 0:
                y_max, y_min = np.max(y_coords), np.min(y_coords)
                params.TRAP_HEIGHT = (y_max - y_min) / roi.shape[0]
                logging.info(f"更新 TRAP_HEIGHT: {params.TRAP_HEIGHT}")

    if not os.path.isdir(args.input_file):
        raise NotADirectoryError(f"{args.input_file} 不是有效的文件夹路径")
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)

    detector = LaneDetector(car_mask=args.car_mask, roi_mask=args.roi_mask)
    detector.detect_folder(args.input_file, args.loc_gt_folder, args.output_file,
                             args.ipm_matrix_path, args.out_wh_path, args.resolution)

if __name__ == '__main__':
    main()
