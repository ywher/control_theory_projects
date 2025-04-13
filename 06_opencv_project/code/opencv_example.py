# -*- coding: utf-8 -*-
import cv2
import os

"""在图像上添加文字标签"""
def add_text_to_image(img, text, pos=(45, 750), bgr=(0, 255, 0)):
    output = img.copy()
    cv2.putText(
        output, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, 2,
        bgr, 5, cv2.LINE_AA
    )
    return output

"""处理单个图像文件"""
def process_image_file(input_path, output_path, group_number):
    # 读入图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    # 处理图像
    processed_images = []
    
    # 0. 原始图像（BGR）
    processed_images.append(add_text_to_image(img, "Original"))
    
    # 1. 转换为RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_images.append(add_text_to_image(rgb, "RGB Mode"))
    
    # 2. 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 转为3通道
    processed_images.append(add_text_to_image(gray, "Gray"))
    
    # 横向拼接所有结果
    result = cv2.hconcat(processed_images)
    
    # 给图片添加组号信息
    result = add_text_to_image(result, f"Group: {group_number}", pos=(540, 85), bgr=(0, 0, 255))
    
    
    # 保存结果
    if not cv2.imwrite(output_path, result):
        print(f"保存失败: {output_path}")
    else:
        print(f"已保存: {output_path}")

"""主函数"""
def main(input_dir, output_dir, group_number=666):
    # 创建输出目录（递归创建）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 支持的图片扩展名后缀（小写）
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # 遍历输入目录
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            # 提取文件的扩展名并检查是否是支持的图片格式
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_exts:
                # 构建输出路径
                output_path = os.path.join(output_dir, filename)
                # 处理图片
                process_image_file(file_path, output_path, group_number)
    
    print(f"\n处理完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    # 配置输入图像的路径和输出结果的路径
    input_dir = "E:\\projects\\opencv_project\\images"
    output_dir = "E:\\projects\\opencv_project\\output"
    # 组号
    group_number = 6
    # 执行主函数功能
    main(input_dir, output_dir, group_number)