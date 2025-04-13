import os  # 导入操作系统模块，用于文件和路径操作
import cv2  # 导入OpenCV库，用于图像处理和绘图

def draw_bounding_boxes(image_dir, gt_dir, output_dir):
    """
    在图像上绘制真值标注框并保存。

    参数：
    - image_dir: 图片文件夹路径
    - gt_dir: 真值文件夹路径，每个txt文件对应一张图片，内容格式为：
              类别id 中心x坐标/图像宽 中心y坐标/图像高 box宽/图像宽 box高/图像高
              由于只有car类别，id都是0
    - output_dir: 保存结果的文件夹路径
    """
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件并排序
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)

    # 遍历每张图像
    for image_file in image_files:
        # 构建图像完整路径和对应的真值文件路径
        image_path = os.path.join(image_dir, image_file)
        gt_file = os.path.join(gt_dir, os.path.splitext(image_file)[0] + '.txt')

        # 检查真值文件是否存在
        if not os.path.exists(gt_file):
            print(f"真值文件不存在：{gt_file}")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件：{image_path}")
            continue

        # 获取图像尺寸
        height, width = image.shape[:2]

        # 读取真值文件并绘制矩形框
        with open(gt_file, 'r') as f:
            for line in f:
                # 解析每行数据
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"真值文件格式错误：{gt_file}")
                    continue

                # 提取边界框信息并转换为像素坐标
                class_id, cx_ratio, cy_ratio, w_ratio, h_ratio = map(float, parts)
                # 计算中心点坐标（像素）
                cx = int(cx_ratio * width)
                cy = int(cy_ratio * height)
                # 计算边界框宽高（像素）
                box_width = int(w_ratio * width)
                box_height = int(h_ratio * height)
                # 计算左上角坐标
                x_min = int(cx - box_width / 2)
                y_min = int(cy - box_height / 2)
                # 计算右下角坐标
                x_max = int(cx + box_width / 2)
                y_max = int(cy + box_height / 2)

                # 绘制矩形框（绿色，线宽2像素）
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # 在左上角显示类别ID（绿色，字体大小0.9，线宽2像素）
                cv2.putText(image, str(int(class_id)), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存带有标注的图像
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)
        print(f"已保存标注图像：{output_path}")

if __name__ == "__main__":
    # 当作为独立程序运行时执行
    import argparse  # 导入命令行参数解析模块

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="在图像上绘制真值标注框并保存。")
    # 添加命令行参数
    parser.add_argument("--image_dir", type=str, default="images/torcs", help="图片文件夹路径")
    parser.add_argument("--gt_dir", type=str, default="images/annotations", help="真值文件夹路径")
    parser.add_argument("--output_dir", type=str, default="images/torcs_with_anno", help="保存结果的文件夹路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数绘制边界框
    draw_bounding_boxes(args.image_dir, args.gt_dir, args.output_dir)