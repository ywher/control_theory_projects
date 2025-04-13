import os
import glob
import argparse

# 解析预测或真值的标注文件
def parse_annotations(file_path, is_prediction=False):
    """
    解析预测或真值的标注文件，每行格式为：
      类别id  box中心x/图像宽  box中心y/图像高  box宽/图像宽  box高/图像高
    预测文件多一个置信度，格式为：
      类别id  cx_ratio  cy_ratio  w_ratio  h_ratio  confidence
    将中心格式转换为 [x_min, y_min, w, h] 形式。
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_prediction:
                if len(parts) < 6:
                    continue
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                confidence = float(parts[5])
            else:
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                confidence = None

            # 将中心坐标转换为左上角坐标
            x_min = cx - w / 2
            y_min = cy - h / 2
            annotations.append({
                'class_id': class_id,
                'bbox': [x_min, y_min, w, h],
                'confidence': confidence
            })
    return annotations

# 计算两个bbox之间的 IoU，bbox 格式为 [x_min, y_min, w, h]
def compute_iou(box1, box2):
    """
    计算两个bbox之间的 IoU，bbox 格式为 [x_min, y_min, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0
    return inter_area / union_area

# 针对单张图计算 TP、FP、FN。预测结果按置信度降序排列，并与真值逐一匹配。
def evaluate_image(predictions, ground_truths, iou_threshold=0.5):
    """
    针对单张图计算 TP、FP、FN。预测结果按置信度降序排列，并与真值逐一匹配。
    """
    TP = 0
    FP = 0
    matched_gt = set()

    predictions = sorted(predictions, key=lambda x: x['confidence'] if x['confidence'] is not None else 0, reverse=True)
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len(ground_truths) - len(matched_gt)
    return TP, FP, FN

# 结果评估主函数
def main(image_dir, pred_dir, gt_dir, iou_threshold=0.5):
    image_files = glob.glob(os.path.join(image_dir, '*.*'))
    image_files = sorted(image_files)
    total_TP, total_FP, total_FN = 0, 0, 0

    # 用于保存评估结果的所有输出行
    results_lines = []

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        pred_file = os.path.join(pred_dir, os.path.splitext(image_name)[0] + '.txt')
        gt_file = os.path.join(gt_dir, os.path.splitext(image_name)[0] + '.txt')

        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            line = f"跳过 {image_name}，缺少预测或真值文件"
            print(line)
            results_lines.append(line)
            continue

        predictions = parse_annotations(pred_file, is_prediction=True)
        ground_truths = parse_annotations(gt_file, is_prediction=False)

        TP, FP, FN = evaluate_image(predictions, ground_truths, iou_threshold)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        line = (f"Image: {image_name}, TP: {TP}, FP: {FP}, FN: {FN}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        print(line)
        results_lines.append(line)

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results_lines.append("\nOverall Evaluation:")
    results_lines.append(f"Total TP: {total_TP}, Total FP: {total_FP}, Total FN: {total_FN}")
    results_lines.append(f"Overall Precision: {overall_precision:.4f}")
    results_lines.append(f"Overall Recall: {overall_recall:.4f}")
    results_lines.append(f"Overall F1 Score: {overall_f1:.4f}")

    # 输出总体评估结果到控制台
    print("\nOverall Evaluation:")
    print(f"Total TP: {total_TP}, Total FP: {total_FP}, Total FN: {total_FN}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    # 将评估结果写入txt文件，保存在predictions文件夹的同级目录下
    result_save_dir = os.path.dirname(os.path.abspath(pred_dir))
    result_file = os.path.join(result_save_dir, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write('\n'.join(results_lines))
    print(f"\n评估结果已保存到：{result_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate object detection results.')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值, 默认0.5, 不做更改')
    
    parser.add_argument('--image_dir', type=str, default="data\\image", help='图像文件夹路径, 用于获取图像及分辨率')
    parser.add_argument('--gt_dir', type=str, default="data\\label", help='真值结果txt文件文件夹路径')
    parser.add_argument('--pred_dir', type=str, default="output\\exp\\pred_txt", help='预测结果txt文件文件夹路径')
    args = parser.parse_args()
    
    main(args.image_dir, args.pred_dir, args.gt_dir, args.iou_threshold)
