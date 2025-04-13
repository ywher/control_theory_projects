# 导入必要的库
import argparse  # 用于解析命令行参数
import time      # 用于计时功能
from pathlib import Path  # 提供面向对象的文件系统路径处理

import cv2       # OpenCV库，用于图像处理
import torch     # PyTorch深度学习框架
from numpy import random  # 用于生成随机颜色

# 导入自定义模块
from models.yolo import Model  # YOLO模型定义
from utils.datasets import LoadImages  # 数据加载器，处理图像和视频输入
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging, increment_path  # 各种通用工具函数
from utils.plots import plot_one_box  # 用于在图像上绘制检测框
from utils.torch_utils import select_device, time_synchronized  # PyTorch相关工具

# 目标检测主函数
def detect(save_img=False):
    """
    主要的目标检测函数，处理输入的图像或视频并进行检测
    :param save_img: 是否保存带有检测结果的图像
    """
    # 从命令行参数获取配置
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # 判断是否需要保存图像结果

    # 创建保存结果的目录
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 自动创建递增编号的目录
    (save_dir / 'pred_txt' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建文本保存目录

    # 初始化环境
    set_logging()  # 设置日志级别
    device = select_device(opt.device)  # 选择运行设备(CPU/GPU)
    print(f"Device: {device}")

    # 加载模型
    model = Model(opt.cfg, ch=3)  # 创建模型实例，输入为3通道RGB图像
    ckpt = torch.load(weights[0], map_location=device)  # 加载预训练权重
    state_dict = ckpt['model'].float().state_dict()  # 转换为FP32格式
    model.load_state_dict(state_dict, strict=True)  # 加载模型权重
    model.fuse().eval()  # 融合Conv+BN层并设置为评估模式
    
    # 获取模型步长并检查图像尺寸
    stride = int(model.stride.max())  # 获取模型最大步长
    imgsz = check_img_size(imgsz, s=stride)  # 确保图像尺寸是步长的倍数

    # 设置数据加载器
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)  # 加载图像或视频数据

    # 获取类别名称和为每个类别分配随机颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 开始计时
    t0 = time.time()
    # 遍历数据集中的每一项（图像或视频帧）
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)  # 将numpy数组转换为torch张量
        img = img.float()  # 根据设置使用半精度或全精度
        img /= 255.0  # 归一化到0-1范围
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 添加batch维度

        # 执行推理
        t1 = time_synchronized()  # 记录推理开始时间
        with torch.no_grad():   # 不计算梯度，节省内存
            pred = model(img, augment=opt.augment)[0]  # 前向推理
        t2 = time_synchronized()  # 记录推理结束时间

        # 应用非极大值抑制(NMS)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()  # 记录NMS结束时间

        # 处理每张图像的检测结果（可能有多个检测结果）
        for i, det in enumerate(pred):  # 每张图像的检测结果
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 转换为Path对象
            save_path = str(save_dir / p.name)  # 结果图像保存路径
            txt_path = str(save_dir / 'pred_txt' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 结果文本保存路径
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化系数 [w,h,w,h]
            
            if len(det):  # 如果检测到目标
                # 将边界框坐标从模型输入尺寸缩放到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果统计信息
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别检测到的数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到输出字符串

                # 处理每个检测结果
                for *xyxy, conf, cls in reversed(det):
                    # 将结果写入文本文件
                    if save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh坐标
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # 保存格式
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在图像上绘制边界框
                    if save_img or view_img:  
                        label = f'{names[int(cls)]} {conf:.2f}'  # 标签格式：类别名称+置信度
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # 打印处理时间信息
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # 实时显示结果（如果启用）
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1毫秒等待键盘输入

            # 保存结果图像或视频
            if save_img:
                if dataset.mode == 'image':  # 处理单张图像
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}\n")
                else:  # 处理视频
                    if vid_path != save_path:  # 新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频捕获
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流媒体
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)  # 写入视频帧

    # 输出统计信息
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('pred_txt/*.txt')))} pred labels saved to {save_dir / 'pred_txt'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')  # 打印总处理时间

if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg\\yolov7.yaml', help='模型配置文件的路径')
    parser.add_argument('--weights', nargs='+', type=str, default=['pretrain\\yolov7.pt'], help='预训练模型的路径')
    parser.add_argument('--source', type=str, default='data\\image', help='图片或视频的路径')
    parser.add_argument('--project', default='output', help='保存结果的项目路径')
    parser.add_argument('--name', default='exp', help='保存结果的项目名称, 保存在project/name')
    parser.add_argument('--exist-ok', action='store_true', default=False, help='是否覆盖已存在的项目, 默认为False')
    ### 需要调整阈值达到指标
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='非极大抑制的IOU阈值')
    
    parser.add_argument('--img-size', type=int, default=640, help='推理时输入图片缩放的尺寸')
    parser.add_argument('--device', default='cpu', help='使用的设备, 例如: 单卡gpu输入数字0, 多卡gpu输入0,1,2,3, 或 cpu')
    parser.add_argument('--view-img', action='store_true', default=False, help='是否显示推理时的图片, 默认为False')
    parser.add_argument('--save-txt', action='store_false', default=True, help='是否保存推理时的边界框结果到txt文件, 默认为True')
    parser.add_argument('--save-conf', action='store_false', default=True, help='是否保存推理时的置信度结果到txt文件, 默认为True')
    parser.add_argument('--nosave', action='store_true', default=False, help='是否不保存推理时的图片或视频, 默认为False')
    parser.add_argument('--classes', nargs='+', type=int, default=[2, 5, 7], help='需要检测的类别的索引, 从COCO的80个类别中选择')
    parser.add_argument('--agnostic-nms', action='store_true', default=True, help='是否使用类别不敏感的非极大抑制, 默认为True')
    parser.add_argument('--augment', action='store_true', default=False, help='是否使用数据增强, 默认为False')
    
    opt = parser.parse_args()
    # 输出命令行参数
    print(opt)

    # 运行检测
    with torch.no_grad():
        detect()
