# YOLOR通用工具函数

import glob  # 用于文件路径匹配
import logging  # 日志记录
import math  # 数学函数
import os  # 操作系统功能
import re  # 正则表达式
import time  # 时间相关函数
from pathlib import Path  # 面向对象的文件系统路径

import cv2  # OpenCV库
import numpy as np  # NumPy数值计算库
import pandas as pd  # 数据分析库
import torch  # PyTorch深度学习库
import torchvision  # PyTorch视觉库

# 全局设置
torch.set_printoptions(linewidth=320, precision=5, profile='long')  # 设置PyTorch打印选项
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # 设置NumPy打印格式
pd.options.display.max_columns = 10  # 设置Pandas显示的最大列数
cv2.setNumThreads(0)  # 防止OpenCV多线程（与PyTorch DataLoader不兼容）
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # 设置NumExpr最大线程数

# 设置日志级别
def set_logging(rank=-1):
    """
    设置日志记录级别
    
    参数:
        rank: 进程等级，-1或0表示主进程
    """
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

# 检查文件是否存在
def check_file(file):
    """
    检查文件是否存在，如未找到则搜索
    
    参数:
        file: 文件路径
        
    返回:
        文件的真实路径
    """
    # 如果文件存在或为空字符串，直接返回
    if Path(file).is_file() or file == '':
        return file
    else:
        # 递归搜索文件
        files = glob.glob('./**/' + file, recursive=True)  # 查找文件
        assert len(files), f'文件未找到: {file}'  # 断言文件被找到
        assert len(files) == 1, f"多个文件匹配'{file}'，请指定确切路径: {files}"  # 断言唯一匹配
        return files[0]  # 返回文件路径

# 检查图像尺寸是否为步长的倍数
def check_img_size(img_size, s=32):
    """
    验证图像尺寸是否为步长s的倍数
    
    参数:
        img_size: 图像尺寸
        s: 步长，默认为32
        
    返回:
        调整后的图像尺寸
    """
    new_size = make_divisible(img_size, int(s))  # 计算步长倍数
    if new_size != img_size:
        print(f'警告: --img-size {img_size}必须是最大步长{s}的倍数，更新为{new_size}')
    return new_size

# 将数值调整为可被除数整除的值
def make_divisible(x, divisor):
    """
    返回能被除数整除的值
    
    参数:
        x: 原始值
        divisor: 除数
        
    返回:
        调整后的值
    """
    return math.ceil(x / divisor) * divisor

# 将边界框坐标从[x1,y1,x2,y2]转换为[x,y,w,h]
def xyxy2xywh(x):
    """
    将nx4的边界框从[x1,y1,x2,y2]转换为[x,y,w,h]
    其中xy1=左上角，xy2=右下角
    
    参数:
        x: 边界框坐标张量或数组
        
    返回:
        转换后的坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x中心点
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y中心点
    y[:, 2] = x[:, 2] - x[:, 0]  # 宽度
    y[:, 3] = x[:, 3] - x[:, 1]  # 高度
    return y


# 将边界框坐标从[x,y,w,h]转换为[x1,y1,x2,y2]
def xywh2xyxy(x):
    """
    将nx4的边界框从[x,y,w,h]转换为[x1,y1,x2,y2]
    其中xy1=左上角，xy2=右下角
    
    参数:
        x: 边界框坐标张量或数组
        
    返回:
        转换后的坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 左上角x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 左上角y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 右下角x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 右下角y
    return y

# 缩放坐标
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将坐标(xyxy)从img1_shape缩放到img0_shape
    
    参数:
        img1_shape: 模型输入尺寸
        coords: 预测的坐标
        img0_shape: 原始图像尺寸
        ratio_pad: 可选的缩放比例和填充值
        
    返回:
        缩放后的坐标
    """
    if ratio_pad is None:  # 根据img0_shape计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 增益 = 旧/新
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x方向填充，x1和x2
    coords[:, [1, 3]] -= pad[1]  # y方向填充，y1和y2
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

# 限制坐标在图像范围内
def clip_coords(boxes, img_shape):
    """
    将xyxy边界框限制在图像形状(高度, 宽度)内
    
    参数:
        boxes: 边界框坐标
        img_shape: 图像形状[高度, 宽度]
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

# 非极大值抑制
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """
    对推理结果执行非极大值抑制(NMS)
    
    参数:
        prediction: 模型预测结果
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        classes: 过滤指定的类别
        agnostic: 是否进行类别无关的NMS
        multi_label: 是否允许每个框有多个标签
        labels: 可选的已知标签
        
    返回:
        检测结果列表，每张图像一个(n,6)张量 [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    # 设置参数
    min_wh, max_wh = 2, 4096  # （像素）边界框的最小和最大宽度和高度
    max_det = 300  # 每张图像的最大检测数量
    max_nms = 30000  # torchvision.ops.nms()的最大框数
    time_limit = 10.0  # 超时秒数
    redundant = True  # 是否要求检测冗余
    multi_label &= nc > 1  # 每个框多个标签（每张图像增加0.5ms）
    merge = False  # 使用合并NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 图像索引，图像推理
        # 应用约束
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # 宽度-高度
        x = x[xc[xi]]  # 置信度

        # 如果自动标注，加入先验标签
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # 框
            v[:, 4] = 1.0  # 置信度
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 类别
            x = torch.cat((x, v), 0)

        # 如果没有框，处理下一张图像
        if not x.shape[0]:
            continue

        # 计算置信度
        if nc == 1:
            x[:, 5:] = x[:, 4:5]  # 对于只有一个类别的模型，类别损失为0，类别置信度始终为0.5，无需乘法
        else:
            x[:, 5:] *= x[:, 4:5]  # 置信度 = 物体置信度 * 类别置信度

        # 将框从(中心x, 中心y, 宽度, 高度)转换为(x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 检测矩阵nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 只取最佳类别
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 应用有限约束
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 检查形状
        n = x.shape[0]  # 框的数量
        if not n:  # 没有框
            continue
        elif n > max_nms:  # 框太多
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序

        # 批量NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框（按类别偏移）和分数
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # 限制检测数量
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # 合并NMS（使用加权平均合并框）
            # 更新框为boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou矩阵
            weights = iou * scores[None]  # 框权重
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # 合并后的框
            if redundant:
                i = i[iou.sum(1) > 1]  # 要求冗余

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'警告：NMS时间限制{time_limit}秒已超出')
            break  # 超时

    return output

# 去除优化器
def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    """
    从'f'中去除优化器以完成训练，可选保存为's'
    
    参数:
        f: 输入模型文件
        s: 可选的输出模型文件
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # 用EMA模型替换模型
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # 需要删除的键
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # 转为FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # 文件大小
    print(f"优化器已从{f}中去除，{('保存为%s，' % s) if s else ''} {mb:.1f}MB")

# 应用二次分类器
def apply_classifier(x, model, img, im0):
    """
    将二次分类器应用于YOLO输出
    
    参数:
        x: YOLO检测结果
        model: 分类器模型
        img: 处理后的图像
        im0: 原始图像
        
    返回:
        过滤后的检测结果
    """
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # 每张图像
        if d is not None and len(d):
            d = d.clone()

            # 重塑和填充裁剪区域
            b = xyxy2xywh(d[:, :4])  # 边界框
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 矩形变正方形
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # 填充
            d[:, :4] = xywh2xyxy(b).long()

            # 将框从img_size缩放到im0大小
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # 类别
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # 每个检测项
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，变为3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8转float32
                im /= 255.0  # 0-255转0.0-1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # 分类器预测
            x[i] = x[i][pred_cls1 == pred_cls2]  # 保留匹配的类别检测

    return x

# 增量路径
def increment_path(path, exist_ok=True, sep=''):
    """
    增量路径，例如 runs/exp --> runs/exp{sep}0, runs/exp{sep}1等
    
    参数:
        path: 基础路径
        exist_ok: 如果路径存在是否可以使用
        sep: 分隔符
        
    返回:
        增量后的路径
    """
    path = Path(path)  # 操作系统无关
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # 相似路径
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # 索引
        n = max(i) + 1 if i else 2  # 增量数
        return f"{path}{sep}{n}"  # 更新路径