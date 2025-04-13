# 自动锚框工具

import numpy as np  # 导入NumPy数值计算库
import torch  # 导入PyTorch深度学习库
import yaml  # 导入YAML解析库
from scipy.cluster.vq import kmeans  # 导入K均值聚类算法
from tqdm import tqdm  # 导入进度条库

# from utils.general import colorstr  # 导入颜色打印函数（已注释）


def check_anchor_order(m):
    """
    检查YOLO检测模块m中锚框的顺序是否与步长顺序一致，并在必要时进行校正
    
    参数:
        m: YOLO的Detect()模块
    """
    a = m.anchor_grid.prod(-1).view(-1)  # 计算锚框面积
    da = a[-1] - a[0]  # 面积差值
    ds = m.stride[-1] - m.stride[0]  # 步长差值
    if da.sign() != ds.sign():  # 如果符号不同（顺序不一致）
        print('Reversing anchor order')  # 打印反转锚框顺序的信息
        m.anchors[:] = m.anchors.flip(0)  # 反转锚框顺序
        m.anchor_grid[:] = m.anchor_grid.flip(0)  # 反转锚框网格顺序


# 以下是被注释掉的代码，但仍添加注释以说明其功能

# def check_anchors(dataset, model, thr=4.0, imgsz=640):
#     """
#     检查锚框与数据的匹配程度，必要时重新计算锚框
#     
#     参数:
#         dataset: 数据集
#         model: 模型
#         thr: 锚框-标签宽高比阈值
#         imgsz: 图像大小
#     """
#     prefix = colorstr('autoanchor: ')  # 添加彩色前缀
#     print(f'\n{prefix}Analyzing anchors... ', end='')
#     m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # 获取检测模块
#     shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算图像形状
#     scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 随机缩放因子
#     wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 计算宽高

#     def metric(k):  # 计算度量标准
#         r = wh[:, None] / k[None]  # 计算比率
#         x = torch.min(r, 1. / r).min(2)[0]  # 比率度量
#         best = x.max(1)[0]  # 最佳比率
#         aat = (x > 1. / thr).float().sum(1).mean()  # 超过阈值的锚框数量
#         bpr = (best > 1. / thr).float().mean()  # 最佳可能召回率
#         return bpr, aat

#     anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # 当前锚框
#     bpr, aat = metric(anchors)
#     print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
#     if bpr < 0.98:  # 如果BPR小于阈值，需要重新计算锚框
#         print('. Attempting to improve anchors, please wait...')
#         na = m.anchor_grid.numel() // 2  # 锚框数量
#         try:
#             anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
#         except Exception as e:
#             print(f'{prefix}ERROR: {e}')
#         new_bpr = metric(anchors)[0]
#         if new_bpr > bpr:  # 如果新锚框更好，则替换
#             anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
#             m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # 更新推理用的锚框网格
#             check_anchor_order(m)
#             m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # 更新损失计算用的锚框
#             print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
#         else:
#             print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
#     print('')  # 换行


# def kmean_anchors(path='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
#     """
#     使用K均值聚类从训练数据集创建进化的锚框
#     
#     参数:
#         path: 数据集YAML路径或已加载的数据集
#         n: 锚框数量
#         img_size: 训练时使用的图像大小
#         thr: 锚框-标签宽高比阈值
#         gen: 使用遗传算法进化的代数
#         verbose: 是否打印详细结果
#     
#     返回:
#         k: K均值进化后的锚框
#     
#     用法:
#         from utils.autoanchor import *; _ = kmean_anchors()
#     """
#     thr = 1. / thr  # 阈值反转
#     prefix = colorstr('autoanchor: ')

#     def metric(k, wh):  # 计算度量标准
#         r = wh[:, None] / k[None]  # 比率
#         x = torch.min(r, 1. / r).min(2)[0]  # 比率度量
#         # x = wh_iou(wh, torch.tensor(k))  # IOU度量
#         return x, x.max(1)[0]  # 返回x和最佳x

#     def anchor_fitness(k):  # 变异适应度计算
#         _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
#         return (best * (best > thr).float()).mean()  # 适应度

#     def print_results(k):  # 打印结果
#         k = k[np.argsort(k.prod(1))]  # 从小到大排序
#         x, best = metric(k, wh0)
#         bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 最佳可能召回率，超过阈值的锚框数量
#         print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
#         print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
#               f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
#         for i, x in enumerate(k):
#             print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # 用于*.cfg文件
#         return k

#     if isinstance(path, str):  # 如果是YAML文件路径
#         with open(path) as f:
#             data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # 加载模型字典
#         from utils.datasets import LoadImagesAndLabels
#         dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
#     else:
#         dataset = path  # 已经是数据集对象

#     # 获取标签宽高
#     shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
#     wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 宽高

#     # 过滤
#     i = (wh0 < 3.0).any(1).sum()
#     if i:
#         print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
#     wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤掉小于2像素的对象
#     # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # 随机缩放0-1

#     # K均值计算
#     print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
#     s = wh.std(0)  # 标准差用于白化
#     k, dist = kmeans(wh / s, n, iter=30)  # 点，平均距离
#     assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
#     k *= s
#     wh = torch.tensor(wh, dtype=torch.float32)  # 已过滤
#     wh0 = torch.tensor(wh0, dtype=torch.float32)  # 未过滤
#     k = print_results(k)

#     # 绘图（注释掉的可视化代码）
#     # k, d = [None] * 20, [None] * 20
#     # for i in tqdm(range(1, 21)):
#     #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
#     # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
#     # ax = ax.ravel()
#     # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
#     # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
#     # ax[0].hist(wh[wh[:, 0]<100, 0],400)
#     # ax[1].hist(wh[wh[:, 1]<100, 1],400)
#     # fig.savefig('wh.png', dpi=200)

#     # 进化
#     npr = np.random
#     f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # 适应度，形状，变异概率，变异幅度
#     pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # 进度条
#     for _ in pbar:
#         v = np.ones(sh)
#         while (v == 1).all():  # 变异直到发生变化（防止重复）
#             v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
#         kg = (k.copy() * v).clip(min=2.0)  # 变异并裁剪
#         fg = anchor_fitness(kg)  # 计算适应度
#         if fg > f:  # 如果变异后更好
#             f, k = fg, kg.copy()  # 接受变异
#             pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
#             if verbose:
#                 print_results(k)

#     return print_results(k)  # 返回并打印最终结果