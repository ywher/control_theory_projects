# 数据集工具和数据加载器

import glob  # 用于使用通配符查找文件
import logging  # 用于日志记录
import os  # 操作系统接口
from pathlib import Path  # 面向对象的文件系统路径

import cv2  # OpenCV库用于图像处理
import numpy as np  # 科学计算库

# 参数定义
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # 支持的图像格式后缀
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # 支持的视频格式后缀
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    调整图像大小并添加填充，同时满足步长的约束
    
    参数:
        img: 输入图像
        new_shape: 目标尺寸，可以是int或(height, width)
        color: 填充颜色
        auto: 是否自动调整填充为步长的倍数
        scaleFill: 是否拉伸图像填充（不保持原始宽高比）
        scaleup: 是否允许放大图像（如果为False，则只缩小不放大）
        stride: 步长，用于保证填充后尺寸是步长的倍数
    
    返回:
        img: 处理后的图像
        ratio: 缩放比例 (width_ratio, height_ratio)
        (dw, dh): 填充量 (宽度填充, 高度填充)
    """
    shape = img.shape[:2]  # 当前图像形状 [高度, 宽度] 例如 [992, 1202]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # 转换为 [高度, 宽度] 例如 [640, 640]

    # 计算缩放比例 (新尺寸 / 原尺寸)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大（提高测试mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度和高度缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的宽度和高度（未填充）
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽度和高度填充量
    if auto:  # 最小矩形填充，保证填充后尺寸是步长的倍数
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 计算宽度和高度填充量
    elif scaleFill:  # 拉伸填充（不保持原始宽高比）
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度和高度缩放比例

    dw /= 2  # 将填充分为两部分（左右两侧）
    dh /= 2  # 将填充分为两部分（上下两侧）

    if shape[::-1] != new_unpad:  # 如果需要调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 使用线性插值进行调整
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下填充像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右填充像素数
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框填充
    return img, ratio, (dw, dh)


class LoadImages:  # 用于推理的图像加载器
    """
    用于加载图像和视频进行推理的类
    
    可以处理:
    - 单个图像文件
    - 单个视频文件
    - 包含图像的目录
    - 包含视频的目录
    - 带有通配符的路径
    """
    def __init__(self, path, img_size=640, stride=32):
        """
        初始化图像/视频加载器
        
        参数:
            path: 图像/视频文件路径、目录路径或带通配符的路径
            img_size: 处理图像的目标尺寸
            stride: 模型的步长，用于确保图像尺寸是步长的倍数
        """
        p = str(Path(path).absolute())  # 转为绝对路径（与操作系统无关）
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # 使用glob匹配通配符路径
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # 如果是目录，获取目录中的所有文件
        elif os.path.isfile(p):
            files = [p]  # 如果是单个文件，创建只包含该文件的列表
        else:
            raise Exception(f'ERROR: {p} does not exist')  # 如果路径不存在，抛出异常

        # 分离图像和视频文件
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]  # 筛选支持的图像文件
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]  # 筛选支持的视频文件
        ni, nv = len(images), len(videos)  # 图像和视频文件数量

        self.img_size = img_size  # 目标图像大小，例如 640
        self.stride = stride  # 模型步长，例如 32
        self.files = images + videos  # 所有图像和视频文件列表
        self.nf = ni + nv  # 文件总数
        self.video_flag = [False] * ni + [True] * nv  # 标记每个文件是图像还是视频
        self.mode = 'image'  # 默认模式为图像
        if any(videos):
            self.new_video(videos[0])  # 如果有视频，初始化第一个视频
        else:
            self.cap = None  # 没有视频时，视频捕获对象为None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'  # 确保至少有一个文件

    def __iter__(self):
        """
        迭代器方法，使LoadImages对象可迭代
        """
        self.count = 0  # 初始化计数器
        return self

    def __next__(self):
        """
        返回下一个图像/视频帧
        """
        if self.count == self.nf:  # 如果已处理完所有文件
            raise StopIteration  # 停止迭代
        path = self.files[self.count]  # 获取当前文件路径

        if self.video_flag[self.count]:  # 如果是视频
            # 读取视频帧
            self.mode = 'video'
            ret_val, img0 = self.cap.read()  # 读取一帧
            if not ret_val:  # 如果读取失败（视频结束）
                self.count += 1  # 移至下一个文件
                self.cap.release()  # 释放视频捕获对象
                if self.count == self.nf:  # 如果是最后一个视频
                    raise StopIteration  # 停止迭代
                else:
                    path = self.files[self.count]  # 获取下一个文件路径
                    self.new_video(path)  # 初始化新视频
                    ret_val, img0 = self.cap.read()  # 读取新视频的第一帧

            self.frame += 1  # 帧计数器递增
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')  # 打印视频进度

        else:  # 如果是图像
            # 读取图像
            self.count += 1  # 计数器递增
            img0 = cv2.imread(path)  # 使用OpenCV读取图像（BGR格式）
            assert img0 is not None, 'Image Not Found ' + path  # 确保图像加载成功
            #print(f'image {self.count}/{self.nf} {path}: ', end='')  # 打印图像进度（已注释）

        # 进行填充调整大小操作
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # 转换图像格式
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，转换为3×416×416（或其他尺寸）
        img = np.ascontiguousarray(img)  # 确保数组内存连续，提高处理速度

        return path, img, img0, self.cap  # 返回路径，处理后的图像，原始图像，视频捕获对象

    def new_video(self, path):
        """
        初始化新视频
        """
        self.frame = 0  # 重置帧计数器
        self.cap = cv2.VideoCapture(path)  # 创建视频捕获对象
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    def __len__(self):
        """
        返回文件总数
        """
        return self.nf  # 文件总数