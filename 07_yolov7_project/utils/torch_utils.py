# YOLOR PyTorch工具函数

import logging  # 导入日志记录模块
import math  # 导入数学函数模块
import os  # 导入操作系统功能模块
import platform  # 导入平台识别模块
import time  # 导入时间处理模块
import datetime  # 导入日期时间模块
import subprocess  # 导入子进程管理模块
from copy import deepcopy  # 导入深拷贝函数
from pathlib import Path  # 导入路径处理模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
import torchvision  # 导入PyTorch视觉库

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

def date_modified(path=__file__):
    """
    返回可读的文件修改日期，例如'2021-3-26'
    
    参数:
        path: 要检查的文件路径，默认为当前文件
        
    返回:
        格式化的日期字符串
    """
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path必须是一个目录
    """
    返回可读的Git描述，例如v5.0-5-g3e25f1e
    
    参数:
        path: Git仓库目录路径
        
    返回:
        Git描述字符串或空字符串(如果不是Git仓库)
    """
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # 不是Git仓库

# 选择运行设备
def select_device(device='', batch_size=None):
    """
    选择运行设备(CPU/GPU)
    
    参数:
        device: 设备标识符，'cpu'表示CPU，'0'表示第一个GPU，'0,1,2,3'表示多个GPU
        batch_size: 批处理大小，用于检查是否与GPU数量兼容
        
    返回:
        torch.device对象
    """
    s = f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # 信息字符串
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制torch.cuda.is_available()返回False
    elif device:  # 如果请求了非CPU设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置环境变量
        assert torch.cuda.is_available(), f'CUDA不可用，无效的设备{device}'  # 检查可用性

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # 检查batch_size是否与GPU数量兼容
            assert batch_size % n == 0, f'批处理大小{batch_size}不是GPU数量{n}的倍数'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # 字节转MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji安全输出
    return torch.device('cuda:0' if cuda else 'cpu')

# 获取同步的时间
def time_synchronized():
    """
    获取PyTorch准确的时间(如果可用，会同步CUDA)
    
    返回:
        当前时间戳
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# 初始化模型权重
def initialize_weights(model):
    """
    初始化模型的权重
    
    参数:
        model: 要初始化的模型
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3  # 设置BatchNorm2d的epsilon值
            m.momentum = 0.03  # 设置BatchNorm2d的动量值
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # 设置激活函数为就地操作模式

# 融合卷积和批归一化层
def fuse_conv_and_bn(conv, bn):
    """
    融合卷积和批归一化层以提高推理速度
    参考: https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    
    参数:
        conv: 卷积层
        bn: 批归一化层
        
    返回:
        融合后的卷积层
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备滤波器
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备空间偏置
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

# 打印模型信息
def model_info(model, verbose=False, img_size=640):
    """
    打印模型信息。img_size可以是整数或列表，例如：img_size=640或img_size=[640, 320]
    
    参数:
        model: 要分析的模型
        verbose: 是否打印详细信息
        img_size: 输入图像尺寸
    """
    n_p = sum(x.numel() for x in model.parameters())  # 参数总数
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 需要梯度的参数总数
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('层', '名称', '梯度', '参数', '形状', '均值', '标准差'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # 计算FLOPs(浮点运算数)
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # 输入
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 如果是int/float则展开
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    logger.info(f"模型摘要: {len(list(model.modules()))}层, {n_p}个参数, {n_g}个梯度{fs}")

# 加载分类器
def load_classifier(name='resnet101', n=2):
    """
    加载预训练模型并重塑为n类输出
    
    参数:
        name: 模型名称，例如'resnet101'
        n: 输出类别数
        
    返回:
        预训练的分类模型
    """
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet模型属性
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # 重塑输出为n个类别
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model

# 缩放图像
def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """
    按比例缩放图像，受gs-multiple约束
    
    参数:
        img: 输入图像，形状为(bs,3,y,x)
        ratio: 缩放比例
        same_shape: 是否保持相同形状
        gs: 网格大小
        
    返回:
        缩放后的图像
    """
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # 新大小
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # 调整大小
        if not same_shape:  # 填充/裁剪图像
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet均值

# 复制属性
def copy_attr(a, b, include=(), exclude=()):
    """
    从b复制属性到a，可以选择只包含[...]并排除[...]
    
    参数:
        a: 目标对象
        b: 源对象
        include: 要包含的属性列表
        exclude: 要排除的属性列表
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

# 批归一化基类
class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """
    通用维度批归一化基类
    """
    def _check_input_dim(self, input):
        """
        重写输入维度检查方法
        
        BatchNorm1d、BatchNorm2d、BatchNorm3d等之间唯一的区别是这个被子类重写的方法。
        该方法的原始目标是进行张量完整性检查。
        如果你可以绕过这些完整性检查(例如，如果你相信你的推理会提供正确维度的输入)，
        那么你可以直接使用这个方法来轻松地从SyncBatchNorm转换
        (不幸的是，SyncBatchNorm不存储原始类 - 如果它这样做了，我们可以返回最初创建的类)
        """
        return

# 转换同步批归一化
def revert_sync_batchnorm(module):
    """
    将SyncBatchNorm转换为BatchNormXd
    
    这与它尝试恢复的函数非常相似:
    https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    
    参数:
        module: 要转换的模块
        
    返回:
        转换后的模块
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

# JIT跟踪模型
class TracedModel(nn.Module):
    """
    使用TorchScript将模型转换为追踪模型，用于提高推理性能
    """
    def __init__(self, model=None, device=None, img_size=(640,640)): 
        """
        初始化追踪模型
        
        参数:
            model: 原始模型
            device: 运行设备
            img_size: 输入图像尺寸
        """
        super(TracedModel, self).__init__()
        
        print(" 将模型转换为追踪模型... ") 
        self.stride = model.stride  # 模型步长 [8, 16, 32] for yolor
        self.names = model.names  # 类别名称
        self.model = model

        self.model = revert_sync_batchnorm(self.model)  # 将同步批归一化转换为标准批归一化
        self.model.to('cpu')  # 移动到CPU
        self.model.eval()  # 设为评估模式

        self.detect_layer = self.model.model[-1]  # 获取检测层
        self.model.traced = True  # 标记为已追踪
        
        rand_example = torch.rand(1, 3, img_size, img_size)  # 创建随机输入
        
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)  # 追踪模型
        #traced_script_module = torch.jit.script(self.model)
        traced_script_module.save("traced_model.pt")  # 保存追踪模型
        print(" 追踪模型已保存! ")
        self.model = traced_script_module  # 更新模型为追踪后的版本
        self.model.to(device)  # 移动到指定设备
        self.detect_layer.to(device)  # 移动检测层到指定设备
        print(" 模型追踪完成! \n") 

    def forward(self, x, augment=False, profile=False):
        """
        前向推理函数
        
        参数:
            x: 输入张量
            augment: 是否使用增强(未使用)
            profile: 是否进行性能分析(未使用)
            
        返回:
            模型输出
        """
        out = self.model(x)  # 运行追踪模型
        out = self.detect_layer(out)  # 应用检测层
        return out