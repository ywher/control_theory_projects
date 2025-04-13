import argparse  # 导入命令行参数解析库
import logging   # 导入日志记录库
import sys
from copy import deepcopy  # 导入深拷贝函数，用于复制对象

sys.path.append('./')  # 将当前目录添加到系统路径，以便运行子目录中的Python文件
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器
import torch  # 导入PyTorch深度学习框架
from models.common import *  # 导入自定义的通用模型组件
from models.experimental import *  # 导入实验性模型组件
from utils.autoanchor import check_anchor_order  # 导入锚框顺序检查函数
from utils.general import make_divisible, check_file, set_logging  # 导入通用工具函数
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device, copy_attr  # 导入PyTorch相关工具函数

try:
    import thop  # 尝试导入FLOPS计算库
except ImportError:
    thop = None  # 如果导入失败，设置为None


class Detect(nn.Module):
    """
    检测层类，YOLO模型的最后一层，用于生成目标检测结果
    """
    stride = None  # 在构建过程中计算的步长
    export = False  # onnx导出标志
    end2end = False  # 端到端推理标志
    include_nms = False  # 是否包含NMS
    concat = False  # 是否连接输出

    def __init__(self, nc=80, anchors=(), ch=()):  # 初始化检测层
        super(Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚框的输出数量（类别数+5个边界框参数）
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 每层的锚框数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # 注册锚框缓冲区，形状为(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # 注册锚框网格缓冲区，形状为(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层列表，每层一个卷积

    def forward(self, x):
        """前向传播函数"""
        # x = x.copy()  # 用于性能分析
        z = []  # 推理输出列表
        self.training |= self.export  # 设置训练模式或导出模式
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 应用卷积
            bs, _, ny, nx = x[i].shape  # 获取特征图形状：批次大小、通道数、高度、宽度
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 重塑张量形状为(bs,na,ny,nx,no)

            if not self.training:  # 如果是推理模式
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)  # 创建网格
                y = x[i].sigmoid()  # 应用sigmoid激活
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # 计算xy坐标
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # 计算宽高
                else:
                    # ONNX导出特殊处理
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # 分割输出
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # 新xy坐标
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # 新宽高
                    y = torch.cat((xy, wh, conf), 4)  # 重新组合
                z.append(y.view(bs, -1, self.no))  # 添加到输出列表

        # 根据不同模式返回不同格式的输出
        if self.training:
            out = x  # 训练模式返回原始特征
        elif self.end2end:
            out = torch.cat(z, 1)  # 端到端模式返回连接后的结果
        elif self.include_nms:
            z = self.convert(z)  # 转换格式以包含NMS
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)  # 连接所有输出
        else:
            out = (torch.cat(z, 1), x)  # 返回预测结果和特征图

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """创建网格函数，用于生成特征图上的坐标网格"""
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        """转换输出格式，用于NMS处理"""
        z = torch.cat(z, 1)  # 连接所有检测层的输出
        box = z[:, :, :4]  # 提取边界框坐标
        conf = z[:, :, 4:5]  # 提取置信度
        score = z[:, :, 5:]  # 提取类别得分
        score *= conf  # 计算最终得分
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)  # 坐标转换矩阵
        box @= convert_matrix  # 应用转换矩阵
        return (box, score)  # 返回转换后的边界框和得分


class Model(nn.Module):
    """
    YOLOv7模型类，用于构建和管理整个网络
    """
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # 初始化模型
        super(Model, self).__init__()
        self.traced = False  # 是否已被追踪标志
        if isinstance(cfg, dict):
            self.yaml = cfg  # 如果cfg是字典，直接使用
        else:  # 如果cfg是yaml文件路径
            import yaml  # 导入yaml库
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # 加载模型配置字典

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 设置输入通道数
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 覆盖yaml中的类别数
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖yaml中的锚框设置
        
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 解析模型结构
        self.names = self.yaml.get('names', [str(i) for i in range(self.yaml['nc'])])  # 设置类别名称
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # 构建步长和锚框
        m = self.model[-1]  # 获取检测层(Detect)
        if isinstance(m, Detect):
            s = 256  # 最小步长的2倍
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # 计算步长
            check_anchor_order(m)  # 检查锚框顺序
            m.anchors /= m.stride.view(-1, 1, 1)  # 根据步长缩放锚框
            self.stride = m.stride
            self._initialize_biases()  # 初始化偏置（只运行一次）
            # print('Strides: %s' % m.stride.tolist())

        # 初始化权重和偏置
        initialize_weights(self)  # 初始化模型权重
        self.info()  # 打印模型信息
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        """
        前向传播函数，支持普通推理、数据增强和性能分析
        """
        if augment:
            # 数据增强推理
            img_size = x.shape[-2:]  # 获取高度和宽度
            s = [1, 0.83, 0.67]  # 缩放比例
            f = [None, 3, None]  # 翻转类型（2-上下翻转，3-左右翻转）
            y = []  # 输出列表
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 缩放和翻转
                yi = self.forward_once(xi)[0]  # 前向推理
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # 保存图像
                yi[..., :4] /= si  # 恢复缩放前的边界框尺寸
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # 恢复上下翻转
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # 恢复左右翻转
                y.append(yi)
            return torch.cat(y, 1), None  # 返回增强后的推理结果
        else:
            return self.forward_once(x, profile)  # 单尺度推理或训练

    def forward_once(self, x, profile=False):
        """
        单次前向传播，可选性能分析
        """
        y, dt = [], []  # 输出和时间列表
        for m in self.model:
            if m.f != -1:  # 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从早期层获取输入

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, Detect):  # 如果是检测层且模型已被追踪
                    break

            if profile:
                # 性能分析
                c = isinstance(m, (Detect))  # 是否为检测类型层
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # 计算FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # 运行模块
            
            y.append(x if m.i in self.save else None)  # 保存输出

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        """
        初始化Detect()层的偏置
        cf是类别频率（可选）
        """
        # https://arxiv.org/abs/1708.02002 第3.3节
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect()模块
        for mi, s in zip(m.m, m.stride):  # 遍历每个检测层
            b = mi.bias.view(m.na, -1)  # 将卷积偏置从(255)转为(3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 物体置信度偏置（每640像素图像8个物体）
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):
        """
        初始化辅助检测层的偏置
        """
        # https://arxiv.org/abs/1708.02002 第3.3节
        m = self.model[-1]  # Detect()模块
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # 遍历主检测层和辅助检测层
            b = mi.bias.view(m.na, -1)  # 主检测层
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 物体置信度偏置
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # 辅助检测层
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 物体置信度偏置
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):
        """
        初始化二进制分类检测层的偏置
        """
        m = self.model[-1]  # Bin()模块
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)  # 卷积偏置
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # 物体置信度偏置
            b[:, (obj_idx+1):].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置
            b[:, (0,1,2,bc+3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):
        """
        初始化关键点检测层的偏置
        """
        m = self.model[-1]  # Detect()模块
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 物体置信度偏置
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        """
        打印检测层的偏置值
        """
        m = self.model[-1]  # Detect()模块
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T  # 转置卷积偏置
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):
        """
        融合模型中的Conv2d和BatchNorm2d层以提高推理速度
        """
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()  # 融合RepVGG块
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()  # 切换到部署模式
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 融合卷积和BN
                delattr(m, 'bn')  # 移除BN
                m.forward = m.fuseforward  # 更新前向传播函数
        self.info()  # 打印融合后模型信息
        return self

    def nms(self, mode=True):
        """
        添加或移除NMS模块
        """
        present = type(self.model[-1]) is NMS  # 最后一层是否为NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # 创建NMS模块
            m.f = -1  # 设置输入来源
            m.i = self.model[-1].i + 1  # 设置索引
            self.model.add_module(name='%s' % m.i, module=m)  # 添加模块
            self.eval()  # 设置为评估模式
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # 移除NMS模块
        return self

    def autoshape(self):
        """
        添加自动形状调整模块
        """
        print('Adding autoShape... ')
        m = autoShape(self)  # 包装模型
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # 复制属性
        return m

    def info(self, verbose=False, img_size=640):
        """
        打印模型信息
        """
        model_info(self, verbose, img_size)


def parse_model(d, ch):
    """
    解析模型配置字典，构建模型结构
    
    参数:
        d: 模型配置字典
        ch: 输入通道数列表
    
    返回:
        nn.Sequential: 模型序列
        list: 保存层输出的索引列表
    """
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']  # 获取模型配置参数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚框数量
    no = na * (nc + 5)  # 输出数量 = 锚框数 * (类别数 + 5)

    layers, save, c2 = [], [], ch[-1]  # 层列表，保存列表，输出通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 遍历主干和头部的每一层配置
        m = eval(m) if isinstance(m, str) else m  # 如果m是字符串，则执行eval
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # 如果参数是字符串，则执行eval
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 根据深度缩放因子调整层数
        if m in [nn.Conv2d, Conv, DWConv, RepConv, RepConv_OREPA, SPPCSPC]:
            c1, c2 = ch[f], args[0]  # 获取输入和输出通道数
            if c2 != no:  # 如果不是输出层
                c2 = make_divisible(c2 * gw, 8)  # 根据宽度缩放因子调整通道数，确保是8的倍数

            args = [c1, c2, *args[1:]]  # 更新参数列表
            if m in [SPPCSPC]:
                args.insert(2, n)  # 插入重复次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # BatchNorm2d的参数是输入通道数
        elif m is Concat:
            c2 = sum([ch[x] for x in f])  # 计算拼接后的通道数
        elif m in [Detect]:  # 检测层特殊处理
            args.append([ch[x] for x in f])  # 添加输入通道数列表
            if isinstance(args[1], int):  # 如果锚框参数是整数
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]  # 使用输入层的通道数

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 创建模块，如果n>1则重复
        t = str(m)[8:-2].replace('__main__.', '')  # 获取模块类型名称
        np = sum([x.numel() for x in m_.parameters()])  # 计算参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 添加索引、来源索引、类型、参数数量属性
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # 打印层信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)  # 添加到层列表
        if i == 0:
            ch = []  # 如果是第一层，清空通道列表
        ch.append(c2)  # 添加输出通道数到列表
    return nn.Sequential(*layers), sorted(save)  # 返回模型序列和排序后的保存列表


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='模型配置文件')
    parser.add_argument('--device', default='', help='计算设备，例如：0或0,1,2,3或cpu')
    parser.add_argument('--profile', action='store_true', help='是否分析模型速度')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # 检查文件
    set_logging()  # 设置日志
    device = select_device(opt.device)  # 选择设备

    # 创建模型
    model = Model(opt.cfg).to(device)  # 创建模型并移动到指定设备
    model.train()  # 设置为训练模式
    
    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)  # 创建随机输入
        y = model(img, profile=True)  # 运行分析

    # 性能分析
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard可视化（已注释）
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # 添加模型到tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # 添加测试图像到tensorboard