import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from models.common import Conv  # 从自定义模块导入Conv类

class Ensemble(nn.ModuleList):
    """
    模型集成类，继承自nn.ModuleList
    用于组合多个模型进行集成推理
    """
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        """
        前向传播函数
        
        参数:
            x: 输入数据
            augment: 是否使用增强推理
        
        返回:
            y: 集成后的预测结果
            None: 训练时的输出占位符
        """
        y = []
        for module in self:
            y.append(module(x, augment)[0])  # 获取每个模型的推理结果
        # y = torch.stack(y).max(0)[0]  # 最大值集成方式（已注释）
        # y = torch.stack(y).mean(0)  # 平均值集成方式（已注释）
        y = torch.cat(y, 1)  # NMS集成方式，沿维度1连接所有模型的输出
        return y, None  # 返回推理结果和占位符


def attempt_load(weights, map_location=None):
    """
    尝试加载模型权重
    支持加载单个模型或模型集成(多个模型)
    
    参数:
        weights: 模型权重路径，可以是单个路径或路径列表
        map_location: 加载模型到指定设备
    
    返回:
        model: 加载好的模型或模型集成
    """
    # 加载模型集成 weights=[a,b,c] 或单个模型 weights=[a] 或 weights=a
    model = Ensemble()  # 创建模型集成实例
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # 加载权重文件
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # 添加FP32格式模型到集成中
    
    # 兼容性更新
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # PyTorch 1.7.0 兼容性设置
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # PyTorch 1.11.0 兼容性设置
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # PyTorch 1.6.0 兼容性设置
    
    if len(model) == 1:
        return model[-1]  # 如果只有一个模型，直接返回该模型
    else:
        print('Ensemble created with %s\n' % weights)  # 打印创建的集成模型信息
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))  # 将最后一个模型的names和stride属性复制到集成模型
        return model  # 返回模型集成