# 损失函数

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # 返回平滑后的正负标签值，用于BCE标签平滑
    return 1.0 - 0.5 * eps, 0.5 * eps  # 正标签和负标签的平滑值

class SigmoidBin(nn.Module):
    """
    Sigmoid Bin模块：结合了分类和回归的预测方法
    用于将连续值预测问题分解为粗粒度分类和细粒度回归
    """
    stride = None  # 步长，在构建过程中计算
    export = False  # onnx导出标志

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        """
        初始化SigmoidBin模块
        
        参数:
            bin_count: bin的数量，用于将连续值离散化
            min: 预测值的最小值
            max: 预测值的最大值
            reg_scale: 回归缩放因子，控制回归修正的影响程度
            use_loss_regression: 是否在损失计算中使用回归损失
            use_fw_regression: 是否在前向传播中使用回归预测
            BCE_weight: 二元交叉熵的正样本权重
            smooth_eps: 标签平滑参数
        """
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count  # bin的数量
        self.length = bin_count + 1  # 输出向量长度(回归值+bin分类)
        self.min = min  # 最小值
        self.max = max  # 最大值
        self.scale = float(max - min)  # 取值范围
        self.shift = self.scale / 2.0  # 中心偏移量

        self.use_loss_regression = use_loss_regression  # 是否使用回归损失
        self.use_fw_regression = use_fw_regression  # 是否使用回归预测
        self.reg_scale = reg_scale  # 回归缩放因子
        self.BCE_weight = BCE_weight  # BCE权重

        # 计算bins的中心值
        start = min + (self.scale/2.0) / self.bin_count  # 第一个bin的中心
        end = max - (self.scale/2.0) / self.bin_count  # 最后一个bin的中心
        step = self.scale / self.bin_count  # bin宽度
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        # 创建bins张量并注册为缓冲区(不会被视为模型参数)
        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               
        # 标签平滑参数
        self.cp = 1.0 - 0.5 * smooth_eps  # 正标签平滑值
        self.cn = 0.5 * smooth_eps  # 负标签平滑值

        # 损失函数
        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))  # bin分类的BCE损失
        self.MSELoss = nn.MSELoss()  # 回归的MSE损失

    def get_length(self):
        """
        返回预测向量的长度
        """
        return self.length

    def forward(self, pred):
        """
        前向传播函数，用于推理阶段
        
        参数:
            pred: 预测向量，包含回归值和bin分类概率
            
        返回:
            result: 最终预测值
        """
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        # 分离回归预测和bin分类预测
        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step  # 回归预测值
        pred_bin = pred[..., 1:(1+self.bin_count)]  # bin分类预测值

        # 找到最可能的bin
        _, bin_idx = torch.max(pred_bin, dim=-1)  # 获取最高概率的bin索引
        bin_bias = self.bins[bin_idx]  # 获取对应bin的中心值

        # 结合回归修正计算最终预测
        if self.use_fw_regression:
            result = pred_reg + bin_bias  # 结合回归偏移和bin中心
        else:
            result = bin_bias  # 仅使用bin中心
        result = result.clamp(min=self.min, max=self.max)  # 限制在有效范围内

        return result

    def training_loss(self, pred, target):
        """
        训练时的损失计算
        
        参数:
            pred: 预测向量，包含回归值和bin分类概率
            target: 目标值
            
        返回:
            loss: 总损失值
            out_result: 预测结果
        """
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device  # 获取计算设备

        # 分离并处理回归预测
        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step  # 回归预测值
        pred_bin = pred[..., 1:(1+self.bin_count)]  # bin分类预测值

        # 找到与目标最接近的bin
        diff_bin_target = torch.abs(target[..., None] - self.bins)  # 目标与各bin的差异
        _, bin_idx = torch.min(diff_bin_target, dim=-1)  # 获取最接近的bin索引
    
        # 获取bin中心值并计算预测结果
        bin_bias = self.bins[bin_idx]  # 对应bin的中心值
        bin_bias.requires_grad = False  # 冻结梯度
        result = pred_reg + bin_bias  # 计算预测结果

        # 创建bin分类目标
        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # 初始化为负标签值
        n = pred.shape[0]  # 批次大小
        target_bins[range(n), bin_idx] = self.cp  # 设置正确bin的标签为正值

        # 计算bin分类损失
        loss_bin = self.BCEbins(pred_bin, target_bins)  # BCE损失

        # 计算总损失
        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # 回归MSE损失        
            loss = loss_bin + loss_regression  # 总损失
        else:
            loss = loss_bin  # 仅使用bin分类损失

        # 限制结果在有效范围内
        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result  # 返回损失和预测结果