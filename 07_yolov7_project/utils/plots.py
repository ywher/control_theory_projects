# 绘图工具函数

import random  # 导入随机数生成模块，用于生成随机颜色
import cv2  # 导入OpenCV库，用于图像处理和绘图
import matplotlib  # 导入matplotlib库，用于绘制统计图表
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot子模块，提供类似MATLAB的绘图API

# 设置matplotlib参数
matplotlib.rc('font', **{'size': 11})  # 设置默认字体大小为11
matplotlib.use('Agg')  # 使用Agg后端，适用于只需保存图表到文件而不显示的情况

def color_list():
    """
    返回一个RGB颜色列表，用于绘图时的颜色区分
    
    返回:
        颜色列表，每个元素是(r,g,b)元组，值范围0-255
    """
    # 将十六进制颜色转换为RGB元组 - 参考自 https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        """将十六进制颜色代码转换为RGB元组"""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    # 返回matplotlib TABLEAU调色板中的颜色，转换为RGB格式
    # 也可以使用 BASE_(8种), CSS4_(148种), XKCD_(949种) 调色板
    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  

# 绘制单个边界框
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    在图像上绘制一个边界框及可选的标签
    
    参数:
        x: 边界框坐标 [x1, y1, x2, y2]，左上角和右下角坐标
        img: 要绘制的图像
        color: 边界框颜色，默认为随机颜色
        label: 标签文本，如果提供则会绘制在边界框上方
        line_thickness: 线条粗细
        
    返回:
        无返回值，直接修改输入的img
    """
    # 确定线条/字体粗细，自适应图像大小
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    
    # 如果未指定颜色，则随机生成一个RGB颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    # 获取边界框的左上角和右下角坐标点
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # 绘制矩形边界框，使用抗锯齿线条(LINE_AA)提高视觉效果
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # 如果提供了标签，则绘制标签背景和文本
    if label:
        tf = max(tl - 1, 1)  # 文字粗细，略小于边框粗细
        
        # 获取文本大小以确定背景矩形尺寸
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        
        # 计算标签背景矩形的右下角坐标
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        
        # 绘制填充的背景矩形，颜色与边框相同
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # -1表示填充矩形
        
        # 在背景矩形上绘制白色文本
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], 
                    thickness=tf, lineType=cv2.LINE_AA)