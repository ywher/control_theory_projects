# SJTU自动化系大二课程控制导论感知定位随堂实验



1. 实验环境安装
   1. 第六周实验前参考06_opencv_project/环境安装说明文档.pdf完成Anaconda, Spyder, OpenCV的安装
   2. 第七周实验前参考07_yolov7_project/感知实验2环境安装说明文档.pdf完成PyTorch、依赖库安装
2. 第六周实验
   1. 阅读示例程序opencv_example.py
   2. 实现水平翻转、高斯模糊、Canny边缘检测三个操作（查询OpenCV官方文档: https://docs.opencv.org/4.11.0/ 实现）
   3. 水平拼接实验结果图，并在图的左上方标注组号
3. 第七周实验
   1. 阅读示例程序detect.py，了解目标检测流程和参数意义
   2. 阅读示例程序eval.py，了解目标检测结果的评估流程
   3. 调整置信度阈值conf-thres、非极大抑制IOU阈值iou-thres，使得评估指标Precision，Recall，F1-Score都达到0.9
4. 第八周实验
   1. 阅读示例程序main.py、lane_detector.py等
   2. 如何使用OpenCV实现车道中心线检测、图像逆透视变换以及车辆与车道中心线横向定位的完整流程
   3. 调整实验参数或流程方法，使平均横向定位精度小于0.2米
