# 简介

车牌检测（retina mnet）与车牌识别（lprnet）的docker部署， 使用基于retinaface的车牌检测和lprnet车牌识别算法实现车牌的检测识别功能，使用CCPD数据集进行迁移学习，在基于25000张广西车牌的训练中，
测试集取得了98.8%的准确率，经回归验证取得95%左右的准确率。模型已经转换onnx，提供对应Dockerfile便于大家生成docker镜像。
该车牌识别系统优势在于模型体积小、准确率高、推理速度快，车牌检测模型和车牌识别模型加起来仅3.4MB， 可在嵌入式平台使用。

# 使用

### 安装依赖

`pip install -r requirements.txt`

### 运行

`python web-onnx.py`

### postman测试

`curl --location --request POST 'ip地址:端口/api/upload_img' \
--form 'obj_flag="false"' \
--form 'file=@"00_guiA40688.jpg"' \
--form 'base64_img="true"'`

* obj_flag:车牌目标检测开关
* file:车牌图片或者车辆图片（需开启车牌检测）
* base64_img:是否返回base64图片流（默认不返回，返回设置为true）

### docker镜像（tar文件）

链接：https://pan.baidu.com/s/1uzD501wa9dPK0Uwl-i3QBA
提取码：zxsi

`docker run -itd -p5001:5001 镜像`

# 致谢

https://github.com/sirius-ai/LPRNet_Pytorch


