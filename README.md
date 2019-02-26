代码实现从图片到smpl参数的预测。数据集基于up数据集以及surreal数据集。数据集为方便使用都经过个人处理，因为该代码并不能用用原始数据集的训练，不具备通用性。

# data_utils
数据预处理
包括
1. 对原版surreal数据集进行预处理
2. 将处理后的数据集转为tfrecord

# tf_smpl
包括numpy版本的smpl运算，tf版本的smpl运算，tf版本的Batch smpl运算

# dataset.py
data io
tfrecord dataset以及 data generator

# model.py
网络结构

# train,py
训练代码

# config.py
配置设置