# TARNET 的实现

自己动手实现一个基于 transformer 的 tarnet 模型，我们只需要使用分类功能，故在源码的基础上进行修改，以便于我们的实验。

# TARNet : Task-Aware Reconstruction for Time-Series Transformer

This is the official PyTorch implementation of the KDD 2022 paper [TARNet : Task-Aware Reconstruction for Time-Series Transformer](https://dl.acm.org/doi/10.1145/3534678.3539329).

![alt text](https://github.com/ranakroychowdhury/TARNet/blob/main/Slide1.jpg)

### 项目源地址

[TARNET](https://github.com/ranakroychowdhury/TARNet.git)

### 项目目录

`transformer.py` transformer 编码器模块，先构建单个 TransformerEncoderLayer 模块，然后堆叠 N 个编码器模块组成 TransformerEncoder 模块
`multitask_transformer_class.py` transformer 分类器的核心模块，包括位置编码模块**PositionalEncoding**，与分类模块**MultitaskTransformerModel**
`utils.py` 项目的功能模块，负责数据的处理，模型的构建，模型的训练，模型的测试
`script.py` 启动文件，用于与用户的交互

### TARNET 模型的核心

TARNET 是基于 transformer 的模型，支持序列的重构、时间序列的分类与回归三种功能。其中序列的重构是模型的核心创新。模型在不同的任务中都会对序列进行重构，通过重构误差与任务误差来训练了模型的模型参数，以此来达到对序列数据的建模型。
