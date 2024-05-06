# SZU-Graduation_Project
本仓库包含本人在SZU完成的毕业设计《基于静息态与任务态脑活动的双相情感障碍及其家族风险的辅助诊断方法研究》的主要代码。

## 项目描述

本研究深入研究了在任务态和静息态下，正常人、双相情感障碍疾病患者及其一级亲属的脑电信号分类问题。通过运用脑网络分析的特征提取技术及图卷积模块的空间信息提取技术，本研究不仅加深了我们对双相情感障碍疾病患者及其家族神经机制的理解，同时也为双相情感障碍疾病的精确诊断与治疗提供了新思路。

## 安装

你可以使用以下命令来安装依赖：

```bash
pip install -r requirements.txt
```

然后，克隆这个仓库到你的本地机器：

```bash
git clone https://github.com/szu2020222001/SZU-Graduation_Project.git
```

## 文件结构

``` markdown
SZU-Graduation_Project/
│
├── CalFeat/
│   ├── clustering_coef_wu.m 计算聚类系数
│   ├── distance_wei.m 计算最短路径长度
│   ├── efficiency_wei.m 计算局部效率
│   ├── eigenvector_centrality_und.m 计算特征向量中心性
│   └── main.m 计算特征的主函数
│
├── DataPreprocess/
│   ├── DataProcessing.py 任务态EEG分窗切片
│   └── RenameMatfile.py 文件重命名和信息记录
│
├── Model/
│   ├── BFA-GCN.py  该文件定义模型
│   ├── layers.py 该文件定义图卷积相关函数
│   └── utils.py  该文件存放工具函数
│
├── T test/
│   ├── fdr.m 该文件定义False Discovery Rate函数 
│   └── T_test.m T检验的主函数
│
├── Paper/ 该文件夹用于存放论文
├── README.md
└── requirements.txt
```

## 使用方法

在`Model`文件夹中，提供了基于脑网络启发与融合注意力机制的图卷积网络模型（Brain Network-inspired and Fusion Attention-based Graph Convolutional Network，简称BFA-GCN）的完整代码。你可以通过调用此模型，并根据您的需求自定义训练与验证策略，从而灵活地实现模型的训练与评估。

## 许可证

这个项目是在MIT许可证下发布的。详情请参阅LICENSE文件。

## 贡献

欢迎提交pull request来改进这个项目。如果你发现了bug或有新的特性请求，请提交issue。

## 联系方式

你可以通过邮箱zhangyulin20020308@outlook.com联系我。

## 致谢

感谢SZU和UESTC所有对我的毕业设计提供帮助的老师、师兄师姐们。
