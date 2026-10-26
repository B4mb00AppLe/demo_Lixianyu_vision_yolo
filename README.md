\# RoboMaster 装甲板检测（阶段一）



使用 Ultralytics YOLO11n 训练 RoboMaster 装甲板检测模型，并导出 ONNX。



\## 类别



类别编号以 `classes.txt` 的行序为准：



| ID | 类别 |

|---:|---|

| 0 | blue3 |

| 1 | blue1 |

| 2 | bluesb |

| 3 | red3 |

| 4 | red1 |

| 5 | redsb |



\## 环境



\- Windows

\- Python 3.10

\- PyTorch 2.7.1 + CUDA 11.8

\- Ultralytics 8.4.174

\- NVIDIA GeForce RTX 3070 Ti Laptop GPU



\## 数据集



图片和标签未包含在仓库中。准备数据后，目录结构应为：



```text

dataset/

├── data.yaml

├── images/

│   ├── train/

│   ├── val/

│   └── test/

└── labels/

&#x20;   ├── train/

&#x20;   ├── val/

&#x20;   └── test/
