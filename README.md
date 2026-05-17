# YOLO Test Document

一个面向单类别目标检测实验的 Ultralytics YOLO 工程模板。项目把自有训练流程、数据处理脚本、配置文件、模型权重和训练产物从官方源码中拆开，方便后续复现实验、迁移机器、维护脚本和发布模型。

## 功能概览

- VOC XML 标注转换为 YOLO TXT 标签。
- 按图片和标签文件名划分 `train`、`val`、`test` 数据集。
- 使用统一入口启动 YOLO 检测模型训练。
- 将训练结果集中保存到 `artifacts/runs/`。
- 提供 PySide6 桌面检测程序，支持摄像头、视频、图片和模型切换。
- 保留旧脚本到 `legacy/`，避免历史实验代码影响新工程。

## 目录布局

```text
.
├── src/yolo_test/              # 项目自有代码：训练、数据处理、视频抽帧、桌面检测
├── configs/                    # 项目配置：数据集 YAML、训练参数模板
├── models/pretrained/          # 本地预训练权重目录，.pt 文件不提交到 Git
├── models/trained/             # 本地训练后权重目录，.pt 文件不提交到 Git
├── artifacts/runs/             # 本地训练输出，包含 best.pt、last.pt、曲线图、日志等
├── vendor/ultralytics/         # 官方 Ultralytics 源码，尽量不放项目自定义代码
├── legacy/                     # 旧版一次性脚本、临时文件和旧目录结构
├── experiments/non_yolo/       # 与 YOLO 主工程无关的实验代码
└── docs/                       # 工程结构和迁移说明
```

更详细的工程设计说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 环境准备

建议使用干净的虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果只做数据处理，不运行桌面 GUI，可以暂时不安装 `PySide6`。

## 常用命令

从 Pascal VOC XML 转 YOLO 标签：

```bash
PYTHONPATH=src python -m yolo_test.data.voc_xml_to_yolo \
  --xml-dir /path/to/Annotations \
  --output-dir /path/to/labels \
  --classes circle
```

划分数据集：

```bash
PYTHONPATH=src python -m yolo_test.data.split_dataset \
  --images /path/to/JPEGImages \
  --labels /path/to/labels \
  --output /path/to/VOCdevkit \
  --train 0.8 --val 0.1 --test 0.1
```

训练模型：

```bash
PYTHONPATH=src python -m yolo_test.training.train_yolo \
  --data configs/datasets/circle_cqu.yaml \
  --model models/pretrained/yolov8n.pt \
  --name circle_yolov8n \
  --epochs 200 \
  --device 0
```

训练结果默认写入：

```text
artifacts/runs/detect/<run-name>/
```

启动桌面检测程序：

```bash
PYTHONPATH=src python -m yolo_test.apps.desktop_detector \
  --model models/pretrained/yolov8m.pt
```

视频抽帧：

```bash
PYTHONPATH=src python -m yolo_test.video.frames \
  /path/to/video.mp4 \
  /path/to/output_frames \
  --interval 5
```

## 配置说明

数据集配置位于 `configs/datasets/`：

- `circle_cqu.yaml`：`circle` 单类别检测任务。
- `middle_78456.yaml`：`middle` 单类别检测任务。

迁移到新机器时，优先修改 YAML 中的 `train`、`val`、`test` 路径，不要随意改变 `names` 的顺序。类别顺序会直接决定 YOLO 标签中的 class id 含义。

默认训练参数模板位于 `configs/training/default_train.yaml`。实际训练入口仍然是 `src/yolo_test/training/train_yolo.py`。

## 发布和数据管理

本仓库适合提交源码、配置、文档和轻量实验说明。以下内容默认不提交到 Git：

- 原始数据集：`datasets/`
- 训练输出：`artifacts/runs/`
- 模型权重：`*.pt`、`*.onnx`、`*.engine` 等
- 本地虚拟环境、缓存和 IDE 配置

如果需要发布重要权重，建议使用 GitHub Releases 上传二进制文件，并在 release note 中记录对应的数据集配置、训练参数和评估结果。

## 维护约定

- 不要把新脚本放进 `vendor/ultralytics/`。那里只保留官方源码。
- 不要把训练输出提交到源码区。训练结果统一放 `artifacts/runs/`。
- 不要在脚本里写死 `C:\Users\...` 这类个人路径。路径通过命令行参数或 `configs/` 管理。
- 旧脚本已经移到 `legacy/ad_hoc_scripts/`，后续优先维护 `src/yolo_test/` 下的新脚本。
