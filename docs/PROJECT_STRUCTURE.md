# 工程结构设计

## 分层目标

本工程重构后的核心目标是高内聚、低耦合：

- `vendor/ultralytics/` 只负责提供第三方 YOLO 能力。
- `src/yolo_test/` 只负责本项目的业务流程。
- `configs/` 只负责描述数据集和训练参数。
- `artifacts/` 只负责保存运行结果。
- `legacy/` 只负责保存历史材料，避免旧脚本影响新工程。

这样做的直接收益是：升级 Ultralytics、清理训练产物、修改数据处理脚本互不干扰。

## 目录职责

### `src/yolo_test/data`

数据集准备工具。当前包含：

- `voc_xml_to_yolo.py`：Pascal VOC XML 转 YOLO TXT。
- `split_dataset.py`：按图片和标签 stem 匹配后划分 train/val/test。
- `file_maintenance.py`：安全删除空文件、删除不匹配文件、整理 XML 和图片。

这些脚本都通过命令行传入路径，不再依赖固定的个人桌面路径。

### `src/yolo_test/training`

训练入口。当前包含：

- `train_yolo.py`：统一训练命令，把输出固定写到 `artifacts/runs/`。

训练逻辑仍由 Ultralytics 执行，本项目只包装路径、默认参数和输出目录。

### `src/yolo_test/apps`

应用层入口。当前包含：

- `desktop_detector.py`：PySide6 桌面检测程序，支持摄像头、视频、图片和模型切换。

### `src/yolo_test/video`

视频预处理工具。当前包含：

- `frames.py`：视频抽帧，用于制作图像数据集。

## 迁移映射

| 旧位置 | 新位置 |
| --- | --- |
| `yolo/ultralytics-main/ultralytics-main/ultralytics` | `vendor/ultralytics/ultralytics` |
| `yolo/ultralytics-main/ultralytics-main/runs` | `artifacts/runs` |
| `yolo/ultralytics-main/ultralytics-main/yolov8n.pt` | `models/pretrained/yolov8n.pt` |
| `yolo/ultralytics-main/ultralytics-main/yolov8m.pt` | `models/pretrained/yolov8m.pt` |
| `yolo/ultralytics-main/ultralytics-main/yolo11n.pt` | `models/pretrained/yolo11n.pt` |
| `data.yaml` | `configs/datasets/circle_cqu.yaml` |
| `circle.yaml` | `configs/datasets/middle_78456.yaml` |
| 根目录中文/零散 Python 脚本 | `legacy/ad_hoc_scripts/` |
| 旧 `unknown/` | `experiments/non_yolo/unknown/` |

## 后续建议

1. 把真实数据集放到 `datasets/` 或外部数据盘，并在 `configs/datasets/*.yaml` 中使用稳定路径。
2. 对最常用的数据处理命令补单元测试。
3. 把最佳模型复制到 `models/trained/`，并用模型版本号命名，例如 `circle-yolov8n-v1.pt`。
4. 若需要多人协作，把 `artifacts/` 和大型 `.pt` 权重加入 `.gitignore`。
