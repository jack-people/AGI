# AGI

下一代模型按两条线来做：**并行**，以及**从视觉开始的模块化**。感知模块只负责把原始信号翻译成同一张场景图，关系推理和自验证不跟某一个视觉实现绑在一起。

## 项目结构

系统分成四层。上一层只消费下一层的 `StructuredSceneGraph`，不读取像素，也不关心检测器是色块规则还是 OWL-ViT。

```
感知模块  →  场景图（实体 + 关系）  →  四条并行分支  →  RCVS 自验证
```

| 层 | 做什么 | 主要文件 |
| --- | --- | --- |
| 协议 | 规定实体、关系、查询和验证结果的字段 | `New_Model/prototype/protocol.py` |
| 插拔 | 按模态槽位注册模块，视觉可以从 V1 换到 V2、V3 | `New_Model/prototype/registry.py`、`engine.py` |
| 感知 | 把图像或关节角变成实体和框 | `modules/vision.py`、`modules/vision_v3.py`、`modules/base.py`、`modules/arm.py` |
| 推理 | 四条异构分支同时找满足查询的实体，再检查余量是否盖过局部线性误差界 | `modules/parallel_core.py`、`modules/verifier.py` |
| 训练与服务 | 三阶段训练、120 题评测、本地页面 | `trainer.py`、`benchmark.py`、`evaluate_checkpoint.py`、`serve.py`、`modules/continual.py` |

视觉有三版，槽位名都是 `vision`，所以换模块不用改推理器：

- `VisionModuleV1` / `VisionModuleV2`：12 种颜色、4 种形状。V2 多了中值滤波和遮挡补全。上传图片且查询框为空时走 V2。
- `VisionModuleV3`：接收开放词汇检测框。页面填了英文词之后，OWL-ViT 出框，SAM 2 把框描成掩膜，再写回同一张场景图。

`data/` 里是 120 张 64×64 合成图和查询清单，用来训练和做消融。`checkpoints/` 里是三阶段训出来的小权重，只服务预设场景的关系推理。OWL-ViT 和 SAM 2 的权重不在仓库里，第一次按文字查找时下载。

文档和代码分开。`架构设计.md` 写模块怎么接，`训练方案与阶段计划.md` 写四阶段训练和已经测到的准确率，`相关研究综述.md` 对照 2023–2026 的相关工作，两份论文手稿是投稿稿。它们不参与运行。

```
README.md
New_Model/
├── 架构设计.md
├── 训练方案与阶段计划.md
├── 相关研究综述.md
├── 论文_下一代并行模块化模型架构.md
├── paper_pmv_architecture.tex
└── prototype/
    ├── protocol.py
    ├── registry.py
    ├── engine.py
    ├── trainer.py
    ├── benchmark.py
    ├── demo.py
    ├── evaluate_checkpoint.py
    ├── serve.py
    ├── requirements-stage4-optional.txt
    ├── modules/
    │   ├── base.py
    │   ├── vision.py
    │   ├── vision_v3.py
    │   ├── parallel_core.py
    │   ├── verifier.py
    │   ├── arm.py
    │   └── continual.py
    ├── tests/test_architecture.py
    ├── checkpoints/
    └── data/
```

`arm.py` 里有平面两连杆正运动学，机械臂仿真先不继续。`continual.py` 只负责记下「严格分支没答对、其他分支改对」的查询，按周增量训练还没接上。

## 运行

在仓库根目录执行。色块识别、预设推理和单元测试只需要 Python 3 和 NumPy。

```bash
python3 -c "import numpy; print(numpy.__version__)"
python3 -m unittest discover -s New_Model/prototype/tests -v
python3 -m New_Model.prototype.serve --self-check
python3 -m New_Model.prototype.serve --host 0.0.0.0 --port 8765
```

浏览器打开 http://127.0.0.1:8765/ 。

- 查询框留空再选图：标出红、橙、黄、绿、青、蓝、紫、粉、白、灰、棕、黑，以及矩形、圆形、三角形、菱形。同一种颜色分成两块时会分别框出。
- 填写英文词（例如 `cup, book`）后点「按文字查找」：先安装 `New_Model/prototype/requirements-stage4-optional.txt`（含 torchvision），重启服务。第一次会下载 OWL-ViT 和 SAM 2 权重。
- 「看杯子和书」或 http://127.0.0.1:8765/open_demo 不下载大模型。预期摘要里有「杯子在书左侧」，`accepted` 为 true。
- http://127.0.0.1:8765/predict?preset=clean 、`occluded`、`unstable`：前两个 `accepted` 为 true，`unstable` 为 false。PowerShell 里请用 `curl.exe`，或 `Invoke-RestMethod`。

其他命令：

```bash
python3 -m New_Model.prototype.trainer
python3 -m New_Model.prototype.evaluate_checkpoint
python3 -m New_Model.prototype.demo
python3 -m New_Model.prototype.benchmark
```
