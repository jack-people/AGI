# AGI

下一代模型按两条线来做：**并行**，以及**从视觉开始的模块化**。感知模块只负责把原始信号翻译成同一张场景图，关系推理和自验证不跟某一个视觉实现绑在一起。

## 模块化在哪里

模块之间只认一种接口。`PerceptionModule.encode_to_scene_graph`（`New_Model/prototype/modules/base.py`）把本模态的原始输入写成 `EntityNode` 和 `RelationEdge`。`ParallelRelationalCore` 和 `RCVSSelfVerifier` 只读这张图，不读像素，也不知道框是色块规则算的还是 OWL-ViT 给的。

插拔发生在模态槽位上。`ModuleRegistry`（`registry.py`）按 `vision`、`proprioception` 这类槽位保存当前模块。`NextGenModelEngine.upgrade_module` 替换槽位里的实现，下游推理代码不动。三版视觉都挂在 `vision` 槽：

- `VisionModuleV1`：颜色、边缘、空间三路直接在原图像素上取实体。
- `VisionModuleV2`：同一输出协议上增加 3×3 中值滤波和遮挡补全。遮挡场景因此能从 V1 的失败变成通过，测试在 `test_modular_hot_upgrade_v1_to_v2_on_occlusion_and_noise`。
- `VisionModuleV3`：检测器插件交出类别和框，SAM 2 再补掩膜，仍然写成同一张场景图。`test_stage4_open_vocab_hotswap_keeps_reasoner` 检查从 V2 热升级到 V3 之后，原来的「杯子在书左侧」查询仍然由同一套推理器回答。

新感官走同一个注册口。`engine.register_module` 把本体感知挂到 `proprioception` 槽。`perceive_parallel` 把各模态的图做并集，`bind_cross_modal_relations` 按距离补上 `near_to`。夹爪和红色矩形能出现在同一张图里，是因为它们都是实体，不是因为推理器认识机械臂。

训练也按模块拆开。`trainer.py` 的第一阶段只对齐视觉投影，第二、三阶段才动关系注意力和分支路由。视觉升级时不必把推理权重一并推倒。

## 并行在哪里

并行分四层，都在一次 `engine.run` 里发生。

1. **一条感官内部并行。** `VisionModuleV1.encode_to_scene_graph` 用线程池同时跑颜色分割、边缘梯度和空间网格。三路结果回来之后才拼成实体。
2. **多种感官同时编码。** `NextGenModelEngine.perceive_parallel` 按已注册的模态各开一个任务，例如视觉和本体感知一起算，算完再合并。
3. **整张图一起传关系。** `ParallelRelationalCore.parallel_relational_attention` 用关系偏置矩阵让所有实体在同一轮里交换信息，推理深度不随实体个数逐个往后排。
4. **四条假设同时推演，再同时自检。** `rollout_hypotheses` 默认拉起四条策略不同的分支：`strict_high_margin`（余量至少 0.05）、`occlusion_compensated`（允许被遮挡物体）、`transitive_relational`（两跳传递）、`counterfactual_boundary`（临界边界）。`RCVSSelfVerifier.verify_group_parallel` 同时检查这四条，再用 `compute_grpo_advantages` 做组内相对排序。选中的是奖励最高且约束和误差界都过的那一条。`benchmark.py` 里 G=4 相对单分支的实测加速大约是 3.34 倍。

所以模块化决定「谁可以换」，并行决定「一次查询里有哪些计算一起跑」。换视觉模块不会拆掉这四条分支；增加一条感官也不会改分支策略的名字。

## 成品与剩余步骤

成品是一个能换视觉模块的关系推理器。上传一张图，它标出物体和关系，四条分支同时判断，再由自检决定接受或拒绝。

页面上应当看到：

- 不填词：标出颜色、形状，以及左右、上下。
- 填英文词：OWL-ViT 出框，SAM 2 描边界，并写出能从框判断的关系。左右上下之外，还有 `on`（底边贴住）、`wearing`（人穿着衣物）、`in`（大部分落在里面）、`has`（整体包住部件）。
- 每条结论能接受或拒绝。拒绝时能看出原因：方向反了、两个物体太近，或者框不满足这条关系的规则。
- 换掉视觉模块后，分支和自检不用改。

做到这里就停。机械臂仿真、Grounding DINO、DINOv2、通用对话都不在这个成品里。听觉或机械臂若以后要加，是新感官插进同一张场景图。

已经有的部分留着：三阶段训练、四条并行分支、RCVS、上传识别、增量回放、方向关系 76.00%。语义关系的合格线是：wearing 83.50%，in 62.50%，has 74.50%。`on` 的 200 条标注里，底边贴住上半部的有 17 条，这 17 条全部接受；另外 183 条是「在某个景物上」，单独计为规则不符，不拿来改规则。

三步都已完成：

1. **页面显示四种新关系。** 已完成。推理里的 `on`、`wearing`、`in`、`has` 会出现在摘要和关系列表里。页面上「看杯子在桌上」的摘要含有「杯子在桌子上」，「看人戴帽子」的摘要含有「人穿着帽子」。这两项不下载大模型。
2. **固定语义关系的合格线。** 已完成。`--semantic` 的报告里 `qualification.on` 分列了「底边贴住」和「宽松标注」。wearing、in、has 仍是 83.50%、62.50%、74.50%。方向关系复测仍是 76.00%。
3. **用检测器自己的框再考一次。** 已完成。15 张 Visual Genome 照片、每种关系 4 条，同一问法分别用标注框和 OWL-ViT 的框。标注框接受率相对检测框：left 75% 对 25%，right 100% 对 50%，above 25% 对 50%，below 100% 对 50%，on 都是 0%，wearing 都是 50%，in 75% 对 25%，has 100% 对 50%。这份对照在 `checkpoints/visual_genome_detector_box_report.json`，不改 800 条方向关系的 76.00%。

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
    ├── scene_graph_eval.py
    ├── serve.py
    ├── requirements-stage4-optional.txt
    ├── modules/
    │   ├── base.py
    │   ├── vision.py
    │   ├── vision_v3.py
    │   ├── relations.py
    │   ├── parallel_core.py
    │   ├── verifier.py
    │   ├── arm.py
    │   └── continual.py
    ├── tests/test_architecture.py
    ├── checkpoints/
    └── data/
```

`arm.py` 里有平面两连杆正运动学，机械臂仿真先不继续。`continual.py` 在推理时把「严格分支没答对、其他分支改对」的查询连同场景图写入回放池。`python3 -m New_Model.prototype.trainer --incremental` 读取这些记录，和 120 题里的清晰样本混在一起再跑一轮 GRPO。清晰、遮挡、临界和总准确率都没有下降时，新权重写到 `checkpoints/pmv_incremental_weights.npz`，原来的 `pmv_trained_weights.npz` 保持不变。

`scene_graph_eval.py` 用 Visual Genome 的真实关系标注检查开放词汇之后的关系推理。标注框走 `VisionModuleV3` 的同一条接口，再交给原来的并行分支和 RCVS。

方向关系是 left of、right of、above、on top of、under。每种 200 条，共 800 条，接受率 76.00%。方向和框中心一致、并且拉开到画面 4% 的样本全部接受。报告在 `checkpoints/visual_genome_relation_report.json`。

另外四种关系由框和名字写成边，色块视觉 V1、V2 不写这些边：

- `on`：上面物体的底边贴住下面物体的上半部，并且水平重叠。`on top of` 仍然算 `above`。
- `in`：一个框的大部分落在另一个框里面。
- `wearing`：人的名字配上衣服或配件的名字，并且衣服框和人的框重叠。
- `has`：整体框包住部件框。只是挨得近、没有包含时不写。

`python3 -m New_Model.prototype.scene_graph_eval --semantic` 对这四种各取 200 条。wearing、in、has 的接受率是 83.50%、62.50%、74.50%。`on` 的 200 条里，底边贴住上半部的 17 条全部接受，其余 183 条按宽松标注另计，不写进这条规则的接受率。报告在 `checkpoints/visual_genome_semantic_report.json`。

`python3 -m New_Model.prototype.scene_graph_eval --detector-boxes` 用 15 张真实照片比较标注框和 OWL-ViT 框。检测框在 left、right、below、in、has 上比标注框少接受约 50 个百分点；above 和 wearing、on 没有更差。报告在 `checkpoints/visual_genome_detector_box_report.json`。原始标注和下载的照片缓存在 `data/visual_genome/`，不进仓库。

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
- 「看杯子在桌上」摘要里有「杯子在桌子上」。「看人戴帽子」摘要里有「人穿着帽子」。这两个示例也不下载大模型。
- http://127.0.0.1:8765/predict?preset=clean 、`occluded`、`unstable`：前两个 `accepted` 为 true，`unstable` 为 false。PowerShell 里请用 `curl.exe`，或 `Invoke-RestMethod`。

其他命令：

```bash
python3 -m New_Model.prototype.trainer
python3 -m New_Model.prototype.trainer --incremental
python3 -m New_Model.prototype.scene_graph_eval
python3 -m New_Model.prototype.scene_graph_eval --semantic
python3 -m New_Model.prototype.scene_graph_eval --detector-boxes
python3 -m New_Model.prototype.evaluate_checkpoint
python3 -m New_Model.prototype.demo
python3 -m New_Model.prototype.benchmark
```
