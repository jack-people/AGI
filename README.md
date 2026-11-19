# AGI

下一代模型按两条线来做：**并行**，以及**从视觉开始的模块化**。感知模块只负责把原始信号翻译成同一张场景图，关系推理和自验证不跟某一个视觉实现绑在一起。

## 当前定位、能力边界与优化路线

> 当前项目是**面向有限场景的、可解释的视觉关系推理研究原型**。优势在于模块化、规则可追踪和快速验证；尚未证明通用 AGI、复杂真实图像泛化、严格全链路鲁棒认证或有效推理的普遍并行加速。
>
> 本节基于提交 `7f7ac75054121a1652717a20ac5d9daf2fe7f349` 的代码审查、非破坏性本地验证与仓库历史报告。**已有接口、已演示任务和已证明能力必须区分。以下优化是待实施建议，不表示源码已经修复。**

### 能力边界

| 能力 | 已有实现或演示 | 当前不能承诺的能力 |
| --- | --- | --- |
| 简单视觉感知 | 受控色块、简单形状、几何位置；V2 有滤波和遮挡补偿规则 | 复杂背景、严重遮挡、小目标、任意材质下的稳定识别 |
| 给定实体框的关系判断 | 左右、上下等方向，以及部分 `on`、`wearing`、`in`、`has` 关系 | 语义关系主要依赖类别和框几何，不等于理解支撑、穿戴、包含等常识 |
| 真实图开放词汇识别 | 可选 OWL-ViT / SAM 2 接口，也允许直接传入检测框 | 接口存在不代表端到端效果可靠；客户端提供框的结果不是检测器独立识别 |
| 多约束、多跳推理 | 四条固定策略，包含传递分支 | 检查点路径只处理第一条关系；二跳验证缺独立路径与余量核实，尚不能保证组合推理正确 |
| 任务学习 | 投影、节点匹配和四策略路由更新 | Q/K/V 未训练；不是端到端视觉语言学习，也不能等同于通用生成式 GRPO |
| 增量纠错 | 图快照回放、路由更新、旧任务不退化验收 | 历史仅一条回放且前后指标持平，未证明持续学习、纠正成功率提高或长期不遗忘 |
| 鲁棒性与拒答 | 固定常数构造的几何误差界和验证状态 | 容忍值门槛有已复现漏洞；不覆盖感知误差和语义不确定性，不是全链路认证 |
| 并行与模块替换 | 线程分支、模态注册和感知器替换 | 至多四条固定策略；小负载并行可能更慢；维度、实体 ID 等兼容性校验不完善 |
| 交互与部署 | 本地上传页面、预设任务、关系解释 | 不提供通用对话、长期规划或机器人闭环；服务不适合直接暴露公网 |

### 两条推理路径需要分开评价

- **符号引擎**：感知 → 场景图 → 四条手写策略候选 → RCVS 打分与验证。`benchmark` 和 `demo` 使用该路径，不加载训练检查点。
- **检查点路径**：固定 V2 场景图 → 16 维投影/注意力/节点匹配 → `policy` 的 `argmax` 选择一条策略 → 手写接受条件。它不是四路并行之后再统一 RCVS 验证。
- **实际训练**：Stage 1 更新 `w_vis_proj` / `b_vis_proj`，以原 embedding 自身对齐；Stage 2 只更新 `w_match`；Stage 3 和增量训练更新 `policy_logits`。Q/K/V 保持初始化，保存的 `branch_margin_thresholds` 未参与相应推理阈值判断。四策略奖励枚举与组相对优势更新，不能直接等同于带生成式轨迹采样、ratio/clipping/KL/reference-policy 的大模型策略训练。

实现入口：[符号引擎](<New_Model/prototype/engine.py>)、[检查点推理](<New_Model/prototype/evaluate_checkpoint.py>)、[训练器](<New_Model/prototype/trainer.py>)。

### 验证结果与证据边界

**本次审查期间的本地验证（不是本次文档提交重新运行）：**

- 筛选后的单元测试 **12/12 通过，1.593 秒**。总共 13 项，排除了会覆盖训练检查点和报告的全训练测试；增量项使用临时目录。测试通过不意味着以下负例已被覆盖。
- 60 题新 seed 的 unseen 合成集：随机权重整体准确率 **20%**；训练检查点整体 **100%**，FAR **0%**。它只证明该任务生成器下的表现，不等于现实分布外泛化。
- 120 题符号引擎 benchmark：A0 整体 **58.33%**，A4 **100%**。无 RCVS 配置仍使用经过验证排序的候选，消融不干净，不能据此单独归因某个模块。
- 去除不影响答案的人工 FFT 负载，在本机同一个 clean 样例、G=4、`simulated_work_steps=0`、每种模式一次预热加 20 次测量时，rollout 中位数为串行 **0.1995 ms**、线程并行 **1.2425 ms**，并行耗时约 **6.23 倍**。这只是小任务探针，不是全面性能结论。

**仓库提交的历史报告（未在本次审查重跑）：**

| 实验 | 报告结果 | 应如何解读 |
| --- | --- | --- |
| 三阶段训练 | 整体 17.50 → 35.83 → 56.67 → 98.33%；最终 clean 95% | 同一份 prepared 数据训练和阶段评估，不是 holdout |
| 增量回放 | 一条记录、8 epoch；before / after 都 100% | 展示聚合指标不退化，没有展示纠正率提升 |
| VG 标注框方向关系 | 608/800 = 76%，371 张图 | 正关系接受率，不是从图片检测到推理的整体准确率 |
| VG 标注框语义关系 | 458/800 = 57.25%；`on` 为 17/200 = 8.5% | `on` 在规则适用的 17 条上 100%，不能替代完整 200 条上的 8.5% |
| OWL 真实图小样本 | 15 张图、32 条关系；OWL 12/32 = 37.5%，标注框 21/32 = 65.625% | 小样本正关系接受率；找到双方实体 17/32 = 53.125%，不能视为全面检测或推理准确率 |

来源：[训练报告](<New_Model/prototype/checkpoints/training_report.json>)、[增量报告](<New_Model/prototype/checkpoints/incremental_report.json>)、[方向关系报告](<New_Model/prototype/checkpoints/visual_genome_relation_report.json>)、[语义关系报告](<New_Model/prototype/checkpoints/visual_genome_semantic_report.json>)、[真实图检测报告](<New_Model/prototype/checkpoints/visual_genome_detector_box_report.json>)。

评估仍有这些限制：120 题合成集均为 `left_of` 单约束，40 条 clean 仅 15 种不同场景模板；40 条 boundary 仅 2 个图像模板、3 个查询模板。FAR 的不稳定题仅 10 条且高度重复。VG 标注框实验以空白图加标注框输入，不衡量视觉检测；缺负关系 precision/FAR，按文件顺序选样且存在重复。报告缺代码版本、检查点哈希和环境绑定，不能把不同历史产物当作同一次实验。

### 已确认的正确性问题：最高优先级

以下反例在内存中构造实体、关系和路由，未修改源文件。强制路由用于覆盖分支边界，**不代表已测出训练检查点下的自然发生频率**。

1. **误差容忍门槛失效**：`error_tolerance=0.001`、`perturbation_radius=0.03`、关系 margin=0.6 时，误差界为 **0.04333**，仍 `PASS`；稳定分 **6.9242**。失败路径没有硬拒绝且稳定分无上限，见 [验证器](<New_Model/prototype/modules/verifier.py#L116-L145>)。
2. **第二条关系被忽略**：安全 `left_of` 外再要求不存在的 `wearing`，检查点推理仍接受；仅处理 `required_relations[0]`，见 [检查点推理](<New_Model/prototype/evaluate_checkpoint.py#L339-L347>)。
3. **第四策略漏检属性**：要求 triangle、实际 rectangle，强制第四路由仍接受；接受条件遗漏 `attr_matched`，见 [检查点推理](<New_Model/prototype/evaluate_checkpoint.py#L376-L377>)。

此外，理论半径使用 `sqrt(tolerance)`，与自身误差模型 `1.4142*r + r²` 不一致；默认 tolerance=.15 报 .3873，仅满足误差容忍的正确正根约 .0991，关系余量还需另加约束。二跳验证使用策略名称和固定余量 .08，也不能替代独立路径证明。

### 优化路线与验收标准

推荐顺序：**修正确性 → 建立可信评估 → 改善真实图表现 → 优化性能与服务 → 扩展学习能力**。

| 优先级 | 方向 | 建议实施内容 | 建议验收标准 |
| --- | --- | --- | --- |
| P0 | 统一结果契约 | 感知与候选生成之后共享验证器，硬检查属性、全部关系、路径、容忍值和安全余量；训练、评估、服务复用；区分 PASS / REJECT / UNKNOWN | 三个反例不再误接受；补多关系、无关系、属性错配和阈值边界测试；正常正例仍通过，不能靠全部拒答降低 FAR |
| P0 | 隔离副作用与版本 | 训练/导出显式指定输出；自检临时回放；修复自检忽略 replay 参数；保存配置、commit 和权重 hash | 默认测试不改变跟踪文件；报告与检查点可追溯 |
| P1 | 独立评估 | 按图像/模板拆分 train / validation / test；冻结测试集；加入负例、多对象、组合关系、背景/尺寸/遮挡变化 | 同时报告 precision、recall、FAR、拒答率、覆盖率及按图像统计的置信区间 |
| P1 | 感知与错误归因 | 检测/分割输出置信度；漏检返回 UNKNOWN；按物体尺寸和遮挡分桶，固定后端版本 | 分开测实体检测、给定正确框的关系、图片到最终答案三层，能定位瓶颈 |
| P1 | 语义关系学习 | 方向关系保留几何规则；`on` / `wearing` / `in` / `has` 引入实体外观、局部图像、掩膜与监督关系分类器 | 相同框布局、不同语义的对照能区分；在含负例的独立测试上超过规则基线 |
| P2 | 真实性能 | 移除人工 FFT；缓存、向量化；小任务串行，大负载再选择并行；消融每次只改一个变量 | 测感知到答案的端到端 p50/p95、吞吐、内存和多图规模；答案不变而有效性能改善 |
| P2 | 持续纠错 | 测回放题 before/after、纠正成功率和旧任务遗忘率；去重、数据版本与回滚 | 能证明错误被修正，而不只是旧聚合指标不下降 |
| P2 | 查询与服务 | 自然语言解析为结构约束，解析与验证解耦；输入限制、鉴权、超时、限流、回放写入和模型加载锁 | 解析误差与推理误差分别评估；不支持或证据不足明确拒答；服务稳定且安全 |
| P3 | 真正可训练的关系模型 | 训练 Q/K/V 或 GNN，属性/关系联合目标；先建立监督分类和简单路由基线；有可变轨迹与延迟奖励需求时再扩展策略优化 | 在独立测试上胜过规则和简单分类器，而非只提高训练集分数 |

统一验证应当把 **PASS（证据满足约束）**、**REJECT（证据与要求矛盾）**、**UNKNOWN（证据不足或漏检）** 分开。评估拒答必须同时看覆盖率，避免全部拒答造成虚假的可靠性。

近期建议定位为“**可解释、能拒答、结果可验证的视觉关系验证器**”，优先可靠性而不是分支数量、参数规模或更复杂的 GRPO 包装。如果目标仍是 AGI，需要新增语言、记忆、规划、工具使用和环境反馈等研究能力；这些不在现有实现或近期成品范围内，接口扩展本身也不是通用智能证据。

### 运行副作用和部署提醒

- 完整单元测试包含全训练，会覆盖检查点与报告；`python -B` 只关闭字节码缓存，不隔离业务写入。
- `evaluate_checkpoint` 无参数会导出/覆盖数据；仅复查合成 unseen 可显式运行 `python3 -B -m New_Model.prototype.evaluate_checkpoint --verify-unseen`。
- `serve --self-check` 会写回放，并且当前自检路径忽略 `--replay`；如要运行应使用隔离检查点目录。
- VG 评估和无客户端框的 OWL/SAM 识别可能下载数据、模型并写入目录或缓存。
- 服务缺认证、请求体上限、超时/限流与共享资源同步，默认只应在本机使用；不要将 `0.0.0.0` 当作安全的公网部署配置。

下面保留原有架构、任务范围与运行说明；其中实验数字应结合本节的证据边界理解。

## 模块化在哪里

模块之间只认一种接口。`PerceptionModule.encode_to_scene_graph`（`New_Model/prototype/modules/base.py`）把本模态的原始输入写成 `EntityNode` 和 `RelationEdge`。`ParallelRelationalCore` 和 `RCVSSelfVerifier` 只读这张图，不读像素，也不知道框是色块规则算的还是 OWL-ViT 给的。

插拔发生在模态槽位上。`ModuleRegistry`（`registry.py`）按 `vision`、`proprioception` 这类槽位保存当前模块。`NextGenModelEngine.upgrade_module` 替换槽位里的实现，下游推理代码不动。三版视觉都挂在 `vision` 槽：

- `VisionModuleV1`：颜色、边缘、空间三路直接在原图像素上取实体。
- `VisionModuleV2`：同一输出协议上增加 3×3 中值滤波和遮挡补全。遮挡场景因此能从 V1 的失败变成通过，测试在 `test_modular_hot_upgrade_v1_to_v2_on_occlusion_and_noise`。
- `VisionModuleV3`：检测器插件交出类别和框，SAM 2 再补掩膜，仍然写成同一张场景图。`test_stage4_open_vocab_hotswap_keeps_reasoner` 检查从 V2 热升级到 V3 之后，原来的「杯子在书左侧」查询仍然由同一套推理器回答。

新感官走同一个注册口。`engine.register_module` 把本体感知挂到 `proprioception` 槽。`perceive_parallel` 把各模态的图做并集，`bind_cross_modal_relations` 按距离补上 `near_to`。夹爪和红色矩形能出现在同一张图里，是因为它们都是实体，不是因为推理器认识机械臂。

训练也按模块拆开。`trainer.py` 的第一阶段对齐视觉投影，第二阶段更新节点匹配矩阵，第三阶段更新分支路由；当前 Q/K/V 不参与训练。模块接口允许独立替换感知器，但升级后的输入分布与检查点兼容性仍需单独验证。

## 并行在哪里

并行分四层，都在一次 `engine.run` 里发生。

1. **一条感官内部并行。** `VisionModuleV1.encode_to_scene_graph` 用线程池同时跑颜色分割、边缘梯度和空间网格。三路结果回来之后才拼成实体。
2. **多种感官同时编码。** `NextGenModelEngine.perceive_parallel` 按已注册的模态各开一个任务，例如视觉和本体感知一起算，算完再合并。
3. **整张图一起传关系。** `ParallelRelationalCore.parallel_relational_attention` 用关系偏置矩阵让所有实体在同一轮里交换信息，推理深度不随实体个数逐个往后排。
4. **四条假设同时推演，再同时自检。** 符号引擎的 `rollout_hypotheses` 默认拉起四条策略不同的分支：`strict_high_margin`（余量至少 0.05）、`occlusion_compensated`（允许被遮挡物体）、`transitive_relational`（两跳传递）、`counterfactual_boundary`（临界边界）。`RCVSSelfVerifier.verify_group_parallel` 检查候选并按奖励等指标排序；当前最高奖励选择并不保证所有硬约束和误差门槛均成立，已知漏洞见前文。`benchmark.py` 的历史约 3.34 倍加速包含不影响答案的人工 FFT 负载，不能直接视为有效推理加速；检查点路径也不执行这套四路流程。

所以模块化决定「谁可以换」，并行决定「一次查询里有哪些计算一起跑」。换视觉模块不会拆掉这四条分支；增加一条感官也不会改分支策略的名字。

## 成品与剩余步骤

成品目标是一个能换视觉模块的关系推理器。当前上传页面、符号引擎与检查点预设使用不同路径，不能概括为所有上传均执行四条分支并通过统一自检。以下是产品目标和已有演示，不代表任意真实图片上的可靠性保证。

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
python3 -m New_Model.prototype.serve --host 127.0.0.1 --port 8765
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
