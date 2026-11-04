"""
端到端演示与架构验证脚本 (demo.py)

运行方式：
    python3 -m New_Model.prototype.demo

演示内容：
1. 【从视觉开始的结构化翻译 + 并行推理】：将 2D 多物体图像翻译为结构化关系图，并行推演并自验证。
2. 【视觉模块热插拔升级 (Vision V1 -> V2)】：在噪声与遮挡场景下，无需修改核心引擎，仅升级视觉模块即解决遮挡识别难题。
3. 【多感官模块化扩展 (视觉 + 机械臂本体感知)】：动态挂载机械臂模块，在同一结构化总线上完成跨模态空间推理。
4. 【并行能力基准测试】：对比多分支并行推演与串行推演的吞吐效率及 GRPO 组内相对优势分布。
"""

from __future__ import annotations

import json
from .engine import NextGenModelEngine
from .modules.base import ProprioceptionModule
from .modules.vision import (
    SceneObjectSpec,
    VisionModuleV1,
    VisionModuleV2,
    VisualSceneBuilder,
)
from .protocol import QueryConstraint


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def demo_1_basic_vision_and_parallel_reasoning() -> None:
    print_banner("实验 1：从视觉开始 —— 原始像素 -> 结构化关系图 -> G 路并行推理与自验证")

    builder = VisualSceneBuilder(height=64, width=64)
    image = builder.render(
        [
            SceneObjectSpec(
                name="red_rect",
                color="red",
                shape="rectangle",
                y_min=0.15,
                x_min=0.10,
                y_max=0.45,
                x_max=0.38,
                z_order=1,
            ),
            SceneObjectSpec(
                name="blue_circle",
                color="blue",
                shape="circle",
                y_min=0.18,
                x_min=0.58,
                y_max=0.48,
                x_max=0.88,
                z_order=1,
            ),
            SceneObjectSpec(
                name="green_triangle",
                color="green",
                shape="triangle",
                y_min=0.60,
                x_min=0.35,
                y_max=0.90,
                x_max=0.65,
                z_order=1,
            ),
        ]
    )

    engine = NextGenModelEngine(embedding_dim=16, num_branches=4)
    query = QueryConstraint(
        goal="寻找位于蓝色圆形左侧、且位于绿色三角形上方的红色矩形物体",
        target_attributes={"color": "red", "shape": "rectangle"},
        required_relations=[
            ("left_of", {"color": "blue", "shape": "circle"}),
            ("above", {"color": "green", "shape": "triangle"}),
        ],
        error_tolerance=0.12,
        perturbation_radius=0.025,
    )

    result = engine.run(sensory_inputs={"vision": image}, query=query, parallel=True)
    sg = result["scene_graph"]
    report = result["verification_report"]

    print(f"[感知翻译] 活跃模块: {sg.source_modules}")
    print(f"[感知翻译] 提取实体数: {len(sg.nodes)}, 关系边数: {len(sg.edges)}")
    for node in sg.nodes:
        print(
            f"  - 实体 {node.entity_id}: color={node.attributes['color']}, "
            f"shape={node.attributes['shape']}, center={tuple(round(c, 2) for c in node.bbox.center)}, "
            f"conf={node.confidence:.2f}"
        )

    print("\n[并行分支 & GRPO 组相对评估结果]:")
    for cand in report.all_candidates:
        print(
            f"  * Branch-{cand.branch_id} ({cand.strategy_name:<24}) | "
            f"匹配={cand.matched_entities} | 约束分={cand.constraint_score:.2f} | "
            f"稳定分={cand.stability_score:.2f} | 总奖励 R={cand.total_reward:.4f} | "
            f"GRPO优势 A={cand.grpo_advantage:+.4f} | 状态={cand.verification_status.value}"
        )

    print(f"\n[RCVS 自验证结论]: 状态 = {report.status.value}")
    print(f"  适用域与稳定性边界: {report.applicable_domain}")
    print(f"  局部线性误差上界 |R_1(x)| <= {report.local_linear_error_bound}")


def demo_2_modular_vision_hot_upgrade() -> None:
    print_banner("实验 2：模块化热升级 —— 遮挡与传感器噪声场景下 VisionModule V1 -> V2 无损升级")

    builder = VisualSceneBuilder(height=64, width=64)
    # 红色矩形被黄色圆形遮挡了一角，且图像存在椒盐噪声
    occluded_noisy_image = builder.render(
        [
            SceneObjectSpec(
                name="red_rect_occluded",
                color="red",
                shape="rectangle",
                y_min=0.20,
                x_min=0.15,
                y_max=0.60,
                x_max=0.55,
                z_order=1,
            ),
            SceneObjectSpec(
                name="yellow_occluder",
                color="yellow",
                shape="circle",
                y_min=0.35,
                x_min=0.35,
                y_max=0.60,
                x_max=0.60,
                z_order=2,  # 覆盖在红色矩形右下方
            ),
            SceneObjectSpec(
                name="blue_circle",
                color="blue",
                shape="circle",
                y_min=0.25,
                x_min=0.70,
                y_max=0.55,
                x_max=0.92,
                z_order=1,
            ),
        ],
        salt_pepper_prob=0.04,
        seed=7,
    )

    query = QueryConstraint(
        goal="在存在遮挡与噪声的场景中，识别位于蓝色物体左侧的红色矩形",
        target_attributes={"color": "red", "shape": "rectangle"},
        required_relations=[("left_of", {"color": "blue"})],
        error_tolerance=0.15,
        perturbation_radius=0.03,
    )

    engine = NextGenModelEngine(
        embedding_dim=16, num_branches=4, vision_module=VisionModuleV1()
    )

    # 1) 使用 V1 运行
    res_v1 = engine.run({"vision": occluded_noisy_image}, query=query)
    rep_v1 = res_v1["verification_report"]
    red_node_v1 = [
        n for n in res_v1["scene_graph"].nodes if n.attributes.get("color") == "red"
    ][0]
    print(
        f"[升级前 VisionModuleV1] 红色物体识别结果: shape={red_node_v1.attributes['shape']}, "
        f"fill_ratio={red_node_v1.attributes['fill_ratio']}, occluded={red_node_v1.attributes['occluded']}"
    )
    print(
        f"[升级前 VisionModuleV1] 推理最终状态: {rep_v1.status.value} "
        f"(命中实体: {rep_v1.selected_hypothesis.matched_entities if rep_v1.selected_hypothesis else []})"
    )

    # 2) 热升级视觉模块至 VisionModuleV2（不改动核心引擎与验证器任何代码）
    upgrade_msg = engine.upgrade_module("vision", VisionModuleV2())
    print(f"\n>>> 执行模块热升级: {upgrade_msg}")

    res_v2 = engine.run({"vision": occluded_noisy_image}, query=query)
    rep_v2 = res_v2["verification_report"]
    red_node_v2 = [
        n for n in res_v2["scene_graph"].nodes if n.attributes.get("color") == "red"
    ][0]
    occ_edges = res_v2["scene_graph"].find_edges(relation_type="occludes")

    print(
        f"[升级后 VisionModuleV2] 红色物体识别结果: shape={red_node_v2.attributes['shape']}, "
        f"fill_ratio={red_node_v2.attributes['fill_ratio']}, occluded={red_node_v2.attributes['occluded']}, "
        f"occluded_by={red_node_v2.attributes.get('occluded_by')}"
    )
    print(
        f"[升级后 VisionModuleV2] 新增遮挡关系边: {[e.to_dict() for e in occ_edges]}"
    )
    print(
        f"[升级后 VisionModuleV2] 推理最终状态: {rep_v2.status.value} "
        f"(命中实体: {rep_v2.selected_hypothesis.matched_entities if rep_v2.selected_hypothesis else []})"
    )


def demo_3_multimodal_plug_and_play() -> None:
    print_banner("实验 3：五感与执行器扩展 —— 视觉模块 + 机械臂本体感知模块并行融合")

    builder = VisualSceneBuilder(height=64, width=64)
    image = builder.render(
        [
            SceneObjectSpec(
                name="target_red_box",
                color="red",
                shape="rectangle",
                y_min=0.25,
                x_min=0.20,
                y_max=0.45,
                x_max=0.40,
            ),
            SceneObjectSpec(
                name="distant_blue_circle",
                color="blue",
                shape="circle",
                y_min=0.65,
                x_min=0.70,
                y_max=0.88,
                x_max=0.92,
            ),
        ]
    )

    engine = NextGenModelEngine(vision_module=VisionModuleV2())
    # 动态接入机械臂本体感知模块
    engine.register_module(ProprioceptionModule())

    query = QueryConstraint(
        goal="定位当前与机械臂末端夹爪处于空间邻近状态 (near_to) 的红色视觉物体",
        target_attributes={"color": "red", "shape": "rectangle"},
        required_relations=[("near_to", {"category": "actuator", "shape": "gripper"})],
    )

    result = engine.run(
        sensory_inputs={
            "vision": image,
            "proprioception": {
                "gripper_y": 0.34,
                "gripper_x": 0.31,
                "aperture": 0.12,
                "force_newton": 2.8,
            },
        },
        query=query,
    )
    sg = result["scene_graph"]
    report = result["verification_report"]

    print(f"[多模态注册状态]: {json.dumps(result['active_modules'], ensure_ascii=False)}")
    print(f"[跨模态融合场景图] 节点总数={len(sg.nodes)}, 跨模态关系边 near_to={ [e.to_dict() for e in sg.find_edges(relation_type='near_to')] }")
    print(
        f"[跨模态推理验证结论] 状态={report.status.value}, "
        f"命中目标={report.selected_hypothesis.matched_entities if report.selected_hypothesis else []}"
    )


def demo_4_parallel_speedup_benchmark() -> None:
    print_banner("实验 4：并行能力基准测试 —— 多分支并行推演 vs. 串行推演性能对比")

    builder = VisualSceneBuilder(height=64, width=64)
    image = builder.render(
        [
            SceneObjectSpec("r", "red", "rectangle", 0.1, 0.1, 0.4, 0.4),
            SceneObjectSpec("b", "blue", "circle", 0.1, 0.6, 0.4, 0.9),
        ]
    )
    engine = NextGenModelEngine(vision_module=VisionModuleV2(), num_branches=4)
    query = QueryConstraint(
        goal="基准测试查询",
        target_attributes={"color": "red"},
        required_relations=[("left_of", {"color": "blue"})],
    )

    # 预热
    engine.run({"vision": image}, query=query, parallel=True, simulated_work_steps=500)

    work_steps = 4000
    res_serial = engine.run(
        {"vision": image}, query=query, parallel=False, simulated_work_steps=work_steps
    )
    res_parallel = engine.run(
        {"vision": image}, query=query, parallel=True, simulated_work_steps=work_steps
    )

    t_serial = res_serial["metrics"]["rollout_time_ms"]
    t_parallel = res_parallel["metrics"]["rollout_time_ms"]
    speedup = t_serial / max(0.1, t_parallel)

    print(f"  4 分支串行推演耗时 (Serial Rollout)  : {t_serial:.2f} ms")
    print(f"  4 分支并行推演耗时 (Parallel Rollout): {t_parallel:.2f} ms")
    print(f"  并行加速比 (Speedup)                 : {speedup:.2f}x")


def main() -> None:
    demo_1_basic_vision_and_parallel_reasoning()
    demo_2_modular_vision_hot_upgrade()
    demo_3_multimodal_plug_and_play()
    demo_4_parallel_speedup_benchmark()
    print("\n" + "=" * 78)
    print("  所有下一代模型架构（并行能力 + 模块化视觉起步）实验验证完成！")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
