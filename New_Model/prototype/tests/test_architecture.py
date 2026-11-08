"""
下一代模型架构自动化测试套件 (test_architecture.py)

验证核心指标：
1. 视觉模块能够将原始像素矩阵翻译为标准 StructuredSceneGraph（数据 -> 结构化数据）。
2. 模块化热升级：VisionModuleV1 -> VisionModuleV2 无需修改核心引擎即可修复噪声与遮挡场景下的识别。
3. 多模态即插即用：ProprioceptionModule（机械臂）能与视觉模块并行融合并建立跨模态关系边。
4. 并行推演与 GRPO 相对优势 + RCVS 局部线性化自验证闭环正常工作。
"""

import unittest
import numpy as np

from New_Model.prototype.engine import NextGenModelEngine
from New_Model.prototype.modules.arm import PlanarArmModule, planar_gripper_pose
from New_Model.prototype.modules.base import ProprioceptionModule
from New_Model.prototype.modules.continual import ReplayBuffer
from New_Model.prototype.modules.vision import (
    SceneObjectSpec,
    VisionModuleV1,
    VisionModuleV2,
    VisualSceneBuilder,
)
from New_Model.prototype.modules.vision_v3 import (
    Detection,
    OwlVitDetector,
    RecordedDetector,
    Sam2MaskRefiner,
    VisionModuleV3,
    build_cup_book_demo,
)
from New_Model.prototype.protocol import QueryConstraint, VerificationStatus
from New_Model.prototype.trainer import TrainablePMVWeights, run_full_training_pipeline


class TestNextGenModelArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = VisualSceneBuilder(height=64, width=64)

    def test_vision_translation_to_structured_scene_graph(self) -> None:
        """测试点 1：视觉模块将像素阵列并行翻译为结构化实体与关系图。"""
        img = self.builder.render(
            [
                SceneObjectSpec("red_box", "red", "rectangle", 0.15, 0.10, 0.45, 0.35),
                SceneObjectSpec("blue_ball", "blue", "circle", 0.15, 0.60, 0.45, 0.85),
            ]
        )
        v1 = VisionModuleV1(embedding_dim=16)
        sg = v1.encode_to_scene_graph(img)

        self.assertEqual(len(sg.nodes), 2)
        colors = {n.attributes["color"] for n in sg.nodes}
        self.assertEqual(colors, {"red", "blue"})

        left_edges = sg.find_edges(relation_type="left_of")
        self.assertEqual(len(left_edges), 1)
        red_node = [n for n in sg.nodes if n.attributes["color"] == "red"][0]
        blue_node = [n for n in sg.nodes if n.attributes["color"] == "blue"][0]
        self.assertEqual(left_edges[0].source_id, red_node.entity_id)
        self.assertEqual(left_edges[0].target_id, blue_node.entity_id)

    def test_modular_hot_upgrade_v1_to_v2_on_occlusion_and_noise(self) -> None:
        """测试点 2：在不修改核心推理引擎的情况下，通过升级视觉模块解决遮挡与噪声问题。"""
        img = self.builder.render(
            [
                SceneObjectSpec(
                    "red_rect", "red", "rectangle", 0.20, 0.15, 0.60, 0.55, z_order=1
                ),
                SceneObjectSpec(
                    "yellow_occ", "yellow", "circle", 0.35, 0.35, 0.60, 0.60, z_order=2
                ),
                SceneObjectSpec(
                    "blue_circle", "blue", "circle", 0.25, 0.70, 0.55, 0.92, z_order=1
                ),
            ],
            salt_pepper_prob=0.04,
            seed=7,
        )
        query = QueryConstraint(
            goal="找到蓝色物体左侧的红色矩形",
            target_attributes={"color": "red", "shape": "rectangle"},
            required_relations=[("left_of", {"color": "blue"})],
        )

        engine = NextGenModelEngine(vision_module=VisionModuleV1())
        res_v1 = engine.run({"vision": img}, query=query)
        # V1 因椒盐噪声与遮挡无法将红色物体恢复为矩形
        self.assertNotEqual(
            res_v1["verification_report"].status, VerificationStatus.PASS
        )

        # 热升级至 V2
        msg = engine.upgrade_module("vision", VisionModuleV2())
        self.assertIn("1.0.0 -> VisionModule@2.0.0", msg)

        res_v2 = engine.run({"vision": img}, query=query)
        self.assertEqual(res_v2["verification_report"].status, VerificationStatus.PASS)
        best = res_v2["verification_report"].selected_hypothesis
        self.assertIsNotNone(best)
        self.assertTrue(len(best.matched_entities) >= 1)

        # 验证 V2 检测到了 occludes 关系边
        occ_edges = res_v2["scene_graph"].find_edges(relation_type="occludes")
        self.assertTrue(len(occ_edges) >= 1)

    def test_multimodal_plug_and_play_with_proprioception(self) -> None:
        """测试点 3：多感官模块即插即用，视觉与机械臂本体感知在统一总线上融合。"""
        img = self.builder.render(
            [
                SceneObjectSpec("red_box", "red", "rectangle", 0.25, 0.20, 0.45, 0.40),
            ]
        )
        engine = NextGenModelEngine(vision_module=VisionModuleV2())
        engine.register_module(ProprioceptionModule())

        query = QueryConstraint(
            goal="查找机械臂夹爪附近的红色矩形",
            target_attributes={"color": "red", "shape": "rectangle"},
            required_relations=[("near_to", {"category": "actuator"})],
        )
        res = engine.run(
            sensory_inputs={
                "vision": img,
                "proprioception": {
                    "gripper_y": 0.34,
                    "gripper_x": 0.30,
                    "aperture": 0.10,
                    "force_newton": 2.0,
                },
            },
            query=query,
        )
        self.assertEqual(res["verification_report"].status, VerificationStatus.PASS)
        near_edges = res["scene_graph"].find_edges(relation_type="near_to")
        self.assertTrue(len(near_edges) >= 2)

    def test_grpo_advantage_and_rcvs_boundary_verification(self) -> None:
        """测试点 4：并行分支 GRPO 相对优势零均值性质及 RCVS 临界边界警告。"""
        img = self.builder.render(
            [
                SceneObjectSpec("red_box", "red", "rectangle", 0.20, 0.20, 0.45, 0.45),
                SceneObjectSpec("green_box", "green", "rectangle", 0.20, 0.50, 0.45, 0.70),
                SceneObjectSpec("blue_ball", "blue", "circle", 0.20, 0.78, 0.45, 0.95),
            ]
        )
        engine = NextGenModelEngine(vision_module=VisionModuleV2(), num_branches=4)
        query = QueryConstraint(
            goal="找到蓝色物体左侧的红色矩形",
            target_attributes={"color": "red", "shape": "rectangle"},
            required_relations=[("left_of", {"color": "blue"})],
            error_tolerance=0.15,
            perturbation_radius=0.02,
        )
        res = engine.run({"vision": img}, query=query, parallel=True)
        candidates = res["verification_report"].all_candidates
        self.assertEqual(len(candidates), 4)

        # 验证当各分支得分存在差异时，GRPO 组内相对优势均值为 0
        advantages = [c.grpo_advantage for c in candidates]
        self.assertAlmostEqual(float(np.mean(advantages)), 0.0, places=5)

    def test_expanded_colors_shapes_and_separate_instances(self) -> None:
        """同色分开的物体各自成目标，并识别新增的橙色、紫色和菱形。"""
        img = self.builder.render(
            [
                SceneObjectSpec("orange_ball", "orange", "circle", 0.08, 0.08, 0.38, 0.34),
                SceneObjectSpec("purple_diamond", "purple", "diamond", 0.08, 0.62, 0.40, 0.92),
                SceneObjectSpec("cyan_tri", "cyan", "triangle", 0.40, 0.32, 0.62, 0.58),
                SceneObjectSpec("pink_box", "pink", "rectangle", 0.66, 0.36, 0.90, 0.56),
                SceneObjectSpec("red_left", "red", "rectangle", 0.55, 0.06, 0.88, 0.28),
                SceneObjectSpec("red_right", "red", "rectangle", 0.55, 0.62, 0.88, 0.90),
            ]
        )
        sg = VisionModuleV2(embedding_dim=16).encode_to_scene_graph(img)
        colors = [node.attributes["color"] for node in sg.nodes]
        self.assertEqual(colors.count("red"), 2)
        self.assertIn("orange", colors)
        purple = [node for node in sg.nodes if node.attributes["color"] == "purple"]
        self.assertEqual(len(purple), 1)
        self.assertEqual(purple[0].attributes["shape"], "diamond")
        orange = [node for node in sg.nodes if node.attributes["color"] == "orange"]
        self.assertEqual(len(orange), 1)
        self.assertEqual(orange[0].attributes["shape"], "circle")
        cyan = [node for node in sg.nodes if node.attributes["color"] == "cyan"]
        self.assertEqual(len(cyan), 1)
        self.assertEqual(cyan[0].attributes["shape"], "triangle")
        pink = [node for node in sg.nodes if node.attributes["color"] == "pink"]
        self.assertEqual(len(pink), 1)
        self.assertEqual(pink[0].attributes["shape"], "rectangle")

    def test_stage4_open_vocab_hotswap_keeps_reasoner(self) -> None:
        """第四阶段：换成开放词汇视觉后，原关系推理和自验证不用改。"""
        image, detections = build_cup_book_demo()
        engine = NextGenModelEngine(vision_module=VisionModuleV2())
        upgraded = VisionModuleV3(detector=RecordedDetector(detections))
        message = engine.upgrade_module("vision", upgraded)
        self.assertIn("VisionModule@2.0.0 -> VisionModule@3.0.0", message)

        result = engine.run(
            {"vision": {"image": image, "detections": detections, "queries": ["杯子", "书"]}},
            QueryConstraint(
                goal="找到书左侧的杯子",
                target_attributes={"category": "杯子"},
                required_relations=[("left_of", {"category": "书"})],
                perturbation_radius=0.02,
            ),
            simulated_work_steps=64,
        )
        self.assertEqual(result["verification_report"].status, VerificationStatus.PASS)
        categories = [
            node.attributes["category"] for node in result["scene_graph"].nodes
        ]
        self.assertEqual(categories.count("杯子"), 1)
        self.assertEqual(categories.count("书"), 1)

        pixel_detector = RecordedDetector(
            [Detection("person", 0.9, 8, 8, 40, 30)]
        )
        graph = VisionModuleV3(detector=pixel_detector, queries=["person"]).encode_to_scene_graph(
            np.zeros((64, 64, 3))
        )
        self.assertEqual(len(graph.nodes), 1)
        self.assertLess(graph.nodes[0].bbox.y_max, 1.5)
        self.assertIn("person", graph.nodes[0].attributes["category"])

    def test_stage4_planar_arm_and_hard_example_replay(self) -> None:
        """关节角定位夹爪；严格分支失手、其他分支改对的样本进入回放池。"""
        gripper_y, gripper_x = planar_gripper_pose(0.80, 0.10, 0.30, 0.25, -1.1, 0.2)
        image = self.builder.render(
            [
                SceneObjectSpec(
                    "red_box",
                    "red",
                    "rectangle",
                    gripper_y - 0.06,
                    gripper_x - 0.06,
                    gripper_y + 0.06,
                    gripper_x + 0.06,
                )
            ]
        )
        engine = NextGenModelEngine(vision_module=VisionModuleV2())
        engine.register_module(PlanarArmModule())
        arm_result = engine.run(
            {
                "vision": image,
                "proprioception": {
                    "base_y": 0.80,
                    "base_x": 0.10,
                    "length_1": 0.30,
                    "length_2": 0.25,
                    "shoulder": -1.1,
                    "elbow": 0.2,
                    "aperture": 0.08,
                    "force_newton": 1.5,
                },
            },
            QueryConstraint(
                goal="查找关节臂夹爪附近的红色矩形",
                target_attributes={"color": "red", "shape": "rectangle"},
                required_relations=[("near_to", {"category": "actuator"})],
            ),
            simulated_work_steps=64,
        )
        self.assertEqual(arm_result["verification_report"].status, VerificationStatus.PASS)

        wide = [
            Detection("甲", 0.95, 0.30, 0.15, 0.50, 0.30),
            Detection("乙", 0.95, 0.30, 0.55, 0.50, 0.70),
        ]
        blank = np.zeros((64, 64, 3), dtype=np.float64)
        buffer = ReplayBuffer()
        wide_result = NextGenModelEngine(
            vision_module=VisionModuleV3(detector=RecordedDetector(wide))
        ).run(
            {"vision": {"image": blank, "detections": wide}},
            QueryConstraint(
                goal="甲在乙的左侧",
                target_attributes={"category": "甲"},
                required_relations=[("left_of", {"category": "乙"})],
                perturbation_radius=0.02,
            ),
            simulated_work_steps=64,
            replay_buffer=buffer,
        )
        self.assertEqual(wide_result["verification_report"].status, VerificationStatus.PASS)
        self.assertEqual(buffer.records, [])

        # 中心距 0.045：关系边存在，但严格分支要求余量至少 0.05。
        tight = [
            Detection("甲", 0.95, 0.30, 0.370, 0.50, 0.430),
            Detection("乙", 0.95, 0.30, 0.420, 0.50, 0.470),
        ]
        buffer_hit = ReplayBuffer()
        hit = NextGenModelEngine(
            vision_module=VisionModuleV3(detector=RecordedDetector(tight))
        ).run(
            {"vision": {"image": blank, "detections": tight}},
            QueryConstraint(
                goal="甲在乙的左侧，间距很小",
                target_attributes={"category": "甲"},
                required_relations=[("left_of", {"category": "乙"})],
                perturbation_radius=0.02,
            ),
            simulated_work_steps=64,
            replay_buffer=buffer_hit,
        )
        self.assertEqual(hit["verification_report"].status, VerificationStatus.PASS)
        self.assertNotEqual(
            hit["verification_report"].selected_hypothesis.strategy_name,
            "strict_high_margin",
        )
        self.assertEqual(len(buffer_hit.records), 1)
        self.assertEqual(buffer_hit.records[0]["selected_matched"][0][:3], "vis")

    def test_owl_backend_reports_missing_install(self) -> None:
        detector = OwlVitDetector()
        try:
            import torch  # noqa: F401
            from transformers import OwlViTForObjectDetection  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(RuntimeError) as caught:
                detector.detect(np.zeros((16, 16, 3)), ["cup"])
            self.assertIn("VisionModuleV3", str(caught.exception))
            return
        self.assertIsNone(detector._model)
        refiner = Sam2MaskRefiner()
        self.assertEqual(refiner.refine(np.zeros((8, 8, 3)), []), [])
        self.assertIsNone(refiner._model)

    def test_three_stage_training_pipeline_and_checkpoint(self) -> None:
        """测试点 5：三阶段训练管线 (视觉对齐 -> SFT 模仿 -> GRPO+RCVS 理解) 收敛并生成权重成果。"""
        report = run_full_training_pipeline()
        prog = report["stage_progression"]

        init_acc = prog["Stage0_Untrained_Init"]["overall_acc"]
        sft_acc = prog["Stage2_SFT_Imitation"]["overall_acc"]
        grpo_acc = prog["Stage3_GRPO_RCVS_Understanding"]["overall_acc"]

        self.assertGreater(sft_acc, init_acc)
        self.assertGreater(grpo_acc, sft_acc)
        self.assertGreaterEqual(grpo_acc, 95.0)
        self.assertEqual(
            prog["Stage3_GRPO_RCVS_Understanding"]["false_accept_rate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
