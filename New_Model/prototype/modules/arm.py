"""
第四阶段机械臂本体感知：由关节角做平面正运动学，再写回原来的夹爪节点。

还不连接 MuJoCo 或 Isaac。关节角先变成夹爪的归一化位置，
后面的 near_to 关系和 RCVS 仍使用第一阶段的本体感知协议。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..protocol import BoundingBox, EntityNode, StructuredSceneGraph
from .base import PerceptionModule


def planar_gripper_pose(
    base_y: float,
    base_x: float,
    length_1: float,
    length_2: float,
    shoulder: float,
    elbow: float,
) -> Tuple[float, float]:
    """两连杆平面臂。0 弧度指向 +x，正角转向画面下方的 +y。"""
    wrist = shoulder + elbow
    gripper_x = base_x + length_1 * math.cos(shoulder) + length_2 * math.cos(wrist)
    gripper_y = base_y + length_1 * math.sin(shoulder) + length_2 * math.sin(wrist)
    return gripper_y, gripper_x


class PlanarArmModule(PerceptionModule):
    """用肩关节和肘关节定位夹爪。给定 gripper_y / gripper_x 时仍直接采用该位置。"""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__(
            module_id="ProprioceptionArmModule",
            modality="proprioception",
            version="2.0.0",
            embedding_dim=embedding_dim,
        )

    def encode_to_scene_graph(self, raw_input: Dict[str, Any]) -> StructuredSceneGraph:
        aperture = float(raw_input.get("aperture", 0.1))
        force = float(raw_input.get("force_newton", 0.0))
        if "shoulder" in raw_input:
            gripper_y, gripper_x = planar_gripper_pose(
                base_y=float(raw_input.get("base_y", 0.5)),
                base_x=float(raw_input.get("base_x", 0.5)),
                length_1=float(raw_input.get("length_1", 0.25)),
                length_2=float(raw_input.get("length_2", 0.20)),
                shoulder=float(raw_input["shoulder"]),
                elbow=float(raw_input.get("elbow", 0.0)),
            )
        else:
            gripper_y = float(raw_input.get("gripper_y", 0.5))
            gripper_x = float(raw_input.get("gripper_x", 0.5))

        half = aperture / 2.0
        emb = np.zeros(self.embedding_dim, dtype=np.float64)
        emb[0:4] = [
            gripper_y,
            gripper_x,
            aperture,
            min(force / 10.0, 1.0),
        ]
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        node = EntityNode(
            entity_id="arm_gripper_0",
            modality=self.modality,
            bbox=BoundingBox(
                y_min=max(0.0, gripper_y - half),
                x_min=max(0.0, gripper_x - half),
                y_max=min(1.0, gripper_y + half),
                x_max=min(1.0, gripper_x + half),
            ),
            attributes={
                "category": "actuator",
                "shape": "gripper",
                "color": "metallic",
                "force_newton": force,
                "is_holding": force > 1.0,
                "gripper_y": round(gripper_y, 4),
                "gripper_x": round(gripper_x, 4),
            },
            embedding=emb,
            confidence=0.98,
        )
        return StructuredSceneGraph(
            nodes=[node],
            edges=[],
            source_modules=[f"{self.module_id}@{self.version}"],
            metadata={"proprioception_active": True, "kinematics": "planar_2link"},
        )
