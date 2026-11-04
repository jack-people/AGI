"""
模块抽象基类与扩展示例模块

定义统一感官模块接口 (PerceptionModule)，所有模态（视觉、听觉、触觉/机械臂本体感知）
只需实现 `encode_to_scene_graph` 即可接入下一代模型架构总线。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

from ..protocol import BoundingBox, EntityNode, RelationEdge, StructuredSceneGraph


class PerceptionModule(ABC):
    """感官模块统一基类：负责将对应模态的原始数据“翻译”为结构化场景图。"""

    def __init__(self, module_id: str, modality: str, version: str, embedding_dim: int = 16) -> None:
        self.module_id = module_id
        self.modality = modality
        self.version = version
        self.embedding_dim = embedding_dim

    @abstractmethod
    def encode_to_scene_graph(self, raw_input: Any) -> StructuredSceneGraph:
        """将原始感知输入翻译为统一结构化表征 (StructuredSceneGraph)。"""
        raise NotImplementedError


class ProprioceptionModule(PerceptionModule):
    """
    扩展感官示例：机械臂/触觉本体感知模块 (对接仓库中的 `机械臂` 方向)。
    用于验证架构的多模态模块化扩展能力：无需修改视觉或核心推理代码，
    即可将机械臂末端与视觉场景中的物体融合到同一张关系图中。
    """

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__(
            module_id="ProprioceptionArmModule",
            modality="proprioception",
            version="1.0.0",
            embedding_dim=embedding_dim,
        )

    def encode_to_scene_graph(self, raw_input: Dict[str, Any]) -> StructuredSceneGraph:
        """
        raw_input 示例:
        {
            "gripper_y": 0.35,
            "gripper_x": 0.28,
            "aperture": 0.12,
            "force_newton": 2.5,
        }
        """
        gy = float(raw_input.get("gripper_y", 0.5))
        gx = float(raw_input.get("gripper_x", 0.5))
        half = float(raw_input.get("aperture", 0.1)) / 2.0
        force = float(raw_input.get("force_newton", 0.0))

        emb = np.zeros(self.embedding_dim, dtype=np.float64)
        emb[0:4] = [gy, gx, half * 2.0, min(force / 10.0, 1.0)]
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        node = EntityNode(
            entity_id="arm_gripper_0",
            modality=self.modality,
            bbox=BoundingBox(
                y_min=max(0.0, gy - half),
                x_min=max(0.0, gx - half),
                y_max=min(1.0, gy + half),
                x_max=min(1.0, gx + half),
            ),
            attributes={
                "category": "actuator",
                "shape": "gripper",
                "color": "metallic",
                "force_newton": force,
                "is_holding": force > 1.0,
            },
            embedding=emb,
            confidence=0.98,
        )

        return StructuredSceneGraph(
            nodes=[node],
            edges=[],
            source_modules=[f"{self.module_id}@{self.version}"],
            metadata={"proprioception_active": True},
        )
