"""
下一代模型架构主引擎 (NextGenModelEngine)

将“模块化感知（从视觉开始）”、“统一结构化表征总线”、“并行关系推演核心”与“RCVS 自验证”
整合为一套高内聚、低耦合、可热升级的完整系统。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict, Optional
import numpy as np

from .modules.base import PerceptionModule
from .modules.parallel_core import ParallelRelationalCore
from .modules.verifier import RCVSSelfVerifier
from .modules.vision import VisionModuleV1
from .protocol import QueryConstraint, StructuredSceneGraph, VerificationReport
from .registry import ModuleRegistry


class NextGenModelEngine:
    """
    下一代并行模块化模型主引擎 (PMV Engine)
    默认装载视觉模块 (VisionModuleV1)，支持随时热升级至 VisionModuleV2 或扩展新感官模块。
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        num_branches: int = 4,
        vision_module: Optional[PerceptionModule] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.registry = ModuleRegistry()
        self.registry.register(
            vision_module
            if vision_module is not None
            else VisionModuleV1(embedding_dim=embedding_dim)
        )
        self.core = ParallelRelationalCore(
            embedding_dim=embedding_dim,
            num_branches=num_branches,
        )
        self.verifier = RCVSSelfVerifier(parallel_workers=num_branches)

    def register_module(self, module: PerceptionModule) -> None:
        """注册新的感官模块（如听觉、触觉、机械臂本体感知）。"""
        self.registry.register(module, replace=True)

    def upgrade_module(self, modality: str, new_module: PerceptionModule) -> str:
        """在不改动核心推理引擎的前提下，热升级指定模态模块（例如 Vision V1 -> V2）。"""
        return self.registry.upgrade(modality, new_module)

    def perceive_parallel(
        self, sensory_inputs: Dict[str, Any]
    ) -> StructuredSceneGraph:
        """
        并行调用所有输入对应的感官模块，将多源原始信号翻译并融合为统一 StructuredSceneGraph。
        """
        active = self.registry.active_modules()
        valid_tasks = [
            (modality, active[modality], raw_data)
            for modality, raw_data in sensory_inputs.items()
            if modality in active and raw_data is not None
        ]
        if not valid_tasks:
            raise ValueError(
                f"未找到匹配的已注册感官模块。输入模态: {list(sensory_inputs.keys())}, "
                f"已注册: {list(active.keys())}"
            )

        with ThreadPoolExecutor(max_workers=len(valid_tasks)) as pool:
            futures = [
                pool.submit(mod.encode_to_scene_graph, data)
                for _, mod, data in valid_tasks
            ]
            graphs = [f.result() for f in futures]

        merged = graphs[0]
        for g in graphs[1:]:
            merged = merged.merge(g)

        return self.core.bind_cross_modal_relations(merged)

    def run(
        self,
        sensory_inputs: Dict[str, Any],
        query: QueryConstraint,
        parallel: bool = True,
        simulated_work_steps: int = 1200,
    ) -> Dict[str, Any]:
        """
        执行完整闭环：
        1. 模块化并行感知（原始数据 -> 结构化场景图）
        2. 并行关系注意力与 G 路分支推演
        3. RCVS 约束与局部线性微扰自验证 + GRPO 组相对优势排序
        """
        t0 = time.perf_counter()
        scene_graph = self.perceive_parallel(sensory_inputs)
        candidates, rollout_ms = self.core.rollout_hypotheses(
            scene_graph=scene_graph,
            query=query,
            parallel=parallel,
            simulated_work_steps=simulated_work_steps,
        )
        report: VerificationReport = self.verifier.verify_group_parallel(
            candidates=candidates,
            scene_graph=scene_graph,
            query=query,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "scene_graph": scene_graph,
            "verification_report": report,
            "active_modules": self.registry.status_summary(),
            "metrics": {
                "parallel_mode": parallel,
                "rollout_time_ms": round(rollout_ms, 3),
                "total_time_ms": round(total_ms, 3),
            },
        }
