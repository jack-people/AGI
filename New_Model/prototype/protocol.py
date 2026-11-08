"""
统一结构化表征协议 (Structured Latent Protocol, SLP)

将“数据 -> 结构化数据”固化为所有感官模块、并行推理核心与自验证模块之间的标准总线契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class VerificationStatus(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BoundingBox:
    """归一化或像素级二维空间包络 (y_min, x_min, y_max, x_max)。"""
    y_min: float
    x_min: float
    y_max: float
    x_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.y_min + self.y_max) / 2.0, (self.x_min + self.x_max) / 2.0)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def area(self) -> float:
        return self.height * self.width


@dataclass
class EntityNode:
    """结构化实体节点：由感知模块（如视觉模块）翻译产出。"""
    entity_id: str
    modality: str
    bbox: BoundingBox
    attributes: Dict[str, Any]
    embedding: np.ndarray
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "modality": self.modality,
            "bbox": {
                "y_min": round(self.bbox.y_min, 3),
                "x_min": round(self.bbox.x_min, 3),
                "y_max": round(self.bbox.y_max, 3),
                "x_max": round(self.bbox.x_max, 3),
                "center": tuple(round(c, 3) for c in self.bbox.center),
            },
            "attributes": self.attributes,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class RelationEdge:
    """结构化关系边：表达两个实体之间的空间、属性或因果联系。"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    margin: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type,
            "weight": round(self.weight, 4),
            "margin": round(self.margin, 4),
        }


@dataclass
class StructuredSceneGraph:
    """
    统一结构化场景关系图（数据 -> 结构化数据的核心载体）。
    无论上游是 VisionModuleV1、VisionModuleV2 还是未来的触觉/听觉模块，
    均将原始信号翻译为该结构，供下游并行推理核心消费。
    """
    nodes: List[EntityNode] = field(default_factory=list)
    edges: List[RelationEdge] = field(default_factory=list)
    source_modules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, entity_id: str) -> Optional[EntityNode]:
        for node in self.nodes:
            if node.entity_id == entity_id:
                return node
        return None

    def find_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[RelationEdge]:
        results: List[RelationEdge] = []
        for edge in self.edges:
            if source_id is not None and edge.source_id != source_id:
                continue
            if target_id is not None and edge.target_id != target_id:
                continue
            if relation_type is not None and edge.relation_type != relation_type:
                continue
            results.append(edge)
        return results

    def merge(self, other: "StructuredSceneGraph") -> "StructuredSceneGraph":
        """合并来自多个并行感官模块的结构化场景图。"""
        merged_nodes = list(self.nodes) + list(other.nodes)
        merged_edges = list(self.edges) + list(other.edges)
        merged_sources = list(dict.fromkeys(self.source_modules + other.source_modules))
        merged_meta = {**self.metadata, **other.metadata}
        return StructuredSceneGraph(
            nodes=merged_nodes,
            edges=merged_edges,
            source_modules=merged_sources,
            metadata=merged_meta,
        )

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "source_modules": self.source_modules,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class QueryConstraint:
    """问题需求与验证约束定义（对应方案一 RCVS）。"""
    goal: str
    target_attributes: Dict[str, Any] = field(default_factory=dict)
    required_relations: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    error_tolerance: float = 0.15
    perturbation_radius: float = 0.03


@dataclass
class HypothesisCandidate:
    """并行推理分支产出的单个候选假设。"""
    branch_id: int
    strategy_name: str
    answer: Any
    matched_entities: List[str]
    reasoning_trace: List[str]
    raw_confidence: float
    constraint_score: float = 0.0
    stability_score: float = 0.0
    total_reward: float = 0.0
    grpo_advantage: float = 0.0
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    verification_notes: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """自验证模块输出的完整报告。"""
    status: VerificationStatus
    selected_hypothesis: Optional[HypothesisCandidate]
    all_candidates: List[HypothesisCandidate]
    local_linear_error_bound: float
    applicable_domain: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
