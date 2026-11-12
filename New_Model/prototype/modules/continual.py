"""
经验回放入口。

收集「严格分支没答对，但其他并行分支经过 RCVS 改对了」的查询。
每条记录带上当时的场景图，供后续增量 GRPO 和清晰样本混在一起更新路由。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..protocol import (
    BoundingBox,
    EntityNode,
    QueryConstraint,
    RelationEdge,
    StructuredSceneGraph,
    VerificationReport,
    VerificationStatus,
)


def _plain(value: Any) -> Any:
    """把 numpy 标量和数组收成 json 能写的 Python 值。"""
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def snapshot_scene_graph(scene_graph: StructuredSceneGraph) -> Dict[str, Any]:
    """把场景图收成可写入 jsonl 的快照，保留节点向量和关系余量。"""
    return {
        "nodes": [
            {
                "entity_id": node.entity_id,
                "modality": node.modality,
                "bbox": {
                    "y_min": float(node.bbox.y_min),
                    "x_min": float(node.bbox.x_min),
                    "y_max": float(node.bbox.y_max),
                    "x_max": float(node.bbox.x_max),
                },
                "attributes": _plain(node.attributes),
                "embedding": _plain(node.embedding),
                "confidence": float(node.confidence),
            }
            for node in scene_graph.nodes
        ],
        "edges": [
            {
                "source": edge.source_id,
                "target": edge.target_id,
                "relation": edge.relation_type,
                "weight": float(edge.weight),
                "margin": float(edge.margin),
            }
            for edge in scene_graph.edges
        ],
        "source_modules": list(scene_graph.source_modules),
    }


def scene_graph_from_snapshot(payload: Dict[str, Any]) -> StructuredSceneGraph:
    """从回放快照还原 StructuredSceneGraph，供增量 GRPO 重新打分。"""
    nodes = [
        EntityNode(
            entity_id=str(item["entity_id"]),
            modality=str(item.get("modality", "vision")),
            bbox=BoundingBox(
                y_min=float(item["bbox"]["y_min"]),
                x_min=float(item["bbox"]["x_min"]),
                y_max=float(item["bbox"]["y_max"]),
                x_max=float(item["bbox"]["x_max"]),
            ),
            attributes=dict(item.get("attributes") or {}),
            embedding=np.asarray(item["embedding"], dtype=np.float64),
            confidence=float(item.get("confidence", 1.0)),
        )
        for item in payload.get("nodes", [])
    ]
    edges = [
        RelationEdge(
            source_id=str(item["source"]),
            target_id=str(item["target"]),
            relation_type=str(item["relation"]),
            weight=float(item.get("weight", 1.0)),
            margin=float(item.get("margin", 0.0)),
        )
        for item in payload.get("edges", [])
    ]
    return StructuredSceneGraph(
        nodes=nodes,
        edges=edges,
        source_modules=list(payload.get("source_modules") or []),
    )


class ReplayBuffer:
    """困难样本池。给出路径时会读入已有 jsonl，并只追加没见过的记录。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self.records: List[Dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self.records = load_replay_records(self.path)

    def observe(
        self,
        query: QueryConstraint,
        report: VerificationReport,
        scene_graph: Optional[StructuredSceneGraph] = None,
    ) -> bool:
        selected = report.selected_hypothesis
        if report.status != VerificationStatus.PASS or selected is None:
            return False
        strict = next(
            (
                candidate
                for candidate in report.all_candidates
                if candidate.strategy_name == "strict_high_margin"
            ),
            None,
        )
        if strict is None:
            return False
        strict_ids = list(strict.matched_entities)
        selected_ids = list(selected.matched_entities)
        if strict_ids == selected_ids:
            return False
        gt_color = ""
        if scene_graph is not None and selected_ids:
            chosen = scene_graph.get_node(selected_ids[0])
            if chosen is not None:
                gt_color = str(chosen.attributes.get("color", ""))
        record = {
            "goal": query.goal,
            "target_attributes": _plain(dict(query.target_attributes)),
            "required_relations": [
                [relation, _plain(dict(target))]
                for relation, target in query.required_relations
            ],
            "error_tolerance": float(query.error_tolerance),
            "perturbation_radius": float(query.perturbation_radius),
            "selected_strategy": selected.strategy_name,
            "strict_matched": strict_ids,
            "selected_matched": selected_ids,
            "selected_reward": float(selected.total_reward),
            "strict_reward": float(strict.total_reward),
            "gt_color": gt_color,
        }
        if scene_graph is not None:
            record["scene_graph"] = snapshot_scene_graph(scene_graph)
        if any(_record_key(old) == _record_key(record) for old in self.records):
            return False
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True


def _record_key(record: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "goal": record.get("goal"),
            "target_attributes": record.get("target_attributes"),
            "required_relations": record.get("required_relations"),
            "selected_strategy": record.get("selected_strategy"),
            "strict_matched": record.get("strict_matched"),
            "selected_matched": record.get("selected_matched"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_replay_records(path: Path) -> List[Dict[str, Any]]:
    """读 jsonl，跳过坏行，并按查询和分支去重。"""
    records: List[Dict[str, Any]] = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records
