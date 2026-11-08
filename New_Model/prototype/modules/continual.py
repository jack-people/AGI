"""
第四阶段的经验回放入口。

收集「严格分支没答对，但其他并行分支经过 RCVS 改对了」的查询。
这些记录留给后续按周增量跑 GRPO。当前不改已训练的权重。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..protocol import QueryConstraint, VerificationReport, VerificationStatus


class ReplayBuffer:
    """内存中的困难样本池，可选地追加写入 jsonl。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self.records: List[Dict[str, Any]] = []

    def observe(self, query: QueryConstraint, report: VerificationReport) -> bool:
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
        record = {
            "goal": query.goal,
            "target_attributes": dict(query.target_attributes),
            "required_relations": [
                [relation, dict(target)] for relation, target in query.required_relations
            ],
            "selected_strategy": selected.strategy_name,
            "strict_matched": strict_ids,
            "selected_matched": selected_ids,
            "selected_reward": selected.total_reward,
            "strict_reward": strict.total_reward,
        }
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
