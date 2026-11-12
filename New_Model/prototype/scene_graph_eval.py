"""
用 Visual Genome 的真实关系标注，检查开放词汇场景图上的关系推理。

检测框走 VisionModuleV3 的同一条接口（和 OWL-ViT / SAM 2 写出的框相同），
再交给原来的并行分支和 RCVS。这里查的是标注里的方向关系：
left of、right of、above、on top of、under。wearing、on、has 这类语义关系不计入分数。

标注来自 Visual Genome（Krishna et al., IJCV 2017），缓存用 Hugging Face 上的
jn12/VisualGenome 镜像。原始 json 不进仓库。

    python3 -m New_Model.prototype.scene_graph_eval
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np

from .engine import NextGenModelEngine
from .modules.vision_v3 import Detection, VisionModuleV3
from .protocol import QueryConstraint, VerificationStatus


CACHE_DIR = Path(__file__).resolve().parent / "data" / "visual_genome"
REPORT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "visual_genome_relation_report.json"
)
RELATION_URL = (
    "https://huggingface.co/datasets/jn12/VisualGenome/resolve/main/relationships.json.zip"
)
IMAGE_URL = (
    "https://huggingface.co/datasets/jn12/VisualGenome/resolve/main/image_data.json.zip"
)

SPATIAL_PREDICATES = {
    "left of": "left_of",
    "to left of": "left_of",
    "to the left of": "left_of",
    "on left of": "left_of",
    "on the left of": "left_of",
    "on left side of": "left_of",
    "on the left side of": "left_of",
    "on left": "left_of",
    "right of": "right_of",
    "to right of": "right_of",
    "to the right of": "right_of",
    "on right of": "right_of",
    "on the right of": "right_of",
    "on right side of": "right_of",
    "on the right side of": "right_of",
    "on right": "right_of",
    "above": "above",
    "are above": "above",
    "on top of": "above",
    "are on top of": "above",
    "on top": "above",
    "sitting on top of": "above",
    "lying on top of": "above",
    "standing on top of": "above",
    "placed on top of": "above",
    "atop": "above",
    "over": "above",
    "hanging over": "above",
    "hanging above": "above",
    "below": "below",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "standing under": "below",
    "sitting under": "below",
    "standing underneath": "below",
}


def canonical_spatial(predicate: str) -> Optional[str]:
    """把 Visual Genome 的方向谓词收成模型里的四种关系。其他谓词返回空。"""
    text = " ".join(str(predicate).lower().replace("_", " ").split())
    for tail in (" a", " an", " the"):
        if text.endswith(tail):
            text = text[: -len(tail)]
    return SPATIAL_PREDICATES.get(text)


def _object_name(node: Dict[str, Any]) -> str:
    if node.get("name"):
        return " ".join(str(node["name"]).lower().split())
    names = node.get("names") or []
    if names:
        return " ".join(str(names[0]).lower().split())
    return ""


def _box(node: Dict[str, Any], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    try:
        x = float(node["x"])
        y = float(node["y"])
        w = float(node["w"])
        h = float(node["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or w <= 1 or h <= 1:
        return None
    y_min = float(np.clip(y / height, 0.0, 1.0))
    x_min = float(np.clip(x / width, 0.0, 1.0))
    y_max = float(np.clip((y + h) / height, 0.0, 1.0))
    x_max = float(np.clip((x + w) / width, 0.0, 1.0))
    if y_max - y_min < 1e-4 or x_max - x_min < 1e-4:
        return None
    return y_min, x_min, y_max, x_max


def cases_from_image(
    record: Dict[str, Any], width: int, height: int
) -> Tuple[List[Dict[str, Any]], int]:
    """从一个图像的关系列表抽出可检查的方向关系。返回 (样本, 跳过的非方向关系数)。"""
    skipped = 0
    objects: Dict[int, Dict[str, Any]] = {}
    pending: List[Tuple[str, int, int, str, str]] = []
    for relation in record.get("relationships") or []:
        kind = canonical_spatial(str(relation.get("predicate", "")))
        subject = relation.get("subject") or {}
        target = relation.get("object") or {}
        if kind is None:
            skipped += 1
            continue
        try:
            subject_id = int(subject["object_id"])
            target_id = int(target["object_id"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if subject_id == target_id:
            skipped += 1
            continue
        subject_name = _object_name(subject)
        target_name = _object_name(target)
        subject_box = _box(subject, width, height)
        target_box = _box(target, width, height)
        if not subject_name or not target_name or subject_box is None or target_box is None:
            skipped += 1
            continue
        objects.setdefault(
            subject_id, {"name": subject_name, "box": subject_box}
        )
        objects.setdefault(
            target_id, {"name": target_name, "box": target_box}
        )
        pending.append((kind, subject_id, target_id, subject_name, target_name))

    name_counts = Counter(item["name"] for item in objects.values())
    labels = {
        object_id: (
            item["name"]
            if name_counts[item["name"]] == 1
            else f"{item['name']} #{object_id}"
        )
        for object_id, item in objects.items()
    }
    detections = [
        Detection(
            labels[object_id],
            1.0,
            item["box"][0],
            item["box"][1],
            item["box"][2],
            item["box"][3],
            attributes={"vg_object_id": object_id},
        )
        for object_id, item in objects.items()
    ]
    cases = []
    for kind, subject_id, target_id, subject_name, target_name in pending:
        cases.append(
            {
                "image_id": int(record.get("image_id", -1)),
                "relation": kind,
                "predicate_subject": subject_name,
                "predicate_object": target_name,
                "subject_id": subject_id,
                "target_id": target_id,
                "subject_label": labels[subject_id],
                "target_label": labels[target_id],
                "detections": detections,
            }
        )
    return cases, skipped


def iter_json_array(handle: Any) -> Iterator[Dict[str, Any]]:
    """逐条读出一个 JSON 数组，避免把整份关系文件装进内存。"""
    decoder = json.JSONDecoder()
    buffer = ""
    opened = False
    while True:
        if not opened:
            start = buffer.find("[")
            if start >= 0:
                buffer = buffer[start + 1 :]
                opened = True
            else:
                chunk = _read_text(handle)
                if not chunk:
                    return
                buffer += chunk
                continue
        buffer = buffer.lstrip()
        if buffer.startswith("]"):
            return
        if buffer.startswith(","):
            buffer = buffer[1:]
            continue
        try:
            record, index = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = _read_text(handle)
            if not chunk:
                raise
            buffer += chunk
            continue
        if isinstance(record, dict):
            yield record
        buffer = buffer[index:]


def _read_text(handle: Any) -> str:
    chunk = handle.read(1 << 20)
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8")
    return chunk or ""


def load_image_sizes(path: Path) -> Dict[int, Tuple[int, int]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(item for item in archive.namelist() if item.endswith(".json"))
            payload = json.loads(archive.read(name))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    sizes = {}
    for item in payload:
        sizes[int(item["image_id"])] = (int(item["width"]), int(item["height"]))
    return sizes


def ensure_annotations(cache_dir: Path) -> Tuple[Path, Path]:
    """下载关系标注和图像尺寸。已有缓存时直接用。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    relations = _cached(cache_dir, "relationships.json.zip", RELATION_URL)
    images = _cached(cache_dir, "image_data.json.zip", IMAGE_URL)
    return relations, images


def _cached(cache_dir: Path, name: str, url: str) -> Path:
    path = cache_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"下载 {name}")
    urlretrieve(url, path)
    return path


def collect_spatial_cases(
    relations_path: Path,
    sizes: Dict[int, Tuple[int, int]],
    per_relation: int = 200,
) -> Dict[str, Any]:
    """扫一遍关系文件，每种方向关系最多保留 per_relation 条。"""
    kept: Dict[str, List[Dict[str, Any]]] = {
        "left_of": [],
        "right_of": [],
        "above": [],
        "below": [],
    }
    images_used = set()
    images_seen = 0
    spatial_seen = 0
    nonspatial = 0
    opener = _open_relations(relations_path)
    with opener as handle:
        for record in iter_json_array(handle):
            images_seen += 1
            size = sizes.get(int(record.get("image_id", -1)))
            if size is None:
                continue
            cases, skipped = cases_from_image(record, size[0], size[1])
            nonspatial += skipped
            if not cases:
                continue
            added = False
            for case in cases:
                bucket = kept[case["relation"]]
                if len(bucket) >= per_relation:
                    spatial_seen += 1
                    continue
                bucket.append(case)
                spatial_seen += 1
                added = True
            if added:
                images_used.add(int(record["image_id"]))
            if all(len(bucket) >= per_relation for bucket in kept.values()):
                break
            if images_seen % 20000 == 0:
                filled = {name: len(rows) for name, rows in kept.items()}
                print(f"  已扫描 {images_seen} 张，收集 {filled}")
    cases = [case for rows in kept.values() for case in rows]
    return {
        "cases": cases,
        "images_seen": images_seen,
        "images_used": len(images_used),
        "spatial_seen": spatial_seen,
        "nonspatial_skipped": nonspatial,
        "collected": {name: len(rows) for name, rows in kept.items()},
    }


def _open_relations(path: Path):
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        name = next(item for item in archive.namelist() if item.endswith(".json"))
        handle = archive.open(name)
        return _ZipText(archive, handle)
    return path.open("r", encoding="utf-8")


class _ZipText:
    def __init__(self, archive: zipfile.ZipFile, handle: Any) -> None:
        self.archive = archive
        self.handle = handle

    def __enter__(self) -> Any:
        return self.handle

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.handle.close()
        self.archive.close()


def evaluate_cases(
    cases: Iterable[Dict[str, Any]], perturbation_radius: float = 0.02
) -> Dict[str, Any]:
    """把每条真实关系送进开放词汇场景图和原来的推理器。"""
    module = VisionModuleV3(score_threshold=0.05)
    engine = NextGenModelEngine(vision_module=module, num_branches=4)
    blank = np.zeros((8, 8, 3), dtype=np.float64)
    by_relation: Dict[str, Dict[str, int]] = {}
    disagreements: List[Dict[str, Any]] = []
    rejected_agreements: List[Dict[str, Any]] = []
    geometry_agreed = 0
    reasoning_accepted = 0
    accepted_without_geometry = 0
    miss_counts = Counter()
    total = 0

    for case in cases:
        total += 1
        bucket = by_relation.setdefault(
            case["relation"],
            {"count": 0, "geometry_agreed": 0, "reasoning_accepted": 0},
        )
        bucket["count"] += 1
        outcome = _score_case(engine, blank, case, perturbation_radius)
        if outcome["geometry_agreed"]:
            geometry_agreed += 1
            bucket["geometry_agreed"] += 1
        if outcome["reasoning_accepted"]:
            reasoning_accepted += 1
            bucket["reasoning_accepted"] += 1
        if outcome["reasoning_accepted"] and not outcome["geometry_agreed"]:
            accepted_without_geometry += 1
        if not outcome["geometry_agreed"]:
            miss_counts[outcome["miss"]] += 1
        example = {
            "image_id": case["image_id"],
            "relation": case["relation"],
            "subject": case["predicate_subject"],
            "object": case["predicate_object"],
            "status": outcome["status"],
            "strategy": outcome["strategy"],
            "center_gap": outcome["center_gap"],
            "miss": outcome["miss"],
        }
        if not outcome["geometry_agreed"] and len(disagreements) < 8:
            disagreements.append(example)
        if (
            outcome["geometry_agreed"]
            and not outcome["reasoning_accepted"]
            and len(rejected_agreements) < 8
        ):
            rejected_agreements.append(example)

    def rate(numerator: int) -> float:
        return round(100.0 * numerator / max(1, total), 2)

    per_relation = {}
    for name, bucket in by_relation.items():
        count = bucket["count"]
        per_relation[name] = {
            **bucket,
            "geometry_agreement": round(100.0 * bucket["geometry_agreed"] / max(1, count), 2),
            "reasoning_accuracy": round(
                100.0 * bucket["reasoning_accepted"] / max(1, count), 2
            ),
        }
    return {
        "spatial_relations": total,
        "geometry_agreed": geometry_agreed,
        "reasoning_accepted": reasoning_accepted,
        "geometry_agreement": rate(geometry_agreed),
        "reasoning_accuracy": rate(reasoning_accepted),
        "accepted_without_geometry": accepted_without_geometry,
        "misses": dict(miss_counts),
        "by_relation": per_relation,
        "geometry_disagreements": disagreements,
        "agreed_but_rejected": rejected_agreements,
    }


def _score_case(
    engine: NextGenModelEngine,
    blank: np.ndarray,
    case: Dict[str, Any],
    perturbation_radius: float,
) -> Dict[str, Any]:
    query = QueryConstraint(
        goal=f"{case['subject_label']} {case['relation']} {case['target_label']}",
        target_attributes={"category": case["subject_label"]},
        required_relations=[(case["relation"], {"category": case["target_label"]})],
        perturbation_radius=perturbation_radius,
    )
    result = engine.run(
        {"vision": {"image": blank, "detections": case["detections"], "queries": []}},
        query,
        parallel=False,
        simulated_work_steps=0,
    )
    graph = result["scene_graph"]
    subject = _node_for(graph, case["subject_id"])
    target = _node_for(graph, case["target_id"])
    center_gap = None
    miss = ""
    geometry_agreed = False
    if subject is not None and target is not None:
        center_gap = round(_signed_gap(subject, target, case["relation"]), 4)
        geometry_agreed = any(
            edge.relation_type == case["relation"] and edge.target_id == target.entity_id
            for edge in graph.find_edges(source_id=subject.entity_id)
        )
        if not geometry_agreed:
            if center_gap < 0:
                miss = "opposite_direction"
            elif center_gap < 0.04:
                miss = "center_gap_below_0.04"
            else:
                miss = "other"
    else:
        miss = "missing_box"
    report = result["verification_report"]
    selected = report.selected_hypothesis
    reasoning_accepted = (
        report.status == VerificationStatus.PASS
        and selected is not None
        and subject is not None
        and subject.entity_id in selected.matched_entities
    )
    return {
        "geometry_agreed": geometry_agreed,
        "reasoning_accepted": reasoning_accepted,
        "status": report.status.value,
        "strategy": None if selected is None else selected.strategy_name,
        "center_gap": center_gap,
        "miss": miss,
    }


def _signed_gap(subject: Any, target: Any, relation: str) -> float:
    """正数表示主体中心已经在标注要求的那一侧。图像坐标向下为正。"""
    sy, sx = subject.bbox.center
    ty, tx = target.bbox.center
    if relation == "left_of":
        return tx - sx
    if relation == "right_of":
        return sx - tx
    if relation == "above":
        return ty - sy
    return sy - ty


def _node_for(graph: Any, object_id: int) -> Any:
    for node in graph.nodes:
        if int(node.attributes.get("vg_object_id", -1)) == object_id:
            return node
    return None


def run_visual_genome_check(
    cache_dir: Path = CACHE_DIR,
    per_relation: int = 200,
    report_path: Path = REPORT_PATH,
) -> Dict[str, Any]:
    relations_path, image_path = ensure_annotations(cache_dir)
    sizes = load_image_sizes(image_path)
    collected = collect_spatial_cases(relations_path, sizes, per_relation=per_relation)
    measured = evaluate_cases(collected["cases"])
    report = {
        "source": "Visual Genome relationships (jn12/VisualGenome mirror of the 1.4 release)",
        "license": "CC-BY-4.0",
        "citation": "Krishna et al., Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations, IJCV 2017.",
        "path": "VisionModuleV3 recorded boxes -> parallel branches -> RCVS",
        "perturbation_radius": 0.02,
        "per_relation_cap": per_relation,
        "images_seen": collected["images_seen"],
        "images_used": collected["images_used"],
        "spatial_seen_before_cap": collected["spatial_seen"],
        "nonspatial_skipped_while_scanning": collected["nonspatial_skipped"],
        "collected": collected["collected"],
        **measured,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Visual Genome 检查开放词汇后的关系推理")
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--per-relation", type=int, default=200)
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()
    report = run_visual_genome_check(
        cache_dir=Path(args.cache),
        per_relation=args.per_relation,
        report_path=Path(args.report),
    )
    print("=" * 78)
    print("  Visual Genome 方向关系检查")
    print("=" * 78)
    print(
        f"  样本 {report['spatial_relations']} 条，来自 {report['images_used']} 张图。"
        f"几何一致 {report['geometry_agreement']:.2f}%，"
        f"推理接受 {report['reasoning_accuracy']:.2f}%。"
    )
    misses = report.get("misses") or {}
    if misses:
        print(
            "  未一致："
            + "，".join(f"{name} {count}" for name, count in misses.items())
        )
    for name, bucket in report["by_relation"].items():
        print(
            f"  - {name:<10} n={bucket['count']:<4} "
            f"几何 {bucket['geometry_agreement']:>6.2f}%  "
            f"推理 {bucket['reasoning_accuracy']:>6.2f}%"
        )
    print(f"  报告: {args.report}")
    print("=" * 78)


if __name__ == "__main__":
    main()
