"""
从物体名字和框写出 on、wearing、in、has。

视觉模块仍然只提供实体和框。这里不看像素，只决定要不要加一条关系边。
余量用重叠或接触的比例，数值和左右关系一样落在 0 到 1，供原来的 RCVS 比较。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..protocol import EntityNode, RelationEdge


PERSON_NAMES = {
    "man",
    "woman",
    "person",
    "boy",
    "girl",
    "child",
    "kid",
    "baby",
    "guy",
    "lady",
    "gentleman",
    "people",
    "human",
    "rider",
    "player",
    "skater",
    "surfer",
    "skier",
    "toddler",
    "teenager",
    "teen",
    "policeman",
    "police",
    "soldier",
    "worker",
    "chef",
    "cook",
    "bride",
    "groom",
    "cowboy",
    "batter",
    "pitcher",
    "catcher",
    "skateboarder",
    "snowboarder",
    "biker",
    "cyclist",
    "motorcyclist",
    "pedestrian",
    "passenger",
    "driver",
    "he",
    "she",
}

CLOTHING_NAMES = {
    "shirt",
    "t-shirt",
    "tee",
    "hat",
    "cap",
    "jacket",
    "coat",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "shoe",
    "shoes",
    "sneaker",
    "sneakers",
    "boot",
    "boots",
    "sandal",
    "sandals",
    "glasses",
    "sunglasses",
    "helmet",
    "glove",
    "gloves",
    "sock",
    "socks",
    "skirt",
    "dress",
    "sweater",
    "hoodie",
    "sweatshirt",
    "scarf",
    "tie",
    "belt",
    "vest",
    "suit",
    "uniform",
    "jersey",
    "gown",
    "blouse",
    "watch",
    "headband",
    "necklace",
    "earring",
    "earrings",
    "mask",
    "goggles",
    "backpack",
}


def propose_semantic_edges(nodes: Sequence[EntityNode]) -> List[RelationEdge]:
    """为每一对实体补上满足框规则的语义关系。"""
    edges: List[RelationEdge] = []
    for source in nodes:
        for target in nodes:
            if source.entity_id == target.entity_id:
                continue
            edges.extend(_edges_between(source, target))
    return edges


def _edges_between(source: EntityNode, target: EntityNode) -> List[RelationEdge]:
    edges: List[RelationEdge] = []
    source_inside = _containment(source, target)
    target_inside = _containment(target, source)
    if source_inside >= 0.6 and source.bbox.area < target.bbox.area * 0.95:
        edges.append(_edge(source, target, "in", source_inside))
    if target_inside >= 0.55 and target.bbox.area < source.bbox.area * 0.85:
        edges.append(_edge(source, target, "has", target_inside))
    if _is_person(source) and _is_clothing(target) and target_inside >= 0.35:
        edges.append(_edge(source, target, "wearing", target_inside))
    on_margin = _on_margin(source, target, source_inside)
    if on_margin is not None:
        edges.append(_edge(source, target, "on", on_margin))
    return edges


def _edge(
    source: EntityNode, target: EntityNode, relation: str, margin: float
) -> RelationEdge:
    score = float(max(0.0, min(1.0, margin)))
    return RelationEdge(
        source_id=source.entity_id,
        target_id=target.entity_id,
        relation_type=relation,
        weight=score,
        margin=score,
    )


def _label(node: EntityNode) -> str:
    raw = node.attributes.get("vg_name") or node.attributes.get("category") or ""
    return str(raw).split(" #", 1)[0].strip().lower()


def _tokens(name: str) -> set:
    return {name, *name.replace("-", " ").split()}


def _is_person(node: EntityNode) -> bool:
    return bool(_tokens(_label(node)) & PERSON_NAMES)


def _is_clothing(node: EntityNode) -> bool:
    return bool(_tokens(_label(node)) & CLOTHING_NAMES)


def _overlap_len(left: float, right: float, other_left: float, other_right: float) -> float:
    return max(0.0, min(right, other_right) - max(left, other_left))


def _containment(inner: EntityNode, outer: EntityNode) -> float:
    """inner 有多少面积落在 outer 里面。"""
    height = _overlap_len(inner.bbox.y_min, inner.bbox.y_max, outer.bbox.y_min, outer.bbox.y_max)
    width = _overlap_len(inner.bbox.x_min, inner.bbox.x_max, outer.bbox.x_min, outer.bbox.x_max)
    return (height * width) / max(1e-6, inner.bbox.area)


def _on_margin(source: EntityNode, target: EntityNode, source_inside: float) -> Optional[float]:
    """source 底边贴住 target 的上半部，并且水平重叠。大部分落在里面时归 in，不写 on。"""
    if source_inside >= 0.6:
        return None
    horizontal = _overlap_len(
        source.bbox.x_min, source.bbox.x_max, target.bbox.x_min, target.bbox.x_max
    )
    horizontal_ratio = horizontal / max(1e-6, min(source.bbox.width, target.bbox.width))
    if horizontal_ratio < 0.35:
        return None
    support_top = target.bbox.y_min
    support_band = target.bbox.y_min + 0.5 * target.bbox.height
    if not (support_top - 0.03 <= source.bbox.y_max <= support_band):
        return None
    if source.bbox.center[0] >= target.bbox.center[0]:
        return None
    return horizontal_ratio
