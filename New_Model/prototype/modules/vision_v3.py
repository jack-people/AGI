"""
第四阶段视觉模块 (VisionModuleV3)

保持 SLP 总线不变：仍然输出 EntityNode 与 RelationEdge。
底层不再用固定的 12 色原型，而是接收开放词汇检测器给出的类别、框和可选掩膜。
检测器可以是外部的 SAM 2 / Grounding DINO / OWL-ViT，也可以是已经算好的检测结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Protocol, Sequence
import numpy as np

from ..protocol import BoundingBox, EntityNode, RelationEdge, StructuredSceneGraph
from .vision import SceneObjectSpec, VisionModuleV1, VisualSceneBuilder


@dataclass
class Detection:
    """一个开放词汇检测结果。坐标用 0–1 归一化；大于 1.5 时按像素解释。"""

    category: str
    score: float
    y_min: float
    x_min: float
    y_max: float
    x_max: float
    mask: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class OpenVocabDetector(Protocol):
    """检测器插件。实现 detect(image, queries) 即可热插到 VisionModuleV3。"""

    def detect(
        self, image: np.ndarray, queries: Sequence[str]
    ) -> List[Detection]:
        """返回与文本查询对应的检测框。queries 为空时由检测器自己决定输出哪些类别。"""


def _category_selected(category: str, queries: Sequence[str]) -> bool:
    if not queries:
        return True
    name = category.casefold().strip()
    for query in queries:
        key = str(query).casefold().strip()
        if key and (key == name or key in name or name in key):
            return True
    return False


class RecordedDetector:
    """把已经算好的检测结果送进视觉模块，用来接外部模型或做可重复测试。"""

    def __init__(self, detections: Sequence[Detection]) -> None:
        self.detections = list(detections)

    def detect(
        self, image: np.ndarray, queries: Sequence[str]
    ) -> List[Detection]:
        del image
        return [
            item
            for item in self.detections
            if _category_selected(item.category, queries)
        ]


class OwlVitDetector:
    """
    可选的 OWL-ViT 开放词汇后端。

    默认环境只有 numpy，不导入 torch。调用 detect 时才加载
    google/owlvit-base-patch32。权重未安装时给出明确错误，不影响色块识别。
    """

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        score_threshold: float = 0.10,
    ) -> None:
        self.model_name = model_name
        self.score_threshold = score_threshold
        self._model = None
        self._processor = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "开放词汇权重还没安装。需要时再执行 pip install torch transformers pillow 。"
                "没装之前，可以把 SAM 2 或 Grounding DINO 的检测框交给 VisionModuleV3，"
                "关系推理和自验证不用重训。"
            ) from exc
        self._processor = OwlViTProcessor.from_pretrained(self.model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def detect(
        self, image: np.ndarray, queries: Sequence[str]
    ) -> List[Detection]:
        labels = [str(item).strip() for item in queries if str(item).strip()]
        if not labels:
            return []
        self._ensure()
        torch = self._torch
        array = np.asarray(image, dtype=np.float64)
        peak = float(array.max()) if array.size else 0.0
        if peak <= 1.0:
            array = array * 255.0
        pixels = np.clip(array, 0, 255).astype(np.uint8)
        from PIL import Image

        picture = Image.fromarray(pixels, mode="RGB")
        inputs = self._processor(text=[labels], images=picture, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        height, width = pixels.shape[:2]
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.score_threshold,
            target_sizes=[(height, width)],
            text_labels=[labels],
        )[0]
        detections: List[Detection] = []
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        names = processed.get("text_labels")
        label_ids = processed["labels"]
        if hasattr(label_ids, "detach"):
            label_ids = label_ids.detach().cpu().numpy()
        for index, (box, score) in enumerate(zip(boxes, scores)):
            if names:
                category = str(names[index])
            else:
                category = labels[int(label_ids[index])]
            x0, y0, x1, y1 = [float(value) for value in box]
            detections.append(
                Detection(
                    category=category,
                    score=float(score),
                    y_min=y0 / max(1.0, height),
                    x_min=x0 / max(1.0, width),
                    y_max=y1 / max(1.0, height),
                    x_max=x1 / max(1.0, width),
                )
            )
        return _suppress_overlaps(detections)


def _box_iou(a: Detection, b: Detection) -> float:
    inter_y = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter_x = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    inter = inter_y * inter_x
    union = a.y_max - a.y_min
    union = union * (a.x_max - a.x_min) + (b.y_max - b.y_min) * (b.x_max - b.x_min) - inter
    return inter / max(1e-6, union)


def _suppress_overlaps(detections: Sequence[Detection], iou_limit: float = 0.45) -> List[Detection]:
    """同一类别里重叠过高的框只保留分数更高的一个。"""
    kept: List[Detection] = []
    for item in sorted(detections, key=lambda det: det.score, reverse=True):
        if any(
            other.category.casefold() == item.category.casefold()
            and _box_iou(item, other) >= iou_limit
            for other in kept
        ):
            continue
        kept.append(item)
        if len(kept) >= 8:
            break
    return kept


class Sam2MaskRefiner:
    """用检测框作为提示，让 SAM 2 画出物体掩膜。权重按需下载，失败时保留原来的框。"""

    def __init__(self, model_name: str = "facebook/sam2.1-hiera-tiny") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._torch = None
        self.error: Optional[str] = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SAM 2 需要 torch、transformers 和 pillow。"
            ) from exc
        self._processor = Sam2Processor.from_pretrained(self.model_name)
        self._model = Sam2Model.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def refine(
        self, image: np.ndarray, detections: Sequence[Detection]
    ) -> List[Detection]:
        items = list(detections)
        if not items:
            return items
        try:
            return self._refine_loaded(image, items)
        except Exception as exc:
            self.error = str(exc)
            return items

    def _refine_loaded(
        self, image: np.ndarray, items: List[Detection]
    ) -> List[Detection]:
        self._ensure()
        torch = self._torch
        array = np.asarray(image, dtype=np.float64)
        peak = float(array.max()) if array.size else 0.0
        if peak <= 1.0:
            array = array * 255.0
        pixels = np.clip(array, 0, 255).astype(np.uint8)
        from PIL import Image

        picture = Image.fromarray(pixels, mode="RGB")
        height, width = pixels.shape[:2]
        boxes = [
            [
                item.x_min * width,
                item.y_min * height,
                item.x_max * width,
                item.y_max * height,
            ]
            for item in items
        ]
        inputs = self._processor(
            images=picture, input_boxes=[boxes], return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        processed = self._processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"]
        )
        mask_batch = processed[0]
        if hasattr(mask_batch, "detach"):
            mask_batch = mask_batch.detach().cpu().numpy()
        else:
            mask_batch = np.asarray(mask_batch)
        refined: List[Detection] = []
        for index, item in enumerate(items):
            mask = _pick_mask(mask_batch, index, (height, width))
            refined.append(
                Detection(
                    category=item.category,
                    score=item.score,
                    y_min=item.y_min,
                    x_min=item.x_min,
                    y_max=item.y_max,
                    x_max=item.x_max,
                    mask=mask,
                    attributes=dict(item.attributes),
                )
            )
        self.error = None
        return refined


def _pick_mask(
    mask_batch: np.ndarray, index: int, shape: tuple[int, int]
) -> Optional[np.ndarray]:
    """从 SAM 输出里取出第 index 个框对应的二值掩膜。"""
    array = np.squeeze(np.asarray(mask_batch))
    if array.ndim == 2:
        chosen = array
    elif array.ndim == 3:
        if array.shape[0] == shape[0] and array.shape[1] == shape[1]:
            chosen = array[:, :, min(index, array.shape[-1] - 1)]
        else:
            chosen = array[min(index, array.shape[0] - 1)]
    elif array.ndim >= 4:
        chosen = array[min(index, array.shape[0] - 1)]
        chosen = np.squeeze(chosen)
        if chosen.ndim == 3:
            chosen = chosen[0]
    else:
        return None
    if chosen.shape != shape:
        return None
    return chosen.astype(bool)


def _stable_unit(text: str) -> np.ndarray:
    """把类别名变成稳定的 3 维特征。不使用内置 hash，避免每次进程启动都变。"""
    acc = 2166136261
    for char in text:
        acc ^= ord(char)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return np.array(
        [
            (acc & 255) / 255.0,
            ((acc >> 8) & 255) / 255.0,
            ((acc >> 16) & 255) / 255.0,
        ],
        dtype=np.float64,
    )


def _as_detection(item: Detection | Dict[str, Any]) -> Detection:
    if isinstance(item, Detection):
        return item
    box = item.get("bbox", item)
    return Detection(
        category=str(item["category"]),
        score=float(item.get("score", 1.0)),
        y_min=float(box["y_min"]),
        x_min=float(box["x_min"]),
        y_max=float(box["y_max"]),
        x_max=float(box["x_max"]),
        attributes=dict(item.get("attributes") or {}),
    )


def _normalize_span(
    y_min: float, x_min: float, y_max: float, x_max: float, height: int, width: int
) -> tuple[float, float, float, float]:
    if max(y_min, x_min, y_max, x_max) > 1.5:
        y_min = y_min / max(1.0, height)
        x_min = x_min / max(1.0, width)
        y_max = y_max / max(1.0, height)
        x_max = x_max / max(1.0, width)
    y_min, y_max = sorted((float(y_min), float(y_max)))
    x_min, x_max = sorted((float(x_min), float(x_max)))
    return (
        float(np.clip(y_min, 0.0, 1.0)),
        float(np.clip(x_min, 0.0, 1.0)),
        float(np.clip(y_max, 0.0, 1.0)),
        float(np.clip(x_max, 0.0, 1.0)),
    )


class VisionModuleV3(VisionModuleV1):
    """
    开放词汇视觉模块。

    热升级到这个版本时，下游 ParallelRelationalCore 和 RCVSSelfVerifier 保持原样。
    新查询用 category 约束，例如「杯子在书的左侧」。
    """

    def __init__(
        self,
        detector: Optional[OpenVocabDetector] = None,
        queries: Optional[Sequence[str]] = None,
        embedding_dim: int = 16,
        score_threshold: float = 0.2,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, parallel_workers=1)
        self.version = "3.0.0"
        self.detector = detector
        self.queries = list(queries or [])
        self.score_threshold = score_threshold

    def encode_to_scene_graph(self, raw_input: Any) -> StructuredSceneGraph:
        image, queries, supplied = self._unpack(raw_input)
        detections = [
            _as_detection(item)
            for item in (
                supplied
                if supplied is not None
                else self._run_detector(image, queries)
            )
        ]
        height, width = image.shape[:2]
        nodes: List[EntityNode] = []
        masks: Dict[str, np.ndarray] = {}
        for detection in detections:
            if detection.score < self.score_threshold:
                continue
            if not _category_selected(detection.category, queries):
                continue
            y_min, x_min, y_max, x_max = _normalize_span(
                detection.y_min,
                detection.x_min,
                detection.y_max,
                detection.x_max,
                height,
                width,
            )
            bbox = BoundingBox(y_min=y_min, x_min=x_min, y_max=y_max, x_max=x_max)
            if bbox.area < 1e-5:
                continue
            entity_id = f"vis_{detection.category}_{len(nodes)}"
            mask = self._mask_for(detection.mask, image.shape[:2], bbox)
            if mask is not None:
                masks[entity_id] = mask
            visible_area = float(bbox.area)
            if mask is not None:
                visible_area = float(np.count_nonzero(mask)) / float(height * width)
            attributes = {
                "category": detection.category,
                "visible_area": round(visible_area, 4),
                "estimated_area": round(float(bbox.area), 4),
                "occluded": False,
                "score": round(float(detection.score), 4),
                "has_mask": mask is not None,
            }
            attributes.update(detection.attributes)
            nodes.append(
                EntityNode(
                    entity_id=entity_id,
                    modality=self.modality,
                    bbox=bbox,
                    attributes=attributes,
                    embedding=self._embed(image, bbox, detection, mask),
                    confidence=float(np.clip(detection.score, 0.0, 1.0)),
                )
            )

        edges = self._extract_relations(nodes) + self._occlusion_edges(nodes, masks)
        return StructuredSceneGraph(
            nodes=nodes,
            edges=edges,
            source_modules=[f"{self.module_id}@{self.version}"],
            metadata={
                "vision_version": self.version,
                "open_vocabulary": True,
                "queries": list(queries),
                "detector": type(self.detector).__name__ if self.detector else "inline",
            },
        )

    def _unpack(
        self, raw_input: Any
    ) -> tuple[np.ndarray, List[str], Optional[List[Any]]]:
        if isinstance(raw_input, dict):
            image = np.asarray(raw_input["image"], dtype=np.float64)
            queries = list(raw_input.get("queries", self.queries))
            supplied = raw_input.get("detections")
            return image, queries, supplied
        return np.asarray(raw_input, dtype=np.float64), list(self.queries), None

    def _run_detector(self, image: np.ndarray, queries: Sequence[str]) -> List[Detection]:
        if self.detector is None:
            raise RuntimeError(
                "VisionModuleV3 还没有检测器。传入 detections，或安装 OWL-ViT 后使用 OwlVitDetector。"
            )
        return list(self.detector.detect(image, queries))

    @staticmethod
    def _mask_for(
        mask: Optional[np.ndarray], shape: tuple[int, int], bbox: BoundingBox
    ) -> Optional[np.ndarray]:
        if mask is None:
            return None
        array = np.asarray(mask, dtype=bool)
        if array.shape != shape:
            return None
        return array

    def _embed(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        detection: Detection,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        emb = np.zeros(self.embedding_dim, dtype=np.float64)
        height, width = image.shape[:2]
        y0 = int(math.floor(bbox.y_min * max(0, height - 1)))
        y1 = int(math.ceil(bbox.y_max * max(1, height - 1))) + 1
        x0 = int(math.floor(bbox.x_min * max(0, width - 1)))
        x1 = int(math.ceil(bbox.x_max * max(1, width - 1))) + 1
        crop = image[max(0, y0) : max(y0 + 1, y1), max(0, x0) : max(x0 + 1, x1)]
        if crop.size:
            emb[0:3] = np.mean(crop.reshape(-1, crop.shape[-1]), axis=0)[:3]
        emb[3:6] = _stable_unit(detection.category)
        cy, cx = bbox.center
        fill_ratio = 1.0
        if mask is not None:
            fill_ratio = float(np.count_nonzero(mask) / max(1.0, height * width * bbox.area))
        emb[6:12] = [cy, cx, bbox.height, bbox.width, bbox.area, fill_ratio]
        emb[12] = float(np.clip(detection.score, 0.0, 1.0))
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb

    @staticmethod
    def _occlusion_edges(
        nodes: List[EntityNode], masks: Dict[str, np.ndarray]
    ) -> List[RelationEdge]:
        edges: List[RelationEdge] = []
        for src in nodes:
            for dst in nodes:
                if src.entity_id == dst.entity_id:
                    continue
                src_mask = masks.get(src.entity_id)
                dst_mask = masks.get(dst.entity_id)
                if src_mask is not None and dst_mask is not None:
                    overlap = float(np.count_nonzero(src_mask & dst_mask))
                    base = float(np.count_nonzero(dst_mask))
                    ratio = overlap / max(1.0, base)
                    if ratio < 0.55:
                        continue
                else:
                    inter_y = max(
                        0.0,
                        min(src.bbox.y_max, dst.bbox.y_max)
                        - max(src.bbox.y_min, dst.bbox.y_min),
                    )
                    inter_x = max(
                        0.0,
                        min(src.bbox.x_max, dst.bbox.x_max)
                        - max(src.bbox.x_min, dst.bbox.x_min),
                    )
                    ratio = (inter_y * inter_x) / max(1e-6, dst.bbox.area)
                    if ratio < 0.25:
                        continue
                if float(src.confidence) < float(dst.confidence):
                    continue
                dst.attributes["occluded"] = True
                dst.attributes["occluded_by"] = src.entity_id
                edges.append(
                    RelationEdge(
                        source_id=src.entity_id,
                        target_id=dst.entity_id,
                        relation_type="occludes",
                        weight=min(1.0, ratio),
                        margin=ratio,
                    )
                )
        return edges


def build_cup_book_demo(
    height: int = 128, width: int = 128
) -> tuple[np.ndarray, List[Detection]]:
    """生成一张杯子在书左侧的示例，检测框与画面里的色块对齐。"""
    cup = SceneObjectSpec("cup", "blue", "circle", 0.22, 0.08, 0.72, 0.40)
    book = SceneObjectSpec("book", "brown", "rectangle", 0.22, 0.55, 0.72, 0.92)
    image = VisualSceneBuilder(height=height, width=width).render([cup, book])
    detections = [
        Detection(
            "杯子",
            0.97,
            cup.y_min,
            cup.x_min,
            cup.y_max,
            cup.x_max,
            attributes={"color": "blue"},
        ),
        Detection(
            "书",
            0.96,
            book.y_min,
            book.x_min,
            book.y_max,
            book.x_max,
            attributes={"color": "brown"},
        ),
    ]
    return image, detections
