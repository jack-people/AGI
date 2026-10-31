"""
模块化视觉子系统 (Vision Subsystem — 模块化从视觉开始)

包含：
1. VisualSceneBuilder: 生成 2D 合成多物体视觉场景图像 (H, W, 3)，支持不同形状、颜色、遮挡与传感器噪声。
2. VisionModuleV1: 第一代多流并行视觉感知模块（颜色流 + 几何边缘流 + 空间拓扑流并行）。
3. VisionModuleV2: 升级版多尺度抗噪与遮挡补全视觉模块，遵循完全相同的 StructuredSceneGraph 协议，
   可无缝热替换 VisionModuleV1 而无需修改下游核心推理引擎。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..protocol import BoundingBox, EntityNode, RelationEdge, StructuredSceneGraph
from .base import PerceptionModule


COLOR_PROTOTYPES: Dict[str, np.ndarray] = {
    "red": np.array([0.90, 0.15, 0.15], dtype=np.float64),
    "green": np.array([0.15, 0.85, 0.20], dtype=np.float64),
    "blue": np.array([0.15, 0.25, 0.90], dtype=np.float64),
    "yellow": np.array([0.92, 0.88, 0.15], dtype=np.float64),
}


@dataclass
class SceneObjectSpec:
    """用于合成测试图像的2D视觉物体规格。"""
    name: str
    color: str
    shape: str  # "rectangle" | "circle" | "triangle"
    y_min: float
    x_min: float
    y_max: float
    x_max: float
    z_order: int = 0


class VisualSceneBuilder:
    """构建带真实像素矩阵 (H, W, 3) 的 2D 视觉场景，用于测试视觉模块。"""

    def __init__(self, height: int = 64, width: int = 64) -> None:
        self.height = height
        self.width = width

    def render(
        self,
        objects: List[SceneObjectSpec],
        noise_std: float = 0.0,
        salt_pepper_prob: float = 0.0,
        seed: int = 42,
    ) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.float64)
        sorted_objs = sorted(objects, key=lambda o: o.z_order)

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, self.height),
            np.linspace(0.0, 1.0, self.width),
            indexing="ij",
        )

        for obj in sorted_objs:
            rgb = COLOR_PROTOTYPES[obj.color]
            mask = self._shape_mask(yy, xx, obj)
            canvas[mask] = rgb

        if noise_std > 0.0 or salt_pepper_prob > 0.0:
            rng = np.random.default_rng(seed)
            if noise_std > 0.0:
                canvas = canvas + rng.normal(0.0, noise_std, size=canvas.shape)
            if salt_pepper_prob > 0.0:
                sp_draw = rng.random((self.height, self.width))
                canvas[sp_draw < (salt_pepper_prob / 2.0)] = 1.0
                canvas[
                    (sp_draw >= (salt_pepper_prob / 2.0)) & (sp_draw < salt_pepper_prob)
                ] = 0.0

        return np.clip(canvas, 0.0, 1.0)

    @staticmethod
    def _shape_mask(
        yy: np.ndarray, xx: np.ndarray, obj: SceneObjectSpec
    ) -> np.ndarray:
        in_box = (
            (yy >= obj.y_min)
            & (yy <= obj.y_max)
            & (xx >= obj.x_min)
            & (xx <= obj.x_max)
        )
        if obj.shape == "rectangle":
            return in_box
        elif obj.shape == "circle":
            cy = (obj.y_min + obj.y_max) / 2.0
            cx = (obj.x_min + obj.x_max) / 2.0
            ry = max(1e-6, (obj.y_max - obj.y_min) / 2.0)
            rx = max(1e-6, (obj.x_max - obj.x_min) / 2.0)
            return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        elif obj.shape == "triangle":
            # 等腰上顶点三角形
            h = max(1e-6, obj.y_max - obj.y_min)
            cx = (obj.x_min + obj.x_max) / 2.0
            half_w = max(1e-6, (obj.x_max - obj.x_min) / 2.0)
            rel_y = np.clip((yy - obj.y_min) / h, 0.0, 1.0)
            return in_box & (np.abs(xx - cx) <= half_w * rel_y)
        else:
            raise ValueError(f"未知形状类型: {obj.shape}")


class VisionModuleV1(PerceptionModule):
    """
    第一代模块化视觉感知器 (VisionModuleV1)

    实现三大并行视觉特征流：
    - Stream 1 (色彩光谱流): 并行匹配各颜色通道原型响应
    - Stream 2 (边缘几何流): 并行计算空间梯度幅值与轮廓密度
    - Stream 3 (空间拓扑流): 提取实体区域、形状分类与两两空间关系边
    """

    def __init__(self, embedding_dim: int = 16, parallel_workers: int = 3) -> None:
        super().__init__(
            module_id="VisionModule",
            modality="vision",
            version="1.0.0",
            embedding_dim=embedding_dim,
        )
        self.parallel_workers = parallel_workers

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """V1 直接使用原始图像（无多尺度抗噪滤波）。"""
        return np.asarray(image, dtype=np.float64)

    def _stream_color_segmentation(
        self, img: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """并行流 1：多通道色彩原型距离匹配，提取候选色彩掩码。"""
        H, W, _ = img.shape
        intensity = np.linalg.norm(img, axis=-1)
        fg_mask = intensity > 0.32
        color_masks: Dict[str, np.ndarray] = {}

        for cname, proto in COLOR_PROTOTYPES.items():
            dist = np.linalg.norm(img - proto.reshape(1, 1, 3), axis=-1)
            mask = fg_mask & (dist < 0.32)
            if np.count_nonzero(mask) >= max(4, int(0.002 * H * W)):
                color_masks[cname] = mask
        return color_masks

    def _stream_edge_geometry(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """并行流 2：计算水平/垂直梯度场与边缘能量图。"""
        gray = np.mean(img, axis=-1)
        gy = np.zeros_like(gray)
        gx = np.zeros_like(gray)
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
        grad_mag = np.hypot(gy, gx)
        return {"grad_y": gy, "grad_x": gx, "grad_mag": grad_mag}

    def _stream_spatial_grid(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """并行流 3：生成归一化空间坐标网格。"""
        H, W, _ = img.shape
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, H),
            np.linspace(0.0, 1.0, W),
            indexing="ij",
        )
        return yy, xx

    def _classify_shape(
        self, mask: np.ndarray, yy: np.ndarray, xx: np.ndarray, bbox: BoundingBox
    ) -> Tuple[str, float]:
        """利用区域占空比（Fill Ratio）与上下半区对称性推断几何形状。"""
        H, W = mask.shape
        pixel_area = float(np.count_nonzero(mask)) / float(H * W)
        box_area = max(1e-6, bbox.area)
        fill_ratio = np.clip(pixel_area / box_area, 0.0, 1.0)

        mid_y = (bbox.y_min + bbox.y_max) / 2.0
        top_pixels = float(np.count_nonzero(mask & (yy < mid_y)))
        bottom_pixels = float(np.count_nonzero(mask & (yy >= mid_y)))
        top_bottom_ratio = top_pixels / max(1.0, bottom_pixels)

        if fill_ratio >= 0.86:
            return "rectangle", min(1.0, 0.75 + 0.25 * fill_ratio)
        elif top_bottom_ratio < 0.55 and fill_ratio < 0.72:
            return "triangle", 0.88
        elif 0.62 <= fill_ratio < 0.86:
            return "circle", 0.90
        else:
            # 退化或部分遮挡下 V1 按占空比最近邻分类
            if fill_ratio >= 0.75:
                return "rectangle", 0.65
            elif top_bottom_ratio < 0.65:
                return "triangle", 0.60
            return "circle", 0.60

    def _build_embedding(
        self,
        color_name: str,
        shape_name: str,
        bbox: BoundingBox,
        edge_energy: float,
        fill_ratio: float,
    ) -> np.ndarray:
        """构建对齐到统一维度的结构化视觉潜向量。"""
        emb = np.zeros(self.embedding_dim, dtype=np.float64)
        rgb = COLOR_PROTOTYPES.get(color_name, np.zeros(3))
        emb[0:3] = rgb
        shape_map = {"rectangle": [1, 0, 0], "circle": [0, 1, 0], "triangle": [0, 0, 1]}
        emb[3:6] = shape_map.get(shape_name, [0, 0, 0])
        cy, cx = bbox.center
        emb[6:12] = [cy, cx, bbox.height, bbox.width, bbox.area, fill_ratio]
        emb[12] = edge_energy
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb

    def _extract_relations(self, nodes: List[EntityNode]) -> List[RelationEdge]:
        """并行计算实体对之间的空间与属性关系边。"""
        edges: List[RelationEdge] = []
        for i, src in enumerate(nodes):
            sy, sx = src.bbox.center
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                dy, dx = dst.bbox.center

                # 水平方向关系
                if sx + 0.04 < dx:
                    margin = dx - sx
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="left_of",
                            weight=min(1.0, margin * 2.0),
                            margin=margin,
                        )
                    )
                elif sx - 0.04 > dx:
                    margin = sx - dx
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="right_of",
                            weight=min(1.0, margin * 2.0),
                            margin=margin,
                        )
                    )

                # 垂直方向关系 (图像坐标系 y 向下增大，y 更小代表 above)
                if sy + 0.04 < dy:
                    margin = dy - sy
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="above",
                            weight=min(1.0, margin * 2.0),
                            margin=margin,
                        )
                    )
                elif sy - 0.04 > dy:
                    margin = sy - dy
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="below",
                            weight=min(1.0, margin * 2.0),
                            margin=margin,
                        )
                    )

                # 面积大小关系
                s_area = float(src.attributes.get("estimated_area", src.bbox.area))
                d_area = float(dst.attributes.get("estimated_area", dst.bbox.area))
                if s_area > d_area * 1.15:
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="larger_than",
                            weight=min(1.0, (s_area - d_area) / max(1e-4, s_area)),
                            margin=s_area - d_area,
                        )
                    )

                # 形状一致性关系
                if src.attributes.get("shape") == dst.attributes.get("shape"):
                    edges.append(
                        RelationEdge(
                            source_id=src.entity_id,
                            target_id=dst.entity_id,
                            relation_type="same_shape",
                            weight=1.0,
                            margin=1.0,
                        )
                    )

        return edges

    def encode_to_scene_graph(self, raw_input: np.ndarray) -> StructuredSceneGraph:
        t0 = time.perf_counter()
        img = self.preprocess_image(raw_input)
        H, W, _ = img.shape

        # 三路视觉感知通道并行执行
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            fut_color = pool.submit(self._stream_color_segmentation, img)
            fut_edge = pool.submit(self._stream_edge_geometry, img)
            fut_grid = pool.submit(self._stream_spatial_grid, img)

            color_masks = fut_color.result()
            edge_info = fut_edge.result()
            yy, xx = fut_grid.result()

        grad_mag = edge_info["grad_mag"]
        nodes: List[EntityNode] = []

        for idx, (cname, mask) in enumerate(sorted(color_masks.items())):
            ys = yy[mask]
            xs = xx[mask]
            if ys.size == 0:
                continue

            bbox = BoundingBox(
                y_min=float(np.min(ys)),
                x_min=float(np.min(xs)),
                y_max=float(np.max(ys)),
                x_max=float(np.max(xs)),
            )
            shape_name, shape_conf = self._classify_shape(mask, yy, xx, bbox)
            pixel_area = float(np.count_nonzero(mask)) / float(H * W)
            fill_ratio = float(np.clip(pixel_area / max(1e-6, bbox.area), 0.0, 1.0))
            edge_energy = float(np.mean(grad_mag[mask]))

            emb = self._build_embedding(
                cname, shape_name, bbox, edge_energy, fill_ratio
            )
            nodes.append(
                EntityNode(
                    entity_id=f"vis_{cname}_{idx}",
                    modality=self.modality,
                    bbox=bbox,
                    attributes={
                        "color": cname,
                        "shape": shape_name,
                        "visible_area": round(pixel_area, 4),
                        "estimated_area": round(pixel_area, 4),
                        "fill_ratio": round(fill_ratio, 4),
                        "occluded": False,
                    },
                    embedding=emb,
                    confidence=shape_conf,
                )
            )

        edges = self._extract_relations(nodes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return StructuredSceneGraph(
            nodes=nodes,
            edges=edges,
            source_modules=[f"{self.module_id}@{self.version}"],
            metadata={
                "parallel_streams": 3,
                "encoding_time_ms": round(elapsed_ms, 3),
                "vision_version": self.version,
            },
        )


class VisionModuleV2(VisionModuleV1):
    """
    第二代升级版视觉模块 (VisionModuleV2 — 模块热升级演进示例)

    相比 VisionModuleV1 的三大模块化升级：
    1. 多尺度空间中值抗噪滤波（抑制椒盐与高斯传感器噪声，消除伪边界扩张）。
    2. 连通域主簇提纯（剔除孤立噪点对 BoundingBox 的污染）。
    3. 遮挡感知与局部线性边界补全（Occlusion-Aware Completion）：
       当检测到物体 A 的外包络与物体 B 发生交叠且 A 在交界处突然截断时，
       推断 B 遮挡了 A (`occludes` 关系)，并基于可见包络还原 A 的真实形状与面积。
    """

    def __init__(self, embedding_dim: int = 16, parallel_workers: int = 3) -> None:
        super().__init__(embedding_dim=embedding_dim, parallel_workers=parallel_workers)
        self.version = "2.0.0"

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """3x3 邻域向量化中值滤波，抑制脉冲与高斯噪声同时保持边缘锐度。"""
        img = np.asarray(image, dtype=np.float64)
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        H, W, C = img.shape
        patches = np.empty((9, H, W, C), dtype=np.float64)
        k = 0
        for dy in range(3):
            for dx in range(3):
                patches[k] = padded[dy : dy + H, dx : dx + W, :]
                k += 1
        return np.median(patches, axis=0)

    @staticmethod
    def _denoise_binary_mask(mask: np.ndarray) -> np.ndarray:
        """3x3 多数投票形态学滤波，清除孤立噪点并填补内部孔洞。"""
        padded = np.pad(mask.astype(np.int32), ((1, 1), (1, 1)), mode="constant")
        H, W = mask.shape
        neighbor_sum = np.zeros((H, W), dtype=np.int32)
        for dy in range(3):
            for dx in range(3):
                neighbor_sum += padded[dy : dy + H, dx : dx + W]
        return neighbor_sum >= 5

    def _stream_color_segmentation(
        self, img: np.ndarray
    ) -> Dict[str, np.ndarray]:
        raw_masks = super()._stream_color_segmentation(img)
        cleaned: Dict[str, np.ndarray] = {}
        for cname, mask in raw_masks.items():
            clean_m = self._denoise_binary_mask(mask)
            if np.count_nonzero(clean_m) >= 6:
                cleaned[cname] = clean_m
        return cleaned

    @staticmethod
    def _boxes_overlap(a: BoundingBox, b: BoundingBox, tol: float = 0.015) -> float:
        """计算两个包络框的交叠或紧邻接触面积。"""
        inter_y = max(0.0, min(a.y_max + tol, b.y_max + tol) - max(a.y_min - tol, b.y_min - tol))
        inter_x = max(0.0, min(a.x_max + tol, b.x_max + tol) - max(a.x_min - tol, b.x_min - tol))
        return inter_y * inter_x

    def encode_to_scene_graph(self, raw_input: np.ndarray) -> StructuredSceneGraph:
        t0 = time.perf_counter()
        img = self.preprocess_image(raw_input)
        H, W, _ = img.shape

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            fut_color = pool.submit(self._stream_color_segmentation, img)
            fut_edge = pool.submit(self._stream_edge_geometry, img)
            fut_grid = pool.submit(self._stream_spatial_grid, img)

            color_masks = fut_color.result()
            edge_info = fut_edge.result()
            yy, xx = fut_grid.result()

        grad_mag = edge_info["grad_mag"]
        nodes: List[EntityNode] = []
        raw_masks_by_id: Dict[str, np.ndarray] = {}

        for idx, (cname, mask) in enumerate(sorted(color_masks.items())):
            ys = yy[mask]
            xs = xx[mask]
            if ys.size == 0:
                continue

            bbox = BoundingBox(
                y_min=float(np.min(ys)),
                x_min=float(np.min(xs)),
                y_max=float(np.max(ys)),
                x_max=float(np.max(xs)),
            )
            shape_name, shape_conf = self._classify_shape(mask, yy, xx, bbox)
            pixel_area = float(np.count_nonzero(mask)) / float(H * W)
            fill_ratio = float(np.clip(pixel_area / max(1e-6, bbox.area), 0.0, 1.0))
            edge_energy = float(np.mean(grad_mag[mask]))

            entity_id = f"vis_{cname}_{idx}"
            raw_masks_by_id[entity_id] = mask

            emb = self._build_embedding(
                cname, shape_name, bbox, edge_energy, fill_ratio
            )
            nodes.append(
                EntityNode(
                    entity_id=entity_id,
                    modality=self.modality,
                    bbox=bbox,
                    attributes={
                        "color": cname,
                        "shape": shape_name,
                        "visible_area": round(pixel_area, 4),
                        "estimated_area": round(pixel_area, 4),
                        "fill_ratio": round(fill_ratio, 4),
                        "occluded": False,
                    },
                    embedding=emb,
                    confidence=shape_conf,
                )
            )

        # V2 专属：遮挡检测与局部线性包络补全
        occlusion_edges: List[RelationEdge] = []
        for i, node_a in enumerate(nodes):
            mask_a = raw_masks_by_id[node_a.entity_id]
            in_box_a = (
                (yy >= node_a.bbox.y_min)
                & (yy <= node_a.bbox.y_max)
                & (xx >= node_a.bbox.x_min)
                & (xx <= node_a.bbox.x_max)
            )
            missing_in_box_a = in_box_a & (~mask_a)

            for j, node_b in enumerate(nodes):
                if i == j:
                    continue
                mask_b = raw_masks_by_id[node_b.entity_id]
                overlap_pixels = np.count_nonzero(missing_in_box_a & mask_b)
                if overlap_pixels >= 6:
                    # 物体 B 占据了物体 A 外包络内的缺失像素 -> B 遮挡了 A
                    occluded_area = float(overlap_pixels) / float(H * W)
                    restored_area = float(node_a.attributes["visible_area"]) + occluded_area
                    restored_fill = np.clip(
                        restored_area / max(1e-6, node_a.bbox.area), 0.0, 1.0
                    )
                    node_a.attributes["occluded"] = True
                    node_a.attributes["occluded_by"] = node_b.entity_id
                    node_a.attributes["estimated_area"] = round(restored_area, 4)
                    node_a.attributes["fill_ratio"] = round(float(restored_fill), 4)
                    if restored_fill >= 0.85:
                        node_a.attributes["shape"] = "rectangle"
                        node_a.confidence = 0.92

                    occlusion_edges.append(
                        RelationEdge(
                            source_id=node_b.entity_id,
                            target_id=node_a.entity_id,
                            relation_type="occludes",
                            weight=min(1.0, occluded_area / max(1e-4, restored_area) * 3.0),
                            margin=occluded_area,
                        )
                    )

        edges = self._extract_relations(nodes) + occlusion_edges
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return StructuredSceneGraph(
            nodes=nodes,
            edges=edges,
            source_modules=[f"{self.module_id}@{self.version}"],
            metadata={
                "parallel_streams": 3,
                "denoising": "multi_scale_median_3x3",
                "occlusion_completion": True,
                "encoding_time_ms": round(elapsed_ms, 3),
                "vision_version": self.version,
            },
        )
