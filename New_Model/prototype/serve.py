"""
本地推理服务：加载训练好的权重，对外提供健康检查和场景预测。

部署（在仓库根目录）：
    python3 -m New_Model.prototype.serve
    python3 -m New_Model.prototype.serve --host 0.0.0.0 --port 8765

冒烟测试（不启动端口，直接跑三条预设场景）：
    python3 -m New_Model.prototype.serve --self-check

服务起来之后，用浏览器打开：
    http://127.0.0.1:8765/

也可以直接访问：
    http://127.0.0.1:8765/health
    http://127.0.0.1:8765/predict?preset=clean
    http://127.0.0.1:8765/predict?preset=occluded
    http://127.0.0.1:8765/predict?preset=unstable

PowerShell 里 curl 是 Invoke-WebRequest 的别名，请用下面两种写法之一：
    curl.exe http://127.0.0.1:8765/predict?preset=clean
    Invoke-RestMethod http://127.0.0.1:8765/predict?preset=clean
"""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .engine import NextGenModelEngine
from .evaluate_checkpoint import CKPT_PATH, infer_with_weights
from .modules.continual import ReplayBuffer
from .modules.vision import (
    COLOR_LABELS,
    SHAPE_LABELS,
    SceneObjectSpec,
    VisionModuleV2,
    VisualSceneBuilder,
)
from .modules.vision_v3 import (
    Detection,
    OwlVitDetector,
    RecordedDetector,
    Sam2MaskRefiner,
    VisionModuleV3,
    build_cup_book_demo,
)
from .protocol import QueryConstraint, StructuredSceneGraph, VerificationStatus
from .trainer import TrainablePMVWeights


def build_presets() -> Dict[str, Dict[str, Any]]:
    """三条可直接调用的部署测试场景。"""
    builder = VisualSceneBuilder(height=64, width=64)
    return {
        "clean": {
            "expect_accepted": True,
            "image": builder.render(
                [
                    SceneObjectSpec("red_box", "red", "rectangle", 0.18, 0.12, 0.48, 0.40),
                    SceneObjectSpec("blue_ball", "blue", "circle", 0.18, 0.58, 0.48, 0.88),
                ]
            ),
            "query": QueryConstraint(
                goal="寻找蓝色圆形左侧的红色矩形",
                target_attributes={"color": "red", "shape": "rectangle"},
                required_relations=[("left_of", {"color": "blue"})],
                error_tolerance=0.15,
                perturbation_radius=0.025,
            ),
        },
        "occluded": {
            "expect_accepted": True,
            "image": builder.render(
                [
                    SceneObjectSpec(
                        "red_rect", "red", "rectangle", 0.20, 0.15, 0.60, 0.55, z_order=1
                    ),
                    SceneObjectSpec(
                        "yellow_occ", "yellow", "circle", 0.35, 0.35, 0.60, 0.60, z_order=2
                    ),
                    SceneObjectSpec(
                        "blue_circle", "blue", "circle", 0.25, 0.70, 0.55, 0.92, z_order=1
                    ),
                ],
                salt_pepper_prob=0.035,
                seed=11,
            ),
            "query": QueryConstraint(
                goal="寻找被遮挡、位于蓝色物体左侧的红色矩形",
                target_attributes={"color": "red", "shape": "rectangle"},
                required_relations=[("left_of", {"color": "blue"})],
                error_tolerance=0.15,
                perturbation_radius=0.025,
            ),
        },
        "unstable": {
            "expect_accepted": False,
            "image": builder.render(
                [
                    SceneObjectSpec("red_box", "red", "rectangle", 0.15, 0.20, 0.40, 0.40),
                    SceneObjectSpec("blue_box", "blue", "rectangle", 0.50, 0.246, 0.75, 0.446),
                ]
            ),
            "query": QueryConstraint(
                goal="判断几乎贴在一起的左右关系在微扰下是否仍然成立",
                target_attributes={"color": "red", "shape": "rectangle"},
                required_relations=[("left_of", {"color": "blue"})],
                error_tolerance=0.15,
                perturbation_radius=0.038,
            ),
        },
    }


def harvest_correction_examples(
    replay: ReplayBuffer, simulated_work_steps: int = 64
) -> int:
    """把预设场景和一条已知的窄间距纠正写进回放池。已有的记录不会重复追加。"""
    before = len(replay.records)
    engine = NextGenModelEngine(vision_module=VisionModuleV2(), num_branches=4)
    for spec in build_presets().values():
        engine.run(
            {"vision": spec["image"]},
            spec["query"],
            simulated_work_steps=simulated_work_steps,
            replay_buffer=replay,
        )
    tight = [
        Detection("甲", 0.95, 0.30, 0.370, 0.50, 0.430),
        Detection("乙", 0.95, 0.30, 0.420, 0.50, 0.470),
    ]
    blank = np.zeros((64, 64, 3), dtype=np.float64)
    NextGenModelEngine(
        vision_module=VisionModuleV3(detector=RecordedDetector(tight)),
        num_branches=4,
    ).run(
        {"vision": {"image": blank, "detections": tight}},
        QueryConstraint(
            goal="甲在乙的左侧，间距很小",
            target_attributes={"category": "甲"},
            required_relations=[("left_of", {"category": "乙"})],
            perturbation_radius=0.02,
        ),
        simulated_work_steps=simulated_work_steps,
        replay_buffer=replay,
    )
    return len(replay.records) - before


COLOR_ZH = COLOR_LABELS
SHAPE_ZH = SHAPE_LABELS
RELATION_ZH = {
    "left_of": "在",
    "above": "在",
    "occludes": "遮挡了",
}
RELATION_SUFFIX = {
    "left_of": "左侧",
    "above": "上方",
    "occludes": "",
}


def _node_label(attributes: Dict[str, Any]) -> str:
    category = str(attributes.get("category", ""))
    if category and category != "actuator":
        return category
    color = str(attributes.get("color", ""))
    shape = str(attributes.get("shape", ""))
    return f"{COLOR_ZH.get(color, color)}{SHAPE_ZH.get(shape, shape)}"


def decode_uploaded_image(payload: Dict[str, Any]) -> np.ndarray:
    """把浏览器传来的 RGBA 像素还原成 0–1 的 RGB 图像。"""
    width = int(payload["width"])
    height = int(payload["height"])
    if width < 8 or height < 8 or width > 512 or height > 512:
        raise ValueError("图片边长需要在 8 到 512 像素之间")
    raw = base64.b64decode(payload["rgba_base64"])
    expected = width * height * 4
    if len(raw) != expected:
        raise ValueError(f"像素长度应为 {expected}，实际为 {len(raw)}")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).astype(np.float64)
    rgb = pixels[..., :3] / 255.0
    alpha = pixels[..., 3:4] / 255.0
    return rgb * alpha + (1.0 - alpha)


class InferenceApp:
    def __init__(self, checkpoint: str, replay_path: Optional[str] = None) -> None:
        self.weights = TrainablePMVWeights.load(checkpoint)
        self.vision = VisionModuleV2(embedding_dim=self.weights.embedding_dim)
        self.presets = build_presets()
        self.checkpoint = checkpoint
        self.open_detector = OwlVitDetector()
        self.mask_refiner = Sam2MaskRefiner()
        if replay_path is None:
            replay_path = str(
                Path(checkpoint).resolve().parent / "replay_live.jsonl"
            )
        self.replay = ReplayBuffer(replay_path)
        self._symbolic = NextGenModelEngine(
            vision_module=VisionModuleV2(embedding_dim=self.weights.embedding_dim),
            num_branches=4,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": self.checkpoint,
            "embedding_dim": self.weights.embedding_dim,
            "presets": list(self.presets.keys()),
            "replay_records": len(self.replay.records),
            "replay_path": str(self.replay.path) if self.replay.path else "",
        }

    def _remember_correction(self, image: np.ndarray, query: QueryConstraint) -> bool:
        """用四条符号分支再看一遍。严格分支失手、其他分支改对时写入回放池。"""
        before = len(self.replay.records)
        self._symbolic.run(
            {"vision": image},
            query,
            simulated_work_steps=64,
            replay_buffer=self.replay,
        )
        return len(self.replay.records) > before

    def predict(self, preset: str) -> Dict[str, Any]:
        if preset not in self.presets:
            raise KeyError(preset)
        spec = self.presets[preset]
        result = infer_with_weights(
            self.weights, spec["image"], spec["query"], self.vision
        )
        result["preset"] = preset
        result["expect_accepted"] = spec["expect_accepted"]
        result["matches_expectation"] = result["accepted"] == spec["expect_accepted"]
        result["replay_recorded"] = self._remember_correction(spec["image"], spec["query"])
        result["replay_records"] = len(self.replay.records)
        return result

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """对一张用户提供的 RGB 图像做色块识别，返回颜色、形状和空间关系。"""
        scene = self.vision.encode_to_scene_graph(np.asarray(image, dtype=np.float64))
        payload = _describe_scene(scene, image.shape[:2])
        payload["vocabulary"] = {
            "colors": list(COLOR_ZH.values()),
            "shapes": list(SHAPE_ZH.values()),
        }
        if payload["entity_count"] == 0:
            payload["summary"] = (
                "没有识别到色块。当前视觉模块识别红、橙、黄、绿、青、蓝、紫、粉、白、灰、棕、黑，"
                "以及矩形、圆形、三角形、菱形。物体需要和画面边缘的背景颜色明显不同。"
            )
        return payload

    def open_demo(self) -> Dict[str, Any]:
        """第四阶段示例：杯子在书的左侧。检测框接入后，原推理器直接回答。"""
        image, detections = build_cup_book_demo()
        return self.recognize_open(
            image,
            queries=["杯子", "书"],
            detections=detections,
            check_relation=True,
        )

    def recognize_open(
        self,
        image: np.ndarray,
        queries: Optional[list] = None,
        detections: Optional[list] = None,
        check_relation: bool = False,
    ) -> Dict[str, Any]:
        """用开放词汇检测结果构建场景图。没有检测框时才尝试加载 OWL-ViT。"""
        query_list = [str(item).strip() for item in (queries or []) if str(item).strip()]
        parsed = [_detection_from_payload(item) for item in (detections or [])]
        masks_ready = False
        if parsed:
            found = parsed
            detector_name = "recorded"
        else:
            if not query_list:
                raise ValueError("按文字查找时需要填写查询词，例如 cup, book")
            found = self.open_detector.detect(image, query_list)
            found = self.mask_refiner.refine(image, found)
            masks_ready = any(item.mask is not None for item in found)
            detector_name = "owlvit+sam2" if masks_ready else "owlvit"
        module = VisionModuleV3(
            detector=RecordedDetector(found),
            queries=query_list,
            score_threshold=0.05,
        )
        raw: Dict[str, Any] = {
            "image": image,
            "queries": query_list,
            "detections": found,
        }
        scene = module.encode_to_scene_graph(raw)
        payload = _describe_scene(scene, image.shape[:2], found)
        payload["stage"] = 4
        payload["queries"] = query_list
        payload["detector"] = detector_name
        if payload["entity_count"] == 0:
            payload["summary"] = (
                "没有找到这些词对应的物体。查询用英文更稳，例如 cup, book。"
                "第一次查找会下载 OWL-ViT 权重。"
            )
        elif masks_ready:
            payload["summary"] += "边界已用 SAM 2 描出。"
        elif detector_name == "owlvit" and self.mask_refiner.error:
            payload["summary"] += "当前显示的是检测框。"
        if check_relation and payload["entity_count"] >= 2:
            engine = NextGenModelEngine(vision_module=module, num_branches=4)
            checked = engine.run(
                {"vision": raw},
                QueryConstraint(
                    goal="杯子在书的左侧",
                    target_attributes={"category": "杯子"},
                    required_relations=[("left_of", {"category": "书"})],
                    perturbation_radius=0.02,
                ),
                simulated_work_steps=64,
                replay_buffer=self.replay,
            )
            accepted = (
                checked["verification_report"].status == VerificationStatus.PASS
            )
            payload["accepted"] = accepted
            payload["goal"] = "杯子在书的左侧"
            if accepted:
                payload["summary"] += "关系结论：接受。"
        return payload


def _detection_from_payload(item: Any) -> Detection:
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


def _describe_scene(
    scene: StructuredSceneGraph,
    image_size: tuple,
    detections: Optional[list] = None,
) -> Dict[str, Any]:
    entities = []
    for node in scene.nodes:
        item = node.to_dict()
        item["label"] = _node_label(node.attributes)
        mask = _mask_for_node(node, detections)
        if mask is not None:
            item["mask_png_base64"] = base64.b64encode(
                _mask_png(np.asarray(mask, dtype=bool))
            ).decode("ascii")
        entities.append(item)
    relations = []
    for edge in scene.edges:
        if edge.relation_type not in RELATION_ZH:
            continue
        source = scene.get_node(edge.source_id)
        target = scene.get_node(edge.target_id)
        if source is None or target is None:
            continue
        sentence = (
            f"{_node_label(source.attributes)}"
            f"{RELATION_ZH[edge.relation_type]}"
            f"{_node_label(target.attributes)}"
            f"{RELATION_SUFFIX[edge.relation_type]}"
        )
        relations.append({**edge.to_dict(), "sentence": sentence})
    if entities:
        names = "、".join(item["label"] for item in entities)
        relation_text = "；".join(
            list(
                dict.fromkeys(
                    item["sentence"]
                    for item in relations
                    if item["relation"] in ("left_of", "above", "occludes")
                )
            )[:6]
        )
        summary = f"识别到 {len(entities)} 个物体：{names}。"
        if relation_text:
            summary += relation_text + "。"
    else:
        summary = "没有识别到物体。"
    return {
        "summary": summary,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "entity_count": len(entities),
        "entities": entities,
        "relations": relations,
        "source_modules": list(scene.source_modules),
    }


def _mask_for_node(node: Any, detections: Optional[list]) -> Optional[np.ndarray]:
    if not detections:
        return None
    best = None
    best_distance = 0.08
    cy, cx = node.bbox.center
    for item in detections:
        mask = getattr(item, "mask", None)
        if mask is None:
            continue
        iy = (float(item.y_min) + float(item.y_max)) / 2.0
        ix = (float(item.x_min) + float(item.x_max)) / 2.0
        distance = abs(iy - cy) + abs(ix - cx)
        if distance < best_distance:
            best = mask
            best_distance = distance
    return None if best is None else np.asarray(best, dtype=bool)


def _mask_png(mask: np.ndarray) -> bytes:
    """把二值掩膜画成半透明绿色 PNG，给页面叠在原图上。"""
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.float64)
    rgba[mask, 1] = 0.85
    rgba[mask, 3] = 0.45
    return png_bytes(rgba)


def png_bytes(image: np.ndarray) -> bytes:
    """把 0–1 的 RGB 或 RGBA 图像编码成 PNG。"""
    import struct
    import zlib

    array = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    height, width, channels = array.shape
    if channels not in (3, 4):
        raise ValueError("PNG 只接受 3 或 4 个通道")
    raw = (array * 255.0).astype(np.uint8)
    rows = b"".join(b"\x00" + raw[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    color_type = 6 if channels == 4 else 2
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def self_check(checkpoint: str) -> None:
    app = InferenceApp(checkpoint)
    failed = []
    for name in app.presets:
        result = app.predict(name)
        mark = "PASS" if result["matches_expectation"] else "FAIL"
        print(
            f"  [{mark}] preset={name:<10} accepted={result['accepted']} "
            f"branch={result['chosen_branch']} entity={result['selected_entity_id']} "
            f"color={result['selected_color']}"
        )
        if not result["matches_expectation"]:
            failed.append(name)
    open_demo = app.open_demo()
    open_ok = open_demo.get("accepted") is True and "杯子" in open_demo["summary"]
    print(
        f"  [{'PASS' if open_ok else 'FAIL'}] open_demo    "
        f"accepted={open_demo.get('accepted')} summary={open_demo['summary']}"
    )
    if not open_ok:
        failed.append("open_demo")
    if failed:
        raise SystemExit(f"部署冒烟未通过: {failed}")
    print("部署冒烟通过：清晰场景接受，遮挡场景接受，临界场景拒绝，开放词汇示例接受。")


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PMV 图片识别</title>
  <style>
    body { font-family: sans-serif; margin: 2rem auto; max-width: 56rem; line-height: 1.5; color: #1c1c1c; }
    canvas { max-width: 100%; background: #f3f3f3; border: 1px solid #ccc; }
    button, input { margin: 0.3rem 0.4rem 0.3rem 0; }
    #summary { font-size: 1.05rem; }
    ul { padding-left: 1.2rem; }
  </style>
</head>
<body>
  <h1>图片识别</h1>
  <p>选择一张图片。当前能识别的颜色：红、橙、黄、绿、青、蓝、紫、粉、白、灰、棕、黑。形状：矩形、圆形、三角形、菱形。同一种颜色如果分成两块，会分别框出来。</p>
  <p>
    <input id="file" type="file" accept="image/*">
    <button id="example" type="button">用内置示例图</button>
  </p>
  <h2>按文字找真实物体</h2>
  <p>先填写英文词，再选择照片，然后点「按文字查找」。OWL-ViT 负责找框，SAM 2 负责把框收成物体边界，原来的左右和上下关系接着用。查询留空时，选图仍只认颜色和形状。第一次查找会下载权重。SAM 2 还需要 torchvision。</p>
  <p>
    <input id="queries" type="text" size="42" placeholder="cup, book">
    <button id="find" type="button">按文字查找</button>
    <button id="open-demo" type="button">看杯子和书</button>
  </p>
  <canvas id="view" width="320" height="320"></canvas>
  <p id="summary">还没有图片。</p>
  <ul id="entities"></ul>
  <ul id="relations"></ul>
  <script>
    const fileInput = document.getElementById("file");
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const summary = document.getElementById("summary");
    const entityList = document.getElementById("entities");
    const relationList = document.getElementById("relations");
    const colors = {
      red: "#d22", orange: "#e80", yellow: "#ca0", green: "#2a2",
      cyan: "#1aa", blue: "#24a", purple: "#72b", pink: "#e59",
      white: "#ddd", gray: "#888", brown: "#852", black: "#111"
    };

    function drawResult(image, result) {
      const scale = Math.min(480 / image.width, 480 / image.height, 4);
      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.font = "14px sans-serif";
      const palette = ["#d22", "#24a", "#e80", "#2a2", "#72b", "#1aa"];
      (result.entities || []).forEach((entity, index) => {
        const box = entity.bbox;
        const x = box.x_min * canvas.width;
        const y = box.y_min * canvas.height;
        const w = (box.x_max - box.x_min) * canvas.width;
        const h = (box.y_max - box.y_min) * canvas.height;
        ctx.strokeStyle = colors[entity.attributes.color] || palette[index % palette.length];
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(entity.label, x + 2, Math.max(14, y - 2));
        if (entity.mask_png_base64) {
          const overlay = new Image();
          overlay.onload = () => {
            ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
          };
          overlay.src = "data:image/png;base64," + entity.mask_png_base64;
        }
      });
      summary.textContent = result.summary || "";
      entityList.replaceChildren(...(result.entities || []).map((entity) => {
        const item = document.createElement("li");
        item.textContent = entity.label + "，置信度 " + entity.confidence;
        return item;
      }));
      const shown = (result.relations || []).filter((item) =>
        ["left_of", "above", "occludes"].includes(item.relation)
      );
      relationList.replaceChildren(...shown.map((item) => {
        const row = document.createElement("li");
        row.textContent = item.sentence;
        return row;
      }));
    }

    function queryList() {
      return document.getElementById("queries").value
        .split(/[,，]/)
        .map((item) => item.trim())
        .filter(Boolean);
    }

    async function recognizePixels(width, height, rgba, sourceImage, queries) {
      summary.textContent = queries.length
        ? "按文字查找中。第一次会下载 OWL-ViT 和 SAM 2 权重，请等待。"
        : "识别中…";
      const bytes = new Uint8Array(rgba);
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      const payload = {
        width,
        height,
        rgba_base64: btoa(binary)
      };
      if (queries.length) payload.queries = queries;
      const response = await fetch(queries.length ? "/recognize_open" : "/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        summary.textContent = result.error || "识别失败";
        return;
      }
      drawResult(sourceImage, result);
    }

    async function recognizeFile(file, queries) {
      const image = new Image();
      image.src = URL.createObjectURL(file);
      await image.decode();
      const maxSide = queries.length ? 512 : 256;
      const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
      const width = Math.max(8, Math.round(image.width * scale));
      const height = Math.max(8, Math.round(image.height * scale));
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      const scratchCtx = scratch.getContext("2d");
      scratchCtx.drawImage(image, 0, 0, width, height);
      const rgba = scratchCtx.getImageData(0, 0, width, height).data;
      await recognizePixels(width, height, rgba, image, queries);
      URL.revokeObjectURL(image.src);
    }

    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) recognizeFile(fileInput.files[0], queryList());
    });
    document.getElementById("find").addEventListener("click", () => {
      const queries = queryList();
      if (!queries.length) {
        summary.textContent = "请先填写要找的英文词，例如 cup, book。";
        return;
      }
      if (!fileInput.files[0]) {
        summary.textContent = "请先选择一张照片。";
        return;
      }
      recognizeFile(fileInput.files[0], queries);
    });
    document.getElementById("example").addEventListener("click", async () => {
      const response = await fetch("/example.png");
      const blob = await response.blob();
      await recognizeFile(new File([blob], "example.png", { type: "image/png" }), []);
    });
    document.getElementById("open-demo").addEventListener("click", async () => {
      summary.textContent = "识别中…";
      const [imageResponse, resultResponse] = await Promise.all([
        fetch("/open_demo.png"),
        fetch("/open_demo")
      ]);
      const blob = await imageResponse.blob();
      const result = await resultResponse.json();
      const image = new Image();
      image.src = URL.createObjectURL(blob);
      await image.decode();
      drawResult(image, result);
      URL.revokeObjectURL(image.src);
    });
  </script>
</body>
</html>
"""


def make_handler(app: InferenceApp):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/health":
                self._send(200, app.health())
                return
            if parsed.path == "/example.png":
                image = app.presets["clean"]["image"]
                self._send_bytes(200, png_bytes(image), "image/png")
                return
            if parsed.path == "/open_demo.png":
                image, _detections = build_cup_book_demo()
                self._send_bytes(200, png_bytes(image), "image/png")
                return
            if parsed.path == "/open_demo":
                self._send(200, app.open_demo())
                return
            if parsed.path == "/predict":
                preset = parse_qs(parsed.query).get("preset", ["clean"])[0]
                try:
                    self._send(200, app.predict(preset))
                except KeyError:
                    self._send(
                        400,
                        {"error": "未知 preset", "presets": list(app.presets.keys())},
                    )
                return
            self._send(
                404,
                {
                    "error": "未知路径",
                    "hint": "打开 / ，或访问 /health 与 /predict?preset=clean",
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"error": f"JSON 无法解析: {exc}"})
                return
            if path == "/recognize":
                try:
                    image = decode_uploaded_image(payload)
                    self._send(200, app.recognize(image))
                except (KeyError, ValueError) as exc:
                    self._send(400, {"error": str(exc)})
                return
            if path == "/recognize_open":
                try:
                    image = decode_uploaded_image(payload)
                    result = app.recognize_open(
                        image,
                        queries=payload.get("queries") or [],
                        detections=payload.get("detections") or [],
                        check_relation=bool(payload.get("check_relation")),
                    )
                    self._send(200, result)
                except (KeyError, ValueError, RuntimeError) as exc:
                    self._send(400, {"error": str(exc)})
                return
            if path != "/predict":
                self._send(
                    404,
                    {"error": "仅支持 POST /predict、POST /recognize 或 POST /recognize_open"},
                )
                return
            try:
                preset = str(payload.get("preset", "clean"))
                self._send(200, app.predict(preset))
            except KeyError:
                self._send(
                    400,
                    {
                        "error": "未知 preset",
                        "presets": list(app.presets.keys()),
                    },
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[pmv] {self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="部署 PMV 推理服务并做冒烟测试")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkpoint",
        default=str(CKPT_PATH),
        help="训练权重路径，默认使用 prototype/checkpoints/pmv_trained_weights.npz",
    )
    parser.add_argument(
        "--replay",
        default="",
        help="困难样本 jsonl。默认写到权重旁边的 replay_live.jsonl",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="不启动端口，直接对三条预设场景做一次推理冒烟",
    )
    args = parser.parse_args()
    replay_path = args.replay or None

    if args.self_check:
        self_check(args.checkpoint)
        return

    app = InferenceApp(args.checkpoint, replay_path=replay_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"PMV 推理服务已启动: http://{args.host}:{args.port}")
    print(f"权重: {args.checkpoint}")
    print("浏览器打开: http://127.0.0.1:%s/" % args.port)
    print("直接预测: http://127.0.0.1:%s/predict?preset=clean" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
