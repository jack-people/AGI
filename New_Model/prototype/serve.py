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
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

import numpy as np

from .evaluate_checkpoint import CKPT_PATH, infer_with_weights
from .modules.vision import SceneObjectSpec, VisionModuleV2, VisualSceneBuilder
from .protocol import QueryConstraint
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


COLOR_ZH = {
    "red": "红色",
    "green": "绿色",
    "blue": "蓝色",
    "yellow": "黄色",
}
SHAPE_ZH = {
    "rectangle": "矩形",
    "circle": "圆形",
    "triangle": "三角形",
}
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
    def __init__(self, checkpoint: str) -> None:
        self.weights = TrainablePMVWeights.load(checkpoint)
        self.vision = VisionModuleV2(embedding_dim=self.weights.embedding_dim)
        self.presets = build_presets()
        self.checkpoint = checkpoint

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": self.checkpoint,
            "embedding_dim": self.weights.embedding_dim,
            "presets": list(self.presets.keys()),
        }

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
        return result

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """对一张用户提供的 RGB 图像做视觉识别，返回物体、形状和空间关系。"""
        scene = self.vision.encode_to_scene_graph(np.asarray(image, dtype=np.float64))
        entities = []
        for node in scene.nodes:
            item = node.to_dict()
            color = str(node.attributes.get("color", ""))
            shape = str(node.attributes.get("shape", ""))
            item["label"] = f"{COLOR_ZH.get(color, color)}{SHAPE_ZH.get(shape, shape)}"
            entities.append(item)
        relations = []
        for edge in scene.edges:
            if edge.relation_type not in RELATION_ZH:
                continue
            source = scene.get_node(edge.source_id)
            target = scene.get_node(edge.target_id)
            if source is None or target is None:
                continue
            source_label = _node_label(source.attributes)
            target_label = _node_label(target.attributes)
            suffix = RELATION_SUFFIX[edge.relation_type]
            sentence = f"{source_label}{RELATION_ZH[edge.relation_type]}{target_label}{suffix}"
            relations.append({**edge.to_dict(), "sentence": sentence})
        if not entities:
            summary = (
                "没有识别到红、绿、蓝、黄的色块。"
                "当前视觉模块只识别这四种颜色，以及矩形、圆形、三角形。"
            )
        else:
            names = "、".join(item["label"] for item in entities)
            relation_text = "；".join(
                item["sentence"]
                for item in relations
                if item["relation"] in ("left_of", "above", "occludes")
            )
            summary = f"识别到 {len(entities)} 个物体：{names}。"
            if relation_text:
                summary += relation_text + "。"
        return {
            "summary": summary,
            "image_size": [int(image.shape[0]), int(image.shape[1])],
            "entity_count": len(entities),
            "entities": entities,
            "relations": relations,
            "source_modules": list(scene.source_modules),
            "vocabulary": {
                "colors": list(COLOR_ZH.values()),
                "shapes": list(SHAPE_ZH.values()),
            },
        }


def png_bytes(image: np.ndarray) -> bytes:
    """把 0–1 RGB 图像编码成 PNG，供浏览器示例按钮使用。"""
    import struct
    import zlib

    rgb = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    height, width, _ = rgb.shape
    raw = (rgb * 255.0).astype(np.uint8)
    rows = b"".join(b"\x00" + raw[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
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
    if failed:
        raise SystemExit(f"部署冒烟未通过: {failed}")
    print("部署冒烟通过：清晰场景接受，遮挡场景接受，临界场景拒绝。")


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
  <p>选择一张图片，视觉模块会标出它能识别的色块、形状和空间关系。当前只识别红、绿、蓝、黄，以及矩形、圆形、三角形。</p>
  <p>
    <input id="file" type="file" accept="image/*">
    <button id="example" type="button">用内置示例图</button>
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
    const colors = { red: "#d22", green: "#2a2", blue: "#24a", yellow: "#ca0" };

    function drawResult(image, result) {
      const scale = Math.min(480 / image.width, 480 / image.height, 4);
      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.font = "14px sans-serif";
      for (const entity of result.entities || []) {
        const box = entity.bbox;
        const x = box.x_min * canvas.width;
        const y = box.y_min * canvas.height;
        const w = (box.x_max - box.x_min) * canvas.width;
        const h = (box.y_max - box.y_min) * canvas.height;
        ctx.strokeStyle = colors[entity.attributes.color] || "#111";
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(entity.label, x + 2, Math.max(14, y - 2));
      }
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

    async function recognizePixels(width, height, rgba, sourceImage) {
      summary.textContent = "识别中…";
      const bytes = new Uint8Array(rgba);
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      const response = await fetch("/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          width,
          height,
          rgba_base64: btoa(binary)
        })
      });
      const result = await response.json();
      if (!response.ok) {
        summary.textContent = result.error || "识别失败";
        return;
      }
      drawResult(sourceImage, result);
    }

    async function recognizeFile(file) {
      const image = new Image();
      image.src = URL.createObjectURL(file);
      await image.decode();
      const maxSide = 256;
      const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
      const width = Math.max(8, Math.round(image.width * scale));
      const height = Math.max(8, Math.round(image.height * scale));
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      const scratchCtx = scratch.getContext("2d");
      scratchCtx.drawImage(image, 0, 0, width, height);
      const rgba = scratchCtx.getImageData(0, 0, width, height).data;
      await recognizePixels(width, height, rgba, image);
      URL.revokeObjectURL(image.src);
    }

    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) recognizeFile(fileInput.files[0]);
    });
    document.getElementById("example").addEventListener("click", async () => {
      const response = await fetch("/example.png");
      const blob = await response.blob();
      await recognizeFile(new File([blob], "example.png", { type: "image/png" }));
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
            if path != "/predict":
                self._send(404, {"error": "仅支持 POST /predict 或 POST /recognize"})
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
        "--self-check",
        action="store_true",
        help="不启动端口，直接对三条预设场景做一次推理冒烟",
    )
    args = parser.parse_args()

    if args.self_check:
        self_check(args.checkpoint)
        return

    app = InferenceApp(args.checkpoint)
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
