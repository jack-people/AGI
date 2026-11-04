"""
本地推理服务：加载训练好的权重，对外提供健康检查和场景预测。

部署（在仓库根目录）：
    python3 -m New_Model.prototype.serve
    python3 -m New_Model.prototype.serve --host 0.0.0.0 --port 8765

冒烟测试（不启动端口，直接跑三条预设场景）：
    python3 -m New_Model.prototype.serve --self-check

服务起来之后：
    curl http://127.0.0.1:8765/health
    curl -X POST http://127.0.0.1:8765/predict -H 'Content-Type: application/json' -d '{"preset":"clean"}'
    curl -X POST http://127.0.0.1:8765/predict -H 'Content-Type: application/json' -d '{"preset":"occluded"}'
    curl -X POST http://127.0.0.1:8765/predict -H 'Content-Type: application/json' -d '{"preset":"unstable"}'
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

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


def make_handler(app: InferenceApp):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/health":
                self._send(200, app.health())
                return
            self._send(
                404,
                {
                    "error": "未知路径",
                    "hint": "GET /health 或 POST /predict {\"preset\":\"clean|occluded|unstable\"}",
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/predict":
                self._send(404, {"error": "仅支持 POST /predict"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
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
            except json.JSONDecodeError as exc:
                self._send(400, {"error": f"JSON 无法解析: {exc}"})

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
    print("测试: curl http://127.0.0.1:%s/health" % args.port)
    print(
        "预测: curl -X POST http://127.0.0.1:%s/predict "
        "-H 'Content-Type: application/json' -d '{\"preset\":\"clean\"}'" % args.port
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
