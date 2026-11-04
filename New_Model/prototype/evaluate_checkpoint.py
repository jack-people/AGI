"""
数据集导出与训练后权重独立验证工具 (evaluate_checkpoint.py)

运行方式：
    # 1. 将原始视觉图像 (.npy/.ppm) 与结构化标注导出到磁盘目录 New_Model/prototype/data/
    python3 -m New_Model.prototype.evaluate_checkpoint --export-data

    # 2. 加载训练好的权重文件 (pmv_trained_weights.npz)，在未见过的新随机种子测试集 (Unseen Holdout Set) 上独立验证
    python3 -m New_Model.prototype.evaluate_checkpoint --verify-unseen

    # 3. 对单张测试图像进行逐步可视化推理对比（未训练权重 vs. 已训练权重）
    python3 -m New_Model.prototype.evaluate_checkpoint --inspect-sample 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from .benchmark import generate_benchmark_suite
from .modules.vision import SceneObjectSpec, VisionModuleV2, VisualSceneBuilder
from .protocol import QueryConstraint
from .trainer import (
    STRATEGY_LIST,
    TrainablePMVWeights,
    encode_query_vector,
    extract_scene_context_features,
    softmax,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
CKPT_PATH = Path(__file__).resolve().parent / "checkpoints" / "pmv_trained_weights.npz"


def save_rgb_as_ppm(image: np.ndarray, path: Path) -> None:
    """将 (H, W, 3) 浮点 RGB 图像保存为可直接查看的二进制 PPM (P6) 图像文件。"""
    h, w, _ = image.shape
    rgb_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgb_uint8.tobytes())


def export_raw_dataset_to_disk() -> Path:
    """
    将内存中程序化合成的 120 个训练/评测原始数据（64x64x3 RGB 像素张量 + JSONL 结构化任务标注 + 可视化样本图）
    持久化落盘到 New_Model/prototype/data/ 目录下，方便直观查看与审计。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples_img_dir = DATA_DIR / "sample_images"
    samples_img_dir.mkdir(parents=True, exist_ok=True)

    suite = generate_benchmark_suite()
    images_array = np.stack([item["image"] for item in suite], axis=0)  # (120, 64, 64, 3)
    npy_path = DATA_DIR / "raw_images_120x64x64x3.npy"
    np.save(npy_path, images_array)

    jsonl_path = DATA_DIR / "dataset_manifest.jsonl"
    exported_preview_count = {"clean": 0, "occluded_noisy": 0, "boundary": 0}

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(suite):
            split = item["split"]
            q: QueryConstraint = item["query"]
            img_file_rel = None
            if exported_preview_count.get(split, 0) < 2:
                ppm_name = f"{idx:03d}_{item['id']}.ppm"
                save_rgb_as_ppm(item["image"], samples_img_dir / ppm_name)
                img_file_rel = f"sample_images/{ppm_name}"
                exported_preview_count[split] = exported_preview_count.get(split, 0) + 1

            record = {
                "index": idx,
                "id": item["id"],
                "split": split,
                "image_shape": list(item["image"].shape),
                "pixel_min": round(float(np.min(item["image"])), 4),
                "pixel_max": round(float(np.max(item["image"])), 4),
                "pixel_mean": round(float(np.mean(item["image"])), 4),
                "preview_image": img_file_rel,
                "query": {
                    "goal": q.goal,
                    "target_attributes": q.target_attributes,
                    "required_relations": [
                        {"relation": r_type, "target_filter": t_filter}
                        for r_type, t_filter in q.required_relations
                    ],
                    "error_tolerance": q.error_tolerance,
                    "perturbation_radius": q.perturbation_radius,
                },
                "ground_truth": {
                    "gt_color": item["gt_color"],
                    "gt_stable": item["gt_stable"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[数据导出完成] 原始图像张量已保存至: {npy_path} (shape={images_array.shape})")
    print(f"[数据导出完成] 任务与标签清单已保存至: {jsonl_path} (共 {len(suite)} 条)")
    print(f"[数据导出完成] 6 张代表性可视化图像 (.ppm) 已保存至: {samples_img_dir}")
    return DATA_DIR


def generate_unseen_holdout_test_suite(seed_offset: int = 9000) -> List[Dict[str, Any]]:
    """
    生成一批全新的、训练阶段从未见过的独立测试集 (Unseen Holdout Set, 共 60 个新场景)：
    采用不同的随机物体尺寸、不同的坐标偏移、不同的噪声随机种子 (seed >= 9000)。
    用于严格检验训练出来的权重是“死记硬背”还是真正具备泛化能力。
    """
    builder = VisualSceneBuilder(height=64, width=64)
    rng = np.random.default_rng(seed_offset)
    holdout: List[Dict[str, Any]] = []
    colors = ["red", "blue", "green"]
    shapes = ["rectangle", "circle", "triangle"]

    # 1. 全新随机位置与尺寸的 Clean 场景 (20 题)
    for i in range(20):
        c_left = colors[(i + 1) % 3]
        c_right = colors[(i + 2) % 3]
        s_left = shapes[(i + 1) % 3]
        s_right = shapes[i % 3]
        dy = float(rng.uniform(-0.05, 0.12))
        dx = float(rng.uniform(-0.03, 0.04))

        img = builder.render(
            [
                SceneObjectSpec(
                    "unseen_target",
                    c_left,
                    s_left,
                    0.22 + dy,
                    0.14 + dx,
                    0.50 + dy,
                    0.42 + dx,
                    z_order=1,
                ),
                SceneObjectSpec(
                    "unseen_ref",
                    c_right,
                    s_right,
                    0.22 + dy,
                    0.60 + dx,
                    0.50 + dy,
                    0.88 + dx,
                    z_order=1,
                ),
            ]
        )
        holdout.append(
            {
                "id": f"unseen_clean_{i}",
                "split": "clean",
                "image": img,
                "query": QueryConstraint(
                    goal=f"[Unseen] Find {c_left} {s_left} left_of {c_right}",
                    target_attributes={"color": c_left, "shape": s_left},
                    required_relations=[("left_of", {"color": c_right})],
                    error_tolerance=0.15,
                    perturbation_radius=0.025,
                ),
                "gt_color": c_left,
                "gt_stable": True,
            }
        )

    # 2. 全新随机噪声种子与不同遮挡位置的 Occluded & Noisy 场景 (20 题)
    for i in range(20):
        c_target = "green" if i % 2 == 0 else "red"
        c_ref = "blue"
        dy = float(rng.uniform(-0.04, 0.08))

        img = builder.render(
            [
                SceneObjectSpec(
                    "unseen_rect",
                    c_target,
                    "rectangle",
                    0.22 + dy,
                    0.14,
                    0.62 + dy,
                    0.54,
                    z_order=1,
                ),
                SceneObjectSpec(
                    "unseen_occluder",
                    "yellow",
                    "circle",
                    0.38 + dy,
                    0.34,
                    0.62 + dy,
                    0.58,
                    z_order=2,
                ),
                SceneObjectSpec(
                    "unseen_blue",
                    c_ref,
                    "circle",
                    0.25 + dy,
                    0.72,
                    0.55 + dy,
                    0.94,
                    z_order=1,
                ),
            ],
            salt_pepper_prob=0.04,
            seed=seed_offset + i,
        )
        holdout.append(
            {
                "id": f"unseen_occ_{i}",
                "split": "occluded_noisy",
                "image": img,
                "query": QueryConstraint(
                    goal=f"[Unseen] Find occluded {c_target} rectangle left_of {c_ref}",
                    target_attributes={"color": c_target, "shape": "rectangle"},
                    required_relations=[("left_of", {"color": c_ref})],
                    error_tolerance=0.15,
                    perturbation_radius=0.025,
                ),
                "gt_color": c_target,
                "gt_stable": True,
            }
        )

    # 3. 全新 Boundary Critical 场景 (20 题：10 个安全余量 + 5 个中等余量 + 5 个临界不稳定)
    for i in range(20):
        if i < 10:
            gap = 0.25
            r_pert = 0.030
            stable = True
        elif i < 15:
            gap = 0.047
            r_pert = 0.018
            stable = True
        else:
            gap = 0.045
            r_pert = 0.039
            stable = False

        img = builder.render(
            [
                SceneObjectSpec(
                    "red_box", "red", "rectangle", 0.18, 0.22, 0.42, 0.42, z_order=1
                ),
                SceneObjectSpec(
                    "blue_box",
                    "blue",
                    "rectangle",
                    0.52,
                    0.22 + gap,
                    0.76,
                    0.42 + gap,
                    z_order=1,
                ),
            ]
        )
        holdout.append(
            {
                "id": f"unseen_bound_{i}",
                "split": "boundary",
                "image": img,
                "query": QueryConstraint(
                    goal="[Unseen] Verify boundary stability",
                    target_attributes={"color": "red", "shape": "rectangle"},
                    required_relations=[("left_of", {"color": "blue"})],
                    error_tolerance=0.15,
                    perturbation_radius=r_pert,
                ),
                "gt_color": "red",
                "gt_stable": stable,
            }
        )

    return holdout


def infer_with_weights(
    weights: TrainablePMVWeights,
    image: np.ndarray,
    query: QueryConstraint,
    vision: VisionModuleV2 | None = None,
) -> Dict[str, Any]:
    """用已加载权重对一张图和一个结构化查询做一次推理，返回可序列化结果。"""
    if vision is None:
        vision = VisionModuleV2(embedding_dim=weights.embedding_dim)
    sg = vision.encode_to_scene_graph(image)
    if not sg.nodes:
        return {
            "accepted": False,
            "status": "UNKNOWN",
            "reason": "视觉模块没有提取到实体",
            "nodes": [],
            "selected_entity_id": None,
            "selected_color": None,
            "chosen_branch": None,
            "branch_probs": {},
            "goal": query.goal,
        }

    q_vec = encode_query_vector(query, sg, weights.embedding_dim)
    ctx_feat = extract_scene_context_features(sg, query)

    raw_embs = np.stack([n.embedding for n in sg.nodes], axis=0)
    proj_embs = raw_embs @ weights.w_vis_proj + weights.b_vis_proj
    q_mat = proj_embs @ weights.w_q
    k_mat = proj_embs @ weights.w_k
    v_mat = proj_embs @ weights.w_v
    attn = softmax((q_mat @ k_mat.T) / math.sqrt(weights.embedding_dim))
    ctx_embs = proj_embs + attn @ v_mat
    node_probs = softmax(ctx_embs @ (q_vec @ weights.w_match))

    branch_probs = softmax(ctx_feat @ weights.policy_logits)
    chosen_branch = int(np.argmax(branch_probs))

    ranked_indices = np.argsort(-node_probs)
    chosen_idx = int(ranked_indices[0])
    if chosen_branch == 1:
        for idx in ranked_indices:
            cand_node = sg.nodes[int(idx)]
            if cand_node.attributes.get("occluded", False) and all(
                cand_node.attributes.get(k) == v
                for k, v in query.target_attributes.items()
            ):
                chosen_idx = int(idx)
                break

    node = sg.nodes[chosen_idx]
    attr_matched = all(
        node.attributes.get(k) == v for k, v in query.target_attributes.items()
    )
    min_margin = 0.0
    rel_matched = False
    if query.required_relations:
        rel_type, ref_filter = query.required_relations[0]
        for edge in sg.find_edges(source_id=node.entity_id, relation_type=rel_type):
            target_node = sg.get_node(edge.target_id)
            if target_node and all(
                target_node.attributes.get(k) == v for k, v in ref_filter.items()
            ):
                rel_matched = True
                min_margin = max(min_margin, edge.margin)

    radius = query.perturbation_radius
    local_err_bound = 1.4142 * radius + (radius ** 2)
    is_occluded = bool(node.attributes.get("occluded", False))

    if chosen_branch == 0:
        pred_accept = (
            attr_matched
            and (not is_occluded)
            and rel_matched
            and min_margin >= 0.055
            and min_margin > local_err_bound
        )
    elif chosen_branch == 1:
        pred_accept = (
            attr_matched
            and is_occluded
            and rel_matched
            and min_margin > local_err_bound
        )
    elif chosen_branch == 2:
        pred_accept = (
            attr_matched
            and (not is_occluded)
            and rel_matched
            and (min_margin > local_err_bound)
            and (min_margin < 0.055)
        )
    else:
        pred_accept = False if local_err_bound >= min_margin else True

    nodes = []
    for idx, entity in enumerate(sg.nodes):
        record = entity.to_dict()
        record["match_prob"] = round(float(node_probs[idx]), 4)
        nodes.append(record)

    return {
        "accepted": bool(pred_accept),
        "status": "PASS" if pred_accept else "REJECT",
        "goal": query.goal,
        "selected_entity_id": node.entity_id,
        "selected_color": node.attributes.get("color"),
        "selected_shape": node.attributes.get("shape"),
        "chosen_branch": STRATEGY_LIST[chosen_branch],
        "branch_probs": {
            STRATEGY_LIST[i]: round(float(branch_probs[i]), 4) for i in range(4)
        },
        "min_margin": round(float(min_margin), 4),
        "local_error_bound": round(float(local_err_bound), 4),
        "relation_matched": bool(rel_matched),
        "nodes": nodes,
        "source_modules": list(sg.source_modules),
    }


def evaluate_weights_on_suite(
    weights: TrainablePMVWeights, suite: List[Dict[str, Any]]
) -> Dict[str, float]:
    """给定一组模型权重，在指定测试集上独立计算准确率与临界误接受率。"""
    vision = VisionModuleV2(embedding_dim=weights.embedding_dim)
    clean_ok, clean_tot = 0, 0
    occ_ok, occ_tot = 0, 0
    bound_ok, bound_tot = 0, 0
    false_accept, unstable_tot = 0, 0

    for item in suite:
        result = infer_with_weights(weights, item["image"], item["query"], vision)
        pred_accept = bool(result["accepted"])
        color_ok = result["selected_color"] == item["gt_color"]

        split = item["split"]
        if split == "clean":
            clean_tot += 1
            if pred_accept and color_ok:
                clean_ok += 1
        elif split == "occluded_noisy":
            occ_tot += 1
            if pred_accept and color_ok:
                occ_ok += 1
        elif split == "boundary":
            bound_tot += 1
            if item["gt_stable"]:
                if pred_accept and color_ok:
                    bound_ok += 1
            else:
                unstable_tot += 1
                if not pred_accept:
                    bound_ok += 1
                else:
                    false_accept += 1

    total_ok = clean_ok + occ_ok + bound_ok
    return {
        "clean_acc": round(100.0 * clean_ok / max(1, clean_tot), 2),
        "occ_acc": round(100.0 * occ_ok / max(1, occ_tot), 2),
        "boundary_acc": round(100.0 * bound_ok / max(1, bound_tot), 2),
        "overall_acc": round(100.0 * total_ok / max(1, len(suite)), 2),
        "false_accept_rate": round(100.0 * false_accept / max(1, unstable_tot), 2),
    }


def inspect_single_sample(sample_idx: int) -> None:
    """对单个样本对比展示【未训练权重】与【已训练权重】的逐步推理差异。"""
    suite = generate_unseen_holdout_test_suite()
    sample = suite[sample_idx % len(suite)]
    vision = VisionModuleV2(embedding_dim=16)
    sg = vision.encode_to_scene_graph(sample["image"])
    q: QueryConstraint = sample["query"]
    q_vec = encode_query_vector(q, sg, 16)
    ctx_feat = extract_scene_context_features(sg, q)

    untrained = TrainablePMVWeights.initialize_random(embedding_dim=16, seed=42)
    trained = TrainablePMVWeights.load(CKPT_PATH)

    print("\n" + "=" * 78)
    print(f"  单样本逐步解剖验证: [{sample['id']}] (类别: {sample['split']})")
    print(f"  查询目标: {q.goal} | 真实目标颜色: {sample['gt_color']} | 真实稳定性: {sample['gt_stable']}")
    print("=" * 78)

    for label, w in [("未训练随机模型 (Untrained)", untrained), ("已训练模型Checkpoint (Trained)", trained)]:
        raw_embs = np.stack([n.embedding for n in sg.nodes], axis=0)
        proj_embs = raw_embs @ w.w_vis_proj + w.b_vis_proj
        q_mat = proj_embs @ w.w_q
        k_mat = proj_embs @ w.w_k
        v_mat = proj_embs @ w.w_v
        attn = softmax((q_mat @ k_mat.T) / math.sqrt(w.embedding_dim))
        ctx_embs = proj_embs + attn @ v_mat
        node_probs = softmax(ctx_embs @ (q_vec @ w.w_match))
        branch_probs = softmax(ctx_feat @ w.policy_logits)
        best_branch = int(np.argmax(branch_probs))

        print(f"\n>>> {label}:")
        for idx, n in enumerate(sg.nodes):
            print(
                f"    节点[{idx}] {n.entity_id:<14} (color={n.attributes['color']:<6}, "
                f"shape={n.attributes['shape']:<9}, occluded={n.attributes['occluded']}) "
                f"-> 匹配概率 P = {node_probs[idx]:.4f}"
            )
        print(
            f"    并行策略路由分布: "
            + ", ".join(
                f"{STRATEGY_LIST[g]}={branch_probs[g]:.3f}" for g in range(4)
            )
        )
        print(f"    最终激活分支: Branch-{best_branch} ({STRATEGY_LIST[best_branch]})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 PMV 原始训练数据并独立验证训练好的 Checkpoint 权重"
    )
    parser.add_argument(
        "--export-data", action="store_true", help="将 120 条原始训练数据与图像落盘到 data/ 目录"
    )
    parser.add_argument(
        "--verify-unseen",
        action="store_true",
        help="在 60 个全新随机种子的未见测试集上对比验证训练前后的模型权重",
    )
    parser.add_argument(
        "--inspect-sample",
        type=int,
        default=None,
        help="指定未见测试集样本编号 (0-59)，打印未训练 vs 已训练的内部概率解剖对比",
    )
    args = parser.parse_args()

    # 若未传任何参数，默认执行全套：导出数据 + 未见测试集验证 + 单样本解剖
    run_all = not (args.export_data or args.verify_unseen or args.inspect_sample is not None)

    if args.export_data or run_all:
        export_raw_dataset_to_disk()

    if args.verify_unseen or run_all:
        print("\n" + "=" * 78)
        print("  独立验证：在 60 个全新未见测试场景 (Unseen Holdout Set, Seed=9000+) 上检验泛化能力")
        print("=" * 78)
        unseen_suite = generate_unseen_holdout_test_suite(seed_offset=9000)
        untrained_w = TrainablePMVWeights.initialize_random(embedding_dim=16, seed=42)
        trained_w = TrainablePMVWeights.load(CKPT_PATH)

        m_untrained = evaluate_weights_on_suite(untrained_w, unseen_suite)
        m_trained = evaluate_weights_on_suite(trained_w, unseen_suite)

        print(f"  [未训练随机权重 Untrained] -> Clean: {m_untrained['clean_acc']:>6.2f}% | 遮挡+噪声: {m_untrained['occ_acc']:>6.2f}% | 临界边界: {m_untrained['boundary_acc']:>6.2f}% | 总准确率: {m_untrained['overall_acc']:>6.2f}%")
        print(f"  [已训练模型权重 Trained  ] -> Clean: {m_trained['clean_acc']:>6.2f}% | 遮挡+噪声: {m_trained['occ_acc']:>6.2f}% | 临界边界: {m_trained['boundary_acc']:>6.2f}% | 总准确率: {m_trained['overall_acc']:>6.2f}% (FAR={m_trained['false_accept_rate']}%)")

    if args.inspect_sample is not None:
        inspect_single_sample(args.inspect_sample)
    elif run_all:
        inspect_single_sample(0)
        inspect_single_sample(20)
        inspect_single_sample(58)


if __name__ == "__main__":
    main()
