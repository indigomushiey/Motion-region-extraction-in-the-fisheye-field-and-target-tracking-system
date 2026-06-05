"""
主入口 — 鱼眼视频运动区域提取与目标跟踪。

路线 A: 鱼眼域直接处理
路线 B: 校正后处理

用法:
    python main.py                          # 使用 config.py 中的默认配置
    python main.py --route A                # 仅路线 A
    python main.py --route B                # 仅路线 B
    python main.py --route BOTH             # 两条路线都跑
    python main.py --method frame_diff      # 帧差法
    python main.py --method optical_flow    # 光流法
    python main.py --method all             # 所有方法
    python main.py --sample 5               # 仅处理 5 个样本
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
from collections import defaultdict

# 将 src 目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import DataLoader
from radial_poly import validate_model
from undistort import Undistorter
from motion_extraction import extract_motion, background_subtraction_mog2, align_frames
from postprocessing import postprocess_mask, adaptive_postprocess_fisheye, postprocess_mask_enhanced
from tracking import SimpleSORT, sparse_optical_flow_tracking
from evaluation import compute_metrics, compute_region_metrics
from visualization import (
    flow_to_color, draw_bboxes, draw_sparse_flow,
    create_comparison_figure, create_region_analysis_figure,
    create_distortion_analysis_figure, create_metrics_summary_figure,
    create_method_comparison_figure, create_region_overlay
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fisheye Video Motion Extraction & Tracking")
    parser.add_argument("--route", type=str, default=config.ROUTE,
                        choices=["A", "B", "BOTH"],
                        help="Processing route: A (fisheye domain), B (rectified domain), BOTH")
    parser.add_argument("--method", type=str, default=config.MOTION_METHOD,
                        choices=["frame_diff", "optical_flow", "background_sub", "all"],
                        help="Motion extraction method")
    parser.add_argument("--projection", type=str, default=config.PROJECTION,
                        choices=["perspective", "cylindrical", "equirectangular"],
                        help="Rectification projection (Route B only)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable saving intermediate results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return parser.parse_args()


def setup_output_dirs(route, method, projection):
    """创建输出目录结构。"""
    base = config.OUTPUT_DIR
    dirs = {
        "eval_comparison": os.path.join(base, "evaluation", "eval_comparison"),
        "metrics": os.path.join(base, "evaluation", "metrics"),
        "analysis": os.path.join(base, "analysis"),
        "tracking": os.path.join(base, "tracking"),
        "undistort": os.path.join(base, "undistortion"),
        "motion_a": os.path.join(base, "motion_masks", "route_A", method),
        "motion_b": os.path.join(base, "motion_masks", "route_B", method, projection),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def process_route_A(loader, sample_ids, method, output_dir, verbose=True):
    """路线 A: 鱼眼域直接处理。"""
    if verbose:
        print("\n" + "=" * 60)
        print(f"路线 A (鱼眼域直接处理) — 方法: {method}")
        print("=" * 60)

    results = []
    undo = Undistorter(config.K1, config.K2, config.K3, config.K4,
                       config.CX, config.CY, config.IMG_W, config.IMG_H)

    for i, data in enumerate(loader.iter_pairs(sample_ids)):
        img_id = data["id"]
        if verbose and i % 20 == 0:
            print(f"  [{i+1}/{len(sample_ids)}] Processing ID={img_id:05d}...")

        curr = data["curr"]
        prev = data["prev"]
        gt = data["gt"]

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # ---- 运动提取 (使用 local 自运动补偿) ----
        if method in ("frame_diff", "all"):
            # 帧差法: 先用对齐减少背景差异
            if config.USE_FEATURE_ALIGNMENT:
                alignment = align_frames(prev_gray, curr_gray,
                                         transform_type=config.ALIGNMENT_TRANSFORM,
                                         max_features=config.ALIGNMENT_MAX_FEATURES,
                                         min_inliers=config.ALIGNMENT_MIN_INLIERS,
                                         ransac_thresh=config.ALIGNMENT_RANSAC_THRESH)
                prev_motion = alignment["aligned"] if alignment["success"] else prev_gray
                th = config.FRAME_DIFF_THRESHOLD_ALIGNED if alignment["success"] else config.FRAME_DIFF_THRESHOLD
            else:
                prev_motion = prev_gray
                th = config.FRAME_DIFF_THRESHOLD
            motion = extract_motion(prev_motion, curr_gray, method="frame_diff",
                                    threshold=th,
                                    blur_ksize=config.FRAME_DIFF_BLUR_KSIZE)
        elif method == "optical_flow":
            motion = extract_motion(prev_gray, curr_gray, method="optical_flow",
                                    compensate_ego_motion=config.COMPENSATE_EGO_MOTION)
        elif method == "sparse_optical_flow":
            motion = extract_motion(prev_gray, curr_gray, method="sparse_optical_flow")
        else:
            motion = {"motion_mask": np.zeros_like(curr_gray)}

        raw_mask = motion["motion_mask"]

        # ---- 自适应后处理 (考虑畸变) ----
        post = adaptive_postprocess_fisheye(
            raw_mask, config.CX, config.CY,
            median_ksize=config.MEDIAN_BLUR_KSIZE,
            open_ksize=config.MORPH_OPEN_KSIZE,
            close_ksize=config.MORPH_CLOSE_KSIZE,
            min_area_center=config.MIN_CONTOUR_AREA,
            min_area_edge=config.MIN_CONTOUR_AREA_FISHEYE,
        )
        mask = post["mask"]

        # ---- 评估 ----
        eval_result = None
        if gt is not None:
            eval_result = compute_metrics(mask, gt)
            region_eval = compute_region_metrics(mask, gt, config.CX, config.CY,
                                                  config.REGION_RATIOS)

        results.append({
            "id": img_id,
            "raw_mask": raw_mask,
            "mask": mask,
            "post": post,
            "metrics": eval_result,
            "region_metrics": region_eval if gt is not None else None,
            "motion": motion,
        })

        # ---- 保存中间结果 ----
        if config.SAVE_INTERMEDIATE:
            fid = f"{img_id:05d}"
            cv2.imwrite(os.path.join(output_dir, f"{fid}_raw_mask.png"), raw_mask)
            cv2.imwrite(os.path.join(output_dir, f"{fid}_mask.png"), mask)
            # 叠加可视化
            overlay = curr.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
            cv2.imwrite(os.path.join(output_dir, f"{fid}_overlay.png"), overlay)

            if "flow" in motion:
                flow_color = flow_to_color(motion["flow"])
                cv2.imwrite(os.path.join(output_dir, f"{fid}_flow.png"), flow_color)

    return results


def process_route_B(loader, sample_ids, method, projection, output_dir, verbose=True):
    """路线 B: 校正后处理。"""
    if verbose:
        print("\n" + "=" * 60)
        print(f"路线 B (校正后处理) — 方法: {method}, 投影: {projection}")
        print("=" * 60)

    undo = Undistorter(config.K1, config.K2, config.K3, config.K4,
                       config.CX, config.CY, config.IMG_W, config.IMG_H)

    results = []

    for i, data in enumerate(loader.iter_pairs(sample_ids)):
        img_id = data["id"]
        if verbose and i % 20 == 0:
            print(f"  [{i+1}/{len(sample_ids)}] Processing ID={img_id:05d}...")

        curr = data["curr"]
        prev = data["prev"]
        gt = data["gt"]

        # ---- 去畸变 ----
        curr_und, curr_valid = undo.undistort(curr, projection=projection,
                                               out_size=config.UNDIST_SIZE,
                                               focal=config.UNDIST_FOCAL)
        prev_und, prev_valid = undo.undistort(prev, projection=projection,
                                               out_size=config.UNDIST_SIZE,
                                               focal=config.UNDIST_FOCAL)

        # 合成有效区域掩膜 (两帧交集, 排除黑边)
        valid_mask = cv2.bitwise_and(curr_valid, prev_valid)

        curr_gray = cv2.cvtColor(curr_und, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_und, cv2.COLOR_BGR2GRAY)

        # ---- ★ 帧间对齐 (校正域中变换更准确) ----
        if config.USE_FEATURE_ALIGNMENT:
            alignment = align_frames(
                prev_gray, curr_gray,
                transform_type=config.ALIGNMENT_TRANSFORM,
                max_features=config.ALIGNMENT_MAX_FEATURES,
                min_inliers=config.ALIGNMENT_MIN_INLIERS,
                ransac_thresh=config.ALIGNMENT_RANSAC_THRESH,
            )
            prev_aligned = alignment["aligned"]
            use_alignment = alignment["success"]
        else:
            use_alignment = False
            prev_aligned = prev_gray

        # ---- 运动提取 ----
        if method in ("frame_diff", "all"):
            th = config.FRAME_DIFF_THRESHOLD_ALIGNED if use_alignment else config.FRAME_DIFF_THRESHOLD
            motion = extract_motion(prev_aligned, curr_gray, method="frame_diff",
                                    threshold=th,
                                    blur_ksize=config.FRAME_DIFF_BLUR_KSIZE)
        elif method == "optical_flow":
            if use_alignment:
                motion = extract_motion(prev_aligned, curr_gray, method="optical_flow",
                                        compensate_ego_motion=False,
                                        farneback_params={
                                            **config.FARNEBACK_PARAMS,
                                            "mag_threshold": config.FLOW_MAG_THRESHOLD,
                                            "mag_percentile": 0,
                                        })
            else:
                motion = extract_motion(prev_aligned, curr_gray, method="optical_flow",
                                        compensate_ego_motion=True)
        elif method == "sparse_optical_flow":
            motion = extract_motion(prev_aligned, curr_gray, method="sparse_optical_flow")
        else:
            motion = {"motion_mask": np.zeros_like(curr_gray)}

        raw_mask = motion["motion_mask"]

        # ---- 用 valid_mask 去除黑边伪运动 ----
        raw_mask = cv2.bitwise_and(raw_mask, valid_mask)

        # ---- 后处理 ----
        post = postprocess_mask(
            raw_mask,
            median_ksize=config.MEDIAN_BLUR_KSIZE,
            open_ksize=config.MORPH_OPEN_KSIZE,
            close_ksize=config.MORPH_CLOSE_KSIZE,
            min_area=config.MIN_CONTOUR_AREA,
        )
        mask = post["mask"]

        # ---- 将校正域掩膜反投回鱼眼域 (用于与 GT 比较) ----
        mask_fisheye = undo.warp_mask_to_fisheye(
            mask, projection=projection,
            out_size=config.UNDIST_SIZE, focal=config.UNDIST_FOCAL
        )

        # ---- 评估 (在鱼眼图像坐标系中) ----
        eval_result = None
        region_eval = None
        if gt is not None:
            eval_result = compute_metrics(mask_fisheye, gt)
            region_eval = compute_region_metrics(mask_fisheye, gt,
                                                  config.CX, config.CY,
                                                  config.REGION_RATIOS)

        results.append({
            "id": img_id,
            "raw_mask": raw_mask,
            "mask": mask,
            "mask_fisheye": mask_fisheye,
            "post": post,
            "metrics": eval_result,
            "region_metrics": region_eval,
            "motion": motion,
            "curr_undistorted": curr_und,
        })

        # ---- 保存中间结果 ----
        if config.SAVE_INTERMEDIATE:
            fid = f"{img_id:05d}"
            cv2.imwrite(os.path.join(output_dir, f"{fid}_undistorted.png"), curr_und)
            cv2.imwrite(os.path.join(output_dir, f"{fid}_raw_mask.png"), raw_mask)
            cv2.imwrite(os.path.join(output_dir, f"{fid}_mask.png"), mask)
            cv2.imwrite(os.path.join(output_dir, f"{fid}_mask_fisheye.png"), mask_fisheye)

            overlay = curr.copy()
            overlay[mask_fisheye > 0] = overlay[mask_fisheye > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
            cv2.imwrite(os.path.join(output_dir, f"{fid}_overlay.png"), overlay)

            if "flow" in motion:
                flow_color = flow_to_color(motion["flow"])
                cv2.imwrite(os.path.join(output_dir, f"{fid}_flow.png"), flow_color)

    return results


def run_tracking(loader, sequence_index, method, projection, output_dir, verbose=True):
    """在连续序列上运行目标跟踪。"""
    seq_ids = loader.get_sequence_ids(sequence_index)
    if len(seq_ids) < 3:
        if verbose:
            print(f"  Sequence {sequence_index} too short for tracking (need >=3 frames)")
        return None

    if verbose:
        print(f"\n  序列 {chr(65+sequence_index)}: {len(seq_ids)} 帧 [{seq_ids[0]:05d}-{seq_ids[-1]:05d}]")

    undo = None
    if method == "B":
        undo = Undistorter(config.K1, config.K2, config.K3, config.K4,
                           config.CX, config.CY, config.IMG_W, config.IMG_H)

    sort_tracker = SimpleSORT(
        iou_threshold=config.TRACKING_IOU_THRESHOLD,
        max_missing=config.TRACKING_MAX_MISSING,
        min_hits=config.TRACKING_MIN_HITS,
    )

    all_tracks = []
    seq_frames = [loader.load_pair(sid) for sid in seq_ids]

    for t in range(len(seq_frames) - 1):
        curr_data = seq_frames[t + 1]
        prev_data = seq_frames[t]

        if undo is not None:
            curr_img, curr_valid = undo.undistort(curr_data["curr"], projection=projection,
                                                  out_size=config.UNDIST_SIZE,
                                                  focal=config.UNDIST_FOCAL)
            prev_img, prev_valid = undo.undistort(prev_data["curr"], projection=projection,
                                                  out_size=config.UNDIST_SIZE,
                                                  focal=config.UNDIST_FOCAL)
            valid_mask = cv2.bitwise_and(curr_valid, prev_valid)
        else:
            curr_img = curr_data["curr"]
            prev_img = prev_data["curr"]

        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        # 运动提取 (含帧间对齐)
        if config.USE_FEATURE_ALIGNMENT:
            alignment = align_frames(prev_gray, curr_gray,
                                     transform_type=config.ALIGNMENT_TRANSFORM,
                                     max_features=config.ALIGNMENT_MAX_FEATURES,
                                     min_inliers=config.ALIGNMENT_MIN_INLIERS,
                                     ransac_thresh=config.ALIGNMENT_RANSAC_THRESH)
            prev_for_motion = alignment["aligned"]
            compensate = not alignment["success"]
        else:
            prev_for_motion = prev_gray
            compensate = True

        motion = extract_motion(prev_for_motion, curr_gray, method="optical_flow",
                                compensate_ego_motion=compensate)

        if undo is not None:
            # Route B: 遮掉黑边伪运动
            motion["motion_mask"] = cv2.bitwise_and(motion["motion_mask"], valid_mask)
            post = postprocess_mask(motion["motion_mask"],
                                     min_area=config.MIN_CONTOUR_AREA)
        else:
            post = adaptive_postprocess_fisheye(
                motion["motion_mask"], config.CX, config.CY,
                min_area_center=config.MIN_CONTOUR_AREA,
                min_area_edge=config.MIN_CONTOUR_AREA_FISHEYE,
            )

        # SORT 更新
        active = sort_tracker.update(post["bboxes"])
        all_tracks.append({
            "frame_idx": t,
            "id_curr": curr_data["id"],
            "active_tracks": active,
        })

        # 保存跟踪可视化
        if config.SAVE_INTERMEDIATE and t % max(1, len(seq_frames) // 10) == 0:
            from visualization import draw_tracks
            track_img = draw_tracks(curr_img, active)
            cv2.imwrite(os.path.join(output_dir, f"track_{seq_ids[0]:05d}_{t:03d}.png"),
                        track_img)

    return all_tracks


def print_summary(results_a, results_b, method):
    """打印评估结果摘要。"""
    print("\n" + "=" * 70)
    print("评估结果摘要")
    print("=" * 70)

    for route_name, results in [("路线 A (鱼眼域)", results_a), ("路线 B (校正域)", results_b)]:
        if not results:
            continue
        valid = [r for r in results if r["metrics"] is not None]
        if not valid:
            print(f"\n  {route_name}: 无有效评估结果")
            continue

        ious = [r["metrics"]["IoU"] for r in valid]
        f1s = [r["metrics"]["F1"] for r in valid]
        precs = [r["metrics"]["Precision"] for r in valid]
        recalls = [r["metrics"]["Recall"] for r in valid]

        print(f"\n  {route_name} ({len(valid)} 个样本):")
        print(f"    IoU:       mean={np.mean(ious):.4f}, std={np.std(ious):.4f}, "
              f"median={np.median(ious):.4f}")
        print(f"    F1:        mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}, "
              f"median={np.median(f1s):.4f}")
        print(f"    Precision: mean={np.mean(precs):.4f}, std={np.std(precs):.4f}")
        print(f"    Recall:    mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}")

        # 分区域统计
        region_stats = defaultdict(list)
        for r in valid:
            if r["region_metrics"]:
                for reg_name in ["center", "transition", "edge"]:
                    if reg_name in r["region_metrics"]:
                        region_stats[reg_name].append(r["region_metrics"][reg_name]["IoU"])

        print(f"\n    分区域 IoU:")
        for reg_name in ["center", "transition", "edge"]:
            vals = region_stats[reg_name]
            if vals:
                print(f"      {reg_name:12s}: mean={np.mean(vals):.4f}, "
                      f"std={np.std(vals):.4f}, median={np.median(vals):.4f}")

    print("=" * 70)


def generate_analysis_figures(loader, results_a, results_b, dirs, method, projection):
    """生成分析图表。"""
    print("\n生成分析图表...")

    # 1. 模型验证
    val_result = validate_model()
    print(f"  radial_poly 模型验证: "
          f"max_error={val_result['max_error_deg']:.6f}°, "
          f"{'PASS' if val_result['passed'] else 'FAIL'}")

    # 2. ★ 鱼眼畸变特征分析 (题目要求: 分析鱼眼成像特点)
    from distortion_analysis import (
        analyze_distortion_characteristics,
        demonstrate_edge_effect,
        analyze_flow_distortion_impact
    )
    analysis_dir = os.path.join(config.OUTPUT_DIR, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    sample_data = loader.load_pair(loader.ids[0])
    curr = sample_data["curr"]

    analyze_distortion_characteristics(
        config.K1, config.K2, config.K3, config.K4,
        config.CX, config.CY, config.IMG_W, config.IMG_H,
        analysis_dir
    )
    demonstrate_edge_effect(
        curr, config.K1, config.K2, config.K3, config.K4,
        config.CX, config.CY, analysis_dir
    )
    # 对第一个有光流结果的样本做分区域光流分析
    if results_a:
        res0 = results_a[0]
        if "flow" in res0.get("motion", {}):
            analyze_flow_distortion_impact(
                res0["motion"]["flow"],
                config.CX, config.CY,
                config.K1, config.K2, config.K3, config.K4,
                analysis_dir
            )

    # 3. 校正前后对比图 (第一个样本)
    undo = Undistorter(config.K1, config.K2, config.K3, config.K4,
                       config.CX, config.CY, config.IMG_W, config.IMG_H)

    curr_p, _ = undo.undistort(curr, projection="perspective",
                                out_size=config.UNDIST_SIZE, focal=config.UNDIST_FOCAL)
    curr_c, _ = undo.undistort(curr, projection="cylindrical",
                                out_size=config.UNDIST_SIZE, focal=config.UNDIST_FOCAL)
    curr_e, _ = undo.undistort(curr, projection="equirectangular",
                                out_size=config.UNDIST_SIZE)

    create_distortion_analysis_figure(curr, curr_p, curr_c, curr_e,
                                       save_path=os.path.join(dirs["undistort"],
                                                              "projection_comparison.png"))
    print(f"  [OK] 多投影对比图 → {dirs['undistort']}")

    # 4. 路线 A vs B 对比图 (选5个样本, 包含最好和最差)
    all_ids_with_gt = [r["id"] for r in results_a if r["metrics"] is not None]
    if len(all_ids_with_gt) > 5:
        # 选: 前2个, IoU最好2个, IoU最差1个
        ious = [(r["id"], r["metrics"]["IoU"]) for r in results_a if r["metrics"] is not None]
        ious.sort(key=lambda x: x[1])
        demo_ids = [all_ids_with_gt[0], all_ids_with_gt[1],
                     ious[-1][0], ious[-2][0], ious[0][0]]
        demo_ids = list(dict.fromkeys(demo_ids))  # 去重
    else:
        demo_ids = all_ids_with_gt[:5]

    for sid in demo_ids:
        data = loader.load_pair(sid)
        curr = data["curr"]
        prev = data["prev"]
        gt = data["gt"]

        res_a = next((r for r in results_a if r["id"] == sid), None) if results_a else None
        res_b = next((r for r in results_b if r["id"] == sid), None) if results_b else None

        mask_a = res_a["mask"] if res_a else np.zeros_like(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        mask_b = res_b["mask_fisheye"] if res_b else np.zeros_like(mask_a)
        gt_mask = (gt * 255).astype(np.uint8) if gt is not None else np.zeros_like(mask_a)

        flow_color = None
        if res_a and "flow" in res_a.get("motion", {}):
            flow_color = flow_to_color(res_a["motion"]["flow"])

        create_comparison_figure(curr, prev, mask_a, mask_b, gt_mask,
                                  flow_color=flow_color,
                                  route_a_name=f"Route A ({method})",
                                  route_b_name=f"Route B ({method}, {projection})",
                                  save_path=os.path.join(dirs["eval_comparison"],
                                                         f"comparison_{sid:05d}.png"))
    print(f"  [OK] A vs B 对比图 ×{len(demo_ids)} → {dirs['eval_comparison']}")

    # 5. 分区域叠加可视化 (选取 IoU 最好的样本)
    if results_a and demo_ids:
        best_id = demo_ids[-1] if len(demo_ids) > 2 else demo_ids[0]
        data_best = loader.load_pair(best_id)
        res_best = next((r for r in results_a if r["id"] == best_id), None)
        if res_best and data_best["gt"] is not None:
            create_region_overlay(
                data_best["curr"], res_best["mask"], data_best["gt"],
                config.CX, config.CY, config.REGION_RATIOS,
                save_path=os.path.join(dirs["analysis"], f"region_overlay_{best_id:05d}.png")
            )
            print(f"  [OK] 分区域叠加图 → {dirs['analysis']}")

    # 6. 评估指标汇总图
    if results_a:
        create_metrics_summary_figure(results_a, f"Route A — {method}",
                                       save_path=os.path.join(dirs["metrics"],
                                                              "summary_route_A.png"))
    if results_b:
        create_metrics_summary_figure(results_b, f"Route B — {method} ({projection})",
                                       save_path=os.path.join(dirs["metrics"],
                                                              "summary_route_B.png"))
    print(f"  [OK] 评估汇总图 → {dirs['metrics']}")

    # 5. 分区域对比图 (A vs B)
    if results_a and results_b:
        # 聚合所有样本的分区域指标
        agg_a = {"center": {}, "transition": {}, "edge": {}}
        agg_b = {"center": {}, "transition": {}, "edge": {}}

        for metric in ["IoU", "Precision", "Recall", "F1"]:
            for reg in ["center", "transition", "edge"]:
                vals_a = [r["region_metrics"][reg][metric]
                          for r in results_a if r["region_metrics"] is not None]
                vals_b = [r["region_metrics"][reg][metric]
                          for r in results_b if r["region_metrics"] is not None]
                agg_a[reg][metric] = np.mean(vals_a) if vals_a else 0.0
                agg_b[reg][metric] = np.mean(vals_b) if vals_b else 0.0

        create_region_analysis_figure(
            agg_a, agg_b, region_names=["Center", "Transition", "Edge"],
            save_path=os.path.join(dirs["analysis"], "region_comparison.png")
        )
        print(f"  [OK] 分区域对比图 → {dirs['analysis']}")

    # 6. 保存评估结果为 JSON
    def convert_to_serializable(results, route_name):
        return [{
            "id": r["id"],
            "IoU": r["metrics"]["IoU"] if r["metrics"] else None,
            "Precision": r["metrics"]["Precision"] if r["metrics"] else None,
            "Recall": r["metrics"]["Recall"] if r["metrics"] else None,
            "F1": r["metrics"]["F1"] if r["metrics"] else None,
            "region_IoU": {
                reg: r["region_metrics"][reg]["IoU"]
                for reg in ["center", "transition", "edge"]
            } if r["region_metrics"] else None,
            "num_regions": r["post"]["num_regions"],
        } for r in results]

    if results_a:
        with open(os.path.join(dirs["metrics"], "results_route_A.json"), "w") as f:
            json.dump(convert_to_serializable(results_a, "A"), f, indent=2, ensure_ascii=False)
    if results_b:
        with open(os.path.join(dirs["metrics"], "results_route_B.json"), "w") as f:
            json.dump(convert_to_serializable(results_b, "B"), f, indent=2, ensure_ascii=False)
    print(f"  [OK] 评估结果 JSON → {dirs['metrics']}")


def main():
    args = parse_args()

    # 覆盖配置
    route = args.route
    method = args.method
    projection = args.projection

    if args.no_save:
        config.SAVE_INTERMEDIATE = False
    if args.quiet:
        config.VERBOSE = False

    verbose = config.VERBOSE

    print("=" * 60)
    print("鱼眼视频运动区域提取与目标跟踪")
    print(f"  路线: {route}  方法: {method}  投影: {projection}")
    print("=" * 60)

    # ---- 数据加载 ----
    loader = DataLoader(
        data_dir=config.DATA_DIR,
        rgb_dir=config.RGB_DIR,
        prev_dir=config.PREV_DIR,
        calib_dir=config.CALIB_DIR,
        gt_dir=config.GT_DIR,
    )
    loader.print_summary()

    # 确定处理样本
    sample_ids = loader.ids
    if args.sample:
        sample_ids = loader.ids[:args.sample]
    config.SAMPLE_IDS = sample_ids

    if verbose:
        print(f"\n处理 {len(sample_ids)} 个样本")

    # ---- 模型验证 ----
    val = validate_model()
    if verbose:
        print(f"\nradial_poly 模型验证: max_error={val['max_error_deg']:.6f}° "
              f"({'PASS' if val['passed'] else 'FAIL'})")
    if not val["passed"]:
        print("  WARNING: 模型验证未通过，校正结果可能不准确！")

    # ---- 创建输出目录 ----
    dirs = setup_output_dirs(route, method, projection)

    # ---- 运行路线 A ----
    results_a = []
    if route in ("A", "BOTH"):
        results_a = process_route_A(loader, sample_ids, method, dirs["motion_a"],
                                     verbose=verbose)

    # ---- 运行路线 B ----
    results_b = []
    if route in ("B", "BOTH"):
        results_b = process_route_B(loader, sample_ids, method, projection,
                                     dirs["motion_b"], verbose=verbose)

    # ---- 打印评估摘要 ----
    print_summary(results_a, results_b, method)

    # ---- 目标跟踪 (在最长连续序列上) ----
    if verbose:
        print("\n" + "=" * 60)
        print("目标跟踪 (连续序列)")
        print("=" * 60)

    # 路线 A 跟踪 (使用最长序列)
    if route in ("A", "BOTH") and len(loader.sequences) > 0:
        best_seq = max(range(len(loader.sequences)), key=lambda i: len(loader.sequences[i]))
        print("\n  路线 A (鱼眼域) 跟踪:")
        tracks_a = run_tracking(loader, best_seq, "A", projection,
                                 dirs["tracking"], verbose)

    # 路线 B 跟踪
    if route in ("B", "BOTH") and len(loader.sequences) > 0:
        best_seq = max(range(len(loader.sequences)), key=lambda i: len(loader.sequences[i]))
        print("\n  路线 B (校正域) 跟踪:")
        tracks_b = run_tracking(loader, best_seq, "B", projection,
                                 dirs["tracking"], verbose)

    # ---- 生成分析图表 ----
    if config.SAVE_INTERMEDIATE:
        generate_analysis_figures(loader, results_a, results_b, dirs, method, projection)

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"  输出目录: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
