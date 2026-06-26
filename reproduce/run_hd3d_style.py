from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UDIS_REPO = REPO_ROOT.parent / "UnsupervisedDeepImageStitching"
METHOD = "unistitch"
DEFAULT_CHECKPOINT = REPO_ROOT / "model" / "epoch_best_model.pth"
DEFAULT_HD3D_MANIFEST = Path(r"D:\HD3D_Result\_global_work\_work_root\manifest.csv")
DEFAULT_RESULT_ROOT = Path(r"D:\HD3D_Result")

for path in (REPO_ROOT / "reproduce", UDIS_REPO / "reproduce", UDIS_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hd3d_eval import evaluate_raw, finite, load_lpips_metric, load_niqe_metric  # noqa: E402
from run_stitchbench import (  # noqa: E402
    RetryablePairError,
    bgr_to_model_tensor,
    extract_matched_keypoints,
    fallback_edges,
    is_oom_error,
    load_feature_models,
    load_network,
    prepare_images,
    resolve_device,
    setup_seed,
    tensor_outputs_to_images,
)
from network import build_output_model  # noqa: E402


PER_PAIR_FIELDS = [
    "scene",
    "pair_id",
    "pair_name",
    "method",
    "status",
    "failure_reason",
    "mdr",
    "niqe",
    "psnr",
    "ssim",
    "lpips",
    "rmse",
    "runtime_seconds",
    "valid_ratio",
    "alignment_matcher",
    "alignment_matches",
    "alignment_inliers",
    "valid_mask_strategy",
    "lpips_max_side",
    "raw_path",
    "aligned_path",
    "valid_mask_path",
    "gt_path",
    "cpp_mdr",
    "cpp_warping_residual_avg",
    "cpp_warping_residual_sd",
    "gt_width",
    "gt_height",
]

SUMMARY_FIELDS = [
    "method",
    "total_runs",
    "successes",
    "failures",
    "failure_rate",
    "mean_mdr",
    "median_mdr",
    "mean_niqe",
    "median_niqe",
    "mean_psnr",
    "median_psnr",
    "mean_ssim",
    "median_ssim",
    "mean_lpips",
    "median_lpips",
    "mean_rmse",
    "median_rmse",
    "mean_runtime",
    "median_runtime",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(number) else f"{number:.5f}"


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def final_pair_dir(row: dict[str, str], result_root: Path) -> Path:
    value = row.get("final_pair_dir")
    if value:
        return Path(value)
    return result_root / row["scene"] / f"pair_{row['pair_id']}"


def base_row(row: dict[str, str], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scene": row["scene"],
        "pair_id": row["pair_id"],
        "pair_name": row["pair_name"],
        "method": METHOD,
        "status": "failed",
        "failure_reason": "",
        "mdr": math.nan,
        "niqe": math.nan,
        "psnr": math.nan,
        "ssim": math.nan,
        "lpips": math.nan,
        "rmse": math.nan,
        "runtime_seconds": math.nan,
        "valid_ratio": math.nan,
        "alignment_matcher": "",
        "alignment_matches": "",
        "alignment_inliers": "",
        "valid_mask_strategy": "",
        "lpips_max_side": args.lpips_max_side,
        "raw_path": str(out_dir / "stitch_result.png"),
        "aligned_path": "",
        "valid_mask_path": "",
        "gt_path": row["gt_path"],
        "cpp_mdr": math.nan,
        "cpp_warping_residual_avg": math.nan,
        "cpp_warping_residual_sd": math.nan,
        "gt_width": "",
        "gt_height": "",
    }


def cache_hit(out_dir: Path, args: argparse.Namespace) -> bool:
    work_dir = out_dir / "work"
    status = load_json(work_dir / "method_status.json")
    metrics = load_json(work_dir / "metrics.json")
    return (
        args.skip_existing
        and not args.force
        and status.get("success")
        and metrics.get("status") == "success"
        and (out_dir / "stitch_result.png").exists()
        and (out_dir / "aligned_to_gt.png").exists()
        and (out_dir / "valid_mask.png").exists()
    )


def load_cached_metrics(out_dir: Path) -> dict[str, Any]:
    return load_json(out_dir / "work" / "metrics.json")


def run_unistitch_pair(
    row: dict[str, str],
    out_dir: Path,
    args: argparse.Namespace,
    net,
    extractor,
    matcher,
    device: torch.device,
) -> dict[str, Any]:
    work_dir = out_dir / "work"
    stitch_path = out_dir / "stitch_result.png"
    last_error: BaseException | None = None

    for edge in fallback_edges(args.max_input_edge):
        try:
            image1, image2, scale = prepare_images(Path(row["left_source"]), Path(row["right_source"]), edge)
            p1, p2, d1, d2, keypoint_info = extract_matched_keypoints(image1, image2, extractor, matcher, device)
            input1 = bgr_to_model_tensor(image1, device)
            input2 = bgr_to_model_tensor(image2, device)
            with torch.inference_mode():
                batch_out, ok = build_output_model(
                    net,
                    input1,
                    input2,
                    p1,
                    p2,
                    d1,
                    d2,
                    max_out_height=args.max_out_height,
                )
            if not ok:
                raise RetryablePairError("Predicted output canvas exceeded max_out_height.")

            fusion, warped_ref, warped_tgt, overlap_mask, overlap_psnr, overlap_ssim = tensor_outputs_to_images(batch_out)
            cv2.imwrite(str(stitch_path), fusion)
            cv2.imwrite(str(work_dir / "reference.png"), image1)
            cv2.imwrite(str(work_dir / "target.png"), image2)
            cv2.imwrite(str(work_dir / "warped_ref.png"), warped_ref)
            cv2.imwrite(str(work_dir / "warped_tgt.png"), warped_tgt)
            cv2.imwrite(str(work_dir / "overlap_mask.png"), overlap_mask)
            torch.save(
                {
                    "point1": p1.detach().cpu(),
                    "point2": p2.detach().cpu(),
                    "descriptor1": d1.detach().cpu(),
                    "descriptor2": d2.detach().cpu(),
                    "info": keypoint_info,
                },
                work_dir / "keypoints.pt",
            )
            return {
                "input_height": int(image1.shape[0]),
                "input_width": int(image1.shape[1]),
                "input_scale": scale,
                "max_input_edge": edge,
                "overlap_psnr": overlap_psnr,
                "overlap_ssim": overlap_ssim,
                **keypoint_info,
            }
        except Exception as exc:
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not (is_oom_error(exc) or isinstance(exc, RetryablePairError)):
                raise

    assert last_error is not None
    raise last_error


def process_pair(
    row: dict[str, str],
    args: argparse.Namespace,
    net,
    extractor,
    matcher,
    device: torch.device,
    niqe_metric,
    lpips_metric,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = final_pair_dir(row, args.result_root) / METHOD
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = work_dir / "metrics.json"
    status_path = work_dir / "method_status.json"
    stitch_path = out_dir / "stitch_result.png"

    if cache_hit(out_dir, args):
        return load_cached_metrics(out_dir)

    result = base_row(row, out_dir, args)
    status = {
        "method": METHOD,
        "pair_name": row["pair_name"],
        "scene": row["scene"],
        "pair_id": row["pair_id"],
        "success": False,
        "runtime_seconds": None,
        "failure_reason": "",
    }

    try:
        if args.force:
            for path in (stitch_path, out_dir / "aligned_to_gt.png", out_dir / "valid_mask.png"):
                if path.exists():
                    path.unlink()
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        render_info = run_unistitch_pair(row, out_dir, args, net, extractor, matcher, device)
        eval_info = evaluate_raw(
            stitch_path,
            Path(row["gt_path"]),
            out_dir,
            niqe_metric,
            lpips_metric,
            feature_max_side=args.feature_max_side,
            min_alignment_inliers=args.min_alignment_inliers,
            min_valid_ratio=args.min_valid_ratio,
            min_niqe_side=args.min_niqe_side,
            valid_black_threshold=args.valid_black_threshold,
            lpips_max_side=args.lpips_max_side,
        )
        runtime = time.perf_counter() - started
        result.update(eval_info)
        result.update(render_info)
        result.update(
            {
                "status": "success",
                "failure_reason": "",
                "runtime_seconds": runtime,
                "raw_path": str(stitch_path),
                "cpp_mdr": math.nan,
                "cpp_warping_residual_avg": math.nan,
                "cpp_warping_residual_sd": math.nan,
            }
        )
        status.update({"success": True, "runtime_seconds": runtime, "raw_path": str(stitch_path), **render_info})
    except Exception as exc:
        runtime = time.perf_counter() - started
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["runtime_seconds"] = runtime
        status.update(
            {
                "success": False,
                "failure_reason": result["failure_reason"],
                "failure_traceback": traceback.format_exc(),
                "runtime_seconds": runtime,
            }
        )
        (work_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if args.stop_on_error:
            write_json(status_path, status)
            write_json(metrics_path, result)
            raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(status_path, status)
    write_json(metrics_path, result)
    return result


def select_rows(rows: list[dict[str, str]], scenes: list[str] | None, limit: int) -> list[dict[str, str]]:
    if scenes:
        wanted = set(scenes)
        rows = [row for row in rows if row["scene"] in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [
        row
        for row in rows
        if row.get("status") == "success"
        and all(finite(row.get(key)) for key in ("mdr", "niqe", "psnr", "ssim", "lpips", "rmse"))
    ]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in successes if finite(row.get(key))]

    summary: dict[str, Any] = {
        "method": METHOD,
        "total_runs": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "failure_rate": (len(rows) - len(successes)) / len(rows) if rows else math.nan,
    }
    for metric in ("mdr", "niqe", "psnr", "ssim", "lpips", "rmse", "runtime"):
        key = "runtime_seconds" if metric == "runtime" else metric
        metric_values = values(key)
        summary[f"mean_{metric}"] = mean(metric_values) if metric_values else math.nan
        summary[f"median_{metric}"] = median(metric_values) if metric_values else math.nan
    return summary


def write_global_outputs(global_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    global_dir.mkdir(parents=True, exist_ok=True)
    write_csv(global_dir / "per_pair_metrics.csv", rows, PER_PAIR_FIELDS)
    write_csv(global_dir / "summary_all.csv", [summarize(rows)], SUMMARY_FIELDS)
    summary = {
        "method": METHOD,
        "manifest": str(args.manifest),
        "result_root": str(args.result_root),
        "checkpoint": str(args.checkpoint),
        **summarize(rows),
    }
    write_json(global_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UniStitch on HD3D-style two-view GT manifests.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_HD3D_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-input-edge", type=int, default=2048)
    parser.add_argument("--max-out-height", type=int, default=8000)
    parser.add_argument("--feature-max-side", type=int, default=1800)
    parser.add_argument("--min-alignment-inliers", type=int, default=12)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    parser.add_argument("--min-niqe-side", type=int, default=96)
    parser.add_argument("--valid-black-threshold", type=int, default=5)
    parser.add_argument("--lpips-max-side", type=int, default=1024)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed()
    device = resolve_device(args.device)
    args.manifest = args.manifest.resolve()
    args.result_root = args.result_root.resolve()
    args.checkpoint = args.checkpoint.resolve()

    rows = select_rows(read_manifest(args.manifest), args.scene, args.limit)
    global_dir = args.result_root / "_global_work" / METHOD
    net = load_network(args.checkpoint, device)
    extractor, matcher = load_feature_models(device)
    niqe_metric, metric_device = load_niqe_metric(args.device)
    lpips_metric, _ = load_lpips_metric(metric_device)

    metrics_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['pair_name']} {METHOD}", flush=True)
        result = process_pair(row, args, net, extractor, matcher, device, niqe_metric, lpips_metric)
        metrics_rows.append(result)
        print(f"  -> {result.get('status')}", flush=True)
        write_global_outputs(global_dir, metrics_rows, args)

    write_global_outputs(global_dir, metrics_rows, args)
    successes = sum(1 for row in metrics_rows if row.get("status") == "success")
    print(f"\nDone. {successes}/{len(metrics_rows)} succeeded. Wrote UniStitch HD3D-style metrics to {global_dir}")
    return 0 if successes == len(metrics_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
