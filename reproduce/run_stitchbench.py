from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim


REPO_ROOT = Path(__file__).resolve().parents[1]
CODES_DIR = REPO_ROOT / "Codes"
LIGHTGLUE_DIR = REPO_ROOT / "keypoint_tool" / "LightGlue"
DEFAULT_MANIFEST = Path(r"D:\StitchBench_Result\_global_work\_shared\manifest.csv")
DEFAULT_RESULT_ROOT = Path(r"D:\StitchBench_Result")
DEFAULT_CHECKPOINT = REPO_ROOT / "model" / "epoch_best_model.pth"
METHOD = "unistitch"
MAX_POINTS = 2048

for path in (CODES_DIR, LIGHTGLUE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lightglue import LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import rbd  # noqa: E402
from network import Network, build_output_model  # noqa: E402


class RetryablePairError(RuntimeError):
    pass


def setup_seed(seed: int = 114514) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_oom_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda" in text and "memory" in text


def fallback_edges(max_edge: int) -> list[int]:
    if max_edge <= 0:
        return [0]
    candidates = [max_edge, 1536, 1024]
    edges: list[int] = []
    for edge in candidates:
        if edge <= max_edge and edge not in edges:
            edges.append(edge)
    return edges or [max_edge]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type != "cuda":
        raise RuntimeError("UniStitch inference expects CUDA because the original warping code calls .cuda().")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    return device


def load_network(checkpoint_path: Path, device: torch.device) -> Network:
    net = Network(backbone_weights=None).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    net.load_state_dict(state_dict)
    net.fuse()
    net.eval()
    return net


def load_feature_models(device: torch.device) -> tuple[SuperPoint, LightGlue]:
    extractor = SuperPoint(max_num_keypoints=None).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def resize_long_edge(image: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    if max_edge <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image, 1.0
    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def prepare_images(image1_path: Path, image2_path: Path, max_edge: int) -> tuple[np.ndarray, np.ndarray, float]:
    image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)
    if image1 is None:
        raise FileNotFoundError(f"Cannot read image: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"Cannot read image: {image2_path}")
    image1, scale = resize_long_edge(image1, max_edge)
    image2, _ = resize_long_edge(image2, max_edge)
    if image1.shape[:2] != image2.shape[:2]:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    return image1, image2, scale


def bgr_to_model_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def bgr_to_lightglue_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.to(device)


def pad_or_truncate(points: torch.Tensor, descriptors: torch.Tensor, target: int = MAX_POINTS) -> tuple[torch.Tensor, torch.Tensor]:
    descriptor_dim = descriptors.shape[-1] if descriptors.ndim == 2 and descriptors.shape[-1] else 256
    count = int(points.shape[0]) if points is not None else 0
    if count <= 0:
        return torch.zeros(target, 2), torch.zeros(target, descriptor_dim)
    points = points.detach().float().cpu()
    descriptors = descriptors.detach().float().cpu()
    if count >= target:
        return points[:target], descriptors[:target]
    extra = target - count
    repeat = torch.arange(extra) % count
    return torch.cat([points, points[repeat]], dim=0), torch.cat([descriptors, descriptors[repeat]], dim=0)


@torch.inference_mode()
def extract_matched_keypoints(
    image1: np.ndarray,
    image2: np.ndarray,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    tensor1 = bgr_to_lightglue_tensor(image1, device)
    tensor2 = bgr_to_lightglue_tensor(image2, device)
    feats1_b = extractor.extract(tensor1)
    feats2_b = extractor.extract(tensor2)
    matches_b = matcher({"image0": feats1_b, "image1": feats2_b})
    feats1, feats2, matches = [rbd(x) for x in (feats1_b, feats2_b, matches_b)]
    match_idx = matches["matches"]
    if isinstance(match_idx, list):
        match_idx = match_idx[0]
    if match_idx.numel() > 0:
        idx1 = match_idx[:, 0].long()
        idx2 = match_idx[:, 1].long()
        points1 = feats1["keypoints"][idx1]
        points2 = feats2["keypoints"][idx2]
        desc1 = feats1["descriptors"][idx1]
        desc2 = feats2["descriptors"][idx2]
    else:
        points1 = feats1["keypoints"]
        points2 = feats2["keypoints"]
        desc1 = feats1["descriptors"]
        desc2 = feats2["descriptors"]

    points1, desc1 = pad_or_truncate(points1, desc1)
    points2, desc2 = pad_or_truncate(points2, desc2)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    points1[:, 0] = points1[:, 0] / max(w1 - 1, 1)
    points1[:, 1] = points1[:, 1] / max(h1 - 1, 1)
    points2[:, 0] = points2[:, 0] / max(w2 - 1, 1)
    points2[:, 1] = points2[:, 1] / max(h2 - 1, 1)

    info = {
        "keypoints1": int(feats1["keypoints"].shape[0]),
        "keypoints2": int(feats2["keypoints"].shape[0]),
        "matches": int(match_idx.shape[0]) if match_idx.ndim else 0,
        "used_points": MAX_POINTS,
    }
    return (
        points1.unsqueeze(0).to(device),
        points2.unsqueeze(0).to(device),
        desc1.unsqueeze(0).to(device),
        desc2.unsqueeze(0).to(device),
        info,
    )


def mask_psnr(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 1e-6:
        return math.nan
    image1 = image1 * mask / 255.0
    image2 = image2 * mask / 255.0
    rmse = math.sqrt(float(np.sum((image1 - image2) ** 2)) / denom)
    if rmse <= 1e-12:
        return math.inf
    return 20.0 * math.log10(1.0 / rmse)


def mask_ssim(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 1e-6:
        return math.nan
    _, ssim = compare_ssim(image1 * mask, image2 * mask, data_range=255, channel_axis=2, full=True)
    return float(np.sum(ssim * mask) / (denom + 1e-6))


def uint8_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def tensor_outputs_to_images(batch_out: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    output_ref = batch_out["output_tps_ref"]
    output_tgt = batch_out["output_tps_tgt"]
    ref = (output_ref[0, 0:3].detach().cpu().numpy().transpose(1, 2, 0) * 127.5)
    tgt = (output_tgt[0, 0:3].detach().cpu().numpy().transpose(1, 2, 0) * 127.5)
    mask = output_ref[0, 3:6] * output_tgt[0, 3:6]
    mask_np = mask.detach().cpu().numpy().transpose(1, 2, 0)
    fusion = ref * (ref / (ref + tgt + 1e-6)) + tgt * (tgt / (ref + tgt + 1e-6))
    psnr = mask_psnr(ref, tgt, mask_np)
    ssim = mask_ssim(ref, tgt, mask_np)
    return uint8_image(fusion), uint8_image(ref), uint8_image(tgt), uint8_image(mask_np * 255.0), psnr, ssim


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def existing_metrics(scene_dir: Path) -> dict[str, Any] | None:
    path = scene_dir / METHOD / "work" / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def process_pair(
    row: dict[str, str],
    args: argparse.Namespace,
    net: Network,
    extractor: SuperPoint,
    matcher: LightGlue,
    device: torch.device,
) -> dict[str, Any]:
    scene = row["dataset"]
    category = row.get("category", "")
    image_names = row["image_files"].split("|")[:2]
    if len(image_names) < 2:
        raise ValueError(f"Manifest row has fewer than two images: {scene}")
    image1_path = Path(row["data_dir"]) / image_names[0]
    image2_path = Path(row["data_dir"]) / image_names[1]
    scene_dir = args.out_root / scene / METHOD
    work_dir = scene_dir / "work"
    result_path = scene_dir / "stitch_result.png"

    if args.skip_existing and not args.force and result_path.exists():
        metrics = existing_metrics(args.out_root / scene)
        if metrics:
            metrics["status"] = metrics.get("status", "ok_existing")
            return metrics

    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_error: BaseException | None = None

    for edge in fallback_edges(args.max_input_edge):
        try:
            image1, image2, scale = prepare_images(image1_path, image2_path, edge)
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

            fusion, warped_ref, warped_tgt, overlap_mask, psnr, ssim = tensor_outputs_to_images(batch_out)
            cv2.imwrite(str(result_path), fusion)
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

            elapsed = time.perf_counter() - started
            metrics = {
                "dataset": scene,
                "category": category,
                "status": "ok",
                "image1": str(image1_path),
                "image2": str(image2_path),
                "result_image": str(result_path),
                "overlap_psnr": psnr,
                "overlap_ssim": ssim,
                "input_height": int(image1.shape[0]),
                "input_width": int(image1.shape[1]),
                "input_scale": scale,
                "max_input_edge": edge,
                "elapsed_sec": elapsed,
                **keypoint_info,
            }
            save_json(work_dir / "metrics.json", metrics)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return metrics
        except Exception as exc:  # retry only memory/canvas-size issues
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not (is_oom_error(exc) or isinstance(exc, RetryablePairError)):
                raise

    assert last_error is not None
    raise last_error


def select_rows(rows: list[dict[str, str]], scenes: list[str] | None, limit: int) -> list[dict[str, str]]:
    if scenes:
        wanted = set(scenes)
        rows = [row for row in rows if row["dataset"] in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def write_global_outputs(global_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fieldnames = [
        "dataset",
        "category",
        "status",
        "image1",
        "image2",
        "result_image",
        "overlap_psnr",
        "overlap_ssim",
        "input_height",
        "input_width",
        "input_scale",
        "max_input_edge",
        "elapsed_sec",
        "keypoints1",
        "keypoints2",
        "matches",
        "used_points",
        "error",
    ]
    write_csv(global_dir / "metrics.csv", rows, fieldnames)
    psnr = [float(row.get("overlap_psnr", math.nan)) for row in rows]
    ssim = [float(row.get("overlap_ssim", math.nan)) for row in rows]
    summary = {
        "method": METHOD,
        "manifest": str(args.manifest),
        "result_root": str(args.out_root),
        "checkpoint": str(args.checkpoint),
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("status") in {"ok", "ok_existing"}),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "mean_overlap_psnr": finite_mean(psnr),
        "mean_overlap_ssim": finite_mean(ssim),
    }
    save_json(global_dir / "summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UniStitch on StitchBench General manifest pairs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-input-edge", type=int, default=2048)
    parser.add_argument("--max-out-height", type=int, default=8000)
    parser.add_argument("--scene", action="append", default=None, help="Run one scene; can be repeated.")
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
    args.out_root = args.out_root.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()

    rows = select_rows(load_manifest(args.manifest), args.scene, args.limit)
    global_dir = args.out_root / "_global_work" / METHOD
    global_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(args.checkpoint, device)
    extractor, matcher = load_feature_models(device)

    metrics_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        scene = row["dataset"]
        print(f"[{index}/{len(rows)}] {scene}", flush=True)
        try:
            metrics_rows.append(process_pair(row, args, net, extractor, matcher, device))
        except Exception as exc:
            image_names = row.get("image_files", "").split("|")[:2]
            failed = {
                "dataset": scene,
                "category": row.get("category", ""),
                "status": "failed",
                "image1": str(Path(row["data_dir"]) / image_names[0]) if image_names else "",
                "image2": str(Path(row["data_dir"]) / image_names[1]) if len(image_names) > 1 else "",
                "result_image": str(args.out_root / scene / METHOD / "stitch_result.png"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            metrics_rows.append(failed)
            failure_dir = args.out_root / scene / METHOD / "work"
            save_json(failure_dir / "metrics.json", failed)
            (failure_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if args.stop_on_error:
                write_global_outputs(global_dir, metrics_rows, args)
                raise

    write_global_outputs(global_dir, metrics_rows, args)
    print(f"Wrote UniStitch metrics to {global_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
