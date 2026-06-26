from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


DEFAULT_MANIFEST = Path(r"D:\StitchBench_Result\_global_work\_shared\manifest.csv")
DEFAULT_RESULT_ROOT = Path(r"D:\StitchBench_Result")
METHOD = "unistitch"
CATEGORIES = ("OBJ-GSP", "AANAP", "APAP", "CAVE", "DFW", "DHW", "GES", "LPC", "REW", "SEAGULL", "SVA", "SPHP")
PER_PAIR_COLUMNS = (
    "dataset",
    "category",
    "result_image",
    "mdr_rmse",
    "warping_residual_avg",
    "warping_residual_sd",
    "niqe",
    "status",
)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def load_niqe_metric(device: str):
    import pyiqa

    return pyiqa.create_metric("niqe", device=device)


def compute_niqe(metric, image_path: Path) -> float:
    if not image_path.exists():
        return math.nan
    try:
        score = metric(str(image_path))
    except Exception:
        import torch

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return math.nan
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        score = metric(tensor)
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    metric = None if args.skip_niqe else load_niqe_metric(args.device)
    rows: list[dict[str, Any]] = []
    for row in manifest:
        dataset = row["dataset"]
        category = row.get("category", "")
        method_dir = args.result_root / dataset / args.method
        result_image = method_dir / "stitch_result.png"
        metrics = read_metrics(method_dir / "work" / "metrics.json")
        niqe = math.nan
        if not result_image.exists():
            status = "missing_result"
        else:
            status = "ok_psnr_ssim_only"
            if metrics.get("status") == "failed":
                status = "failed"
            if metric is not None:
                try:
                    niqe = compute_niqe(metric, result_image)
                except Exception:
                    status = "failed"
        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "result_image": str(result_image),
                "mdr_rmse": math.nan,
                "warping_residual_avg": math.nan,
                "warping_residual_sd": math.nan,
                "niqe": niqe,
                "status": status,
                "overlap_psnr": metrics.get("overlap_psnr", ""),
                "overlap_ssim": metrics.get("overlap_ssim", ""),
            }
        )
    return rows


def write_by_category(output_root: Path, rows: list[dict[str, Any]]) -> None:
    category_rows = []
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        niqe_values = [float(row["niqe"]) for row in group if isinstance(row["niqe"], float)]
        psnr_values = [float(row["overlap_psnr"]) for row in group if row["overlap_psnr"] not in ("", None)]
        ssim_values = [float(row["overlap_ssim"]) for row in group if row["overlap_ssim"] not in ("", None)]
        category_rows.append(
            {
                "category": category,
                "total_count": len(group),
                "ok_count": len([row for row in group if row["status"] == "ok_psnr_ssim_only"]),
                "niqe_mean": finite_mean(niqe_values),
                "overlap_psnr_mean": finite_mean(psnr_values),
                "overlap_ssim_mean": finite_mean(ssim_values),
            }
        )
    write_csv(
        output_root / "by_category.csv",
        category_rows,
        ["category", "total_count", "ok_count", "niqe_mean", "overlap_psnr_mean", "overlap_ssim_mean"],
    )


def write_report(output_root: Path, rows: list[dict[str, Any]], method: str) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok_psnr_ssim_only"]
    niqe_mean = finite_mean([float(row["niqe"]) for row in ok_rows if isinstance(row["niqe"], float)])
    psnr_mean = finite_mean([float(row["overlap_psnr"]) for row in ok_rows if row["overlap_psnr"] not in ("", None)])
    ssim_mean = finite_mean([float(row["overlap_ssim"]) for row in ok_rows if row["overlap_ssim"] not in ("", None)])
    lines = [
        f"# {method} StitchBench General Report",
        "",
        f"- Total scenes: {len(rows)}",
        f"- OK PSNR/SSIM scenes: {len(ok_rows)}",
        f"- Missing/failed scenes: {len(rows) - len(ok_rows)}",
        f"- Mean overlap PSNR: {format_float(psnr_mean)}",
        f"- Mean overlap SSIM: {format_float(ssim_mean)}",
        f"- Mean NIQE: {format_float(niqe_mean)}",
        "",
        "MDR is intentionally left blank for UniStitch in this first integration because the available runner exposes warped overlap images, not the OBJ-GSP mesh-RMSE interface.",
        "Use `D:\\StitchBench_Result\\_global_work\\unistitch\\metrics.csv` for overlap PSNR/SSIM.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate UniStitch StitchBench outputs for MethodManagement.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-niqe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.result_root = args.result_root.resolve()
    output_root = args.result_root / "_global_work" / args.method
    rows = evaluate(args)
    write_csv(output_root / "per_pair.csv", rows, PER_PAIR_COLUMNS)
    write_by_category(output_root, rows)
    write_report(output_root, rows, args.method)
    print(f"Wrote UniStitch evaluation files to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
