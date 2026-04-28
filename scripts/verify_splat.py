#!/usr/bin/env python3
"""
Step 6: Verify Gaussian Splat (.ply) files -- point counts and basic stats.

Matches facelift_pipeline.ipynb Cell 9.
Simple scan: enumerate splat directories, load each .ply header, report stats.

Usage:
    python scripts/verify_splat.py
    python scripts/verify_splat.py --splat-dir data/splats
    python scripts/verify_splat.py --report eval/splat_report.json
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_ply_header(ply_path):
    """Read PLY header to get vertex count and properties."""
    info = {"num_points": 0, "properties": [], "valid": False, "error": None}
    try:
        with open(ply_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
                if len(header) > 10000:
                    info["error"] = "header too large"
                    return info

            for line_bytes in header.split(b"\n"):
                line_str = line_bytes.decode("ascii", errors="replace").strip()
                if line_str.startswith("element vertex"):
                    info["num_points"] = int(line_str.split()[-1])
                elif line_str.startswith("property"):
                    info["properties"].append(line_str)

        info["valid"] = info["num_points"] > 0
        info["file_size_mb"] = ply_path.stat().st_size / (1024 * 1024)
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    parser = argparse.ArgumentParser(description="Verify Gaussian Splat files")
    parser.add_argument("--splat-dir", type=str, default=None)
    parser.add_argument("--report", type=str, default=None,
                        help="Save verification report to JSON")
    args = parser.parse_args()

    config = load_config()
    splat_dir = Path(args.splat_dir or config["paths"]["splat_output"]).resolve()

    print("=" * 60)
    print("  Gaussian Splat Verification")
    print("=" * 60)
    print(f"  Directory: {splat_dir}")

    if not splat_dir.exists():
        print(f"  ERROR: directory does not exist")
        sys.exit(1)

    # Find all PLY files
    ply_files = sorted([
        d / "gaussians.ply"
        for d in splat_dir.iterdir()
        if d.is_dir() and (d / "gaussians.ply").exists()
    ])

    if not ply_files:
        print(f"  No gaussians.ply files found")
        sys.exit(1)

    print(f"  Found {len(ply_files)} PLY files\n")

    # Verify each
    results = []
    valid_count = 0
    point_counts = []

    for i, ply_path in enumerate(ply_files):
        name = ply_path.parent.name
        info = read_ply_header(ply_path)
        info["name"] = name
        results.append(info)

        if info["valid"]:
            valid_count += 1
            point_counts.append(info["num_points"])
            if (i + 1) % 50 == 0 or (i + 1) == len(ply_files):
                print(f"  [{i+1}/{len(ply_files)}] {name}: "
                      f"{info['num_points']:,} pts, "
                      f"{info.get('file_size_mb', 0):.1f} MB")
        else:
            print(f"  [{i+1}/{len(ply_files)}] {name}: INVALID - {info['error']}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Total:   {len(ply_files)}")
    print(f"  Valid:   {valid_count}")
    print(f"  Invalid: {len(ply_files) - valid_count}")

    if point_counts:
        arr = np.array(point_counts)
        print(f"  Points:  min={arr.min():,}  max={arr.max():,}  "
              f"median={int(np.median(arr)):,}  mean={int(arr.mean()):,}")

    # Save report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "total": len(ply_files),
            "valid": valid_count,
            "invalid": len(ply_files) - valid_count,
            "results": results,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
