"""
Clean orphan downstream files: those whose raw_faces entry has been deleted.

After you manually prune raw_faces (e.g. removing samples whose FaceLift
render looked bad), this script removes the corresponding files from
cropped_faces, rgb_rendered, depth_maps, normal_maps, opacity_maps and splats
so they don't get pulled back into the training set by downstream cells.

Usage:
    python scripts/clean_orphans.py                  # dry-run, show what would be deleted
    python scripts/clean_orphans.py --apply          # actually delete
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

# Downstream folders that mirror raw_faces (with _face suffix appended on stem)
PNG_DIRS = ["cropped_faces", "rgb_rendered", "depth_maps", "normal_maps", "opacity_maps"]
SPLAT_DIR = "splats"  # subdirectory per sample, contains gaussians.ply


def base_stem(stem: str) -> str:
    """Strip the _face suffix added by prepare_images.py."""
    return stem[:-5] if stem.endswith("_face") else stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Actually delete files (default: dry-run)")
    args = p.parse_args()

    raw_dir = DATA / "raw_faces"
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found")
        return

    raw_stems = {p.stem for p in raw_dir.glob("*") if p.is_file()}
    print(f"raw_faces: {len(raw_stems)} files\n")

    total_orphans = 0
    actions = []  # (path, kind)

    # Check each PNG dir
    for d_name in PNG_DIRS:
        d = DATA / d_name
        if not d.exists():
            print(f"  [skip] {d_name} not found")
            continue
        files = list(d.glob("*.png"))
        orphans = [f for f in files if base_stem(f.stem) not in raw_stems]
        print(f"  {d_name:15s}: {len(files):4d} total, {len(orphans):4d} orphans")
        for f in orphans:
            actions.append((f, "file"))
        total_orphans += len(orphans)

    # Splats subdirs
    sp = DATA / SPLAT_DIR
    if sp.exists():
        subs = [d for d in sp.iterdir() if d.is_dir()]
        sp_orphans = [d for d in subs if base_stem(d.name) not in raw_stems]
        print(f"  {SPLAT_DIR:15s}: {len(subs):4d} dirs , {len(sp_orphans):4d} orphans")
        for d in sp_orphans:
            actions.append((d, "dir"))
        total_orphans += len(sp_orphans)

    print(f"\nTotal orphan items: {total_orphans}")

    if total_orphans == 0:
        print("Nothing to clean. You're good to run the pipeline.")
        return

    # Show first 10 examples
    print("\nFirst 10 to delete:")
    for path, kind in actions[:10]:
        print(f"  [{kind}] {path.relative_to(PROJECT_ROOT)}")
    if len(actions) > 10:
        print(f"  ... and {len(actions) - 10} more")

    if not args.apply:
        print("\n[DRY-RUN] Nothing was deleted.")
        print("Run again with --apply to actually delete:")
        print("  python scripts/clean_orphans.py --apply")
        return

    # Actually delete
    print("\nDeleting...")
    n_ok, n_fail = 0, 0
    for path, kind in actions:
        try:
            if kind == "file":
                path.unlink()
            else:
                shutil.rmtree(path)
            n_ok += 1
        except Exception as e:
            print(f"  FAIL: {path}: {e}")
            n_fail += 1
    print(f"\nDeleted {n_ok} items, {n_fail} failures")
    print("Now you can run: python run_pipeline.py --start-from 2 --stop-at 4")


if __name__ == "__main__":
    main()
