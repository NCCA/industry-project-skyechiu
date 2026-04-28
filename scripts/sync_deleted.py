"""Find files missing from image/ and delete matching files from all other subfolders."""
from pathlib import Path
import sys

SUBDIRS = ["depth", "depth_lr_8bit", "depth_lr_16bit", "normal", "normal_dsine",
           "mask", "mask_lr", "opacity"]

for split in ["train", "val"]:
    root = Path(f"data/dataset/{split}")
    image_dir = root / "image"
    if not image_dir.exists():
        continue

    # get stems that exist in image/
    image_stems = {p.stem for p in image_dir.glob("*.png")}
    image_stems |= {p.stem for p in image_dir.glob("*.jpg")}

    deleted = []
    for sub in SUBDIRS:
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for p in sorted(sub_dir.glob("*.*")):
            if p.stem not in image_stems:
                deleted.append(p)

    if not deleted:
        print(f"{split}: nothing to delete")
        continue

    print(f"\n{split}: will delete {len(deleted)} files:")
    for p in deleted:
        print(f"  {p.relative_to(root)}")

    if "--dry" not in sys.argv:
        for p in deleted:
            p.unlink()
        print(f"  => deleted {len(deleted)} files")
    else:
        print("  (dry run, pass without --dry to actually delete)")
