"""
Remove duplicate images in raw_faces by comparing file content hash.
Keeps the first occurrence, deletes duplicates.
"""

import os
import hashlib
from pathlib import Path

_base = "/sessions/epic-zen-shannon/mnt/facelift_pipeline/data"
if not os.path.exists(_base):
    _base = r"D:\zmm\facelift_pipeline\data"

RAW_DIR = os.path.join(_base, "raw_faces")


def file_hash(filepath):
    """MD5 hash of file content."""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    files = sorted(os.listdir(RAW_DIR))
    print("Total files: " + str(len(files)))

    seen = {}  # hash -> first filename
    duplicates = []

    for i, fname in enumerate(files):
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        h = file_hash(fpath)
        if h in seen:
            duplicates.append((fname, seen[h]))
        else:
            seen[h] = fname

        if (i + 1) % 500 == 0:
            print("  Checked " + str(i + 1) + "/" + str(len(files)))

    print("\nFound " + str(len(duplicates)) + " duplicates")

    if duplicates:
        print("\nExamples:")
        for dup, orig in duplicates[:10]:
            print("  " + dup + "  ==  " + orig)

        # Delete duplicates
        for dup, orig in duplicates:
            os.remove(os.path.join(RAW_DIR, dup))

        remaining = len(os.listdir(RAW_DIR))
        print("\nDeleted " + str(len(duplicates)) + " duplicates")
        print("Remaining: " + str(remaining) + " images")
    else:
        print("No duplicates found.")


if __name__ == "__main__":
    main()
