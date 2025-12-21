#!/usr/bin/env python3
"""Convert ACDC 3D H5 volumes into 2D slice datasets.

Usage:
    python scripts/prepare_acdc_2d.py \
        --dataset-root ACDC-dataset/ACDC \
        --output-root  ACDC-dataset/ACDC/2d

Only slices that contain foreground pixels (label > 0) are kept.
Output layout:
    <output-root>/image/{train,val}/*.png
    <output-root>/mask/{train,val}/*.png
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ACDC 3D H5 volumes into 2D png slices.")
    parser.add_argument("--dataset-root", type=Path, default=Path("ACDC-dataset/ACDC"),
                        help="Root folder that contains data/ and train/val list files.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Destination root for the 2D slices. Default: <dataset-root>/2d")
    parser.add_argument("--min-foreground", type=int, default=1,
                        help="Minimum number of foreground pixels required to keep a slice.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files if they already exist.")
    return parser.parse_args()


def read_split_list(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Split list not found: {file_path}")
    with file_path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_output_tree(root: Path, splits: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    tree: Dict[str, Dict[str, Path]] = {"image": {}, "mask": {}}
    for kind in ("image", "mask"):
        for split in splits:
            out_dir = root / kind / split
            out_dir.mkdir(parents=True, exist_ok=True)
            tree[kind][split] = out_dir
    return tree


def normalize_to_uint8(volume_slice: np.ndarray) -> np.ndarray:
    min_v = volume_slice.min()
    max_v = volume_slice.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(volume_slice, dtype=np.uint8)
    normalized = (volume_slice - min_v) / (max_v - min_v)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def save_slice(image_arr: np.ndarray, mask_arr: np.ndarray, out_dir: Path,
               base_name: str, slice_idx: int, overwrite: bool) -> Tuple[Path, Path]:
    img_name = f"{base_name}_slice{slice_idx:03d}.png"
    img_path = out_dir["image"] / img_name
    mask_path = out_dir["mask"] / img_name

    if not overwrite and img_path.exists() and mask_path.exists():
        return img_path, mask_path

    image_uint8 = normalize_to_uint8(image_arr)
    mask_uint8 = mask_arr.astype(np.uint8)

    Image.fromarray(image_uint8).save(img_path)
    Image.fromarray(mask_uint8).save(mask_path)
    return img_path, mask_path


def process_case(case_name: str, split: str, data_root: Path, out_tree: Dict[str, Dict[str, Path]],
                 min_foreground: int, overwrite: bool) -> Tuple[int, int]:
    h5_path = data_root / f"{case_name}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 file for {case_name}: {h5_path}")

    saved = 0
    skipped = 0
    with h5py.File(h5_path, "r") as handle:
        images = handle["image"]
        masks = handle["label"]
        depth = images.shape[0]
        for idx in range(depth):
            mask_slice = masks[idx]
            if np.count_nonzero(mask_slice) < min_foreground:
                skipped += 1
                continue
            image_slice = images[idx]
            save_slice(image_slice, mask_slice,
                       {"image": out_tree["image"][split], "mask": out_tree["mask"][split]},
                       case_name,
                       idx,
                       overwrite)
            saved += 1
    return saved, skipped


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = (args.output_root or dataset_root / "2d").resolve()

    data_dir = dataset_root / "data"
    train_list = dataset_root / "train.list"
    val_list = dataset_root / "val.list"

    splits = {
        "train": read_split_list(train_list),
        "val": read_split_list(val_list)
    }
    out_tree = ensure_output_tree(output_root, splits.keys())

    summary: Dict[str, Dict[str, int]] = {split: {"saved": 0, "skipped": 0} for split in splits}

    for split_name, case_ids in splits.items():
        iterator = tqdm(case_ids, desc=f"Processing {split_name}")
        for case in iterator:
            saved, skipped = process_case(case, split_name, data_dir, out_tree,
                                          args.min_foreground, args.overwrite)
            summary[split_name]["saved"] += saved
            summary[split_name]["skipped"] += skipped
            iterator.set_postfix(saved=summary[split_name]["saved"], skipped=summary[split_name]["skipped"])

    print("\nConversion finished. Summary:")
    for split_name, stats in summary.items():
        print(f"  {split_name}: kept {stats['saved']} slices, discarded {stats['skipped']} slices without foreground.")
    print(f"Slices stored under {output_root}")


if __name__ == "__main__":
    main()
