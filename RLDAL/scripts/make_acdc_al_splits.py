import argparse
import os
import random
import numpy as np


# Fixed random seed for reproducibility
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ACDC active-learning splits and save to data/acdc_al_splits.npy"
    )
    parser.add_argument(
        "--train-dir",
        default=os.path.join("ACDC-dataset", "ACDC", "2d", "image", "train"),
        help="Path to ACDC train images (PNG files).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "acdc_al_splits.npy"),
        help="Output npy path for the splits dictionary.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for shuffling."
    )
    parser.add_argument(
        "--pt-ratio", type=float, default=0.10, help="Pre-train percentage of train set."
    )
    parser.add_argument(
        "--tq-ratio", type=float, default=0.20, help="Teacher queue percentage of train set."
    )
    parser.add_argument(
        "--ds-size", type=int, default=10, help="Fixed number of images for d_s (state set)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")

    image_ids = [f for f in os.listdir(args.train_dir) if f.lower().endswith(".png")]
    if not image_ids:
        raise RuntimeError(f"No PNG images found in {args.train_dir}")

    image_ids = sorted(image_ids)
    random.seed(args.seed)
    random.shuffle(image_ids)

    total = len(image_ids)
    pt_size = int(total * args.pt_ratio)
    tq_size = int(total * args.tq_ratio)
    ds_size = args.ds_size
    dt_size = total - pt_size - ds_size - tq_size

    if dt_size < 0:
        raise ValueError(
            f"Split sizes exceed dataset: total={total}, pt={pt_size}, d_s={ds_size}, t_q={tq_size}"
        )

    splits = {
        "p_t": image_ids[:pt_size],
        "d_s": image_ids[pt_size : pt_size + ds_size],
        "t_q": image_ids[pt_size + ds_size : pt_size + ds_size + tq_size],
        "d_t": image_ids[pt_size + ds_size + tq_size :],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, splits)

    print("Saved splits to", args.output)
    print(
        f"Counts -> total: {total}, p_t: {len(splits['p_t'])}, d_s: {len(splits['d_s'])}, "
        f"t_q: {len(splits['t_q'])}, d_t: {len(splits['d_t'])}"
    )


if __name__ == "__main__":
    main()
