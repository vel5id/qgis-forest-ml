#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import Affine

# ==== Parameters ==== 
BASE_ROOT   = Path("C:/Users/vladi/Downloads/AllForScience/RGB - Новая модель- Без Data Leak/Images/new_ver/base")
AUG_ROOT    = Path("C:/Users/vladi/Downloads/AllForScience/RGB - Новая модель- Без Data Leak/Images/new_ver")

# Split ratios (sum should be <= 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Directory name of birch class
BIRCH_DIR   = "birch_inner"

# Noise levels for birch only
BIRCH_NOISE_LEVEL = [0.005, 0.01]

# Spatial transforms: map name -> (function, rotation_k)
SPATIAL_TRANSFORMS = {
    "orig":   (lambda x: x,              0),
    "flip_h": (lambda x: x[:, :, ::-1], 0),
    "flip_v": (lambda x: x[:, ::-1, :], 0),
    "rot90":  (lambda x: np.rot90(x, 1, axes=(1,2)), 1),
    "rot180": (lambda x: np.rot90(x, 2, axes=(1,2)), 2),
    "rot270": (lambda x: np.rot90(x, 3, axes=(1,2)), 3),
}

# Affine rotation helper
def _rot_affine(transform: Affine, k: int, h: int, w: int) -> Affine:
    if k == 0:
        return transform
    return transform * Affine.rotation(90 * k) * Affine.translation(
        *({1: (0, -h), 2: (-w, -h), 3: (-w, 0)}[k])
    )

# Helpers: clip array to valid dtype range
def dtype_max(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return np.finfo(dtype).max

def dtype_min(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    return np.finfo(dtype).min

# Additional birch-only augmentations
def brightness_jitter(img: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.2)
    mx = dtype_max(img.dtype)
    mn = dtype_min(img.dtype)
    out = img.astype(np.float64) * factor
    return np.clip(out, mn, mx).astype(img.dtype)

def channel_jitter(img: np.ndarray) -> np.ndarray:
    mx = dtype_max(img.dtype)
    mn = dtype_min(img.dtype)
    out = img.astype(np.float64)
    out[1] = np.clip(out[1] * np.random.uniform(0.9,1.1), mn, mx)
    return out.astype(img.dtype)

# Core augment function

def augment(tile: np.ndarray, transform: Affine, class_name: str):
    ops = []
    bands, h, w = tile.shape
    # For birch, include spatial + noise + color + brightness
    if class_name == BIRCH_DIR:
        # 1) Spatial transforms
        for name, (fn, k) in SPATIAL_TRANSFORMS.items():
            arr2 = fn(tile)
            tf2 = _rot_affine(transform, k, h, w)
            ops.append((arr2, tf2, name))
        # 2) Two noise variants
        for sigma in random.sample(BIRCH_NOISE_LEVEL, 2):
            arr2 = np.clip(tile + np.random.normal(0, sigma, tile.shape),
                            dtype_min(tile.dtype), dtype_max(tile.dtype))
            ops.append((arr2.astype(tile.dtype), transform, f"noise_{sigma:.3f}"))
        # 3) One color jitter
        arr_c = channel_jitter(tile)
        ops.append((arr_c, transform, "color"))
        # 4) Two brightness jitters
        for i in range(2):
            arr_b = brightness_jitter(tile)
            ops.append((arr_b, transform, f"bright_{i}"))
        return ops
    # For other classes, only spatial
    for name, (fn, k) in SPATIAL_TRANSFORMS.items():
        arr2 = fn(tile)
        tf2 = _rot_affine(transform, k, h, w)
        ops.append((arr2, tf2, name))
    return ops

# Process list of files into destination folder

def process_files(src_files, dst_dir: Path, class_name: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for tif in src_files:
        with rasterio.open(tif) as src:
            tile      = src.read()  # Read once, preserving original dtype
            profile   = src.profile
            transform = src.transform
            nodata    = src.nodata
        for arr_aug, tf_aug, name in augment(tile, transform, class_name):
            prof = profile.copy()
            prof.update(transform=tf_aug, dtype=arr_aug.dtype, nodata=nodata)
            out_path = dst_dir / f"{tif.stem}_{name}.tif"
            with rasterio.open(out_path, 'w', **prof) as dst:
                dst.write(arr_aug)

# Split files into train/val/test

def split_files(files):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train = files[:n_train]
    val   = files[n_train:n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test

# Main

if __name__ == "__main__":
    random.seed(42)
    for sub in BASE_ROOT.iterdir():
        if not sub.is_dir():
            continue
        class_name = sub.name
        files = list(sub.glob("*.tif"))
        train_files, val_files, test_files = split_files(files)
        for split_name, split_list in [
            ("augmented_train", train_files),
            ("augmented_validation", val_files),
            ("augmented_test", test_files)
        ]:
            dst_sub = AUG_ROOT / split_name / f"{class_name}_augmented"
            print(f"{class_name} -> {split_name}: base={len(split_list)}", end=" -> ")
            process_files(split_list, dst_sub, class_name)
            aug_count = len(list(dst_sub.glob("*.tif")))
            print(f"augmented={aug_count}")
