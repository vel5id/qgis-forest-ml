import os
import time
import psutil
import joblib
import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from tqdm import tqdm
from multiprocessing import freeze_support
from rasterio.warp import calculate_default_transform

# new imports for texture features
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# === Parameters ===
mosaic_path      = r"C:\Users\vladi\Downloads\AllForScience\results\Калининское 2 - Валидация\result.tif"
model_path       = "./multiclass_main_without_data_leak_0.85weight_birch_300_trees_modified_256levels.pkl"
output_class_map = r"C:\Users\vladi\Downloads\AllForScience\results\Калининское 2 - Валидация\class_map.tif"
output_birch     = r"C:\Users\vladi\Downloads\AllForScience\results\Калининское 2 - Валидация\birch_mask.tif"
output_excel     = r"C:\Users\vladi\Downloads\AllForScience\results\Калининское 2 - Валидация\birch_report_0.85weight_birch256levels.xlsx"

patch_size       = 64  # size of each patch in pixels
stride           = 64  # step between patches
block_size       = 100 # number of patches per processing block
temp_threshold   = 70.0 # °C, CPU temperature throttle threshold
cool_factor      = 0.5  # seconds delay per °C above threshold
n_workers        = max(1, int((os.cpu_count() or 1) * 0.8))  # number of parallel workers
BEST_THRESH      = 0.537 # probability threshold for birch class

# number of quantization levels for GLCM and LBP (matches model name)
LEVELS = 8

def predict_with_threshold(proba_block, t1=BEST_THRESH):
    """Apply threshold to birch probabilities and assign other classes by highest remaining probability."""
    proba_block = np.asarray(proba_block)
    # Vectorized: find argmax among classes [2..6] for all samples
    other_labels = np.argmax(proba_block[:, 1:], axis=1) + 2
    # Apply threshold: if birch probability >= t1, assign 1; otherwise use other_labels
    y_pred = np.where(proba_block[:, 0] >= t1, 1, other_labels)
    return y_pred.tolist()


def get_cpu_temp():
    """Retrieve current CPU temperature or return None if unavailable."""
    try:
        for entries in psutil.sensors_temperatures().values():
            for e in entries:
                if e.current:
                    return e.current
    except:
        pass
    return None


def init_worker():
    """Initialize worker process: load the model and open the raster source."""
    global model, src
    model = joblib.load(model_path)
    src   = rasterio.open(mosaic_path)


def extract_features(patch):
    """Convert RGB patch to color, Haralick, and LBP feature vector."""
    R, G, B = patch[0].astype(float), patch[1].astype(float), patch[2].astype(float)

    # 1) Spectral color features
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    feats = [mR, mG, mB, exg, exr, exgr]

    # 2) Haralick (GLCM) on the green channel
    patchG = (G * (LEVELS - 1) / (G.max() + 1e-6)).astype(np.uint8)
    glcm = graycomatrix(
        patchG,
        distances=[1, 5],
        angles=[0, np.pi/2],
        levels=LEVELS,
        symmetric=True,
        normed=True
    )
    for prop in ("contrast", "correlation"):
        feats.append(graycoprops(glcm, prop).mean())

    # 3) LBP histogram (uniform, P=8, R=1)
    lbp = local_binary_pattern(patchG, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(hist.tolist())

    return feats


def process_block(coords):
    """Read patches by coordinates, extract features, predict probabilities, and apply threshold."""
    feats = []
    for top, left in coords:
        win   = Window(left, top, patch_size, patch_size)
        patch = src.read(window=win)
        feats.append(extract_features(patch))
    proba  = model.predict_proba(feats)
    labels = predict_with_threshold(proba)
    return list(zip(coords, labels))


def count_birch_clusters(birch_mask, patch_size):
    """Count connected birch clusters in the birch mask grid."""
    H, W = birch_mask.shape
    M, N = H // patch_size, W // patch_size
    # Vectorized: reshape to (M, patch_size, N, patch_size), then take max over patch dimensions
    trimmed = birch_mask[:M * patch_size, :N * patch_size]
    reshaped = trimmed.reshape(M, patch_size, N, patch_size)
    grid = (reshaped.max(axis=(1, 3)) == 1).astype(np.uint8)

    struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labels, n = label(grid, structure=struct)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "total_clusters": int(n),
        "single_patch":   int((sizes == 1).sum()),
        "multi_patch":    int((sizes > 1).sum()),
        "cluster_sizes":  sizes.tolist()
    }


def main():
    start_time = time.time()
    freeze_support()
    # lower process priority to reduce system impact
    try:
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except:
        pass

    if not os.path.isfile(mosaic_path):
        raise FileNotFoundError(f"Mosaic not found: {mosaic_path}")

    # --- Read metadata and compute area in meters ---
    with rasterio.open(mosaic_path) as tmp:
        meta = tmp.meta.copy()
        H, W = tmp.height, tmp.width
        deg_dx, deg_dy = abs(tmp.res[0]), abs(tmp.res[1])
        transform_m, Wm, Hm = calculate_default_transform(
            tmp.crs, "EPSG:3857", W, H, *tmp.bounds
        )
        dx, dy = transform_m.a, -transform_m.e
        pixel_area_m2 = dx * dy
        image_area_m2 = pixel_area_m2 * Wm * Hm

    # prepare empty arrays for output maps
    class_map = np.zeros((H, W), dtype=np.uint8)
    birch_map = np.zeros((H, W), dtype=np.uint8)

    # generate all patch coordinates and split into blocks
    all_coords = [
        (i, j)
        for i in range(0, H - patch_size + 1, stride)
        for j in range(0, W - patch_size + 1, stride)
    ]
    blocks = [all_coords[i:i+block_size] for i in range(0, len(all_coords), block_size)]

    # process blocks in parallel
    pbar = tqdm(total=len(blocks), desc="Blocks")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as exe:
        futures = {exe.submit(process_block, blk): blk for blk in blocks}
        for fut in as_completed(futures):
            for (top, left), lbl in fut.result():
                class_map[top:top+patch_size, left:left+patch_size] = lbl
                birch_map[top:top+patch_size, left:left+patch_size] = (1 if lbl == 1 else 0)
            pbar.update(1)
            # thermal throttling if CPU exceeds threshold
            t = get_cpu_temp()
            if t and t > temp_threshold:
                sl = min((t - temp_threshold) * cool_factor, 5.0)
                pbar.write(f"[WARN] CPU {t:.1f}°C > {temp_threshold}°C → sleep {sl:.1f}s")
                time.sleep(sl)
    pbar.close()

    # --- Save GeoTIFF outputs ---
    meta.update(count=1, dtype="uint8")
    with rasterio.open(output_class_map, 'w', **meta) as dst:
        dst.write(class_map, 1)
    with rasterio.open(output_birch, 'w', **meta) as dst:
        dst.write(birch_map, 1)

    # count clusters and compute summary statistics
    stats      = count_birch_clusters(birch_map, patch_size)
    mdl        = joblib.load(model_path)
    error_prob = 1.0 - mdl.oob_score_ if hasattr(mdl, "oob_score_") else None

    # compute seedling density per m²
    num_seedlings  = stats["total_clusters"]
    density_per_m2 = num_seedlings / image_area_m2

    # export summary to Excel
    runtime = time.time() - start_time
    df = pd.DataFrame([{ ... }])  # fields: image dimensions, resolutions, areas, clusters, density, runtime
    df.to_excel(output_excel, index=False)

    print("Done! Maps saved and summary written to", output_excel)

if __name__ == "__main__":
    main()
