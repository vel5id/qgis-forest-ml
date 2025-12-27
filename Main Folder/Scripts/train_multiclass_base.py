import os
import pandas as pd
import numpy as np
import rasterio
import joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve
)
from tqdm import tqdm

# --- Parameters -----------------------------------------------------------
LABELS_CSV   = "labels_multiclass_all.csv"
BASE_DIR     = Path("../Images/new_ver")   # the same BASE_DIR used in the CSV generator
RANDOM_SEED  = 42
N_TREES      = 300
# -------------------------------------------------------------------------

def extract_features(path: str) -> list[float]:
    """Extract simple spectral features from an RGB patch."""
    with rasterio.open(path) as src:
        img = src.read().astype(float)
    R, G, B = img[0], img[1], img[2]
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    return [mR, mG, mB, exg, exr, exgr]


def build_dataset(df_split: pd.DataFrame):
    """Given a DataFrame with columns 'filepath' and 'class', builds X (an N×6 array) and y."""
    X = []
    paths = [str(BASE_DIR / fp) for fp in df_split.filepath]
    for full_path in paths:
        feats = extract_features(full_path)
        X.append(feats)
    y = df_split["class"].astype(int).tolist()
    return np.array(X), y


def get_original_ids(relpaths):
    """From a list of relative paths, returns a set of (class_name, patch_id) pairs,
    where patch_id is the original patch name before augmentation."""
    ids = set()
    for rp in relpaths:
        parts = Path(rp).parts
        if len(parts) < 3:
            continue
        class_aug = parts[1]
        class_name = class_aug.removesuffix("_augmented")
        filename = parts[2]
        stem = Path(filename).stem
        patch_id = stem.rsplit("_", 1)[0]
        ids.add((class_name, patch_id))
    return ids


def main() -> None:
    # 1) Read the CSV
    df = pd.read_csv(LABELS_CSV)

    # 2) Strict split based on filepath prefix
    df_train = df[df.filepath.str.startswith("augmented_train/")].reset_index(drop=True)
    df_val   = df[df.filepath.str.startswith("augmented_validation/")].reset_index(drop=True)
    df_test  = df[df.filepath.str.startswith("augmented_test/")].reset_index(drop=True)

    # 2.5) Smoke tests for data leakage
    ids_train = get_original_ids(df_train.filepath)
    ids_val   = get_original_ids(df_val.filepath)
    ids_test  = get_original_ids(df_test.filepath)
    assert ids_train.isdisjoint(ids_val),  f"LEAK between TRAIN/VAL: {ids_train & ids_val}"
    assert ids_train.isdisjoint(ids_test), f"LEAK between TRAIN/TEST: {ids_train & ids_test}"
    assert ids_val.isdisjoint(ids_test),   f"LEAK between VAL/TEST:   {ids_val & ids_test}"
    print("✅ No data leakage detected between splits.\n")

    print(f"Train samples: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

    # 3) Build features
    print("Building train features")
    X_train_f, y_train = build_dataset(df_train)
    print("Building validation features")
    X_val_f,   y_val   = build_dataset(df_val)
    print("Building test features")
    X_test_f,  y_test  = build_dataset(df_test)

    # 4) Manual class weights (no automatic balancing)
    classes = np.unique(y_train)
    # assign a base weight of 1.0 to all classes
    class_w = {c: 1.0 for c in classes}

    # boost birch (label=1) by a factor
    boost_factor = 0.85
    if 1 in class_w:
        class_w[1] *= boost_factor

    print("Using manual class weights:", class_w, "\n")

    # 5) Train RandomForest with progress bar
    print("Training RandomForest with progress bar:")
    clf = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        class_weight=class_w,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        oob_score=True
    )
    for i in tqdm(range(N_TREES), desc="Building trees", unit="tree"):
        clf.n_estimators = i + 1
        clf.fit(X_train_f, y_train)

    # 5.1) Collect OOB errors **on a clone** to avoid altering `clf`
    oob_errors = []
    for n in [1, 5, 10, 25, 50, 100, 150, 200, 300]:
        temp = clone(clf)  # use the same set of parameters
        temp.set_params(n_estimators=n, warm_start=False)  # fresh forest
        temp.fit(X_train_f, y_train)
        oob_errors.append((n, 1 - temp.oob_score_))
    print("OOB errors by n_estimators:", oob_errors)

    # 6) Optional cross-validation on train+validation
    X_tv_f = np.vstack([X_train_f, X_val_f])
    y_tv   = y_train + y_val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        clf, X_tv_f, y_tv,
        cv=skf, scoring="f1_macro", n_jobs=-1
    )
    print(f"\n5-fold CV Macro-F1 on train+val: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 7.1) Find threshold for class 1 on validation
    # binarize y_val: 1 = birch, 0 = everything else
    y_val_bin = np.array(y_val) == 1

    # probabilities for "birch" on validation
    proba_val = clf.predict_proba(X_val_f)[:, 0]  # column 0 corresponds to class 1

    # compute precision, recall, and thresholds
    prec, rec, thresh = precision_recall_curve(y_val_bin, proba_val)

    # calculate F1 scores for each threshold
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)

    # skip the last element of prec/rec which has no threshold
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresh[best_idx]
    print(f"Best threshold for class 1: {best_thresh:.3f} → "
        f"Precision={prec[best_idx]:.3f}, Recall={rec[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")

    # 8) Evaluation on the test set
    print("\n=== Test ===")
    yt_pred = clf.predict(X_test_f)
    print(classification_report(y_test, yt_pred, digits=3))
    print("Confusion Matrix (test):\n", confusion_matrix(y_test, yt_pred))
    print("Balanced accuracy (test):", balanced_accuracy_score(y_test, yt_pred))
    print("Cohen's kappa (test):", cohen_kappa_score(y_test, yt_pred))

    # 8.1) Prediction with threshold for class 1
    proba_test = clf.predict_proba(X_test_f)

    def predict_with_threshold(proba, t):
        proba = np.asarray(proba)
        # Vectorized: find argmax among classes [2..6] for all samples
        other_labels = np.argmax(proba[:, 1:], axis=1) + 2
        # Apply threshold: if birch probability >= t, assign 1; otherwise use other_labels
        y_pred = np.where(proba[:, 0] >= t, 1, other_labels)
        return y_pred.tolist()

    y_test_adj = predict_with_threshold(proba_test, best_thresh)

    # then use y_test_adj instead of clf.predict(X_test_f)
    print(classification_report(y_test, y_test_adj, digits=3))

    # 9) ROC-AUC One-vs-Rest
    proba = clf.predict_proba(X_test_f)
    y_onehot = np.zeros((len(y_test), clf.n_classes_))
    for i, lab in enumerate(y_test):
        y_onehot[i, lab-1] = 1
    auc = roc_auc_score(y_onehot, proba, multi_class="ovr")
    print(f"ROC-AUC (ovr) on test: {auc:.3f}")

    # 10) Feature importances
    print("\nFeature importances:")
    feature_names = ["mR","mG","mB","ExG","ExR","ExG-ExR"]
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    # 11) Save the model
    out_model = "multiclass_main_without_data_leak_0.85weight_birch.pkl"
    joblib.dump(clf, out_model)
    print(f"\nModel saved as {out_model}")

if __name__ == "__main__":
    main()
