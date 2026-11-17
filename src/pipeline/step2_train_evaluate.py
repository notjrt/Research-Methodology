"""
Step 2: 5-Fold Stratified Cross-Validation Pipeline
Strictly enforces zero data leakage across all 5 folds:
  - Fold Partitioning: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
  - PCA Compression (150-D): Fitted exclusively on each fold's internal train partition.
  - Latent SMOTE: Applied strictly to each fold's internal train partition.
  - Early Stopping: Tuned on each fold's internal validation partition.
  - Final Evaluation: Evaluated on the 1,000 untouched official test images.
"""
import os
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.dim_reduction import PCAReducer
from src.models.balancer import LatentSMOTEBalancer
from src.models.classifiers import get_xgboost_classifier
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed

def train_and_benchmark(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["random_seed"]
    set_seed(seed)

    train_feat_path = config["feature_extraction"]["export_paths"]["train_features"]
    train_lbl_path = config["feature_extraction"]["export_paths"]["train_labels"]
    test_feat_path = config["feature_extraction"]["export_paths"]["test_features"]
    test_lbl_path = config["feature_extraction"]["export_paths"]["test_labels"]

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print("[-] Pre-extracted feature matrices not found.")
        print(f"    Expected: {train_feat_path} and {test_feat_path}")
        print("    Please run Step 1 (python -m src.pipeline.step1_extract_features) first.")
        return

    print("[+] Loading pre-extracted 768-D ViT feature matrices...")
    X_train_raw = np.load(train_feat_path)
    y_train_raw = np.load(train_lbl_path)
    X_test_raw = np.load(test_feat_path)
    y_test_raw = np.load(test_lbl_path)

    print(f"    Raw Train Matrix: {X_train_raw.shape}")
    print(f"    Raw Test Matrix:  {X_test_raw.shape} (Official 1,000 balanced benchmark)")

    n_splits = config["training"].get("n_splits", 5)
    n_components = config["training"]["pca"]["n_components"]
    k_neighbors = config["training"]["smote"]["k_neighbors"]

    print(f"\n" + "=" * 65)
    print(f"🔬 RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION (ZERO DATA LEAKAGE)")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_val_metrics = []
    fold_test_metrics = []
    best_overall_model = None
    best_val_acc = -1.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train_raw), 1):
        print(f"\n>>> [Fold {fold_idx}/{n_splits}] Starting Fold Pipeline...")
        X_tr, y_tr = X_train_raw[train_idx], y_train_raw[train_idx]
        X_val, y_val = X_train_raw[val_idx], y_train_raw[val_idx]

        print(f"    Internal Train: {X_tr.shape[0]} samples | Internal Val: {X_val.shape[0]} samples")

        # Step 4: PCA Dimensionality Reduction (Fit strictly on this fold's train partition)
        pca_reducer = PCAReducer(n_components=n_components)
        X_tr_pca, exp_var = pca_reducer.fit_transform(X_tr)
        X_val_pca = pca_reducer.transform(X_val)
        X_test_pca = pca_reducer.transform(X_test_raw)
        print(f"    [+] PCA ({n_components}-D) fitted. Variance captured: {exp_var * 100:.2f}%")

        # Step 5: Latent-Space SMOTE (Applied strictly on this fold's train partition)
        balancer = LatentSMOTEBalancer(k_neighbors=k_neighbors, random_state=seed)
        X_tr_bal, y_tr_bal = balancer.fit_resample(X_tr_pca, y_tr)
        print(f"    [+] SMOTE applied. Balanced train size: {len(X_tr_bal)} samples")

        # Step 6: Train XGBoost Classifier with Early Stopping on Fold Validation
        print("    [+] Fitting Tuned XGBoost with early stopping...")
        model = get_xgboost_classifier(config)
        model.fit(
            X_tr_bal,
            y_tr_bal,
            eval_set=[(X_val_pca, y_val)],
            verbose=False
        )

        # Evaluate on Fold Internal Validation Set
        y_val_pred = model.predict(X_val_pca)
        val_metric = compute_classification_metrics(y_val, y_val_pred)
        fold_val_metrics.append(val_metric)
        print(f"    [✓] Fold {fold_idx} Validation Accuracy: {val_metric['accuracy'] * 100:.2f}%")

        # Evaluate on the untouched 1,000 Official Test Set
        y_test_pred = model.predict(X_test_pca)
        y_test_prob = model.predict_proba(X_test_pca)
        test_metric = compute_classification_metrics(y_test_raw, y_test_pred, y_test_prob)
        fold_test_metrics.append(test_metric)
        print(f"    [✓] Fold {fold_idx} Test Accuracy:       {test_metric['accuracy'] * 100:.2f}%")

        if val_metric["accuracy"] > best_val_acc:
            best_val_acc = val_metric["accuracy"]
            best_overall_model = model

    # 5-Fold Aggregated Summary Statistics
    val_accs = [m["accuracy"] * 100 for m in fold_val_metrics]
    test_accs = [m["accuracy"] * 100 for m in fold_test_metrics]
    test_sens = [m["macro_sensitivity"] * 100 for m in fold_test_metrics]
    test_spec = [m["macro_specificity"] * 100 for m in fold_test_metrics]
    test_f1 = [m["macro_f1"] * 100 for m in fold_test_metrics]

    print("\n" + "=" * 65)
    print("🏆 FINAL 5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  • Validation Accuracy (Mean ± Std): {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
    print(f"  • Test Accuracy       (Mean ± Std): {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")
    print(f"  • Macro Sensitivity   (Mean ± Std): {np.mean(test_sens):.2f}% ± {np.std(test_sens):.2f}%")
    print(f"  • Macro Specificity   (Mean ± Std): {np.mean(test_spec):.2f}% ± {np.std(test_spec):.2f}%")
    print(f"  • Macro F1-Score      (Mean ± Std): {np.mean(test_f1):.2f}% ± {np.std(test_f1):.2f}%")
    print("=" * 65)

    return best_overall_model, fold_test_metrics

if __name__ == "__main__":
    train_and_benchmark()
