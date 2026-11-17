"""
Step 2: Leak-Free PCA Compression, Latent-Space SMOTE, and XGBoost Training
Runs quickly on CPU (taking ~3-5 minutes) on the pre-extracted .npy feature matrices.
"""
import os
import yaml
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.dim_reduction import PCAReducer
from src.models.balancer import LatentSMOTEBalancer
from src.models.classifiers import get_xgboost_classifier, get_baseline_classifiers
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed

def train_and_benchmark(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["random_seed"])

    train_feat_path = config["feature_extraction"]["export_paths"]["train_features"]
    train_lbl_path = config["feature_extraction"]["export_paths"]["train_labels"]
    test_feat_path = config["feature_extraction"]["export_paths"]["test_features"]
    test_lbl_path = config["feature_extraction"]["export_paths"]["test_labels"]

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print(f"[-] Feature matrices not found. Please run Step 1 (or Colab notebook) first.")
        print(f"    Expected: {train_feat_path} and {test_feat_path}")
        return

    print("[+] Loading extracted 768-D feature matrices...")
    X_train_raw = np.load(train_feat_path)
    y_train_raw = np.load(train_lbl_path)
    X_test_raw = np.load(test_feat_path)
    y_test_raw = np.load(test_lbl_path)

    print(f"    Raw Train shape: {X_train_raw.shape}, Raw Test shape: {X_test_raw.shape}")

    # Step 4: Strict Leak-Free Holdout Partition (85% Train / 15% Val)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=config["training"]["random_seed"])
    train_idx, val_idx = next(sss.split(X_train_raw, y_train_raw))

    X_tr = X_train_raw[train_idx]
    y_tr = y_train_raw[train_idx]
    X_val = X_train_raw[val_idx]
    y_val = y_train_raw[val_idx]

    # Step 5: PCA Compression (fit strictly on internal training set)
    n_components = config["training"]["pca"]["n_components"]
    print(f"[+] Fitting PCA ({n_components} components) strictly on internal train set...")
    pca_reducer = PCAReducer(n_components=n_components)
    X_tr_pca, exp_var = pca_reducer.fit_transform(X_tr)
    X_val_pca = pca_reducer.transform(X_val)
    X_test_pca = pca_reducer.transform(X_test_raw)
    print(f"[✓] PCA Explained Variance: {exp_var * 100:.2f}%")

    # Step 6: SMOTE Balancing on internal train set only
    print("[+] Applying SMOTE oversampling to internal training partition...")
    balancer = LatentSMOTEBalancer(
        k_neighbors=config["training"]["smote"]["k_neighbors"],
        random_state=config["training"]["random_seed"]
    )
    X_tr_balanced, y_tr_balanced = balancer.fit_resample(X_tr_pca, y_tr)
    print(f"    Balanced train samples: {len(X_tr_balanced)}")

    # Step 7: Train XGBoost Classifier
    print("[+] Training Tuned XGBoost classifier with early stopping...")
    xgb_model = get_xgboost_classifier(config)
    xgb_model.fit(
        X_tr_balanced,
        y_tr_balanced,
        eval_set=[(X_val_pca, y_val)],
        verbose=50
    )

    # Evaluate on Official Benchmark Test Set
    y_test_pred = xgb_model.predict(X_test_pca)
    y_test_prob = xgb_model.predict_proba(X_test_pca)
    metrics = compute_classification_metrics(y_test_raw, y_test_pred, y_test_prob)

    print("\n" + "=" * 50)
    print("🏆 FINAL BENCHMARK RESULTS (1,000 Official Test Images)")
    print("=" * 50)
    print(f"  • Test Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Macro Sensitivity:    {metrics['macro_sensitivity'] * 100:.2f}%")
    print(f"  • Macro Specificity:    {metrics['macro_specificity'] * 100:.2f}%")
    print(f"  • Macro Precision:      {metrics['macro_precision'] * 100:.2f}%")
    print(f"  • Macro F1-Score:       {metrics['macro_f1'] * 100:.2f}%")
    if metrics.get("macro_roc_auc"):
        print(f"  • Macro ROC-AUC:        {metrics['macro_roc_auc'] * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    train_and_benchmark()
