import os
import sys
import json
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

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
        print(f"Feature files not found: {train_feat_path}, {test_feat_path}")
        return

    print("Loading feature matrices...")
    X_train_raw = np.load(train_feat_path)
    y_train_raw = np.load(train_lbl_path)
    X_test_raw = np.load(test_feat_path)
    y_test_raw = np.load(test_lbl_path)

    print(f"Train matrix: {X_train_raw.shape}, Test matrix: {X_test_raw.shape}")

    n_splits = config["training"].get("n_splits", 5)
    n_components = config["training"]["pca"]["n_components"]
    k_neighbors = config["training"]["smote"]["k_neighbors"]

    print(f"Running {n_splits}-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_val_metrics = []
    fold_test_metrics = []
    fold_details = []
    best_overall_model = None
    best_val_acc = -1.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train_raw), 1):
        X_tr, y_tr = X_train_raw[train_idx], y_train_raw[train_idx]
        X_val, y_val = X_train_raw[val_idx], y_train_raw[val_idx]

        pca_reducer = PCAReducer(n_components=n_components)
        X_tr_pca, exp_var = pca_reducer.fit_transform(X_tr)
        X_val_pca = pca_reducer.transform(X_val)
        X_test_pca = pca_reducer.transform(X_test_raw)

        balancer = LatentSMOTEBalancer(k_neighbors=k_neighbors, random_state=seed)
        X_tr_bal, y_tr_bal = balancer.fit_resample(X_tr_pca, y_tr)

        model = get_xgboost_classifier(config)
        model.fit(
            X_tr_bal,
            y_tr_bal,
            eval_set=[(X_val_pca, y_val)],
            verbose=False
        )

        y_val_pred = model.predict(X_val_pca)
        val_metric = compute_classification_metrics(y_val, y_val_pred)
        fold_val_metrics.append(val_metric)

        y_test_pred = model.predict(X_test_pca)
        y_test_prob = model.predict_proba(X_test_pca)
        test_metric = compute_classification_metrics(y_test_raw, y_test_pred, y_test_prob)
        fold_test_metrics.append(test_metric)

        fold_info = {
            "fold": fold_idx,
            "val_accuracy": round(val_metric["accuracy"] * 100, 2),
            "test_accuracy": round(test_metric["accuracy"] * 100, 2),
            "macro_sensitivity": round(test_metric["macro_sensitivity"] * 100, 2),
            "macro_specificity": round(test_metric["macro_specificity"] * 100, 2),
            "macro_f1": round(test_metric["macro_f1"] * 100, 2)
        }
        fold_details.append(fold_info)

        print(f"Fold {fold_idx}/{n_splits} | Val Acc: {fold_info['val_accuracy']:.2f}% | Test Acc: {fold_info['test_accuracy']:.2f}%")

        if val_metric["accuracy"] > best_val_acc:
            best_val_acc = val_metric["accuracy"]
            best_overall_model = model

    val_accs = [m["accuracy"] * 100 for m in fold_val_metrics]
    test_accs = [m["accuracy"] * 100 for m in fold_test_metrics]
    test_sens = [m["macro_sensitivity"] * 100 for m in fold_test_metrics]
    test_spec = [m["macro_specificity"] * 100 for m in fold_test_metrics]
    test_f1 = [m["macro_f1"] * 100 for m in fold_test_metrics]

    summary_text = (
        "5-Fold Stratified Cross-Validation Benchmark Results\n\n"
        "Fold Details:\n"
    )
    for f in fold_details:
        summary_text += f"Fold {f['fold']} | Val: {f['val_accuracy']:.2f}% | Test: {f['test_accuracy']:.2f}% | Sens: {f['macro_sensitivity']:.2f}% | Spec: {f['macro_specificity']:.2f}% | F1: {f['macro_f1']:.2f}%\n"

    summary_text += (
        f"\nSummary (Mean +/- Std):\n"
        f"Validation Accuracy: {np.mean(val_accs):.2f}% +/- {np.std(val_accs):.2f}%\n"
        f"Test Accuracy:       {np.mean(test_accs):.2f}% +/- {np.std(test_accs):.2f}%\n"
        f"Macro Sensitivity:   {np.mean(test_sens):.2f}% +/- {np.std(test_sens):.2f}%\n"
        f"Macro Specificity:   {np.mean(test_spec):.2f}% +/- {np.std(test_spec):.2f}%\n"
        f"Macro F1-Score:      {np.mean(test_f1):.2f}% +/- {np.std(test_f1):.2f}%\n"
    )

    print(f"\n{summary_text}")

    os.makedirs("outputs", exist_ok=True)
    result_path = "outputs/benchmark_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    json_path = "outputs/benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "folds": fold_details,
            "summary": {
                "val_accuracy_mean": round(float(np.mean(val_accs)), 2),
                "val_accuracy_std": round(float(np.std(val_accs)), 2),
                "test_accuracy_mean": round(float(np.mean(test_accs)), 2),
                "test_accuracy_std": round(float(np.std(test_accs)), 2),
                "macro_sensitivity_mean": round(float(np.mean(test_sens)), 2),
                "macro_sensitivity_std": round(float(np.std(test_sens)), 2),
                "macro_specificity_mean": round(float(np.mean(test_spec)), 2),
                "macro_specificity_std": round(float(np.std(test_spec)), 2),
                "macro_f1_mean": round(float(np.mean(test_f1)), 2),
                "macro_f1_std": round(float(np.std(test_f1)), 2),
            }
        }, f, indent=2)

    print(f"Saved results: {result_path}")

    return best_overall_model, fold_test_metrics

if __name__ == "__main__":
    train_and_benchmark()
