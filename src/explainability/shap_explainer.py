import shap
import matplotlib.pyplot as plt
import os

class TreeSHAPExplainer:
    """
    Computes game-theoretic feature attribution using TreeSHAP
    for tree ensemble models on the latent PCA feature space.
    """
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_sample):
        return self.explainer.shap_values(X_sample)

    def summary_plot(self, shap_values, X_sample, feature_names=None, save_path=None):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
