from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAReducer:
    """
    Orthogonal Dimensionality Reduction with zero data leakage.
    Fitted strictly on training partition only.
    """
    def __init__(self, n_components: int = 150):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)

    def fit_transform(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train)
        X_pca = self.pca.fit_transform(X_scaled)
        explained_variance_sum = sum(self.pca.explained_variance_ratio_)
        return X_pca, explained_variance_sum

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
