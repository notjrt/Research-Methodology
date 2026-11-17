from imblearn.over_sampling import SMOTE

class LatentSMOTEBalancer:
    """
    Synthesizes vectors along minority class segments directly
    in the 150-D orthogonal latent space to balance class distributions.
    Strictly applied only to internal training partition.
    """
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    def fit_resample(self, X_train, y_train):
        return self.smote.fit_resample(X_train, y_train)
