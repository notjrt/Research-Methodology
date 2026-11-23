import cv2
import numpy as np
from PIL import Image


class CLAHEPreprocessor:
    """OCT B-scan preprocessor with optional CLAHE enhancement and ImageNet normalization."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        target_size: tuple = (224, 224),
        use_clahe: bool = True,
    ):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.target_size = tuple(target_size)
        self.use_clahe = bool(use_clahe)
        self._clahe = None

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
        return self._clahe

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe = None

    def preprocess_image(self, image_input) -> np.ndarray:
        """Load image, apply optional CLAHE, resize to 224x224, and normalize to (3, H, W)."""
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, Image.Image):
            gray = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) if image_input.ndim == 3 else image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if gray is None:
            raise FileNotFoundError(f"Failed to load image: {image_input}")

        processed = self.clahe.apply(gray) if self.use_clahe else gray
        resized = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        return np.transpose(normalized, (2, 0, 1))
