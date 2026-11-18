import os
import sys
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.clahe_preprocessor import CLAHEPreprocessor
from src.models.vit_extractor import FrozenViTExtractor
from src.utils.seed import set_seed

class SimpleOCTFolderDataset(Dataset):
    def __init__(self, root_dir, class_names, preprocessor):
        self.samples = []
        self.labels = []
        self.preprocessor = preprocessor
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            image_paths = glob.glob(os.path.join(cls_dir, "*.jpeg")) + glob.glob(os.path.join(cls_dir, "*.jpg"))
            for p in image_paths:
                self.samples.append(p)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        img_tensor = self.preprocessor.preprocess_image(path)
        return torch.from_numpy(img_tensor).float(), label

def extract_and_export(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    preprocessor = CLAHEPreprocessor(
        clip_limit=config["data"]["clahe"]["clip_limit"],
        tile_grid_size=tuple(config["data"]["clahe"]["tile_grid_size"]),
        target_size=tuple(config["data"]["image_size"])
    )

    model = FrozenViTExtractor(
        model_name=config["feature_extraction"]["backbone"],
        pretrained=config["feature_extraction"]["pretrained"]
    ).to(device)

    data_dir = config["data"]["data_dir"]
    classes = config["data"]["classes"]
    batch_size = config["feature_extraction"]["batch_size"]

    num_workers = config["feature_extraction"].get("num_workers", 0)
    if sys.platform == "win32" and device.type == "cpu":
        num_workers = 0

    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory not found: {split_dir}")
            continue

        dataset = SimpleOCTFolderDataset(split_dir, classes, preprocessor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
        print(f"Extracting {split} set ({len(dataset)} images)...")

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=f"{split}"):
                imgs = imgs.to(device)
                feats = model(imgs).cpu().numpy()
                all_feats.append(feats)
                all_labels.append(lbls.numpy())

        X = np.vstack(all_feats)
        y = np.concatenate(all_labels)

        out_feat_path = config["feature_extraction"]["export_paths"][f"{split}_features"]
        out_lbl_path = config["feature_extraction"]["export_paths"][f"{split}_labels"]
        os.makedirs(os.path.dirname(out_feat_path), exist_ok=True)

        np.save(out_feat_path, X)
        np.save(out_lbl_path, y)
        print(f"Saved {split}: {out_feat_path} ({X.shape})")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    extract_and_export()
