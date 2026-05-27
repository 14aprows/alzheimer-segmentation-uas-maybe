from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

class BrainSegmentationDataset(Dataset):
    def __init__(self, data_dirs, transform=None, strict=True):
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transform = transform
        self.strict = strict
        self.samples = self._collect_samples()

        if self.strict and len(self.samples) == 0:
            raise ValueError("No image-mask pairs found. Check dataset paths and naming format.")

    def _collect_samples(self):
        samples = []

        for data_dir in self.data_dirs:
            image_paths = sorted([
                path for path in data_dir.glob("*")
                if path.suffix.lower() in IMAGE_EXTENSIONS
                and not path.stem.endswith("_mask")
            ])

            for image_path in image_paths:
                mask_path = image_path.parent / f"{image_path.stem}_mask{image_path.suffix}"

                if mask_path.exists():
                    samples.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "tumor_type": data_dir.name,
                    })
                elif self.strict:
                    raise FileNotFoundError(f"Mask not found for image: {image_path}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {sample['image_path']}")

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {sample['mask_path']}")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "tumor_type": sample["tumor_type"],
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }