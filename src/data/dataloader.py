import torch
from torch.utils.data import DataLoader, Subset
from src.configs.config import GLIOMA_DIR, MENINGIOMA_DIR, PITUITARY_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
from src.data.dataset import BrainSegmentationDataset
from src.data.transforms import get_train_transform, get_val_transform

def get_data_dirs():
    return [
        GLIOMA_DIR,
        MENINGIOMA_DIR,
        PITUITARY_DIR,
    ]

def split_indices(dataset_size, train_ratio=0.8, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(train_ratio * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return train_indices, val_indices

def create_dataloaders():
    data_dirs = get_data_dirs()

    base_dataset = BrainSegmentationDataset(
        data_dirs=data_dirs,
        transform=None,
    )

    train_indices, val_indices = split_indices(
        dataset_size=len(base_dataset),
        train_ratio=0.8,
        seed=SEED,
    )

    train_dataset = BrainSegmentationDataset(
        data_dirs=data_dirs,
        transform=get_train_transform(IMAGE_SIZE),
    )

    val_dataset = BrainSegmentationDataset(
        data_dirs=data_dirs,
        transform=get_val_transform(IMAGE_SIZE),
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader