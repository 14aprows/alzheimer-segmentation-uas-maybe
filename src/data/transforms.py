import albumentations as A
import cv2

def get_train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.3),
    ])

def get_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])