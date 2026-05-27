import albumentations as A

def get_train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
    ])

def get_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])