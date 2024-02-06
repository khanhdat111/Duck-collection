# Defining the augmentations
import albumentations as albu
import numpy as np

# aug_train = albu.Compose([
#     albu.HorizontalFlip(),
#     albu.VerticalFlip(),
#     albu.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
#     albu.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), always_apply=True),
# ])

aug_train = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.9),
    albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    albu.RandomGamma(gamma_limit=(80, 120), p=0.9),
    albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    albu.Affine(scale=(0.8,1.2), translate_percent=(-0.2,0.2), rotate=(-180,180), shear=(-8,8), p=0.9),
    albu.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, fill_value=0, p=0.5),
    albu.RandomSizedCrop(min_max_height=(256, 256), height=512, width=512, p=0.5)
])

def augment_images(x_train,y_train):
    x_train_out = []
    y_train_out = []

    for i in range (len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])  
        y_train_out.append(ug['mask'])

    return np.array(x_train_out), np.array(y_train_out)
