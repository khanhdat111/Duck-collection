import glob

import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize
from skimage.color import rgb2gray 

folder_path = "/kaggle/input/cvc-clinic-png/"  # Add the path to your data directory


# def load_data(img_height, img_width, images_to_be_loaded, dataset):
#     IMAGES_PATH = folder_path + 'Original/'
#     MASKS_PATH = folder_path + 'Ground Truth/'

#     if dataset == 'kvasir':
#         train_ids = glob.glob(IMAGES_PATH + "*.jpg")

#     if dataset == 'cvc-clinicdb':
#         train_ids = glob.glob(IMAGES_PATH + "*.png")
#         #train_ids = train_ids[:30]

#     if dataset == 'cvc-colondb' or dataset == 'etis-laribpolypdb':
#         train_ids = glob.glob(IMAGES_PATH + "*.png")

#     if images_to_be_loaded == -1:
#         images_to_be_loaded = len(train_ids)
#         print(images_to_be_loaded)

#     X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
#     Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

#     print('Resizing training images and masks: ' + str(images_to_be_loaded))
#     for n, id_ in tqdm(enumerate(train_ids)):
#         if n == images_to_be_loaded:
#             break

#         image_path = id_
#         mask_path = image_path.replace("images", "masks")

#         image = imread(image_path)
#         mask_ = imread(mask_path)

#         mask = np.zeros((img_height, img_width), dtype=np.bool_)

#         pillow_image = Image.fromarray(image)

#         pillow_image = pillow_image.resize((img_height, img_width))
#         image = np.array(pillow_image)

#         X_train[n] = image / 255

#         pillow_mask = Image.fromarray(mask_)
#         pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
#         mask_ = np.array(pillow_mask)

#         for i in range(img_height):
#             for j in range(img_width):
#                 if (mask_[i, j] >= 127).all():
#                     mask[i, j] = 1

#         Y_train[n] = mask

#     Y_train = np.expand_dims(Y_train, axis=-1)

#     return X_train, Y_train


def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = folder_path + 'Original/'
    MASKS_PATH = folder_path.replace("Original", "Ground Truth")

    pattern = "*.jpg" if dataset == 'kvasir' else "*.png"
    train_ids = glob.glob(IMAGES_PATH + pattern)

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)
    print(f'Resizing training images and masks: {images_to_be_loaded}')

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width, 1), dtype=np.uint8) 

    for n, id_ in tqdm(enumerate(train_ids), total=images_to_be_loaded):
        if n >= images_to_be_loaded:
            break

        image = imread(id_) / 255.0
        mask_path = id_.replace("Original", "Ground Truth").replace(".jpg", ".png").replace(".png", ".png")
        mask_ = imread(mask_path)
        if mask_.ndim == 3:
            mask_ = rgb2gray(mask_)

        image_resized = resize(image, (img_height, img_width), anti_aliasing=True)
        mask_resized = resize(mask_, (img_height, img_width), order=0, preserve_range=True)

        mask_resized = (mask_resized >= 0.5).astype(np.uint8) * 255

        X_train[n] = image_resized
        Y_train[n] = np.expand_dims(mask_resized, axis=-1)
        
    return X_train, Y_train
