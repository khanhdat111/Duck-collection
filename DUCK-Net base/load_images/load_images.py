import glob
import numpy as np
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

folder_path = "/kaggle/input/cvc-clinic-png/"  # Adjust the path to your data directory accordingly

def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = f"{folder_path}Original/"
    MASKS_PATH = f"{folder_path}Ground Truth/"

    # Simplify dataset pattern matching
    pattern = "*.jpg" if dataset == 'kvasir' else "*.png"
    train_ids = glob.glob(IMAGES_PATH + pattern)

    # Optionally limit the number of images to be loaded
    if images_to_be_loaded == -1 or images_to_be_loaded > len(train_ids):
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width, 1), dtype=np.bool_)

    print(f'Resizing training images and masks: {images_to_be_loaded}')
    for n, id_ in tqdm(enumerate(train_ids[:images_to_be_loaded]), total=images_to_be_loaded):
        image = np.array(Image.open(id_).resize((img_width, img_height)))
        mask_path = id_.replace("Original", "Ground Truth").replace(".jpg", ".png").replace(".png", ".png")
        mask = np.array(Image.open(mask_path).resize((img_width, img_height), resample=Image.LANCZOS))

        X_train[n] = image / 255.0
        Y_train[n] = np.expand_dims(mask >= 127, axis=-1)

    return X_train, Y_train
