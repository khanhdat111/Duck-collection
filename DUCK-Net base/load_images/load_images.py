import glob
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

folder_path = "/kaggle/input/cvc-clinic-png/"  # Add the path to your data directory

def read_and_resize_image(image_path, img_height, img_width):
    # Đọc và resize ảnh
    image = imread(image_path)
    image = resize(image, (img_height, img_width), anti_aliasing=True)
    return image

def read_and_process_mask(mask_path, img_height, img_width):
    # Đọc và resize mask
    mask = imread(mask_path)
    mask = resize(mask, (img_height, img_width), anti_aliasing=False, order=0)
    # Xử lý mask để chuyển thành binary mask
    binary_mask = mask >= 127
    return binary_mask

def get_train_ids(dataset, IMAGES_PATH):
    # Lấy danh sách các ID ảnh tùy theo dataset
    if dataset in ['kvasir', 'cvc-clinicdb', 'cvc-colondb', 'etis-laribpolypdb']:
        extension = "*.jpg" if dataset == 'kvasir' else "*.png"
        train_ids = glob.glob(IMAGES_PATH + extension)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    return train_ids

def load_data(img_height, img_width, images_to_be_loaded, dataset):
    IMAGES_PATH = folder_path + 'Original/'
    MASKS_PATH = folder_path + 'Ground Truth/'

    train_ids = get_train_ids(dataset, IMAGES_PATH)

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)
        print(f"Number of images to be loaded: {images_to_be_loaded}")

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width, 1), dtype=np.bool_)

    print(f'Resizing training images and masks: {images_to_be_loaded}')
    for n, id_ in tqdm(enumerate(train_ids), total=images_to_be_loaded):
        if n == images_to_be_loaded:
            break

        # Xử lý ảnh và mask
        image = read_and_resize_image(id_, img_height, img_width)
        mask_path = id_.replace("Original", "Ground Truth").replace(".jpg", ".png").replace(".png", ".png")
        mask = read_and_process_mask(mask_path, img_height, img_width)

        # Normalize ảnh và thêm vào X_train
        X_train[n] = image / 255.0
        Y_train[n, ..., 0] = mask

    return X_train, Y_train
