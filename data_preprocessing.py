import os
import shutil
import numpy as np
from PIL import Image


def split_data(source_dir: str, train_size=0.8, val_size=0.1, test_size=0.1):
    """
        Splits the data in the given source directory into train, validation, and test sets.
        :param source_dir: The path to the directory containing the data.
        :param train_size: The proportion of data to be used for training. Defaults to 0.8.
        :param val_size: The proportion of data to be used for validation. Defaults to 0.1.
        :param test_size: The proportion of data to be used for testing. Defaults to 0.1.
        :raise AssertionError: If the sum of the sizes is not equal to 1.
        :return: None
    """
    assert train_size + val_size + test_size == 1, "The sum of sizes must be 1"

    files = os.listdir(os.path.join(source_dir, 'images'))
    np.random.shuffle(files)

    train_files = files[:int(len(files) * train_size)]
    val_files = files[int(len(files) * train_size):int(len(files) * (train_size + val_size))]
    test_files = files[int(len(files) * (train_size + val_size)):]
    dirs = ["train", "val", "test"]
    for i in dirs:
        os.makedirs(os.path.join(source_dir, i), exist_ok=True)
        os.makedirs(os.path.join(source_dir, i, 'images'), exist_ok=True)
        os.makedirs(os.path.join(source_dir, i, 'labels'), exist_ok=True)
    files_type = [train_files, val_files, test_files]
    for number, file in enumerate(files_type):
        for i in file:
            shutil.move(os.path.join(source_dir, 'images', i), os.path.join(source_dir, dirs[number], 'images', i))
            shutil.move(os.path.join(source_dir, 'labels', get_filename(i) + '.txt'),
                        os.path.join(source_dir, dirs[number], 'labels', get_filename(i) + '.txt'))


def resize_image(image_path: str, output_path: str, size=(512, 512)) -> None:
    image = Image.open(image_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(output_path, format='JPEG')


def resize_mask(mask_path: str, output_path: str, size=(512, 512)) -> None:
    mask = Image.open(mask_path)
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    mask.save(output_path, format='PNG')


# Изменение размера изображений
def resize(images_dir="augment/masks", output_images_dir="augment", size=(512, 512)) -> None:
    """
    Resize images in the given directory to the specified size.
    :param images_dir: directory to the source images
    :param output_images_dir: output directory
    :param size: size to resize the images to
    :return: None
    """
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(output_images_dir, filename)
            if filename.endswith(".jpg"):
                resize_image(image_path, output_path, size)
            elif filename.endswith(".png"):
                resize_mask(image_path, output_path, size)


def get_filename(file: str) -> str:
    """
    get filename without extension
    :param file: name of the file
    :return: file name without extension
    """
    file = file.split(".")[0]
    return file.rstrip(".")


def copy_images_from_files(source_txt_dir: str, source_image_dir: str, target_image_dir: str):
    """
    Copies images from files to target directory. For example, if the source directory contains files with text files,
    this function can be used to copy the corresponding images with the same name as the text files
    to the target directory.
    :param source_txt_dir: The path to the directory containing the original files
    :param source_image_dir: The path to the directory containing the images
    :param target_image_dir: The path to the directory where the images will be copied
    :return: None
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir, exist_ok=True)
    files = os.listdir(source_txt_dir)
    for txt_file_name in files:
        try:
            shutil.copy2(os.path.join(source_image_dir, get_filename(txt_file_name) + '.jpg'),
                         target_image_dir)
        except FileNotFoundError:
            try:
                shutil.copy2(os.path.join(source_image_dir, get_filename(txt_file_name) + '.jpeg'),
                             target_image_dir)
            except FileNotFoundError:
                shutil.copy2(os.path.join(source_image_dir, get_filename(txt_file_name) + '.png'),
                             target_image_dir)



if __name__ == '__main__':
    # split_data('datasets')
    text_dir = os.path.join('dataset', 'labels')
    image_dir = os.path.join('data', 'full quality')
    # target_dir = os.path.join('dataset', 'images')
    target_dir = os.path.join('unet-train', 'images')
    copy_images_from_files(text_dir, image_dir, target_dir)
