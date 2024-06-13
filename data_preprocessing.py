import os
import shutil
import numpy as np


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


def get_filename(file: str) -> str:
    """
    get filename without extension
    :param file: name of the file
    :return: file name without extension
    """
    file = file.split(".")[0]
    return file.rstrip(".")


def copy_images_from_text_files(source_txt_dir: str, source_image_dir: str, target_image_dir: str):
    """
    Copies images from text files to target directory.
    :param source_txt_dir:
    :param source_image_dir:
    :param target_image_dir:
    :param image_extensions:
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
        # shutil.copy2(os.path.join(source_image_dir, get_filename(txt_file_name) + '.jpg'),
        #              os.path.join(target_image_dir, get_filename(txt_file_name) + '.jpg'))
        # print(f"Copied: {os.path.join(source_image_dir, get_filename(txt_file_name) + '.jpg')} to"
        #       f" {os.path.join(target_image_dir, get_filename(txt_file_name) + '.jpg')}")


if __name__ == '__main__':
    # split_data('datasets')
    text_dir = os.path.join('dataset', 'labels')

    image_dir = os.path.join('data', 'full quality')
    # target_dir = os.path.join('dataset', 'images')
    target_dir = os.path.join('unet-train', 'images')
    copy_images_from_text_files(text_dir, image_dir, target_dir)
