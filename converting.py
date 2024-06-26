import os
import cv2
import numpy as np
from skimage import measure


def preprocess_image(img):
    # Delete black pixels
    img[img == 0] = np.nan
    img = np.nan_to_num(img)

    # Normalize remaining pixels
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Create YOLO segmentation mask
    mask = np.zeros_like(img)
    mask[img > 0] = 1

    return img, mask


def preprocess_and_save(img_path, output_dir):
    img = cv2.imread(img_path)

    # Preprocess image
    img, mask = preprocess_image(img)

    # Get image filename
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # Create YOLO format
    yolo_str = f"{filename}\n"
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                yolo_str += f"{j} {i} 0 0\n"

    # Write to text file
    with open(os.path.join(output_dir, f"{filename}.txt"), "w") as f:
        f.write(yolo_str)


def mask_to_polygons(mask):
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)  # Flip to (x, y) format
        contour = contour.ravel().tolist()
        polygons.append(contour)
    return polygons


def normalize_polygons(polygons, img_width, img_height):
    normalized_polygons = []
    for polygon in polygons:
        normalized_polygon = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            normalized_polygon.extend([x, y])
        normalized_polygons.append(normalized_polygon)
    return normalized_polygons


def save_yolo_format(image_id, polygons, output_dir, class_id=0):
    with open(os.path.join(output_dir, f'{image_id}.txt'), 'w') as f:
        for polygon in polygons:
            polygon_str = ' '.join(map(str, polygon))
            f.write(f'{class_id} {polygon_str}\n')


def convert_masks_to_yolo_format(image_dir: str, mask_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all masks and convert them
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_id = os.path.splitext(mask_filename)[0]
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_image_path = os.path.join(image_dir, image_id + ext)
                if os.path.exists(potential_image_path):
                    image_path = potential_image_path
                    break

            if image_path is None:
                print(f"No corresponding image found for mask: {mask_filename}")
                continue

            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]

            mask_path = os.path.join(mask_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            polygons = mask_to_polygons(mask)

            normalized_polygons = normalize_polygons(polygons, img_width, img_height)

            save_yolo_format(image_id, normalized_polygons, output_dir)


if __name__ == '__main__':

    # Directory paths
    image_dir = os.path.join('data', 'full quality')
    mask_dir = os.path.join("segmentation", "SegmentationClass", "full quality")
    output_dir = os.path.join("dataset", "labels", "train")

    # Convert masks to YOLO format
    # convert_masks_to_yolo_format(image_dir, mask_dir, output_dir)
    # for mask_filename in os.listdir(mask_dir):
    #     preprocess_and_save(os.path.join(mask_dir, mask_filename), output_dir)
