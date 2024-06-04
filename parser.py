from yamager import Yamager
import os
import shutil
import requests


def get_extension(filename: str) -> str:
    """
    Returns the extension of a filename
    """
    # Разделяем имя файла на части по разделителю '/' или '\'
    parts = filename.split(os.sep)
    # Последняя часть - это имя файла
    filename = parts[-1]
    # Разделяем имя файла на основу и расширение
    base, ext = os.path.splitext(filename)
    # Возвращаем расширение с точкой
    return ext


def parsing(query: str, rename=False, should_print=False):
    """
    Parsing images from google
    with should_print = False process will be faster
    rename = True if you want to rename images in count order
    """
    count = 0
    yamager = Yamager()
    images = yamager.search_google_images(query)
    if should_print:
        print(f"Found {len(images)} images for query: {query}")
        print(images)
    if not os.path.exists(os.path.join('parsing', query)):
        os.makedirs(os.path.join('parsing', query))

    for image in images:
        try:
            response = requests.get(image, stream=True)
        except Exception as e:
            if should_print:
                print(f"Error requesting image: {image}, error: {e}")
            continue
        if response.status_code == 200:
            try:
                if rename:
                    with open(os.path.join('parsing', query, str(count) + get_extension(image)), 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                    count += 1
                else:
                    with open(os.path.join('parsing', query, image.split('/')[-1]), 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                if should_print:
                    if not rename:
                        print(f"Image saved: {image.split('/')[-1]}")
                    else:
                        print(f"Image saved: {str(count) + get_extension(image)}")

            except Exception as e:
                if should_print:
                    print(f"Error saving image: {image.split('/')[-1]}, error: {e}")
        else:
            if should_print:
                print(f"Error downloading image: {image}, status code: {response.status_code}")
        del response


if __name__ == '__main__':
    parsing('металлическая лестница', False, True)