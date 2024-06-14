import os
import shutil
import zipfile
import torch
import concurrent.futures
import requests
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
from vk_api.longpoll import VkLongPoll, VkEventType
import time
import numpy as np
from skimage import io
import urllib.request
import tempfile
from threading import Thread
import cv2
from background_remove import UNet, get_image_without_background
from segment_train import train
from sending_functions import send_message, send_document, send_photo, vk_session, vk


def check_attachments(event) -> str:
    """
    Check attachments in event, if there is a photo, document or None
    :param event: vk event
    :return: string with type of attachment: photo, doc or None
    """
    # Получаем полную информацию о сообщении по его id
    message = vk.messages.getById(message_ids=event.message_id)['items'][0]

    # Проверяем, есть ли вложения в сообщении
    if 'attachments' in message:
        for attachment in message['attachments']:
            # Проверяем тип вложения
            if attachment['type'] in ['photo']:
                return "photo"
            elif attachment['type'] in ['doc']:
                return "doc"
    return "None"

def is_number(msg):
    """
    Check if the given text message can be converted to a number (integer or float).

    Parameters:
    msg (str): The text message to check.

    Returns:
    bool: True if the message can be converted to a number, False otherwise.
    """
    try:
        float(msg)  # Attempt to convert to float
        return True
    except ValueError:
        try:
            int(msg)  # Attempt to convert to integer
            return True
        except ValueError:
            return False


def transform_photo(img_path: str) -> np.ndarray:
    """
    Fix image by duplicating axis or deleting alpha channel
    :param img_path: path to image
    :return: image in numpy array
    """
    img = cv2.imread(img_path)
    if img.shape[2] == 2:
        img = img[:, :, None]
        img = np.repeat(img, 3, axis=2)
        print("duplicated axis")
    if len(img.shape) == 1:
        img = img[:, :, None]
        img = np.repeat(img, 3, axis=1)
        print("duplicated axis")
    if img.shape[2] == 4:
        image = img[:, :, :3]
        print("removed alpha channel")
    return img


def get_models(user_id: int) -> list:
    """
    get list of the models of user
    :param user_id: vk id
    :return: list of paths to the models
    """
    models = []
    for i in os.listdir('models'):
        if os.path.isfile(os.path.join("models", i)):
            models.append(i.split('.')[0])
    for i in os.listdir(os.path.join('models', str(user_id))):
        if os.path.isfile(os.path.join("models", str(user_id), i)):
            models.append(i.split('.')[0])
    return models


def save_image_from_url(image_url: str, file_name: str):
    """
    Save image from url
    :param image_url: url of the image
    :param file_name: name of the file
    :return: None
    """
    urllib.request.urlretrieve(image_url, file_name)
    print(F'File "{file_name}" was saved to temporary')


def process_single_image(image_path: str, model, device: torch.device, temp_dir, count=None, user_id=None):
    return get_image_without_background(image_path, model, device, temp_dir, count, user_id)


def remove_background(user_id: int, event, model, device: torch.device, image_path: str, temp_dir, many=False) -> None:
    """
    Remove background from the image
    :param user_id: vk id
    :param event: vk event
    :param model: pytorch model of unet
    :param device: cuda or cpu
    :param image_path: path to the image
    :param temp_dir: temp
    :param many: bool, if there are many images
    :return:
    """
    if not many:
        new_image = get_image_without_background(image_path, model, device, temp_dir)
        send_document(user_id, event, new_image)
    else:
        image_files = [os.path.join(image_path, file) for file in os.listdir(image_path) if
                       file.endswith(('.png', '.jpg', '.jpeg'))]

        # Использование ThreadPoolExecutor для параллельного выполнения
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_image, image_file, model, device, temp_dir, count, user_id): image_file
                for count, image_file in enumerate(image_files)}

            # Ожидание завершения всех потоков
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")

        new_extract = os.path.join(temp_dir.name, "file")
        extract_to = os.path.join(temp_dir.name, f"{user_id}", "without")
        shutil.make_archive(new_extract, 'zip', root_dir=extract_to)
        send_document(user_id, event, new_extract + ".zip")
    temp_dir.cleanup()


def main(longpoll):
    """
    main function of the program, which run the vk-bot
    """
    user_action = {}
    user_models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)
    model.load_state_dict(torch.load(os.path.join('models', 'base_segmentation.pt')))
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            id = event.user_id
            if id not in user_action:
                user_action[id] = ""
                user_models[id] = model
            try:
                msg = event.text.lower()
            except AttributeError:
                msg = ""
            if user_action[id] == "choose_model":
                models = get_models(id)
                if not is_number(msg):
                    user_action[id] = ""
                    send_message(id, "Сообщение не является числом. Попробуйте ещё раз, введя команду 'модели'.")
                else:
                    if int(msg) > len(models) or int(msg) <= 0:
                        user_action[id] = ""
                        send_message(id, "Такой модели нет. Попробуйте ещё раз, введя команду 'модели'.")
                    else:
                        user_action[id] = ""
                        model_name = models[int(msg) - 1]
                        model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)
                        if model_name == "base_segmentation" or model_name == "extra_remove":
                            model.load_state_dict(torch.load(os.path.join('models', model_name + '.pt')))
                        else:
                            model.load_state_dict(torch.load(os.path.join('models', str(id), model_name + '.pt')))
                        user_models[id] = model
                        send_message(id, "Выбрана модель " + model_name)
            elif msg == "начать" or msg == "помощь":
                keyboard = VkKeyboard(inline=True)
                keyboard.add_button('Формат файлов для обучения модели', color=VkKeyboardColor.PRIMARY)
                text = "Данный бот создан для того, чтобы вырезать фон у фотографий с лестницами-стремянками."\
                                 "Для этого отправьте изображение или документ с фотографией. Отправляйте изображения "\
                                 "по одному. Если вы хотите отправить несколько изображений, то загрузите "\
                                 "файл с расширением .zip. После этого бот отправит вам изображения. Изображения "\
                                 "поддерживаются только в формате jpg, jpeg и png.\n"\
                                 " Напишите 'модели' чтобы получить список моделей и выбрать, какую использовать.\n"\
                                 "Напишите 'обучить 10' и прикрепите архив необходимого формата, чтобы обучить модель. "\
                                 "Позже мы сможете её выбрать, написав 'модели'. Номер после слова 'обучить' обозначает"\
                                 " количество эпох для обучения. Максимум 100. По умолчанию 10."
                vk.messages.send(
                    user_id=id,
                    random_id=get_random_id(),
                    message=text, keyboard=keyboard.get_keyboard())
            elif "обучить" in msg:
                is_ok = False
                if not check_attachments(event) == "doc":
                    send_message(id, "Для обучения модели нужен архив с изображениями.")
                if msg == "обучить":
                    n_epochs = 10
                    is_ok = True
                elif not is_number(msg.split()[-1]):
                    send_message(id, "После слова 'обучить' должно быть число, обозначающее количество эпох. "
                                     "Можете оставить пустым для обучения на 10 эпох.")
                else:
                    n_epochs = int(msg.split()[-1])
                    if n_epochs > 100 or n_epochs <= 0:
                        send_message(id, "Количество эпох должно быть от 1 до 100.")
                    else:
                        is_ok = True
                if is_ok:
                    temp_dir = tempfile.TemporaryDirectory()
                    api_message = vk.messages.getById(message_ids=event.message_id)['items'][0]
                    document = api_message["attachments"][0]["doc"]
                    save_path = os.path.join(temp_dir.name, f"{id}.zip")
                    save_image_from_url(document["url"], save_path)
                    os.mkdir(os.path.join(temp_dir.name, f"{id}"))
                    extract_to = os.path.join(temp_dir.name, f"{id}")
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(path=extract_to)
                    send_message(id,
                                 "Файл загружен, начинаю обучение. В это время можно также отправить изображение для обработки.")
                    Thread(target=train, args=(extract_to, id, temp_dir, n_epochs)).start()

            elif 'формат' in msg:
                send_photo(id, "tip.png", "Ознакомьтесь с изображением. На нём представлен формат"
                                          "данных, необходимый для обучения. В сообщении с обучением должен присутствовать"
                                          "архив с данными в данном формате. После обучения модели вы сможете выбрать её.\n"
                                          "Фотографии маски должны быть в формате png, сами фотографии в формате jpg.\n"
                                          "Маски должны быть в папке 'labels', фотографии в папке 'images'. Должен присутствовать"
                                          "json файл, в котором указаны классы и их цвета. Если классов будет больше двух "
                                          "(фон и само изображение, которое необходимо вырезать), то это приведёт к "
                                          "неправильному вырезанию фото. Сама структура архива также представлена на изображении.")

            elif "модели" in msg:
                os.makedirs(os.path.join('models', str(id)), exist_ok=True)
                models = get_models(id)
                message = "Список доступных моделей:\n"
                for i, model in enumerate(models, start=1):
                    if i == 1:
                        message += f"{i}. {model} – базовая модель для вырезания фона\n"
                    elif i == 2:
                        message += f"{i}. {model} – более 'агрессивная' модель для вырезания\n"
                    else:
                        message += f"{i}. {model}\n"
                message += "\nНапишите номер модели для выбора. Базовая модель - 'base_segmentation.pt'"
                send_message(id, message)
                user_action[id] = "choose_model"
            elif check_attachments(event) == "photo":
                api_message = vk.messages.getById(message_ids=event.message_id)['items'][0]
                photo = api_message["attachments"][0]["photo"]
                temp_dir = tempfile.TemporaryDirectory()
                save_path = os.path.join(temp_dir.name, "remove_background.jpg")
                save_image_from_url(photo["sizes"][2]["url"], save_path)
                Thread(target=remove_background, args=(id, event, user_models[id], device, save_path, temp_dir)).start()
            elif check_attachments(event) == "doc":
                api_message = vk.messages.getById(message_ids=event.message_id)['items'][0]
                document = api_message["attachments"][0]["doc"]
                temp_dir = tempfile.TemporaryDirectory()
                if document["ext"] == "zip":
                    save_path = os.path.join(temp_dir.name, f"{id}.zip")
                    save_image_from_url(document["url"], save_path)
                    os.mkdir(os.path.join(temp_dir.name, f"{id}"))
                    extract_to = os.path.join(temp_dir.name, f"{id}")
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(path=extract_to)
                    send_message(id, "Файл загружен, начинаю обработку.")
                    Thread(target=remove_background, args=(id, event, user_models[id], device, extract_to, temp_dir, True)).start()
                elif document["ext"] == "jpg" or document["ext"] == "jpeg" or document["ext"] == "png":
                    save_path = os.path.join(temp_dir.name, f"{id}.jpg")
                    save_image_from_url(document["url"], save_path)
                    Thread(target=remove_background, args=(id, event, user_models[id], device, save_path, temp_dir)).start()
                else:
                    send_message(id, "Ваш тип документа не поддерживается, отправьте в формате zip, jpg, jpeg или png.")
                    temp_dir.cleanup()
            else:
                send_message(id, "В сообщении нет никаких вложений. Отправьте изображение или документ, чтобы "
                                 "получить результат.")


if __name__ == '__main__':
    longpoll = VkLongPoll(vk_session)
    print("Bot started")
    while True:
        try:
            main(longpoll)
        except requests.exceptions.ReadTimeout:
            print("read-timeout")
            time.sleep(600)
        except Exception as ex:
            print("Restarted", ex)
