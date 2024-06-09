import json
import os
import shutil
import zipfile

from ultralytics import YOLO
import requests
from vk_api.utils import get_random_id
import vk_api
from private_api import VK_API
from vk_api.longpoll import VkLongPoll, VkEventType
import time
import numpy as np
from skimage import io
import urllib.request
import tempfile
from threading import Thread
import cv2

vk_session = vk_api.VkApi(token=VK_API)
vk = vk_session.get_api()


def send_message(user_id: int, msg: str, stiker=None, attach=None) -> None:
    """
    Send message to user
    :param user_id: user id
    :param msg: text message
    :param stiker: id of the sticker in vk
    :param attach: attachment
    :return: None
    """
    try:
        vk.messages.send(
            user_id=user_id,
            random_id=get_random_id(),
            message=msg,
            sticker_id=stiker,
            attachment=attach
        )
    except BaseException as ex:
        print(ex)
        return


def send_document(user_id: int, event, doc_path: str) -> None:
    """
    Send document to user
    :param user_id: vk id
    :param event: event of vk
    :param doc_path: path to the document
    :return: None
    """
    result = json.loads(requests.post(vk.docs.getMessagesUploadServer(type='doc', peer_id=user_id)['upload_url'],
                                      files={'file': open(doc_path, 'rb')}).text)
    json_answer = vk.docs.save(file=result['file'], title='title', tags=[])
    try:
        vk.messages.send(
            peer_id=user_id,
            random_id=0,
            attachment=f"doc{json_answer['doc']['owner_id']}_{json_answer['doc']['id']}"
        )
    except BaseException:
        send_message(user_id, "Не удалось отправить документ")



def send_photo(user_id: int, img_path: str, message=None):
    """
        Send photo to user
        :param user_id: vk id
        :param img_path: path to the document
        :param message: message
        :return: None
        """
    upload = vk_api.VkUpload(vk_session)
    photo = upload.photo_messages(img_path)[0]
    owner_id = photo['owner_id']
    photo_id = photo['id']
    attachment = f'photo{owner_id}_{photo_id}'
    post = {'user_id': user_id, 'random_id': 0, "attachment": attachment}
    if message is not None:
        post['message'] = message
    try:
        vk_session.method('messages.send', post)
    except BaseException:
        send_message(user_id, "Не удалось отправить картинку")
        return


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


def save_image_from_url(image_url: str, file_name: str):
    """
    Save image from url
    :param image_url: url of the image
    :param file_name: name of the file
    :return: None
    """
    urllib.request.urlretrieve(image_url, file_name)
    print(F'File "{file_name}" was saved to temporary')


def remove_background(user_id: int, model: YOLO, image_path: str, temp_dir) -> None:
    import matplotlib.pyplot as plt
    image = transform_photo(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    path = ""
    results = model(image_rgb)
    print(results[0].masks.xyn)
    mask = results[0].masks[0]  # Если у вас несколько каналов, выберите нужный

    # Преобразование тензора в массив numpy для визуализации
    mask_np = mask.numpy()

    # Визуализация маски
    plt.imshow(mask_np, cmap='gray')  # Используйте cmap='gray' для отображения в градациях серого
    plt.axis('off')  # Отключение осей для чистого изображения
    plt.show()

    # cv2.imwrite(mask_filename, obj_mask)
    # path = os.path.join(temp_dir, f"{user_id}_{i}.png")
    # cv2.imwrite(path, segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    temp_dir.cleanup()
    # send_photo(user_id, "mask.png")


def main(longpoll):
    """
    main function of the program
    """
    model = YOLO('segmentation.pt')
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            id = event.user_id
            try:
                msg = event.text.lower()
            except AttributeError:
                msg = ""

            if msg == "начать" or msg == "помощь":
                send_message(id, "Данный бот создан для того, чтобы вырезать фон у фотографий с лестницами-стремянками."
                                 "Для этого отправьте изображение или документ с фотографией. Отправляйте изображения "
                                 "по одному. Если вы хотите отправить несколько изображений, то загрузите "
                                 "файл с расширением .zip. После этого бот отправит вам изображения. Изображения "
                                 "поддерживаются только в формате jpg, jpeg и png.")
            if check_attachments(event) == "photo":
                api_message = vk.messages.getById(message_ids=event.message_id)['items'][0]
                photo = api_message["attachments"][0]["photo"]
                temp_dir = tempfile.TemporaryDirectory()
                save_path = os.path.join(temp_dir.name, "remove_background.jpg")
                save_image_from_url(photo["sizes"][2]["url"], save_path)
                Thread(target=remove_background, args=(id, model, save_path, temp_dir)).start()
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
                    new_extract = os.path.join(temp_dir.name, "file")
                    shutil.make_archive(new_extract, 'zip', root_dir=extract_to)
                    send_document(id, event, new_extract + ".zip")
                    temp_dir.cleanup()
                elif document["ext"] == "jpg" or document["ext"] == "jpeg" or document["ext"] == "png":
                    save_path = os.path.join(temp_dir.name, f"{id}.jpg")
                    save_image_from_url(document["url"], save_path)
                    send_document(id, event, save_path)
                    temp_dir.cleanup()
                else:
                    send_message(id, "Ваш тип документа не поддерживается, отправьте в формате zip, jpg, jpeg или png.")
                    temp_dir.cleanup()
                # print(document)
                # save_path = os.path.join(temp_dir.name, "remove_background.jpg")
                # save_image_from_url(document["sizes"][2]["url"], save_path)
            else:
                send_document(id, event, os.path.join("data", "full quality", "4.jpg"))
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
