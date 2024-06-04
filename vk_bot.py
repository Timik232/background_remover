import requests
from vk_api.utils import get_random_id
import vk_api
from private_api import VK_API
from vk_api.longpoll import VkLongPoll, VkEventType
import time
import numpy as np
from skimage import io
import urllib.request


vk_session = vk_api.VkApi(token=VK_API)
vk = vk_session.get_api()


def send_message(user_id: int, msg: str, stiker=None, attach=None) -> None:
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


def send_document(user_id, doc_req, message=None):
    upload = vk.VkUpload(vk_session)
    document = upload.document_message(doc_req)[0]
    print(document)
    owner_id = document['owner_id']
    doc_id = document['id']
    attachment = f'doc{owner_id}_{doc_id}'
    post = {'user_id': user_id, 'random_id': 0, "attachment": attachment}
    if message is not None:
        post['message'] = message
    try:
        vk_session.method('messages.send', post)
    except BaseException:
        send_message(user_id, "Не удалось отправить документ")
        return


def send_photo(user_id, img_req, message=None):
    upload = vk_api.VkUpload(vk_session)
    photo = upload.photo_messages(img_req)[0]
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
    img = io.imread(img_path)
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


def main(longpoll):
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
                                 "файл с расширением .zip. После этого бот отправит вам изображения.")
            if check_attachments(event) == "photo":
                send_message(id, "В сообщении есть изображение")
            elif check_attachments(event) == "doc":
                send_message(id, "В сообщении есть документ")
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

