import vk_api
from private_api import VK_API
from vk_api.utils import get_random_id
import json
import requests

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