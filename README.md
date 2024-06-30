# Выпускной проект академии Samsung по треку «Искусственный интеллект»
Данный проект создан для вырезания фона у строительных предметов, в первую очередь у лестниц-стремянок и лестниц-трансформеров. Для работы используется модель VIT Transformer + Unet для семантической сегментации. Сам проект работает в vk-api, можно отправлять как изображения, так и архив с изображениями. После отправки изображения, бот пришлёт файлом изображение с вырезанным фоном. Присутствует возможность дообучать модель. \
Ссылка на веса модели: https://www.kaggle.com/models/ser13volk/ladder-segmentation \
Ссылка на датасет: https://www.kaggle.com/datasets/ser13volk/ladder-segmentation
# Запуск
Необходимо создать файл private_api, в котором нужно добавить переменную VK_API с api из сообщества ВКонтакте, чтобы бот мог отправлять сообщения. \
Затем установить зависимости из requirements.txt: ``` pip install -r requirements.txt```\
Запустить файл vk_bot, после чего отправить сообщение "старт" либо "помощь" боту, и он подскажет дальнейшие инструкции. 


