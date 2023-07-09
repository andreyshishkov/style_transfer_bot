import torch
import os
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from settings.config import TOKEN
from settings.messages import START_MESSAGE
from data_base.dbalchemy import DBManager
from upgrade_style_transfer.transfer_style import transfer_style as transfer
from upgrade_style_transfer.model import Net


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

data_base = DBManager()


model = Net(ngf=128)
model_dict = torch.load('upgrade_style_transfer/21styles.model')
model_dict_clone = model_dict.copy()
for key, value in model_dict_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del model_dict[key]
model.load_state_dict(model_dict, False)


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    if message.chat.id not in data_base.get_all_chat_ids():
        data_base.insert_chat_id(message.chat.id)
    await message.reply(START_MESSAGE)


@dp.message_handler(commands=['change_style'])
async def change_style(message: types.Message):
    chat_id = message.chat.id
    data_base.change_mode(chat_id, False)
    await message.reply('Включен режим "Ввод стиля". Отправьте изображение стиля')


@dp.message_handler(commands=['transfer_style'])
async def transfer_style(message: types.Message):
    chat_id = message.chat.id
    data_base.change_mode(chat_id, True)
    await message.reply('Включен режим "Перенос стиля". Отправьте изображение, на которое хотите перенести стиль')


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    chat_id = message.chat.id
    is_transfer_mode = data_base.get_curr_mode(chat_id)

    if is_transfer_mode:
        tmp_image_name = message.photo[-1].file_id
        image_path = f'tmp/{tmp_image_name}.jpg'
        style_path = data_base.get_curr_style(chat_id)
        await message.photo[-1].download(image_path)
        await message.answer('Ваше фото загружено. Начался перенос стиля. Это может занять пару минут...')

        new_image_path = transfer(image_path, style_path, model)
        torch.cuda.empty_cache()
        new_image = types.InputFile(new_image_path)
        await bot.send_photo(chat_id, new_image, caption='Ваше фото))')

        os.remove(image_path)
        os.remove(new_image_path)

    else:
        curr_style_path = data_base.get_curr_style(chat_id)
        new_style_path = f'styles/{message.photo[-1].file_id}.jpg'
        await message.photo[-1].download(new_style_path)

        if curr_style_path != 'styles/default.jpg':
            os.remove(curr_style_path)

        data_base.change_path(chat_id, new_style_path)

        await message.answer('Стиль успешно изменен')


if __name__ == '__main__':
    executor.start_polling(dp)
