from aiogram import *
from aiogram.types import Message, FSInputFile
import asyncio
import requests
from io import BytesIO
from aiogram.filters import CommandStart
# from constains import*

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model_path = 'best5.pt'
token_api = "7126742135:AAEs4-OVgJGXaZK2JAe3spml5lR9AxiJHZI"
image_path = 'flower_image0.jpg'  # Замените на путь к вашему изображению
model = YOLO(model_path)

bot = Bot(token=token_api)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(msg: types.Message):
    await msg.answer("Привет! Я бот, который поможет вам определить качество сварки. Просто отправьте мне изображение сварного шва, и я постараюсь оценить его качество. \nДля начала отправьте фото! 🛠️"+"\n" + "")
@dp.message(F.photo)
async def handle_photo(message: Message):
    global photo_url, model, model_path, image_path
    # Получаем информацию о фотографии
    photo = message.photo[-1]  # Берем последнюю (самую большую) версию фото
    file_id = photo.file_id
    i = 0

    # Получаем URL фотографии
    file_path = await bot.get_file(file_id)
    photo_url = f"https://api.telegram.org/file/bot{token_api}/{file_path.file_path}"
    print(photo_url)
    response = requests.get(photo_url)
    if response.status_code == 200:
        # Сохранить изображение на диск
        with open(f"flower_image{i}.jpg", "wb") as file:
            file.write(response.content)
            # driveSkript(i)
            i += 1

            file.close()
        print("Изображение успешно скачано и сохранено как flower_image.jpg")
    else:
        print("Не удалось скачать изображение. Пожалуйста, проверьте URL и попробуйте снова.")
    # await message.answer(f"URL адрес фото: {photo_url}")
    results = model.predict(source=image_path)
    file = BytesIO()

    # Получаем изображение с предсказаниями
    predicted_image = results[0].save('flower.jpg') # Метод plot() возвращает изображение с нарисованными предсказаниями
    file.seek(0)
    file = FSInputFile('flower.jpg', filename='photo.jpg')


    predicted_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]  # Для возврата

    await message.answer_photo(file)
    if len(predicted_classes) == 0:
        await message.answer("Сварка не обнаружена")
    for predict in range(0, len(predicted_classes)):
        if (predicted_classes[predict] == 'Good Welding'):
            for r in model(image_path):
                for x in range(0,4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'True\nGood Welding:\n{string}')
        elif (predicted_classes[predict] == 'Bad Welding'):
            for r in model(image_path):
                for x in range(0, 4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'Flase\nBad Welding:\n{string}')
        elif (predicted_classes[predict] == 'Crack'):
            for r in model(image_path):
                for x in range(0, 4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'Crack:\n{string}')
        elif (predicted_classes[predict] == 'Excess Reinforcement'):
            for r in model(image_path):
                for x in range(0,4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'Flase\nExcess Reinforcement:\n{string}')
                # await message.answer(f'Excess Reinforcement:\n{r.boxes.xywh.cpu().numpy()[predict]}')
        elif (predicted_classes[predict] == 'Porosity'):
            for r in model(image_path):
                for x in range(0,4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'Flase\nPorosity:\n{string}')
                # await message.answer(f'Porosity:\n{r.boxes.xywh.cpu().numpy()[predict]}')
        elif (predicted_classes[predict] == 'Spatters'):
            for r in model(image_path):
                for x in range(0,4):
                    if x == 0:
                        string = f'Start x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 1:
                        string += f'\nStart y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 2:
                        string += f'\nEnd x: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                    elif x == 3:
                        string += f'\nEnd y: {r.boxes.xyxy.cpu().numpy()[predict][x]}'
                await message.answer(f'Flase\nSpatters:\n{string}')
                # await message.answer(f'Spatters:\n{r.boxes.xywh.cpu().numpy()[predict]}')




                # await message.answer(f'{r.boxes.xywh.cpu().numpy()[predict]}')
            # await message.answer_photo(file)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

# 7126742135:AAEs4-OVgJGXaZK2JAe3spml5lR9AxiJHZI
  