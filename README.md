# Классификация изображений Intel

Классификация спутниковых изображений на 6 классов: buildings, forest, glacier, mountain, sea, street.

## Точность модели

- Тестовая точность: ~83%
- Лучшая валидационная точность: 76.6%

## Быстрый запуск веб-приложения

```bash
cd webapp
pip install -r ../requirements.txt
python manage.py migrate
python manage.py runserver


Запуск Jupyter ноутбука
bash
pip install -r requirements.txt
jupyter notebook notebook/Intel_Image_Classification.ipynb
Классы
buildings (здания)

forest (лес)

glacier (ледник)

mountain (гора)

sea (море)

street (улица)