# Классификация изображений 

Классификация спутниковых изображений на 6 классов: buildings, forest, glacier, mountain, sea, street.

## Запуск веб-приложения Django

```bash
git clone https://github.com/ZeroUzer/intel-image-classification.git
cd intel-image-classification
pip install -r requirements.txt
cd webapp
python manage.py migrate
python manage.py runserver
```

## Как пользоваться
1. Нажмите на область загрузки
2. Выберите изображение (JPG, JPEG, PNG, до 10 МБ)
3. Нажмите "Распознать"
4. Результат покажет:
- Предсказанный класс
- Уверенность в процентах

Примеры изображений:
- В папке notebook/test_photos/ есть примеры для тестирования.

## Запуск Jupyter ноутбука (для изучения обучения модели)
```bash
jupyter notebook notebook/Intel_Image_Classification.ipynb
```

## Структура проекта
1. intel-image-classification
    1. webapp/ # Django веб-приложение (основная демонстрация)
    - classifier/ # Основное приложение
        - views.py # Логика предсказания
        - classifier.py # Класс ImageClassifier
            - index.html # Форма загрузки
            - result.html # Страница результата
        - model/ 
            - final_model.h5 # Обученная модель
        - manage.py
    2. notebook/ # Jupyter ноутбук (обучение модели)
    - Intel_Image_Classification.ipynb
    - model/final_model.h5
    - results/ # Графики и метрики
- requirements.txt # Все зависимости
- README.md

