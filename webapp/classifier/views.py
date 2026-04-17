import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Импортируем класс ImageClassifier
from .classifier import ImageClassifier

# Создаём экземпляр классификатора напрямую из кода
classifier = ImageClassifier()

def index(request):
    return render(request, 'classifier/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        # Сохраняем загруженный файл
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = fs.path(filename)
        
        # Предсказание
        try:
            result = classifier.predict(file_path)
            
            # Формируем данные для шаблона
            context = {
                'image_url': fs.url(filename),
                'predicted_class': result['predicted_class'],
                'confidence': round(result['confidence'], 1),
                'top_3': result['top_3'],
                'all_predictions': result['all_predictions'],
                'error': None
            }
        except Exception as e:
            context = {
                'error': str(e),
                'image_url': None,
                'predicted_class': None,
                'confidence': None,
                'top_3': [],
                'all_predictions': {}
            }
        
        return render(request, 'classifier/result.html', context)
    
    return render(request, 'classifier/index.html')