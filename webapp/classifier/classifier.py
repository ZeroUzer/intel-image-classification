import os
import cv2
import numpy as np
import pickle
from datetime import datetime
from tensorflow import keras

class ImageClassifier:
    """Классификатор изображений для Intel dataset"""
    
    def __init__(self, model_path=None, class_names=None, img_size=(150, 150)):
        """
        Инициализация классификатора
        
        Args:
            model_path: путь к файлу модели (.h5)
            class_names: список названий классов
            img_size: размер изображения для модели (height, width)
        """
        print("Инициализация классификатора...")
        
        # Определяем пути к файлам
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, 'model', 'final_model.h5')
        
        # Загружаем модель
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        print(f"Модель загружена: {model_path}")
        
        # Загружаем классы
        if class_names is None:
            class_info_path = os.path.join(base_dir, 'results', 'class_info.json')
            if os.path.exists(class_info_path):
                import json
                with open(class_info_path, 'r', encoding='utf-8') as f:
                    class_info = json.load(f)
                    self.class_names = class_info['classes']
            else:
                self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
                print("Используются стандартные классы Intel")
        else:
            self.class_names = class_names
        
        self.img_size = img_size
        print("Классификатор готов")
        print(f"  Классы: {len(self.class_names)}")
        print(f"  Размер изображений: {self.img_size[0]}x{self.img_size[1]}")
    
    def predict(self, image_path):
        """
        Предсказание класса для изображения
        
        Args:
            image_path: путь к файлу изображения
            
        Returns:
            dict: результат предсказания
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        if not any(image_path.endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Неподдерживаемый формат файла: {image_path}")
        
        # Загрузка и обработка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]
        
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Предсказание
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        pred_idx = np.argmax(predictions)
        pred_class = self.class_names[pred_idx]
        confidence = predictions[pred_idx] * 100
        
        # Топ-3 предсказания
        top_indices = predictions.argsort()[-3:][::-1]
        top_predictions = [
            (self.class_names[i], predictions[i] * 100) 
            for i in top_indices
        ]
        
        result = {
            'file_name': os.path.basename(image_path),
            'file_path': image_path,
            'original_size': (original_width, original_height),
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_predictions': {self.class_names[i]: float(predictions[i]) 
                               for i in range(len(self.class_names))},
            'top_3': top_predictions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def predict_and_show(self, image_path):
        """
        Предсказание с визуализацией
        
        Args:
            image_path: путь к файлу изображения
            
        Returns:
            dict: результат предсказания
        """
        import matplotlib.pyplot as plt
        
        result = self.predict(image_path)
        
        fig = plt.figure(figsize=(16, 6))
        
        # Исходное изображение
        ax1 = plt.subplot(1, 3, 1)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.set_title('Ваше изображение')
        ax1.set_xlabel(f"Размер: {result['original_size'][0]}x{result['original_size'][1]}")
        ax1.axis('off')
        
        # Результат предсказания
        ax2 = plt.subplot(1, 3, 2)
        ax2.axis('off')
        
        text_y = 0.7
        ax2.text(0.5, text_y, 'Результат:', 
                fontsize=16, ha='center', va='center', fontweight='bold')
        
        text_y -= 0.15
        color = 'green' if result['confidence'] > 70 else 'orange'
        ax2.text(0.5, text_y, result['predicted_class'], 
                fontsize=32, ha='center', va='center', 
                color=color, fontweight='bold')
        
        text_y -= 0.15
        ax2.text(0.5, text_y, f"{result['confidence']:.1f}%", 
                fontsize=24, ha='center', va='center')
        
        text_y -= 0.1
        ax2.text(0.5, text_y, 'уверенность', 
                fontsize=14, ha='center', va='center')
        
        # График вероятностей
        ax3 = plt.subplot(1, 3, 3)
        
        classes = list(result['all_predictions'].keys())
        probabilities = [result['all_predictions'][cls] * 100 for cls in classes]
        
        colors = []
        for i, cls in enumerate(classes):
            if cls == result['predicted_class']:
                colors.append('green')
            elif probabilities[i] > 10:
                colors.append('orange')
            else:
                colors.append('lightgray')
        
        bars = ax3.barh(classes, probabilities, color=colors)
        ax3.set_xlabel('Вероятность (%)')
        ax3.set_title('Вероятности по классам')
        ax3.set_xlim([0, 100])
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')
        
        for bar, prob in zip(bars, probabilities):
            if prob > 1:
                ax3.text(bar.get_width() + 1, 
                        bar.get_y() + bar.get_height()/2,
                        f'{prob:.1f}%', va='center', fontsize=9)
        
        plt.suptitle(f'Классификация: {result["file_name"]}')
        plt.tight_layout()
        plt.show()
        
        # Текстовый вывод
        print(f"Файл: {result['file_name']}")
        print(f"Размер: {result['original_size'][0]}x{result['original_size'][1]}")
        print(f"Предсказанный класс: {result['predicted_class']}")
        print(f"Уверенность: {result['confidence']:.1f}%")
        print("Топ-3 предсказания:")
        for i, (cls, conf) in enumerate(result['top_3'], 1):
            print(f"  {i}. {cls}: {conf:.1f}%")
        
        return result