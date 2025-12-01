"""
Конфігураційний файл
handwrite2print/app/config.py
"""
import os
from pathlib import Path


class Config:
    """Налаштування програми"""
    
    # Версія програми
    VERSION = "1.0.0"
    APP_NAME = "Система перетворення рукописного тексту в друкований"
    
    # Шляхи
    BASE_DIR = Path(__file__).parent.parent
    APP_DIR = BASE_DIR / "app"
    RESOURCES_DIR = BASE_DIR / "resources"
    MODELS_DIR = RESOURCES_DIR / "models"
    PADDLEOCR_MODELS_DIR = MODELS_DIR / "paddleocr"  # Папка для моделей PaddleOCR в проекті
    TEMP_DIR = Path(os.getenv('TEMP', '/tmp'))
    
    # Створення директорій, якщо не існують
    RESOURCES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    PADDLEOCR_MODELS_DIR.mkdir(exist_ok=True, parents=True)  # Створюємо папку для PaddleOCR моделей
    
    # OCR налаштування
    DEFAULT_OCR_ENGINE = "tesseract"
    DEFAULT_LANGUAGE = "eng"
    
    TESSERACT_PATHS = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\%USERNAME%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    ]
    
    # Підтримувані формати
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    SUPPORTED_EXPORT_FORMATS = ['.txt', '.docx', '.pdf', '.html']
    
    # Обмеження
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_WIDTH = 4000
    MAX_IMAGE_HEIGHT = 4000
    
    # Налаштування попередньої обробки
    PREPROCESSING = {
        'binarization': True,
        'denoise': True,
        'deskew': True,
        'resize': True,
        'enhance_contrast': False,
        'remove_borders': False
    }
    
    # Налаштування розпізнавання
    OCR_CONFIG = {
        'confidence_threshold': 60,
        'psm_mode': 6,  # Tesseract PSM mode
        'oem_mode': 3,  # Tesseract OEM mode
    }
    
    # Налаштування інтерфейсу
    UI_CONFIG = {
        'window_width': 1400,
        'window_height': 900,
        'canvas_min_width': 600,
        'canvas_min_height': 400,
        'default_pen_width': 4,
        'default_pen_color': '#000000'
    }
    
    # Кольорова схема
    COLORS = {
        'primary': '#2196F3',
        'primary_dark': '#1976D2',
        'primary_light': '#BBDEFB',
        'accent': '#4CAF50',
        'error': '#F44336',
        'warning': '#FF9800',
        'text_primary': '#212121',
        'text_secondary': '#757575',
        'background': '#FAFAFA',
        'surface': '#FFFFFF'
    }
    
    # Мовні мапінги
    LANGUAGE_MAPPING = {
        'Українська': 'ukr',
        'Англійська': 'eng'
    }
    
    # Мапінг рушіїв OCR
    ENGINE_MAPPING = {
        'Tesseract': 'tesseract',
        'EasyOCR': 'easyocr',
        'PaddleOCR': 'paddleocr'
    }
    
    # Логування
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'app.log'
    
    @classmethod
    def get_tesseract_path(cls):
        """Пошук шляху до Tesseract"""
        for path in cls.TESSERACT_PATHS:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return None
        
    @classmethod
    def is_valid_image_format(cls, file_path):
        """Перевірка формату зображення"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_IMAGE_FORMATS
        
    @classmethod
    def is_valid_export_format(cls, file_path):
        """Перевірка формату експорту"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXPORT_FORMATS
        
    @classmethod
    def get_model_path(cls, model_name):
        """Отримання шляху до моделі"""
        return cls.MODELS_DIR / model_name
        
    @classmethod
    def save_settings(cls, settings):
        """Збереження налаштувань користувача"""
        import json
        settings_file = cls.BASE_DIR / 'settings.json'
        
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Помилка збереження налаштувань: {e}")
            return False
            
    @classmethod
    def load_settings(cls):
        """Завантаження налаштувань користувача"""
        import json
        settings_file = cls.BASE_DIR / 'settings.json'
        
        if not settings_file.exists():
            return {}
            
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Помилка завантаження налаштувань: {e}")
            return {}


# Експорт конфігурації
config = Config()
