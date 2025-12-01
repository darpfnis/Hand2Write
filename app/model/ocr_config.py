"""
Конфігурація OCR та LLM
handwrite2print/app/model/ocr_config.py
"""
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os


class OCRConfig:
    """Конфігурація для OCR та LLM"""
    
    # Дефолтні налаштування препроцесингу
    DEFAULT_PREPROCESSING = {
        'binarization': True,
        'denoise': True,
        'deskew': True,
        'enhance_contrast': True,
        'sharpen': True,
        'morphology': True,
        'upscale_small': True,
        'remove_borders': False
    }
    
    # Дефолтні налаштування LLM
    DEFAULT_LLM_CONFIG = {
        'api_type': 'local',  # 'openai', 'ollama', 'local'
        'api_key': None,
        'api_url': 'http://localhost:11434',  # Для Ollama
        'model': 'llama3.2:1b',  # Дефолтна модель - дуже легка для обмеженої пам'яті (~1-2 GB)
        'enabled': False
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Ініціалізація конфігурації
        
        Args:
            config_file: шлях до файлу конфігурації
        """
        self.config_file = config_file or Path(__file__).parent.parent.parent / 'ocr_config.json'
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Завантаження конфігурації"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Дефолтна конфігурація
        return {
            'preprocessing': self.DEFAULT_PREPROCESSING.copy(),
            'llm': self.DEFAULT_LLM_CONFIG.copy(),
            'default_engine': 'tesseract',
            'fallback_enabled': True
        }
    
    def save_config(self):
        """
        Збереження конфігурації з безпечним обробленням API ключів
        
        API ключі не зберігаються у файлі, якщо вони доступні через змінні середовища
        """
        try:
            self.config_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Створюємо копію конфігурації для збереження
            config_to_save = self.config.copy()
            
            # Безпека: не зберігаємо API ключі у файлі, якщо вони доступні через змінні середовища
            if 'llm' in config_to_save:
                llm_config = config_to_save['llm'].copy()
                # Якщо API ключ доступний через змінну середовища, не зберігаємо його
                if llm_config.get('api_type') == 'openai':
                    env_key = os.getenv('OPENAI_API_KEY')
                    if env_key and llm_config.get('api_key') == env_key:
                        # Не зберігаємо ключ, якщо він зі змінної середовища
                        llm_config['api_key'] = None
                        llm_config['_use_env_key'] = True
                config_to_save['llm'] = llm_config
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Помилка збереження конфігурації: {e}")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Отримання конфігурації препроцесингу"""
        return self.config.get('preprocessing', self.DEFAULT_PREPROCESSING.copy())
    
    def _apply_env_api_key(self, llm_config: Dict[str, Any]) -> None:
        """Застосування API ключа зі змінної середовища"""
        env_api_key = os.getenv('OPENAI_API_KEY')
        if env_api_key:
            llm_config['api_key'] = env_api_key
            llm_config['_use_env_key'] = True
        elif not llm_config.get('api_key') and os.getenv('OPENAI_API_KEY'):
            llm_config['api_key'] = os.getenv('OPENAI_API_KEY')
            llm_config['_use_env_key'] = True
    
    def _apply_env_api_url(self, llm_config: Dict[str, Any]) -> None:
        """Застосування API URL зі змінної середовища"""
        env_api_url = os.getenv('LLM_API_URL')
        if env_api_url:
            llm_config['api_url'] = env_api_url
        elif not llm_config.get('api_url') and os.getenv('LLM_API_URL'):
            llm_config['api_url'] = os.getenv('LLM_API_URL')
    
    def _auto_detect_llm_availability(self, llm_config: Dict[str, Any]) -> None:
        """Автоматична перевірка доступності LLM"""
        if llm_config.get('enabled') is None or llm_config.get('auto_detect', True):
            try:
                from .llm_postprocessor import LLMPostProcessor
                processor = LLMPostProcessor(
                    api_type=llm_config.get('api_type', 'local'),
                    api_key=llm_config.get('api_key'),
                    api_url=llm_config.get('api_url')
                )
                if processor.is_available():
                    llm_config['enabled'] = True
                    llm_config['auto_detected'] = True
                else:
                    llm_config['enabled'] = False
            except Exception:
                llm_config['enabled'] = False
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Отримання конфігурації LLM з автоматичною перевіркою доступності"""
        llm_config = self.config.get('llm', self.DEFAULT_LLM_CONFIG.copy())
        # Переконаємося, що модель є в конфігурації (якщо не вказана, використовуємо дефолтну)
        if 'model' not in llm_config or not llm_config.get('model'):
            llm_config['model'] = self.DEFAULT_LLM_CONFIG.get('model', 'llama3.2:1b')
        
        # Перевірка змінних середовища (пріоритет над файлом конфігурації)
        self._apply_env_api_key(llm_config)
        self._apply_env_api_url(llm_config)
        
        # Автоматична перевірка доступності LLM
        self._auto_detect_llm_availability(llm_config)
        
        return llm_config
    
    def get_default_engine(self) -> str:
        """Отримання дефолтного рушія"""
        return self.config.get('default_engine', 'tesseract')
    
    def is_fallback_enabled(self) -> bool:
        """Чи увімкнено fallback"""
        return self.config.get('fallback_enabled', True)
    
    def set_preprocessing_config(self, config: Dict[str, Any]):
        """Встановлення конфігурації препроцесингу"""
        self.config['preprocessing'] = config
        self.save_config()
    
    def set_llm_config(self, config: Dict[str, Any]):
        """Встановлення конфігурації LLM"""
        self.config['llm'] = config
        self.save_config()
    
    def set_default_engine(self, engine: str):
        """Встановлення дефолтного рушія"""
        self.config['default_engine'] = engine
        self.save_config()

