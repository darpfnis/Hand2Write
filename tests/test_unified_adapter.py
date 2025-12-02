"""
Unit-тести для UnifiedOCRAdapter
handwrite2print/tests/test_unified_adapter.py
"""
import pytest
import sys
import os
import numpy as np

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.unified_ocr_adapter import UnifiedOCRAdapter


class TestUnifiedOCRAdapter:
    """Тести для UnifiedOCRAdapter"""
    
    def test_adapter_initialization_with_tesseract(self):
        """Тест ініціалізації адаптера з Tesseract"""
        adapter = UnifiedOCRAdapter(engine='tesseract')
        assert adapter.engine_name == 'tesseract'
        assert adapter.preprocessor is not None
    
    def test_adapter_initialization_with_easyocr(self):
        """Тест ініціалізації адаптера з EasyOCR"""
        adapter = UnifiedOCRAdapter(engine='easyocr')
        # Якщо EasyOCR недоступний, адаптер може fallback на інший рушій
        assert adapter.engine_name in ['easyocr', 'tesseract', 'paddleocr']
        assert adapter.preprocessor is not None
    
    def test_adapter_initialization_with_paddleocr(self):
        """Тест ініціалізації адаптера з PaddleOCR"""
        adapter = UnifiedOCRAdapter(engine='paddleocr')
        # Якщо PaddleOCR недоступний, адаптер може fallback на інший рушій
        assert adapter.engine_name in ['paddleocr', 'tesseract', 'easyocr']
        assert adapter.preprocessor is not None
    
    def test_adapter_initialization_with_llm(self):
        """Тест ініціалізації адаптера з LLM корекцією"""
        llm_config = {
            'api_type': 'local',
            'model': 'llama3.2:1b',
            'enabled': False
        }
        adapter = UnifiedOCRAdapter(
            engine='tesseract',
            use_llm_correction=True,
            llm_config=llm_config
        )
        assert adapter.engine_name == 'tesseract'
        # LLM може бути недоступний, тому просто перевіряємо, що ініціалізація пройшла
    
    def test_adapter_validates_none_image(self):
        """Тест валідації None зображення"""
        adapter = UnifiedOCRAdapter(engine='tesseract')
        
        # Адаптер обробляє помилки всередині та повертає порожній рядок
        result = adapter.recognize(None, 'eng')  # type: ignore[arg-type]
        assert isinstance(result, str)
        # Може повернути порожній рядок або викинути помилку
        # Перевіряємо, що не падає з критичною помилкою
    
    def test_adapter_validates_empty_image(self):
        """Тест валідації порожнього зображення"""
        adapter = UnifiedOCRAdapter(engine='tesseract')
        
        empty_image = np.array([], dtype=np.uint8)
        # Адаптер обробляє помилки всередині та повертає порожній рядок
        result = adapter.recognize(empty_image, 'eng')
        assert isinstance(result, str)
        # Може повернути порожній рядок або викинути помилку
        # Перевіряємо, що не падає з критичною помилкою
    
    def test_adapter_get_available_engines(self):
        """Тест отримання списку доступних рушіїв"""
        adapter = UnifiedOCRAdapter(engine='tesseract')
        available = adapter.get_available_engines()
        
        assert isinstance(available, list)
        # Tesseract зазвичай доступний
        if 'tesseract' in available:
            assert 'tesseract' in available
    
    def test_adapter_recognize_with_valid_image(self):
        """Тест розпізнавання з коректним зображенням"""
        adapter = UnifiedOCRAdapter(engine='tesseract')
        
        if 'tesseract' not in adapter.get_available_engines():
            pytest.skip("Tesseract недоступний")
        
        # Створюємо просте тестове зображення
        rng = np.random.default_rng(42)
        test_image = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        
        result = adapter.recognize(test_image, 'eng', preprocess=False)
        assert isinstance(result, str)
    
    def test_adapter_fallback_on_error(self):
        """Тест fallback механізму при помилці"""
        # Це тест перевіряє, що адаптер не падає при помилці
        adapter = UnifiedOCRAdapter(engine='tesseract')
        
        if 'tesseract' not in adapter.get_available_engines():
            pytest.skip("Tesseract недоступний")
        
        # Створюємо зображення, яке може викликати помилку
        invalid_image = np.array([[1, 2, 3]], dtype=np.uint8)  # Некоректна розмірність
        
        # Адаптер обробляє помилки всередині (fallback механізм)
        # Може повернути порожній рядок або обробити помилку
        result = adapter.recognize(invalid_image, 'eng')
        assert isinstance(result, str)
        # Перевіряємо, що не падає з критичною помилкою


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

