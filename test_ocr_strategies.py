"""
Unit-тести для OCR стратегій
handwrite2print/tests/test_ocr_strategies.py
"""
import pytest
import sys
import os
import numpy as np

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.ocr_strategies import OCRStrategy, TesseractStrategy, EasyOCRStrategy, PaddleOCRStrategy


class TestOCRStrategy:
    """Тести для базового класу OCRStrategy"""
    
    def test_strategy_is_abstract(self):
        """Тест, що OCRStrategy є абстрактним класом"""
        # Спробуємо створити екземпляр - має викликати TypeError
        with pytest.raises(TypeError):
            OCRStrategy()  # type: ignore[abstract]


class TestTesseractStrategy:
    """Тести для TesseractStrategy"""
    
    def test_tesseract_initialization(self):
        """Тест ініціалізації Tesseract стратегії"""
        strategy = TesseractStrategy()
        assert isinstance(strategy, OCRStrategy)
        assert isinstance(strategy.is_available(), bool)
        assert strategy.get_name() in ['Tesseract', 'tesseract']
    
    def test_tesseract_recognize_with_invalid_image(self):
        """Тест розпізнавання з некоректним зображенням"""
        strategy = TesseractStrategy()
        if not strategy.is_available():
            pytest.skip("Tesseract недоступний")
        
        # Порожнє зображення
        empty_image = np.array([], dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError)):
            strategy.recognize(empty_image, 'eng')
    
    def test_tesseract_recognize_with_valid_image(self):
        """Тест розпізнавання з коректним зображенням"""
        strategy = TesseractStrategy()
        if not strategy.is_available():
            pytest.skip("Tesseract недоступний")
        
        # Створюємо просте тестове зображення (білий квадрат)
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Tesseract може повернути порожній рядок для білого зображення
        result = strategy.recognize(test_image, 'eng')
        assert isinstance(result, str)


class TestEasyOCRStrategy:
    """Тести для EasyOCRStrategy"""
    
    def test_easyocr_initialization(self):
        """Тест ініціалізації EasyOCR стратегії"""
        strategy = EasyOCRStrategy()
        assert isinstance(strategy, OCRStrategy)
        assert isinstance(strategy.is_available(), bool)
        assert strategy.get_name() in ['EasyOCR', 'easyocr']
    
    def test_easyocr_recognize_with_invalid_image(self):
        """Тест розпізнавання з некоректним зображенням"""
        strategy = EasyOCRStrategy()
        if not strategy.is_available():
            pytest.skip("EasyOCR недоступний")
        
        # Порожнє зображення
        empty_image = np.array([], dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError, Exception)):
            strategy.recognize(empty_image, 'eng')
    
    def test_easyocr_recognize_with_valid_image(self):
        """Тест розпізнавання з коректним зображенням"""
        strategy = EasyOCRStrategy()
        if not strategy.is_available():
            pytest.skip("EasyOCR недоступний")
        
        # Створюємо просте тестове зображення
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # EasyOCR може повернути порожній рядок для білого зображення
        result = strategy.recognize(test_image, 'eng')
        assert isinstance(result, str)


class TestPaddleOCRStrategy:
    """Тести для PaddleOCRStrategy"""
    
    def test_paddleocr_initialization(self):
        """Тест ініціалізації PaddleOCR стратегії"""
        strategy = PaddleOCRStrategy()
        assert isinstance(strategy, OCRStrategy)
        assert isinstance(strategy.is_available(), bool)
        assert strategy.get_name() in ['PaddleOCR', 'paddleocr']
    
    def test_paddleocr_recognize_with_invalid_image(self):
        """Тест розпізнавання з некоректним зображенням"""
        strategy = PaddleOCRStrategy()
        if not strategy.is_available():
            pytest.skip("PaddleOCR недоступний")
        
        # Порожнє зображення
        empty_image = np.array([], dtype=np.uint8)
        with pytest.raises((ValueError, RuntimeError, Exception)):
            strategy.recognize(empty_image, 'eng')
    
    def test_paddleocr_recognize_with_valid_image(self):
        """Тест розпізнавання з коректним зображенням"""
        strategy = PaddleOCRStrategy()
        if not strategy.is_available():
            pytest.skip("PaddleOCR недоступний")
        
        # Створюємо просте тестове зображення
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # PaddleOCR може повернути порожній рядок для білого зображення
        result = strategy.recognize(test_image, 'eng')
        assert isinstance(result, str)


class TestOCRStrategyInterface:
    """Тести для перевірки інтерфейсу стратегій"""
    
    @pytest.mark.parametrize("strategy_class", [
        TesseractStrategy,
        EasyOCRStrategy,
        PaddleOCRStrategy
    ])
    def test_strategy_interface(self, strategy_class):
        """Тест, що всі стратегії реалізують необхідний інтерфейс"""
        strategy = strategy_class()
        
        # Перевірка наявності методів
        assert hasattr(strategy, 'recognize')
        assert hasattr(strategy, 'is_available')
        assert hasattr(strategy, 'get_name')
        
        # Перевірка типів повернених значень
        assert isinstance(strategy.is_available(), bool)
        assert isinstance(strategy.get_name(), str)
        
        # Перевірка, що recognize приймає правильні параметри
        import inspect
        sig = inspect.signature(strategy.recognize)
        params = list(sig.parameters.keys())
        assert 'image' in params
        assert 'language' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

