"""
Тести для валідації вхідних даних
handwrite2print/tests/test_validation.py
"""
import pytest
import sys
import os
import numpy as np

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.optimized_preprocessor import OptimizedPreprocessor
from app.handwrite_config import Config


class TestImageValidation:
    """Тести для валідації зображень"""
    
    def test_preprocessor_validates_none_image(self):
        """Тест валідації None зображення"""
        preprocessor = OptimizedPreprocessor()
        
        with pytest.raises(ValueError, match="не може бути None"):
            preprocessor.process(None)  # type: ignore[arg-type]
    
    def test_preprocessor_validates_empty_image(self):
        """Тест валідації порожнього зображення"""
        preprocessor = OptimizedPreprocessor()
        
        empty_image = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError, match="порожнє"):
            preprocessor.process(empty_image)
    
    def test_preprocessor_validates_invalid_dimensions(self):
        """Тест валідації некоректної розмірності"""
        preprocessor = OptimizedPreprocessor()
        
        # Одновимірний масив
        invalid_image = np.array([1, 2, 3], dtype=np.uint8)
        with pytest.raises(ValueError, match="Невірна розмірність"):
            preprocessor.process(invalid_image)
    
    def test_preprocessor_processes_valid_image(self):
        """Тест обробки коректного зображення"""
        preprocessor = OptimizedPreprocessor()
        
        # Валідне зображення
        rng = np.random.default_rng(7)
        valid_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocessor.process(valid_image)
        
        assert result is not None
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8


class TestConfigValidation:
    """Тести для валідації конфігурації"""
    
    def test_image_format_validation(self):
        """Тест валідації формату зображення"""
        # Валідні формати
        assert Config.is_valid_image_format("test.jpg")
        assert Config.is_valid_image_format("test.png")
        assert Config.is_valid_image_format("test.jpeg")
        assert Config.is_valid_image_format("test.bmp")
        assert Config.is_valid_image_format("test.tiff")
        
        # Невалідні формати
        assert not Config.is_valid_image_format("test.gif")
        assert not Config.is_valid_image_format("test.pdf")
        assert not Config.is_valid_image_format("test.txt")
    
    def test_export_format_validation(self):
        """Тест валідації формату експорту"""
        # Валідні формати
        assert Config.is_valid_export_format("test.txt")
        assert Config.is_valid_export_format("test.docx")
        assert Config.is_valid_export_format("test.pdf")
        assert Config.is_valid_export_format("test.html")
        
        # Невалідні формати
        assert not Config.is_valid_export_format("test.jpg")
        assert not Config.is_valid_export_format("test.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

