"""
Базові тести для програми
handwrite2print/tests/test_basic.py
"""
import pytest
import sys
import os

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.handwrite_preprocess import ImagePreprocessor
from app.model.handwrite_export import TextExporter
import numpy as np


class TestImagePreprocessor:
    """Тести для препроцесора зображень"""
    
    def test_denoise(self):
        """Тест видалення шумів"""
        preprocessor = ImagePreprocessor()
        
        # Створення тестового зображення з шумом
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        result = preprocessor.denoise(image)
        
        assert result is not None
        assert result.shape == image.shape
        
    def test_binarize(self):
        """Тест бінаризації"""
        preprocessor = ImagePreprocessor()
        
        # Створення тестового зображення
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        result = preprocessor.binarize(image)
        
        assert result is not None
        assert result.shape == image.shape
        # Перевірка, що всі значення 0 або 255
        assert np.all((result == 0) | (result == 255))
        
    def test_resize_image(self):
        """Тест зміни розміру"""
        preprocessor = ImagePreprocessor()
        
        # Велике зображення
        image = np.random.randint(0, 255, (3000, 3000), dtype=np.uint8)
        
        result = preprocessor.resize_image(image, max_width=2000, max_height=2000)
        
        assert result is not None
        assert result.shape[0] <= 2000
        assert result.shape[1] <= 2000


class TestTextExporter:
    """Тести для експортера"""
    
    def test_export_txt(self, tmp_path):
        """Тест експорту в TXT"""
        exporter = TextExporter()
        
        test_text = "Тестовий текст\nДругий рядок"
        file_path = tmp_path / "test.txt"
        
        result = exporter.export_txt(test_text, str(file_path))
        
        assert result is True
        assert file_path.exists()
        
        # Перевірка вмісту
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == test_text
        
    def test_export_html(self, tmp_path):
        """Тест експорту в HTML"""
        exporter = TextExporter()
        
        test_text = "Тестовий <текст>"
        file_path = tmp_path / "test.html"
        
        result = exporter.export_html(test_text, str(file_path))
        
        assert result is True
        assert file_path.exists()
        
        # Перевірка, що HTML екрановано
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '&lt;' in content
        assert '&gt;' in content


def test_imports():
    """Тест імпорту основних модулів"""
    try:
        from app.model.handwrite_preprocess import ImagePreprocessor
        from app.model.handwrite_export import TextExporter
        from app.model.unified_ocr_adapter import UnifiedOCRAdapter
        assert True
    except ImportError as e:
        pytest.fail(f"Помилка імпорту: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
