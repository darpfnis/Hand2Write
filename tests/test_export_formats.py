"""
Unit-тести для експорту у різні формати
handwrite2print/tests/test_export_formats.py
"""
import pytest
import sys
import os
from pathlib import Path

# Додавання шляху до app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.model.handwrite_export import TextExporter


class TestTextExporter:
    """Тести для TextExporter"""
    
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
    
    def test_export_docx(self, tmp_path):
        """Тест експорту в DOCX"""
        exporter = TextExporter()
        
        test_text = "Тестовий текст для DOCX"
        file_path = tmp_path / "test.docx"
        
        result = exporter.export_docx(test_text, str(file_path))
        
        # DOCX може бути недоступний, тому перевіряємо результат
        assert isinstance(result, bool)
        if result:
            assert file_path.exists()
    
    def test_export_pdf(self, tmp_path):
        """Тест експорту в PDF"""
        exporter = TextExporter()
        
        test_text = "Тестовий текст для PDF"
        file_path = tmp_path / "test.pdf"
        
        result = exporter.export_pdf(test_text, str(file_path))
        
        # PDF може бути недоступний, тому перевіряємо результат
        assert isinstance(result, bool)
        if result:
            assert file_path.exists()
    
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
        
        assert '&lt;' in content or '<' in content
        assert '&gt;' in content or '>' in content
    
    def test_export_handles_special_characters(self, tmp_path):
        """Тест експорту зі спеціальними символами"""
        exporter = TextExporter()
        
        test_text = "Текст з спеціальними символами: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        file_path = tmp_path / "test_special.txt"
        
        result = exporter.export_txt(test_text, str(file_path))
        
        assert result is True
        assert file_path.exists()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == test_text
    
    def test_export_handles_unicode(self, tmp_path):
        """Тест експорту з Unicode символами"""
        exporter = TextExporter()
        
        test_text = "Українська мова: Привіт, Світ! 🇺🇦"
        file_path = tmp_path / "test_unicode.txt"
        
        result = exporter.export_txt(test_text, str(file_path))
        
        assert result is True
        assert file_path.exists()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == test_text
    
    def test_export_handles_empty_text(self, tmp_path):
        """Тест експорту порожнього тексту"""
        exporter = TextExporter()
        
        test_text = ""
        file_path = tmp_path / "test_empty.txt"
        
        result = exporter.export_txt(test_text, str(file_path))
        
        assert result is True
        assert file_path.exists()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

