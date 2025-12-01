"""
Експорт розпізнаного тексту у різні формати
handwrite2print/app/model/export.py
"""
import os
from datetime import datetime


class TextExporter:
    """Клас для експорту тексту у різні формати"""
    
    def __init__(self):
        """Ініціалізація експортера.
        
        Метод порожній, оскільки клас не потребує початкової конфігурації.
        Всі необхідні налаштування виконуються в методах експорту.
        """
        pass
        
    def export_txt(self, text, file_path):
        """
        Експорт у текстовий файл
        
        Args:
            text: текст для експорту
            file_path: шлях до файлу
            
        Returns:
            True при успіху, False при помилці
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"Помилка при збереженні TXT: {e}")
            return False
            
    def export_docx(self, text, file_path):
        """
        Експорт у Word документ
        
        Args:
            text: текст для експорту
            file_path: шлях до файлу
            
        Returns:
            True при успіху, False при помилці
        """
        try:
            from docx import Document
            from docx.shared import Pt
            
            doc = Document()
            
            # Додавання заголовку
            doc.add_heading('Розпізнаний текст', level=1)
            
            # Додавання дати
            date_paragraph = doc.add_paragraph()
            date_paragraph.add_run(
                f'Дата створення: {datetime.now().strftime("%d.%m.%Y %H:%M")}'
            ).italic = True
            
            # Додавання тексту
            for line in text.split('\n'):
                if line.strip():
                    p = doc.add_paragraph(line)
                    # Налаштування шрифту
                    for run in p.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(12)
                else:
                    doc.add_paragraph()  # Порожній рядок
                    
            # Збереження
            doc.save(file_path)
            return True
            
        except ImportError:
            print("Модуль python-docx не встановлено. Встановіть: pip install python-docx")
            return False
        except Exception as e:
            print(f"Помилка при збереженні DOCX: {e}")
            return False
            
    def export_pdf(self, text, file_path):
        """
        Експорт у PDF файл
        
        Args:
            text: текст для експорту
            file_path: шлях до файлу
            
        Returns:
            True при успіху, False при помилці
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            from reportlab.lib.colors import HexColor
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # Створення PDF
            doc = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            story = []
            styles = getSampleStyleSheet()
            
            # Реєстрація шрифту, який підтримує кирилицю
            # Використовуємо системні шрифти або вбудовані шрифти ReportLab
            font_name = 'Helvetica'  # За замовчуванням
            try:
                # Спробуємо знайти системний шрифт, який підтримує кирилицю
                import platform
                system = platform.system()

                if system == 'Windows':
                    # Windows має шрифти, які підтримують кирилицю
                    # Спробуємо використати Arial або Times New Roman
                    try:
                        # Шукаємо шрифт у стандартних місцях Windows
                        font_paths = [
                            'C:/Windows/Fonts/arial.ttf',
                            'C:/Windows/Fonts/times.ttf',
                            'C:/Windows/Fonts/timesnr.ttf',
                        ]
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                pdfmetrics.registerFont(TTFont('CyrillicFont', font_path))
                                font_name = 'CyrillicFont'
                                break
                    except Exception:
                        pass
                elif system == 'Linux':
                    # Linux - спробуємо DejaVu Sans
                    try:
                        font_paths = [
                            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                        ]
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                pdfmetrics.registerFont(TTFont('CyrillicFont', font_path))
                                font_name = 'CyrillicFont'
                                break
                    except Exception:
                        pass
                elif system == 'Darwin':  # macOS
                    try:
                        font_paths = [
                            '/System/Library/Fonts/Supplemental/Arial.ttf',
                            '/Library/Fonts/Arial.ttf',
                        ]
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                pdfmetrics.registerFont(TTFont('CyrillicFont', font_path))
                                font_name = 'CyrillicFont'
                                break
                    except Exception:
                        pass
            except Exception as e:
                print(f"Не вдалося зареєструвати шрифт для кирилиці: {e}")
                # Використовуємо стандартний шрифт, але спробуємо обійти проблему
                
            # Стиль заголовку
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=HexColor("#2196F3"),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName=font_name
            )
            
            # Стиль дати
            date_style = ParagraphStyle(
                'DateStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=HexColor("#000000"),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName=font_name
            )
            
            # Стиль основного тексту
            body_style = ParagraphStyle(
                'BodyStyle',
                parent=styles['Normal'],
                fontSize=12,
                leading=16,
                alignment=TA_LEFT,
                fontName=font_name
            )
            
            # Додавання заголовку
            story.append(Paragraph("Розпізнаний текст", title_style))
            
            # Додавання дати
            date_text = f"Дата створення: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            story.append(Paragraph(date_text, date_style))
            
            story.append(Spacer(1, 0.5*cm))
            
            # Додавання тексту
            for line in text.split('\n'):
                if line.strip():
                    # Екранування спеціальних символів для ReportLab
                    # ReportLab вимагає екранування XML-символів
                    safe_line = (
                        line.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#39;')
                    )
                    try:
                        story.append(Paragraph(safe_line, body_style))
                        story.append(Spacer(1, 0.2 * cm))
                    except Exception as e:
                        # Якщо не вдалося додати параграф (наприклад, через проблеми з кодуванням),
                        # спробуємо додати як простий текст
                        print(f"Помилка при додаванні рядка в PDF: {e}")
                        # Додаємо як простий текст без форматування
                        from reportlab.platypus import Preformatted

                        story.append(
                            Preformatted(line, body_style, maxLineLength=80)
                        )
                        story.append(Spacer(1, 0.2 * cm))
                else:
                    story.append(Spacer(1, 0.3 * cm))
                    
            # Побудова PDF
            doc.build(story)
            return True
            
        except ImportError:
            print("Модуль reportlab не встановлено. Встановіть: pip install reportlab")
            return False
        except Exception as e:
            print(f"Помилка при збереженні PDF: {e}")
            return False
            
    def export_html(self, text, file_path):
        """
        Експорт у HTML файл
        
        Args:
            text: текст для експорту
            file_path: шлях до файлу
            
        Returns:
            True при успіху, False при помилці
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Розпізнаний текст</title>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2196F3;
            text-align: center;
        }}
        .date {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-bottom: 30px;
        }}
        .content {{
            text-align: justify;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>Розпізнаний текст</h1>
    <p class="date">Дата створення: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
    <div class="content">{self._escape_html(text)}</div>
</body>
</html>
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return True
            
        except Exception as e:
            print(f"Помилка при збереженні HTML: {e}")
            return False
            
    def _escape_html(self, text):
        """Екранування HTML спецсимволів"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
