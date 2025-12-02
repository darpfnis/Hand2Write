"""
Допоміжні діалогові вікна
handwrite2print/app/view/dialogs.py
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QGroupBox, QCheckBox,
                             QSpinBox, QComboBox, QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt

# Стиль для QMessageBox з білим текстом
MESSAGE_BOX_STYLE = """
    QMessageBox {
        background-color: #2c3e50;
        color: white;
    }
    QMessageBox QLabel {
        color: white;
        background-color: transparent;
    }
    QMessageBox QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 4px;
    }
    QMessageBox QPushButton:hover {
        background-color: #2980b9;
    }
    QMessageBox QPushButton:pressed {
        background-color: #21618c;
    }
"""


def _create_styled_message_box(parent, icon, title, text, buttons=None):
    """Створення QMessageBox з білим текстом"""
    msg = QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStyleSheet(MESSAGE_BOX_STYLE)
    if buttons:
        msg.setStandardButtons(buttons)
    return msg


class SettingsDialog(QDialog):
    """Діалог налаштувань"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Налаштування OCR")
        self.setMinimumWidth(600)
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout(self)
        
        # Налаштування попередньої обробки
        preprocess_group = QGroupBox("Попередня обробка")
        preprocess_layout = QVBoxLayout()
        
        self.binarization_check = QCheckBox("Бінаризація зображення")
        self.binarization_check.setChecked(True)
        preprocess_layout.addWidget(self.binarization_check)
        
        self.denoise_check = QCheckBox("Видалення шумів")
        self.denoise_check.setChecked(True)
        preprocess_layout.addWidget(self.denoise_check)
        
        self.deskew_check = QCheckBox("Вирівнювання тексту")
        self.deskew_check.setChecked(True)
        preprocess_layout.addWidget(self.deskew_check)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Налаштування розпізнавання
        ocr_group = QGroupBox("Параметри розпізнавання")
        ocr_layout = QVBoxLayout()
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Мінімальна впевненість (%):"))
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(60)
        confidence_layout.addWidget(self.confidence_spin)
        ocr_layout.addLayout(confidence_layout)
        
        self.spellcheck_check = QCheckBox("Перевірка орфографії")
        self.spellcheck_check.setChecked(False)
        ocr_layout.addWidget(self.spellcheck_check)
        
        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)
        
        # Налаштування LLM корекції
        llm_group = QGroupBox("ШІ корекція результатів (покращує точність)")
        llm_layout = QVBoxLayout()
        
        self.llm_enabled_check = QCheckBox("Використовувати ШІ для виправлення помилок OCR")
        self.llm_enabled_check.setToolTip("Автоматично виправляє помилки OCR через штучний інтелект для підвищення точності")
        llm_layout.addWidget(self.llm_enabled_check)
        
        # Тип API
        api_type_layout = QHBoxLayout()
        api_type_layout.addWidget(QLabel("Тип API:"))
        self.api_type_combo = QComboBox()
        self.api_type_combo.addItems(["Ollama", "Local"])
        self.api_type_combo.currentTextChanged.connect(self.on_api_type_changed)
        api_type_layout.addWidget(self.api_type_combo)
        llm_layout.addLayout(api_type_layout)
        
        # API URL (для Ollama/Local)
        self.api_url_layout = QHBoxLayout()
        self.api_url_layout.addWidget(QLabel("API URL:"))
        self.api_url_edit = QLineEdit()
        self.api_url_edit.setPlaceholderText("http://localhost:11434")
        self.api_url_layout.addWidget(self.api_url_edit)
        llm_layout.addLayout(self.api_url_layout)
        
        # Модель (для Ollama/Local)
        self.model_layout = QHBoxLayout()
        self.model_layout.addWidget(QLabel("Модель:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("llama3.2:latest")
        self.model_edit.setToolTip(
            "Назва моделі:\n"
            "• Для Ollama/Local: llama3.2:1b, llama3.2:latest\n"
            "• Залиште порожнім для автоматичного вибору"
        )
        self.model_layout.addWidget(self.model_edit)
        llm_layout.addLayout(self.model_layout)
        
        # Статус доступності
        self.llm_status_label = QLabel("Статус: не перевірено")
        self.llm_status_label.setStyleSheet("color: gray;")
        llm_layout.addWidget(self.llm_status_label)
        
        # Кнопка перевірки
        test_btn = QPushButton("Перевірити доступність")
        test_btn.clicked.connect(self.test_llm_connection)
        llm_layout.addWidget(test_btn)
        
        # Інформаційна кнопка
        info_btn = QPushButton("ℹ️ Як отримати ШІ?")
        info_btn.setToolTip("Відкрити інструкцію з налаштування ШІ")
        info_btn.clicked.connect(self.show_llm_info)
        llm_layout.addWidget(info_btn)
        
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        
        # Оновлюємо видимість полів
        self.on_api_type_changed()
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Скасувати")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
    def load_config(self):
        """Завантаження конфігурації"""
        try:
            from model.ocr_config import OCRConfig
            config = OCRConfig()
            llm_config = config.get_llm_config()
            
            self.llm_enabled_check.setChecked(llm_config.get('enabled', False))
            
            api_type = llm_config.get('api_type', 'local').capitalize()
            if api_type == 'Local':
                api_type = 'Local'
            index = self.api_type_combo.findText(api_type)
            if index >= 0:
                self.api_type_combo.setCurrentIndex(index)
            
            # API key більше не використовується (тільки Ollama)
            self.api_url_edit.setText(llm_config.get('api_url', 'http://localhost:11434'))
            self.model_edit.setText(llm_config.get('model', ''))
            
            # Перевірка статусу
            if llm_config.get('enabled'):
                self.llm_status_label.setText("Статус: увімкнено")
                self.llm_status_label.setStyleSheet("color: green;")
            else:
                self.llm_status_label.setText("Статус: вимкнено")
                self.llm_status_label.setStyleSheet("color: gray;")
        except Exception as e:
            print(f"Помилка завантаження конфігурації: {e}")
    
    def on_api_type_changed(self):
        """Оновлення видимості полів при зміні типу API"""
        api_type = self.api_type_combo.currentText().lower()
        
        # Показуємо/ховаємо поля залежно від типу API
        for i in range(self.api_url_layout.count()):
            item = self.api_url_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(api_type in ['ollama', 'local'])
    
    def test_llm_connection(self):
        """Перевірка з'єднання з LLM"""
        try:
            from model.llm_postprocessor import LLMPostProcessor
            
            api_type = self.api_type_combo.currentText().lower()
            api_key = None
            api_url = self.api_url_edit.text() if api_type in ['ollama', 'local'] else None
            
            processor = LLMPostProcessor(
                api_type=api_type,
                api_key=api_key,
                api_url=api_url
            )
            
            if processor.is_available():
                self.llm_status_label.setText("Статус: ✓ Доступний")
                self.llm_status_label.setStyleSheet("color: green;")
                msg = _create_styled_message_box(self, QMessageBox.Icon.Information, "Успіх", "LLM API доступний та готовий до використання!")
                msg.exec()
            else:
                self.llm_status_label.setText("Статус: ✗ Недоступний")
                self.llm_status_label.setStyleSheet("color: red;")
                msg = _create_styled_message_box(self, QMessageBox.Icon.Warning, "Помилка", "LLM API недоступний. Перевірте налаштування.")
                msg.exec()
        except Exception as e:
            self.llm_status_label.setText(f"Статус: ✗ Помилка ({str(e)[:30]})")
            self.llm_status_label.setStyleSheet("color: red;")
            msg = _create_styled_message_box(self, QMessageBox.Icon.Critical, "Помилка", f"Помилка перевірки: {e}")
            msg.exec()
    
    def show_llm_info(self):
        """Показати інформацію про отримання ШІ"""
        from pathlib import Path
        
        info_text = """
<h3>Як отримати ШІ для корекції OCR?</h3>

<p><b>Ollama (безкоштовний, локальний)</b></p>
<ul>
<li>Завантажте з <a href='https://ollama.ai'>ollama.ai</a></li>
<li>Встановіть та запустіть: <code>ollama serve</code></li>
<li>Завантажте модель: <code>ollama pull llama3.2:1b</code></li>
<li>Повністю безкоштовно, працює офлайн</li>
<li>Вимоги: мінімум 8GB RAM</li>
</ul>

<p><b>Детальна інструкція:</b></p>
<p>Перегляньте файл <b>LLM_SETUP_GUIDE.md</b> у корені проекту</p>
"""
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Як отримати ШІ?")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(info_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Перевірка, чи існує файл з інструкцією
        guide_path = Path(__file__).parent.parent.parent / "LLM_SETUP_GUIDE.md"
        if guide_path.exists():
            msg.setInformativeText(f"Повна інструкція знаходиться тут:\n{guide_path}")
        
        msg.exec()
        
    def get_settings(self):
        """Отримання налаштувань"""
        api_type = self.api_type_combo.currentText().lower()
        api_key = None
        api_url = self.api_url_edit.text() if api_type in ['ollama', 'local'] else 'http://localhost:11434'
        
        model = self.model_edit.text().strip() or None
        
        llm_config = {
            'enabled': self.llm_enabled_check.isChecked(),
            'api_type': api_type,
            'api_key': api_key,
            'api_url': api_url,
            'model': model
        }
        
        # Зберігаємо конфігурацію
        try:
            from model.ocr_config import OCRConfig
            config = OCRConfig()
            config.set_llm_config(llm_config)
        except Exception as e:
            print(f"Помилка збереження конфігурації: {e}")
        
        return {
            'binarization': self.binarization_check.isChecked(),
            'denoise': self.denoise_check.isChecked(),
            'deskew': self.deskew_check.isChecked(),
            'confidence': self.confidence_spin.value(),
            'spellcheck': self.spellcheck_check.isChecked(),
            'llm': llm_config
        }


class AboutDialog(QDialog):
    """Діалог 'Про програму'"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Про програму")
        self.setFixedSize(450, 350)
        self.init_ui()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Система перетворення рукописного\nтексту в друкований")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Версія
        version = QLabel("Версія 1.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)
        
        layout.addSpacing(20)
        
        # Опис
        description = QLabel(
            "Desktop-застосунок для автоматичного розпізнавання\n"
            "та перетворення рукописного тексту в цифровий формат.\n\n"
            "Підтримує:\n"
            "• Завантаження зображень (JPG, PNG)\n"
            "• Ручне введення тексту\n"
            "• Розпізнавання українською та англійською мовами\n"
            "• Експорт у формати TXT, DOCX, PDF\n"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        
        layout.addSpacing(20)
        
        # Технології
        tech = QLabel(
            "Технології:\n"
            "Python 3.8+, PyQt6, OpenCV, Tesseract OCR,\n"
            "EasyOCR, PaddleOCR"
        )
        tech.setStyleSheet("font-size: 11px; color: #666;")
        tech.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tech)
        
        layout.addSpacing(20)
        
        # Авторство
        author = QLabel("КПІ ім. Ігоря Сікорського\nКурсова робота, 2025")
        author.setStyleSheet("font-size: 11px; color: #666;")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author)
        
        layout.addStretch()
        
        # Кнопка закриття
        close_btn = QPushButton("Закрити")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class ProgressDialog(QDialog):
    """Діалог прогресу"""
    
    def __init__(self, parent=None, title="Обробка..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 150)
        self.init_ui()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Виконується обробка зображення...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        from PyQt6.QtWidgets import QProgressBar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Невизначений прогрес
        layout.addWidget(self.progress)
        
    def set_message(self, message):
        """Встановлення повідомлення"""
        self.label.setText(message)
