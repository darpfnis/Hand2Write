"""
Панель налаштувань OCR з динамічними підказками
"""
from PyQt6.QtWidgets import (QComboBox, QLabel, QCheckBox, QProgressBar,
                             QVBoxLayout, QHBoxLayout, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from model.ocr_manager import OCREngine, OCRLanguage, ENGINE_METADATA, OCRManager, OCREngineMetadata

# Константи для назв мов
LANG_UKRAINIAN = "Українська"
LANG_ENGLISH = "Англійська"
LANG_BOTH = "Обидві"


class OCRSettingsPanel(QGroupBox):
    """Панель налаштувань OCR з динамічними підказками"""
    
    def __init__(self, parent=None):
        super().__init__("Налаштування розпізнавання", parent)
        self._engine_map = {}
        self.setVisible(True)  # Завжди видима
        self.init_ui()
    
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # === Вибір OCR рушія ===
        engine_layout = QHBoxLayout()
        self.engine_label = QLabel("OCR рушій:")
        engine_layout.addWidget(self.engine_label)
        
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(200)
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        engine_layout.addWidget(self.engine_combo)
        
        layout.addLayout(engine_layout)
        
        # === Підказка для рушія (прихована, інформація тільки в tooltip) ===
        self.engine_info_label = QLabel()
        self.engine_info_label.setWordWrap(True)
        self.engine_info_label.hide()  # Приховуємо блок, інформація тільки в tooltip
        
        # === Вибір мови ===
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Мова тексту:"))
        
        self.language_combo = QComboBox()
        self.language_combo.addItems([LANG_UKRAINIAN, LANG_ENGLISH, LANG_BOTH])
        self.language_combo.setMinimumWidth(150)
        self.language_combo.setToolTip(
            "Оберіть мову тексту для кращого розпізнавання:\n"
            "• Українська - для українського тексту\n"
            "• Англійська - для англійського тексту\n"
            "• Обидві - для змішаного тексту"
        )
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.language_combo)
        
        layout.addLayout(lang_layout)
        
        # === Автоматичний вибір найкращого рушія ===
        self.best_engine_check = QCheckBox("Автоматично вибирати найкращий OCR рушій за мовою")
        self.best_engine_check.setChecked(True)  # За замовчуванням увімкнено
        self.best_engine_check.setToolTip(
            "<b>Автоматичний вибір найкращого OCR рушія</b><br><br>"
            "При увімкненні система запускає всі доступні OCR рушії та автоматично вибирає найкращий результат "
            "на основі оцінки якості (fallback scoring).<br><br>"
            "<b>Як це працює:</b><br>"
            "• Запускаються всі доступні рушії (Tesseract, EasyOCR, PaddleOCR)<br>"
            "• Результати фільтруються та оцінюються за критеріями якості<br>"
            "• Автоматично вибирається найкращий результат<br><br>"
            "<b>Переваги:</b><br>"
            "• Вища точність розпізнавання<br>"
            "• Автоматичний вибір оптимального рушія<br>"
            "• Швидка робота (не потребує AI)<br><br>"
            "<b>Рекомендації:</b><br>"
            "• Рекомендується для рукописного тексту<br>"
            "• Може збільшити час обробки (запускаються всі рушії)"
        )
        self.best_engine_check.toggled.connect(self._on_best_engine_toggled)
        layout.addWidget(self.best_engine_check)
        # Встановлюємо початкову видимість вибору рушія
        self._on_best_engine_toggled(self.best_engine_check.isChecked())
        
        # Описовий текст для чекбоксу "найкращий рушій"
        best_engine_desc = QLabel(
            "Система запустить всі доступні OCR рушії та автоматично вибере найкращий результат "
            "на основі оцінки якості (наявність кирилиці, правильні слова, читабельність тощо)."
        )
        best_engine_desc.setWordWrap(True)
        best_engine_desc.setStyleSheet("color: #666; font-size: 9pt; padding-left: 20px;")
        layout.addWidget(best_engine_desc)
        
        # === AI корекція ===
        self.ai_correction_check = QCheckBox("Використовувати AI для виправлення помилок розпізнавання")
        self.ai_correction_check.setToolTip(
            "<b>AI корекція результатів OCR</b><br><br>"
            "Автоматично виправляє помилки розпізнавання через штучний інтелект.<br><br>"
            "<b>Як це працює:</b><br>"
            "• Після розпізнавання текст відправляється в AI модель<br>"
            "• Модель аналізує контекст та виправляє помилки<br>"
            "• Результат стає більш точним та читабельним<br><br>"
            "<b>Вимоги:</b><br>"
            "• Потребує налаштований Ollama або OpenAI API<br>"
            "• Може збільшити час обробки<br><br>"
            "<b>Рекомендації:</b><br>"
            "• Увімкніть для важкого рукописного тексту<br>"
            "• Вимкніть для швидшого розпізнавання<br>"
            "• Особливо корисна для української мови"
        )
        layout.addWidget(self.ai_correction_check)
        
        # Описовий текст для чекбоксу "AI корекція"
        ai_correction_desc = QLabel(
            "Після розпізнавання текст буде відправлено в AI модель (Ollama або OpenAI) "
            "для автоматичного виправлення помилок. Це покращує точність, особливо для "
            "рукописного тексту, але потребує налаштований AI сервер."
        )
        ai_correction_desc.setWordWrap(True)
        ai_correction_desc.setStyleSheet("color: #666; font-size: 9pt; padding-left: 20px;")
        layout.addWidget(ai_correction_desc)
        
        # === Прогрес-бар (спочатку прихований) ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # === Статус повідомлення ===
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 10pt;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)  # Підтримка HTML
        layout.addWidget(self.status_label)
    
    def load_available_engines(self, available_engines):
        """Завантаження доступних рушіїв
        
        Args:
            available_engines: список OCREngine enum або рядків
        """
        self.engine_combo.clear()
        
        # Мапінг для відображення
        engine_names = {
            OCREngine.TESSERACT: "Tesseract",
            OCREngine.EASYOCR: "EasyOCR",
            OCREngine.PADDLEOCR: "PaddleOCR"
        }
        
        # Мапінг рядків до enum (для сумісності)
        string_to_enum = {
            "tesseract": OCREngine.TESSERACT,
            "easyocr": OCREngine.EASYOCR,
            "paddleocr": OCREngine.PADDLEOCR
        }
        
        # Зберігаємо мапінг для зворотного пошуку
        self._engine_map = {}
        
        for engine in available_engines:
            # Обробляємо як enum, так і рядки
            if isinstance(engine, OCREngine):
                engine_enum = engine
            elif isinstance(engine, str):
                engine_enum = string_to_enum.get(engine, OCREngine.TESSERACT)
            else:
                continue
            
            display_name = engine_names.get(engine_enum, engine_enum.value.capitalize())
            self.engine_combo.addItem(display_name)
            self._engine_map[display_name] = engine_enum
            
            # Додаємо tooltip для кожного елемента
            metadata = ENGINE_METADATA.get(engine_enum)
            if metadata:
                tooltip = self._get_engine_tooltip(engine_enum, metadata)
                # Встановлюємо tooltip для конкретного індексу
                index = self.engine_combo.count() - 1
                self.engine_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
        
        # Встановлюємо дефолтний вибір та tooltip для комбобоксу
        if available_engines:
            self.engine_combo.setCurrentIndex(0)
            self._on_engine_changed(self.engine_combo.currentText())
            
            # Додаємо обробник для показу tooltip при наведенні
            view = self.engine_combo.view()
            if view is not None:
                view.setMouseTracking(True)
                view.entered.connect(self._on_item_hovered)
    
    def _on_engine_changed(self, text: str):
        """Обробник зміни рушія"""
        if text not in self._engine_map:
            return
        
        engine = self._engine_map[text]
        metadata = ENGINE_METADATA.get(engine)
        
        if metadata:
            # Оновлюємо tooltip для комбобоксу з детальною інформацією
            detailed_tooltip = self._get_engine_tooltip(engine, metadata)
            self.engine_combo.setToolTip(detailed_tooltip)
    
    def _on_item_hovered(self, index):
        """Обробник наведення на елемент комбобоксу"""
        # index може бути QModelIndex, отримуємо рядок
        if hasattr(index, 'row'):
            row = index.row()
        else:
            row = index
        
        if 0 <= row < self.engine_combo.count():
            tooltip = self.engine_combo.itemData(row, Qt.ItemDataRole.ToolTipRole)
            if tooltip:
                # Tooltip автоматично показується через setItemData
                pass
    
    def _on_language_changed(self, language: str):
        """Обробник зміни мови"""
        pass
    
    def _get_engine_tooltip(self, engine: OCREngine, metadata: OCREngineMetadata) -> str:
        """Створення детального tooltip для рушія"""
        tooltips = {
            OCREngine.TESSERACT: (
                "<b>Ефективність Tesseract OCR:</b><br><br>"
                "<b>Швидкість:</b> Дуже висока (найшвидший з доступних)<br>"
                "<b>Точність:</b><br>"
                "   • Друкований текст: 85-95%<br>"
                "   • Рукописний текст: 60-75%<br>"
                "   • Українська мова: Відмінна підтримка<br><br>"
                "<b>Рекомендації:</b><br>"
                "   • Найкраще для друкованого тексту<br>"
                "   • Швидке розпізнавання<br>"
                "   • Підтримка багатьох мов<br>"
                "   • Менше ресурсів для обробки"
            ),
            OCREngine.EASYOCR: (
                "<b>Ефективність EasyOCR:</b><br><br>"
                "<b>Швидкість:</b> Середня (залежить від GPU)<br>"
                "<b>Точність:</b><br>"
                "   • Друкований текст: 80-90%<br>"
                "   • Рукописний текст: 70-85%<br>"
                "   • Українська мова: Добра підтримка<br><br>"
                "<b>Рекомендації:</b><br>"
                "   • Найкраще для рукописного тексту<br>"
                "   • Універсальність (різні типи тексту)<br>"
                "   • Підтримка багатомовності<br>"
                "   • Працює на CPU та GPU"
            ),
            OCREngine.PADDLEOCR: (
                "<b>Ефективність PaddleOCR:</b><br><br>"
                "<b>Швидкість:</b> Середня (оптимізовано для CPU)<br>"
                "<b>Точність:</b><br>"
                "   • Друкований текст: 90-95%<br>"
                "   • Рукописний текст: 75-85%<br>"
                "   • Українська мова: Обмежена (краще для англійської)<br><br>"
                "<b>Рекомендації:</b><br>"
                "   • Найкраще для англійської мови<br>"
                "   • Баланс швидкості та точності<br>"
                "   • Оптимізовано для CPU<br>"
                "   • Добре для складних макетів"
            )
        }
        return tooltips.get(engine, f"<b>{metadata.name}</b><br>{metadata.description}")
    
    def get_selected_engine(self) -> OCREngine:
        """Отримання вибраного рушія"""
        # Якщо увімкнено "найкращий рушій", повертаємо поточний вибір
        # (він вже автоматично встановлений як найкращий)
        text = self.engine_combo.currentText()
        return self._engine_map.get(text, OCREngine.TESSERACT)
    
    def is_best_engine_enabled(self) -> bool:
        """Чи увімкнено автоматичний вибір найкращого рушія"""
        return self.best_engine_check.isChecked()
    
    def _on_best_engine_toggled(self, checked: bool):
        """Обробник зміни стану чекбокса 'найкращий рушій'"""
        # Якщо увімкнено автоматичний вибір, можна приховати вибір рушія
        # Якщо вимкнено - показати вибір рушія
        self.set_engine_selection_visible(not checked)
    
    def get_selected_language(self) -> OCRLanguage:
        """Отримання вибраної мови"""
        lang_map = {
            LANG_UKRAINIAN: OCRLanguage.UKRAINIAN,
            LANG_ENGLISH: OCRLanguage.ENGLISH,
            LANG_BOTH: OCRLanguage.BOTH
        }
        return lang_map.get(self.language_combo.currentText(), OCRLanguage.UKRAINIAN)
    
    def get_selected_language_string(self) -> str:
        """Отримання вибраної мови як рядок (для сумісності)"""
        lang_map = {
            LANG_UKRAINIAN: LANG_UKRAINIAN,
            LANG_ENGLISH: LANG_ENGLISH,
            LANG_BOTH: LANG_ENGLISH  # Для "Обидві" використовуємо англійську як fallback
        }
        return lang_map.get(self.language_combo.currentText(), LANG_UKRAINIAN)
    
    def is_ai_correction_enabled(self) -> bool:
        """Чи увімкнена AI корекція"""
        return self.ai_correction_check.isChecked()
    
    def show_progress(self, value: int, message: str = ""):
        """Показати прогрес"""
        self.progress_bar.show()
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
    
    def hide_progress(self):
        """Приховати прогрес"""
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        # Не очищаємо статус-лейбл, якщо там показано використаний рушій
        # (він буде очищений тільки при наступному розпізнаванні або через hide_used_engine)
    
    def set_engine_selection_visible(self, visible: bool):
        """Показати/приховати вибір рушія"""
        self.engine_combo.setVisible(visible)
        self.engine_label.setVisible(visible)
    
    def show_used_engine(self, engine_name: str):
        """Показати який рушій виконав розпізнавання"""
        # Використовуємо status_label для показу використаного рушія
        self.status_label.setText(f"✓ Розпізнавання виконано рушієм: <b>{engine_name}</b>")
        self.status_label.setStyleSheet("color: #2196F3; font-size: 10pt; font-weight: bold;")
    
    def hide_used_engine(self):
        """Приховати інформацію про використаний рушій"""
        # Очищаємо статус-лейбл
        self.status_label.clear()
        self.status_label.setStyleSheet("color: #666; font-size: 10pt;")

