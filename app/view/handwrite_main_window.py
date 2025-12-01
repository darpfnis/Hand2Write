"""
Головне вікно програми
handwrite2print/app/view/main_window.py
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QFileDialog, 
                             QMessageBox, QToolBar, QStatusBar, QProgressBar,
                             QComboBox, QGroupBox, QSplitter, QMenu, QApplication,
                             QStackedWidget, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QFont
import os
import sys
import logging

# Додаємо шлях до app для правильних імпортів
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view.handwrite_canvas import CanvasWidget
from view.handwrite_dialogs import SettingsDialog, AboutDialog
from view.welcome_screen import WelcomeScreen
from view.handwrite_toolbar import HandwriteToolBar
from view.ocr_settings_panel import OCRSettingsPanel
from handwrite_controller import Controller
from improved_controller import ImprovedController

logger = logging.getLogger(__name__)

# Константи для назв мов
LANG_UKRAINIAN = "Українська"
LANG_ENGLISH = "Англійська"
LANG_BOTH = "Обидві"

# Константи для повідомлень
MSG_ERROR = "Помилка"
MSG_INFO = "Інформація"
MSG_SAVE_RESULT = "Зберегти результат"
MSG_RECOGNIZED_TEXT = "розпізнаний текст"

# Константа для стилю CSS
TITLE_STYLE = "color: #2c3e50; padding: 5px;"


class MainWindow(QMainWindow):
    """Головне вікно програми"""
    
    # Інформація про OCR рушії для різних мов
    OCR_ENGINE_INFO = {
        LANG_UKRAINIAN: {
            "Tesseract": {
                "description": "Найкраща підтримка української мови. Швидкий та точний для рукописного тексту.",
                "accuracy": 85
            },
            "EasyOCR": {
                "description": "Добра підтримка української мови. Може працювати повільніше, але дає точні результати для чіткого тексту.",
                "accuracy": 75
            },
            "PaddleOCR": {
                "description": "Не рекомендується для української мови. Може давати неточні результати.",
                "accuracy": 40
            }
        },
        LANG_ENGLISH: {
            "Tesseract": {
                "description": "Надійний та швидкий рушій для англійської мови. Добра точність для рукописного тексту.",
                "accuracy": 80
            },
            "EasyOCR": {
                "description": "Хороша підтримка англійської мови. Може працювати повільніше, особливо для рукописного тексту.",
                "accuracy": 70
            },
            "PaddleOCR": {
                "description": "Відмінна підтримка англійської мови. Найвища точність для друкованого та рукописного тексту.",
                "accuracy": 95
            }
        }
    }
    
    def __init__(self):
        super().__init__()
        # Використовуємо покращений контролер
        self.controller = ImprovedController(self)
        self.current_image_path = None
        self._temp_image_path = None
        self.current_mode = None  # 'draw' або 'upload'
        self.current_language = LANG_UKRAINIAN  # Поточна мова для tooltip
        
        # Створюємо окремі панелі налаштувань OCR для кожного режиму
        # (один віджет не може бути в двох місцях одночасно)
        self.ocr_settings_draw = OCRSettingsPanel()
        self.ocr_settings_upload = OCRSettingsPanel()
        
        # Завантажуємо доступні рушії
        available_engines = self.controller.get_available_engines()
        # Передаємо enum безпосередньо (панель підтримує обидва формати)
        self.ocr_settings_draw.load_available_engines(available_engines)
        self.ocr_settings_upload.load_available_engines(available_engines)
        
        # Синхронізуємо вибір між панелями
        self._sync_ocr_settings()
        
        self.init_ui()
        
    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        self.setWindowTitle("Система розпізнавання рукописного тексту")
        # Встановлюємо мінімальний розмір для уникнення помилок геометрії
        self.setMinimumSize(1200, 800)
        # Показуємо вікно в повноекранному режимі
        self.showMaximized()
        
        # Створення меню
        self.create_menu()
        
        # Створення toolbar (спочатку прихований)
        self.toolbar = HandwriteToolBar(self)
        self.addToolBar(self.toolbar)
        self.toolbar.hide_toolbar()  # Приховуємо на головному меню
        
        # Підключення сигналів toolbar
        self.toolbar.back_btn.clicked.connect(self.show_welcome_screen)
        self.toolbar.load_btn.clicked.connect(self.load_image)
        self.toolbar.convert_btn.clicked.connect(self.convert_text)
        self.toolbar.export_btn.clicked.connect(self.export_text)
        self.toolbar.clear_btn.clicked.connect(self.clear_all)
        
        # Центральний віджет зі стеком
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Стек віджетів для перемикання між меню та основним інтерфейсом
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Початкове меню
        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.draw_mode_selected.connect(self.show_draw_mode)
        self.welcome_screen.upload_mode_selected.connect(self.show_upload_mode)
        self.stacked_widget.addWidget(self.welcome_screen)
        
        # Окремі інтерфейси для кожного режиму
        self.draw_widget = self.create_draw_interface()
        self.stacked_widget.addWidget(self.draw_widget)
        
        self.upload_widget = self.create_upload_interface()
        self.stacked_widget.addWidget(self.upload_widget)
        
        # Ініціалізуємо result_text як None, буде встановлено в create_right_panel
        self.result_text = None
        
        # Показати початкове меню
        self.stacked_widget.setCurrentIndex(0)
        
        # Статус-бар
        self.create_statusbar()
        
        # Застосування стилів
        self.apply_styles()
        
    def create_menu(self):
        """Створення меню"""
        menubar = self.menuBar()
        assert menubar is not None  # menuBar() never returns None for QMainWindow
        
        # Меню Файл
        file_menu = menubar.addMenu("Файл")
        assert file_menu is not None  # addMenu() never returns None
        
        load_action = QAction("Завантажити зображення", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        save_action = QAction(MSG_SAVE_RESULT, self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.export_text)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Вихід", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Правка
        edit_menu = menubar.addMenu("Правка")
        assert edit_menu is not None  # addMenu() never returns None
        
        clear_action = QAction("Очистити все", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.clear_all)
        edit_menu.addAction(clear_action)
        
        copy_action = QAction("Копіювати текст", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_text)
        edit_menu.addAction(copy_action)
        
        # Меню Налаштування
        settings_menu = menubar.addMenu("Налаштування")
        assert settings_menu is not None  # addMenu() never returns None
        
        config_action = QAction("Параметри OCR", self)
        config_action.triggered.connect(self.show_settings)
        settings_menu.addAction(config_action)
        
        # Меню Допомога
        help_menu = menubar.addMenu("Допомога")
        assert help_menu is not None  # addMenu() never returns None
        
        about_action = QAction("Про програму", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_draw_interface(self):
        """Створення інтерфейсу для малювання (без кнопки завантаження)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Використовуємо панель налаштувань OCR для режиму малювання
        self.ocr_settings_draw.setVisible(True)  # Завжди видима
        layout.addWidget(self.ocr_settings_draw)
        
        # Основний контент (розділений на дві панелі)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Ліва панель - малювання
        left_panel = self.create_draw_panel()
        splitter.addWidget(left_panel)
        
        # Права панель - результат (використовує спільний result_text)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_upload_interface(self):
        """Створення інтерфейсу для завантаження зображення (з кнопкою завантаження)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Використовуємо панель налаштувань OCR для режиму завантаження
        self.ocr_settings_upload.setVisible(True)  # Завжди видима
        layout.addWidget(self.ocr_settings_upload)
        
        # Основний контент (розділений на дві панелі)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Ліва панель - завантаження зображення
        left_panel = self.create_upload_panel()
        splitter.addWidget(left_panel)
        
        # Права панель - результат (використовує спільний result_text)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        return widget
    
    def _sync_ocr_settings(self):
        """Синхронізація налаштувань між панелями"""
        # Синхронізуємо зміни з draw панелі на upload панель
        self.ocr_settings_draw.engine_combo.currentTextChanged.connect(
            lambda text: self.ocr_settings_upload.engine_combo.setCurrentText(text)
        )
        self.ocr_settings_draw.language_combo.currentTextChanged.connect(
            lambda text: self.ocr_settings_upload.language_combo.setCurrentText(text)
        )
        self.ocr_settings_draw.ai_correction_check.toggled.connect(
            lambda checked: self.ocr_settings_upload.ai_correction_check.setChecked(checked)
        )
        
        # Синхронізуємо зміни з upload панелі на draw панель
        self.ocr_settings_upload.engine_combo.currentTextChanged.connect(
            lambda text: self.ocr_settings_draw.engine_combo.setCurrentText(text)
        )
        self.ocr_settings_upload.language_combo.currentTextChanged.connect(
            lambda text: self.ocr_settings_draw.language_combo.setCurrentText(text)
        )
        self.ocr_settings_upload.ai_correction_check.toggled.connect(
            lambda checked: self.ocr_settings_draw.ai_correction_check.setChecked(checked)
        )
    
    def _get_current_ocr_settings(self):
        """Отримання поточної панелі налаштувань залежно від режиму"""
        if self.current_mode == 'draw':
            return self.ocr_settings_draw
        elif self.current_mode == 'upload':
            return self.ocr_settings_upload
        else:
            # Fallback на draw панель
            return self.ocr_settings_draw
    
    def _sync_combo_values(self, source_combo, target_combo):
        """Синхронізація значень між комбобоксами"""
        if source_combo.currentText() != target_combo.currentText():
            target_combo.setCurrentText(source_combo.currentText())
    
    def _create_combo_pair(self):
        """Створення пари комбобоксів (мова та рушій) з синхронізацією"""
        language_combo = QComboBox()
        language_combo.addItems([LANG_UKRAINIAN, LANG_ENGLISH])
        language_combo.setMinimumWidth(200)
        
        ocr_engine_combo = QComboBox()
        ocr_engine_combo.setMinimumWidth(150)
        
        # Якщо вже є спільні комбобокси, синхронізуємо значення
        if hasattr(self, 'language_combo') and self.language_combo is not None:
            language_combo.setCurrentText(self.language_combo.currentText())
            # Підключаємо синхронізацію в обидва боки
            self.language_combo.currentTextChanged.connect(
                lambda text, target=language_combo: self._sync_combo_values(self.language_combo, target)
            )
            language_combo.currentTextChanged.connect(
                lambda text, source=language_combo: self._sync_combo_values(source, self.language_combo)
            )
        else:
            # Перший раз - створюємо спільні комбобокси
            self.language_combo = language_combo
            self.ocr_engine_combo = ocr_engine_combo
            # Підключаємо сигнал зміни мови для оновлення списку рушіїв
            self.language_combo.currentTextChanged.connect(self.update_ocr_engines)
            self.update_ocr_engines(LANG_UKRAINIAN)
        
        # Синхронізуємо рушії
        if hasattr(self, 'ocr_engine_combo') and self.ocr_engine_combo is not None and self.ocr_engine_combo != ocr_engine_combo:
            # Копіюємо елементи з спільного комбобоксу
            ocr_engine_combo.clear()
            for i in range(self.ocr_engine_combo.count()):
                ocr_engine_combo.addItem(self.ocr_engine_combo.itemText(i))
            ocr_engine_combo.setCurrentText(self.ocr_engine_combo.currentText())
            # Підключаємо синхронізацію в обидва боки
            self.ocr_engine_combo.currentTextChanged.connect(
                lambda text, target=ocr_engine_combo: self._sync_combo_values(self.ocr_engine_combo, target)
            )
            ocr_engine_combo.currentTextChanged.connect(
                lambda text, source=ocr_engine_combo: self._sync_combo_values(source, self.ocr_engine_combo)
            )
        elif not hasattr(self, 'ocr_engine_combo') or self.ocr_engine_combo is None:
            # Якщо спільний комбобокс ще не створений, ініціалізуємо його
            self.ocr_engine_combo = ocr_engine_combo
        
        # Підключаємо оновлення рушіїв при зміні мови
        language_combo.currentTextChanged.connect(
            lambda text, lang_combo=language_combo, eng_combo=ocr_engine_combo: self._update_combo_engines(lang_combo, eng_combo)
        )
        
        # Якщо це перший комбобокс, ініціалізуємо список рушіїв
        if ocr_engine_combo.count() == 0:
            self._update_combo_engines(language_combo, ocr_engine_combo)
        
        return language_combo, ocr_engine_combo
    
    def _update_combo_engines(self, language_combo, engine_combo):
        """Оновлення списку рушіїв для конкретного комбобоксу"""
        language = language_combo.currentText()
        current_selection = engine_combo.currentText()
        
        # Перевіряємо доступність рушіїв
        from model.unified_ocr_adapter import UnifiedOCRAdapter
        temp_adapter = UnifiedOCRAdapter(engine='tesseract')
        available_engines = temp_adapter.get_available_engines()
        
        # Для обох мов однаковий набір OCR рушіїв
        possible_engines = ["Tesseract", "EasyOCR", "PaddleOCR"]
        if language == LANG_UKRAINIAN:
            default_engine = "Tesseract"
        else:
            default_engine = None
        
        # Зберігаємо інформацію про доступність для tooltip
        if not hasattr(self, '_engine_availability'):
            self._engine_availability = {
                'tesseract': 'tesseract' in available_engines,
                'easyocr': 'easyocr' in available_engines,
                'paddleocr': 'paddleocr' in available_engines
            }
        
        # Оновлюємо список
        engine_combo.blockSignals(True)  # Блокуємо сигнали під час оновлення
        engine_combo.clear()
        engine_combo.addItems(possible_engines)
        
        # Відновлюємо попередній вибір, якщо він доступний
        if current_selection in possible_engines:
            engine_combo.setCurrentText(current_selection)
        elif default_engine and default_engine in possible_engines:
            engine_combo.setCurrentText(default_engine)
        else:
            engine_combo.setCurrentIndex(0)
        engine_combo.blockSignals(False)
    
    def create_settings_panel(self):
        """Панель швидких налаштувань"""
        group = QGroupBox("Налаштування")
        group.setMinimumHeight(80)  # Встановлюємо мінімальну висоту
        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)  # Додаємо відступи
        
        # Створюємо пару комбобоксів для цієї панелі
        language_combo, ocr_engine_combo = self._create_combo_pair()
        
        # Вибір мови
        lang_label = QLabel("Мова:")
        lang_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        layout.addWidget(lang_label)
        layout.addWidget(language_combo)
        
        layout.addSpacing(20)
        
        # Вибір OCR рушія
        engine_label = QLabel("OCR рушій:")
        engine_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        layout.addWidget(engine_label)
        layout.addWidget(ocr_engine_combo)
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
        
    def get_engine_tooltip(self, language, engine):
        """Отримання тексту підказки для рушія OCR"""
        if language in self.OCR_ENGINE_INFO and engine in self.OCR_ENGINE_INFO[language]:
            info = self.OCR_ENGINE_INFO[language][engine]
            return f"{info['description']}\n\nТочність розпізнавання: {info['accuracy']}%"
        return ""
    
    def update_ocr_engines(self, language):
        """Оновлення списку доступних OCR рушіїв залежно від вибраної мови"""
        # Зберігаємо поточну мову
        self.current_language = language
        
        current_selection = self.ocr_engine_combo.currentText()
        
        # Перевіряємо доступність рушіїв для tooltip
        from model.unified_ocr_adapter import UnifiedOCRAdapter
        temp_adapter = UnifiedOCRAdapter(engine='tesseract')
        available_engines = temp_adapter.get_available_engines()
        
        # Для української мови приховуємо PaddleOCR та встановлюємо Tesseract як основний
        # Tesseract має найкращу підтримку української мови
        if language == LANG_UKRAINIAN:
            engines = ["Tesseract", "EasyOCR"]
            # Встановлюємо Tesseract як основний для української
            default_engine = "Tesseract"
        else:
            engines = ["Tesseract", "EasyOCR", "PaddleOCR"]
            default_engine = None
        
        # Зберігаємо інформацію про доступність для tooltip
        self._engine_availability = {
            'tesseract': 'tesseract' in available_engines,
            'easyocr': 'easyocr' in available_engines,
            'paddleocr': 'paddleocr' in available_engines
        }
        
        # Відключаємо сигнал перед очищенням, щоб уникнути зайвих викликів
        try:
            self.ocr_engine_combo.currentTextChanged.disconnect()
        except TypeError:
            # Сигнал не був підключений, це нормально
            pass
        
        # Оновлюємо список
        self.ocr_engine_combo.clear()
        self.ocr_engine_combo.addItems(engines)
        
        # Встановлюємо tooltip для кожного рушія в списку
        for i, engine in enumerate(engines):
            tooltip_text = self.get_engine_tooltip(language, engine)
            
            # Додаємо інформацію про доступність до tooltip
            engine_lower = engine.lower()
            if engine_lower in self._engine_availability:
                is_available = self._engine_availability[engine_lower]
                if not is_available:
                    availability_note = "\n\n⚠️ Увага: Рушій зараз недоступний (проблеми з PyTorch DLL). Буде автоматично використано Tesseract."
                    tooltip_text = (tooltip_text or "") + availability_note
            
            if tooltip_text:
                # Встановлюємо tooltip через модель комбобоксу
                model = self.ocr_engine_combo.model()
                if model:
                    index = model.index(i, 0)
                    model.setData(index, tooltip_text, Qt.ItemDataRole.ToolTipRole)
        
        # Відновлюємо попередній вибір, якщо він все ще доступний
        if current_selection in engines:
            self.ocr_engine_combo.setCurrentText(current_selection)
        else:
            # Якщо попередній вибір недоступний, вибираємо рекомендований або перший доступний
            if default_engine and default_engine in engines:
                self.ocr_engine_combo.setCurrentText(default_engine)
            else:
                self.ocr_engine_combo.setCurrentIndex(0)
        
        # Встановлюємо tooltip для вибраного рушія
        self.update_engine_tooltip()
        
        # Підключаємо сигнал для оновлення tooltip при зміні вибору
        self.ocr_engine_combo.currentTextChanged.connect(self.update_engine_tooltip)
    
    def update_engine_tooltip(self):
        """Оновлення tooltip комбобоксу з інформацією про вибраний рушій"""
        current_engine = self.ocr_engine_combo.currentText()
        tooltip_text = self.get_engine_tooltip(self.current_language, current_engine)
        
        # Додаємо інформацію про доступність до tooltip
        engine_lower = current_engine.lower()
        if hasattr(self, '_engine_availability') and engine_lower in self._engine_availability:
            is_available = self._engine_availability[engine_lower]
            if not is_available:
                availability_note = "\n\n⚠️ Увага: Рушій зараз недоступний (проблеми з PyTorch DLL). Буде автоматично використано Tesseract."
                tooltip_text = (tooltip_text or "") + availability_note
        
        if tooltip_text:
            self.ocr_engine_combo.setToolTip(tooltip_text)
        else:
            self.ocr_engine_combo.setToolTip("")
        
    def create_draw_panel(self):
        """Ліва панель - малювання (тільки полотно)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Заголовок
        title = QLabel("Малювання тексту")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(TITLE_STYLE)
        layout.addWidget(title)
        
        # Полотно для малювання
        self.canvas = CanvasWidget()
        layout.addWidget(self.canvas)
        
        # Панель інструментів для малювання
        self.canvas_toolbar = self.create_canvas_toolbar()
        layout.addWidget(self.canvas_toolbar)
        
        return widget
    
    def create_upload_panel(self):
        """Ліва панель - завантаження зображення"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Заголовок
        title = QLabel("Завантажене зображення")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(TITLE_STYLE)
        layout.addWidget(title)
        
        # Лейбл для зображення
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px dashed #bdc3c7; 
            background-color: #ecf0f1; 
            border-radius: 8px;
            color: #7f8c8d;
            padding: 20px;
        """)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setText("Завантажте зображення з рукописним текстом")
        layout.addWidget(self.image_label)
        
        return widget
        
    def create_canvas_toolbar(self):
        """Панель інструментів для малювання"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)
        
        pen_btn = QPushButton("Перо")
        pen_btn.clicked.connect(lambda: self.canvas.set_tool("pen"))
        layout.addWidget(pen_btn)
        
        eraser_btn = QPushButton("Гумка")
        eraser_btn.clicked.connect(lambda: self.canvas.set_tool("eraser"))
        layout.addWidget(eraser_btn)
        
        thickness_label = QLabel("Товщина:")
        thickness_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(thickness_label)
        thickness_combo = QComboBox()
        thickness_combo.addItems(["2", "4", "6", "8", "10"])
        thickness_combo.setCurrentText("4")
        thickness_combo.currentTextChanged.connect(
            lambda t: self.canvas.set_pen_width(int(t))
        )
        layout.addWidget(thickness_combo)
        
        clear_canvas_btn = QPushButton("Очистити полотно")
        clear_canvas_btn.clicked.connect(self.canvas.clear)
        layout.addWidget(clear_canvas_btn)
        
        layout.addStretch()
        
        return widget
        
    def create_right_panel(self):
        """Права панель - результат"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Заголовок
        title = QLabel("Результат розпізнавання")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(TITLE_STYLE)
        layout.addWidget(title)
        
        # Створюємо текстове поле для результату (спільне для обох режимів)
        result_text = QTextEdit()
        result_text.setPlaceholderText("Тут з'явиться розпізнаний текст...")
        result_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                background-color: white;
                color: #2c3e50;
            }
        """)
        layout.addWidget(result_text)
        
        # Зберігаємо посилання на result_text в самому віджеті для подальшого пошуку
        # Використовуємо setProperty для збереження посилання
        widget.setProperty("result_text", result_text)
        # Також зберігаємо як атрибут для зворотної сумісності
        setattr(widget, 'result_text', result_text)
        
        # Оновлюємо посилання на result_text (завжди вказує на поточний активний)
        # Але не перезаписуємо, якщо вже є встановлений для поточного режиму
        if not hasattr(self, 'result_text') or self.result_text is None:
            self.result_text = result_text
        
        # Кнопки керування результатом
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        copy_btn = QPushButton("Копіювати")
        copy_btn.clicked.connect(self.copy_text)
        buttons_layout.addWidget(copy_btn)
        
        save_txt_btn = QPushButton("Зберегти TXT")
        save_txt_btn.clicked.connect(lambda: self.export_text("txt"))
        buttons_layout.addWidget(save_txt_btn)
        
        save_docx_btn = QPushButton("Зберегти DOCX")
        save_docx_btn.clicked.connect(lambda: self.export_text("docx"))
        buttons_layout.addWidget(save_docx_btn)
        
        save_pdf_btn = QPushButton("Зберегти PDF")
        save_pdf_btn.clicked.connect(lambda: self.export_text("pdf"))
        buttons_layout.addWidget(save_pdf_btn)
        
        layout.addLayout(buttons_layout)
        
        return widget
        
    def create_statusbar(self):
        """Створення статус-бару"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        self.statusbar.showMessage("Готово до роботи")
        
    def show_welcome_screen(self):
        """Показати початкове меню"""
        self.stacked_widget.setCurrentIndex(0)
        self.current_mode = None
        # Приховуємо toolbar на головному меню
        if hasattr(self, 'toolbar'):
            self.toolbar.hide_toolbar()
        
    def show_draw_mode(self):
        """Показати режим малювання"""
        self.current_mode = 'draw'
        # draw_widget знаходиться на індексі 1 (після welcome_screen)
        self.stacked_widget.setCurrentIndex(1)
        # Показуємо toolbar для режиму малювання (без кнопки завантаження)
        if hasattr(self, 'toolbar'):
            self.toolbar.show_for_draw_mode()
        # Оновлюємо посилання на result_text для поточного режиму
        self._update_result_text_reference()
        
    def show_upload_mode(self):
        """Показати режим завантаження"""
        self.current_mode = 'upload'
        # upload_widget знаходиться на індексі 2 (після welcome_screen та draw_widget)
        self.stacked_widget.setCurrentIndex(2)
        # Показуємо toolbar для режиму завантаження (з кнопкою завантаження)
        if hasattr(self, 'toolbar'):
            self.toolbar.show_for_upload_mode()
        # Оновлюємо посилання на result_text для поточного режиму
        self._update_result_text_reference()
            
    def load_image(self):
        """Завантаження зображення"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # Перевірка розміру файлу (обмеження 10MB)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > 10:
                QMessageBox.warning(
                    self, MSG_ERROR,
                    "Файл занадто великий. Максимальний розмір: 10 MB"
                )
                return
                
            # Валідація формату зображення
            from handwrite_config import Config
            if not Config.is_valid_image_format(file_path):
                QMessageBox.warning(
                    self, MSG_ERROR,
                    f"Непідтримуваний формат файлу. Підтримувані формати: {', '.join(Config.SUPPORTED_IMAGE_FORMATS)}"
                )
                return
            
            # Перевірка коректності зображення
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(
                    self, MSG_ERROR,
                    "Не вдалося завантажити зображення. Файл може бути пошкоджений або має непідтримуваний формат."
                )
                return
            
            # Перевірка розмірів зображення
            width, height = pixmap.width(), pixmap.height()
            if width > Config.MAX_IMAGE_WIDTH or height > Config.MAX_IMAGE_HEIGHT:
                QMessageBox.warning(
                    self, MSG_ERROR,
                    f"Зображення занадто велике ({width}x{height}). Максимальні розміри: {Config.MAX_IMAGE_WIDTH}x{Config.MAX_IMAGE_HEIGHT}"
                )
                return
                
            self.current_image_path = file_path
            
            # Масштабування зображення
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.statusbar.showMessage(f"Завантажено: {os.path.basename(file_path)}")
            
    def convert_text(self):
        """Запуск розпізнавання тексту"""
        image_path = self._get_image_path_for_current_mode()
        if image_path is None:
            return

        if not self._validate_image_path(image_path):
            return

        (
            engine,
            language_enum,
            use_ai,
            use_best_engine,
            llm_config,
        ) = self._get_ocr_settings_and_llm_config()

        self._start_async_recognition(
            image_path=image_path,
            engine=engine,
            language_enum=language_enum,
            use_ai=use_ai,
            use_best_engine=use_best_engine,
            llm_config=llm_config,
        )

    def _get_image_path_for_current_mode(self) -> str | None:
        """Отримання шляху до зображення залежно від поточного режиму."""
        if self.current_mode == 'upload':
            if not self.current_image_path:
                QMessageBox.warning(self, MSG_ERROR, "Завантажте зображення!")
                return None
            return self.current_image_path

        if self.current_mode == 'draw':
            if not hasattr(self, 'canvas') or self.canvas is None:
                QMessageBox.warning(self, MSG_ERROR, "Полотно не ініціалізовано!")
                return None
            image_path = self.canvas.save_to_temp()
            if not image_path:
                QMessageBox.warning(self, MSG_ERROR, "Намалюйте текст!")
                return None
            return image_path

        QMessageBox.warning(self, MSG_ERROR, "Оберіть режим роботи!")
        return None

    def _validate_image_path(self, image_path: str) -> bool:
        """Перевірка існування файлу зображення."""
        if not os.path.exists(image_path):
            QMessageBox.critical(
                self, MSG_ERROR, f"Файл зображення не знайдено: {image_path}"
            )
            return False
        return True

    def _get_ocr_settings_and_llm_config(self):
        """Отримання налаштувань OCR та LLM конфігурації."""
        ocr_settings = self._get_current_ocr_settings()
        engine = ocr_settings.get_selected_engine()
        language_enum = ocr_settings.get_selected_language()
        use_ai = ocr_settings.is_ai_correction_enabled()
        use_best_engine = ocr_settings.is_best_engine_enabled()

        from model.ocr_config import OCRConfig

        config = OCRConfig()
        llm_config = config.get_llm_config() if use_ai else None

        if use_ai and llm_config:
            llm_config["enabled"] = True
        else:
            llm_config = {"enabled": False}

        return engine, language_enum, use_ai, use_best_engine, llm_config

    def _start_async_recognition(
        self,
        image_path: str,
        engine,
        language_enum,
        use_ai: bool,
        use_best_engine: bool,
        llm_config: dict,
    ) -> None:
        """Запуск асинхронного розпізнавання через покращений контролер."""
        logger.info(
            "[MainWindow] Запуск розпізнавання: режим=%s, мова=%s, рушій=%s, найкращий=%s",
            self.current_mode,
            language_enum.value,
            engine.value,
            use_best_engine,
        )
        self.controller.recognize_text_async(
            image_path=image_path,
            engine=engine,
            language=language_enum,
            use_ai_correction=use_ai,
            use_best_engine=use_best_engine,
            llm_config=llm_config,
        )
        
    def _update_result_text_reference(self):
        """Оновлення посилання на result_text для поточного режиму"""
        current_widget = self.stacked_widget.currentWidget()
        if current_widget:
            # Шукаємо result_text в поточному віджеті
            for widget in current_widget.findChildren(QTextEdit):
                placeholder = widget.placeholderText()
                if placeholder and MSG_RECOGNIZED_TEXT in placeholder:
                    self.result_text = widget
                    logger.info(f"[MainWindow] Оновлено посилання на result_text для режиму {self.current_mode}")
                    return
    
    def _get_result_text(self):
        """Отримання посилання на result_text (створює, якщо не існує)"""
        # Спочатку шукаємо в поточному активному віджеті через атрибут
        current_widget = self.stacked_widget.currentWidget()
        if current_widget:
            # Перевіряємо, чи є атрибут result_text у поточного віджету
            # Використовуємо getattr для безпечного доступу
            result_text = getattr(current_widget, 'result_text', None)
            if result_text is None:
                # Спробуємо через property
                result_text = current_widget.property("result_text")
            if result_text is not None:
                try:
                    # Перевіряємо, чи віджет все ще існує
                    _ = result_text.placeholderText()
                    self.result_text = result_text
                    logger.info("[MainWindow] Знайдено result_text через атрибут поточного віджету")
                    return result_text
                except RuntimeError:
                    # Віджет був видалений
                    pass
            
            # Шукаємо result_text в поточному віджеті через findChildren
            for widget in current_widget.findChildren(QTextEdit):
                placeholder = widget.placeholderText()
                if placeholder and MSG_RECOGNIZED_TEXT in placeholder:
                    self.result_text = widget
                    logger.info("[MainWindow] Знайдено result_text в поточному віджеті через findChildren")
                    return widget
        
        # Якщо не знайдено, шукаємо в обох інтерфейсах
        for interface_widget in [self.draw_widget, self.upload_widget]:
            if interface_widget:
                # Спочатку перевіряємо через property, потім через hasattr
                result_text = interface_widget.property("result_text")
                if result_text is None and hasattr(interface_widget, 'result_text'):
                    result_text = getattr(interface_widget, 'result_text', None)
                    if result_text is not None:
                        try:
                            _ = result_text.placeholderText()
                            self.result_text = result_text
                            logger.info("[MainWindow] Знайдено result_text через атрибут інтерфейсу")
                            return result_text
                        except RuntimeError:
                            pass
                
                # Потім шукаємо через findChildren
                for text_edit in interface_widget.findChildren(QTextEdit):
                    placeholder = text_edit.placeholderText()
                    if placeholder and MSG_RECOGNIZED_TEXT in placeholder:
                        self.result_text = text_edit
                        logger.info("[MainWindow] Знайдено result_text в інтерфейсі через findChildren")
                        return text_edit
        
        logger.warning("[MainWindow] Не вдалося знайти result_text")
        return None
        
    def on_progress_updated(self, value: int, message: str):
        """Оновлення прогресу"""
        ocr_settings = self._get_current_ocr_settings()
        ocr_settings.show_progress(value, message)
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage(message)
    
    def on_recognition_complete(self, text, engine_name=None):
        """Обробник завершення розпізнавання"""
        ocr_settings = self._get_current_ocr_settings()
        ocr_settings.hide_progress()
        self.progress_bar.hide()
        logger.info(
            "[MainWindow] Отримано результат розпізнавання: %s символів", len(text)
        )

        self._cleanup_after_recognition()
        self._show_used_engine_if_needed(ocr_settings, engine_name)
        self._update_statusbar_after_recognition(text)
        self._display_recognition_result(text)

    def _cleanup_after_recognition(self) -> None:
        """Видалення тимчасових файлів після завершення обробки."""
        if self.current_mode == 'draw' and hasattr(self, 'canvas') and self.canvas:
            self.canvas.cleanup_temp_files()
        elif (
            self.current_mode == 'upload'
            and hasattr(self, '_temp_image_path')
            and self._temp_image_path
        ):
            try:
                if os.path.exists(self._temp_image_path):
                    os.remove(self._temp_image_path)
                    logger.info(
                        "[MainWindow] Видалено тимчасовий файл: %s",
                        self._temp_image_path,
                    )
            except Exception as error:
                logger.warning(
                    "[MainWindow] Не вдалося видалити тимчасовий файл: %s", error
                )

    def _show_used_engine_if_needed(self, ocr_settings, engine_name: str | None) -> None:
        """Показуємо використаний рушій в режимі малювання."""
        if self.current_mode == 'draw' and engine_name:
            ocr_settings.show_used_engine(engine_name)

    def _update_statusbar_after_recognition(self, text: str) -> None:
        """Оновлення статус-бару після завершення розпізнавання."""
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage(f"Готово ({len(text)} символів)", 3000)

        try:
            from model.ocr_config import OCRConfig

            config = OCRConfig()
            llm_config = config.get_llm_config()
            if llm_config.get("enabled", False):
                self.statusbar.showMessage(
                    f"Розпізнавання завершено з ШІ корекцією ({len(text)} символів)",
                    3000,
                )
            else:
                self.statusbar.showMessage(
                    f"Розпізнавання завершено ({len(text)} символів)", 3000
                )
        except Exception:
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage(
                    f"Розпізнавання завершено ({len(text)} символів)", 3000
                )

    def _display_recognition_result(self, text: str) -> None:
        """Відображення результату розпізнавання у відповідному полі або діалозі."""
        self._update_result_text_reference()
        result_text = self._get_result_text()
        if result_text is not None:
            try:
                result_text.setPlainText(text)
                result_text.ensureCursorVisible()  # Прокручуємо до початку тексту
                logger.info(
                    "[MainWindow] Результат виведено в поле результату"
                )
            except Exception as error:
                logger.error(
                    "[MainWindow] Помилка при виведенні результату: %s", error
                )
                QMessageBox.warning(
                    self, MSG_ERROR, f"Не вдалося вивести результат: {error}"
                )
        else:
            logger.error(
                "[MainWindow] Не вдалося знайти result_text для відображення результату"
            )
            QMessageBox.information(
                self,
                "Результат розпізнавання",
                text[:500] + ("..." if len(text) > 500 else ""),
            )
        
    def on_recognition_error(self, error_msg: str):
        """Помилка розпізнавання"""
        ocr_settings = self._get_current_ocr_settings()
        ocr_settings.hide_progress()
        # Приховуємо інформацію про використаний рушій при помилці
        if self.current_mode == 'draw':
            ocr_settings.hide_used_engine()
        self.progress_bar.hide()
        if hasattr(self, 'statusbar'):
            self.statusbar.showMessage(MSG_ERROR, 5000)
        QMessageBox.critical(self, MSG_ERROR, error_msg)
        """Обробник помилки розпізнавання"""
        self.progress_bar.hide()
        self.statusbar.showMessage(MSG_ERROR, 5000)
        QMessageBox.critical(self, "Помилка розпізнавання", error_msg)
        
    def copy_text(self):
        """Копіювання тексту в буфер обміну"""
        result_text = self._get_result_text()
        if result_text is None:
            QMessageBox.information(self, MSG_INFO, "Немає тексту для копіювання")
            return
        
        text = result_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            assert clipboard is not None  # clipboard() never returns None
            clipboard.setText(text)
            self.statusbar.showMessage("Текст скопійовано в буфер обміну", 3000)
        else:
            QMessageBox.information(self, MSG_INFO, "Немає тексту для копіювання")
            
    def export_text(self, format_type=None):
        """Експорт тексту у файл"""
        result_text = self._get_result_text()
        if result_text is None:
            QMessageBox.information(self, MSG_INFO, "Немає тексту для збереження")
            return
        
        text = result_text.toPlainText()
        if not text:
            QMessageBox.information(self, MSG_INFO, "Немає тексту для збереження")
            return
            
        if format_type is None:
            # Показати діалог вибору формату
            file_path, _ = QFileDialog.getSaveFileName(
                self, MSG_SAVE_RESULT, "",
                "Text files (*.txt);;Word documents (*.docx);;PDF files (*.pdf)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, MSG_SAVE_RESULT, "",
                f"{format_type.upper()} files (*.{format_type})"
            )
            
        if file_path:
            success = self.controller.export_text(text, file_path)
            if success:
                self.statusbar.showMessage(f"Збережено: {os.path.basename(file_path)}", 5000)
                QMessageBox.information(self, "Успіх", "Файл успішно збережено!")
            else:
                QMessageBox.critical(self, MSG_ERROR, "Не вдалося зберегти файл")
                
    def clear_all(self):
        """Очищення всіх даних"""
        reply = QMessageBox.question(
            self, "Підтвердження",
            "Ви впевнені, що хочете очистити всі дані?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.current_image_path = None
            
            # Очищення залежить від поточного режиму
            if self.current_mode == 'upload' and hasattr(self, 'image_label'):
                self.image_label.clear()
                self.image_label.setText("Завантажте зображення з рукописним текстом")
                self.image_label.setStyleSheet("""
                    border: 2px dashed #bdc3c7; 
                    background-color: #ecf0f1; 
                    border-radius: 8px;
                    color: #7f8c8d;
                    padding: 20px;
                """)
            elif self.current_mode == 'draw' and hasattr(self, 'canvas'):
                self.canvas.clear()
            
            # Очищення результату (якщо існує)
            result_text = self._get_result_text()
            if result_text:
                result_text.clear()
            
            self.statusbar.showMessage("Дані очищено", 3000)
            
    def show_settings(self):
        """Показати діалог налаштувань"""
        dialog = SettingsDialog(self)
        dialog.exec()
        
    def show_about(self):
        """Показати діалог 'Про програму'"""
        dialog = AboutDialog(self)
        dialog.exec()
        
    def apply_styles(self):
        """Застосування стилів"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QPushButton {
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                background-color: #3498db;
                color: white;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QComboBox {
                padding: 6px 12px;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background-color: white;
                color: #2c3e50;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                selection-background-color: #3498db;
                selection-color: white;
            }
        """)
