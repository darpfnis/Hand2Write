"""
Панель інструментів для головного вікна
handwrite2print/app/view/handwrite_toolbar.py
"""
from PyQt6.QtWidgets import (QToolBar, QPushButton, QWidget, QSizePolicy)
from PyQt6.QtCore import Qt


class HandwriteToolBar(QToolBar):
    """Панель інструментів для програми розпізнавання рукописного тексту"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self._init_load_button_state()
        self.init_ui()
    
    def _init_load_button_state(self):
        """Ініціалізація стану кнопки завантаження"""
        self.load_btn = QPushButton("Завантажити")
        self.load_btn.setToolTip("Завантажити зображення з файлу")
        self.load_separator_before = None
        self.load_separator_after = None
        self.load_btn_added = False
    
    def init_ui(self):
        """Ініціалізація інтерфейсу toolbar"""
        self._add_back_button()
        self._add_left_spacer()
        self._add_main_buttons()
        self._add_right_spacer()
    
    def _add_back_button(self):
        """Додавання кнопки повернення до меню"""
        self.back_btn = QPushButton("Назад до меню")
        self.back_btn.setToolTip("Повернутися до початкового меню")
        self.addWidget(self.back_btn)
        self.addSeparator()
    
    def _add_left_spacer(self):
        """Додавання лівого spacer для центрування"""
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(left_spacer)
    
    def _add_main_buttons(self):
        """Додавання основних кнопок toolbar"""
        self._add_convert_button()
        self.addSeparator()
        self._add_export_button()
        self.addSeparator()
        self._add_clear_button()
    
    def _add_convert_button(self):
        """Додавання кнопки розпізнавання"""
        self.convert_btn = QPushButton("Розпізнати текст")
        self.convert_btn.setToolTip("Запустити розпізнавання тексту")
        self.convert_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px 20px;")
        self.addWidget(self.convert_btn)
    
    def _add_export_button(self):
        """Додавання кнопки збереження"""
        self.export_btn = QPushButton("Зберегти")
        self.export_btn.setToolTip("Зберегти результат у файл")
        self.addWidget(self.export_btn)
    
    def _add_clear_button(self):
        """Додавання кнопки очищення"""
        self.clear_btn = QPushButton("Очистити")
        self.clear_btn.setToolTip("Очистити всі дані")
        self.addWidget(self.clear_btn)
    
    def _add_right_spacer(self):
        """Додавання правого spacer для центрування"""
        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.right_spacer = right_spacer
        self.addWidget(right_spacer)

    # ===========================
    # Допоміжні методи для load_btn
    # ===========================

    def _find_action_for_widget(self, widget: QWidget):
        """Знаходить action, який містить даний widget у toolbar."""
        for action in self.actions():
            try:
                if self.widgetForAction(action) == widget:
                    return action
            except (AttributeError, RuntimeError, TypeError):
                continue
        return None

    def _remove_load_button_actions(self) -> None:
        """Видаляє кнопку завантаження та оточуючі separators, якщо вони є."""
        load_action = self._find_action_for_widget(self.load_btn)

        if load_action:
            actions_list = list(self.actions())
            try:
                load_idx = actions_list.index(load_action)
                # Видаляємо separator перед кнопкою (якщо є)
                if load_idx > 0:
                    prev_action = actions_list[load_idx - 1]
                    if prev_action.isSeparator():
                        self.removeAction(prev_action)
                # Видаляємо саму кнопку
                self.removeAction(load_action)
                # Видаляємо separator після кнопки (якщо є)
                if load_idx < len(actions_list) - 1:
                    next_action = actions_list[load_idx + 1]
                    if next_action.isSeparator():
                        self.removeAction(next_action)
            except (ValueError, IndexError, AttributeError):
                # Якщо не вдалося коректно знайти індекси, просто видаляємо action
                self.removeAction(load_action)

        # Також пробуємо видалити збережені separators, якщо вони ще присутні
        for sep_attr in ("load_separator_before", "load_separator_after"):
            sep_action = getattr(self, sep_attr, None)
            if sep_action:
                try:
                    self.removeAction(sep_action)
                except Exception:
                    pass
                setattr(self, sep_attr, None)

        self.load_btn_added = False

        # Додаткова перевірка - переконаємося, що кнопка дійсно видалена
        residual_actions = list(self.actions())
        for action in residual_actions:
            try:
                if self.widgetForAction(action) == self.load_btn:
                    self.removeAction(action)
            except (AttributeError, RuntimeError, TypeError):
                pass

    def _insert_load_button_before_action(self, anchor_action) -> None:
        """Вставляє кнопку завантаження перед вказаним action з separators."""
        self.load_separator_before = self.insertSeparator(anchor_action)
        load_action = self.insertWidget(self.load_separator_before, self.load_btn)
        self.load_separator_after = self.insertSeparator(load_action)
        self.load_btn_added = True

    def _append_load_button_to_end(self) -> None:
        """Додає кнопку завантаження та separators в кінець toolbar."""
        self.load_separator_before = self.addSeparator()
        self.addWidget(self.load_btn)
        self.load_separator_after = self.addSeparator()
        self.load_btn_added = True
    
    def show_for_draw_mode(self):
        """Показати toolbar для режиму малювання (без кнопки завантаження)"""
        self.setVisible(True)
        # Завжди намагаємося видалити кнопку завантаження (навіть якщо прапорець не встановлений)
        try:
            self._remove_load_button_actions()
        except (RuntimeError, AttributeError):
            # Якщо toolbar у некоректному стані, просто ігноруємо помилку
            pass
    
    def show_for_upload_mode(self):
        """Показати toolbar для режиму завантаження (з кнопкою завантаження)"""
        self.setVisible(True)
        if not self.load_btn_added:
            self._add_load_button_to_toolbar()
    
    def _add_load_button_to_toolbar(self):
        """Додавання кнопки завантаження до toolbar"""
        try:
            anchor_action = self._find_load_button_anchor()
            if anchor_action:
                self._insert_load_button_before_action(anchor_action)
            else:
                self._append_load_button_to_end()
        except (RuntimeError, AttributeError):
            try:
                self._append_load_button_to_end()
            except Exception:
                pass
    
    def _find_load_button_anchor(self):
        """Знаходження anchor action для вставки кнопки завантаження"""
        right_spacer_action = self._find_action_for_widget(self.right_spacer)
        if right_spacer_action:
            return right_spacer_action
        
        clear_action = self._find_action_for_widget(self.clear_btn)
        if clear_action:
            return self._get_next_action_after_clear(clear_action)
        
        return None
    
    def _get_next_action_after_clear(self, clear_action):
        """Отримання наступного action після кнопки очищення"""
        actions_list = list(self.actions())
        try:
            clear_idx = actions_list.index(clear_action)
            if clear_idx < len(actions_list) - 1:
                return actions_list[clear_idx + 1]
            return clear_action
        except (ValueError, IndexError):
            return clear_action
    
    def hide_toolbar(self):
        """Приховати toolbar (для головного меню)"""
        self.setVisible(False)
