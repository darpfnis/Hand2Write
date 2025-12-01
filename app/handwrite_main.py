"""
Головний файл програми - точка входу
handwrite2print/app/main.py

КРИТИЧНО: PyTorch завантажується ДО створення QApplication
"""
import sys
import os
import logging

# ========================================
# КРОК 1: Налаштування PADDLE_PDX_CACHE_HOME (ДУЖЕ РАНО)
# ========================================
try:
    from pathlib import Path
    base_dir = Path(__file__).parent.parent
    paddleocr_models_dir = base_dir / "resources" / "models" / "paddleocr"
    paddleocr_models_dir.mkdir(exist_ok=True, parents=True)

    paddleocr_models_path = str(paddleocr_models_dir.absolute())
    os.environ['PADDLE_PDX_CACHE_HOME'] = paddleocr_models_path
    os.environ['PADDLEX_HOME'] = paddleocr_models_path
    os.environ['PADDLE_HOME'] = paddleocr_models_path
except (OSError, PermissionError, Exception):
    paddlex_home = r'D:\paddlex_models'
    try:
        os.makedirs(paddlex_home, exist_ok=True)
        os.environ['PADDLE_PDX_CACHE_HOME'] = paddlex_home
        os.environ['PADDLEX_HOME'] = paddlex_home
        os.environ['PADDLE_HOME'] = paddlex_home
    except (OSError, PermissionError):
        pass

# ========================================
# КРОК 2: Приховування попереджень
# ========================================
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*MPS.*')
warnings.filterwarnings('ignore', message='.*accelerator.*')
warnings.filterwarnings('ignore', message='.*pinned memory.*')
warnings.filterwarnings('ignore', message='.*defaulting to CPU.*')
warnings.filterwarnings('ignore', message='.*much faster with a GPU.*')

# ========================================
# КРОК 3: Налаштування логування
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger('app.model.ocr_strategies').setLevel(logging.INFO)
logging.getLogger('app.model.unified_ocr_adapter').setLevel(logging.INFO)
logging.getLogger('app.model.ocr_adapter').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# ========================================
# КРОК 4: КРИТИЧНО - Завантаження PyTorch ДО QApplication
# ========================================
def preload_pytorch():
    """
    Попереднє завантаження PyTorch ДО створення QApplication
    Це вирішує конфлікт DLL між PyQt6 та PyTorch
    """
    try:
        import site
        from pathlib import Path

        logger.info("Налаштування PyTorch PATH...")

        # Знаходимо torch/lib
        torch_lib_path = None
        for site_packages in site.getsitepackages():
            potential_path = Path(site_packages) / "torch" / "lib"
            if potential_path.exists():
                torch_lib_path = potential_path
                break
        
        if torch_lib_path is None:
            venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "torch" / "lib"
            if venv_lib.exists():
                torch_lib_path = venv_lib
        
        if torch_lib_path and torch_lib_path.exists():
            torch_lib_str = str(torch_lib_path.absolute())
            current_path = os.environ.get("PATH", "")

            # Додаємо на ПОЧАТОК PATH для пріоритету
            if torch_lib_str not in current_path:
                os.environ["PATH"] = torch_lib_str + os.pathsep + current_path
            else:
                # Переміщуємо на початок
                path_parts = current_path.split(os.pathsep)
                if torch_lib_str in path_parts:
                    path_parts.remove(torch_lib_str)
                    os.environ["PATH"] = torch_lib_str + os.pathsep + os.pathsep.join(path_parts)

            logger.info("✓ PyTorch lib додано до PATH: %s", torch_lib_str)

            # Явне завантаження DLL через helper
            try:
                from model.pytorch_helper import setup_pytorch_path
                setup_pytorch_path()
                logger.info("✓ PyTorch DLL завантажені явно")
            except Exception as e:
                logger.debug(f"Часткова помилка налаштування: {e}")

            # КРИТИЧНО: Імпортуємо та ініціалізуємо PyTorch ДО QApplication
            logger.info("Завантаження PyTorch (до QApplication)...")
            import time
            time.sleep(0.1)  # Невелика затримка для DLL
            
            import torch
            logger.info("✓ PyTorch %s завантажено успішно", torch.__version__)
            
            # Тестовий виклик для повної ініціалізації
            _ = torch.tensor([1.0])
            logger.info("✓ PyTorch повністю ініціалізовано")
            
            return True
    except Exception as e:
        error_msg = str(e)
        if "DLL" in error_msg or "1114" in error_msg:
            logger.warning("⚠️ PyTorch недоступний (DLL помилка)")
            logger.warning("   Програма працюватиме з Tesseract")
            logger.info("   Детальне рішення: FINAL_SOLUTION_PYTORCH.md")
        else:
            logger.warning("⚠️ PyTorch недоступний: %s", error_msg[:100])
        return False


def main():
    """Запуск програми"""
    try:
        logger.info("=" * 60)
        logger.info("Запуск програми розпізнавання рукописного тексту")
        logger.info("=" * 60)

        # ========================================
        # КРОК 5: Завантажуємо PyTorch ПЕРЕД QApplication
        # ========================================
        pytorch_loaded = preload_pytorch()
        
        # ========================================
        # КРОК 6: ТЕПЕР створюємо QApplication
        # ========================================
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        app = QApplication(sys.argv)
        app.setApplicationName("Система перетворення рукописного тексту")
        app.setOrganizationName("KPI")
        app.setStyle("Fusion")
        
        # ========================================
        # КРОК 7: Показуємо splash screen
        # ========================================
        from view.splash_screen import SplashScreen
        
        splash = SplashScreen()
        splash.update_message("Ініціалізація середовища...")
        QApplication.processEvents()
        
        # ========================================
        # КРОК 8: Попереднє завантаження моделей OCR (якщо PyTorch доступний)
        # ========================================
        # КРИТИЧНО: Завантажуємо моделі ДО створення головного вікна
        # Це переносить 78-секундну затримку на старт програми
        if pytorch_loaded:
            try:
                from model.model_preloader import ModelPreloaderThread
                
                def update_progress(message: str):
                    """Оновлення повідомлення на splash screen"""
                    splash.update_message(message)
                    QApplication.processEvents()

                # Створюємо потік для завантаження моделей
                preloader = ModelPreloaderThread(progress_callback=update_progress)
                preloader.progress_updated.connect(update_progress)
                
                # Запускаємо завантаження
                splash.update_message("Завантаження моделей OCR (це може зайняти ~1 хвилину)...")
                QApplication.processEvents()
                preloader.start()
                
                # КРИТИЧНО: Чекаємо завершення завантаження (з обмеженням часу - максимум 2 хвилини)
                # Це переносить затримку на старт програми, але забезпечує миттєве використання
                logger.info("[Preloader] Початок завантаження моделей OCR...")
                logger.info("[Preloader] Це може зайняти до 2 хвилин...")
                
                if preloader.wait(120000):  # 120 секунд = 2 хвилини
                    logger.info("[Preloader] ✓ Моделі успішно завантажено")
                    splash.update_message("Моделі готові")
                else:
                    logger.warning("[Preloader] Завантаження моделей зайняло занадто багато часу, продовжуємо...")
                    splash.update_message("Завершення завантаження...")
                    # Не перериваємо потік, він продовжить в фоні

                QApplication.processEvents()

            except Exception as e:
                logger.warning("[Preloader] Помилка попереднього завантаження моделей: %s", e)
                # Продовжуємо роботу навіть якщо не вдалося завантажити моделі
        
        # ========================================
        # КРОК 10: Створюємо головне вікно (ПІСЛЯ завантаження моделей)
        # ========================================
        splash.update_message("Створення інтерфейсу...")
        QApplication.processEvents()
        
        from view.handwrite_main_window import MainWindow
        window = MainWindow()
        
        # ========================================
        # КРОК 11: Показуємо вікно та закриваємо splash
        # ========================================
        splash.update_message("Завершення завантаження...")
        QApplication.processEvents()
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(300, lambda: None)
        QApplication.processEvents()
        
        window.show()
        splash.finish(window)
        
        logger.info("=" * 60)
        logger.info("Програма готова до роботи")
        if pytorch_loaded:
            logger.info("PyTorch: ДОСТУПНИЙ")
        else:
            logger.info("PyTorch: НЕДОСТУПНИЙ (використовується Tesseract)")
        logger.info("=" * 60)

        sys.exit(app.exec())

    except KeyboardInterrupt:
        logger.info("Завантаження перервано користувачем")
        sys.exit(0)
    except Exception:
        import traceback
        print("\n" + "=" * 60)
        print("КРИТИЧНА ПОМИЛКА ПРИ ЗАПУСКУ:")
        print("=" * 60)
        print(traceback.format_exc())
        print("=" * 60)
        try:
            input("\nНатисніть Enter для виходу...")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
