"""
Допоміжний модуль для роботи з PyTorch та діагностики проблем
handwrite2print/app/model/pytorch_helper.py
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import ctypes
from ctypes import wintypes

logger = logging.getLogger(__name__)

# Глобальний кеш для відстеження спроб завантаження DLL
_DLL_LOAD_ATTEMPTED = False
_DLL_LOAD_SUCCESS = False


def _load_pytorch_dlls_explicitly(torch_lib_path: Path) -> bool:
    """
    Явне завантаження PyTorch DLL через LoadLibraryEx з LOAD_WITH_ALTERED_SEARCH_PATH
    Це допомагає уникнути конфліктів з PyQt6 та іншими бібліотеками
    
    Returns:
        True якщо DLL успішно завантажені
    """
    global _DLL_LOAD_ATTEMPTED, _DLL_LOAD_SUCCESS
    
    if _DLL_LOAD_ATTEMPTED and not _DLL_LOAD_SUCCESS:
        return False
    
    if _DLL_LOAD_SUCCESS:
        return True
    
    _DLL_LOAD_ATTEMPTED = True
    
    try:
        torch_lib_str = str(torch_lib_path.absolute())
        _add_dll_directory(torch_lib_str)
        
        loaded_dlls = _load_dlls_with_loadlibraryex(torch_lib_path)
        if loaded_dlls:
            logger.info(f"✓ Явно завантажено PyTorch DLL: {', '.join(loaded_dlls)}")
            _DLL_LOAD_SUCCESS = True
            return True
        
        if _try_fallback_windll(torch_lib_path):
            _DLL_LOAD_SUCCESS = True
            return True
        
        _DLL_LOAD_SUCCESS = False
        return False
    except Exception as e:
        logger.debug(f"Помилка явного завантаження PyTorch DLL: {e}")
        _DLL_LOAD_SUCCESS = False
        return False


def _add_dll_directory(torch_lib_str: str) -> None:
    """Додавання torch/lib до DLL search path через os.add_dll_directory"""
    try:
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib_str)
            logger.info(f"✓ Додано torch/lib до DLL search path: {torch_lib_str}")
    except Exception as e:
        logger.debug(f"os.add_dll_directory не працює: {e}")


def _load_dlls_with_loadlibraryex(torch_lib_path: Path) -> list[str]:
    """Завантаження критичних DLL через LoadLibraryEx"""
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        load_library_ex_w = kernel32.LoadLibraryExW
        load_library_ex_w.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
        load_library_ex_w.restype = wintypes.HMODULE
        
        LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        critical_dlls = ["c10.dll", "torch_cpu.dll"]
        
        loaded_dlls = []
        for dll_name in critical_dlls:
            dll_path = torch_lib_path / dll_name
            if dll_path.exists():
                if _try_load_dll_with_loadlibraryex(load_library_ex_w, dll_path, dll_name, LOAD_WITH_ALTERED_SEARCH_PATH):
                    loaded_dlls.append(dll_name)
                elif _try_load_dll_with_windll(dll_path, dll_name):
                    loaded_dlls.append(dll_name)
        
        return loaded_dlls
    except Exception as e:
        logger.debug(f"Помилка LoadLibraryEx: {e}")
        return []


def _try_load_dll_with_loadlibraryex(load_library_ex_w, dll_path: Path, dll_name: str, flag: int) -> bool:
    """Спроба завантажити DLL через LoadLibraryEx"""
    try:
        dll_path_str = str(dll_path.absolute())
        hmodule = load_library_ex_w(dll_path_str, None, flag)
        if hmodule:
            logger.info(f"✓ Завантажено {dll_name} через LoadLibraryEx")
            return True
        
        _log_dll_load_error(dll_name, ctypes.get_last_error())
        return False
    except Exception:
        return False


def _try_load_dll_with_windll(dll_path: Path, dll_name: str) -> bool:
    """Спроба завантажити DLL через WinDLL (fallback)"""
    try:
        ctypes.WinDLL(str(dll_path))
        logger.info(f"✓ Завантажено {dll_name} через WinDLL")
        return True
    except Exception as e:
        logger.debug(f"Не вдалося завантажити {dll_name}: {e}")
        return False


def _log_dll_load_error(dll_name: str, error_code: int) -> None:
    """Логування помилки завантаження DLL"""
    if error_code == 1114:
        logger.warning(f"✗ {dll_name} не може бути ініціалізована (WinError 1114)")
        logger.warning("  Це системна проблема. Перевірте Visual C++ Redistributables та перевстановіть PyTorch.")
    else:
        logger.debug(f"Не вдалося завантажити {dll_name}, код помилки: {error_code}")


def _try_fallback_windll(torch_lib_path: Path) -> bool:
    """Fallback завантаження через простий WinDLL"""
    try:
        critical_dlls = ["c10.dll", "torch_cpu.dll"]
        for dll_name in critical_dlls:
            dll_path = torch_lib_path / dll_name
            if dll_path.exists():
                ctypes.WinDLL(str(dll_path))
                logger.info(f"✓ Завантажено {dll_name} через WinDLL (fallback)")
                return True
    except Exception:
        pass
    return False


def setup_pytorch_path() -> bool:
    """
    Налаштування PATH для PyTorch DLL та явне завантаження критичних DLL
    ВАЖЛИВО: Викликайте це ДУЖЕ РАНО, до будь-яких імпортів torch
    
    Returns:
        True якщо PATH успішно налаштовано
    """
    try:
        import site
        torch_lib_path = None
        
        # Шукаємо torch lib в site-packages
        for site_packages in site.getsitepackages():
            potential_path = Path(site_packages) / "torch" / "lib"
            if potential_path.exists():
                torch_lib_path = potential_path
                break
        
        # Спробуємо в venv
        if torch_lib_path is None:
            venv_lib = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "torch" / "lib"
            if venv_lib.exists():
                torch_lib_path = venv_lib
        
        if torch_lib_path and torch_lib_path.exists():
            torch_lib_str = str(torch_lib_path.absolute())
            current_path = os.environ.get("PATH", "")
            # Додаємо на початок PATH для пріоритету
            if torch_lib_str not in current_path:
                os.environ["PATH"] = torch_lib_str + os.pathsep + current_path
                logger.info(f"✓ PyTorch lib додано до PATH: {torch_lib_str}")
            # Також перевіряємо, чи не потрібно додати на початок
            path_parts = current_path.split(os.pathsep)
            if torch_lib_str in path_parts and path_parts.index(torch_lib_str) > 0:
                # Переміщуємо на початок
                path_parts.remove(torch_lib_str)
                os.environ["PATH"] = torch_lib_str + os.pathsep + os.pathsep.join(path_parts)
                logger.info("✓ PyTorch lib переміщено на початок PATH")
            
            # Спробуємо явно завантажити критичні DLL перед імпортом torch
            logger.info(f"Спроба явного завантаження PyTorch DLL з {torch_lib_str}...")
            _load_pytorch_dlls_explicitly(torch_lib_path)
            
            return True
        else:
            logger.warning("✗ torch/lib не знайдено")
        
        return False
    except Exception as e:
        logger.warning(f"Помилка налаштування PyTorch PATH: {e}")
        return False


def check_pytorch_availability() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Перевірка доступності PyTorch з детальною діагностикою
    Використовує явне завантаження DLL для уникнення конфліктів
    
    Returns:
        Tuple[is_available, version, error_message]
    """
    # Спочатку налаштовуємо PATH та завантажуємо DLL явно
    setup_pytorch_path()
    
    # Додаткова затримка для завантаження DLL
    import time
    time.sleep(0.2)  # Збільшено затримку для надійності
    
    try:
        # Спробуємо імпортувати torch
        logger.info("Спроба імпорту torch...")
        import torch
        version = torch.__version__
        logger.info(f"✓ PyTorch {version} успішно імпортовано")
        
        # Тестовий виклик для перевірки роботи
        try:
            logger.info("Тестування PyTorch tensor...")
            test_tensor = torch.tensor([1.0])
            test_value = test_tensor.item()
            
            # Уникаємо прямого порівняння з плаваючою комою
            if abs(test_value - 1.0) < 1e-6:
                logger.info("✓ PyTorch працює коректно")
                return True, version, None
            return False, version, "PyTorch tensor тест не пройдено"
        except Exception as e:
            logger.error(f"✗ Помилка тесту PyTorch: {e}")
            return False, version, f"Помилка тесту PyTorch: {e}"
            
    except ImportError as e:
        logger.error(f"✗ PyTorch не встановлено: {e}")
        return False, None, f"PyTorch не встановлено: {e}"
    except OSError as e:
        error_msg = str(e)
        if "DLL" in error_msg or "1114" in error_msg:
            # WinError 1114 - це системна проблема, яку не можна вирішити кодом
            # Можливі причини:
            # 1. Відсутність Visual C++ Redistributables
            # 2. Конфлікт версій DLL
            # 3. Пошкоджені файли PyTorch
            # 4. Проблеми з правами доступу
            logger.warning("✗ PyTorch DLL не може бути ініціалізована (WinError 1114)")
            logger.warning("  Це системна проблема. Програма працюватиме з Tesseract.")
            logger.warning("  Для вирішення проблеми:")
            logger.warning("  1. Перевстановіть Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            logger.warning("  2. Перезапустіть комп'ютер")
            logger.warning("  3. Перевстановіть PyTorch:")
            logger.warning("     pip uninstall torch torchvision torchaudio -y")
            logger.warning("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            return False, None, f"Помилка завантаження PyTorch DLL: {e}"
        logger.error(f"✗ Помилка ОС при завантаженні PyTorch: {error_msg}")
        return False, None, f"Помилка ОС при завантаженні PyTorch: {e}"
    except Exception as e:
        logger.error(f"✗ Невідома помилка PyTorch: {e}")
        return False, None, f"Невідома помилка PyTorch: {e}"


def diagnose_pytorch_environment() -> Dict[str, Any]:
    """
    Діагностика середовища PyTorch
    
    Returns:
        Словник з інформацією про середовище
    """
    diagnosis = {
        'pytorch_installed': False,
        'pytorch_version': None,
        'pytorch_working': False,
        'torch_lib_path': None,
        'torch_lib_in_path': False,
        'vc_redist_installed': None,  # Можна перевірити через реєстр
        'error': None
    }
    
    # Перевірка встановлення
    try:
        import torch
        diagnosis['pytorch_installed'] = True
        diagnosis['pytorch_version'] = torch.__version__
    except ImportError:
        diagnosis['error'] = "PyTorch не встановлено"
        return diagnosis
    except Exception as e:
        diagnosis['error'] = f"Помилка імпорту PyTorch: {e}"
        return diagnosis
    
    # Перевірка шляху до lib
    try:
        import site
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
        
        if torch_lib_path:
            diagnosis['torch_lib_path'] = str(torch_lib_path.absolute())
            current_path = os.environ.get("PATH", "")
            diagnosis['torch_lib_in_path'] = str(torch_lib_path.absolute()) in current_path
    except Exception:
        pass
    
    # Перевірка роботи
    is_available, _, error = check_pytorch_availability()
    diagnosis['pytorch_working'] = is_available
    if error:
        diagnosis['error'] = error
    
    return diagnosis


def get_pytorch_error_solution() -> str:
    """
    Отримання інструкцій для вирішення проблеми PyTorch
    
    Returns:
        Текст з інструкціями
    """
    return """
═══════════════════════════════════════════════════════════════
ВИПРАВЛЕННЯ ПРОБЛЕМИ PYTORCH DLL
═══════════════════════════════════════════════════════════════

Крок 1: Встановіть Visual C++ Redistributables
─────────────────────────────────────────────────
1. Завантажте: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Встановіть завантажений файл
3. ПЕРЕЗАПУСТІТЬ КОМП'ЮТЕР (обов'язково!)

Крок 2: Перевстановіть PyTorch
────────────────────────────────
В PowerShell (в корені проекту):

.venv\\Scripts\\Activate.ps1
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Крок 3: Перевірка
──────────────────
python -c "import torch; print('PyTorch:', torch.__version__); x = torch.tensor([1.0]); print('Тест:', x.item())"

Якщо виводить версію та число - PyTorch працює! ✅

═══════════════════════════════════════════════════════════════
"""

