"""
Скрипт для запуска тестов детекции персональной информации
"""
import subprocess
import sys
import os


def run_tests():
    """Запускает тесты"""
    print("=" * 60)
    print("Запуск тестов детекции персональной информации")
    print("=" * 60)
    
    # Проверяем, доступен ли API
    try:
        import requests
        response = requests.get("http://localhost:8000/docs", timeout=2)
        print("✓ API доступен")
    except:
        print("⚠ ВНИМАНИЕ: API недоступен на http://localhost:8000")
        print("  Убедитесь, что сервер запущен:")
        print("  python main.py")
        print("  или")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        print()
    
    # Запускаем тесты
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_anonymization.py", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

