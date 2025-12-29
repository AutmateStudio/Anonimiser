# Используем официальный легкий образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости для сборки некоторых библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python-библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модель русского языка (lg — большая, для лучшей точности)
# Обратите внимание: скачивание модели происходит на этапе сборки образа
RUN python -m spacy download ru_core_news_lg

# Копируем основной код приложения
COPY main.py .

# Открываем порт
EXPOSE 8000

# Запускаем приложение через uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]