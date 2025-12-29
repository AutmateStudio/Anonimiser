import re
import time
import requests
import pandas as pd
from docx import Document
from typing import List, Dict
import json

# Конфигурация
API_ENDPOINT = "http://155.212.191.224:8000/anonymize"
INPUT_FILE = "переписки с клиентами реальные.docx"
OUTPUT_FILE = "anonymization_results.xlsx"

def read_docx(file_path: str) -> str:
    """Читает содержимое .docx файла"""
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)
    return "\n".join(text)

def parse_messages(text: str) -> List[Dict[str, str]]:
    """Парсит сообщения из текста, разделяя по меткам 'Компания:' и 'Клиент:'"""
    messages = []
    
    # Паттерн для поиска сообщений
    pattern = r'(Компания:|Клиент:)(.*?)(?=Компания:|Клиент:|$)'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        sender_type = match.group(1).strip().replace(':', '')
        message_text = match.group(2).strip()
        
        if message_text:  # Игнорируем пустые сообщения
            messages.append({
                'sender': sender_type,
                'text': message_text
            })
    
    return messages

def anonymize_text(text: str) -> Dict:
    """Отправляет текст на API для анонимизации"""
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None

def process_conversations():
    """Основная функция обработки переписки"""
    print(f"Чтение файла: {INPUT_FILE}")
    
    # Читаем документ
    try:
        full_text = read_docx(INPUT_FILE)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    
    # Парсим сообщения
    print("Парсинг сообщений...")
    messages = parse_messages(full_text)
    print(f"Найдено сообщений: {len(messages)}")
    
    if not messages:
        print("Сообщения не найдены. Проверьте формат файла.")
        return
    
    # Обрабатываем каждое сообщение
    results = []
    
    for i, message in enumerate(messages, 1):
        print(f"\nОбработка сообщения {i}/{len(messages)} ({message['sender']})...")
        print(f"Текст: {message['text'][:100]}...")
        
        # Отправляем на анонимизацию
        start_time = time.time()
        api_response = anonymize_text(message['text'])
        request_time = time.time() - start_time
        
        if api_response:
            results.append({
                'Номер сообщения': i,
                'Отправитель': message['sender'],
                'Изначальное сообщение': message['text'],
                'Анонимный результат': api_response.get('anonymized_text', ''),
                'Время обработки (сек)': api_response.get('processing_time', request_time),
                'Время запроса (сек)': request_time,
                'Маппинг': json.dumps(api_response.get('mapping', {}), ensure_ascii=False)
            })
            print(f"✓ Успешно обработано за {api_response.get('processing_time', request_time):.3f} сек")
        else:
            results.append({
                'Номер сообщения': i,
                'Отправитель': message['sender'],
                'Изначальное сообщение': message['text'],
                'Анонимный результат': 'ОШИБКА',
                'Время обработки (сек)': 0,
                'Время запроса (сек)': request_time,
                'Маппинг': ''
            })
            print("✗ Ошибка при обработке")
        
        # Небольшая задержка между запросами
        time.sleep(0.5)
    
    # Сохраняем результаты в Excel
    print(f"\nСохранение результатов в {OUTPUT_FILE}...")
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    
    print(f"\n✓ Готово! Обработано {len(results)} сообщений.")
    print(f"Результаты сохранены в {OUTPUT_FILE}")

if __name__ == "__main__":
    process_conversations()

