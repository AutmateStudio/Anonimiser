"""
Улучшенная версия сервиса анонимизации с использованием мощной NER модели
Использует transformers с русской BERT моделью для лучшего распознавания именованных сущностей
"""
import time
import re
from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

app = FastAPI(title="Advanced Anonymization Service")

# --- Инициализация мощной NER модели ---

# Используем русскую BERT модель для NER
# Можно использовать разные модели:
# - "Gherman/bert-base-NER-Russian" - специализированная модель для русского NER
# - "DeepPavlov/rubert-base-cased-conversational" - более мощная модель
# - "ai-forever/ruBert-base" - базовая модель от AI Forever

MODEL_NAME = "Gherman/bert-base-NER-Russian"  # Можно заменить на более мощную модель

try:
    # Инициализация NER pipeline
    print(f"Загрузка модели {MODEL_NAME}...")
    ner_pipeline = pipeline(
        "ner",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1  # Используем GPU если доступно
    )
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели {MODEL_NAME}: {e}")
    print("Попытка загрузки альтернативной модели...")
    try:
        MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
        ner_pipeline = pipeline(
            "ner",
            model=MODEL_NAME,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Альтернативная модель загружена успешно!")
    except Exception as e2:
        print(f"Ошибка загрузки альтернативной модели: {e2}")
        ner_pipeline = None


# --- Паттерны для распознавания персональных данных ---

class PatternRecognizer:
    """Класс для распознавания сущностей по паттернам"""
    
    def __init__(self):
        self.patterns = {
            "PHONE_NUMBER": [
                r'(\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',
            ],
            "INN": [
                r'\b\d{10}\b',
                r'\b\d{12}\b',
            ],
            "PASSPORT": [
                r'(?:паспорт\s*)?(?:серия\s*)?(?:\d{2}\s?\d{2}|\d{4})[\s\-]?(?:номер\s*)?\d{6}',
            ],
            "ADDRESS": [
                # Проспект с сокращением "пр" - МАКСИМАЛЬНЫЙ ПРИОРИТЕТ
                r'(?i)(?:северный|южный|восточный|западный|центральный|красный|зеленый|синий|новый|старый)\s+пр\s+\d+',
                # Линия с домом через точку
                r'(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия\s+д\.\d+[А-ЯЁа-яё]?',
                # Проспект с прилагательным
                r'(?i)[А-ЯЁа-яё]+(?:ый|ий|ой|ая|ое)\s+пр\s+\d+',
                # Линия с номером дома
                r'(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия(?:\s+[А-ЯЁа-яё\.]+)?(?:\s*[,]?\s*)?(?:д\.?\s*\d+[А-ЯЁа-яё]?|дом\s*\d+[А-ЯЁа-яё]?)?',
                # Полный адрес
                r'(?i)(?:г|город|г\.)\s+[А-ЯЁа-яё\-]+(?:\s*,\s*)?(?:(?:ул|улица|ул\.|пр-т|проспект|пр\.|наб|набережная|наб\.|пер|переулок|пер\.|ш|шоссе|ш\.|б-р|бульвар|б-р\.)\s+[А-ЯЁа-яё0-9\-\.]+)?(?:\s*,\s*)?(?:(?:д|дом|д\.|стр|строение|стр\.|корп|корпус|корп\.|к|к\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:кв|квартира|кв\.|оф|офис|оф\.)\s*[А-ЯЁа-яё0-9\-]+)?',
                # Улица с номером дома
                r'(?i)\b[А-ЯЁа-яё][А-ЯЁа-яё\-]+(?:ская|скаяя|ской|ая|ий|ый|ой|ое|ов|а|ы|и|е)\s+(?:д\.?\s*)?\d+[А-ЯЁа-яё]?\b',
                # Шоссе с домом
                r'(?i)\b[А-ЯЁа-яё]+(?:ое|ая|ий|ый|ой)\s+(?:ш|шоссе|ш\.)(?:\s*,\s*)?(?:(?:д|дом|д\.)\s*[А-ЯЁа-яё0-9\-]+)?',
            ]
        }
    
    def recognize(self, text: str) -> List[Dict]:
        """Распознает сущности по паттернам"""
        results = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    results.append({
                        "entity": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "score": 0.9  # Высокий score для паттернов
                    })
        
        return results


pattern_recognizer = PatternRecognizer()


# --- Функции постобработки ---

def clean_name_text(text: str) -> str:
    """Очищает захваченный текст имени от лишних данных"""
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text.strip())
    
    stop_words = ['время', 'место', 'номер', 'телефон', 'адрес', 'дата', 
                  'день', 'месяц', 'год', 'лет', 'часов', 'минут',
                  'квартира', 'подъезд', 'этаж', 'дом', 'улица', 'сообщу']
    
    words = text.split()
    cleaned_words = []
    
    for word in words:
        word_clean = word.strip('.,!?;:()[]{}"\'-')
        
        if not word_clean:
            continue
        
        if word_clean.lower() in stop_words:
            break
        
        if word_clean[0].isupper() and (word_clean.replace('-', '').isalpha() or 
                                       (len(word_clean) <= 3 and word_clean.isdigit())):
            cleaned_words.append(word_clean)
        elif not cleaned_words:
            if word_clean[0].isupper() and word_clean.replace('-', '').isalpha():
                cleaned_words.append(word_clean)
    
    if cleaned_words:
        result = ' '.join(cleaned_words)
        if len(result) > 50:
            result = result[:50]
        return result
    
    first_word = words[0] if words else text
    first_word = first_word.strip('.,!?;:()[]{}"\'-')
    
    if len(first_word) > 50:
        first_word = first_word[:50]
    
    return first_word


def clean_address_text(text: str) -> str:
    """Очищает захваченный текст адреса от лишних данных"""
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.rstrip('.,!?;:')
    
    if len(text) > 200:
        for delimiter in [',', '.', ';']:
            idx = text[:200].rfind(delimiter)
            if idx > 100:
                text = text[:idx].strip()
                break
        else:
            text = text[:200].strip()
    
    return text


# --- Логика управления метками ---

class RequestAnonymizer:
    """Класс для обработки одного конкретного запроса"""
    
    def __init__(self):
        self.counters = {}
        self.mapping = {}
        self.label_map = {
            "PER": "ИМЯ",
            "PERSON": "ИМЯ",
            "ORG": "ОРГАНИЗАЦИЯ",
            "LOC": "АДРЕС",
            "PHONE_NUMBER": "ТЕЛЕФОН",
            "INN": "ИНН",
            "PASSPORT": "ПАСПОРТ",
            "ADDRESS": "АДРЕС",
            "LOCATION": "АДРЕС"
        }
    
    def get_replacement(self, original_text: str, entity_type: str) -> str:
        """Получает метку-замену для сущности"""
        ru_label = self.label_map.get(entity_type, entity_type)
        
        if ru_label not in self.counters:
            self.counters[ru_label] = 1
        
        placeholder = f"{{{ru_label}_{self.counters[ru_label]}}}"
        
        if placeholder not in self.mapping:
            if entity_type in ["PERSON", "PER"]:
                cleaned_text = clean_name_text(original_text)
                self.mapping[placeholder] = cleaned_text if cleaned_text else original_text
            elif entity_type in ["ADDRESS", "LOCATION", "LOC"]:
                cleaned_text = clean_address_text(original_text)
                self.mapping[placeholder] = cleaned_text if cleaned_text else original_text
            else:
                self.mapping[placeholder] = original_text
            self.counters[ru_label] += 1
        
        return placeholder


# --- Модели API ---

class AnonymizeRequest(BaseModel):
    text: str


class AnonymizeResponse(BaseModel):
    anonymized_text: str
    mapping: Dict[str, str]
    processing_time: float


class DeanonymizeRequest(BaseModel):
    text: str
    mapping: Dict[str, str]


class DeanonymizeResponse(BaseModel):
    restored_text: str


# --- Основная функция распознавания ---

def recognize_entities(text: str) -> List[Dict]:
    """Распознает все сущности в тексте используя NER модель и паттерны"""
    all_results = []
    
    # 1. Используем NER модель если доступна
    if ner_pipeline:
        try:
            ner_results = ner_pipeline(text)
            
            for result in ner_results:
                entity_type = result.get("entity_group", "").upper()
                # Маппинг типов сущностей из модели
                if entity_type in ["PER", "PERSON"]:
                    entity_type = "PER"
                elif entity_type in ["LOC", "LOCATION"]:
                    entity_type = "ADDRESS"
                elif entity_type in ["ORG", "ORGANIZATION"]:
                    # Организации пока не анонимизируем, но можем добавить
                    continue
                
                all_results.append({
                    "entity": entity_type,
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                    "text": result.get("word", ""),
                    "score": result.get("score", 0.8)
                })
        except Exception as e:
            print(f"Ошибка при использовании NER модели: {e}")
    
    # 2. Используем паттерны для дополнительного распознавания
    pattern_results = pattern_recognizer.recognize(text)
    
    # 3. Объединяем результаты, удаляя дубликаты
    all_results.extend(pattern_results)
    
    # 4. Удаляем перекрывающиеся результаты (оставляем с большим score)
    filtered_results = []
    all_results = sorted(all_results, key=lambda x: (x["start"], -x["score"]))
    
    for result in all_results:
        overlaps = False
        for existing in filtered_results:
            if not (result["end"] <= existing["start"] or result["start"] >= existing["end"]):
                overlaps = True
                if result["score"] > existing["score"]:
                    filtered_results.remove(existing)
                    filtered_results.append(result)
                break
        
        if not overlaps:
            filtered_results.append(result)
    
    return sorted(filtered_results, key=lambda x: x["start"])


# --- Эндпоинты ---

@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize_text(req: AnonymizeRequest):
    start_time = time.time()
    
    manager = RequestAnonymizer()
    
    # Распознаем сущности
    entities = recognize_entities(req.text)
    
    # Создаем анонимизированный текст
    anonymized_text = req.text
    # Заменяем с конца, чтобы не сбить индексы
    for entity in reversed(entities):
        entity_type = entity["entity"]
        original_text = entity["text"]
        placeholder = manager.get_replacement(original_text, entity_type)
        
        start = entity["start"]
        end = entity["end"]
        
        anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]
    
    return AnonymizeResponse(
        anonymized_text=anonymized_text,
        mapping=manager.mapping,
        processing_time=time.time() - start_time
    )


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize_text(req: DeanonymizeRequest):
    text = req.text
    mapping = req.mapping
    
    for placeholder in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    
    return DeanonymizeResponse(restored_text=text)


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "ok",
        "model_loaded": ner_pipeline is not None,
        "model_name": MODEL_NAME if ner_pipeline else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

