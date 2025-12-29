import time
import re
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.entities import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

app = FastAPI(title="Anonymization Service")

# --- Инициализация Presidio ---

configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "ru", "model_name": "ru_core_news_lg"}]
}
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

registry = RecognizerRegistry()
registry.load_predefined_recognizers(nlp_engine=nlp_engine)

# Кастомные рекогнайзеры
address_recognizer = PatternRecognizer(
    supported_entity="ADDRESS",
    name="RU_ADDRESS",
    supported_language="ru",
    patterns=[Pattern(
        name="address",
        regex=r"(?i)\b(?:г|ул|пр-т|проспект|наб|пер|д|дом|корп|стр|кв|обл|район)\.?\s+[А-ЯЁа-яё0-9\-\.]+(?:[\s,]+(?:г|ул|пр-т|проспект|наб|пер|д|дом|корп|стр|кв|обл|район)\.?\s+[А-ЯЁа-яё0-9\-\.]+)*",
        score=0.6
    )]
)

phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    name="RU_PHONE",
    supported_language="ru",
    patterns=[
        Pattern(name="phone", regex=r'(\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}', score=0.8)]
)

inn_recognizer = PatternRecognizer(
    supported_entity="INN",
    name="RU_INN",
    supported_language="ru",
    patterns=[Pattern(name="inn", regex=r'\b\d{10}\b|\b\d{12}\b', score=0.8)]
)

passport_recognizer = PatternRecognizer(
    supported_entity="PASSPORT",
    name="RU_PASSPORT",
    supported_language="ru",
    patterns=[Pattern(
        name="passport",
        regex=r"(?<!\d)(?:паспорт\s*)?(?:серия\s*)?(?:\d{2}\s?\d{2}|\d{4})[\s\-]?(?:номер\s*)?\d{6}(?!\d)",
        score=0.95
    )]
)

registry.add_recognizer(address_recognizer)
registry.add_recognizer(phone_recognizer)
registry.add_recognizer(inn_recognizer)
registry.add_recognizer(passport_recognizer)

analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
anonymizer_engine = AnonymizerEngine()


# --- Функции постобработки для улучшения качества распознавания имен ---

def clean_name_text(text: str) -> str:
    """Очищает захваченный текст имени от лишних данных"""
    if not text:
        return text
    
    # Удаляем переносы строк и лишние пробелы
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Список ключевых слов, которые не должны быть в имени
    stop_words = ['время', 'место', 'номер', 'телефон', 'адрес', 'дата', 
                  'день', 'месяц', 'год', 'лет', 'часов', 'минут',
                  'квартира', 'подъезд', 'этаж', 'дом', 'улица', 'сообщу']
    
    # Разбиваем на слова
    words = text.split()
    
    # Фильтруем слова: берем только те, которые не являются стоп-словами
    # и выглядят как имена (начинаются с заглавной буквы, содержат только буквы)
    cleaned_words = []
    for word in words:
        # Удаляем знаки препинания в начале/конце
        word_clean = word.strip('.,!?;:()[]{}"\'-')
        
        # Пропускаем пустые слова
        if not word_clean:
            continue
        
        # Проверяем, что это не стоп-слово
        if word_clean.lower() in stop_words:
            break  # Прерываем, если встретили стоп-слово
        
        # Проверяем, что слово выглядит как имя (начинается с заглавной, только буквы)
        # Имена обычно состоят из букв и могут содержать дефис
        if word_clean[0].isupper() and (word_clean.replace('-', '').isalpha() or 
                                         (len(word_clean) <= 3 and word_clean.isdigit())):
            cleaned_words.append(word_clean)
        elif not cleaned_words:
            # Если еще нет имен, но слово начинается с заглавной - возможно имя
            if word_clean[0].isupper() and word_clean.replace('-', '').isalpha():
                cleaned_words.append(word_clean)
    
    # Если нашли очищенные слова, возвращаем их
    if cleaned_words:
        result = ' '.join(cleaned_words)
        # Ограничиваем длину (имена обычно не длиннее 50 символов)
        if len(result) > 50:
            result = result[:50]
        return result
    
    # Если не удалось очистить, возвращаем первое слово (до пробела/переноса строки)
    first_word = words[0] if words else text
    first_word = first_word.strip('.,!?;:()[]{}"\'-')
    
    # Ограничиваем длину имени
    if len(first_word) > 50:
        first_word = first_word[:50]
    
    return first_word


def filter_and_clean_results(results: List[RecognizerResult], text: str) -> List[RecognizerResult]:
    """Фильтрует результаты распознавания имен по качеству"""
    filtered_results = []
    
    for result in results:
        # Для имен применяем дополнительную фильтрацию
        if result.entity_type in ["PERSON", "PER"]:
            # Получаем захваченный текст
            captured_text = text[result.start:result.end]
            
            # Проверяем длину (имена обычно не длиннее 50 символов)
            if len(captured_text) > 50:
                continue  # Пропускаем слишком длинные результаты
            
            # Проверяем наличие стоп-слов в середине текста
            # (если стоп-слово в начале, это нормально - может быть частью контекста)
            stop_words = ['время', 'место', 'номер', 'телефон', 'адрес']
            text_lower = captured_text.lower()
            
            # Если в тексте есть стоп-слово не в начале - пропускаем
            has_stop_word = False
            for stop_word in stop_words:
                idx = text_lower.find(stop_word)
                if idx > 0:  # Стоп-слово не в начале
                    has_stop_word = True
                    break
            
            if has_stop_word:
                continue  # Пропускаем результаты со стоп-словами
            
            # Проверяем наличие переносов строк (обычно имена не содержат переносы)
            if '\n' in captured_text:
                # Если есть перенос строки, берем только первую часть
                first_line = captured_text.split('\n')[0].strip()
                if first_line and len(first_line) <= 50:
                    # Создаем новый результат только для первой строки
                    new_start = result.start
                    new_end = result.start + len(first_line)
                    new_result = RecognizerResult(
                        entity_type=result.entity_type,
                        start=new_start,
                        end=new_end,
                        score=result.score
                    )
                    filtered_results.append(new_result)
                continue
            
            # Если все проверки пройдены, добавляем результат
            filtered_results.append(result)
        else:
            # Для других типов сущностей оставляем как есть
            filtered_results.append(result)
    
    return filtered_results


# --- Логика управления метками (Stateful per request) ---

class RequestAnonymizer:
    """Класс для обработки одного конкретного запроса"""

    def __init__(self):
        self.counters = {}
        self.mapping = {}
        self.label_map = {
            "PERSON": "ИМЯ",
            "PER": "ИМЯ",
            "PHONE_NUMBER": "ТЕЛЕФОН",
            "INN": "ИНН",
            "PASSPORT": "ПАСПОРТ",
            "LOCATION": "АДРЕС",
            "ADDRESS": "АДРЕС"
        }

    def get_replacement(self, original_text, entity_type):
        ru_label = self.label_map.get(entity_type, entity_type)
        if ru_label not in self.counters:
            self.counters[ru_label] = 1

        placeholder = f"{{{ru_label}_{self.counters[ru_label]}}}"

        # Сохраняем маппинг
        if placeholder not in self.mapping:
            # Для имен применяем очистку текста
            if entity_type in ["PERSON", "PER"]:
                cleaned_text = clean_name_text(original_text)
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


# --- Эндпоинты ---

@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize_text(req: AnonymizeRequest):
    start_time = time.time()

    # Создаем менеджер меток для текущего запроса
    manager = RequestAnonymizer()

    # Анализ
    results = analyzer.analyze(
        text=req.text,
        language='ru',
        entities=["PERSON", "PER", "PHONE_NUMBER", "INN", "PASSPORT", "LOCATION", "ADDRESS"]
    )
    
    # Фильтруем и очищаем результаты распознавания имен
    results = filter_and_clean_results(results, req.text)
    
    # Сортируем по позиции начала
    results = sorted(results, key=lambda x: x.start)

    # Анонимизация
    # Используем default аргумент в lambda, чтобы передать entity_type правильно
    operators = {
        ent: OperatorConfig("custom", {"lambda": lambda x, et=ent: manager.get_replacement(x, et)})
        for ent in ["PERSON", "PER", "PHONE_NUMBER", "INN", "PASSPORT", "LOCATION", "ADDRESS"]
    }

    anonymized_result = anonymizer_engine.anonymize(
        text=req.text,
        analyzer_results=results,
        operators=operators
    )

    return AnonymizeResponse(
        anonymized_text=anonymized_result.text,
        mapping=manager.mapping,
        processing_time=time.time() - start_time
    )


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize_text(req: DeanonymizeRequest):
    text = req.text
    mapping = req.mapping
    # Сортируем ключи по длине (от длинных к коротким), чтобы не сломать замену
    # (например, чтобы {ИМЯ_10} не превратилось в значение {ИМЯ_1} + "0")
    for placeholder in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])

    return DeanonymizeResponse(restored_text=text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)