"""
Сервис анонимизации персональных данных с использованием DeepPavlov NER
"""
import time
import re
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel

# Импорт DeepPavlov
try:
    from deeppavlov import build_model, configs
    DEEPPAVLOV_AVAILABLE = True
except ImportError:
    print("DeepPavlov не установлен. Установите: pip install deeppavlov")
    DEEPPAVLOV_AVAILABLE = False

app = FastAPI(title="Anonymization Service (DeepPavlov)")

# --- Инициализация DeepPavlov NER модели ---

ner_model = None
if DEEPPAVLOV_AVAILABLE:
    try:
        print("Загрузка DeepPavlov NER модели...")
        # Используем модель ner_rus_bert для русского языка
        ner_model = build_model('ner_rus_bert', download=True)
        print("DeepPavlov NER модель загружена успешно!")
    except Exception as e:
        print(f"Ошибка загрузки DeepPavlov модели: {e}")
        ner_model = None


# --- Паттерны для распознавания персональных данных ---

class PatternRecognizer:
    """Класс для распознавания сущностей по паттернам"""
    
    def __init__(self):
        self.patterns = {
            "PHONE_NUMBER": [
                PatternInfo(
                    regex=r'(\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',
                    score=0.9
                )
            ],
            "INN": [
                PatternInfo(
                    regex=r'\b\d{10}\b',
                    score=0.9
                ),
                PatternInfo(
                    regex=r'\b\d{12}\b',
                    score=0.9
                ),
            ],
            "PASSPORT": [
                PatternInfo(
                    regex=r'(?:паспорт\s*)?(?:серия\s*)?(?:\d{2}\s?\d{2}|\d{4})[\s\-]?(?:номер\s*)?\d{6}',
                    score=0.95
                ),
            ],
            "ADDRESS": [
                # Проспект с сокращением "пр" - МАКСИМАЛЬНЫЙ ПРИОРИТЕТ
                PatternInfo(
                    regex=r"(?i)(?:северный|южный|восточный|западный|центральный|красный|зеленый|синий|новый|старый)\s+пр\s+\d+",
                    score=0.95
                ),
                # Линия с домом через точку
                PatternInfo(
                    regex=r"(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия\s+д\.\d+[А-ЯЁа-яё]?",
                    score=0.95
                ),
                # Проспект с прилагательным
                PatternInfo(
                    regex=r"(?i)[А-ЯЁа-яё]+(?:ый|ий|ой|ая|ое)\s+пр\s+\d+",
                    score=0.9
                ),
                # Линия с номером дома
                PatternInfo(
                    regex=r"(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия(?:\s+[А-ЯЁа-яё\.]+)?(?:\s*[,]?\s*)?(?:д\.?\s*\d+[А-ЯЁа-яё]?|дом\s*\d+[А-ЯЁа-яё]?)?",
                    score=0.9
                ),
                # Полный адрес
                PatternInfo(
                    regex=r"(?i)(?:г|город|г\.)\s+[А-ЯЁа-яё\-]+(?:\s*,\s*)?(?:(?:ул|улица|ул\.|пр-т|проспект|пр\.|наб|набережная|наб\.|пер|переулок|пер\.|ш|шоссе|ш\.|б-р|бульвар|б-р\.)\s+[А-ЯЁа-яё0-9\-\.]+)?(?:\s*,\s*)?(?:(?:д|дом|д\.|стр|строение|стр\.|корп|корпус|корп\.|к|к\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:кв|квартира|кв\.|оф|офис|оф\.)\s*[А-ЯЁа-яё0-9\-]+)?",
                    score=0.85
                ),
                # Улица с номером дома
                PatternInfo(
                    regex=r"(?i)\b[А-ЯЁа-яё][А-ЯЁа-яё\-]+(?:ская|скаяя|ской|ая|ий|ый|ой|ое|ов|а|ы|и|е)\s+(?:д\.?\s*)?\d+[А-ЯЁа-яё]?\b",
                    score=0.8
                ),
                # Шоссе с домом
                PatternInfo(
                    regex=r"(?i)\b[А-ЯЁа-яё]+(?:ое|ая|ий|ый|ой)\s+(?:ш|шоссе|ш\.)(?:\s*,\s*)?(?:(?:д|дом|д\.)\s*[А-ЯЁа-яё0-9\-]+)?",
                    score=0.8
                ),
                # Квартира с подъездом
                PatternInfo(
                    regex=r"(?i)(?:кв|квартира|кв\.)\s*[А-ЯЁа-яё0-9\-]+(?:\s*,\s*)?(?:(?:\d+\s+)?(?:парадная|подъезд|подъезд\s*\d+))?(?:\s*,\s*)?(?:(?:этаж|эт\.)\s*\d+)?",
                    score=0.7
                ),
                # Метро и адрес
                PatternInfo(
                    regex=r"(?i)метро\s+[А-ЯЁа-яё\-]+(?:\s+[А-ЯЁа-яё\-]+)*(?:\s*,\s*)?(?:\d+\s+минут\s+от\s+метро)?(?:\s*,\s*)?[А-ЯЁа-яё\-]+\s+\d+",
                    score=0.7
                ),
            ]
        }
    
    def recognize(self, text: str) -> List[Dict]:
        """Распознает сущности по паттернам"""
        results = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern_info in patterns:
                for match in re.finditer(pattern_info.regex, text):
                    results.append({
                        "entity": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "score": pattern_info.score
                    })
        
        return results


class PatternInfo:
    """Информация о паттерне"""
    def __init__(self, regex: str, score: float):
        self.regex = regex
        self.score = score


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

def recognize_entities_with_deeppavlov(text: str) -> List[Dict]:
    """Распознает сущности используя DeepPavlov NER модель"""
    results = []
    
    if ner_model:
        try:
            # DeepPavlov ожидает список предложений
            # Разбиваем текст на предложения для лучшей обработки
            sentence_endings = re.finditer(r'[.!?]\s+', text)
            sentence_starts = [0]
            sentence_ends = []
            
            for match in sentence_endings:
                sentence_ends.append(match.end())
                sentence_starts.append(match.end())
            sentence_ends.append(len(text))
            
            # Обрабатываем каждое предложение
            for i in range(len(sentence_starts)):
                start_pos = sentence_starts[i]
                end_pos = sentence_ends[i] if i < len(sentence_ends) else len(text)
                sentence = text[start_pos:end_pos].strip()
                
                if not sentence:
                    continue
                
                try:
                    # Применяем модель NER
                    model_result = ner_model([sentence])
                    
                    # Результат может быть в разных форматах
                    if isinstance(model_result, list) and len(model_result) > 0:
                        if isinstance(model_result[0], tuple) and len(model_result[0]) == 2:
                            tokens, tags = model_result[0]
                        elif isinstance(model_result[0], list):
                            tokens = model_result[0]
                            tags = model_result[1] if len(model_result) > 1 else []
                        else:
                            continue
                    else:
                        continue
                    
                    # Обрабатываем результаты BIO тегов
                    current_entity = None
                    current_tokens = []
                    current_start_in_sentence = 0
                    
                    # Восстанавливаем позиции токенов в предложении
                    sentence_lower = sentence.lower()
                    token_positions = []
                    search_pos = 0
                    
                    for token in tokens:
                        token_lower = token.lower()
                        # Ищем токен в предложении
                        pos = sentence_lower.find(token_lower, search_pos)
                        if pos == -1:
                            # Если не нашли точное совпадение, используем приблизительную позицию
                            pos = search_pos
                        token_positions.append((pos, pos + len(token)))
                        search_pos = pos + len(token)
                    
                    # Обрабатываем теги
                    for i, (token, tag, (token_start, token_end)) in enumerate(zip(tokens, tags, token_positions)):
                        # Обрабатываем теги BIO
                        if tag and (tag.startswith('B-') or tag.startswith('I-')):
                            entity_type = tag[2:]  # Убираем префикс B- или I-
                            
                            if tag.startswith('B-') or current_entity != entity_type:
                                # Сохраняем предыдущую сущность если есть
                                if current_entity and current_tokens:
                                    entity_text = ' '.join(current_tokens)
                                    entity_start_in_text = start_pos + current_start_in_sentence
                                    entity_end_in_text = entity_start_in_text + len(entity_text)
                                    
                                    # Для LOC (локаций) расширяем контекст для захвата номеров домов
                                    if current_entity == "LOC":
                                        expanded_text = entity_text
                                        expanded_end = entity_end_in_text
                                        
                                        # Ищем номер дома после локации
                                        context_after = sentence[entity_end_in_text - start_pos:entity_end_in_text - start_pos + 30]
                                        number_match = re.search(r'\s+(?:д\.?\s*)?(\d+[А-ЯЁа-яё]?)', context_after)
                                        if number_match:
                                            expanded_text = entity_text + number_match.group()
                                            expanded_end = entity_end_in_text + len(number_match.group())
                                        
                                        # Ищем номер перед локацией
                                        if not any(char.isdigit() for char in entity_text):
                                            context_before = sentence[max(0, entity_start_in_text - start_pos - 20):entity_start_in_text - start_pos]
                                            number_match_before = re.search(r'(\d+(?:-я|-й|-е|-ая|-ый|-ое)?)\s+', context_before)
                                            if number_match_before:
                                                expanded_text = number_match_before.group(1) + ' ' + expanded_text
                                                expanded_start = entity_start_in_text - len(number_match_before.group(1)) - 1
                                                entity_start_in_text = max(start_pos, expanded_start)
                                        
                                        results.append({
                                            "entity": "ADDRESS",  # LOC -> ADDRESS
                                            "start": entity_start_in_text,
                                            "end": expanded_end,
                                            "text": expanded_text,
                                            "score": 0.9
                                        })
                                    else:
                                        results.append({
                                            "entity": current_entity,
                                            "start": entity_start_in_text,
                                            "end": entity_end_in_text,
                                            "text": entity_text,
                                            "score": 0.85
                                        })
                                
                                # Начинаем новую сущность
                                current_entity = entity_type
                                current_tokens = [token]
                                current_start_in_sentence = token_start
                            else:
                                # Продолжаем текущую сущность
                                current_tokens.append(token)
                        else:
                            # Сохраняем предыдущую сущность если есть
                            if current_entity and current_tokens:
                                entity_text = ' '.join(current_tokens)
                                entity_start_in_text = start_pos + current_start_in_sentence
                                entity_end_in_text = entity_start_in_text + len(entity_text)
                                
                                # Для LOC (локаций) расширяем контекст для захвата номеров домов
                                if current_entity == "LOC":
                                    expanded_text = entity_text
                                    expanded_end = entity_end_in_text
                                    
                                    # Ищем номер дома после локации
                                    context_after = sentence[entity_end_in_text - start_pos:entity_end_in_text - start_pos + 30]
                                    number_match = re.search(r'\s+(?:д\.?\s*)?(\d+[А-ЯЁа-яё]?)', context_after)
                                    if number_match:
                                        expanded_text = entity_text + number_match.group()
                                        expanded_end = entity_end_in_text + len(number_match.group())
                                    
                                    # Ищем номер перед локацией
                                    if not any(char.isdigit() for char in entity_text):
                                        context_before = sentence[max(0, entity_start_in_text - start_pos - 20):entity_start_in_text - start_pos]
                                        number_match_before = re.search(r'(\d+(?:-я|-й|-е|-ая|-ый|-ое)?)\s+', context_before)
                                        if number_match_before:
                                            expanded_text = number_match_before.group(1) + ' ' + expanded_text
                                            expanded_start = entity_start_in_text - len(number_match_before.group(1)) - 1
                                            entity_start_in_text = max(start_pos, expanded_start)
                                    
                                    results.append({
                                        "entity": "ADDRESS",  # LOC -> ADDRESS
                                        "start": entity_start_in_text,
                                        "end": expanded_end,
                                        "text": expanded_text,
                                        "score": 0.9
                                    })
                                else:
                                    results.append({
                                        "entity": current_entity,
                                        "start": entity_start_in_text,
                                        "end": entity_end_in_text,
                                        "text": entity_text,
                                        "score": 0.85
                                    })
                            
                            current_entity = None
                            current_tokens = []
                    
                    # Сохраняем последнюю сущность если есть
                    if current_entity and current_tokens:
                        entity_text = ' '.join(current_tokens)
                        entity_start_in_text = start_pos + current_start_in_sentence
                        entity_end_in_text = entity_start_in_text + len(entity_text)
                        
                        # Для LOC (локаций) расширяем контекст для захвата номеров домов
                        if current_entity == "LOC":
                            # Ищем номера домов рядом с локацией
                            expanded_text = entity_text
                            expanded_end = entity_end_in_text
                            
                            # Ищем номер дома после локации (в пределах 30 символов)
                            context_after = sentence[entity_end_in_text - start_pos:entity_end_in_text - start_pos + 30]
                            number_match = re.search(r'\s+(?:д\.?\s*)?(\d+[А-ЯЁа-яё]?)', context_after)
                            if number_match:
                                expanded_text = entity_text + number_match.group()
                                expanded_end = entity_end_in_text + len(number_match.group())
                            
                            # Ищем номер перед локацией (если локация не содержит цифр)
                            if not any(char.isdigit() for char in entity_text):
                                context_before = sentence[max(0, entity_start_in_text - start_pos - 20):entity_start_in_text - start_pos]
                                number_match_before = re.search(r'(\d+(?:-я|-й|-е|-ая|-ый|-ое)?)\s+', context_before)
                                if number_match_before:
                                    expanded_text = number_match_before.group(1) + ' ' + expanded_text
                                    expanded_start = entity_start_in_text - len(number_match_before.group(1)) - 1
                                    entity_start_in_text = max(start_pos, expanded_start)
                            
                            results.append({
                                "entity": "ADDRESS",  # LOC -> ADDRESS
                                "start": entity_start_in_text,
                                "end": expanded_end,
                                "text": expanded_text,
                                "score": 0.9  # Высокий score для адресов из NER
                            })
                        else:
                            results.append({
                                "entity": current_entity,
                                "start": entity_start_in_text,
                                "end": entity_end_in_text,
                                "text": entity_text,
                                "score": 0.85
                            })
                
                except Exception as e:
                    print(f"Ошибка при обработке предложения '{sentence[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except Exception as e:
            print(f"Ошибка при использовании DeepPavlov модели: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def recognize_entities(text: str) -> List[Dict]:
    """Распознает все сущности в тексте используя DeepPavlov NER и паттерны"""
    all_results = []
    
    # 1. Используем DeepPavlov NER модель
    deeppavlov_results = recognize_entities_with_deeppavlov(text)
    
    # Маппинг типов сущностей из DeepPavlov
    # LOC уже обработан как ADDRESS в recognize_entities_with_deeppavlov
    for result in deeppavlov_results:
        entity_type = result["entity"]
        # PER и ADDRESS уже правильно обработаны
        if entity_type != "ORG":  # Пропускаем организации
            all_results.append(result)
    
    # 2. Используем паттерны для дополнительного распознавания
    pattern_results = pattern_recognizer.recognize(text)
    all_results.extend(pattern_results)
    
    # 3. Объединяем результаты, удаляя дубликаты
    # Сортируем по позиции начала
    all_results = sorted(all_results, key=lambda x: (x["start"], -x["score"]))
    
    # Удаляем перекрывающиеся результаты (оставляем с большим score)
    filtered_results = []
    
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
        "model_loaded": ner_model is not None,
        "deeppavlov_available": DEEPPAVLOV_AVAILABLE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
