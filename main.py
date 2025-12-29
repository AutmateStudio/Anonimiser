import time
import re
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.entities import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
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

# --- Улучшенный рекогнайзер адресов с комбинацией паттернов и NLP ---

class EnhancedAddressRecognizer(PatternRecognizer):
    """Улучшенный рекогнайзер адресов, использующий паттерны и NLP"""
    
    def __init__(self, nlp_engine, supported_language="ru"):
        # Улучшенные паттерны для различных форматов адресов
        patterns = [
            # Полный адрес с городом, улицей, домом, квартирой
            Pattern(
                name="full_address",
                regex=r"(?i)(?:г|город|г\.)\s+[А-ЯЁа-яё\-]+(?:\s*,\s*)?(?:(?:ул|улица|ул\.|пр-т|проспект|пр\.|наб|набережная|наб\.|пер|переулок|пер\.|ш|шоссе|ш\.|б-р|бульвар|б-р\.)\s+[А-ЯЁа-яё0-9\-\.]+)?(?:\s*,\s*)?(?:(?:д|дом|д\.|стр|строение|стр\.|корп|корпус|корп\.|к|к\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:кв|квартира|кв\.|оф|офис|оф\.)\s*[А-ЯЁа-яё0-9\-]+)?",
                score=0.85
            ),
            # Улица с номером дома (Кавалергардская 12Б, Комсомола 7, Авиаконструкторов 54)
            Pattern(
                name="street_with_number",
                regex=r"(?i)\b[А-ЯЁа-яё][А-ЯЁа-яё\-]+(?:ская|скаяя|ской|ая|ий|ый|ой|ое|ов|а|ы|и|е)\s+(?:д\.?\s*)?\d+[А-ЯЁа-яё]?\b",
                score=0.8
            ),
            # Проспект с прилагательным и номером (северный проспект 69, центральный проспект 10)
            Pattern(
                name="prospect_with_adj",
                regex=r"(?i)\b[А-ЯЁа-яё]+(?:ый|ий|ой|ая|ое)\s+(?:пр-т|проспект|пр\.)\s+\d+[А-ЯЁа-яё]?\b",
                score=0.8
            ),
            # Улица с домом и квартирой (ул. Новая, д. 1, кв. 5)
            Pattern(
                name="street_house_apt",
                regex=r"(?i)(?:ул|улица|ул\.|пр-т|проспект|пр\.|наб|набережная|наб\.|пер|переулок|пер\.|ш|шоссе|ш\.)\s+[А-ЯЁа-яё\-]+(?:\s*,\s*)?(?:(?:д|дом|д\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:к|к\.|корп|корпус|корп\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:кв|квартира|кв\.|оф|офис|оф\.)\s*[А-ЯЁа-яё0-9\-]+)?",
                score=0.75
            ),
            # Проспект с сокращением "пр" (северный пр 69, центральный пр 10)
            Pattern(
                name="prospect_short",
                regex=r"(?i)\b[А-ЯЁа-яё]+(?:ый|ий|ой|ая|ое)\s+пр\s+\d+[А-ЯЁа-яё]?\b",
                score=0.8
            ),
            # Линия с номером дома (4 линия д.41, 1-я линия, 2-я линия В.О. д.5)
            Pattern(
                name="line_address",
                regex=r"(?i)\b\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия(?:\s+[А-ЯЁа-яё\.]+)?(?:\s*,\s*)?(?:(?:д|дом|д\.)\s*\d+[А-ЯЁа-яё]?)?",
                score=0.8
            ),
            # Шоссе с домом (Южное шоссе, д. 53 к 4)
            Pattern(
                name="highway_address",
                regex=r"(?i)\b[А-ЯЁа-яё]+(?:ое|ая|ий|ый|ой)\s+(?:ш|шоссе|ш\.)(?:\s*,\s*)?(?:(?:д|дом|д\.)\s*[А-ЯЁа-яё0-9\-]+)?(?:\s*,\s*)?(?:(?:к|к\.|корп|корпус|корп\.)\s*[А-ЯЁа-яё0-9\-]+)?",
                score=0.8
            ),
            # Квартира с подъездом и этажом (Квартира 23, 3 парадная, Этаж 10)
            Pattern(
                name="apartment_details",
                regex=r"(?i)(?:кв|квартира|кв\.)\s*[А-ЯЁа-яё0-9\-]+(?:\s*,\s*)?(?:(?:\d+\s+)?(?:парадная|подъезд|подъезд\s*\d+))?(?:\s*,\s*)?(?:(?:этаж|эт\.)\s*\d+)?",
                score=0.7
            ),
            # Метро и адрес рядом (метро площадь Ленина, 10 минут от метро)
            Pattern(
                name="metro_address",
                regex=r"(?i)метро\s+[А-ЯЁа-яё\-]+(?:\s+[А-ЯЁа-яё\-]+)*(?:\s*,\s*)?(?:\d+\s+минут\s+от\s+метро)?(?:\s*,\s*)?[А-ЯЁа-яё\-]+\s+\d+",
                score=0.7
            ),
            # Адрес с навигационными указаниями (по навигатору Яндекса - это Кирочная 54К)
            Pattern(
                name="navigation_address",
                regex=r"(?i)(?:по\s+навигатору|по\s+навигации|это)\s+[А-ЯЁа-яё\-]+\s+\d+[А-ЯЁа-яё]?",
                score=0.65
            ),
            # Простой адрес: название улицы/площади с номером
            Pattern(
                name="simple_street",
                regex=r"(?i)\b(?:площадь|пл\.|сквер|парк)\s+[А-ЯЁа-яё\-]+(?:\s+[А-ЯЁа-яё\-]+)*(?:\s*,\s*)?\d+[А-ЯЁа-яё]?",
                score=0.7
            ),
            # Адрес с подъездом (3 парадная, подъезд 38)
            Pattern(
                name="entrance_address",
                regex=r"(?i)(?:подъезд|парадная|парадная\s*\d+)\s*\d+",
                score=0.65
            ),
            # Адрес с этажом (Этаж 10, 10 этаж)
            Pattern(
                name="floor_address",
                regex=r"(?i)(?:этаж|эт\.)\s*\d+",
                score=0.6
            ),
        ]
        
        super().__init__(
            supported_entity="ADDRESS",
            name="RU_ADDRESS_ENHANCED",
            supported_language=supported_language,
            patterns=patterns
        )
        self.nlp_engine = nlp_engine
    
    def enhance_using_nlp(self, text: str, nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """Использует NLP для улучшения распознавания адресов"""
        results = []
        
        # Используем Spacy для распознавания локаций
        doc = nlp_artifacts.doc
        
        # Ищем именованные сущности типа LOC (локация) и GPE (геополитическая сущность)
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"] and len(ent.text) > 3:
                # Проверяем, что это похоже на адрес (содержит цифры или адресные слова)
                text_lower = ent.text.lower()
                address_indicators = ['ул', 'улица', 'проспект', 'пр', 'пр-т', 'шоссе', 'площадь', 'дом', 'квартира', 
                                     'метро', 'район', 'город', 'область', 'д.', 'кв.', 'стр.', 'линия']
                
                # Если содержит индикаторы адреса или цифры - считаем адресом
                if any(indicator in text_lower for indicator in address_indicators) or \
                   any(char.isdigit() for char in ent.text):
                    # Ищем позицию в исходном тексте
                    start = ent.start_char
                    end = ent.end_char
                    
                    # Расширяем контекст - ищем номер дома/квартиры рядом
                    context_start = max(0, start - 50)
                    context_end = min(len(text), end + 50)
                    context = text[context_start:context_end]
                    
                    # Ищем паттерны номеров рядом с локацией
                    number_pattern = r'\d+[А-ЯЁа-яё]?'
                    numbers_before = re.findall(number_pattern, text[max(0, start-20):start])
                    numbers_after = re.findall(number_pattern, text[end:min(len(text), end+20)])
                    
                    # Если есть номера рядом, расширяем границы
                    if numbers_after:
                        # Ищем номер после локации
                        match_after = re.search(r'\s+\d+[А-ЯЁа-яё]?', text[end:end+30])
                        if match_after:
                            end = end + match_after.end()
                    
                    if numbers_before and not any(char.isdigit() for char in ent.text):
                        # Если номер перед локацией, расширяем начало
                        match_before = re.search(r'\d+[А-ЯЁа-яё]?\s+[А-ЯЁа-яё]+', text[max(0, start-30):start+len(ent.text)])
                        if match_before:
                            start = max(0, start - 30) + match_before.start()
                    
                    result = RecognizerResult(
                        entity_type="ADDRESS",
                        start=start,
                        end=end,
                        score=0.75  # Средний score для NLP результатов
                    )
                    results.append(result)
        
        return results
    
    def analyze(self, text: str, entities=None, nlp_artifacts=None):
        """Переопределяем analyze для комбинации паттернов и NLP"""
        # Сначала используем паттерны
        pattern_results = super().analyze(text, entities, nlp_artifacts)
        
        # Затем используем NLP, если доступны nlp_artifacts
        nlp_results = []
        if nlp_artifacts:
            nlp_results = self.enhance_using_nlp(text, nlp_artifacts)
        
        # Объединяем результаты
        all_results = list(pattern_results) + nlp_results
        
        # Удаляем дубликаты и перекрывающиеся результаты
        filtered_results = self._merge_results(all_results)
        
        return filtered_results
    
    def _merge_results(self, results: List[RecognizerResult]) -> List[RecognizerResult]:
        """Объединяет и фильтрует перекрывающиеся результаты"""
        if not results:
            return []
        
        # Сортируем по позиции начала
        sorted_results = sorted(results, key=lambda x: x.start)
        merged = []
        
        for result in sorted_results:
            # Проверяем, не перекрывается ли с уже добавленными
            overlaps = False
            for existing in merged:
                # Если перекрываются, берем результат с большим score
                if not (result.end <= existing.start or result.start >= existing.end):
                    overlaps = True
                    if result.score > existing.score:
                        # Заменяем существующий результат
                        merged.remove(existing)
                        merged.append(result)
                    break
            
            if not overlaps:
                merged.append(result)
        
        return merged


# Создаем улучшенный рекогнайзер адресов
address_recognizer = EnhancedAddressRecognizer(nlp_engine=nlp_engine)

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


def clean_address_text(text: str) -> str:
    """Очищает захваченный текст адреса от лишних данных"""
    if not text:
        return text
    
    # Удаляем переносы строк и лишние пробелы
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Удаляем лишние знаки препинания в конце
    text = text.rstrip('.,!?;:')
    
    # Ограничиваем длину адреса (обычно адреса не длиннее 200 символов)
    if len(text) > 200:
        # Пытаемся найти естественную границу (запятая, точка)
        for delimiter in [',', '.', ';']:
            idx = text[:200].rfind(delimiter)
            if idx > 100:  # Если нашли разделитель не слишком близко к началу
                text = text[:idx].strip()
                break
        else:
            text = text[:200].strip()
    
    return text


def filter_and_clean_results(results: List[RecognizerResult], text: str) -> List[RecognizerResult]:
    """Фильтрует результаты распознавания имен и адресов по качеству"""
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
        elif result.entity_type in ["ADDRESS", "LOCATION"]:
            # Для адресов применяем фильтрацию по длине
            captured_text = text[result.start:result.end]
            
            # Проверяем длину (адреса обычно не длиннее 200 символов)
            if len(captured_text) > 200:
                # Пытаемся обрезать до разумной длины
                # Ищем последнюю запятую или точку в первых 200 символах
                trimmed_text = captured_text[:200]
                for delimiter in [',', '.', ';']:
                    idx = trimmed_text.rfind(delimiter)
                    if idx > 50:  # Если нашли разделитель не слишком близко к началу
                        new_end = result.start + idx
                        result = RecognizerResult(
                            entity_type=result.entity_type,
                            start=result.start,
                            end=new_end,
                            score=result.score
                        )
                        break
            
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
            elif entity_type in ["ADDRESS", "LOCATION"]:
                # Для адресов также применяем очистку
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