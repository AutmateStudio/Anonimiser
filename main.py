import time
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
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