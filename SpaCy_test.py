import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import time

# Предположим, example_4 импортируется или определен здесь
example_4 = "Меня зовут Иван Иванов, мой ИНН 7712345678, телефон +7 900 123-45-67. Прописан: г. Москва, ул. Ленина, дом 5, кв. 12. Паспорт: 4510 123456"

configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "ru", "model_name": "ru_core_news_lg"}]
}
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

registry = RecognizerRegistry()
registry.load_predefined_recognizers(nlp_engine=nlp_engine)

# 1. Регулярное выражение для российских адресов
# Ищет сокращения: г, ул, пр-т, наб, пер, д, корп, стр, кв и последующие названия/номера
address_recognizer = PatternRecognizer(
    supported_entity="ADDRESS",
    name="RU_ADDRESS",
    supported_language="ru",
    patterns=[
        Pattern(
            name="address",
            regex=r"(?i)\b(?:г|ул|пр-т|проспект|наб|пер|д|дом|корп|стр|кв|обл|район)\.?\s+[А-ЯЁа-яё0-9\-\.]+(?:[\s,]+(?:г|ул|пр-т|проспект|наб|пер|д|дом|корп|стр|кв|обл|район)\.?\s+[А-ЯЁа-яё0-9\-\.]+)*",
            score=0.6
        )
    ]
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
    patterns=[
        Pattern(
            name="passport",
            regex=r"(?<!\d)(?:паспорт\s*)?(?:серия\s*)?(?:\d{2}\s?\d{2}|\d{4})[\s\-]?(?:номер\s*)?\d{6}(?!\d)",
            score=0.95
        )
    ]
)

registry.add_recognizer(address_recognizer)
registry.add_recognizer(phone_recognizer)
registry.add_recognizer(inn_recognizer)
registry.add_recognizer(passport_recognizer)

analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
anonymizer_engine = AnonymizerEngine()


class AnonymizationManager:
    def __init__(self):
        self.counters = {}
        self.mapping = {}
        self.label_map = {
            "PERSON": "ИМЯ",
            "PER": "ИМЯ",
            "PHONE_NUMBER": "ТЕЛЕФОН",
            "INN": "ИНН",
            "PASSPORT": "ПАСПОРТ",
            "LOCATION": "АДРЕС",  # Для встроенных локаций spaCy
            "ADDRESS": "АДРЕС"  # Для нашего кастомного рекогнайзера
        }

    def get_replacement(self, original_text, entity_type):
        ru_label = self.label_map.get(entity_type, entity_type)

        if ru_label not in self.counters:
            self.counters[ru_label] = 1

        placeholder = f"{{{ru_label}_{self.counters[ru_label]}}}"

        # Чтобы не дублировать одни и те же сущности в маппинге
        if placeholder not in self.mapping:
            self.mapping[placeholder] = original_text
            self.counters[ru_label] += 1

        return placeholder


manager = AnonymizationManager()

text = example_4

# Добавляем LOCATION и ADDRESS в список сущностей
results = analyzer.analyze(
    text=text,
    language='ru',
    entities=["PERSON", "PER", "PHONE_NUMBER", "INN", "PASSPORT", "LOCATION", "ADDRESS"]
)

# Сортировка для корректной работы анонимизатора
results = sorted(results, key=lambda x: x.start)

time_start = time.time()

# Формируем операторы динамически на основе label_map
operators = {
    entity: OperatorConfig("custom", {"lambda": lambda x, et=entity: manager.get_replacement(x, et)})
    for entity in ["PERSON", "PER", "PHONE_NUMBER", "INN", "PASSPORT", "LOCATION", "ADDRESS"]
}

anonymized_result = anonymizer_engine.anonymize(
    text=text,
    analyzer_results=results,
    operators=operators
)
time_end = time.time()

print("--- АНОНИМИЗИРОВАННЫЙ ТЕКСТ ---")
print(anonymized_result.text)
print(f"Время обработки: {time_end - time_start:.4f} сек")

print("\n--- КАРТА ДАННЫХ ---")
for k, v in manager.mapping.items():
    print(f"{k}: {v}")


def restore(text, mapping):
    # Сортируем ключи по длине (от длинных к коротким), чтобы избежать частичной замены
    for placeholder in sorted(mapping.keys(), key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text


print("\n--- ВОССТАНОВЛЕННЫЙ ТЕКСТ ---")
print(restore(anonymized_result.text, manager.mapping))