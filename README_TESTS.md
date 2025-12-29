# Тесты для проверки детекции персональной информации

## Описание

Набор тестов для проверки работы системы анонимизации персональных данных. Тесты проверяют распознавание:
- Имен (PERSON)
- Телефонов (PHONE_NUMBER)
- Адресов (ADDRESS)
- ИНН (INN)
- Паспортов (PASSPORT)

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск тестов

### Вариант 1: Через pytest напрямую

```bash
pytest test_anonymization.py -v
```

### Вариант 2: Через скрипт

```bash
python run_tests.py
```

### Вариант 3: Запуск конкретного теста

```bash
pytest test_anonymization.py::TestPersonalDataDetection::test_prospect_short -v
```

## Требования

Перед запуском тестов убедитесь, что:

1. **API сервер запущен** на `http://localhost:8000`
   ```bash
   python main.py
   ```
   или
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Установлены все зависимости** из `requirements.txt`

## Структура тестов

### Тесты для имен
- `test_simple_name_detection` - Простое имя
- `test_name_with_age` - Имя с возрастом
- `test_multiple_names` - Несколько имен
- `test_name_with_stop_word` - Имя не должно захватывать стоп-слова

### Тесты для телефонов
- `test_phone_with_plus` - Телефон с плюсом
- `test_phone_with_8` - Телефон начинающийся с 8
- `test_phone_with_spaces` - Телефон с пробелами
- `test_phone_with_name` - Телефон с именем

### Тесты для адресов
- `test_full_address` - Полный адрес
- `test_street_with_number` - Улица с номером дома
- `test_prospect_short` - Проспект с сокращением 'пр' (северный пр 69)
- `test_line_address` - Линия с номером дома (4 линия д.41)
- `test_highway_address` - Шоссе с домом
- `test_metro_address` - Адрес с метро
- `test_apartment_details` - Квартира с подъездом
- `test_navigation_address` - Адрес с навигационными указаниями

### Тесты для ИНН
- `test_inn_10_digits` - ИНН физического лица (10 цифр)
- `test_inn_12_digits` - ИНН юридического лица (12 цифр)

### Тесты для паспортов
- `test_passport_with_series` - Паспорт с серией
- `test_passport_format` - Паспорт в формате 12 34 567890

### Комбинированные тесты
- `test_multiple_entities` - Несколько типов сущностей
- `test_example_1` - Пример 1 из examples.py
- `test_example_2` - Пример 2 из examples.py
- `test_example_3` - Пример 3 из examples.py
- `test_example_4` - Пример 4 из examples.py

### Тесты производительности и качества
- `test_processing_time` - Время обработки должно быть разумным
- `test_mapping_consistency` - Консистентность маппинга
- `test_no_false_positives` - Отсутствие ложных срабатываний

## Пример вывода

```
test_anonymization.py::TestPersonalDataDetection::test_simple_name_detection PASSED
test_anonymization.py::TestPersonalDataDetection::test_phone_with_plus PASSED
test_anonymization.py::TestPersonalDataDetection::test_prospect_short PASSED
...
```

## Настройка URL API

По умолчанию тесты используют `http://localhost:8000/anonymize`. 

Чтобы изменить URL, отредактируйте переменную `API_URL` в файле `test_anonymization.py`:

```python
API_URL = "http://your-server:8000/anonymize"
```

## Отладка

Если тесты не проходят:

1. Проверьте, что API сервер запущен и доступен
2. Проверьте логи сервера на наличие ошибок
3. Запустите тесты с флагом `-v` для подробного вывода
4. Используйте `--tb=long` для полного traceback

```bash
pytest test_anonymization.py -v --tb=long
```

