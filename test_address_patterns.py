"""
Быстрый тест для проверки паттернов адресов
"""
import re

# Тестовый текст
test_text = """Здравствуйте, подскажите день рождения 4 года, девочке ,хотим видеть на праздник леди баг, дата празднования 16.04 есть два {ИМЯ_2} места проведения, не определиться ещё одно северный пр 69, а другое {АДРЕС_2} 4 линия д.41. Если заказать аниматора на дату, а место сообщить за два дня до начала. Хотелось бы узнать цену и возможно ли это?"""

# Паттерны
prospect_patterns = [
    r"(?i)(?:северный|южный|восточный|западный|центральный|красный|зеленый|синий|новый|старый)\s+пр\s+\d+",
    r"(?i)[А-ЯЁа-яё]+(?:ый|ий|ой|ая|ое)\s+пр\s+\d+",
]

line_patterns = [
    r"(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия\s+д\.\d+",
    r"(?i)\d+(?:-я|-й|-е|-ая|-ый|-ое)?\s+линия(?:\s+[А-ЯЁа-яё\.]+)?(?:\s*[,]?\s*)?(?:д\.?\s*\d+[А-ЯЁа-яё]?|дом\s*\d+[А-ЯЁа-яё]?)?",
]

print("Тестирование паттернов адресов:")
print("=" * 60)
print(f"Текст: {test_text}")
print("=" * 60)

print("\nПоиск 'северный пр 69':")
for i, pattern in enumerate(prospect_patterns):
    matches = list(re.finditer(pattern, test_text))
    print(f"  Паттерн {i+1}: {pattern}")
    if matches:
        for match in matches:
            print(f"    ✓ Найдено: '{match.group()}' (позиция {match.start()}-{match.end()})")
    else:
        print(f"    ✗ Не найдено")

print("\nПоиск '4 линия д.41':")
for i, pattern in enumerate(line_patterns):
    matches = list(re.finditer(pattern, test_text))
    print(f"  Паттерн {i+1}: {pattern}")
    if matches:
        for match in matches:
            print(f"    ✓ Найдено: '{match.group()}' (позиция {match.start()}-{match.end()})")
    else:
        print(f"    ✗ Не найдено")

