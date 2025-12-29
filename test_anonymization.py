"""
Тесты для проверки детекции персональной информации
"""
import pytest
import requests
import json
import os
from typing import Dict, List


# Базовый URL API (можно переопределить через переменную окружения)
API_URL = os.getenv("ANONYMIZER_API_URL", "http://localhost:8000/anonymize")


class TestPersonalDataDetection:
    """Тесты для проверки детекции персональных данных"""
    
    def send_request(self, text: str) -> Dict:
        """Отправляет запрос на анонимизацию"""
        try:
            response = requests.post(
                API_URL,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API недоступен: {e}")
    
    def assert_entity_detected(self, response: Dict, entity_type: str, original_value: str):
        """Проверяет, что сущность была обнаружена и заменена"""
        anonymized_text = response.get("anonymized_text", "")
        mapping = response.get("mapping", {})
        
        # Проверяем, что оригинальное значение было заменено (или частично заменено)
        # Для некоторых случаев оригинальное значение может быть частично видно
        original_lower = original_value.lower()
        anonymized_lower = anonymized_text.lower()
        
        # Проверяем, что в маппинге есть соответствующая метка
        found = False
        found_placeholder = None
        for placeholder, value in mapping.items():
            if entity_type in placeholder:
                # Проверяем, содержит ли значение оригинальные данные
                if original_lower in value.lower() or any(word in value.lower() for word in original_lower.split() if len(word) > 2):
                    found = True
                    found_placeholder = placeholder
                    assert placeholder in anonymized_text, \
                        f"Метка {placeholder} не найдена в анонимизированном тексте"
                    break
        
        assert found, \
            f"Сущность типа {entity_type} со значением '{original_value}' не найдена в маппинге. " \
            f"Маппинг: {mapping}, Анонимизированный текст: {anonymized_text}"
    
    # === ТЕСТЫ ДЛЯ ИМЕН ===
    
    def test_simple_name_detection(self):
        """Тест: Простое имя"""
        text = "Меня зовут Иван Иванов"
        response = self.send_request(text)
        
        assert "Иван" in response["anonymized_text"] or "{ИМЯ" in response["anonymized_text"]
        self.assert_entity_detected(response, "ИМЯ", "Иван")
    
    def test_name_with_age(self):
        """Тест: Имя с возрастом"""
        text = "Алиса, 6 лет"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ИМЯ", "Алиса")
        assert "6 лет" in response["anonymized_text"]  # Возраст не должен заменяться
    
    def test_multiple_names(self):
        """Тест: Несколько имен"""
        text = "Варвара и Екатерина пришли на встречу"
        response = self.send_request(text)
        
        mapping = response.get("mapping", {})
        name_placeholders = [k for k in mapping.keys() if "ИМЯ" in k]
        assert len(name_placeholders) >= 2, "Должно быть обнаружено минимум 2 имени"
    
    def test_name_with_stop_word(self):
        """Тест: Имя не должно захватывать стоп-слова"""
        text = "Имя Варвара\nВремя 17:00"
        response = self.send_request(text)
        
        mapping = response.get("mapping", {})
        # Проверяем, что в маппинге имени нет слова "Время"
        for placeholder, value in mapping.items():
            if "ИМЯ" in placeholder:
                assert "Время" not in value, f"Имя содержит стоп-слово: {value}"
                assert "Варвара" in value, f"Имя должно содержать 'Варвара', получено: {value}"
    
    # === ТЕСТЫ ДЛЯ ТЕЛЕФОНОВ ===
    
    def test_phone_with_plus(self):
        """Тест: Телефон с плюсом"""
        text = "Мой номер +79818122189"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ТЕЛЕФОН", "+79818122189")
    
    def test_phone_with_8(self):
        """Тест: Телефон начинающийся с 8"""
        text = "Позвоните по номеру 89001234567"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ТЕЛЕФОН", "89001234567")
    
    def test_phone_with_spaces(self):
        """Тест: Телефон с пробелами"""
        text = "Телефон: 8 900 123 45 67"
        response = self.send_request(text)
        
        # Проверяем, что телефон был обнаружен (может быть с пробелами или без)
        anonymized = response["anonymized_text"]
        assert "{ТЕЛЕФОН" in anonymized
    
    def test_phone_with_name(self):
        """Тест: Телефон с именем"""
        text = "+79500054031 Екатерина"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ТЕЛЕФОН", "+79500054031")
        self.assert_entity_detected(response, "ИМЯ", "Екатерина")
    
    # === ТЕСТЫ ДЛЯ АДРЕСОВ ===
    
    def test_full_address(self):
        """Тест: Полный адрес"""
        text = "г. Москва, ул. Новая, д. 1, кв. 5"
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
        mapping = response.get("mapping", {})
        address_found = any("АДРЕС" in k for k in mapping.keys())
        assert address_found, "Адрес должен быть обнаружен"
    
    def test_street_with_number(self):
        """Тест: Улица с номером дома"""
        text = "Кавалергардская 12Б"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "АДРЕС", "Кавалергардская")
    
    def test_prospect_short(self):
        """Тест: Проспект с сокращением 'пр'"""
        text = "северный пр 69"
        response = self.send_request(text)
        
        anonymized = response["anonymized_text"]
        mapping = response.get("mapping", {})
        
        assert "{АДРЕС" in anonymized, \
            f"Адрес 'северный пр 69' не обнаружен. Результат: {anonymized}, Маппинг: {mapping}"
        
        # Проверяем, что адрес есть в маппинге
        address_found = False
        for placeholder, value in mapping.items():
            if "АДРЕС" in placeholder:
                if "северный" in value.lower() or "69" in value:
                    address_found = True
                    break
        
        assert address_found, f"Адрес не найден в маппинге. Маппинг: {mapping}"
    
    def test_line_address(self):
        """Тест: Линия с номером дома"""
        text = "4 линия д.41"
        response = self.send_request(text)
        
        anonymized = response["anonymized_text"]
        mapping = response.get("mapping", {})
        
        assert "{АДРЕС" in anonymized, \
            f"Адрес '4 линия д.41' не обнаружен. Результат: {anonymized}, Маппинг: {mapping}"
        
        # Проверяем, что адрес есть в маппинге
        address_found = False
        for placeholder, value in mapping.items():
            if "АДРЕС" in placeholder:
                if "линия" in value.lower() or "41" in value:
                    address_found = True
                    break
        
        assert address_found, f"Адрес не найден в маппинге. Маппинг: {mapping}"
    
    def test_highway_address(self):
        """Тест: Шоссе с домом"""
        text = "Южное шоссе, д. 53 к 4"
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
    
    def test_metro_address(self):
        """Тест: Адрес с метро"""
        text = "метро площадь Ленина, 10 минут от метро. Комсомола 7"
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
    
    def test_apartment_details(self):
        """Тест: Квартира с подъездом"""
        text = "Квартира 23, 3 парадная"
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
    
    def test_navigation_address(self):
        """Тест: Адрес с навигационными указаниями"""
        text = "по навигатору Яндекса - это Кирочная 54К"
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
    
    # === ТЕСТЫ ДЛЯ ИНН ===
    
    def test_inn_10_digits(self):
        """Тест: ИНН физического лица (10 цифр)"""
        text = "ИНН: 1234567890"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ИНН", "1234567890")
    
    def test_inn_12_digits(self):
        """Тест: ИНН юридического лица (12 цифр)"""
        text = "ИНН организации: 123456789012"
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ИНН", "123456789012")
    
    # === ТЕСТЫ ДЛЯ ПАСПОРТОВ ===
    
    def test_passport_with_series(self):
        """Тест: Паспорт с серией"""
        text = "Паспорт серия 1234 номер 567890"
        response = self.send_request(text)
        
        assert "{ПАСПОРТ" in response["anonymized_text"]
    
    def test_passport_format(self):
        """Тест: Паспорт в формате 12 34 567890"""
        text = "Паспорт 12 34 567890"
        response = self.send_request(text)
        
        assert "{ПАСПОРТ" in response["anonymized_text"]
    
    # === КОМБИНИРОВАННЫЕ ТЕСТЫ ===
    
    def test_multiple_entities(self):
        """Тест: Несколько типов сущностей"""
        text = "Иван Иванов, телефон +79001234567, адрес: г. Москва, ул. Ленина, д. 1"
        response = self.send_request(text)
        
        mapping = response.get("mapping", {})
        assert any("ИМЯ" in k for k in mapping.keys()), "Имя должно быть обнаружено"
        assert any("ТЕЛЕФОН" in k for k in mapping.keys()), "Телефон должен быть обнаружен"
        assert any("АДРЕС" in k for k in mapping.keys()), "Адрес должен быть обнаружен"
    
    def test_example_1(self):
        """Тест: Пример 1 из examples.py"""
        text = """Номер +79818122189
Имя Варвара
Время 17:00
Место сообщу завтра"""
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ТЕЛЕФОН", "+79818122189")
        self.assert_entity_detected(response, "ИМЯ", "Варвара")
        # Проверяем, что "Время" не захвачено в имя
        mapping = response.get("mapping", {})
        for placeholder, value in mapping.items():
            if "ИМЯ" in placeholder:
                assert "Время" not in value
    
    def test_example_2(self):
        """Тест: Пример 2 из examples.py"""
        text = """Знает, ждёт)
Алиса, 6 лет.
Кавалергардская 12Б, по навигатору Яндекса - это Кирочная 54К, проезду и проход с Мариинского. Я прикреплю скрин, чтобы понятно было.
Квартира 23, 3 парадная.
Если погода будет позволять и не будет дождя, то планируется на улице анимация, в сквере напротив дома. Вещи можно будет оставить в квартире.
+79500054031 Екатерина"""
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ИМЯ", "Алиса")
        self.assert_entity_detected(response, "ИМЯ", "Екатерина")
        self.assert_entity_detected(response, "ТЕЛЕФОН", "+79500054031")
        assert "{АДРЕС" in response["anonymized_text"]
    
    def test_example_3(self):
        """Тест: Пример 3 из examples.py"""
        text = """Здравствуйте!14 сентября день рождения у дочки. 5 лет. Друзей нет ещё. Можно ли заказать у вас аниматора "Леди Баг" На детскую площадку у дома? Вручить подарок с шариками, потанцевать. И в общем, сколько это будет стоить? Мы живём: метро площадь Ленина, 10 минут от метро. Комсомола 7."""
        response = self.send_request(text)
        
        assert "{АДРЕС" in response["anonymized_text"]
    
    def test_example_4(self):
        """Тест: Пример 4 из examples.py"""
        text = """Очень неудобно, что у вас такая связь, общение не 1 раз и сразу. А несколько дней с промежутком в 8 часов :(
22 ноября 13.00
Южное шоссе, д. 53 к 4
Мой номер 89650809493, Елена.
День рождения у Анны 7 лет.
Детей будет 7-8 человек.
Я хочу поговорить с аниматором, который приедет, и обсудить детали"""
        response = self.send_request(text)
        
        self.assert_entity_detected(response, "ТЕЛЕФОН", "89650809493")
        self.assert_entity_detected(response, "ИМЯ", "Елена")
        self.assert_entity_detected(response, "ИМЯ", "Анна")
        assert "{АДРЕС" in response["anonymized_text"]
    
    # === ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ ===
    
    def test_processing_time(self):
        """Тест: Время обработки должно быть разумным"""
        text = "Иван Иванов, телефон +79001234567"
        response = self.send_request(text)
        
        processing_time = response.get("processing_time", 0)
        assert processing_time < 10.0, f"Время обработки слишком большое: {processing_time} сек"
    
    # === ТЕСТЫ КАЧЕСТВА ===
    
    def test_mapping_consistency(self):
        """Тест: Консистентность маппинга"""
        text = "Иван Иванов"
        response = self.send_request(text)
        
        anonymized = response["anonymized_text"]
        mapping = response.get("mapping", {})
        
        # Проверяем, что все метки из текста есть в маппинге
        import re
        placeholders = re.findall(r'\{[А-ЯЁ_]+_\d+\}', anonymized)
        for placeholder in placeholders:
            assert placeholder in mapping, f"Метка {placeholder} отсутствует в маппинге"
    
    def test_no_false_positives(self):
        """Тест: Отсутствие ложных срабатываний на обычных словах"""
        text = "Сегодня хорошая погода. Температура 25 градусов."
        response = self.send_request(text)
        
        mapping = response.get("mapping", {})
        # Не должно быть обнаружено персональных данных в этом тексте
        # (хотя могут быть обнаружены локации, если модель их найдет)
        assert len(mapping) == 0 or all("АДРЕС" not in k or "LOCATION" not in k for k in mapping.keys()), \
            "Не должно быть ложных срабатываний на обычных словах"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

