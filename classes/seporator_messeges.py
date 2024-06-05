import re


class MessageSeparator:
    def __init__(self):
        pass

    def predict(self, answer: str) -> int:
        """
        Проверка наличия хотя бы одной даты в ответе.
        1 - дата есть в ответе
        0 - даты нет в ответе
        """

        def extract_dates(text: str) -> set:
            """
            Выделение дат
            """
            # Паттерн для выделения дат:
            # \b\d{3,4}\b - число от 3 до 4 цифр (годы)
            # \b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b - дата в формате дд.мм.гггг, дд/мм/гггг или дд-мм-гггг
            pattern = r"\b\d{3,4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
            matches = re.findall(pattern, text)
            return set(matches)

        answer_dates = extract_dates(answer)

        return int(bool(answer_dates))


# # Usage example:
# if __name__ == "__main__":
#     separator = MessageSeparator()

#     answer = "В каком городе проходил чемпионат мира по хоккею с шайбой в 1936 году?"
#     print(separator.predict(answer))  # Output: 1

