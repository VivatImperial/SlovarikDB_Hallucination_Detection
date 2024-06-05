import os
from slovnet import NER
from navec import Navec

class CityExtractor:
    def __init__(self, path_navec = "seporator messages/navec_news_v1_1B_250K_300d_100q (1).tar", path_ner = "seporator messages/slovnet_ner_news_v1.tar"):
        if not os.path.exists(path_navec):
            raise FileNotFoundError(f"Navec model not found at {path_navec}")
        if not os.path.exists(path_ner):
            raise FileNotFoundError(f"NER model not found at {path_ner}")
        
        self.navec = Navec.load(path_navec)
        self.ner = NER.load(path_ner)
        self.ner.navec(self.navec)

    def predict(self, text: str) -> int:
        """
        Проверка наличия хотя бы одного города в тексте.
        1 - город есть в тексте
        0 - города нет в тексте
        """
        if not text.strip():
            return 0

        markup = self.ner(text)
        cities = {text[span.start:span.stop] for span in markup.spans if span.type == 'LOC'}

        return int(bool(cities))


# # Usage example:
# if __name__ == "__main__":
#     path_navec = "./Pipeline/SeporatorMesseges/navec_news_v1_1B_250K_300d_100q (1).tar"
#     path_ner = "./Pipeline/SeporatorMesseges/slovnet_ner_news_v1.tar"
    
#     city_extractor = CityExtractor(path_navec, path_ner)

#     answer = "В каком городе проходил чемпионат мира по хоккею с шайбой в 1936 году?"
#     print(city_extractor.predict(answer))  # Output: 1

#     # Чтение данных
#     df = pd.read_csv(r"C:\Users\Mi\Downloads\Telegram Desktop\prob_all_sintetic.csv")

#     # Применение функции extract_cities к каждому ряду
#     df['is_city'] = df.apply(lambda row: city_extractor.predict(row['answer']), axis=1)

#     # Сохранение результата в новый CSV файл
#     df.to_csv(r"C:\Users\Mi\Downloads\Telegram Desktop\prob_all_sintetic_with_is_city.csv", index=False)

#     print("Файл сохранен успешно.")
