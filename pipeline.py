from vsevolo_de_bert import VsevoloDeBERT
from classes.seporator_messeges import MessageSeparator
from classes.LLM import LLM_predictor
from classes.city_detector import CityExtractor
import gc


def pipeline(model_path: str, data: list, use_NER = True):
    pred = []
    sep_labels = []
    city_labels = []

    # First Step - len check

    for i in data:
        if len(i['answer']) > 100:
            pred.append(1)
            sep_labels.append(-1)
            city_labels.append(-1)
        else:
            pred.append(0)
            sep_labels.append(0)
            city_labels.append(0)
    print ('check1')
    # Second Step - separation
    separator = MessageSeparator()
    if use_NER:
        city_detector = CityExtractor()

    for num, i in enumerate(data):
        if sep_labels[num] != -1:
            sep_labels[num] = separator.predict(i['answer'])
            if use_NER:
                city_labels[num] = city_detector.predict(i['answer'])

    print ('check2')

    del separator
    if use_NER:
        del city_detector
    gc.collect()

    # Third Step - classification
    classifier = VsevoloDeBERT(model_path=model_path)

    for num, i in enumerate(data):
        if sep_labels[num] == 0 and city_labels[num] == 0:
            pred[num] = classifier.predict([i['summary'], i['answer']])

    del classifier
    gc.collect()

    print ('check3')

    # Fourth Step - LLM
    llm = LLM_predictor()

    for num, i in enumerate(data):
        if sep_labels[num] == 1:
            pred[num] = llm.predict([i['summary'], i['question'], i['answer']], 0)

        elif use_NER:
            if city_labels[num] == 1:
                pred[num] = llm.predict([i['summary'], i['question'], i['answer']], 1)

    del llm
    gc.collect()

    return pred
