from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import requests


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def correct_sentence(string):
    query = 'http://speller.yandex.net/services/spellservice.json/checkText?text=%s' % "+".join(string.split())
    response = requests.get(query)
    mistakes = {element['word']:element['s'][0] for element in response.json()}
    correct_string = ""
    for word in string.split():
        if word not in mistakes.keys():
            correct_string += word + " "
        else:
            correct_string += mistakes[word] + " "
    return correct_string




def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def interactive_parser(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")
        
        
        sentence = correct_sentence(sentence)
        
        words_raw = sentence.strip().split(" ")
        
        words_cut = [x for x in words_raw if x.lower() not in ('a', 'an', 'the')]
        
        
        if words_raw == ["exit"]:
            break

        preds = model.predict(words_cut)
            
        result = ""
        for word, prediction in zip(words_cut, preds):
            if prediction.lower() in ('a', 'an', 'the'):
                result += prediction.lower() + " "
            result += word + " "
        
        to_print = {"output": result}
        
        #to_print = align_data({"input": words_raw, "output": result})

        for key, seq in to_print.items():
            model.logger.info(seq)
            


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session("./results/server_test/model.weights/")

    # create dataset
    test  = CoNLLDataset("../movie-dialogs/benchmark_test.txt", config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    predictions, correct_predictions = model.predict_all(test)


if __name__ == "__main__":
    main()
