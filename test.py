import os
import getpass
import sys
import time


import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from model.q2_initialization import xavier_weight_init
import model.window_utils as du
import model.window_ner as ner
from model.utils import data_iterator
from model.model import LanguageModel
from model.config import Config

from THE_model import NERModel

from os import listdir
from os.path import isfile, join







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



def interactive_parser(sess, model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")
        
        
        #sentence = correct_sentence(sentence)
        
        words_raw = sentence.strip().split(" ")
        
        words_cut = [x for x in words_raw if x.lower() not in ('a', 'an', 'the')]
        
        
        if words_raw == ["exit"]:
            break

        preds = model.predict_labels(sess,words_cut)

            
        result = ""
        for word, prediction in zip(words_cut, preds):
            if prediction.lower() in ('a', 'an', 'the'):
                result += prediction.lower() + " "
            result += word + " "
        
        print "output>", result
        

        #to_print = align_data({"input": words_raw, "output": result})
            





def main():
    config = Config()
    model = NERModel(config)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, config.dir_model)

    print "Session restored"


    interactive_parser(sess, model)




if __name__ == "__main__":
    main()
