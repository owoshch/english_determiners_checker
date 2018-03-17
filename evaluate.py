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

from TESTModel import NERModel, Config

from os import listdir
from os.path import isfile, join


dir_model = './weights/ner.weights'


seq = "I live in White House"


def main():
    print ("Hello world!")
    config = Config()
    model = NERModel(config)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, dir_model)

    #_ , word_to_num, num_to_word = ner.load_wv(
    #  'data/ner/vocab.txt', 'data/ner/wordVectors.txt')

    #tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    #num_to_tag = dict(enumerate(tagnames))
    #tag_to_num = {v:k for k,v in num_to_tag.iteritems()}
    word_to_num = config.vocab_words
    #num_to_word = {v:k for k,v in word_to_num}
    tag_to_num = config.vocab_tags
    num_to_tag = {v:k for k, v in tag_to_num.items()}



    words_raw = seq.split()
    pad = (config.window_size - 1)/2
    words_raw = pad * ["<s>"] + words_raw + pad * ["</s>"]
    print 'words', words_raw
    words = [config.processing_word(w) for w in words_raw]
    print 'preprocessed words', words
    X = []
    for i in range(len(words)):
        if words[i] == "<s>" or words[i] == "</s>":
            continue # skip sentence delimiters
        idxs = [word_to_num[words[ii]]
                for ii in range(i - pad, i + pad + 1)]
        X.append(idxs)

    print X


    '''
    windows = du.input_to_windows(test_string, word_to_num, tag_to_num)

    _, predictions = model.predict(sess, windows)

    predictions = [num_to_tag[tag] for tag in predictions]

    print (predictions)

    result = ""
    for word, prediction in zip(test_string.split(), predictions):
        #if prediction.lower() in ('a', 'an', 'the'):
        if prediction in ('A', 'THE', 'AN'):
            result += prediction.lower() + " "
        result += word + " "

    print (result)
    '''

if __name__ == "__main__":
    main()