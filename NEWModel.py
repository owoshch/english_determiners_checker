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


from model.data_utils import minibatches, pad_sequences, get_chunks
from model.general_utils import Progbar
from model.model import Model



class NERModel(LanguageModel):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, (None, self.config.window_size))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.label_size))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        print ('placeholders are added')




