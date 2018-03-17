import os

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word



class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary

        #dictionaries 
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # dataset
    #filename_train = filename_dev = filename_test = "./data/ner/dev"
    #filename_train = "./data/ner/train"
    #filename_test = filename_dev = "./data/ner/dev"
    filename_dev = "./data/det/dev_preprocessed_articles_lines.txt" 
    filename_test = "./data/det/test_preprocessed_articles_lines.txt"
    filename_train = "./data/det/train_preprocessed_articles_lines.txt"
    max_iter = None # if not None, max number of examples in Dataset

    # general config
    dir_output = "results/window/"
    dir_model  = dir_output + "model.weights/"
    dir_weights = './weights/ner.weights'
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 50


    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    #filename_glove = "../../embeddings/glove.840B.300d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True


    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"


    # training
    embed_size = 50
    batch_size = 64
    label_size = 5
    hidden_size = 100
    max_epochs = 20
    early_stopping = 2
    dropout = 0.9
    lr = 0.001
    l2 = 0.0001
    window_size = 3


    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU


