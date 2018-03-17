from model.data_utils import CoNLLDataset
from NEWModel import NERModel
from model.config import Config



def main():
    # create instance of config
    config = Config()
    print ('config is loaded')
    model = NERModel(config)
    print ('model is loaded')
    model.build()
    print ('model is built')




if __name__ == "__main__":
    main()
