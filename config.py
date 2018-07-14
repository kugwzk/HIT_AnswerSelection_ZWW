class Config:
    def __init__(self):
        self.data_dir = "data/"
        self.train_set_file_name = self.data_dir + "training.data"
        self.dev_set_file_name = self.data_dir + "develop.data"
        self.stop_words_file_name = self.data_dir + "stopwords.data"
        self.result_dir = "results/"
        self.result_file_name = self.result_dir + "result.data"
        self.low_frequency = 50
        self.high_frequency = 2000
