from config import Config
from utils.data_utils import read_data, write_scores, write_predictions, get_segment, build_dictionary, remove_low_words
from models.BowModel import BowModel
from models.BocModel import BocModel
from models.Ensemble import Ensemble

if __name__ == "__main__":
    config = Config()

    # read data
    train_set = read_data(config.train_set_file_name)
    dev_set = read_data(config.dev_set_file_name)

    # segmentation
    train_set = get_segment(train_set)
    dev_set = get_segment(dev_set)

    # remove words with low frequency
    dictionary = build_dictionary([train_set, dev_set], config.low_frequency, config.high_frequency)
    train_set = remove_low_words(train_set, dictionary)
    dev_set = remove_low_words(dev_set, dictionary)

    # get predictions
    ensemble_model = Ensemble(config, [BowModel(config), BocModel(config)])
    scores = ensemble_model.test(dev_set)

    # write predictions
    # write_predictions(dev_set, labels, config.result_file_name)
    write_scores(scores, config.result_file_name)
