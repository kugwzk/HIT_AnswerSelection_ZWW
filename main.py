from config import Config
from utils.data_utils import read_data, write_scores, write_predictions
from models.BowModel import BowModel

if __name__ == "__main__":
    config = Config()

    train_set = read_data(config.train_set_file_name)
    dev_set = read_data(config.dev_set_file_name)

    model = BowModel(config)
    labels, scores = model.test(dev_set)
    write_predictions(dev_set, labels, config.result_file_name)
    write_scores(scores, config.result_file_name)
