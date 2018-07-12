from config import Config
from utils.data_utils import read_data, get_segment, write_predictions
from models.BowModel import BowModel

if __name__ == "__main__":
    config = Config()

    train_set = read_data(config.train_set_file_name)
    train_set = get_segment(train_set)
    dev_set = read_data(config.dev_set_file_name)
    dev_set = get_segment(dev_set)

    model = BowModel()
    predictions = model.test(dev_set)
    write_predictions(predictions)
