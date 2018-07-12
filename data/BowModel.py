from data_utils import get_word_processing_function
import numpy as np


class BowModel:
    def __init__(self, word_indices):
        self.word_indices = word_indices
        self.get_word_id = get_word_processing_function(self.word_indices)
        pass

    def train(self, data_set):
        pass

    def test(self, data_set):
        for answers_of_1q in data_set:
            for qa_pair in answers_of_1q:
                question = [self.get_word_id(i) for i in qa_pair[0]]
                answer = [self.get_word_id(i) for i in qa_pair[1]]
