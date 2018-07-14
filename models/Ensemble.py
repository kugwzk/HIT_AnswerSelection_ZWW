import numpy as np


class Ensemble:
    def __init__(self, config, models):
        self.config = config
        self.models = models

    def train(self, data_set):
        for model in self.models:
            model.train(data_set)

    def test(self, data_set):
        scores = None
        for model in self.models:
            scores = np.array(model.test(data_set)) if scores is None \
                else scores + np.array(model.test(data_set))

        indices, sentences, _ = data_set
        for i in range(len(indices) - 1):
            al_id = indices[i] + 1 - (i + 1)
            ar_id = indices[i + 1] - (i + 1)
            scores[al_id:ar_id] /= np.sum(scores[al_id:ar_id])
        return scores
