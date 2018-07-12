import gensim
import numpy as np
from utils.data_utils import get_segment


class BowModel:
    def __init__(self):
        pass

    def train(self, data_set):
        pass

    def test(self, data_set):
        indices, sentences, _ = get_segment(data_set)
        dictionary = gensim.corpora.Dictionary(sentences)
        sentences = [dictionary.doc2bow(sen) for sen in sentences]
        tf_idf = gensim.models.TfidfModel(sentences)
        sentences = tf_idf[sentences]
        lsi_model = gensim.models.LsiModel(sentences, id2word=dictionary, num_topics=1000)
        sentences = lsi_model[sentences]
        index = gensim.similarities.MatrixSimilarity(sentences)
        labels = np.zeros(len(sentences))
        scores = list()
        for i in range(len(indices) - 1):
            que = sentences[indices[i]]
            al_id = indices[i] + 1
            ar_id = indices[i + 1]
            similarities_of_1q = index[que][al_id: ar_id]
            similarities_of_1q /= np.sum(similarities_of_1q)
            scores.extend(similarities_of_1q)
            ans_pos = np.argmax(similarities_of_1q) + (indices[i] + 1)
            labels[ans_pos] = 1
        return labels, scores
