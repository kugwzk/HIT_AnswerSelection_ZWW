import gensim
import numpy as np


class BowModel:
    def __init__(self):
        pass

    def train(self, data_set):
        pass

    def test(self, data_set):
        indices, sentences, _ = data_set
        labels = np.zeros(len(sentences))
        dictionary = gensim.corpora.Dictionary(sentences)
        sentences_bow = [dictionary.doc2bow(sen) for sen in sentences]
        tf_idf = gensim.models.TfidfModel(sentences_bow)
        sentences_tf_idf = tf_idf[sentences_bow]
        lsi_model = gensim.models.LsiModel(sentences_tf_idf, id2word=dictionary, num_topics=2)
        sentences_lsi = lsi_model[sentences_tf_idf]
        similarities = gensim.similarities.MatrixSimilarity(sentences_lsi)
        for i in range(len(indices) - 1):
            que_id = indices[i]
            al_id = indices[i] + 1
            ar_id = indices[i + 1]
            similarities_of_1q = [similarities[que_id, a] for a in range(al_id, ar_id)]
            ans_pos = np.argmax(similarities_of_1q) + (indices[i] + 1)
            labels[ans_pos] = 1
        return labels
