import gensim
import numpy as np


class BocModel:
    def __init__(self, config):
        self.config = config

    def save(self):
        pass

    def restore(self):
        pass

    def train(self, data_set):
        pass

    def test(self, data_set):
        indices, sentences, _ = data_set
        dictionary = gensim.corpora.Dictionary(sentences)
        sentences = [list("".join(sen)) for sen in sentences]
        sentences = [dictionary.doc2bow(sen) for sen in sentences]
        tf_idf = gensim.models.TfidfModel(sentences)
        sentences = tf_idf[sentences]
        lsi_model_100 = gensim.models.LsiModel(sentences, id2word=dictionary, num_topics=100)
        lsi_model_200 = gensim.models.LsiModel(sentences, id2word=dictionary, num_topics=200)
        # lda_model = gensim.models.LdaModel(sentences, id2word=dictionary, num_topics=1000)

        scores = list()
        for i in range(len(indices) - 1):
            que = sentences[indices[i]]
            al_id = indices[i] + 1
            ar_id = indices[i + 1]

            # tf-idf similarity
            # index_tf_idf = gensim.similarities.MatrixSimilarity(sentences[(al_id - 1):ar_id])
            # similarities_of_1q = index_tf_idf[que][1: ar_id - al_id + 1]

            # lsi similarity
            sens_lsi = lsi_model_100[sentences[(al_id - 1):ar_id]]
            index_lsi = gensim.similarities.MatrixSimilarity(sens_lsi[0:ar_id - al_id + 1])
            similarities_of_1q = index_lsi[sens_lsi[0]][1: ar_id - al_id + 1]

            sens_lsi = lsi_model_200[sentences[(al_id - 1):ar_id]]
            index_lsi = gensim.similarities.MatrixSimilarity(sens_lsi[0:ar_id - al_id + 1])
            similarities_of_1q += index_lsi[sens_lsi[0]][1: ar_id - al_id + 1]

            # lda similarity
            # sens_lda = lda_model[sentences[(al_id - 1):ar_id]]
            # index_lda = gensim.similarities.MatrixSimilarity(sens_lda[0:ar_id - al_id + 1])
            # similarities_of_1q += index_lda[sens_lda[0]][1: ar_id - al_id + 1]

            similarities_of_1q /= np.sum(similarities_of_1q)
            scores.extend(similarities_of_1q)
        return scores
