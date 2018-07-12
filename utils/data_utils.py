import jieba

UNK = "$UNK$"
PAD = "$PAD$"


def read_data(file_name):
    # data_set format:
    # indices, sentences, labels
    # indices is positions of questions
    num = 0
    indices = [0]
    sentences = list()
    labels = list()
    with open(file_name, "r", encoding="utf-8") as fin:
        last_question = None
        for line in fin.readlines():
            s = line.strip().split()
            if len(s) != 0:
                question = s[0]
                answer = s[1]
                label = None if len(s) == 2 else s[2]
                if last_question is not None and question != last_question:
                    indices.append(num)
                    sentences.append(question)
                    labels.append(-1)
                    last_question = question
                sentences.append(answer)
                labels.append(label)
                num += 1
            else:
                indices.append(num)
                break
    return indices, sentences, labels


def get_segment(data_set):
    indices, sentences, labels = data_set
    sentences_seg = list()
    for sen in sentences:
        sen = jieba.cut(sen)
        sentences_seg.append(sen)
    return indices, sentences_seg, labels


def get_processed(data_set):
    indices, sentences, labels = data_set
    pass


def build_dictionary(data_set):
    vocabulary = set()
    indices, sentences, labels = data_set
    for sen in sentences:
        vocabulary.update(sen)
    vocabulary.add(PAD)
    vocabulary.add(UNK)
    vocabulary = list(vocabulary)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    return word_indices


def get_word_processing_function(word_indices):
    def f(word):
        if word in word_indices.keys():
            return word_indices[word]
        return word_indices[UNK]

    return f


def write_predictions(data_set, labels, file_name):
    indices, sentences, _ = data_set
    with open(file_name, "w", encoding="utf-8") as fout:
        for i in range(len(indices) - 1):
            que_id = indices[i]
            al_id = indices[i] + 1
            ar_id = indices[i + 1]
            question = sentences[que_id]
            for j in range(al_id, ar_id):
                answer = sentences[j]
                label = labels[j]
                fout.write(question + "\t" + answer + "\t" + str(label) + "\n")
