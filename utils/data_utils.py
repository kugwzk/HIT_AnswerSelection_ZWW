import jieba

UNK = "$UNK$"
PAD = "$PAD$"


def read_data(file_name):
    # data_set format:
    # indices, sentences, labels
    # indices is positions of questions
    num = 0
    indices = list()
    sentences = list()
    labels = list()
    with open(file_name, "r", encoding="utf-8") as fin:
        last_question = None
        for line in fin.readlines():
            s = line.strip().split("\t")
            # s = [i.strip() for i in s]
            if len(s) != 0:
                question = s[0]
                answer = s[1]
                label = None if len(s) == 2 else eval(s[2])
                if last_question is None or question != last_question:
                    indices.append(num)
                    sentences.append(question)
                    labels.append(-1)
                    num += 1
                last_question = question
                sentences.append(answer)
                labels.append(label)
                num += 1
            else:
                indices.append(num)
                break
        if indices[-1] != num:
            indices.append(num)
    return indices, sentences, labels


def get_segment(data_set, stop_words_file_name):
    # load stop words
    stop_words_set = set()
    with open(stop_words_file_name, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            word = line.strip()
            if len(word) != 0:
                stop_words_set.add(word)

    # get word segmentation
    indices, sentences, labels = data_set
    sentences_seg = list()
    for sen in sentences:
        sen = list(jieba.cut(sen))
        sen = [sen[i] for i in range(len(sen)) if sen[i] not in stop_words_set]
        sentences_seg.append(sen)
    return indices, sentences_seg, labels


def remove_low_words(data_set, dictionary):
    indices, sentences, labels = data_set
    dictionary = dictionary.keys()
    sentences_ret = list()
    for sen in sentences:
        sen = [word for word in sen if word in dictionary]
        sentences_ret.append(sen)
    return indices, sentences_ret, labels


def get_processed(data_set):
    indices, sentences, labels = data_set
    pass


def build_dictionary(data_sets, low_frequency):
    words_count = dict()
    for data_set in data_sets:
        indices, sentences, labels = data_set
        for sen in sentences:
            for word in sen:
                if word in words_count.keys():
                    words_count[word] += 1
                else:
                    words_count[word] = 1
    vocabulary = set([word for word in words_count.keys() if words_count[word] > low_frequency])
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
                fout.write("".join(question) + "\t" + "".join(answer) + "\t" + str(label) + "\n")


def write_scores(scores, file_name):
    with open(file_name, "w", encoding="utf-8") as fout:
        for i in scores:
            fout.write(str(i) + '\n')
