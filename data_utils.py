import jieba


def read_data(file_name):
    # data_set format:
    # [
    #   [(Q1, A1-1, label), (Q1, A1-2, label), (Q1, A1-3, label), (Q1, A1-4, label)...]
    #   [(Q2, A2-1, label), (Q2, A2-2, label), (Q2, A2-3, label), (Q2, A2-4, label)...]
    #   ...
    # ]
    data_set = list()
    with open(file_name, "r", encoding="utf-8") as fin:
        last_question = None
        answers_of_1q = list()
        for line in fin.readlines():
            s = line.strip().split()
            if len(s) != 0:
                question = s[0]
                answer = s[1]
                label = None if len(s) == 2 else s[2]
                if last_question is None or question == last_question:
                    answers_of_1q.append((question, answer, label))
                else:
                    data_set.append(answers_of_1q)
                    answers_of_1q = list()
                    last_question = question
            else:
                if len(answers_of_1q) != 0:  # solve the blank line
                    data_set.append(answers_of_1q)
                    answers_of_1q = list()
                    last_question = None
        if len(answers_of_1q) != 0:  # solve end of file
            data_set.append(answers_of_1q)
    return data_set


def get_segment(data_set):
    data_set_seg = list()
    for answers_of_1q in data_set:
        answers_of_1q_seg = list()
        for qa_pair in answers_of_1q:
            question = jieba.cut(qa_pair[0])
            answer = jieba.cut(qa_pair[1])
            label = qa_pair[2]
            answers_of_1q_seg.append((question, answer, label))
        data_set_seg.append(answers_of_1q_seg)
    return data_set_seg
