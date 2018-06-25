# /usr/bin/env python
# coding=utf-8

import jieba
import sys
import logging
import os
import pickle
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import KFold

reload(sys)
sys.setdefaultencoding('utf8')

FILE_PATH_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/atec_nlp_sim_train.csv"))
FILE_PATH_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/atec_nlp_sim_train_add.csv"))

WORD_DICT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/word_dict.pkl"))
WORD2VEC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/word2vec_model"))
EMBEDDING_MATRIX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/embedding_matrix.pkl"))

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/data.pkl"))
KFOLD_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kfold_data.pkl"))

X1_MAX_LENGTH = 75
X2_MAX_LENGTH = 90
MODEL_NUMBER = 7

jieba.load_userdict(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/userdict.txt")))


def sentence_split(sentence):
    return jieba.cut(sentence)

sentences = []


def word2vec_train():


    with open(FILE_PATH_1, 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            new_sentence_1 = ' '.join(sentence_split(example_split[1])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            new_sentence_2 = ' '.join(sentence_split(example_split[2])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            sentences.append(new_sentence_1.split(' '))
            sentences.append(new_sentence_2.split(' '))

    with open(FILE_PATH_2, 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            new_sentence_1 = ' '.join(sentence_split(example_split[1])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            new_sentence_2 = ' '.join(sentence_split(example_split[2])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            sentences.append(new_sentence_1.split(' '))
            sentences.append(new_sentence_2.split(' '))


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    word2vec_model = word2vec.Word2Vec(sentences, size=50, min_count=1,window=3)
    word2vec_model.save(WORD2VEC_PATH)
    return

def dump_matrix_word (word2vec_model):

    embedding_matrix = []
    word_dict = {}
    number = 0
    for key, value in word2vec_model.wv.vocab.items():
        word_dict[key] = number
        embedding_matrix.append(word2vec_model[key])
        number += 1

    _PAD_ = len(embedding_matrix)

    word_dict['_PAD_'] = _PAD_
    embedding_matrix.append(np.zeros(50))

    with open(EMBEDDING_MATRIX_PATH, 'wb') as f:
        pickle.dump([embedding_matrix], f)

    with open(WORD_DICT_PATH, 'wb') as f:
        pickle.dump([word_dict], f)

    return


def data_code(input_file_path, output_file_path):

    with open(WORD_DICT_PATH, 'rb') as f:
        word_dict = pickle.load(f)[0]
    print("the word_dict length is %s"%(len(word_dict)))

    X1_code = []
    X2_code = []
    X1_len = []
    X2_len = []
    Y_label = []

    with open(input_file_path, 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            new_sentence_1 = ' '.join(sentence_split(example_split[1])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            new_sentence_2 = ' '.join(sentence_split(example_split[2])).replace('，', '').replace('。', '').replace('？','').replace('！', '') \
                .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')', '') \
                .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('*', '')

            code_1 = []
            X1_len.append(len(new_sentence_1.split(' ')))
            for word in new_sentence_1.split(' '):
                code_1.append(word_dict[word])

            code_2 = []
            X2_len.append(len(new_sentence_2.split(' ')))
            for word in new_sentence_2.split(' '):
                code_2.append(word_dict[word])

            X1_code.append(code_1)
            X2_code.append(code_2)

            if int(example_split[3]) == 1:
                Y_label.append([1, 0])
            else:
                Y_label.append([0, 1])

    Y_label = np.array(Y_label)
    X1_code = np.array(X1_code)
    X2_code = np.array(X2_code)

    X1_data = []
    X2_data = []

    for x1 in X1_code:
        tmp = np.zeros(X1_MAX_LENGTH)
        tmp.fill(len(word_dict)-1)
        tmp[:len(x1)] = x1
        X1_data.append(tmp)

    for x2 in X2_code:
        tmp = np.zeros(X2_MAX_LENGTH)
        tmp.fill(len(word_dict)-1)
        tmp[:len(x2)] = x2
        X2_data.append(tmp)

    X1_data = np.array(X1_data)
    X2_data = np.array(X2_data)

    X1_len = np.array(X1_len).reshape((-1, 1))
    X2_len = np.array(X2_len).reshape((-1, 1))

    with open(output_file_path, 'wb') as f:
        pickle.dump([X1_data,X2_data,X1_len,X2_len,Y_label], f)
    return



def kfold_dev(file_path):

    with open(file_path, 'rb') as f:
        X1_data, X2_data, X1_len, X2_len, Y_label = pickle.load(f)

    example = np.concatenate((X1_data,X1_len,X2_data,X2_len,Y_label),axis=1)

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(example):
        example_train, example_test = example[train_index], example[test_index]
        x1_train = example_train[:,:X1_MAX_LENGTH]
        x1_train_len = example_train[:,X1_MAX_LENGTH:X1_MAX_LENGTH + 1]
        x2_train = example_train[:, X1_MAX_LENGTH + 1:X1_MAX_LENGTH + 1 + X2_MAX_LENGTH]
        x2_train_len = example_train[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 1:X1_MAX_LENGTH + X2_MAX_LENGTH + 2]
        y_train = example_train[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 2:]

        x1_dev = example_test[:,:X1_MAX_LENGTH]
        x1_dev_len = example_test[:,X1_MAX_LENGTH:X1_MAX_LENGTH + 1]
        x2_dev = example_test[:, X1_MAX_LENGTH + 1:X1_MAX_LENGTH + 1 + X2_MAX_LENGTH]
        x2_dev_len = example_test[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 1:X1_MAX_LENGTH + X2_MAX_LENGTH + 2]
        y_dev = example_test[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 2:]

        yield x1_train, x1_train_len, x2_train, x2_train_len, y_train, x1_dev, x1_dev_len, x2_dev, x2_dev_len, y_dev

def sub_data(file_path):

    with open(file_path, 'rb') as f:
        X1_data, X2_data, X1_len, X2_len, Y_label = pickle.load(f)

    example = np.concatenate((X1_data, X1_len, X2_data, X2_len, Y_label), axis=1)

    x1_train = example[:,:X1_MAX_LENGTH]
    x1_train_len = example[:,X1_MAX_LENGTH:X1_MAX_LENGTH + 1]
    x2_train = example[:, X1_MAX_LENGTH + 1:X1_MAX_LENGTH + 1 + X2_MAX_LENGTH]
    x2_train_len = example[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 1:X1_MAX_LENGTH + X2_MAX_LENGTH + 2]
    y_train = example[:,X1_MAX_LENGTH + X2_MAX_LENGTH + 2:]

    return x1_train, x1_train_len, x2_train, x2_train_len, y_train




def main():
    # print("训练word2vec")
    # word2vec_train()
    # print("加载word2vec")
    # word2vec_model = word2vec.Word2Vec.load(WORD2VEC_PATH)
    # print("dump embedding_matrix and word_dict")
    # dump_matrix_word(word2vec_model)
    # print("编码数据")
    # data_code()

    # data_code()

    FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/"))

    for i in range(MODEL_NUMBER):
        input_file_path = FILE_PATH + "/sub_data" + str(i)
        output_file_path = input_file_path + ".pkl"
        data_code(input_file_path,output_file_path)
    return






if __name__ == '__main__':
    main()