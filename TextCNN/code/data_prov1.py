#/usr/bin/env python
#coding=utf-8

import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

vocab_size = 1133
embedding_size = 128
max_document_length = 493

FILE_PATH_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/atec_nlp_sim_train.csv"))
FILE_PATH_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/atec_nlp_sim_train_add.csv"))
VPROCESSOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/VocabularyProcessor_v1"))
SUB_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/sub_data_v1"))
EMBEDDING_MATRIX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/embedding_matrix_v1"))

def pro_data():

    Y_label = []
    sentences = []

    with open(FILE_PATH_1, 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")

            sentences.append(example_split[1] + "^" + example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1, 0])
            else:
                Y_label.append([0, 1])

    with open(FILE_PATH_2, 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            sentences.append(example_split[1] + "^" + example_split[2])
            if int(example_split[3]) == 1:
                Y_label.append([1, 0])
            else:
                Y_label.append([0, 1])
    Y_label = np.array(Y_label)

    max_document_length = max([len(s) for s in sentences])

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_data = np.array(list(vocab_processor.fit_transform(sentences)))
    vocab_size = len(vocab_processor.vocabulary_)

    vocab_processor.save(VPROCESSOR_PATH)

    return X_data, Y_label, max_document_length, vocab_size



def under_sample(X_data, Y_label):

    neg_num = 18685.0
    under_sample_num = int(neg_num * (10.0 / 6.0))

    pos_samples = []  # 0
    pos_label = []    # [0, 1]
    neg_samples = []  # 1
    neg_label = []    # [1, 0]

    for i in range(len(Y_label)):
        if Y_label[i][0] == 0:
            pos_samples.append(X_data[i])
            pos_label.append(Y_label[i])
        else:
            neg_samples.append(X_data[i])
            neg_label.append(Y_label[i])

    pos_index = [i for i in range(len(pos_samples))]

    for i in range(5):
        sub_sample = []
        sub_label = []
        pos_slice = random.sample(pos_index, under_sample_num)
        for index in pos_slice:
            sub_sample.append(pos_samples[index])
            sub_label.append(pos_label[index])

        for index in range(len(neg_samples)):
            sub_sample.append(neg_samples[index])
            sub_label.append(neg_label[index])

        sub_sample = np.array(sub_sample)
        sub_label = np.array(sub_label)
        with open(SUB_DATA_PATH + str(i), 'wb') as fw:
            pickle.dump([sub_sample, sub_label], fw)
    return


def get_embedding_matrix(vocab_size,embedding_size):
    with tf.Session() as sess:
        embedding_matrix = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0, dtype=tf.float32,)
        with open(EMBEDDING_MATRIX_PATH,'wb') as fw:
            pickle.dump([sess.run(embedding_matrix)],fw)

def batch_iterator(x_train,y_train,batch_size = 64,epoch = 100):

    data_size = x_train.shape
    example = np.concatenate((x_train,y_train),axis=1)
    batch_nums = int(data_size[0]/batch_size) + 1

    for i in range(epoch):
        np.random.shuffle(example)
        x_data = example[:,:data_size[1]]
        y_data = example[:,data_size[1]:]
        for j in range(batch_nums):
            start = j * batch_size
            end = start + batch_size
            if end > data_size[0]:
                yield x_data[start:],y_data[start:], batch_nums
            else:
                yield x_data[start:end],y_data[start:end], batch_nums


def main():
    X_data, Y_label, max_document_length, vocab_size = pro_data()

    under_sample(X_data, Y_label)
    get_embedding_matrix(vocab_size, embedding_size)
    return

if __name__ == '__main__':
    main()