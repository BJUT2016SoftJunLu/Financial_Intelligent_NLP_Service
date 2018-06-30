# /usr/bin/env python
# coding=utf-8

import sys
import pickle
import numpy as np
import tensorflow as tf

from train import assign_pretrained_word_embedding
from model_bilstm_cnn import BiLSTM_CNN
from data_pro import load_word_vec, load_tfidf_dict, token_string_as_list, data_mining_features


def predict(input_path, out_path):


    print("加载词表")
    with open("../data/vocabulary",'rb') as f:
        vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = pickle.load(f)

    fin=open(input_path, 'r')
    X1=[]
    X2=[]
    lineno_list=[]
    BLUE_SCORE = []
    count=0

    word_vec_fasttext_dict=load_word_vec('../data/fasttext_fin_model_50.vec')
    word_vec_word2vec_dict = load_word_vec('../data/word2vec.txt')
    tfidf_dict=load_tfidf_dict('../data/atec_nl_sim_tfidf.txt')

    PAD_ID = 0
    UNK_ID = 1
    sentence_len = 39

    vocab_size = len(vocabulary_word2index)
    num_classes = len(vocabulary_index2label)

    for i,line in enumerate(fin):
        lineno, sen1, sen2 = line.strip().split('\t')
        lineno_list.append(lineno)

        x1_list = token_string_as_list(sen1, tokenize_style='word')
        x1 = [vocabulary_word2index.get(x, UNK_ID) for x in x1_list]
        x2_list = token_string_as_list(sen2, tokenize_style='word')
        x2 = [vocabulary_word2index.get(x, UNK_ID) for x in x2_list]

        X1.append(x1)
        X2.append(x2)

        print("padding.....")
        for i in range(len(X1)):
            if len(X1[i]) > sentence_len:
                X1[i] = X1[i][0:sentence_len]
                print(len(X1[i]))
            else:
                X1[i].extend([0.] * (sentence_len - len(X1[i])))

        for i in range(len(X2)):
            if len(X2[i]) > sentence_len:
                X2[i] = X2[i][0:sentence_len]
                print(len(X2[i]))
            else:
                X2[i].extend([0.] * (sentence_len - len(X2[i])))


        features_vector = data_mining_features(i, sen1, sen2, vocabulary_word2index, word_vec_fasttext_dict,word_vec_word2vec_dict, tfidf_dict, n_gram=8)
        features_vector=[float(x) for x in features_vector]
        BLUE_SCORE.append(features_vector)

        length_data_mining_features = len(BLUE_SCORE[0])


    pre_model = BiLSTM_CNN(filter_sizes_list=[2, 3, 4], num_filters=10,
                           num_classes=2,
                           learning_rate=0.001,
                           batch_size=64,
                           decay_steps=1000,
                           decay_rate=1.0,
                           sequence_length=39,
                           vocab_size=13422,
                           embed_size=50,
                           initializer=tf.random_normal_initializer(stddev=0.1),
                           clip_gradients=3.0,
                           decay_rate_big=0.50,
                           model="dual_bilstm_cnn",
                           similiarity_strategy='additive',
                           top_k=3,
                           max_pooling_style="k_max_pooling",
                           length_data_mining_features=length_data_mining_features)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        word2vec_model_path = "../data/fasttext_fin_model_50.vec"
        assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, pre_model, word2vec_model_path,50)
        saver.restore(sess, tf.train.latest_checkpoint('../model/'))

        number_of_test_data = len(X1)
        batch_size = 4
        divide_equally=(number_of_test_data%batch_size==0)
        if divide_equally:
            steps=int(number_of_test_data/batch_size)
        else:
            steps=int(number_of_test_data/batch_size)+1
        logits_result=np.zeros((number_of_test_data,len(vocabulary_index2label)))
        for i in range(steps):
            print("i:",i)
            start=i*batch_size
            if i!=steps or divide_equally:
                end=(i+1)*batch_size
                feed_dict = {pre_model.input_x1: X1[start:end],
                             pre_model.input_x2: X2[start:end],
                             pre_model.input_bluescores: BLUE_SCORE[start:end],
                             pre_model.dropout_keep_prob: 1.0}
            else:
                end=number_of_test_data-(batch_size*int(number_of_test_data%batch_size))
                feed_dict = {pre_model.input_x1: X1[start:end],
                             pre_model.input_x2: X2[start:end],
                             pre_model.input_bluescores: BLUE_SCORE[start:end],
                             pre_model.dropout_keep_prob: 1.0}
            logits_batch=sess.run(pre_model.logits,feed_dict)
            logits_result[start:end]=logits_batch

        file_object = open(out_path, 'a')
        for index, logit in enumerate(logits_result):
            label_index = np.argmax(logit)
            label = vocabulary_index2label[label_index]
            file_object.write(lineno_list[index] + "\t" + label + "\n")
        file_object.close()
        return

def main():
    # predict(sys.argv[1], sys.argv[2])
    predict("../data/test.csv","../data/result.csv")
    return

if __name__ == '__main__':
    main()