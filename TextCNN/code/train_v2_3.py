#/usr/bin/env python
#coding=utf-8

import time
from data_prov2 import *
from model import *
import tensorflow as tf

vocab_size = 13343
embedding_size = 128
max_document_length = 125


with open(EMBEDDING_MATRIX_PATH, 'rb') as fr:
    embedding_matrix = pickle.load(fr)[0]


def train(text_cnn, embedding_matrix, model_number):
    loss_list = []
    acc_list = []
    min_loss = sys.maxint

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with open(SUB_DATA_PATH + str(model_number),'rb') as fr:
        sub_sample, sub_label = pickle.load(fr)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        for x_data, y_data, batch_nums in batch_iterator(sub_sample,sub_label):
            feed_dict = {
                text_cnn.input_x:x_data,
                text_cnn.input_y:y_data,
                text_cnn.embedding_matrix:embedding_matrix,
                text_cnn.keep_prob:0.5
            }
            _,train_loss,train_acc,current_step = sess.run([text_cnn.optimizer,text_cnn.loss,text_cnn.accuracy,text_cnn.global_step],feed_dict=feed_dict)
            loss_list.append(train_loss)
            acc_list.append(train_acc)

            if len(loss_list) == batch_nums:
                np_loss = np.array(loss_list)
                np_acc = np.array(acc_list)
                loss_mean = np.mean(np_loss)
                acc_mean = np.mean(np_acc)
                print("%s the step is %s , the mean train loss is %s , the  mean train acc is %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),current_step, loss_mean, acc_mean))
                loss_list = []
                acc_list = []
                if loss_mean < min_loss:
                    min_loss = loss_mean
                    save_step = current_step
        tf.train.Saver().save(sess, save_path="../model/mod_v2/NLP"+str(model_number), global_step=save_step)
    return



def main():

    text_cnn_0 = TextCNN(max_document_length=max_document_length,
                         vocab_size=vocab_size,
                         embedding_size=embedding_size,
                         filter_size=[3,4,5],
                         class_nums=2,
                         filter_channel=128,
                         l2_reg_lambda=0.0,
                         learning_rate=1e-3)

    print("训练 text_cnn_3")
    train(text_cnn_0, embedding_matrix, 3)

    return


if __name__ == '__main__':
    main()