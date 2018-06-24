#/usr/bin/env python
#coding=utf-8


import time
import tensorflow as tf
import model_v4 as model_v4
from data_prov4 import *


reload(sys)
sys.setdefaultencoding('utf8')



X1_MAX_LENGTH = 75
X2_MAX_LENGTH = 90
embedding_size = 50
vocab_size = 13407

WORD_DICT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/word_dict.pkl"))
EMBEDDING_MATRIX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/embedding_matrix.pkl"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/data.pkl"))


def dynamic_pooling_index(len1, len2, max_len1, max_len2):
    def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
        stride1 = 1.0 * max_len1 / len1_one
        stride2 = 1.0 * max_len2 / len2_one
        idx1_one = [int(i / stride1) for i in range(max_len1)]
        idx2_one = [int(i / stride2) for i in range(max_len2)]
        mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
        index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
        return index_one

    index = []
    for i in range(len(len1)):
        index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
    return np.array(index)


def train(matchpyramid,batch_size=64):


    with open(EMBEDDING_MATRIX_PATH,'rb') as f:
        embedding_matrix = pickle.load(f)[0]

    for train_number in range(10):
        acc_dev_list = []
        acc_train_list = []
        loss_train_list = []

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/sub_data" + str(train_number) + ".pkl"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for x1_train, x1_train_len, x2_train, x2_train_len, y_train, x1_dev, x1_dev_len, x2_dev, x2_dev_len, y_dev in kfold_dev(file_path):

                x1_train_len = x1_train_len.reshape(1, -1).tolist()[0]
                x2_train_len = x2_train_len.reshape(1, -1).tolist()[0]

                x1_dev_len = x1_dev_len.reshape(1, -1).tolist()[0]
                x2_dev_len = x2_dev_len.reshape(1, -1).tolist()[0]

                batch_nums = int(len(y_train)/batch_size) + 1

                for j in range(batch_nums):
                    start = j * batch_size
                    end = start + batch_size
                    if end > len(y_train):
                        feed_dict = {
                            matchpyramid.input_x1: x1_train[start:],
                            matchpyramid.input_x2: x2_train[start:],
                            matchpyramid.x1_len: np.array(x1_train_len[start:]),
                            matchpyramid.x2_len: np.array(x2_train_len[start:]),
                            matchpyramid.input_y: y_train[start:],
                            matchpyramid.dpool_index: dynamic_pooling_index(np.array(x1_train_len[start:]), np.array(x2_train_len[start:]),X1_MAX_LENGTH, X2_MAX_LENGTH),
                            matchpyramid.embedding_matrix: embedding_matrix
                        }
                        _, train_loss, train_acc, current_step = sess.run([matchpyramid.optimizer, matchpyramid.loss, matchpyramid.accuracy, matchpyramid.global_step],feed_dict=feed_dict)
                        loss_train_list.append(train_loss)
                        acc_train_list.append(train_acc)
                    else:

                        feed_dict = {
                            matchpyramid.input_x1: x1_train[start:end],
                            matchpyramid.input_x2: x2_train[start:end],
                            matchpyramid.x1_len: np.array(x1_train_len[start:end]),
                            matchpyramid.x2_len: np.array(x2_train_len[start:end]),
                            matchpyramid.input_y: y_train[start:end],
                            matchpyramid.dpool_index: dynamic_pooling_index(np.array(x1_train_len[start:end]), np.array(x2_train_len[start:end]),X1_MAX_LENGTH, X2_MAX_LENGTH),
                            matchpyramid.embedding_matrix: embedding_matrix
                        }
                        _, train_loss, train_acc, current_step = sess.run([matchpyramid.optimizer, matchpyramid.loss, matchpyramid.accuracy, matchpyramid.global_step],feed_dict=feed_dict)
                        loss_train_list.append(train_loss)
                        acc_train_list.append(train_acc)

                np_train_acc = np.array(acc_train_list)
                np_train_loss = np.array(loss_train_list)
                print("%s the step is %s , the mean train loss is %s , the  mean train acc is %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), current_step, np.mean(np_train_loss), np.mean(np_train_acc)))

                # dev
                dev_dict = {
                    matchpyramid.input_x1:x1_dev,
                    matchpyramid.input_x2:x2_dev,
                    matchpyramid.x1_len:np.array(x1_dev_len),
                    matchpyramid.x2_len:np.array(x2_dev_len),
                    matchpyramid.input_y:y_dev,
                    matchpyramid.dpool_index:dynamic_pooling_index(np.array(x1_dev_len), np.array(x2_dev_len), X1_MAX_LENGTH, X2_MAX_LENGTH),
                    matchpyramid.embedding_matrix:embedding_matrix
                }

                acc = sess.run(matchpyramid.accuracy, feed_dict=dev_dict)
                acc_dev_list.append(acc)

                print("******************************")
                print("%s the step is %s , the dev acc is %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), current_step, acc))
                print("******************************")

            np_dev_acc = np.array(acc_dev_list)
            print("###### the final mean acc is %s ########" % ( np.mean(np_dev_acc) ))
            tf.train.Saver().save(sess, save_path="../model/NLP"+str(train_number), global_step=current_step)
    return


def main():

    print("创建模型")
    matchpyramid = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH,
                                     x2_max_length=X2_MAX_LENGTH,
                                     vocab_size=vocab_size,
                                     embedding_size=embedding_size,
                                     class_nums=2,
                                     learning_rate=1e-3)

    print("开始训练")
    train(matchpyramid)

    return


if __name__ == '__main__':
    main()