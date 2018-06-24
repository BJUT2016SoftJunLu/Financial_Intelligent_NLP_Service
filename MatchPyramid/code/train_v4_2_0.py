#/usr/bin/env python
#coding=utf-8

import time
import tensorflow as tf
import model_v4 as model_v4
from data_prov4 import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

reload(sys)
sys.setdefaultencoding('utf8')

batch_nums = 1442


X1_MAX_LENGTH = 75
X2_MAX_LENGTH = 90
embedding_size = 50
vocab_size = 13407

WORD_DICT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/word_dict.pkl"))
EMBEDDING_MATRIX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/embedding_matrix.pkl"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/data.pkl"))


with open(EMBEDDING_MATRIX_PATH, 'rb') as f:
    embedding_matrix = pickle.load(f)[0]

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


def train(matchpyramid, model_number, batch_size=64):

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/sub_data" + str(model_number) + ".pkl"))
    x1_train, x1_train_len, x2_train, x2_train_len, y_train = sub_data(file_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    x1_train_len = x1_train_len.reshape(1, -1).tolist()[0]
    x2_train_len = x2_train_len.reshape(1, -1).tolist()[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(300):
            min_loss = sys.maxint
            acc_train_list = []
            loss_train_list = []

            batch_nums = int(len(y_train) / batch_size) + 1

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
                        matchpyramid.dpool_index: dynamic_pooling_index(np.array(x1_train_len[start:]),
                                                                        np.array(x2_train_len[start:]),
                                                                        X1_MAX_LENGTH, X2_MAX_LENGTH),
                        matchpyramid.embedding_matrix: embedding_matrix
                    }
                    _, train_loss, train_acc, current_step = sess.run(
                        [matchpyramid.optimizer, matchpyramid.loss, matchpyramid.accuracy,
                         matchpyramid.global_step], feed_dict=feed_dict)
                    loss_train_list.append(train_loss)
                    acc_train_list.append(train_acc)
                else:

                    feed_dict = {
                        matchpyramid.input_x1: x1_train[start:end],
                        matchpyramid.input_x2: x2_train[start:end],
                        matchpyramid.x1_len: np.array(x1_train_len[start:end]),
                        matchpyramid.x2_len: np.array(x2_train_len[start:end]),
                        matchpyramid.input_y: y_train[start:end],
                        matchpyramid.dpool_index: dynamic_pooling_index(np.array(x1_train_len[start:end]),
                                                                        np.array(x2_train_len[start:end]),
                                                                        X1_MAX_LENGTH, X2_MAX_LENGTH),
                        matchpyramid.embedding_matrix: embedding_matrix
                    }
                    _, train_loss, train_acc, current_step = sess.run([matchpyramid.optimizer, matchpyramid.loss, matchpyramid.accuracy,matchpyramid.global_step], feed_dict=feed_dict)
                    loss_train_list.append(train_loss)
                    acc_train_list.append(train_acc)

            np_train_acc = np.array(acc_train_list)
            np_train_loss = np.array(loss_train_list)
            if np.mean(np_train_loss) < min_loss:
                min_loss = np.mean(np_train_loss)
                save_step = current_step
            print("%s the step is %s , the mean train loss is %s , the  mean train acc is %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), current_step, np.mean(np_train_loss),np.mean(np_train_acc)))

        tf.train.Saver().save(sess, save_path="../model/NLP"+str(model_number), global_step=save_step)
    return


def main():

    print("创建模型")
    matchpyramid_0 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size, embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_1 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_2 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_3 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_4 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_5 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
    # matchpyramid_6 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)

    print("开始训练 model__0 ....")
    train(matchpyramid_0, model_number=0)
    # print("开始训练 model__1 ....")
    # train(matchpyramid_1, model_number=1)
    # print("开始训练 model__2 ....")
    # train(matchpyramid_2, model_number=2)
    # print("开始训练 model__3 ....")
    # train(matchpyramid_3, model_number=3)
    # print("开始训练 model__4 ....")
    # train(matchpyramid_4, model_number=4)
    # print("开始训练 model__5 ....")
    # train(matchpyramid_5, model_number=5)
    # print("开始训练 model__6 ....")
    # train(matchpyramid_6, model_number=6)

    return


if __name__ == '__main__':
    main()