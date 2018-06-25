#/usr/bin/env python
#coding=utf-8



import time
from data_pro import *
from MatchSRNN_Model import *


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


def train(matchsrnn, model_number, batch_size=64):

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/sub_data" + str(model_number) + ".pkl"))
    x1_train, x1_train_len, x2_train, x2_train_len, y_train = sub_data(file_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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
                        matchsrnn.input_x1: x1_train[start:],
                        matchsrnn.input_x2: x2_train[start:],
                        matchsrnn.input_y: y_train[start:],
                        matchsrnn.embedding_matrix: embedding_matrix
                    }
                    _, train_loss, train_acc, current_step = sess.run([matchsrnn.optimizer, matchsrnn.loss, matchsrnn.accuracy,matchsrnn.global_step], feed_dict=feed_dict)
                    loss_train_list.append(train_loss)
                    acc_train_list.append(train_acc)
                else:

                    feed_dict = {
                        matchsrnn.input_x1: x1_train[start:end],
                        matchsrnn.input_x2: x2_train[start:end],
                        matchsrnn.input_y: y_train[start:end],
                        matchsrnn.embedding_matrix: embedding_matrix
                    }
                    _, train_loss, train_acc, current_step = sess.run([matchsrnn.optimizer, matchsrnn.loss, matchsrnn.accuracy,matchsrnn.global_step], feed_dict=feed_dict)
                    loss_train_list.append(train_loss)
                    acc_train_list.append(train_acc)

            np_train_acc = np.array(acc_train_list)
            np_train_loss = np.array(loss_train_list)
            if np.mean(np_train_loss) < min_loss:
                min_loss = np.mean(np_train_loss)
                save_step = current_step
            print("%s the step is %s , the mean train loss is %s , the  mean train acc is %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), current_step, np.mean(np_train_loss),np.mean(np_train_acc)))
            break
        tf.train.Saver().save(sess, save_path="../model/NLP"+str(model_number), global_step=save_step)
    return


def main():

#     print("创建模型")
#     matchpyramid_0 = model_v4.MPyramid(x1_max_length=X1_MAX_LENGTH, x2_max_length=X2_MAX_LENGTH, vocab_size=vocab_size, embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
# =vocab_size,embedding_size=embedding_size, class_nums=2, learning_rate=1e-3)
#
#     print("开始训练 model__0 ....")
#     train(matchpyramid_0, model_number=0)

    return


if __name__ == '__main__':
    main()