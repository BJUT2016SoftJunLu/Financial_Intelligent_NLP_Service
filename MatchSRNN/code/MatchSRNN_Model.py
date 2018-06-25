#/usr/bin/env python
#coding=utf-8

from MatchSRNN_MatchTensor import *
from MatchSRNN_SpatialGRU import *
from keras.layers import *

class MyMactchSRNN(object):

    def __init__(self, x1_max_length, x2_max_length, vocab_size, embedding_size, class_nums, learning_rate, channel):

        self.x1_max_length = x1_max_length
        self.x2_max_length = x2_max_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.class_nums = class_nums
        self.learning_rate = learning_rate
        self.channel = channel


        # Input layer
        self.input_x1 = tf.placeholder(tf.int32, shape=[None, self.x1_max_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, self.x2_max_length], name="input_x2")
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.class_nums], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, shape=[None,], name="keep_prob")

        # Embedding layer
        self.embedding_matrix = tf.placeholder(tf.float32, shape=[self.vocab_size, self.embedding_size], name="embedding_matrix")
        self.embedding_input_x1 = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x1)
        self.embedding_input_x2 = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x2)

        # MatchTensor layer
        self.match_tensor = MatchTensor(channel=self.channel)([self.embedding_input_x1, self.embedding_input_x2])
        self.match_tensor_permute = Permute((2, 3, 1))(self.match_tensor)

        # SpatialGRU layer
        self.h_ij = SpatialGRU()(self.match_tensor)

        # dropout layer
        # self.h_ij_drop = Dropout(rate=self.dropout_rate)(self.h_ij)
        self.h_ij_drop = tf.nn.dropout(self.h_ij,keep_prob=self.keep_prob)
        self.prob = Dense(2, activation='softmax', name="prob")(self.h_ij_drop)

        # loss and accuracy layer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prob,labels=self.input_y), name="loss")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob,1),tf.argmax(self.input_y,1)),"float"), name="accuracy")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)


def main():

    model_matchsrnn = MyMactchSRNN(x1_max_length=75,
                                   x2_max_length=90,
                                   vocab_size=13407,
                                   embedding_size=125,
                                   class_nums=2,
                                   learning_rate=1e-3,
                                   channel=3)

if __name__ == '__main__':
    main()