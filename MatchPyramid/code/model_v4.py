#/usr/bin/env python
#coding=utf-8
import tensorflow as tf

class MPyramid(object):

    def __init__(self,x1_max_length,x2_max_length,vocab_size,embedding_size,
                 class_nums,learning_rate):

        self.x1_max_length = x1_max_length
        self.x2_max_length = x2_max_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.class_nums = class_nums
        self.learning_rate = learning_rate


        self.input_x1 = tf.placeholder(tf.int32,shape=[None,self.x1_max_length],name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, self.x2_max_length], name="input_x2")
        self.x1_len = tf.placeholder(tf.int32, name='x1_len', shape=(None, ))
        self.x2_len = tf.placeholder(tf.int32, name='x2_len', shape=(None,))
        self.input_y = tf.placeholder(tf.float32,shape=[None,class_nums],name="input_y")
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',shape=(None, self.x1_max_length, self.x2_max_length, 3))

        self.batch_size = tf.shape(self.input_x1)[0]

        # Embedding layer
        self.embedding_matrix = tf.placeholder(tf.float32, shape=[self.vocab_size, self.embedding_size],name="embedding_matrix")
        self.embed1 = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x1)
        self.embed2 = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x2)

        # batch_size * X1_maxlen * X2_maxlen
        self.cross = tf.einsum('abd,acd->abc', self.embed1, self.embed2)
        self.cross_img = tf.expand_dims(self.cross, 3)

        # convolution
        self.w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32) , dtype=tf.float32, shape=[2, 10, 1, 8])
        self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[8])
        # batch_size * X1_maxlen * X2_maxlen * feat_out
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)


        # dynamic pooling
        data1_psize = 3
        data2_psize = 10
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
        self.pool1 = tf.nn.max_pool(self.conv1_expand,
                        [1, self.x1_max_length / data1_psize, self.x2_max_length / data2_psize, 1],
                        [1, self.x1_max_length / data1_psize, self.x2_max_length / data2_psize, 1], "VALID")

        self.fc1 = tf.nn.relu(tf.contrib.layers.linear(tf.reshape(self.pool1, [self.batch_size, data1_psize * data2_psize * 8]), 20))
        self.pred = tf.contrib.layers.linear(self.fc1, 2)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.input_y))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.prob = tf.nn.softmax(self.pred,name="prob")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.input_y,1)),"float"), name="accuracy")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)




