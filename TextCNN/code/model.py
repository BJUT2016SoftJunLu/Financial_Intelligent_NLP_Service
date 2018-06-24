#/usr/bin/env python
#coding=utf-8
import tensorflow as tf
class TextCNN(object):

    def __init__(self,max_document_length,vocab_size,embedding_size,
                 filter_size,class_nums,filter_channel,l2_reg_lambda,learning_rate):

        self.max_document_length = max_document_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.class_nums = class_nums
        self.filter_channel = filter_channel
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate

        self.input_x = tf.placeholder(tf.int32,shape=[None,self.max_document_length],name="input_x")
        self.input_y = tf.placeholder(tf.float32,shape=[None,class_nums],name="input_y")
        self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        self.embedding_matrix = tf.placeholder(tf.float32,shape=[self.vocab_size, self.embedding_size],name="embedding_matrix")

        # Embedding layer
        embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)



        # Conver and pool layer
        pool_output = []
        for size in self.filter_size:
            filter_W = tf.Variable(tf.truncated_normal(shape=[size,self.embedding_size,1,self.filter_channel],stddev=0.1))
            # filter = 4-D tensor [filter_height, filter_width, in_channels, out_channels]
            conv = tf.nn.conv2d(embedded_chars_expanded,
                                filter=filter_W,
                                strides=[1,1,1,1],
                                padding="VALID")
            bais = tf.Variable(tf.constant(0.1,shape=[self.filter_channel]))
            filter_output = tf.nn.relu(conv + bais)
            # kszie =  1-D int Tensor of 4 elements [batch_size(1), height_size, width_size, channels(1)]
            pool = tf.nn.max_pool(filter_output,
                                  ksize=[1, self.max_document_length - size + 1,1,1],
                                  strides=[1,1,1,1],
                                  padding="VALID")
            pool_output.append(pool)

        feature_nums = self.filter_channel * len(self.filter_size)
        pool_concat = tf.concat(pool_output, 3)
        pool_reshape = tf.reshape(pool_concat, shape=[-1, feature_nums])

        # dropout layer
        dropout_output = tf.nn.dropout(pool_reshape,keep_prob=self.keep_prob)

        # connected layer
        Weight = tf.Variable(tf.truncated_normal(shape=[feature_nums,self.class_nums],stddev=0.1))
        Bais = tf.Variable(tf.constant(0.1,shape=[self.class_nums]))
        tf.add_to_collection("losses",tf.nn.l2_loss(Weight))
        tf.add_to_collection("losses",tf.nn.l2_loss(Bais))
        conn_output = tf.matmul(dropout_output,Weight) + Bais

        # loss and accuracy layer
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conn_output,labels=self.input_y))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.loss = cross_entropy_loss + tf.add_n(tf.get_collection("losses")) * self.l2_reg_lambda
        self.prob = tf.nn.softmax(conn_output,name="prob")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(conn_output,1),tf.argmax(self.input_y,1)),"float"), name="accuracy")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
