# /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn

class MyOSScheme():

    def __init__(self,filter_sizes_list,
                 num_filters,
                 num_classes,
                 learning_rate,
                 batch_size,
                 decay_steps,
                 decay_rate,
                 sequence_length,
                 vocab_size,
                 embed_size,
                 initializer = tf.random_normal_initializer(stddev=0.1),
                 clip_gradients = 3.0,
                 decay_rate_big = 0.50,
                 model = 'dual_bilstm_cnn',
                 similiarity_strategy = 'additive',
                 top_k=3,
                 max_pooling_style='k_max_pooling',
                 length_data_mining_features=25):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes_list = filter_sizes_list
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes_list)
        self.clip_gradients = clip_gradients
        self.model = model
        self.similiarity_strategy = similiarity_strategy
        self.max_pooling_style = max_pooling_style
        self.top_k = top_k
        self.length_data_mining_features = length_data_mining_features
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate



        # input placeholder layer
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")
        self.input_bluescores = tf.placeholder(tf.float32, [None, self.length_data_mining_features], name="input_bluescores")
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")
        self.weights = tf.placeholder(tf.float32, [None,], name="weights_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # input variable layer
        self.b1_conv1=tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b1_conv2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b1 = tf.Variable(tf.ones([self.hidden_size]) / 10)
        self.b2 = tf.Variable(tf.ones([self.hidden_size]) / 10)
        self.b3 = tf.Variable(tf.ones([self.hidden_size*2]) / 10)

        self.Embedding = tf.Variable(name="Embedding", initial_value=tf.random_normal(stddev=0.1,shape=[self.vocab_size, self.embed_size]))
        self.W_projection = tf.Variable(name="W_projection", initial_value=tf.random_normal(shape=[self.hidden_size * 2, self.num_classes],stddev=0.1))
        self.b_projection = tf.Variable(name="b_projection",initial_value=tf.random_normal(shape=[self.num_classes]))
        self.W_LR = tf.Variable(name="W_LR",initial_value=tf.random_normal(shape=[self.length_data_mining_features, self.num_classes]))
        self.b_LR = tf.Variable(name="b_LR",initial_value=tf.random_normal(shape=[self.num_classes]))
        self.W_projection_bilstm = tf.Variable(name="W_projection_bilstm", initial_value=tf.random_normal(shape=[self.hidden_size*2, self.num_classes],stddev=0.1))
        self.b_projection_bilstm = tf.Variable(name="b_projection_bilstm", initial_value=tf.random_normal(shape=[self.num_classes]))
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # model layer
        if self.model == 'dual_bilstm':
            self.logits = self.inference_bilstm()
        elif self.model == 'dual_cnn':
            self.logits = self.inference_cnn()
        elif self.model == 'dual_bilstm_cnn':
            self.logits = self.inference_bilstm_cnn()
        else:
            self.logits = self.inference_mix()

        # loss layer
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        print(self.accuracy.shape)

    def inference_mix(self):

        x1_rnn = self.bi_lstm(self.input_x1, 1)
        x2_rnn = self.bi_lstm(self.input_x2, 1, reuse_flag=True)
        x3_rnn = tf.abs(x1_rnn - x2_rnn)
        x4_rnn = tf.multiply(x1_rnn, x2_rnn)

        h_rnn = tf.concat([x1_rnn, x2_rnn, x3_rnn, x4_rnn], axis=1)
        h_rnn = tf.layers.dense(h_rnn, self.hidden_size, use_bias=True, activation=tf.nn.relu)

        h_bluescore = tf.layers.dense(self.input_bluescores, self.hidden_size, use_bias=True)
        h_bluescore = tf.nn.relu(h_bluescore)

        h = tf.concat([h_rnn, h_bluescore], axis=1)
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)

        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        logits = tf.layers.dense(h, self.num_classes, use_bias=False)

        return logits

    def inference_cnn(self):

        h_bluescore = tf.layers.dense(self.input_bluescores, self.hidden_size / 2, use_bias=True)
        h_bluescore = tf.nn.relu(h_bluescore)

        x1 = self.conv_layers(self.input_x1, 1)
        x2 = self.conv_layers(self.input_x2, 1, reuse_flag=True)
        h_cnn = self.additive_attention(x1, x2, self.hidden_size / 2, "cnn_attention")

        h = tf.concat([h_cnn, h_bluescore], axis=1)
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        logits = tf.layers.dense(h, self.num_classes, use_bias=False)

        return logits

    def inference_bilstm(self):

        x1 = self.bi_lstm(self.input_x1, 1)
        x2 = self.bi_lstm(self.input_x2, 2)

        if self.similiarity_strategy == 'multiply':
            x1 = tf.layers.dense(x1, self.hidden_size*2)
            h_bilstm = tf.multiply(x1, x2)
        elif self.similiarity_strategy == 'additive':
            h_bilstm = self.additive_attention(x1, x2, self.hidden_size, "bilstm_attention")

        h_bluescore = tf.layers.dense(self.input_bluescores, self.hidden_size/2, use_bias=True)
        h_bluescore = tf.nn.relu(h_bluescore)

        h = tf.concat([h_bilstm, h_bluescore], axis=1)

        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        logits = tf.layers.dense(h, self.num_classes, use_bias=False)

        return logits

    def inference_bilstm_cnn(self):

        x1_bilstm = self.bi_lstm(self.input_x1, 1)
        x2_bilstm = self.bi_lstm(self.input_x2, 2)

        x1_bilstm = tf.layers.dense(x1_bilstm, self.hidden_size*2)
        h_bilstm = tf.multiply(x1_bilstm, x2_bilstm)

        x1_cnn = self.conv_layers(self.input_x1, 1)
        x2_cnn = self.conv_layers(self.input_x2, 2)


        x1_cnn = tf.layers.dense(x1_cnn, self.num_filters_total)
        x2_cnn = tf.layers.dense(x2_cnn, self.num_filters_total)


        h_cnn = tf.multiply(x1_cnn, x2_cnn)
        h = tf.concat([h_bilstm, h_cnn], axis=1)
        h = tf.layers.dense(h, self.hidden_size*2, activation=tf.nn.tanh)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        logits = tf.matmul(h, self.W_projection_bilstm) + self.b_projection_bilstm

        return logits


    def bi_lstm(self,input_x,name_scope,reuse_flag=False):

        embedded_words = tf.nn.embedding_lookup(self.Embedding, input_x)

        with tf.variable_scope("bi_lstm_" + str(name_scope), reuse=True):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words, dtype = tf.float32)

        feature = tf.concat([hidden_states[0][1],hidden_states[1][1]],axis=1)

        return feature

    def additive_attention(self,x1, x2, dimension_size, vairable_scope):

        g = tf.Variable(name="attention_g",initial_value=tf.sqrt(1.0 / self.hidden_size))
        b = tf.Variable(name="bias",initial_value=tf.zeros(shape=[dimension_size]))
        x1 = tf.layers.dense(x1, dimension_size)
        x2 = tf.layers.dense(x2, dimension_size)
        h = g*tf.nn.relu(x1 + x2 + b)
        return h

    def conv_layers(self,input_x, name_scope, reuse_flag=False):

        embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)
        sentence_embeddings_expanded=tf.expand_dims(embedded_words,-1)


        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes_list):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=False):
                filter = tf.Variable(name="filter-%s" % filter_size, initial_value=tf.random_normal(stddev=0.1,shape=[filter_size, self.embed_size, 1, self.num_filters]))
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                b = tf.Variable(name="b-%s"%filter_size,initial_value=tf.random_normal(shape=[self.num_filters]))
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1,self.num_filters])
                h = tf.transpose(h, [0, 2, 1])
                h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]
                h = tf.reshape(h, [-1, self.num_filters*self.top_k])
                pooled_outputs.append(h)
        h_pool = tf.concat(pooled_outputs, 1)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total*3])
        h = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)
        return h

    def loss(self, l2_lambda=0.0003):
        losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits, weights=self.weights)
        loss_main = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss_main+l2_losses
        return loss

    def train(self):

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
