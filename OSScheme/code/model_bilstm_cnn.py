# /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn

class BiLSTM_CNN():

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
        embedded_words_x1 = tf.nn.embedding_lookup(self.Embedding, self.input_x1)
        with tf.variable_scope("input_x1"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            outputs_x1, hidden_states_x1 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words_x1, dtype = tf.float32)

        feature_x1 = tf.concat([hidden_states_x1[0][1],hidden_states_x1[1][1]],axis=1)
        x1_bilstm = tf.layers.dense(feature_x1, self.hidden_size * 2)

        embedded_words_x2 = tf.nn.embedding_lookup(self.Embedding, self.input_x2)
        with tf.variable_scope("input_x2"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            outputs_x2, hidden_states_x2 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words_x2, dtype = tf.float32)

        x2_bilstm = tf.concat([hidden_states_x2[0][1],hidden_states_x2[1][1]],axis=1)

        h_bilstm = tf.multiply(x1_bilstm, x2_bilstm)

        embedded_words_x1 = tf.nn.embedding_lookup(self.Embedding,self.input_x1)
        sentence_embeddings_expanded_x1=tf.expand_dims(embedded_words_x1,-1)
        pooled_outputs_x1 = []
        for i,filter_size in enumerate(self.filter_sizes_list):
            filter = tf.Variable(initial_value=tf.random_normal(stddev=0.1,shape=[filter_size, self.embed_size, 1, self.num_filters]))
            conv = tf.nn.conv2d(sentence_embeddings_expanded_x1, filter, strides=[1, 1, 1, 1], padding="VALID")
            b = tf.Variable(initial_value=tf.random_normal(shape=[self.num_filters]))
            h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
            h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1,self.num_filters])
            h = tf.transpose(h, [0, 2, 1])
            h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]
            h = tf.reshape(h, [-1, self.num_filters*self.top_k])
            pooled_outputs_x1.append(h)
        h_pool_x1 = tf.concat(pooled_outputs_x1, 1)
        h_pool_flat_x1 = tf.reshape(h_pool_x1, [-1, self.num_filters_total*3])
        h_x1 = tf.nn.dropout(h_pool_flat_x1, keep_prob=self.dropout_keep_prob)
        x1_cnn = tf.layers.dense(h_x1, self.num_filters_total)

        embedded_words_x2 = tf.nn.embedding_lookup(self.Embedding,self.input_x2)
        sentence_embeddings_expanded_x2=tf.expand_dims(embedded_words_x2,-1)
        pooled_outputs_x2 = []
        for i,filter_size in enumerate(self.filter_sizes_list):
            filter = tf.Variable(initial_value=tf.random_normal(stddev=0.1,shape=[filter_size, self.embed_size, 1, self.num_filters]))
            conv = tf.nn.conv2d(sentence_embeddings_expanded_x2, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            b = tf.Variable(initial_value=tf.random_normal(shape=[self.num_filters]))
            h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
            h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1,self.num_filters])
            h = tf.transpose(h, [0, 2, 1])
            h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]
            h = tf.reshape(h, [-1, self.num_filters*self.top_k])
            pooled_outputs_x2.append(h)
        h_pool_x2 = tf.concat(pooled_outputs_x2, 1)
        h_pool_flat_x2 = tf.reshape(h_pool_x2, [-1, self.num_filters_total*3])
        h_x2 = tf.nn.dropout(h_pool_flat_x2, keep_prob=self.dropout_keep_prob)
        x2_cnn = tf.layers.dense(h_x2, self.num_filters_total)

        h_cnn = tf.multiply(x1_cnn, x2_cnn)
        h = tf.concat([h_bilstm, h_cnn], axis=1)
        h = tf.layers.dense(h, self.hidden_size * 2, activation=tf.nn.tanh)
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        self.logits = tf.matmul(h, self.W_projection_bilstm) + self.b_projection_bilstm

        # loss layer
        l2_lambda = 0.0003
        losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits, weights=self.weights)
        loss_main = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        self.loss = loss_main + l2_losses

        # train layer
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        self.correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y, name="correct_prediction")
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")
