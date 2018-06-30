# /usr/bin/env python
# coding=utf-8

from model_bilstm_cnn import BiLSTM_CNN

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pickle
import random
from weight_boosting import *
from gensim.models import KeyedVectors
small_value = 0.00001

file_object = open('../data/log_predict_error.txt','a')


word2vec_model_path = "../data/fasttext_fin_model_50.vec"

def train(trainX1, trainX2, trainBlueScores, trainY,
          validX1, validX2, validBlueScores, validY,
          testX1, testX2, testBlueScores, testY,
          vocabulary_word2index,
          vocabulary_index2word,
          vocabulary_label2index,
          vocabulary_index2label,
          train_model,
          total_epochs=15,
          batch_size=64):

    vocab_size = len(vocabulary_word2index)


    #2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, train_model, word2vec_model_path, 50)

        number_of_training_data = len(trainX1)
        iteration = 0
        best_acc = 0.60
        best_f1_score = 0.20
        weights_dict = init_weights_dict(vocabulary_label2index)
        for epoch in range(total_epochs):

            trainX1, trainX2, trainBlueScores, trainY = shuffle_data(trainX1, trainX2,trainBlueScores, trainY)

            loss, eval_acc, counter = 0.0, 0.0, 0

            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration+1

                input_x1, input_x2, input_bluescores, input_y = generate_batch_training_data(trainX1, trainX2, trainBlueScores, trainY, number_of_training_data, batch_size)

                weights = get_weights_for_current_batch(input_y, weights_dict)

                feed_dict = {train_model.input_x1: input_x1,
                             train_model.input_x2: input_x2,
                             train_model.input_bluescores:input_bluescores,
                             train_model.input_y:input_y,
                             train_model.weights: np.array(weights),
                             train_model.dropout_keep_prob: 0.5}

                curr_loss, curr_acc, curr_lr, _ = sess.run([train_model.loss, train_model.accuracy, train_model.learning_rate, train_model.train_op], feed_dict)
                loss, eval_acc, counter = loss + curr_loss, eval_acc + curr_acc, counter+1
                if counter % 100 == 0:
                     print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),eval_acc/float(counter),curr_lr))

            eval_loss, eval_accc, f1_scoree, precision, recall, weights_label = do_eval(sess, train_model, validX1, validX2, validBlueScores, validY, vocabulary_index2word)
            weights_dict = get_weights_label_as_standard_dict(weights_label)
            print("label accuracy(used for label weight):==========>>>>", weights_dict)

            # if eval_accc * 1.05 > best_acc and f1_scoree > best_f1_score:
            print("going to save model. eval_f1_score:",f1_scoree,";previous best f1 score:",best_f1_score, ";eval_acc",str(eval_accc),";previous best_acc:",str(best_acc))
            saver.save(sess, "../model/model.ckpt", global_step=epoch)
            best_acc = eval_accc
            best_f1_score = f1_scoree

            if True and (epoch != 0 and (epoch == 2 or epoch == 5 or epoch == 9 or epoch == 13)):
                for i in range(2):
                    print(i, "Going to decay learning rate by half.")
                    sess.run(train_model.learning_rate_decay_half_op)

    test_loss, acc_t, f1_score_t, precision, recall, weights_label = do_eval(sess, train_model, testX1, testX2, testBlueScores, testY, iteration, vocabulary_index2word)
    print("Test Loss:%.3f\tAcc:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f:" % ( test_loss,acc_t,f1_score_t,precision,recall))


def shuffle_data(trainX1, trainX2, trainFeatures, trainY):

    c = list(zip(trainX1,trainX2,trainFeatures,trainY))
    random.shuffle(c)
    trainX1[:], trainX2[:], trainFeatures[:],trainY[:]= zip(*c)
    return trainX1, trainX2,trainFeatures, trainY

def generate_batch_training_data(X1, X2, trainBlueScores, Y, num_data, batch_size):

    index_list_ = random.sample(range(0, num_data), batch_size*5)

    index_list = []
    countt_true = 0
    count_false = 0

    for i,index in enumerate(index_list_):
        if Y[index] == 1 and countt_true < 20:
            index_list.append(index)
            countt_true = countt_true+1
        if Y[index] == 0 and count_false < 44:
            index_list.append(index)
            count_false=count_false+1

    input_x1 = [X1[index] for index in index_list]
    input_x2 = [X2[index] for index in index_list]
    input_bluescore = [trainBlueScores[index] for index in index_list]
    input_y = [Y[index] for index in index_list]
    return input_x1, input_x2, input_bluescore, input_y

def do_eval(sess, textCNN, evalX1, evalX2, evalBlueScores, evalY, vocabulary_index2word):

    number_examples=len(evalX1)

    eval_loss = 0.0
    eval_accc = 0.0
    eval_counter = 0

    eval_true_positive = 0
    eval_false_positive = 0
    eval_true_negative = 0
    eval_false_negative = 0

    batch_size = 1
    weights_label = {}
    weights = np.ones((batch_size))

    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):

        feed_dict = {textCNN.input_x1: evalX1[start:end],
                     textCNN.input_x2: evalX2[start:end],
                     textCNN.input_bluescores:evalBlueScores[start:end],
                     textCNN.input_y:evalY[start:end],
                     textCNN.weights:weights,
                     textCNN.dropout_keep_prob: 1.0}

        curr_eval_loss, curr_accc, logits = sess.run([textCNN.loss, textCNN.accuracy, textCNN.logits], feed_dict)
        true_positive, false_positive, true_negative, false_negative = compute_confuse_matrix(logits[0], evalY[start:end][0])
        write_predict_error_to_file(start, file_object, logits[0], evalY[start:end][0], vocabulary_index2word, evalX1[start:end], evalX2[start:end])
        eval_loss, eval_accc, eval_counter = eval_loss + curr_eval_loss, eval_accc + curr_accc, eval_counter + 1
        eval_true_positive,eval_false_positive = eval_true_positive + true_positive, eval_false_positive + false_positive
        eval_true_negative,eval_false_negative = eval_true_negative + true_negative, eval_false_negative + false_negative
        weights_label = compute_labels_weights(weights_label, logits, evalY[start:end])

    print("true_positive:",eval_true_positive,";false_positive:",eval_false_positive,";true_negative:",eval_true_negative,";false_negative:",eval_false_negative)
    p=float(eval_true_positive)/float(eval_true_positive+eval_false_positive+small_value)
    r=float(eval_true_positive)/float(eval_true_positive+eval_false_negative+small_value)
    f1_score=(2*p*r)/(p+r+small_value)
    print("eval_counter:",eval_counter,";eval_acc:",eval_accc)
    return eval_loss/float(eval_counter),eval_accc/float(eval_counter),f1_score,p,r,weights_label


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path, embed_size):

    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')
    word2vec_dict = {}

    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector/np.linalg.norm(vector)


    word_embedding_2dlist = [[]] * vocab_size
    word_embedding_2dlist[0] = np.zeros(embed_size)
    word_embedding_2dlist[1] = np.zeros(embed_size)
    bound = np.sqrt(1.0) / np.sqrt(vocab_size)
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):
        word = vocabulary_index2word[i]
        embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, embed_size)
            count_not_exist = count_not_exist + 1

    word_embedding_final = np.array(word_embedding_2dlist)

    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    return

def compute_confuse_matrix(logit, label):

    predict = np.argmax(logit)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    if predict == 1 and label == 1:
        true_positive = 1
    elif predict == 1 and label == 0:
        false_positive = 1
    elif predict == 0 and label == 0:
        true_negative = 1
    elif predict == 0 and label == 1:
        false_negative = 1
    return true_positive, false_positive, true_negative, false_negative


def write_predict_error_to_file(index,file_object,logit,label,vocabulary_index2word,x1_index_list,x2_index_list):

    predict = np.argmax(logit)
    if predict!=label:
        x1=[vocabulary_index2word[x] for x in list(x1_index_list[0])]
        x2 = [vocabulary_index2word[x] for x in list(x2_index_list[0])]
        file_object.write(str(index)+"-------------------------------------------------------\n")
        file_object.write("label:"+str(label)+";predict:"+str(predict)+"\n")
        file_object.write("".join(x1)+"\n")
        file_object.write("".join(x2) + "\n")


def main():

    print("加载数据")
    with open("../data/vocabulary",'rb') as f:
        vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label = pickle.load(f)

    with open("../data/data_v1") as f:
        train_data, valid_data, test_data, true_label_percent = pickle.load(f)


    trainX1, trainX2, trainBlueScores, trainY = train_data
    validX1, validX2, validBlueScores, validY = valid_data
    testX1, testX2, testBlueScores, testY = test_data

    length_data_mining_features = len(trainBlueScores[0])


    print("创建模型")
    train_model = BiLSTM_CNN(filter_sizes_list = [2,3,4],
                                     num_filters = 10,
                                     num_classes = 2,
                                     learning_rate = 0.001,
                                     batch_size = 64,
                                     decay_steps = 1000,
                                     decay_rate = 1.0,
                                     sequence_length = 39,
                                     vocab_size = 13422,
                                     embed_size = 50,
                                     initializer = tf.random_normal_initializer(stddev=0.1),
                                     clip_gradients = 3.0,
                                     decay_rate_big = 0.50,
                                     model = "dual_bilstm_cnn",
                                     similiarity_strategy = 'additive',
                                     top_k = 3,
                                     max_pooling_style = "k_max_pooling",
                                     length_data_mining_features = length_data_mining_features)

    print("训练模型")
    train(trainX1, trainX2, trainBlueScores, trainY,
          validX1, validX2, validBlueScores, validY,
          testX1, testX2, testBlueScores, testY,
          vocabulary_word2index,
          vocabulary_index2word,
          vocabulary_label2index,
          vocabulary_index2label,
          train_model,
          total_epochs=15,
          batch_size=64)
    return

if __name__ == '__main__':
    main()

