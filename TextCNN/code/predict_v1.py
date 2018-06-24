#/usr/bin/env python
#coding=utf-8

import sys
from data_prov1 import *
import tensorflow as tf

vocab_size = 1133
embedding_size = 128
max_document_length = 493


with open(EMBEDDING_MATRIX_PATH, 'rb') as fr:
    embedding_matrix = pickle.load(fr)[0]


class Predict:

    def __init__(self,model_number,step_number):

        self.graph = tf.Graph()
        with self.graph.as_default():
             self.saver = tf.train.import_meta_graph("../model/mod_v1/NLP" + str(model_number) + "-" +str(step_number) + ".meta")#创建恢复器

        self.sess=tf.Session(graph=self.graph)
        with self.sess.as_default():
             with self.graph.as_default():
                self.saver.restore(self.sess,"../model/mod_v1/NLP" + str(model_number) + "-" +str(step_number))

                self.input_x = self.graph.get_tensor_by_name('input_x:0')
                self.input_y = self.graph.get_tensor_by_name('input_y:0')
                self.embedding_matrix = self.graph.get_tensor_by_name('embedding_matrix:0')
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
                self.prob = self.graph.get_operation_by_name('prob').outputs[0]

    def predict(self, INPUT_PATH, batch_size=64):

        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(VPROCESSOR_PATH)

        line_number = []
        sentences = []
        predict_result = []

        with open(INPUT_PATH, 'r') as f:
            for line in f.readlines():
                example_split = line.split("\t")
                line_number.append(int(example_split[0]))
                sentences.append(example_split[1] + "^" + example_split[2])

        predict_sentence = np.array(sentences)
        data_size = predict_sentence.shape
        batch_nums = int(data_size[0] / batch_size) + 1

        for j in range(batch_nums):
            start = j * batch_size
            end = start + batch_size
            print(start, end)

            if end > data_size[0]:
                if start == data_size[0]:
                    break
                x_test = np.array(list(vocab_processor.transform(predict_sentence[start:])))
                test_dict = {
                    self.input_x:x_test,
                    self.embedding_matrix:embedding_matrix,
                    self.keep_prob:1.0
                }
                test_prob = self.sess.run(self.prob, feed_dict=test_dict)
                for index in np.argmax(test_prob, 1):
                    if index == 0:
                        predict_result.append(1)
                    else:
                        predict_result.append(0)
            else:
                x_test = np.array(list(vocab_processor.transform(predict_sentence[start:end])))
                test_dict = {
                    self.input_x: x_test,
                    self.embedding_matrix: embedding_matrix,
                    self.keep_prob: 1.0
                }
                test_prob = self.sess.run(self.prob, feed_dict=test_dict)
                for index in np.argmax(test_prob, 1):
                    if index == 0:
                        predict_result.append(1)
                    else:
                        predict_result.append(0)

        self.sess.close()
        return predict_result


def bagging(line_number, total_predict, OUTPUT_PATH):
    line_number = np.array(line_number)
    total_predict = np.array(total_predict)
    total_predict = np.sum(total_predict, axis=0)

    result = []
    for pre in total_predict:
        if pre >= 3:
            result.append(1)
        else:
            result.append(0)

    result = np.array(result)
    line_number = line_number.reshape((len(line_number), 1))
    result = result.reshape((len(result), 1), )
    result = np.concatenate([line_number, result], axis=1)
    with open(OUTPUT_PATH, 'w') as f:
        for line in result:
            f.write(line[0] + "\t" + line[1] + "\n")
    return


def main():

    args = sys.argv

    # args = ["","../data/test_data_one","../data/test_output"]
    line_number = []
    total_predict = []

    with open(args[1], 'r') as f:
        for line in f.readlines():
            example_split = line.split("\t")
            line_number.append(example_split[0])

    predict_0 = Predict(model_number=0, step_number=3115)
    predict_1 = Predict(model_number=1, step_number=3115)
    predict_2 = Predict(model_number=2, step_number=3115)
    predict_3 = Predict(model_number=3, step_number=3115)
    predict_4 = Predict(model_number=4, step_number=3115)
    predict_5 = Predict(model_number=5, step_number=3115)
    predict_6 = Predict(model_number=6, step_number=3115)
    predict_7 = Predict(model_number=7, step_number=3115)
    predict_8 = Predict(model_number=8, step_number=3115)



    res_0 = predict_0.predict(args[1])
    res_1 = predict_1.predict(args[1])
    res_2 = predict_2.predict(args[1])
    res_3 = predict_3.predict(args[1])
    res_4 = predict_4.predict(args[1])
    res_5 = predict_5.predict(args[1])
    res_6 = predict_6.predict(args[1])
    res_7 = predict_7.predict(args[1])
    res_8 = predict_8.predict(args[1])


    total_predict.append(res_0)
    total_predict.append(res_1)
    total_predict.append(res_2)
    total_predict.append(res_3)
    total_predict.append(res_4)
    total_predict.append(res_5)
    total_predict.append(res_6)
    total_predict.append(res_7)
    total_predict.append(res_8)


    bagging(line_number, total_predict, args[2])



    return

if __name__ == '__main__':
    main()