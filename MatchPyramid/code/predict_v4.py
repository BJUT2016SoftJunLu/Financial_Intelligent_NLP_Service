# /usr/bin/env python
# coding=utf-8


from data_prov4 import *
import tensorflow as tf
from train_v4_1 import dynamic_pooling_index

with open(WORD_DICT_PATH, 'rb') as f:
    word_dict = pickle.load(f)[0]

with open(EMBEDDING_MATRIX_PATH, 'rb') as f:
    embedding_matrix = pickle.load(f)[0]



class Predict:

    def __init__(self,model_number,step_number):

        self.graph=tf.Graph()
        with self.graph.as_default():
             self.saver=tf.train.import_meta_graph("./MatchPyramid/model/NLP" + str(model_number) + "-" +str(step_number) + ".meta")#创建恢复器

        self.sess=tf.Session(graph=self.graph)
        with self.sess.as_default():
             with self.graph.as_default():
                self.saver.restore(self.sess,"./MatchPyramid/model/NLP" + str(model_number) + "-" +str(step_number))

                self.input_x1 = self.graph.get_tensor_by_name('input_x1:0')
                self.input_x2 = self.graph.get_tensor_by_name('input_x2:0')
                self.input_x1_len = self.graph.get_tensor_by_name('x1_len:0')
                self.input_x2_len = self.graph.get_tensor_by_name('x2_len:0')
                self.input_dpool_index = self.graph.get_tensor_by_name('dpool_index:0')
                self.input_y = self.graph.get_tensor_by_name('input_y:0')
                self.input_embedding_matrix = self.graph.get_tensor_by_name('embedding_matrix:0')
                self.output_prob = self.graph.get_operation_by_name('prob').outputs[0]

    def predict(self, INPUT_PATH, batch_size=64):

        X1_code = []
        X2_code = []
        X1_len = []
        X2_len = []
        Y_label = []

        with open(INPUT_PATH, 'r') as f:
            for line in f.readlines():
                example_split = line.split("\t")
                new_sentence_1 = ' '.join(sentence_split(example_split[1])).replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')','').replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '').replace('’', '').replace('*', '')
                new_sentence_2 = ' '.join(sentence_split(example_split[2])).replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('(', '').replace(')','').replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '').replace('’', '').replace('*', '')

                code_1 = []
                X1_len.append(len(new_sentence_1.split(' ')))
                for word in new_sentence_1.split(' '):
                    if word in word_dict.keys():
                        code_1.append(word_dict[word])
                    else:
                        code_1.append(word_dict['_PAD_'])

                code_2 = []
                X2_len.append(len(new_sentence_2.split(' ')))
                for word in new_sentence_2.split(' '):
                    if word in word_dict.keys():
                        code_2.append(word_dict[word])
                    else:
                        code_2.append(word_dict['_PAD_'])

                X1_code.append(code_1)
                X2_code.append(code_2)


        X1_code = np.array(X1_code)
        X2_code = np.array(X2_code)

        X1_data = []
        X2_data = []

        for x1 in X1_code:
           tmp = np.zeros(X1_MAX_LENGTH)
           tmp.fill(len(word_dict) - 1)
           tmp[:len(x1)] = x1
           X1_data.append(tmp)

        for x2 in X2_code:
           tmp = np.zeros(X2_MAX_LENGTH)
           tmp.fill(len(word_dict) - 1)
           tmp[:len(x2)] = x2
           X2_data.append(tmp)

        X1_data = np.array(X1_data)
        X2_data = np.array(X2_data)

        data_size = X1_data.shape
        batch_nums = int(data_size[0] / batch_size) + 1

        predict_resutlt = []

        for j in range(batch_nums):
            start = j * batch_size
            end = start + batch_size

            if end > data_size[0]:
                if start == data_size[0]:
                    break
                test_dict = {
                    self.input_x1: X1_data[start:],
                    self.input_x2: X2_data[start:],
                    self.input_x1_len: np.array(X1_len[start:]),
                    self.input_x2_len: np.array(X2_len[start:]),
                    self.input_dpool_index: dynamic_pooling_index(np.array(X1_len[start:]), np.array(X2_len[start:]),X1_MAX_LENGTH, X2_MAX_LENGTH),
                    self.input_embedding_matrix: embedding_matrix
                }
                test_prob = self.sess.run(self.output_prob, feed_dict=test_dict)

                for index in np.argmax(test_prob, axis=1):
                    predict_resutlt.append(index)

            else:
                test_dict = {
                    self.input_x1: X1_data[start:end],
                    self.input_x2: X2_data[start:end],
                    self.input_x1_len: np.array(X1_len[start:end]),
                    self.input_x2_len: np.array(X2_len[start:end]),
                    self.input_dpool_index: dynamic_pooling_index(np.array(X1_len[start:end]), np.array(X2_len[start:end]),X1_MAX_LENGTH, X2_MAX_LENGTH),
                    self.input_embedding_matrix: embedding_matrix
                }
                test_prob = self.sess.run(self.output_prob, feed_dict=test_dict)
                for index in np.argmax(test_prob, axis=1):
                    if index == 1:
                        # (0, 1)
                        predict_resutlt.append(0)
                    else:
                        # (1, 0)
                        predict_resutlt.append(1)

        self.sess.close()
        return predict_resutlt


def bagging(line_number, total_predict, OUTPUT_PATH):
    line_number = np.array(line_number)
    total_predict = np.array(total_predict)
    total_predict = np.sum(total_predict, axis=0)

    result = []
    for pre in total_predict:
        if pre >= 4:
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

    predict_0 = Predict(model_number=0, step_number=233700)
    predict_1 = Predict(model_number=1, step_number=233700)
    predict_2 = Predict(model_number=2, step_number=233700)
    predict_3 = Predict(model_number=3, step_number=233700)
    predict_4 = Predict(model_number=4, step_number=233700)
    predict_5 = Predict(model_number=5, step_number=233700)
    predict_6 = Predict(model_number=6, step_number=233700)





    res_0 = predict_0.predict(args[1])
    res_1 = predict_1.predict(args[1])
    res_2 = predict_2.predict(args[1])
    res_3 = predict_3.predict(args[1])
    res_4 = predict_4.predict(args[1])
    res_5 = predict_5.predict(args[1])
    res_6 = predict_6.predict(args[1])


    #
    # print(res_0)
    # print(res_1)
    # print(res_2)
    # print(res_3)
    # print(res_4)
    # print(res_5)
    # print(res_6)
    # print(res_7)
    # print(res_8)

    total_predict.append(res_0)
    total_predict.append(res_1)
    total_predict.append(res_2)
    total_predict.append(res_3)
    total_predict.append(res_4)
    total_predict.append(res_5)
    total_predict.append(res_6)



    bagging(line_number, total_predict, args[2])



    return

if __name__ == '__main__':
    main()