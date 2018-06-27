# /usr/bin/env python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import numpy as np

from collections import Counter
import pickle
import csv
import jieba


jieba.add_word('花呗')
jieba.add_word('借呗')
PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"
TRUE_LABEL='1'
splitter="&|&"
special_start_token=[u'怎么',u'如何',u'为什么',u'为何']


def create_data_v1(traning_data_path,
                   vocab_word2index,
                   vocab_label2index,
                   sentence_len,
                   training_portion=0.95,
                   tokenize_style='word'):

    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """

    csvfile = open(traning_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    label_size=len(vocab_label2index)
    X1_ = []
    X2_ = []
    Y_ = []

    tfidf_source_file = '.../data/atec_nl_sim_train.txt'
    tfidf_target_file = '.../data/atec_nl_sim_tfidf.txt'

    get_tfidf_score_and_save(tfidf_source_file,tfidf_target_file)

    BLUE_SCORES_=[]
    word_vec_fasttext_dict=load_word_vec('../data/fasttext_fin_model_50.vec')
    word_vec_word2vec_dict = load_word_vec('../data/word2vec.txt')
    tfidf_dict=load_tfidf_dict('../data/atec_nl_sim_tfidf.txt')

    for i, row in enumerate(spamreader):
        x1_list=token_string_as_list(row[1],tokenize_style=tokenize_style)
        x1 = [vocab_word2index.get(x, UNK_ID) for x in x1_list]
        x2_list=token_string_as_list(row[2],tokenize_style=tokenize_style)
        x2 = [vocab_word2index.get(x, UNK_ID) for x in x2_list]
        #add blue score features 2018-05-06
        features_vector=data_mining_features(i,row[1], row[2],vocab_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict, n_gram=8)
        features_vector=[float(x) for x in features_vector]
        BLUE_SCORES_.append(features_vector)
        y_=row[3]
        y=vocab_label2index[y_]
        X1_.append(x1)
        X2_.append(x2)
        Y_.append(y)


    number_examples = len(Y_)

    #shuffle
    X1 = []
    X2 = []
    Y = []
    BLUE_SCORES = []
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X1.append(X1_[index])
        X2.append(X2_[index])
        Y.append(Y_[index])
        BLUE_SCORES.append(BLUE_SCORES_[index])

    # padding 0
    X1_zero = np.zeros(sentence_len)
    X2_zero = np.zeros(sentence_len)
    for i in range(len(X1)):
        X1_zero[i] = X1[i]
    for i in range(len(X2)):
        X2_zero[i] = X2[i]

    X1 = list(X1_zero)
    X2 = list(X2_zero)

    valid_number = min(1600,int((1-training_portion)*number_examples))
    test_number = 800
    training_number = number_examples-valid_number-test_number
    valid_end = training_number+valid_number

    #generate more training data, while still keep data distribution for valid and test.
    X1_final, X2_final, BLUE_SCORE_final,Y_final,training_number_big=get_training_data(X1[0:training_number], X2[0:training_number], BLUE_SCORES[0:training_number],Y[0:training_number], training_number)
    train = (X1_final,X2_final, BLUE_SCORE_final,Y_final)
    valid = (X1[training_number+ 1:valid_end], X2[training_number+ 1:valid_end],BLUE_SCORES[training_number + 1:valid_end],Y[training_number + 1:valid_end])
    test=(X1[valid_end+1:],X2[valid_end:],BLUE_SCORES[valid_end:],Y[valid_end:])

    true_label_numbers = len([y for y in Y if y == 1])
    true_label_pert = float(true_label_numbers)/float(number_examples)

    with open("../data/data_v1", 'ab') as data_f:
        pickle.dump((train, valid, test, true_label_pert),data_f)

    return


def create_data_v2(traning_data_path,
                   vocab_word2index,
                   vocab_label2index,
                   sentence_len,
                   tokenize_style='word'):

    csvfile = open(traning_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    label_size = len(vocab_label2index)
    X1_ = []
    X2_ = []
    Y_ = []

    tfidf_source_file = '../data/atec_nl_sim_train.txt'
    tfidf_target_file = '../data/atec_nl_sim_tfidf.txt'

    get_tfidf_score_and_save(tfidf_source_file, tfidf_target_file)

    BLUE_SCORES_ = []
    word_vec_fasttext_dict = load_word_vec('../data/fasttext_fin_model_50.vec')
    word_vec_word2vec_dict = load_word_vec('../data/word2vec.txt')
    tfidf_dict = load_tfidf_dict('../data/atec_nl_sim_tfidf.txt')

    for i, row in enumerate(spamreader):
        x1_list = token_string_as_list(row[1], tokenize_style=tokenize_style)
        x1 = [vocab_word2index.get(x, UNK_ID) for x in x1_list]
        x2_list = token_string_as_list(row[2], tokenize_style=tokenize_style)
        x2 = [vocab_word2index.get(x, UNK_ID) for x in x2_list]
        # add blue score features 2018-05-06
        features_vector = data_mining_features(i, row[1], row[2], vocab_word2index, word_vec_fasttext_dict, word_vec_word2vec_dict, tfidf_dict, n_gram=8)
        features_vector = [float(x) for x in features_vector]
        BLUE_SCORES_.append(features_vector)
        y_ = row[3]
        y = vocab_label2index[y_]
        X1_.append(x1)
        X2_.append(x2)
        Y_.append(y)

    number_examples = len(Y_)

    # shuffle
    X1 = []
    X2 = []
    Y = []
    BLUE_SCORES = []
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X1.append(X1_[index])
        X2.append(X2_[index])
        Y.append(Y_[index])
        BLUE_SCORES.append(BLUE_SCORES_[index])

    # padding 0
    X1_zero = np.zeros(sentence_len)
    X2_zero = np.zeros(sentence_len)
    for i in range(len(X1)):
        X1_zero[i] = X1[i]
    for i in range(len(X2)):
        X2_zero[i] = X2[i]

    X1 = list(X1_zero)
    X2 = list(X2_zero)

    with open("../data/data_v2", 'ab') as data_f:
        pickle.dump((X1, X2, Y, BLUE_SCORES), data_f)

    return


#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_vocabulary(training_data_path,
                      vocab_size,
                      name_scope='cnn',
                      tokenize_style='word'):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """

    vocabulary_word2index={}
    vocabulary_index2word={}
    vocabulary_word2index[_PAD]=PAD_ID
    vocabulary_index2word[PAD_ID]=_PAD
    vocabulary_word2index[_UNK]=UNK_ID
    vocabulary_index2word[UNK_ID]=_UNK

    vocabulary_label2index={'0':0,'1':1}
    vocabulary_index2label={0:'0',1:'1'}

    #1.load raw data
    csvfile = open(training_data_path, 'r')
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

    #2.loop each line,put to counter
    c_inputs=Counter()
    c_labels=Counter()
    for i,row in enumerate(spamreader):
        string_list_1=token_string_as_list(row[1], tokenize_style=tokenize_style)
        string_list_2 = token_string_as_list(row[2], tokenize_style=tokenize_style)
        c_inputs.update(string_list_1)
        c_inputs.update(string_list_2)

    #return most frequency words
    vocab_list=c_inputs.most_common(vocab_size)
    #put those words to dict
    for i,tuplee in enumerate(vocab_list):
        word,_=tuplee
        vocabulary_word2index[word]=i+2
        vocabulary_index2word[i+2]=word


    with open("../data/vocabulary", 'ab') as data_f:
        pickle.dump((vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label), data_f)

    save_vocab_as_file(vocabulary_word2index,vocabulary_index2label,vocab_list,name_scope=name_scope)
    return vocabulary_word2index,vocabulary_index2word,vocabulary_label2index,vocabulary_index2label


def save_vocab_as_file(vocab_word2index,vocab_index2label,vocab_list,name_scope='cnn'):

    #1.1save vocabulary_word2index
    vocab_word2index_object=open("../data/vocabulary_word2index",mode='a')
    for word,index in vocab_word2index.items():
        vocab_word2index_object.write(word+splitter+str(index)+"\n")
    vocab_word2index_object.close()

    #1.2 save word and frequent
    word_freq_object=open("../data/word_frequent", mode='a')
    for tuplee in vocab_list:
        word,count=tuplee
        word_freq_object.write(word+"|||"+str(count)+"\n")
    word_freq_object.close()

    #2.vocabulary_index2label
    vocab_index2label_object = open("../data/vocabulary_index2label",mode='a')
    for index,label in vocab_index2label.items():
        vocab_index2label_object.write(str(index)+splitter+str(label)+"\n")
    vocab_index2label_object.close()

    return


def get_training_data(X1,X2,BLUE_SCORES,Y,training_number,shuffle_word_flag=False):

    # 1.form more training data by swap sentence1 and sentence2
    X1_big = []
    X2_big = []
    BLUE_SCORE_big=[]
    Y_big = []

    X1_final = []
    X2_final = []
    BLUE_SCORE_final=[]
    Y_final = []
    for index in range(0, training_number):
        X1_big.append(X1[index])
        X2_big.append(X2[index])
        BLUE_SCORE_big.append(BLUE_SCORES[index])
        y_temp = Y[index]
        Y_big.append(y_temp)
        #a.swap sentence1 and sentence2
        if str(y_temp) == TRUE_LABEL:
            X1_big.append(X2[index])
            X2_big.append(X1[index])
            BLUE_SCORE_big.append(BLUE_SCORES[index])
            Y_big.append(y_temp)

        #b.random change location of words
        if shuffle_word_flag:
            for x in range(5):
                x1=X1[index]
                x2=X2[index]
                x1_random=[x1[i] for i in range(len(x1))]
                x2_random = [x2[i] for i in range(len(x2))]
                random.shuffle(x1_random)
                random.shuffle(x2_random)
                X1_big.append(x1_random)
                X2_big.append(x2_random)
                BLUE_SCORE_big.append(BLUE_SCORES[index])
                Y_big.append(Y[index])

    # shuffle data
    training_number_big = len(X1_big)
    permutation2 = np.random.permutation(training_number_big)
    for index in permutation2:
        X1_final.append(X1_big[index])
        X2_final.append(X2_big[index])
        BLUE_SCORE_final.append(BLUE_SCORE_big[index])
        Y_final.append(Y_big[index])

    return X1_final,X2_final,BLUE_SCORE_final,Y_final,training_number_big

def token_string_as_list(string,tokenize_style='word'):
    string=string.decode("utf-8")
    string=string.replace("***","*")
    length=len(string)
    if tokenize_style == 'char':
        listt=[string[i] for i in range(length)]
    else:
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt


def data_mining_features(index,input_string_x1,input_string_x2,vocab_word2index,word_vec_fasttext_dict,word_vec_word2vec_dict,tfidf_dict,n_gram=8):
    """
    get data mining feature given two sentences as string.
    1)n-gram similiarity(blue score);
    2) get length of questions, difference of length
    3) how many words are same, how many words are unique
    4) question 1,2 start with how/why/when(为什么，怎么，如何，为何）
    5）edit distance
    6) cos similiarity using bag of words
    :param input_string_x1:
    :param input_string_x2:
    :return:
    """
    input_string_x1=input_string_x1.decode("utf-8")
    input_string_x2 = input_string_x2.decode("utf-8")
    #1. get blue score vector
    feature_list=[]
    #get blue score with n-gram
    for i in range(n_gram):
        x1_list=split_string_as_list_by_ngram(input_string_x1,i+1)
        x2_list = split_string_as_list_by_ngram(input_string_x2, i + 1)
        blue_score_i_1 = compute_blue_ngram(x1_list,x2_list)
        blue_score_i_2 = compute_blue_ngram(x2_list,x1_list)
        feature_list.append(blue_score_i_1)
        feature_list.append(blue_score_i_2)

    #2. get length of questions, difference of length
    length1=float(len(input_string_x1))
    length2=float(len(input_string_x2))
    length_diff=(float(abs(length1-length2)))/((length1+length2)/2.0)
    feature_list.append(length_diff)

    #3. how many words are same, how many words are unique
    sentence_diff_overlap_features_list=get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2)
    feature_list.extend(sentence_diff_overlap_features_list)


    #5.edit distance
    edit_distance=float(edit(input_string_x1, input_string_x2))/30.0
    feature_list.append(edit_distance)

    #6.cos distance from sentence embedding
    x1_list=token_string_as_list(input_string_x1, tokenize_style='word')
    x2_list = token_string_as_list(input_string_x2, tokenize_style='word')
    distance_list_fasttext = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_fasttext_dict, tfidf_dict)
    distance_list_word2vec = cos_distance_bag_tfidf(x1_list, x2_list, word_vec_word2vec_dict, tfidf_dict)
    feature_list.extend(distance_list_fasttext)
    feature_list.extend(distance_list_word2vec)
    return feature_list

def load_word_vec(file_path):
    source_object = open(file_path, 'r')
    word_vec_dict={}
    for i,line in enumerate(source_object):
        if i==0 and 'word2vec' in file_path:
            continue
        line=line.strip()
        line_list=line.split()
        word=line_list[0].decode("utf-8")
        vec_list=[float(x) for x in line_list[1:]]
        word_vec_dict[word]=np.array(vec_list)
    return word_vec_dict


def load_tfidf_dict(file_path):
    source_object = open(file_path, 'r')
    tfidf_dict={}
    for line in source_object:
        word,tfidf_value=line.strip().split(splitter)
        word=word.decode("utf-8")
        tfidf_dict[word]=float(tfidf_value)
    return tfidf_dict

def get_special_start_token(input_string_x1,input_string_x2,special_token_list):

    feature_list1=[0.0 for i in range(len(special_token_list))]
    feature_list2=[0.0 for i in range(len(special_token_list))]

    for i,speical_token in enumerate(special_token_list):
        if input_string_x1.find(speical_token) > 0:
            feature_list1[i] = 1.0
        if input_string_x2.find(speical_token) > 0:
            feature_list2[i] = 1.0

    feature_list1.extend(feature_list2)
    return feature_list1


def get_sentence_diff_overlap_pert(index,input_string_x1,input_string_x2):

    #0. get list from string
    input_list1=[input_string_x1[token] for token in range(len(input_string_x1)) if input_string_x1[token].strip()]
    input_list2 = [input_string_x2[token] for token in range(len(input_string_x2)) if input_string_x2[token].strip()]
    length1=len(input_list1)
    length2=len(input_list2)

    num_same=0
    same_word_list=[]

    #1.compute percentage of same tokens
    for word1 in input_list1:
        for word2 in input_list2:
           if word1==word2:
               num_same=num_same+1
               same_word_list.append(word1)
               continue
    num_same_pert_min=float(num_same)/float(max(length1,length2))
    num_same_pert_max = float(num_same) / float(min(length1, length2))
    num_same_pert_avg = float(num_same) / (float(length1+length2)/2.0)

    #2.compute percentage of unique tokens in each string
    input_list1_unique=set([x for x in input_list1 if x not in same_word_list])
    input_list2_unique = set([x for x in input_list2 if x not in same_word_list])
    num_diff_x1=float(len(input_list1_unique))/float(length1)
    num_diff_x2= float(len(input_list2_unique)) / float(length2)


    diff_overlap_list=[num_same_pert_min,num_same_pert_max, num_same_pert_avg,num_diff_x1, num_diff_x2]

    return diff_overlap_list


def split_string_as_list_by_ngram(input_string,ngram_value):
    input_string="".join([string for string in input_string if string.strip()])
    length = len(input_string)
    result_string=[]
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i+ngram_value])
    return result_string


def compute_blue_ngram(x1_list,x2_list):
    """
    compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict={}
    count_dict_clip={}
    #1. count for each token at predict sentence side.
    for token in x1_list:
        if token not in count_dict:
            count_dict[token]=1
        else:
            count_dict[token]=count_dict[token]+1
    count=np.sum([value for key,value in count_dict.items()])

    #2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            if token not in count_dict_clip:
                count_dict_clip[token]=1
            else:
                count_dict_clip[token]=count_dict_clip[token]+1

    #3. clip value to ceiling value for that token
    count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
    count_clip=np.sum([value for key,value in count_dict_clip.items()])
    result=float(count_clip)/(float(count)+0.00000001)
    return result


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in xrange(1, len(str1) + 1):
        for j in xrange(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


def main():

    csv_fw = csv.writer(open("../data/atec_nlp_sim_train_data.csv"))
    csv_fr_v1 = csv.reader(open("../data/atec_nlp_sim_train.csv", 'r'))
    csv_fr_v2 = csv.reader(open("../data/atec_nlp_sim_train_add.csv", 'r'))

    for i, row in enumerate(csv_fr_v1):
        csv_fw.writerow(row)

    for i, row in enumerate(csv_fr_v2):
        csv_fw.writerow(row)


    # vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, vocabulary_index2label= create_vocabulary("../atec_nlp_sim_train_data.csv",vocab_size=30000)
    # vocab_size = len(vocabulary_word2index)
    # num_classes = len(vocabulary_index2label)
    #
    #
    #
    # create_data_v1("../data/atec_nlp_sim_train_data.csv", vocabulary_word2index, vocabulary_label2index, sentence_len=21)
    # create_data_v2("../data/atec_nlp_sim_train_data.csv", vocabulary_word2index, vocabulary_label2index, sentence_len=21)
    return

if __name__ == '__main__':
    main()