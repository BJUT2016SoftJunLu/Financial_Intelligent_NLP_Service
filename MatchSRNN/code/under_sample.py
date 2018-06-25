import os
import random

from data_pro import FILE_PATH_1,FILE_PATH_2

FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sub_data/sub_data"))
MODEL_NUMBER = 7

neg_num = 18685.0
total_num = 102477.0

under_sample_num = int(neg_num*(10.0/6.0))


pos_samples = [] # 0
neg_samples = [] # 1

with open(FILE_PATH_1, 'r') as fr:
    for line in fr.readlines():
        example_split = line.split("\t")
        sign = int(example_split[-1])
        if sign == 1:
            neg_samples.append(line)
        else:
            pos_samples.append(line)

with open(FILE_PATH_2, 'r') as fr:
    for line in fr.readlines():
        example_split = line.split("\t")
        sign = int(example_split[-1])
        if sign == 1:
            neg_samples.append(line)
        else:
            pos_samples.append(line)


pos_index = [i for i in range(len(pos_samples))]


for i in range(MODEL_NUMBER):
    sub_sample = []
    pos_slice = random.sample(pos_index, under_sample_num)
    for index in pos_slice:
        sub_sample.append(pos_samples[index])

    for neg_line in neg_samples:
        sub_sample.append(neg_line)

    random.shuffle(sub_sample)
    with open(FILE_PATH + str(i), 'w') as fw:
        for line in sub_sample:
            fw.write(line)










