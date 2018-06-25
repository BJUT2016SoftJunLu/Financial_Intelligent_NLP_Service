#/usr/bin/env python
#coding=utf-8

from MatchSRNN_Model import *


model_matchsrnn = MyMactchSRNN(x1_max_length=75,
                               x2_max_length=90,
                               vocab_size=13407,
                               embedding_size=125,
                               class_nums=2,
                               learning_rate=1e-3)