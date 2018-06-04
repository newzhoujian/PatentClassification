# encoding=utf-8
import jieba
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import sys

f = open('../data/StopWords_con.txt', 'r')

w = set()


for lines in f:
    arr = lines.split('\n')
    # print type(arr[0].decode('utf-8'))
    w.add(arr[0].decode('utf-8'))

a = (u'现在' in w)

print a
