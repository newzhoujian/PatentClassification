#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
from itertools import chain
import sys
import numpy as np
import jieba
import keras
import keras.backend as K
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import gensim
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, GRU, recurrent, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split

#f = pd.read_excel('../data/1000.xlsx', header=0)
# f = pd.read_excel('../data/30.xlsx', header=0)
# f = pd.read_csv('data/data.csv', header=None, sep=',')
# f = open('data/smartPatent_20180512.xlsx', 'r')
f = open('../data/cutxslxaddstopword.txt', 'r')
ff = open('../data/cutxslxaddstopwordlabel.txt', 'r')

reload(sys)
sys.setdefaultencoding('utf-8')

def get_str_X():
    X = []
    for lines in f:
        lines = lines.encode('utf8')
        arr = lines.split(u' ')
        X.append(arr)
    f.close()
    return X


def get_embedding_X():
    X = get_str_X()

    modelword2vec = Word2Vec.load('../word2vec/word2vecaddstopword.model')
    sent_X = []
    fin_X = []
    for i in X:
        for j in i:
            # print chars2ids[k]
            tempword = gensim.utils.to_unicode(j)
            if tempword in modelword2vec:
                sent_X.append(np.array([w for w in modelword2vec[tempword]]))
            else:
                sent_X.append(np.array([0.] * modelword2vec.vector_size))
        sent_X = np.array(sent_X)
        fin_X.append(sent_X)
        sent_X = []
    print 'embedding down!'
    return fin_X, modelword2vec.vector_size


def get_y():
    tags = set()
    y = []
    for i in ff:
        tags.add(i[0])
        y.append(i[0])

    tags2ids = {}

    cnt = 0
    for i in tags:
        tags2ids[i] = cnt
        cnt += 1

    for i in range(len(y)):
        y[i] = tags2ids[y[i]]

    ids2onehot = {}
    j = 0
    for i in tags:
        temp = [0] * len(tags)
        temp[j] = 1
        ids2onehot[tags2ids[i]] = temp
        j += 1

    y_onehot = []
    for i in range(len(y)):
        y_onehot.append(ids2onehot[y[i]])

    print 'get y!'
    ff.close()
    return y_onehot, tags2ids


def get_padding_X(sentmaxlen):
    X, vecsize = get_embedding_X()
    for i in range(len(X)):
        if len(X[i]) >= sentmaxlen:
            X[i] = X[i][:sentmaxlen]
        else:
            temp = np.array([np.array([0.] * vecsize)] * (sentmaxlen - len(X[i])))
            X[i] = np.concatenate((temp, X[i]), axis=0)

    print 'padding down!'
    return X, vecsize


sent_maxlen = 200
word_size = 200
sent_size = 200
sess_size = 200
batch_size = 20


X, vecsize = get_padding_X(sentmaxlen=sent_maxlen)
y, tags2ids = get_y()

n_class = len(tags2ids)

y = np.array(y)

X_word2vec = np.array([i for i in X])
X_word2vec = np.reshape(X_word2vec, (len(X), -1, vecsize))
print 'data and label convert down!'

X_train, X_test, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=33)

print 'begin training...'
model_input = Input(shape=(sent_maxlen, vecsize))
sentence_oneline = Reshape((-1,))(model_input)
sen2vec = Dense(sent_size, activation='tanh')(sentence_oneline)
model_output = Dense(len(tags2ids), activation='softmax')(sen2vec)
# model_output = Dense(1)(sess2vec)
model = Model(inputs=model_input, outputs=model_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20)
print 'end training!'
print 'save model!'
model.save('../model/ANNPC_model.h5')
print 'save model down!'

print 'predicting...'
y_pred = model.predict(X_test)
print 'predicted done'


def check(a):
    max_pos = -1
    temp = -1
    for i in range(len(a)):
        if a[i] > temp:
            temp = a[i]
            max_pos = i
    return max_pos


def cal_acc_test(y1, y2):
    cnt = 0
    for i in range(len(y1)):
        a = check(y1[i])
        if y2[i][a] == 1:
            cnt += 1
    return cnt*1./(len(y1)*1.)


print cal_acc_test(y_pred, y_test)