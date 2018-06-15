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
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, GRU, recurrent, Reshape, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
jieba.load_userdict('../data/keyword.txt')

f = pd.read_excel('../data/random1000.xlsx', header=0)
# f = pd.read_excel('../data/30.xlsx', header=0)
# f = pd.read_csv('data/data.csv', header=None, sep=',')
# f = open('data/smartPatent_20180512.xlsx', 'r')
stopwordfile = open('../data/StopWords_con.txt', 'r')


def getstopwordset():
    w = set()
    for lines in stopwordfile:
        arr = lines.split('\n')
        # print type(arr[0].decode('utf-8'))
        w.add(arr[0].decode('utf-8'))
    return w


def cut_title(str):
    seg_list = jieba.cut(str)
    res = " ".join(seg_list)
    r = []
    temp = ''
    for i in range(len(res)):
        if res[i] == ' ':
            r.append(temp)
            temp = ''
        elif res[i] == '\n':
            r.append(temp)
            temp = ''
        elif i == len(res) - 1:
            temp += res[i]
            r.append(temp)
            break
        else:
            temp += res[i]
    return r


def cut_abstract(str):
    # print np.isnan(str)
    if str is np.nan:
        return [u'']

    seg_list = jieba.cut(str)
    res = " ".join(seg_list)
    fin_res = []
    r = []
    temp = ''
    for i in range(len(res)):
        if res[i] == ' ':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == '\n':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == u'，':
            if temp != '':
                r.append(temp)
            fin_res.append(r)
            r = []
            temp = ''
        elif res[i] == u'。':
            if temp != '':
                r.append(temp)
            fin_res.append(r)
            r = []
            temp = ''
        elif res[i] == u'；':
            if temp != '':
                r.append(temp)
            fin_res.append(r)
            r = []
            temp = ''
        elif res[i] == u'(':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == u')':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == u'、':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == u'：':
            if temp != '':
                r.append(temp)
            temp = ''
        elif res[i] == u'-':
            if temp != '':
                r.append(temp)
            temp = ''
        else:
            temp += res[i]
    return fin_res


def get_str_X():
    f[u'标题'] = f[u'标题'].apply(cut_title)
    f[u'摘要'] = f[u'摘要'].apply(cut_abstract)
    f[u'首项权利要求'] = f[u'首项权利要求'].apply(cut_abstract)
    print 'cut word down!'
    print 'get stop word!'
    stopwordset = getstopwordset()
    print 'get down!'
    tempX = []
    X = []
    # sys.exit()
    for i in range(1, len(f[u'标题'])+1):
        # tempX.append(f[u'标题'][i])
        for j in f[u'标题'][i]:
            isin = (j in stopwordset)
            if not isin:
                tempX.append(j)
        for j in f[u'摘要'][i]:
            for k in j:
                isin = (k in stopwordset)
                if not isin:
                    tempX.append(k)

        for j in f[u'首项权利要求'][i]:
            for k in j:
                isin = (k in stopwordset)
                if not isin:
                    tempX.append(k)
        X.append(tempX)
        tempX = []

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
    for i in f[u'主IPC分类号-小类'].values:
        tags.add(i[0])

    tags2ids = {}

    cnt = 0
    for i in tags:
        tags2ids[i] = cnt
        cnt += 1

    f[u'主IPC分类号-小类'] = f[u'主IPC分类号-小类'].apply(lambda str_y: tags2ids[str_y[0]])
    y = f[u'主IPC分类号-小类'].values

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
'''
X_train = X_word2vec[:800]
X_test = X_word2vec[800:]
y_train = y[:800]
y_test = y[800:]
'''



print 'begin training...'
model_input = Input(shape=(sent_maxlen, vecsize))
sen2vec = GRU(sent_size, activation='relu', return_sequences=True)(model_input)
# sen2vec = Dropout(0.25)(sen2vec)
ses2vec = GRU(sess_size, activation='relu', return_sequences=False)(sen2vec)
model_output = Dense(len(tags2ids), activation='softmax')(ses2vec)
# model_output = Dense(1)(sess2vec)
model = Model(inputs=model_input, outputs=model_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=10)
print 'end training!'
print 'save model!'
model.save('../model/GRUGRUPC_model.h5')
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