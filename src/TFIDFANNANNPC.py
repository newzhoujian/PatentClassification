#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import sys
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import random
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, GRU, recurrent, Reshape, Dropout
from keras.models import Model
from keras import regularizers

from sklearn.decomposition import PCA

f = pd.read_excel('../data/1000.xlsx', header=0)
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
    tempX = u''
    X = []
    # sys.exit()
    for i in range(1, len(f[u'标题'])+1):
        # tempX.append(f[u'标题'][i])
        for j in f[u'标题'][i]:
            isin = (j in stopwordset)
            if not isin:
                tempX += j
                tempX += u' '
        for j in f[u'摘要'][i]:
            for k in j:
                isin = (k in stopwordset)
                if not isin:
                    tempX += k
                    tempX += u' '

        for j in f[u'首项权利要求'][i]:
            for k in j:
                isin = (k in stopwordset)
                if not isin:
                    tempX += k
                    tempX += u' '
        X.append(tempX)
        tempX = u''

    return X


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


X = get_str_X()


y, tags2ids = get_y()

y = np.array(y)
batch_size = 20
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)
X_tfidf = X_tfidf.toarray()
# print X_tfidf


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=33)


print 'begin training...'
model_input = Input(shape=(X_train.shape[1],))
sentence = Dense(1000, activation='tanh')(model_input)
sen2vec = Dense(200, activation='tanh')(sentence)
model_output = Dense(len(tags2ids), activation='softmax')(sen2vec)
model = Model(inputs=model_input, outputs=model_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20)
print 'end training!'
print 'save model!'
model.save('../model/TFIDFANNANNPC_model.h5')
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

