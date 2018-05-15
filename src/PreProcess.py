#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
from itertools import chain
import sys
import numpy as np
import jieba
import keras
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, GRU
from keras.models import Model

f = pd.read_excel('../data/temp.xlsx', header=0)
# f = pd.read_csv('data/data.csv', header=None, sep=',')
# f = open('data/smartPatent_20180512.xlsx', 'r')


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

    tempX = []
    X = []
    for i in range(1, len(f[u'标题'])):
        tempX.append(f[u'标题'][i])
        for j in f[u'摘要'][i]:
            ed = len(j)
            for k in range(len(j)):
                if j[k] == u'<':
                    ed = k
                    break
            tempX.append(j[:ed])

        for j in f[u'首项权利要求'][i]:
            ed = len(j)
            for k in range(len(j)):
                if j[k] == u'<':
                    ed = k
                    break
            tempX.append(j[:ed])
        X.append(tempX)
        tempX = []
    return X


def get_X():
    X = get_str_X()
    chars = set()

    for i in X:
        for j in i:
            for k in j:
               chars.add(k)

    chars2ids = {}

    cnt = 1
    for i in chars:
        chars2ids[i] = cnt
        cnt += 1

    for i in range(len(X)):
        for j in range(len(X[i])):
            for k in range(len(X[i][j])):
                # print chars2ids[k]
                X[i][j][k] = chars2ids[X[i][j][k]]

    return X, cnt


def get_y():
    tags = set()
    for i in f[u'主IPC分类号-小类'].values:
        tags.add(i)

    tags2ids = {}

    cnt = 0
    for i in tags:
        tags2ids[i] = cnt
        cnt += 1

    f[u'主IPC分类号-小类'] = f[u'主IPC分类号-小类'].apply(lambda str_y: tags2ids[str_y])
    y = f[u'主IPC分类号-小类'].values

    return y


def get_padding_X(sentmaxlen, sessmaxlen):
    X, cnt = get_X()
    for i in range(len(X)):
        for j in range(len(X[i])):
            if len(X[i][j]) >= sentmaxlen:
                X[i][j] = X[i][j][:sentmaxlen]
            else:
                temp = [0] * (sentmaxlen - len(X[i][j]))
                X[i][j] = temp + X[i][j]
    '''
    if len(X) >= sessmaxlen:
        X = X[:sessmaxlen]
    else:
        temp = [[0] * sentmaxlen] * (sessmaxlen - len(X))
        X = temp + X
    '''
    for i in range(len(X)):
        if len(X[i]) >= sessmaxlen:
            X[i] = X[i][:sessmaxlen]
        else:
            temp = [[0] * sentmaxlen] * (sessmaxlen - len(X[i]))
            X[i] = temp + X[i]
    return X


sent_maxlen = 32
sess_maxlen = 40
word_size = 200

X, cnt = get_padding_X(sentmaxlen=sent_maxlen, sessmaxlen=sess_maxlen)
y = get_y()


sequence = Input(shape=(sent_maxlen,), dtype='int32')
embedded = Embedding(cnt, word_size, input_length=sent_maxlen, mask_zero=True)(sequence)
sen2vec = GRU(200, return_sequences=True)(embedded)
output = TimeDistributed(Dense(1))(sen2vec)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
all_tags = list(chain(*f[u'主IPC分类号-小类'].values))
sr_alltags = pd.Series(all_tags)
sr_alltags = sr_alltags.value_counts()
set_tags = sr_alltags.index
set_tagids = range(0, len(set_tags))

tag2id = pd.Series(set_tagids, index=set_tags)
id2tag = pd.Series(set_tags, index=set_tagids)
'''

'''
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
        elif i == len(res) - 1:
            temp += res[i]
            if temp != '':
                r.append(temp)
            break
        else:
            temp += res[i]
    return r


f[u'标题'] = f[u'标题'].apply(cut_title)
f[u'摘要'] = f[u'摘要'].apply(cut_abstract)




tags = set()

for i in f[u'主IPC分类号-小类'].values:
    tags.add(i)

tags2ids = {}

cnt = 0
for i in tags:
    tags2ids[i] = cnt
    cnt += 1

print "tags:"
print tags
print len(tags)
print "end"


def tagstoids(str):
    return tags2ids[str]


y = f[u'主IPC分类号-小类'].apply(tagstoids).values

print y[:20]


def run():
    f[u'标题'] = f[u'标题'].apply(cut_title)
    f[u'摘要'] = f[u'摘要'].apply(cut_abstract)
    f[u'首项权利要求'] = f[u'首项权利要求'].apply(cut_abstract)

    X = []
    for i in range(len(f[u'标题'])):
        X.append(f[u'标题'][i].values)
        for j in f[u'摘要'][i].values:
            X.append(j)
        for j in f[u'首项权利要求'][i].values:
            X.append(j)

    print X[0]

if __name__ == "__main__":
    run()

'''


