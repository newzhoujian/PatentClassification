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
    ids2tags = {}
    cnt = 0
    for i in tags:
        tags2ids[i] = cnt
        ids2tags[cnt] = i
        cnt += 1

    f[u'主IPC分类号-小类'] = f[u'主IPC分类号-小类'].apply(lambda str_y: tags2ids[str_y[0]])
    y = f[u'主IPC分类号-小类'].values

    print 'get y!'
    return y, tags2ids, ids2tags


X = get_str_X()


y, tags2ids, ids2tags = get_y()
print y
print ids2tags[0]
tfidf = TfidfVectorizer(max_features=75000)
X_tfidf = tfidf.fit_transform(X)
X_tfidf = X_tfidf.toarray()

pca = PCA()
pca.fit(X_tfidf)
X_tfidf = pca.components_
# print X_tfidf

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=33)

print X_train.shape
print y_train.shape

print X_test.shape
print y_test.shape
print type(X_train[0])
print y_train[0]
print len(tags2ids)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(tags2ids),
    'min_data_in_bin': 1,
    'min_data': 1,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 1,
    'verbose': 0
}

gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=20)

gbm.save_model('model.txt')


# yprob = gbm.predict(X_test).reshape(y_test.shape[0], len(tags2ids))
yprob = gbm.predict(X_test)
print yprob

ylabel = np.argmax(yprob, axis=1)
error = sum( int(ylabel[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))
acc = 1. - error
print ('predicting, classification acc=%f' % (acc))





