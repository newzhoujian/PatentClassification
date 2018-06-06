#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
from itertools import chain
import sys
import numpy as np
import jieba
import keras
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import gensim
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, GRU, recurrent
from keras.models import Model
# f = pd.read_excel('../data/temp.xlsx', header=0)
f = pd.read_excel('../data/temp.xlsx', header=0)
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
        print tempX
        X.append(tempX)
        tempX = []

    return X


def get_embedding_X():
    X = get_str_X()
    #sys.exit()
    modelword2vec = Word2Vec.load('../word2vec/word2vec.model')
    sess_X = []
    sent_X = []
    fin_X = []
    for i in X:
        for j in i:
            for k in j:
                # print chars2ids[k]
                tempword = gensim.utils.to_unicode(k)
                if tempword in modelword2vec:
                    sent_X.append(np.array([w for w in modelword2vec[tempword]]))
                else:
                    sent_X.append(np.array([0.] * modelword2vec.vector_size))
            sent_X = np.array(sent_X)
            sess_X.append(sent_X)
            sent_X = []
        sess_X = np.array(sess_X)
        fin_X.append(sess_X)
        sess_X = []

    print 'embedding down!'
    return fin_X, modelword2vec.vector_size

'''
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
'''



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


def get_padding_X(sentmaxlen, sessmaxlen):
    X, vecsize = get_embedding_X()
    for i in range(len(X)):
        for j in range(len(X[i])):
            if len(X[i][j]) >= sentmaxlen:
                X[i][j] = X[i][j][:sentmaxlen]
            else:
                temp = np.array([np.array([0.] * vecsize)] * (sentmaxlen - len(X[i][j])))
                if len(X[i][j]) != 0:
                    X[i][j] = np.concatenate((temp, X[i][j]), axis=0)

                else:
                    X[i][j] = temp



    '''
    if len(X) >= sessmaxlen:
        X = X[:sessmaxlen]
    else:
        temp = [[0] * sentmaxlen] * (sessmaxlen - len(X))
        X = temp + X
    '''
    for i in range(len(X)):
        if len(X[i]) >= sessmaxlen:
            temp = np.array([X[i][0]])
            for j in range(1, 40):
                temp = np.concatenate((temp, [X[i][j]]), axis=0)
            X[i] = temp
        else:
            temp = np.array([[np.array([0.] * vecsize)] * sentmaxlen] * (sessmaxlen - len(X[i])))
            for j in X[i]:
                temp = np.concatenate((temp, [j]), axis=0)
            X[i] = temp

    print 'padding down!'
    return X, vecsize


sent_maxlen = 32
sess_maxlen = 40
word_size = 200
sent_size = 200
sess_size = 200
batch_size = 1000

# X, vecsize = get_embedding_X()


X, vecsize = get_padding_X(sentmaxlen=sent_maxlen, sessmaxlen=sess_maxlen)
y, tags2ids = get_y()

n_class = len(tags2ids)

y = np.array(y)

X_word2vec = np.array([i for i in X])
X_word2vec = np.reshape(X_word2vec, (len(X), -1, vecsize))
print 'data and label convert down!'
# rint type(X_word2vec)

# print type(y[0])

# print X_word2vec.shape
# print y.shape
#print X[0]
#print X_word2vec[:40]
# print y

print 'begin training...'
model_input = Input(shape=(sent_maxlen*sess_maxlen, vecsize), dtype=tf.float32)
sen2vec = GRU(sent_size, activation='tanh', return_sequences=True)(model_input)
sess2vec = GRU(sess_size, return_sequences=False)(sen2vec)
model_output = Dense(len(tags2ids), activation='softmax')(sess2vec)
# model_output = Dense(1)(sess2vec)
model = Model(inputs=model_input, outputs=model_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_word2vec, y, batch_size=batch_size, epochs=50)
print 'end training!'
print 'save model!'
model.save('../model/PatentClassification.h5')
print 'save model down!'



'''
sent = []
sess = []
sent2vec = Input(shape=(sent_maxlen,), dtype='int32')  # (32,)
print type(sent2vec)
for i in range(sess_maxlen):
    sent.append(sent2vec)  # 40 * (32, )
for word2vecs in sent:
    embedded = Embedding(cnt, word_size, input_length=sent_maxlen, mask_zero=True)(word2vecs)  # (32, 200)
    sen2vec = GRU(sent_size, activation='tanh', return_sequences=True)(embedded)  # (32, 200)
    sess.append(sen2vec)

sess = tf.convert_to_tensor(sess)  # (40, 32, 200)
sess = tf.reshape(sess, (sess_maxlen, sent_maxlen, -1))
sess2vec = GRU(sess_size, return_sequences=False)(sess)
# sess2vec = tf.reshape(sess2vec, (-1, ))
# output = TimeDistributed(Dense(5, activation='softmax'))(sess2vec)
# output = Dense(len(tags2ids), activation='softmax')(sess2vec)
output = Dense(5, activation='softmax')(sess2vec)
model = Model(inputs=sent2vec, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_word2vec, y, batch_size=batch_size, epochs=50)



sent = []
sess = []
sent2vec = Input(shape=(sent_maxlen, vecsize), dtype=tf.float32)
for i in range(sess_maxlen):
    sent.append(sent2vec)
for sent in sent:
    sent2vec_res = GRU(sent_size, activation='tanh', return_sequences=True)(sent)  # (32, 200)
    sess.append(sent2vec_res)

sess = tf.convert_to_tensor(sess)
sess = tf.reshape(sess, (sess_maxlen, sent_size, -1))
sess2vec = GRU(sess_size, return_sequences=False)(sess)
output = TimeDistributed(Dense(5, activation='softmax'))(sess2vec)
model = Model(inputs=sent2vec, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_word2vec, y, batch_size=batch_size, epochs=50)
'''


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


