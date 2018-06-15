#!/usr/bin/env python
# -*- coding:utf-8 -*-
import jieba
import pandas as pd
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

jieba.load_userdict('../data/keyword.txt')
# f = pd.read_excel('../data/30.xlsx', header=0)
f = pd.read_excel('../data/smartPatent_20180512.xlsx', header=0)
# f = pd.read_csv('data/data.csv', header=None, sep=',')
# f = open('data/smartPatent_20180512.xlsx', 'r')
stopwordfile = open('../data/StopWords_con.txt', 'r')
ff = open('../data/cutxslxaddstopword.txt', 'w')
fff = open('../data/cutxslxaddstopwordlabel.txt', 'w')

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


f[u'标题'] = f[u'标题'].apply(cut_title)
f[u'摘要'] = f[u'摘要'].apply(cut_abstract)
f[u'首项权利要求'] = f[u'首项权利要求'].apply(cut_abstract)
print 'cut word down!'
print 'get stop word!'
stopwordset = getstopwordset()
print 'get down!'
tempX = u''
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
    ff.write(tempX)
    ff.write('\n')
    tempX = u''




for i in f[u'主IPC分类号-小类'].values:
    fff.write(i)
    fff.write(u'\n')

