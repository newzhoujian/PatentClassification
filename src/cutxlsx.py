#!/usr/bin/env python
# -*- coding:utf-8 -*-
import jieba
import pandas as pd
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

f = pd.read_excel('../data/smartPatent_20180512.xlsx', header=0)

ff = open('cut.txt', 'w')

print len(f)
for i in range(1, len(f)):
    temp_title = f[u'标题'][i]
    temp_title.replace('\t', '').replace(u'\n', '').replace(' ', '')
    seg_list = jieba.cut(temp_title, cut_all=False)
    ff.write(" ".join(seg_list))
    ff.write(' ')

    temp_abstract = f[u'摘要'][i]
    if temp_abstract is not np.nan:
        temp_abstract.replace('\t', '').replace(u'\n', '').replace(' ', '')
        seg_list = jieba.cut(temp_abstract, cut_all=False)
        ff.write(" ".join(seg_list))
        ff.write(' ')

    temp_top = f[u'首项权利要求'][i]
    temp_top.replace('\t', '').replace(u'\n', '').replace(' ', '')
    seg_list = jieba.cut(temp_top, cut_all=False)
    ff.write(" ".join(seg_list))
    ff.write(' ')

