#!/usr/bin/env python
# -*- coding:utf-8 -*-

from gensim.models import word2vec

sentences=word2vec.Text8Corpus('cut.txt')
model=word2vec.Word2Vec(sentences, size=200)
model.save('../data/word2vec.model')
