# encoding=utf-8
import jieba

X = [467, 464, 804, 983, 768]
maxlen = 10
if len(X) >= maxlen:
    X = X[:maxlen]
else:
    temp = [0] * (maxlen - len(X))
    X = temp + X

print X

