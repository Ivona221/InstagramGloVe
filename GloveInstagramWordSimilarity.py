"""
Created on Tue Sep 24 00:21:36 2019

@author: imi
"""

import numpy as np
import datetime, shelve
import sys
import math
import operator

# Change to the path
shelf = shelve.open("/media/ivona/6D1D6F436C827C1C/glove")
w2v = shelf['w2v']
vocab = shelf['vocab']
shelf.close()

def cos_sim(u,v):
    """
    u: vector of 1st word
    v: vector of 2nd Word
    """
    numerator_ = u.dot(v)
    denominator_= np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))
    return numerator_/denominator_

def KNN(word, K):
    try:
        cosineSimDict = {}
        u = w2v[word]
        for word, vector in w2v.items():
            v = vector
            numerator_ = u.dot(v)
            denominator_= np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))
            cosineSim = numerator_/denominator_
            cosineSimDict.update({word : cosineSim})
        sorted_cosine = sorted(cosineSimDict.items(), key=operator.itemgetter(1),reverse=True)
        sorted_cosine = sorted_cosine[0:K]
        
        return sorted_cosine
    except:
        return []

word = sys.argv[1]
K = int(sys.argv[2])

if "#" in word:
    wordList = [word]
else:
    wordList = ["#" + word]
gloveSimilarWords = []
for word in wordList:
    similarWords = KNN(word, K)
    similarWords = [i[0] for i in similarWords]
    gloveSimilarWords.extend(similarWords)
 
gloveSimilarWords = set(gloveSimilarWords)
gloveSimilarWords = list(gloveSimilarWords)

gloveSimilarWords =  [x for x in gloveSimilarWords if len(x) > 2]
print(gloveSimilarWords)
