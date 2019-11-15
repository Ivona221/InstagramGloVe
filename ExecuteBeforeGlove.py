import numpy as np
import datetime, shelve
import sys
from mxnet import nd
from mxnet.contrib import text

path = sys.argv[1]

def read_data(file_name):
    with open(file_name,'r', encoding="utf-8") as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            floatArray = [float(i) for i in words_Vec[1:]]
            word2vector[words_Vec[0]] = np.array(floatArray,dtype=float)
    return word_vocab, word2vector

vocab, w2v = read_data("glove.instagram.new1.100d.txt")

glove_6b50d = text.embedding.create(
   'glove', pretrained_file_name='glove.twitter.27B.100d.txt')

# Change to path
shelf = shelve.open(path)
shelf['vocab'] = vocab
shelf['w2v'] = w2v
#shelf['glove_6b50d'] = glove_6b50d
shelf.close()
