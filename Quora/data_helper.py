# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

KERAS_DATASETS_DIR = '' 
GLOVE_FILE = './data/glove.840B.300d.txt'

# 读取数据
def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = f.readlines()
    data = [re.split('\t', i) for i in data]
    q1 = [i[1] for i in data]
    
    q2 = [i[2] for i in data]
    label = [int(i[0]) for i in data]
    return q1, q2, label


train_q1, train_q2, train_label = read_data('./data/train.tsv')
test_q1, test_q2, test_label = read_data('./data/test.tsv')
dev_q1, dev_q2, dev_label = read_data('./data/dev.tsv')

# 构造训练word2vec的语料库
corpus = train_q1 + train_q2 + test_q1 + test_q2 + dev_q1 + dev_q2
w2v_corpus = [i.split() for i in corpus]
#词表
word_set = set(' '.join(corpus).split())


MAX_SEQUENCE_LENGTH = 30  # sequence最大长度为30个词
EMB_DIM = 300  # 词向量为300维

tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(corpus)
#生成的词表长度，等于num_words=len(word_set)
L = len(tokenizer.word_index)


embeddings_index = {}
with open(KERAS_DATASETS_DIR + GLOVE_FILE) as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))

# Prepare word embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index), EMB_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



train_q1 = tokenizer.texts_to_sequences(train_q1)
train_q2 = tokenizer.texts_to_sequences(train_q2)

test_q1 = tokenizer.texts_to_sequences(test_q1)
test_q2 = tokenizer.texts_to_sequences(test_q2)

dev_q1 = tokenizer.texts_to_sequences(dev_q1)
dev_q2 = tokenizer.texts_to_sequences(dev_q2)

train_pad_q1 = pad_sequences(train_q1, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_q2 = pad_sequences(train_q2, maxlen=MAX_SEQUENCE_LENGTH)

test_pad_q1 = pad_sequences(test_q1, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_q2 = pad_sequences(test_q2, maxlen=MAX_SEQUENCE_LENGTH)

dev_pad_q1 = pad_sequences(dev_q1, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_q2 = pad_sequences(dev_q2, maxlen=MAX_SEQUENCE_LENGTH)

def save_pickle(fileobj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(fileobj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj


model_data = {'train_q1': train_pad_q1, 'train_q2': train_pad_q2, 'train_label': train_label,
              'test_q1': test_pad_q1, 'test_q2': test_pad_q2, 'test_label': test_label,
              'dev_q1': dev_pad_q1, 'dev_q2': dev_pad_q2, 'dev_label': dev_label}

save_pickle(corpus, './data/corpus.pkl')
save_pickle(model_data, './data/model_data.pkl')
save_pickle(embedding_matrix, './data/embedding_matrix.pkl')
save_pickle(tokenizer, './data/tokenizer.pkl')
