import os
import sys
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

np.random.seed(123)

_file_path = os.path.dirname(os.path.realpath(__file__))

EMBEDDING_FILE = _file_path + '/embeddings/wiki-news-300d-1M.vec'
MAX_WORDS_NUM = 50000
MAX_SEQ_LEN = 300


class DataLoader:
    def __init__(self):
        self.embeddings_index = {}
        self.data = None

        self.read_data()
        self.load_embedding()
        self.get_embeddings()

    def read_data(self):
        fake = pd.read_csv("data/Fake.csv")
        true = pd.read_csv("data/True.csv")
        fake['label'] = 1
        true['label'] = 0
        self.data = pd.concat([fake, true])

        def count(data, words_to_cnt):
            for sample in data:
                for k, v in sample.items():
                    if k == 'title' or k == 'text':
                        v = v.strip().split(' ')

                        for word in v:
                            words_to_cnt[word] = words_to_cnt.get(word, 0) + 1

            words_to_cnt = sorted(words_to_cnt.items(), key=lambda x: x[1], reverse=True)
            print(words_to_cnt[50000])

        # count(fake_news, {})
        # count(true_news, {})

        print("data count: ", len(self.data))

    def load_embedding(self):
        with open(EMBEDDING_FILE, 'r') as f:
            for line in f:
                line = line.strip().split(" ")

                if len(line) != 301:
                    continue

                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')

                self.embeddings_index[word] = coefs

        print('embedding count: ', len(self.embeddings_index))

    def to_sequence(self):
        train_x, train_y, test_x, test_y = train_test_split(self.data.text, self.data.label, test_size=0.2,
                                                            random_state=123)

        tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
        tokenizer.fit_on_texts(train_x)

        tokenizer_train = tokenizer.texts_to_sequences(train_x)
        x_train = pad_sequences(tokenizer_train, maxlen=MAX_SEQ_LEN)

        tokenizer_test = tokenizer.texts_to_sequences(test_x)
        x_test = pad_sequences(tokenizer_test, maxlen=MAX_SEQ_LEN)

    def get_embeddings(self):
        all_embs = np.stack(self.embeddings_index.values())
        print(all_embs[0])


data_loader = DataLoader()
