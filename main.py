import os
import sys
import re
import string

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalAveragePooling1D, Embedding, Dense, Activation, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

np.random.seed(123)

_file_path = os.path.dirname(os.path.realpath(__file__))

EMBEDDING_FILE = _file_path + '/embeddings/wiki-news-300d-1M.vec'
MAX_WORDS_NUM = 50000

MAX_SEQ_LEN = 300
EMBEDDING_SIZE = 300
BATCH_SIZE = 256
EPOCHS = 10


class DataProcess:
    def __init__(self):
        self.data_cleaner = DataCleaner()

        self.embeddings_index = {}
        self.tokenizer = None
        self.embedding_matrix = None
        self.token_num = None
        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None

        data = self.read_data()
        self.to_sequence(data)

        word_to_embedding = self.load_embedding()
        self.calc_embedding_matrix(word_to_embedding)

    def read_data(self):
        fake = pd.read_csv("data/Fake.csv")
        true = pd.read_csv("data/True.csv")
        fake['label'] = 1
        true['label'] = 0
        data = pd.concat([fake, true])

        data['text'] = data['text'] + ' ' + data['title']
        data['text'] = data['text'].apply(self.data_cleaner.denoise_text)
        del data['title']
        del data['subject']
        del data['date']

        # wordcloud visualize
        plt.figure(figsize=(20, 20))
        # fake
        wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(
            ' '.join(data[data.label == 1].text))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig('fake_wc.png')
        # real
        wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=STOPWORDS).generate(
            ' '.join(data[data.label == 0].text))
        plt.imshow(wc, interpolation='bilinear')
        plt.savefig('real_wc.png')

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

        print("data count: ", len(data))

        return data

    def load_embedding(self):
        word_to_embedding = {}

        with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(" ")

                if len(line) != 301:
                    continue

                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')

                word_to_embedding[word] = coefs

        print('embedding count: ', len(word_to_embedding))
        return word_to_embedding

    def to_sequence(self, data):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data.text, data.label,
                                                                                test_size=0.2,
                                                                                random_state=123)

        self.tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
        self.tokenizer.fit_on_texts(self.train_x)

        tokenizer_train = self.tokenizer.texts_to_sequences(self.train_x)
        self.train_x = pad_sequences(tokenizer_train, maxlen=MAX_SEQ_LEN)

        tokenizer_test = self.tokenizer.texts_to_sequences(self.test_x)
        self.test_x = pad_sequences(tokenizer_test, maxlen=MAX_SEQ_LEN)

    def calc_embedding_matrix(self, word_to_embedding):
        all_embs = np.stack(word_to_embedding.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_to_index = self.tokenizer.word_index
        self.token_num = min(MAX_WORDS_NUM, len(word_to_index)) + 1
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.token_num, EMBEDDING_SIZE))

        for word, idx in word_to_index.items():
            if idx >= MAX_WORDS_NUM:
                continue

            embedding_vector = word_to_embedding.get(word, None)
            if embedding_vector is not None:
                self.embedding_matrix[idx] = embedding_vector

        print('embedding matrix cnt: ', len(self.embedding_matrix))

    def get_token_nums(self):
        return self.token_num

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_data(self):
        return self.train_x, self.test_x, self.train_y, self.test_y


class DataCleaner:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        self.stop.update(punctuation)

    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def rm_between_square_brackets(self, text):
        return re.sub('\[[^]]*\]', "", text)

    def rm_url(self, text):
        return re.sub(r'http\S+', '', text)

    def rm_stopwords(self, text):
        res = []
        for i in text.split():
            if i.strip().lower() not in self.stop:
                res.append(i.strip())

        return ' '.join(res)

    def denoise_text(self, text):
        text = self.strip_html(text)
        text = self.rm_between_square_brackets(text)
        text = self.rm_url(text)
        text = self.rm_stopwords(text)

        return text


class Models:
    """define neural network"""

    def __init__(self,
                 num_words,
                 embedding_matrix,
                 embedding_size=300,
                 max_seq_len=300):
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix

    def fast_text(self):
        input = Input(shape=(self.max_seq_len,), dtype='int32')
        embedded_sequences = Embedding(self.num_words,
                                       self.embedding_size,
                                       weights=[self.embedding_matrix],
                                       input_length=self.max_seq_len)(input)  # batch_size * max_seq * embedding_dim
        hidden = GlobalAveragePooling1D()(embedded_sequences)  # batch_size * embedding_dim
        output = Dense(1, activation='sigmoid')(hidden)  # b * 1

        model = Model(inputs=input, outputs=output)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy']
                      )

        print(model.summary())

        return model


if __name__ == '__main__':
    data_loader = DataProcess()

    num_words = data_loader.get_token_nums()
    train_x, test_x, train_y, test_y = data_loader.get_data()
    print('train cnt: ', len(train_x))
    print('test cnt: ', len(test_x))

    embedding_matrix = data_loader.get_embedding_matrix()

    model_repo = Models(num_words,
                        embedding_matrix,
                        EMBEDDING_SIZE,
                        MAX_SEQ_LEN
                        )

    model = model_repo.fast_text()
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=1e-5)
    history = model.fit(train_x, train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=(test_x, test_y),
                        epochs=EPOCHS,
                        callbacks=[learning_rate_reduction],
                        verbose=1)

    print('Accuracy of the model on Training Data is - ', model.evaluate(train_x, train_y)[1])
    print('Accuracy of the model on Testing Data is - ', model.evaluate(test_x, test_y)[1])

    pred = model.predict(test_x)
    print(pred[:10])
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    print(classification_report(test_y, pred, target_names=['Not Fake', 'Fake']))

    cm = confusion_matrix(test_y, pred)
    print(cm)

    epochs = [i for i in range(EPOCHS)]
    fig, ax = plt.subplots(1, 2)

    train_acc = history.history['acc']
    train_loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, label='Training Acc')
    ax[0].plot(epochs, val_acc, label='Testing Acc')
    ax[0].set_title('Training & Testing Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Acc")

    ax[1].plot(epochs, train_loss, label='Training Loss')
    ax[1].plot(epochs, val_loss, label='Testing Loss')
    ax[1].set_title('Training & Testing Loss')
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Loss")

    plt.savefig('fig_after_denoise.png')
