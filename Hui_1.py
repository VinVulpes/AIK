'''
docstring
'''
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pymorphy2
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import random
import os
import json
json_dict=""
cntgrnti=0
cntall=0
grnti = {}
texts = []
texts_grnti = []

def getfrom_file(filename):
    if os.path.isfile(filename):
        with open(filename,'r',encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}
# Sp
def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))
    x_train = strings[:]
    y_train = labels[:]
    x_test = []
    y_test = []
    print("\nTraining set:")
    for _ in range(test_size):
        rand = random.randint(0, len(x_train)-1)
        x_test.append(x_train[rand])
        y_test.append(y_train[rand])
        del x_train[rand]
        del y_train[rand]
    # x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    # y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))
    
    print("\nTesting set:")
    # x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    # y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test


def filter_stop_words(train_sentences, stop_words):
    train_sentences2 = []
    for _, sentence in enumerate(train_sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences2.append(' '.join(new_sent))
    return train_sentences2
def clean_text(text):
    # text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    snowball = SnowballStemmer(language="russian")
    # text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    # text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
    # text = " ".join(ma.parse((word))[0].normal_form for word in text.split())

    text = " ".join(snowball.stem(word) for word in text.split())

    # text = ' '.join(word for word in text.split() if len(word)>3)
    # text = text.encode("utf-8")

    return text

with open("theses_grnti.json",encoding='utf-8') as f:
    json_dict = json.load(f)
    # print(json_dict.keys())
    print("len ",len(json_dict.keys()))
    for key,values in json_dict.items():
        cntall += 1
        if values.get("grnti"):
            cntgrnti += 1
            if values.get("grnti")[:2].isdigit():
                texts.append(values['text'])
                texts_grnti.append(values['grnti'][:2])
            assert len(texts) == len(texts_grnti)
            # if(len(grnti)>500): break
            # print("lenss:", len(texts), len(texts_grnti))
            # if values.get("grnti") not in texts_grnti:
        if grnti.get(str(values.get("grnti"))[:2]):
            grnti[str(values.get("grnti"))[:2]]+= 1
        else:
            grnti[str(values.get("grnti"))[:2]] = 1
    # print("json_dict['7']: ",json_dict['7'].keys())
    # print("json_dict['14949']: ",json_dict['14949'].keys())
    # print(json_dict[2])
    # print(json_dict[2]["data"][0] as key, values)
print(f"Количество кодов: {cntgrnti}\nВсего: {cntall}")
print("grnti_len: ", len(grnti))
for i,code in enumerate(texts_grnti):
    assert (len(code) == 2), code
    # print(list(grnti)[i])
# 1. Токенизируем слова
nltk.download('stopwords')
print("len txt:",len(texts))
# texts2 = {}
texts2 = getfrom_file("texts.json")
if not texts2 or len(texts2) != len(texts):
    texts = filter_stop_words(texts,stopwords.words('russian'))
    for i, text in enumerate(texts):
        texts[i] = clean_text(texts[i])
    with open("texts.json",'w',encoding='utf-8') as f:
        json.dump(texts,f)
else:
    texts = texts2
print(len(texts))
num_words = 3000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&amp;()*+,-—./:;&lt;=>?@[\\]^_`{|}~\t\n\xa0', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

# x_train_tokenized = tokenizer.texts_to_matrix(texts, mode='count')
# x_train = pad_sequences(x_train_tokenized, maxlen=num_words)
# print(x_train_tokenized)
# print(x_train)

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(texts)
X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, texts_grnti, train_test_split=0.90)
total_words = len(tokenizer.word_index)
with open("words",'w',encoding='utf-8') as f:
    json.dump(tokenizer.word_index,f)
print(f'В словаре {total_words} слов')

# количество наиболее часто используемых слов
# num_words = 500

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)


import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(texts_grnti)
for i, n in enumerate(y_test):
    assert n in texts_grnti, n
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

num_classes = np.max(y_train) + 1
print('Количество категорий для классификации: {}'.format(num_classes))
# total_categories = num_classes = np.max(y_train) + 1
total_categories = num_classes = len(texts_grnti) + 1


print('Преобразуем категории в матрицу двоичных чисел '
      '(для использования categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

##################


#'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Convolution3D

# количество эпох\итераций для обучения
epochs = 10

print('Собираем модель...')
model = Sequential()
# model =  Convolution3D() #Это вообще там хуета??
# model.add(Dense(1000, input_shape=(num_words,)))
# model.add(Dense(512, input_shape=(num_words,)))
# model.add(Activation('relu'))
model.add(Dense(63, input_shape=(num_words,)))
# model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('softsign'))
# model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(total_categories))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    verbose=1,)
score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)

#'''
####################


# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM

# epochs = 2
# # максимальное количество слов для анализа
# vocab_size = 4
# max_features = vocab_size
# maxSequenceLength = 3
# print(u'Собираем модель...')
# model = Sequential()
# model.add(Embedding(max_features, maxSequenceLength))
# model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(num_classes, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# print (model.summary())

####################
'''
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# создание модели
model = Sequential()
# Добавляем слой
model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape=(num_words,)))
# Второй сверточный слой
model.add(Conv2D(32, kernel_size=1, activation='relu'))
# Создаем вектор для полносвязной сети.
model.add(Flatten())
# Создадим однослойный перцептрон
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=X.shape[1:])) ???
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
print(hist.history)
'''

####################
'''
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D,Activation
model = Sequential()
max_features = 1
maxSequenceLength = 20
filters = 30
kernel_size = 1
hidden_dims = 30
model.add(Embedding(max_features, maxSequenceLength))
model.add(Dropout(0.2))
#convlution layer 10
model.add(Conv1D(filters,
kernel_size,
padding='valid',
activation='relu', strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
#hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout (0.2))
model.add(Activation('relu'))
#output Layer:
model.add(Dense(2469))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)
print(hist.history)
#tf.reshape(data, [25, 25])
'''
####################
print()
print(f'Оценка теста: {score[0]}')
print(f'Оценка точности модели: {score[1]}')
# 2. Превращаем в BoW
# 3. Запихиваем в нейронку
# 4. Оно обучается и выдает хуйню
# 5. ???
# 6. Profit!