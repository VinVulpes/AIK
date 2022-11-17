'''
docstring
'''
REBUILD = 0
# TODO:


# 1. Протестировать различные функции активации и кол-во нейронов
# 2. Попробовать другие типы нейронных сетей (которые хорошо работают на малых данных)
# 3. Написать Кейно и попросить БД побольше
# 4. Убрать лишниие комментарии и # print
# 5. посмотреть какие слова оставляет
# 6. рефакторинг кода
# 2466/61 = 40.42
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
    # print("Test size: {}".format(test_size))
    x_train = strings[:]
    y_train = labels[:]
    x_test = []
    y_test = []
    # print("\nTraining set:")
    for _ in range(test_size):
        rand = random.randint(0, len(x_train)-1)
        x_test.append(x_train[rand])
        y_test.append(y_train[rand])
        del x_train[rand]
        del y_train[rand]
    # x_train = strings[test_size:]
    # print("\t - x_train: {}".format(len(x_train)))
    # y_train = labels[test_size:]
    # print("\t - y_train: {}".format(len(y_train)))
    
    # print("\nTesting set:")
    # x_test = strings[:test_size]
    # print("\t - x_test: {}".format(len(x_test)))
    # y_test = labels[:test_size]
    # print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test


def filter_stop_words(train_sentences, stop_words):
    train_sentences2 = []
    for _, sentence in enumerate(train_sentences):
        # print([word for word in sentence.split() if len(word)>2])
        new_sent = ["".join(filter(str.isalpha, word)) for word in sentence.split() if ((word not in stop_words) and (len(word)>2))]
        # new_sent = 
        # new_sent = ''.join([s for s in (new_sent) if s not in '!"#$%&amp;()*+,–-—./:;&lt;=>?@[\\]^_`{|}~\t\n\xa0']) 
        train_sentences2.append(' '.join(new_sent))
    # print(train_sentences2[0])
    return train_sentences2
def clean_text(text):
    global cnt
    text = text.lower()
    snowball_ru = SnowballStemmer(language="russian")
    snowball_en = SnowballStemmer(language="english")
    text_ret = ""
    for word in text.split():
        if(word[0].lower() in 'abcdefghigklmnopqrstuwxyz'):
            text_ret += " " + (snowball_en.stem(word))
            cnt += 1
        else:
            text_ret += " " + (snowball_ru.stem(word))
    # text = " ".join(snowball_ru.stem(word) for word in text.split())

    return text_ret
json_dict=""
cntgrnti=0
cntall=0
grnti = {}
texts = []
texts_grnti = []
with open("theses_grnti.json",encoding='utf-8') as f:
    
    json_dict = json.load(f)
    # # print(json_dict.keys())
    # print("len ",len(json_dict.keys()))
    for key,values in json_dict.items():
        cntall += 1
        if values.get("grnti"):
            cntgrnti += 1
            # if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','59','28','81','82','89','06']:
            if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','27','06','50','89','28']:
                texts.append(values['text'])
                texts_grnti.append(values['grnti'][:2])
            assert len(texts) == len(texts_grnti)
            # if(len(grnti)>500): break
            # # print("lenss:", len(texts), len(texts_grnti))
            # if values.get("grnti") not in texts_grnti:
        if grnti.get(str(values.get("grnti"))[:2]):
            grnti[str(values.get("grnti"))[:2]]+= 1
        else:
            grnti[str(values.get("grnti"))[:2]] = 1
    # # print("json_dict['7']: ",json_dict['7'].keys())
    # # print("json_dict['14949']: ",json_dict['14949'].keys())
    # # print(json_dict[2])
    # # print(json_dict[2]["data"][0] as key, values)
# print(f"Количество кодов: {cntgrnti}\nВсего: {cntall}")
print("grnti: ", grnti) #
for i,code in enumerate(texts_grnti):
    assert (len(code) == 2), code
    # # print(list(grnti)[i])
# 1. Токенизируем слова
nltk.download('stopwords')
# print("len txt:",len(texts))
texts2 = {}
if not REBUILD:
    texts2 = getfrom_file("texts.json")
if not texts2 or len(texts2) != len(texts):
    stopwords_t = stopwords.words('russian')
    stopwords_t += ['эт', 'общ']
    texts = filter_stop_words(texts,stopwords_t)

    print('Stemming')
    cnt = 0
    for i, text in enumerate(texts):
        texts[i] = clean_text(texts[i])
    print("cnt: ",cnt)
    # print("1:",len(texts))
    texts = filter_stop_words(texts,stopwords_t)
    # print("2:",len(texts))
    with open("texts.json",'w',encoding='utf-8') as f:
        json.dump(texts,f)
else:
    texts = texts2
# print(len(texts))
num_words = 4000

##############################;


from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
# textSequences = tokenizer.texts_to_sequences(texts)

X_train, y_train, X_test, y_test = load_data_from_arrays(texts, texts_grnti, train_test_split=0.9)
# X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, texts_grnti, train_test_split=0.9)
# Максимальное количество слов в самом длинном описании заявки
max_words = 0
for desc in texts:
    words = len(desc.split())
    if words > max_words:
        max_words = words
print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

total_unique_words = len(tokenizer.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

maxSequenceLength = max_words

vocab_size = round(total_unique_words/10)




print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=maxSequenceLength)
X_test = pad_sequences(X_test, maxlen=maxSequenceLength)

import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

num_classes = np.max(y_train) + 1
print('Количество категорий для классификации: {}'.format(num_classes))


print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
##################

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# максимальное количество слов для анализа
max_features = vocab_size

print(u'Собираем модель...')
model = Sequential()
model.add(Embedding(max_features, maxSequenceLength))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print (model.summary())

batch_size = 32
epochs = 3

print(u'Тренируем модель...')
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))


def nwtg(activation):
    #'''
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Activation
    from keras.layers import Dropout
    from keras.layers import Convolution3D
    from keras.layers import LSTM
    from keras.layers import Embedding

    # количество эпох\итераций для обучения
    epochs = 3

    # print('Собираем модель...')
    model = Sequential()
    # model =  Convolution3D() #Это вообще там хуета??
    model.add(Dense(2000, input_shape=(num_words,)))
    model.add(Dense(2000, input_shape=(num_words,)))
    # model.add(Dense(512, input_shape=(num_words,)))
    # model.add(Activation('relu'))
    # model.add(Embedding(1000, 150))
    # model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(63, input_shape=(num_words,)))
    # model.add(Dense(512, input_shape=(num_words,)))
    model.add(Activation(activation))
    # model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(total_categories))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # print(model.summary())
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        verbose=1,)
    score = model.evaluate(X_test, y_test,
                        batch_size=32, verbose=1)

    # print()
    print(f'Оценка теста: {score[0]}')
    print(f'Оценка точности модели: {score[1]}')
    return score[1]
    # 2. Превращаем в BoW
    # 3. Запихиваем в нейронку
    # 4. Оно обучается и выдает хуйню
    # 5. ???
    # 6. Profit!
activations = [
'softmax',
'softplus',
'softsign',
'tanh',
'selu',
'elu',
'exponential',
'relu',
'sigmoid',]
res = {}
# for i in activations:
#     res[i] = nwtg(i)
# print(res)


# for i in ['relu','elu','softplus','tanh']:
#     print(i)
#     res[i] = nwtg(i)
# print(res)

# nwtg('softplus')
# {'softmax': 0.23170731961727142, 'softplus': 0.5040650367736816, 'softsign': 0.39024388790130615, 'tanh': 0.5284552574157715, 'selu': 0.57317072153009143, 'elu': 0.47154471278190613, 'exponential': 0.4918699264526367, 'relu': 0.5406504273414612, 'sigmoid': 0.3943089544773102}
