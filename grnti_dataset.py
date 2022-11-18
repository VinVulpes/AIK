from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import numpy as np
import random
import os
import json
REBUILD = 0

def getfrom_file(filename):
    if os.path.isfile(filename):
        with open(filename,'r',encoding='utf-8') as file_io:
            return json.load(file_io)
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

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    # return np.stack(np.array(x_train), axis=0),np.stack(np.array(y_train), axis=0), np.stack(np.array(x_test), axis=0) ,np.stack(np.array(y_test), axis=0)


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
    text = text.lower()
    snowball_ru = SnowballStemmer(language="russian")
    snowball_en = SnowballStemmer(language="english")
    text_ret = ""
    for word in text.split():
        if(word[0].lower() in 'abcdefghigklmnopqrstuwxyz'):
            text_ret += " " + (snowball_en.stem(word))
        else:
            text_ret += " " + (snowball_ru.stem(word))
    # text = " ".join(snowball_ru.stem(word) for word in text.split())

    return text_ret
def get_grnti_data():
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
        for _,values in json_dict.items():
            cntall += 1
            if values.get("grnti"):
                cntgrnti += 1
                # if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','59','28','81','82','89','06']:
                # if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','27','06','50','89','28']:
                # if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','89']:
                # if values.get("grnti")[:2].isdigit() and values.get("grnti")[:2] in ['55','30','89','06','50','27','28']:
                if values.get("grnti")[:2].isdigit():
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
    srt = sorted(grnti,key=grnti.get,reverse=True)
    print("grnti: ", [val+":"+str(grnti[val]) for val in srt if val.isdigit()]) #

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
        for i, _ in enumerate(texts):
            texts[i] = clean_text(texts[i])
        print("cnt: ",cnt)
        # print("1:",len(texts))
        texts = filter_stop_words(texts,stopwords_t)
        # print("2:",len(texts))
        with open("texts.json",'w',encoding='utf-8') as f:
            json.dump(texts,f)
    else:
        texts = texts2
    return texts, texts_grnti
