import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import grnti_dataset
from keras.preprocessing.text import Tokenizer

import keras.models
import json

# Model configuration
additional_metrics = ['accuracy',"categorical_accuracy"]
batch_size = 128
embedding_output_dims = 15
loss_function = BinaryCrossentropy(from_logits=True)
max_sequence_length = 300
num_distinct_words = 5000
number_of_epochs = 3
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load dataset
texts, grnti = grnti_dataset.get_grnti_data()
x_train, y_train, x_test, y_test = grnti_dataset.load_data_from_arrays(texts,grnti,0.9)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
# print(x_train[0])
# exit()
num_words = 5000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts)
print(x_train.shape)

# print(x_train[1])
print(x_test.shape)
x_train = tokenizer.texts_to_sequences(x_train)
# y_train = tokenizer.texts_to_sequences(y_train)
x_test = tokenizer.texts_to_sequences(x_test)
# y_test = tokenizer.texts_to_sequences(y_test)
# exit()
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(grnti)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

num_classes = np.max(y_train) + 1
total_categories = num_classes
# print('Количество категорий для классификации: {}'.format(num_classes))
# total_categories = num_classes = np.max(y_train) + 1

# Pad all sequences
padded_inputs = pad_sequences(x_train, maxlen=max_sequence_length, value = 0.0) # 0.0 because it corresponds with <PAD>
padded_inputs_test = pad_sequences(x_test, maxlen=max_sequence_length, value = 0.0) # 0.0 because it corresponds with <PAD>

print(y_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
with open("y_test.json",'w', encoding='utf-8') as fp1:
    json.dump(y_test.tolist(),fp1)
print(y_test)
# exit()
# Define the Keras model
model = Sequential()
model.add(Embedding(num_distinct_words, 700, input_length=max_sequence_length))
# model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
# model.add(LSTM(60))
model.add(LSTM(700))
model.add(Dropout(0.3))
model.add(Dense(total_categories, activation='softmax'))

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=additional_metrics)

# Give a summary
model.summary()

# Train the model
history = model.fit(padded_inputs, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# Test the model after training

test_results = model.evaluate(padded_inputs_test, y_test, verbose=True)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')

x = model.predict(padded_inputs_test[1:])
print('result:', x)
with open("res.json",'w', encoding='utf-8') as fp1:
    json.dump(x.tolist(),fp1)
model.save('my_model.h5')
