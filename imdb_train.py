from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot as plt

num_of_words=10000
max_len = 100

training_set, testing_set = imdb.load_data(num_words=num_of_words)
x_train, y_train = training_set
x_test, y_test = testing_set

x_train_padded = sequence.pad_sequences(x_train, maxlen=max_len)
x_test_padded = sequence.pad_sequences(x_test, maxlen=max_len)


model = Sequential()
model.add(Embedding(input_dim=num_of_words, output_dim=128))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=["acc"])

scores = model.fit(x_train_padded, y_train, batch_size=128, epochs=10, validation_data=(x_test_padded, y_test), verbose=0)


plt.plot(range(1,11), scores.history['acc'], label='training')
plt.plot(range(1,11), scores.history['val_acc'], label='testing')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Plt')
plt.legend()
plt.show()

model.save('./Sentiement')