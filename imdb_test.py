import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb

maxlen=100
model = keras.models.load_model('./Sentiement')

#dictionary
word_to_id = imdb.get_word_index()

while True:
    text = input("Please enter a review of this film:\n")
    words = text_to_word_sequence(text)
    x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=10000) else 0 for word in words]]
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print(x_test.shape)
    model.predict_classes(x_test)
    y = float(model.predict_classes(x_test))

    if y == 1:
        print("You think this film was bad\n")
    elif y == 0:
        print("You think this film was good\n")
    else:
        print("Unsure\n")
