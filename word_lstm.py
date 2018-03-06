# harry_potter.py LSTM to generate text characters
import numpy
np = numpy
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Activation, Lambda
import keras.backend as K
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
f = open('data/shakespeare.txt', 'r')
lines = f.readlines()

# remove numbers
for line in lines:
    try:
        num = int(line.strip())
        if num > 0:
            lines.remove(line)
    except ValueError:
        pass
# remove unnecessary spaces, so that we can focus on just words and sentences
text = ''.join(lines).lower().replace("\r\n", " \r\n ")
most_spaces = "                   "

for i in range(len(most_spaces), 1, -1):
    text = text.replace(most_spaces[:i], " ")

text = text.split(" ")
input_sequences=[]
output = []
print ("Finished reading text")

# every fifth character creates a new sequence of 50 characters.
# Hence the count must stop at len(lines)-41 to get the last sequence
# correctly, along wuth an output value for prediction
for i in range(0, len(text)-11, 1):
    input_sequences.append(text[i:i+10])
    output.append(text[i+10])

print ("Sliced text into sequences")

# encoding
unique_words = list(set(text))
num_unique_words = len(unique_words)
print unique_words
print (num_unique_words)
print ("Encoding")

def word_to_enc(word):
    enc = [0]*num_unique_words
    enc[unique_words.index(word)] = 1
    return enc

def enc_to_word(enc):
    return unique_words[enc.index(1)]

encoded_input = []

for seq in tqdm(input_sequences):
    encoded_seq = []
    for char in seq:
        encoded_seq.append(word_to_enc(char))
    encoded_input.append(encoded_seq)

print ("Finished input encoding")

encoded_output = []
for char in tqdm(output):
    encoded_output.append(word_to_enc(char))

print ("Finished output encoding")


# numpy conversion
print ("Converting to numpy arrays")
encoded_input = numpy.array(encoded_input)
encoded_output = numpy.array(encoded_output)

print (numpy.shape(encoded_input))
print (numpy.shape(encoded_output))



model = Sequential()

model.add(LSTM(200, input_shape=(10,num_unique_words)))

model.add(Dense(num_unique_words, activation='softmax'))

model.compile(optimizer='adam',
          loss='categorical_crossentropy')

print ("Learning")
#checkpointer = ModelCheckpoint(filepath='deep_shakespeare_word_adam.h5', verbose=1, save_best_only=True, monitor='loss')
#hist = model.fit(encoded_input, encoded_output, epochs=50, batch_size=100, callbacks=[checkpointer])
model = load_model('deep_shakespeare_word_adam.h5')

seed_text = "Once upon a time long passed into sweet oblivion \r\n".lower()
print seed_text.split(" ")
def seq_encode(seq):
    encoded_seq = np.zeros((1,10,num_unique_words))
    for char in range(len(seq)):

        encoded_seq[0,char,:] = np.array(word_to_enc(seq[char]))
    return encoded_seq

results_file = open('deep_shakespeare_word_gen.txt', 'w')

all_text = seed_text
for i in tqdm(range(5000)):

    pred = model.predict(seq_encode(all_text.split(" ")[-10:]))
    enc_lst = [0]*num_unique_words
    enc_lst[pred[0].tolist().index(max(pred[0]))] = 1
    all_text += " "+enc_to_word(enc_lst)

print ('no temperature and 20 epochs')
print (all_text)
results_file.write('no temperature and 20 epochs\n')
results_file.write(all_text)
results_file.write('\n')
