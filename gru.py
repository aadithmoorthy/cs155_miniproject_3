# harry_potter.py LSTM to generate text characters
import numpy
from keras.models import Sequential
from keras.models import load_model
from keras.layers import GRU, Dense, Activation, Lambda
import keras.backend as K
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

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
text = ''.join(lines).lower().replace("\n ", "\n").replace("\n\n", "\n")
most_spaces = "                   "

for i in range(len(most_spaces), 1, -1):
    text = text.replace(most_spaces[:i], " ")
print (text)
input_sequences=[]
output = []
print ("Finished reading text")

# every fifth character creates a new sequence of 50 characters.
# Hence the count must stop at len(lines)-41 to get the last sequence
# correctly, along wuth an output value for prediction
for i in range(0, len(text)-41, 5):
    input_sequences.append(text[i:i+40])
    output.append(text[i+40])

print ("Sliced text into sequences")

# encoding
unique_chars = list(set(text))
num_unique_chars = len(unique_chars)
print (num_unique_chars)
print ("Encoding")

def char_to_enc(char):
    enc = [0]*num_unique_chars
    enc[unique_chars.index(char)] = 1
    return enc

def enc_to_char(enc):
    return unique_chars[enc.index(1)]

encoded_input = []

for seq in input_sequences:
    encoded_seq = []
    for char in seq:
        encoded_seq.append(char_to_enc(char))
    encoded_input.append(encoded_seq)

print ("Finished input encoding")

encoded_output = []
for char in output:
    encoded_output.append(char_to_enc(char))

print ("Finished output encoding")


# numpy conversion
print ("Converting to numpy arrays")
encoded_input = numpy.array(encoded_input)
encoded_output = numpy.array(encoded_output)

print (numpy.shape(encoded_input))
print (numpy.shape(encoded_output))



model = Sequential()

model.add(GRU(200, input_shape=(40,39)))

model.add(Dense(num_unique_chars, activation='softmax'))

model.compile(optimizer='adam',
          loss='categorical_crossentropy')

print ("Learning")
checkpointer = ModelCheckpoint(filepath='deep_shakespeare_gru_adam.h5', verbose=1, save_best_only=True, monitor='loss')
hist = model.fit(encoded_input, encoded_output, nb_epoch=100, batch_size=100, callbacks=[checkpointer])
model = load_model('deep_shakespeare_gru_adam.h5')

seed_text = "shall i compare thee to a summer's day?\n".lower()

def seq_encode(seq):
    encoded_seq = []
    for char in seq:
        encoded_seq.append(char_to_enc(char))
    return encoded_seq

results_file = open('deep_shakespeare_gen_gru.txt', 'w')

all_text = seed_text
for i in range(50*50):
    pred = model.predict(numpy.reshape(numpy.array(seq_encode(all_text[-40:])), (1,40,39)))
    enc_lst = [0]*num_unique_chars
    enc_lst[pred[0].tolist().index(max(pred[0]))] = 1
    all_text += enc_to_char(enc_lst)

print ('no temperature and 20 epochs')
print (all_text)
results_file.write('no temperature and 20 epochs\n')
results_file.write(all_text)
results_file.write('\n')
