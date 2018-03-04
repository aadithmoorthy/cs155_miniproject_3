# harry_potter.py LSTM to generate text characters
import numpy
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense
import keras.backend as K
import matplotlib.pyplot as plt


f = open('harry_potter.txt', 'r')
lines = f.readlines()
# remove new lines and unnecessary spaces, so that we can focus on just words and sentences
text = ' '.join(lines).replace("\n","").replace("    "," ").replace("   "," ").replace("  "," ").lower()
input_sequences=[]
output = []
print "Finished reading text"

# every fifth character creates a new sequence of 50 characters.
# Hence the count must stop at len(lines)-51 to get the last sequence
# correctly, along wuth an output value for prediction
for i in range(0, len(text)-51, 5):
    input_sequences.append(text[i:i+50])
    output.append(text[i+50])

print "Sliced text into sequences"

# encoding
unique_chars = list(set(text))
num_unique_chars = len(unique_chars)

print "Encoding"

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

print "Finished input encoding"

encoded_output = []
for char in output:
    encoded_output.append(char_to_enc(char))

print "Finished output encoding"


# numpy conversion
print "Converting to numpy arrays"
encoded_input = numpy.array(encoded_input)
encoded_output = numpy.array(encoded_output)

print numpy.shape(encoded_input)
print numpy.shape(encoded_output)



model = Sequential()

model.add(LSTM(128, input_shape=(50,52)))

model.add(Dense(num_unique_chars, activation='softmax'))

model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy')

print "Learning"
hist = model.fit(encoded_input, encoded_output, nb_epoch=20, batch_size=100)
model.save('deep_shakespeare_rmsprop_b10.h5')

seed_text = 'They each seized a broomstick and kicked off into '.lower()

def seq_encode(seq):
    encoded_seq = []
    for char in seq:
        encoded_seq.append(char_to_enc(char))
    return encoded_seq

results_file = open('deep_shakespeare_gen.txt', 'w')

all_text = seed_text
for i in range(200):
    pred = model.predict(numpy.reshape(numpy.array(seq_encode(all_text[-50:])), (1,50,52)))
    enc_lst = [0]*num_unique_chars
    enc_lst[pred[0].tolist().index(max(pred[0]))] = 1
    all_text += enc_to_char(enc_lst)

print 'no temperature and 20 epochs'
print all_text
results_file.write('no temperature and 20 epochs\n')
results_file.write(all_text)
results_file.write('\n')


# with temperature study
temps = [0.25,0.75,1.5]
epochs_list = [1,5,20]

for t in temps:
    print "Learning " + str(t)
    # Custom Activation with temperature as defined in HW
    def softmax_temp(x):
        return K.softmax(K.log(K.softmax(x))/t)

    # for studies at different stages of learning
    for epochs in epochs_list:
        model = Sequential()

        model.add(LSTM(128, input_shape=(50,52)))

        model.add(Dense(num_unique_chars, activation=softmax_temp))

        model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

        print "Learning " + str(epochs)
        hist = model.fit(encoded_input, encoded_output, nb_epoch=epochs, batch_size=100)

        model.save('deep_shakespeare_'+str(t)+'_'+str(epochs)+'_rmsprop_b100.h5')

        # generation
        all_text = seed_text
        for i in range(200):
            pred = model.predict(numpy.reshape(numpy.array(seq_encode(all_text[-50:])), (1,50,52)))
            enc_lst = [0]*num_unique_chars
            enc_lst[pred[0].tolist().index(max(pred[0]))] = 1
            all_text += enc_to_char(enc_lst)
        print str(t) + ' temperature and epochs '+ str(epochs)
        print all_text
        results_file.write(str(t) + ' temperature and epochs '+ str(epochs)+'\n')
        results_file.write(all_text)
        results_file.write('\n')
