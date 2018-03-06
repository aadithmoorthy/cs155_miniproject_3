from keras.models import load_model
import numpy
from tqdm import tqdm
seed_text = "For thy sweet love remembered such wealth brings,\r\n".lower()
model = load_model('deep_shakespeare_0.25_100_adam_b100.h5', custom_objects={'t':.25})
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

def seq_encode(seq):
    encoded_seq = []
    for char in seq:
        encoded_seq.append(char_to_enc(char))
    return encoded_seq

results_file = open('deep_shakespeare_gen2.txt', 'w')
num_unique_chars=39
all_text = seed_text
for i in tqdm(range(50*50)):
    pred = model.predict(numpy.reshape(numpy.array(seq_encode(all_text[-40:])), (1,40,39)))
    enc_lst = [0]*num_unique_chars
    enc_lst[pred[0].tolist().index(max(pred[0]))] = 1
    all_text += enc_to_char(enc_lst)

print ('no temperature and 100 epochs')
print (all_text)
results_file.write('no temperature and 100 epochs\n')
results_file.write(all_text)
results_file.write('\n')
