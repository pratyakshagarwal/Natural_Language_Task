import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

# Define the path to the file
file_path = os.path.join("Next_Word_Predictor", "file.txt")

# Read the contents of the file
with open(file_path, 'r') as file:
    text = file.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

input_sequences = []
for sentence in text.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequences.append(tokenized_sentence[:i+1])


max_len = max([len(x) for x in input_sequences])
# parameters 
vocab_size = len(tokenizer.word_index)
max_len = max([len(x) for x in input_sequences]) 

# some print statement 
# print('vocab szie: ', vocab_size)
# print('max len:', max_len)


# padding input sequnece according to the max lenght
padedd_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
# print('padded sequneces shape:', padedd_sequences.shape)

# creating input and target data
inputs = padedd_sequences[:, :-1]
targets = padedd_sequences[:,-1]
categorial_targets = to_categorical(targets, num_classes=vocab_size+1)

# some print statement
# print('input shape: ',inputs.shape)
# print('target shape:',targets.shape)
# print(targets)
# print('y categorial shape:', categorial_targets.shape)