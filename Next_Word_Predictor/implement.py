import time
import numpy as np
from keras.models import load_model
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from data import  max_len, tokenizer


def get_prediction(text, model, times):
    for i in range(times):
        tokenized_sentence = tokenizer.texts_to_sequences([text])[0]
        print(len(tokenized_sentence))
        paded_sequnece = pad_sequences([tokenized_sentence], maxlen=max_len, padding='pre')
        print(paded_sequnece.shape)
        pos = np.argmax(model(paded_sequnece))
        text = text + ' ' + tokenizer.index_word[pos]
    return text

def main_implement(model, times):
   st.title('Text Generation')
   prompt = st.text_input('Input Prompt', placeholder='about the program')

   if prompt:
      if st.button('Generate'):
        with st.spinner('Generating'):
            text = get_prediction(prompt, model=model, times=times)
            st.write(text)


if __name__ == '__main__':
    model = load_model('Next_Word_Predictor/next_word_prediction_model')
    text = 'The fees of '
    main_implement(model=model, times=20)