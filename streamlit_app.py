import streamlit as st
import torch
import pickle
from model import EncoderRNN, AttnDecoderRNN, evaluate  # Make sure to import from your model file
import matplotlib.pyplot as plt

# Load the full models directly
encoder = torch.load('encoder_full.pth')
decoder = torch.load('decoder_full.pth')

encoder.eval()
decoder.eval()

# Load input_lang and output_lang
with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)

with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)

# Function to show attention
def show_attention(input_sentence, output_words, attentions):
    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    st.pyplot(fig)

# Function to translate and show attention
def translate_and_show_attention(sentence):
    output_words, attentions = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    st.write("Translated sentence:", ' '.join(output_words))
    show_attention(sentence, output_words, attentions)

# Streamlit app main function
def main():
    st.title("Finnish-English Translation App")

    sentence = st.text_input("Enter a Finnish sentence:")

    if st.button("Translate"):
        translate_and_show_attention(sentence)

if __name__ == '__main__':
    main()
