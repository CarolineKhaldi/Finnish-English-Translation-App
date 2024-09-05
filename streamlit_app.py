import streamlit as st
import torch
import pickle
import re  # For string normalization
from model import EncoderRNN, AttnDecoderRNN, evaluate, tensorFromSentence  # Ensure tensorFromSentence is imported
import matplotlib.pyplot as plt
from model import Lang 

# Define the device globally
device = torch.device('cpu')  # Force the use of CPU

# Load the full models directly, forcing the model to load on CPU
encoder = torch.load('encoder_full.pth', map_location=device)
decoder = torch.load('decoder_full.pth', map_location=device)

encoder.eval()
decoder.eval()

# Load input_lang and output_lang
with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)

with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)

# Define normalizeString function
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)  # Separate punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove anything that's not a letter or punctuation
    return s

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

# Updated Function to translate and show attention (with vocabulary checks)
def translate_and_show_attention(sentence):
    sentence = normalizeString(sentence)  # Normalize the input sentence
    st.write(f"Normalized sentence: {sentence}")
    
    # Check that input_lang and output_lang are loaded correctly
    st.write(f"input_lang n_words: {input_lang.n_words}")
    st.write(f"output_lang n_words: {output_lang.n_words}")

    # Check if the word exists in input_lang vocabulary
    for word in sentence.split(' '):
        if word not in input_lang.word2index:
            st.error(f"Word '{word}' not in input_lang vocabulary.")
            return

    # Check the tokenization process
    try:
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)  # Move to the correct device
        st.write(f"Input tensor size: {input_tensor.size()}")
    except Exception as e:
        st.error(f"Error converting sentence to tensor: {e}")
        return
    
    # Try the evaluation function, pass the device
    try:
        output_words, attentions = evaluate(encoder, decoder, sentence, input_lang, output_lang, device)  # Pass device here
        st.write("Translated sentence:", ' '.join(output_words))
        show_attention(sentence, output_words, attentions)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Streamlit app main function
def main():
    st.title("Finnish-English Translation App")

    sentence = st.text_input("Enter a Finnish sentence:")

    if st.button("Translate"):
        translate_and_show_attention(sentence)

if __name__ == '__main__':
    main()
