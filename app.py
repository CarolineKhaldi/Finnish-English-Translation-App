import streamlit as st
import torch
from your_model_file import EncoderRNN, AttnDecoderRNN, evaluate, input_lang, output_lang
import matplotlib.pyplot as plt

# Load pre-trained models
encoder_model_path = "encoder.pth"
decoder_model_path = "decoder.pth"

# Function to load models
def load_models(encoder_path, decoder_path):
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder

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
    encoder, decoder = load_models(encoder_model_path, decoder_model_path)
    output_words, attentions = evaluate(encoder, decoder, sentence)
    
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
