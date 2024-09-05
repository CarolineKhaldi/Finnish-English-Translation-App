import streamlit as st
import torch
import pickle
import re
from model import EncoderRNN, AttnDecoderRNN, evaluate, tensorFromSentence  # Ensure tensorFromSentence is imported
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import Lang 

# App title and subtitle
st.title("🌍 Finnish-English Translation App")
st.markdown("Translate Finnish sentences into English with an attention mechanism visualization!")

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
- Enter a Finnish sentence in the input box.
- Press the **Translate** button.
- The translation and attention heatmap will be displayed.
""")

# Load the full models directly, forcing the model to load on CPU
encoder = torch.load('encoder_full.pth', map_location=torch.device('cpu'))
decoder = torch.load('decoder_full.pth', map_location=torch.device('cpu'))

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
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size for better readability
    cax = ax.matshow(attentions.numpy(), cmap='viridis')  # Use 'viridis' colormap for a nicer heatmap
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    st.pyplot(fig)

# Updated Function to translate and show attention (with vocabulary checks)
def translate_and_show_attention(sentence):
    sentence = normalizeString(sentence)  # Normalize the input sentence
    st.markdown(f"**Normalized sentence:** `{sentence}`")
    
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
        input_tensor = tensorFromSentence(input_lang, sentence)  # No device needed here
        st.write(f"Input tensor size: {input_tensor.size()}")
    except Exception as e:
        st.error(f"Error converting sentence to tensor: {e}")
        return
    
    # Try the evaluation function
    try:
        output_words, attentions = evaluate(encoder, decoder, sentence, input_lang, output_lang)  # No device needed
        st.markdown(f"**Translated sentence:** `{' '.join(output_words)}`")
        show_attention(sentence, output_words, attentions)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Streamlit app main function
def main():
    st.markdown("## Translate a Sentence")
    sentence = st.text_input("Enter a Finnish sentence:")

    if st.button("Translate"):
        translate_and_show_attention(sentence)

if __name__ == '__main__':
    main()
