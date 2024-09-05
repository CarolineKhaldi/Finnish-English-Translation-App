import streamlit as st
import torch
import pickle
import re
from model import EncoderRNN, AttnDecoderRNN, evaluate, tensorFromSentence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import Lang
from nltk.translate.bleu_score import sentence_bleu

# Load the reference translations from the file
def load_reference_data(eng_fin.txt):
    references = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            fin, eng = line.strip().split('\t')  # Assuming the file is tab-separated
            references[fin] = eng
    return references

# Load the original reference data
reference_translations = load_reference_data('eng_fin.txt')

# Function to calculate BLEU score using reference translations
def calculate_bleu(output_words, sentence):
    if sentence in reference_translations:
        reference = [reference_translations[sentence].split(' ')]  # Use the correct English translation
        candidate = output_words[:-1]  # Ignore the <EOS> token in the output
        bleu_score = sentence_bleu(reference, candidate)
        return bleu_score * 100  # Return percentage BLEU score
    else:
        return 0  # No reference available

# Define the normalizeString function
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)  # Separate punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Remove anything that's not a letter or punctuation
    return s

# Function to show attention with vibrant colors
def show_attention(input_sentence, output_words, attentions):
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size for better readability
    cax = ax.matshow(attentions.numpy(), cmap='plasma')  # Use 'plasma' colormap for vibrant heatmap colors
    fig.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    st.pyplot(fig)

# Updated function to translate, show attention, and calculate BLEU score
def translate_and_show_attention(sentence):
    sentence = normalizeString(sentence)  # Normalize the input sentence
    st.markdown(f"**Normalized sentence:** `{sentence}`")
    
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
        # Display the translated sentence
        st.markdown(f"**Translated sentence:** {' '.join(output_words)}")
        
        # Calculate and display BLEU score using the reference translation
        bleu_score = calculate_bleu(output_words, sentence)
        st.markdown(f"**BLEU Score:** {bleu_score:.2f}%")
        
        show_attention(sentence, output_words, attentions)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Streamlit app main function
def main():
    st.title("Finnish-English Translation App")
    st.write("Translate Finnish sentences into English with BLEU score and attention visualization!")

    sentence = st.text_input("Enter a Finnish sentence:")

    if st.button("Translate"):
        translate_and_show_attention(sentence)

    # Footer with credits
    st.markdown("""
    <hr>
    <center><small>Made with 💡 and ❤️ by [Your Name]</small></center>
    """, unsafe_allow_html=True)

# Load models and language data
encoder = torch.load('encoder_full.pth', map_location=torch.device('cpu'))
decoder = torch.load('decoder_full.pth', map_location=torch.device('cpu'))
encoder.eval()
decoder.eval()

with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)

with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)

if __name__ == '__main__':
    main()
