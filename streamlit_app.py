import streamlit as st
import torch
import pickle
import re
from model import EncoderRNN, AttnDecoderRNN, evaluate, tensorFromSentence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import Lang 
from nltk.translate.bleu_score import sentence_bleu  # Import BLEU score calculation

# App-wide CSS for centering content and enhancing UI
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f8ff;
        color: #333;
        display: flex;
        justify-content: center;
    }
    .stTextInput, .stButton {
        max-width: 80%;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        box-shadow: 2px 2px 5px #888888;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 2px 2px 10px #666666;
    }
    input {
        border-radius: 8px;
        padding: 10px;
        width: 100%;
    }
    .translated-box {
        background-color: #ffebcd;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        animation: fadeIn 0.8s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and subtitle with emojis
st.title("🌍 Finnish-English Translation App")
st.markdown("Translate Finnish sentences into English with a beautiful attention visualization!")

# Sidebar for instructions
st.sidebar.header("📖 Instructions")
st.sidebar.write("""
- Enter a Finnish sentence in the input box.
- Press the **Translate** button.
- The translation and a colorful attention heatmap will be displayed.
""")

# Sidebar credits
st.sidebar.write("Made with 💡 and ❤️ by [Your Name]")

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

# Function to calculate BLEU score
def calculate_bleu(output_words, sentence):
    reference = [sentence.split(' ')]  # Reference translation
    candidate = output_words[:-1]  # Ignore the <EOS> token in the output
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score * 100  # Return percentage BLEU score

# Updated Function to translate and show attention (with vocabulary checks and BLEU score)
def translate_and_show_attention(sentence):
    sentence = normalizeString(sentence)  # Normalize the input sentence
    st.markdown(f"**Normalized sentence:** `{sentence}`")
    
    # Display language information in a card
    st.markdown("""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
        <b>Input language words:</b> {}<br>
        <b>Output language words:</b> {}
        </div>
    """.format(input_lang.n_words, output_lang.n_words), unsafe_allow_html=True)

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
        # Use a floating box to highlight the translated sentence
        st.markdown(f"""
        <div class="translated-box">
        <b>Translated sentence:</b> {' '.join(output_words)}
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate and display BLEU score
        bleu_score = calculate_bleu(output_words, sentence)
        st.markdown(f"**BLEU Score:** {bleu_score:.2f}%")
        
        show_attention(sentence, output_words, attentions)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Streamlit app main function
def main():
    st.markdown("## 🎨 Translate a Sentence")
    sentence = st.text_input("Enter a Finnish sentence:", help="Write a sentence in Finnish and see its translation!")

    if st.button("✨ Translate"):
        translate_and_show_attention(sentence)

    # Footer with credits
    st.markdown("""
    <hr>
    <center><small>Made with 💡 and ❤️ by [Your Name]</small></center>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
