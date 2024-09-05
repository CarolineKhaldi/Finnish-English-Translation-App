import streamlit as st
import torch
import pickle
import re
from model import EncoderRNN, AttnDecoderRNN, evaluate, tensorFromSentence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import Lang

# Custom Google Font and custom CSS styling for improvements
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f8ff;
    }
    
    .reportview-container {
        background-color: #f0f8ff;
        color: #333;
        display: flex;
        justify-content: center;
    }
    .stTextInput>div>div>input {
        border: 2px solid #ADD8E6;  /* Light blue border */
        border-radius: 12px;
        padding: 10px;
    }
    .stTextInput>div>div>input:focus {
        outline: none;
        border: 2px solid #87CEEB;  /* Lighter blue border on focus */
    }
    .stButton>button {
        background-color: #87CEEB;  /* Light blue button */
        color: white;
        font-size: 18px;
        border-radius: 12px;
        box-shadow: 2px 2px 5px #888888;
        transition: all 0.3s ease;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #00BFFF;
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
    .progress-bar {
        width: 100%;
        background-color: #f3f3f3;
        border-radius: 20px;
        margin-bottom: 20px;
    }
    .progress-bar-fill {
        height: 24px;
        border-radius: 20px;
        width: 0;
        background-color: #4CAF50;
        transition: width 0.5s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and subtitle with emojis
st.title("🌍 Finnish to English Translation")
st.markdown("Translate Finnish sentences into English with a beautiful attention visualization!")

# Sidebar for instructions
st.sidebar.header("📖 Instructions")
st.sidebar.write("""
- Enter a Finnish sentence in the input box.
- Press the **Translate** button.
- The translation and a colorful attention heatmap will be displayed.
""")

# Sidebar credits
st.sidebar.write("")

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

# Show progress bar when translating
def show_progress_bar():
    st.markdown("""
        <div class="progress-bar">
            <div class="progress-bar-fill"></div>
        </div>
        """, unsafe_allow_html=True)

# Updated Function to translate and show attention (with vocabulary checks)
def translate_and_show_attention(sentence):
    # Normalize the input sentence
    normalized_sentence = normalizeString(sentence)

    # Check the tokenization process and input tensor size
    try:
        input_tensor = tensorFromSentence(input_lang, normalized_sentence)  # No device needed here
        tensor_size = input_tensor.size()
    except Exception as e:
        st.error(f"Error converting sentence to tensor: {e}")
        return

    # Add all details, including normalized sentence and tensor size, into the dropdown list
    with st.expander("🔍 Translation Details"):
        st.markdown(f"""
        **Normalized sentence:** `{normalized_sentence}`<br>
        **Input language words:** {input_lang.n_words}<br>
        **Output language words:** {output_lang.n_words}<br>
        **Input tensor size:** {tensor_size}
        """, unsafe_allow_html=True)

    # Show progress bar
    show_progress_bar()

    # Check if the word exists in input_lang vocabulary
    for word in normalized_sentence.split(' '):
        if word not in input_lang.word2index:
            st.error(f"Word '{word}' not in input_lang vocabulary.")
            return

    # Try the evaluation function
    try:
        output_words, attentions = evaluate(encoder, decoder, normalized_sentence, input_lang, output_lang) 
        # Use a floating box to highlight the translated sentence
        st.markdown(f"""
        <div class="translated-box">
        <b> 📝 Translated sentence:</b> {' '.join(output_words)}
        </div>
        """, unsafe_allow_html=True)

        # Add attention heatmap inside a dropdown (expander)
        with st.expander("Attention Visualization"):
            show_attention(normalized_sentence, output_words, attentions)

    except Exception as e:
        st.error(f"Error during evaluation: {e}")

# Streamlit app main function
def main():
    st.markdown("## Translate a Sentence")
    sentence = st.text_input(" ✍ Enter a Finnish sentence:", help="Write a sentence in Finnish and see its translation!")

    if st.button("📖 Translate"):
        translate_and_show_attention(sentence)

    # Footer with credits
    st.markdown("""
    <hr>
    <center><small>Made by Caroline Marquis </small></center>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
