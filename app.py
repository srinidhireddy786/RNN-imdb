
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("imdb_rnn_model.keras")

# IMDb word dictionary
word_index = imdb.get_word_index()

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review and predict whether it is Positive or Negative.")

review = st.text_area("Enter Review")

MAX_LEN = 500

def encode_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        if word in word_index:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # unknown word

    return encoded

if st.button("Predict"):

    if review.strip():

        encoded_review = encode_review(review)

        padded_review = pad_sequences(
            [encoded_review],
            maxlen=MAX_LEN
        )

        prediction = model.predict(padded_review)

        score = prediction[0][0]

        st.subheader("Prediction")

        if score >= 0.5:
            st.success(f"Positive Review 😊")
            st.write(f"Confidence: {score:.2%}")
        else:
            st.error(f"Negative Review 😞")
            st.write(f"Confidence: {(1-score):.2%}")