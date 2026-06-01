import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# MUST be the first Streamlit command
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("imdb_rnn_model.keras")

# Load IMDb word index
word_index = imdb.get_word_index()

MAX_LEN = 500
VOCAB_SIZE = 10000

# App UI
st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review and predict whether it is Positive or Negative.")

review = st.text_area(
    "Enter Review",
    height=200
)

# Convert text to IMDb indices
def encode_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        idx = word_index.get(word)

        if idx is not None and (idx + 3) < VOCAB_SIZE:
            encoded.append(idx + 3)
        else:
            encoded.append(2)  # Unknown word

    return encoded

# Prediction button
if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")
    else:

        encoded_review = encode_review(review)

        if len(encoded_review) == 0:
            st.error("Unable to process review.")
        else:

            padded_review = pad_sequences(
                [encoded_review],
                maxlen=MAX_LEN
            )

            padded_review = padded_review.astype("int32")

            prediction = model.predict(
                padded_review,
                verbose=0
            )

            score = float(prediction[0][0])

            st.subheader("Prediction Result")

            if score >= 0.5:
                st.success("😊 Positive Review")
                st.write(f"Confidence: {score:.2%}")
            else:
                st.error("😞 Negative Review")
                st.write(f"Confidence: {(1 - score):.2%}")

            st.progress(score)