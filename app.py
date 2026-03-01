import streamlit as st
import pickle
import re
import numpy as np


st.set_page_config(
    page_title="Veda AI",
    page_icon="🎓",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #4ea8de;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #c9d1d9;
}
.section-title {
    font-size: 26px;
    font-weight: bold;
    color: #4ea8de;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("difficulty_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

col1, col2, col3 = st.columns([1.5,4,1])

with col1:
    st.image("media/college_logo.png", width=120)

with col2:
    st.markdown('<div class="main-title">Veda AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Exam Difficulty Analyzer</div>', unsafe_allow_html=True)

with col3:
    st.image("media/cbse_logo.png", width=90)

st.divider()

st.markdown('<div class="section-title">🧪 Example Questions</div>', unsafe_allow_html=True)

st.markdown("""
1. What is a primary key? *(SQL)*
2. Define a list in Python. *(Python)*
3. Write a Python program to reverse a string. *(Python)*
4. Explain the difference between WHERE and HAVING. *(SQL)*
5. Describe file handling in Python. *(Python)*
6. Write a SQL query to perform INNER JOIN. *(SQL)*
7. Analyze the time complexity of binary search. *(Data Structures)*
8. Design a student record management system using file handling. *(Python)*
9. Compare normalization and denormalization with examples. *(DBMS)*
10. Explain TCP/IP model layers. *(Networking)*
""")

st.divider()


st.markdown('<div class="section-title">🔍 Live Prediction</div>', unsafe_allow_html=True)

question_input = st.text_area(
    "Enter Question Text",
    height=120
)

topic_list = ["Python", "SQL", "DBMS", "Networking", "Data Structures", "General CS"]
topic_input = st.selectbox("Select Topic", topic_list)

predict_button = st.button("Predict Difficulty")


if predict_button:

    if question_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        combined_text = clean_text(question_input + " " + topic_input)
        vectorized_text = vectorizer.transform([combined_text])

        prediction = model.predict(vectorized_text)[0]
        confidence = np.max(model.predict_proba(vectorized_text))

        st.divider()
        st.markdown("## 📊 Prediction Result")

        if prediction.lower() == "easy":
            st.success("🟢 EASY")
        elif prediction.lower() == "medium":
            st.info("🟡 MEDIUM")
        else:
            st.error("🔴 HARD")

        st.write(f"Confidence Score: **{round(confidence*100,2)}%**")


st.divider()
st.markdown('<div class="section-title">📊 Dataset Analytics</div>', unsafe_allow_html=True)

colX, colY = st.columns(2)

with colX:
    st.image("media/pic2.png", use_container_width=True)

with colY:
    st.image("media/pic3.png", use_container_width=True)

st.image("media/pic4.png", use_container_width=True)


st.markdown("---")
st.caption("Developed as part of GenAI | Veda")