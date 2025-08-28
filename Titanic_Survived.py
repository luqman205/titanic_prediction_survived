import streamlit as st
import pandas as pd
import pickle

# Load trained model

from pathlib import Path, PurePath
import pickle

MODEL_PATH = Path(__file__).resolve().parent / "titanic_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


MODEL_PATH = Path(__file__).resolve().parent / "titanic_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --- Page config ---
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background: #f8fafc;
        }
        .title {
            font-size: 8rem;
            text-align: center;
            font-weight: bold;
            color: #fff;
        }
        .subtitle {
            text-align: center;
            color: #fff;
            font-size: 1.4rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #1e40af;
            color: white;
            border-radius: 8px;
            font-size: 1.1rem;
            padding: 0.6em 1.5em;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<p class="title">🚢 Titanic Survival Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter passenger details below and find out if they might have survived.</p>',
            unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", min_value=0, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- DataFrame ---
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked],
})

# Show user input
with st.expander("🔎 See Passenger Data"):
    st.dataframe(input_df, use_container_width=True)

# --- Prediction ---
if st.button("✨ Predict Survival"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🎉 The passenger **would have survived!** 🟢")
    else:
        st.error("💀 Unfortunately, the passenger **would not have survived.** 🔴")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:gray;">Made with ❤️ LUQMAN ARIF</p>',
    unsafe_allow_html=True
)
