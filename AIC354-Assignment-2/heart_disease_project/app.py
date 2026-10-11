import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title='Heart Disease Predictor', page_icon='❤️', layout='centered')
MODEL_PATH = Path(__file__).parent / 'heart_svc_trained_model.pkl'
SEX = ['Male', 'Female']
CHEST_PAIN = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
REST_ECG = ['Normal', 'ST-T Abnormality', 'LV Hypertrophy']
EXERCISE_ANGINA = ['Yes', 'No']

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders(project_name):
    return {'Sex': LabelEncoder().fit(SEX), 'ChestPain': LabelEncoder().fit(CHEST_PAIN), 'RestECG': LabelEncoder().fit(REST_ECG), 'ExerciseAngina': LabelEncoder().fit(EXERCISE_ANGINA)}
model = load_model(str(MODEL_PATH))
encoders = build_label_encoders('heart_disease_project')
st.title('❤️ Heart Disease Prediction System')
st.markdown('\n    Enter the four clinical attributes below and the trained Support Vector\n    Classifier (SVC) will predict whether heart disease is likely.\n    Trained on the UCI Heart Disease (Cleveland) dataset.\n\n    *Educational demo only. This is not medical advice or a diagnostic tool.*\n    ')
st.divider()
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox('Sex', SEX)
    chest_pain = st.selectbox('Chest Pain Type', CHEST_PAIN, index=3)
with col2:
    rest_ecg = st.selectbox('Resting ECG Result', REST_ECG)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
if st.button('Predict', type='primary', use_container_width=True):
    user_input = pd.DataFrame({'Sex': [sex], 'ChestPain': [chest_pain], 'RestECG': [rest_ecg], 'ExerciseAngina': [exercise_angina]})
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})
    prediction = model.predict(encoded)[0]
    st.divider()
    if prediction == 1:
        st.error('### ⚠️ HEART DISEASE LIKELY')
    else:
        st.success('### ✅ NO HEART DISEASE LIKELY')
    with st.expander('See the encoded feature vector sent to the model'):
        st.dataframe(encoded, hide_index=True)
st.divider()
st.caption('Built for AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus.')
st.caption('Mubashir Ali | Fa24-bse-140 | AIC354 Assignment 2')
