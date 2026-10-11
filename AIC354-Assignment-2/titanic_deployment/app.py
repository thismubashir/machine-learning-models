import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title='Titanic Survival Predictor', page_icon='🚢', layout='centered')
MODEL_PATH = Path(__file__).parent / 'svc_trained_model.pkl'

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders(project_name):
    encoders = {}
    encoders['PClass'] = LabelEncoder().fit(['First', 'Second', 'Third'])
    encoders['Gender'] = LabelEncoder().fit(['Male', 'Female'])
    encoders['Sibling'] = LabelEncoder().fit(['Zero', 'One', 'Two', 'Three'])
    encoders['Embarked'] = LabelEncoder().fit(['Southampton', 'Cherbourg', 'Queenstown'])
    return encoders
model = load_model(str(MODEL_PATH))
encoders = build_label_encoders('titanic_deployment')
st.title('🚢 Titanic Passenger Survival Predictor')
st.markdown("\n    This is a teaching demo built from the **Titanic Passenger Survival Prediction**\n    notebook. Pick a passenger's details below and the trained Support Vector\n    Classifier (SVC) will predict whether they would have survived.\n\n    *Model accuracy on held-out test data: **80%** (100-row toy dataset).*\n    ")
st.divider()
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox('Passenger Class (PClass)', ['First', 'Second', 'Third'], index=2)
    gender = st.selectbox('Gender', ['Male', 'Female'])
with col2:
    sibling = st.selectbox('Siblings/Spouses Aboard', ['Zero', 'One', 'Two', 'Three'])
    embarked = st.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])
if st.button('Predict Survival', type='primary', use_container_width=True):
    user_input = pd.DataFrame({'PClass': [pclass], 'Gender': [gender], 'Sibling': [sibling], 'Embarked': [embarked]})
    encoded = pd.DataFrame({'PClass': encoders['PClass'].transform(user_input['PClass']), 'Gender': encoders['Gender'].transform(user_input['Gender']), 'Sibling': encoders['Sibling'].transform(user_input['Sibling']), 'Embarked': encoders['Embarked'].transform(user_input['Embarked'])})
    prediction = model.predict(encoded)[0]
    st.divider()
    if prediction == 1:
        st.success('### ✅ SURVIVED')
    else:
        st.error('### ❌ NOT SURVIVED')
    with st.expander('See the encoded feature vector sent to the model'):
        st.dataframe(encoded, hide_index=True)
st.divider()
st.caption('Built for teaching purposes from a 100-row simplified Titanic dataset. Predictions reflect patterns in this small toy dataset, not a rigorous historical analysis.')
st.caption('Mubashir Ali | Fa24-bse-140 | AIC354 Assignment 2')
