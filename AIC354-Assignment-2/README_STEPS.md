# AIC354 Assignment 2

**Student:** Mubashir Ali  
**Registration No.:** Fa24-bse-140  
**Course:** Machine Learning Fundamentals  
**Institution:** COMSATS University Islamabad, Lahore Campus

## Task 1: Titanic Passenger Survival

Folder: `titanic_deployment/`

The notebook is included and executed. It follows the instructor's four-input LabelEncoder and SVC approach, using the original sample and ordered 80/20 split.

- `Titanic_Survival_Prediction.ipynb`
- `sample-data.csv`
- `svc_trained_model.pkl`
- `app.py`, `requirements.txt`, `runtime.txt`
- Encoded data, train/test CSVs and model predictions

Held-out accuracy: **80.00%** on 20 test records.

## Task 2: Heart Disease Prediction

Folder: `heart_disease_project/`

The notebook follows the reference's 66-cell structure. Its exact four inputs are **Sex, ChestPain, RestECG and ExerciseAngina**. The target is HeartDisease. The raw Cleveland data, 303-row sample, label encoding, stratified 80/20 split, SVC training, testing, application and feedback sections are included.

- `Heart_Disease_Prediction.ipynb`
- `sample-data.csv`, `processed.cleveland.data`
- `heart_svc_trained_model.pkl`
- `app.py`, `requirements.txt`, `runtime.txt`
- Encoded data, train/test CSVs and model predictions

Held-out accuracy: **81.97%** on 61 test records.

## Run locally

Open the notebook from its matching folder and run the cells in order. To start a web app from that folder:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

All Python code comments and explanatory docstrings have been removed. Explanations remain in separate notebook markdown cells.

## Cloud deployment

Current demo: https://mubashir-ml-assignment-two.mellowllama2.chatgpt.site

The current demo is owner-private and uses the exact trained SVC parameters. All 120 possible categorical input combinations matched Python predictions. Streamlit form submissions were tested for both apps. Full browser visual testing was unavailable.

To create public Python deployments:

1. Upload this package's two project folders to a GitHub repository.
2. Sign in at https://share.streamlit.io and create two apps.
3. Set the main files to `titanic_deployment/app.py` and `heart_disease_project/app.py`, respectively.
4. Select Python 3.12 and deploy.
5. Paste each public URL into the first cell of its corresponding notebook.

Public instructor-accessible deployment remains to be completed. No placeholder URL is presented as a live public app.

## Submission

Upload each notebook, its dataset and its trained model to Google Classroom. Keep the final public deployment URL at the top of each notebook.

## Feedback and attribution

Feedback examples are explicitly illustrative. No actual user reviews or cardiologist consultation are claimed. Suggested Heart Disease improvements retain the four-input constraint.

Titanic starter/sample: Mohsin Afzal and Dr. Rao Muhammad Adeel Nawab, https://github.com/haqnawaz99/ml-teaching-notebooks.

Heart Disease: Janosi, Steinbrunn, Pfisterer and Detrano (1989), UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X. License: CC BY 4.0.
