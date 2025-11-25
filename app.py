import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# CONFIGURATION & TITLE
st.set_page_config(page_title="Heart Disease Risk", page_icon="📊")
st.title("📊 Heart Disease Risk Predictor")
st.markdown("Enter patient details below to estimate the risk of heart disease")

# LOAD & TRAIN MODEL
# We use @st.cache_resource so the model only trains ONCE when the app starts,
# not every time you click a button.
@st.cache_resource
def train_model():
    try:
        # Load and Clean
        df = pd.read_csv('heart.csv')
        df = df.drop_duplicates()
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Tuned Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=4,
            max_depth=None,
            bootstrap=True,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    
    except FileNotFoundError:
        return None, None

model, scaler = train_model()

if model is None:
    st.error("❌ Could not find 'heart.csv'. Please make sure the dataset is in the same folder as this script.")
    st.stop()

# SIDEBAR: USER INPUTS
st.sidebar.header("Patient Data")

def user_input_features():
    # 1. Age
    age = st.sidebar.slider("Age", 29, 77, 54)
    
    # 2. Sex
    sex_option = st.sidebar.selectbox("Sex", ("Male", "Female"))
    sex = 1 if sex_option == "Male" else 0
    
    # 3. Chest Pain Type (cp)
    cp_option = st.sidebar.selectbox("Chest Pain Type", (
        "Typical Angina (0)",
        "Atypical Angina (1)",
        "Non-anginal Pain (2)",
        "Asymptomatic (3)"
    ))
    cp = int(cp_option.split("(")[1][0])
    
    # 4. Resting Blood Pressure (trestbps)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    
    # 5. Cholesterol (chol)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 246)
    
    # 6. Fasting Blood Sugar (fbs)
    fbs_option = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl?", ("False", "True"))
    fbs = 1 if fbs_option == "True" else 0
    
    # 7. Resting ECG (restecg)
    restecg = st.sidebar.selectbox("Resting ECG Results", (0, 1, 2))
    
    # 8. Max Heart Rate (thalach)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    
    # 9. Exercise Induced Angina (exang)
    exang_option = st.sidebar.radio("Exercise Induced Angina?", ("No", "Yes"))
    exang = 1 if exang_option == "Yes" else 0
    
    # 10. ST Depression (oldpeak)
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
    
    # 11. Slope
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST", (0, 1, 2))
    
    # 12. Number of Major Vessels (ca)
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    
    # 13. Thalassemia (thal)
    thal_option = st.sidebar.selectbox("Thalassemia", (
        "Null (0)", "Fixed Defect (1)", "Normal (2)", "Reversable Defect (3)"
    ))
    thal = int(thal_option.split("(")[1][0])

    # Store in DataFrame
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get input from user
input_df = user_input_features()

# Display input parameters
st.subheader("Patient Summary")
st.write(input_df)

# PREDICTION LOGIC
if st.button("Predict Risk"):
    # Scale the input using the SAME scaler fitted on training data
    input_scaled = scaler.transform(input_df)
    
    # Get Prediction and Probability
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0][1] # Prob of class 1 (Disease)
    
    st.subheader("Results")
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK DETECTED")
        st.write(f"The model predicts a **{prediction_prob*100:.1f}%** probability of heart disease.")
    else:
        st.success(f"✅ LOW RISK / HEALTHY")
        st.write(f"The model predicts a **{prediction_prob*100:.1f}%** probability of heart disease.")
        
    st.info("Note: This is an AI estimation and should not replace professional medical advice.")