# 🫀 Heart Disease Risk Predictor

A machine learning web application that predicts the likelihood of a patient having heart disease based on medical attributes. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
Heart disease is one of the leading causes of death globally. This project aims to assist in early detection by analyzing key physiological factors.

The model uses a **Tuned Random Forest Classifier** trained on patient data to classify individuals as either **"Healthy"** or **"High Risk"** (Disease). It provides a probability score (Risk Score) alongside the classification.

## 🧠 Dataset Details
The dataset contains 14 attributes used for prediction:

* Age: Age in years.

* Sex: 1 = Male, 0 = Female.

* CP: Chest pain type (0-3).

* Trestbps: Resting blood pressure (mm Hg).

* Chol: Serum cholestoral (mg/dl).

* Fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).

* Restecg: Resting electrocardiographic results (0-2).

* Thalach: Maximum heart rate achieved.

* Exang: Exercise induced angina (1 = yes; 0 = no).

* Oldpeak: ST depression induced by exercise relative to rest.

* Slope: The slope of the peak exercise ST segment.

* Ca: Number of major vessels (0-3) colored by flourosopy.

* Thal: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect).

## ⚙️ Features
* **Interactive Web Interface:** User-friendly form to input patient data (Age, BP, Cholesterol, etc.).
* **Real-time Prediction:** Instant classification with probability estimation.
* **Robust Preprocessing:** Handles data scaling and duplicate removal automatically.
* **Tuned Model:** Uses hyperparameters optimized via GridSearchCV for better generalization.

## 📊 Model Performance
After extensive testing and cleaning the dataset (removing ~700 duplicate rows to prevent data leakage), the model achieves:
* **Accuracy:** ~82% - 84% on the test set.
* **Algorithm:** Random Forest Classifier.
* **Key Parameters:** `n_estimators=100`, `min_samples_leaf=4`, `min_samples_split=10`.

## 🛠️ Installation & Local Run

To run this project on your local machine, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/heart-disease-predictor.git](https://github.com/your-username/heart-disease-predictor.git)
    cd heart-disease-predictor
    ```

2.  **Install dependencies**
    Make sure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.
