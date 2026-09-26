# ❤️ Heart Disease Prediction using Machine Learning

A Machine Learning-based **Heart Disease Prediction System** that predicts the risk of heart disease using patient health parameters.

The project implements a complete machine learning workflow including **data preprocessing, variable transformation, outlier handling, feature selection, class balancing using SMOTE, feature scaling, model training, evaluation, hyperparameter tuning, model serialization, and Flask-based prediction**.

---
# 🚀 Live Demo

🌐 Try the Heart Disease Prediction Application:
Live Application – https://heart-disease-prediction-x5b5.onrender.com/

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Machine Learning can be used to analyze patient-related medical attributes and identify patterns associated with heart disease.

This project develops a classification pipeline that processes patient data and predicts whether the patient has a **high or low risk of heart disease**.

The trained model is integrated into a **Flask web application**, allowing users to enter patient information and receive a prediction.

---

## 🎯 Objectives

* Perform data preprocessing and cleaning.
* Transform variables using **Yeo-Johnson transformation**.
* Handle outliers using **IQR-based trimming/capping**.
* Remove constant and quasi-constant features.
* Perform feature selection.
* Handle class imbalance using **SMOTE**.
* Scale features using **StandardScaler**.
* Train and compare multiple Machine Learning algorithms.
* Evaluate models using classification metrics.
* Generate **ROC/AUC curves** for model comparison.
* Train a Gaussian Naive Bayes model.
* Save the trained model and scaler using Pickle.
* Deploy the prediction functionality through a Flask web application.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Data Processing

* Pandas
* NumPy
* SciPy

### Visualization

* Matplotlib
* Seaborn

### Web Framework

* Flask

### Model Serialization

* Pickle

### Logging

* Custom Python logging module

---

## 🧠 Machine Learning Workflow

The project follows the following pipeline:

```text
Dataset
   ↓
Train-Test Split
   ↓
Yeo-Johnson Transformation
   ↓
Outlier Handling
   ↓
Feature Selection
   ↓
SMOTE Class Balancing
   ↓
StandardScaler
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model
   ↓
Model Serialization
   ↓
Flask Web Application
   ↓
Heart Disease Prediction
```

---

## 📊 Dataset

The project uses a heart disease dataset containing patient-related attributes.

The target variable represents the presence or absence of heart disease.

The final prediction model uses the following processed features:

```text
age_tr
sex_tr
cp_tr
thalach_tr
oldpeak_tr
slope_tr
thal_tr
```

The project initially processes **13 input features** and performs feature selection before model training. The final preprocessing stage reduces the feature set to **7 features**.

---

## 🔄 Data Preprocessing

### 1. Train-Test Split

The dataset is divided into training and testing datasets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces an **80:20 train-test split**.

---

### 2. Yeo-Johnson Transformation

The project applies the **Yeo-Johnson transformation** to the input variables.

This transformation helps make variables more suitable for Machine Learning models by reducing skewness and improving the distribution of numerical features.

---

### 3. Outlier Handling

After transformation, the project uses the **Interquartile Range (IQR)** method.

Upper and lower limits are calculated using:

```text
Upper Limit = Q3 + 1.5 × IQR
Lower Limit = Q1 - 1.5 × IQR
```

Values outside these limits are capped at the corresponding boundary.

---

### 4. Feature Selection

The project applies feature selection techniques including:

* Constant feature removal
* Quasi-constant feature removal
* Feature reduction based on the selected preprocessing strategy

Features removed during the pipeline include:

```text
fbs_tr
trestbps_tr
chol_tr
exang_tr
ca_tr
restecg_tr
```

The final feature set contains:

```text
age_tr
sex_tr
cp_tr
thalach_tr
oldpeak_tr
slope_tr
thal_tr
```

The feature-selection module uses `VarianceThreshold` for constant and quasi-constant feature identification.

---

### 5. Handling Class Imbalance with SMOTE

The training data is balanced using **Synthetic Minority Oversampling Technique (SMOTE)**.

Before SMOTE:

```text
Class 1: 133
Class 0: 109
```

After SMOTE:

```text
Class 1: 133
Class 0: 133
```

This ensures that both classes have equal representation in the training data.

---

### 6. Feature Scaling

The project uses `StandardScaler` to standardize the selected features.

```python
self.sc_obj = StandardScaler()

self.sc_obj.fit(self.X_train)

self.X_train_scaled = self.sc_obj.transform(self.X_train)
self.X_test_scaled = self.sc_obj.transform(self.X_test)
```

The scaler is saved separately as:

```text
scaled.pkl
```

---

# 🤖 Machine Learning Algorithms

The project contains implementations for multiple classification algorithms:

### 1. K-Nearest Neighbors

```python
KNeighborsClassifier(n_neighbors=5)
```

### 2. Gaussian Naive Bayes

```python
GaussianNB()
```

### 3. Logistic Regression

```python
LogisticRegression()
```

### 4. Decision Tree

```python
DecisionTreeClassifier(criterion='entropy')
```

### 5. Random Forest

```python
RandomForestClassifier(
    criterion='entropy',
    n_estimators=10
)
```

### 6. AdaBoost

```python
AdaBoostClassifier(
    estimator=LogisticRegression(),
    n_estimators=10
)
```

### 7. Gradient Boosting

```python
GradientBoostingClassifier(n_estimators=10)
```

### 8. XGBoost

```python
XGBClassifier()
```

All models are evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

The project also generates ROC curves for model comparison.

---

# 🏆 Model Training

For the final prediction pipeline, **Gaussian Naive Bayes** is trained using the scaled training data.

The model uses:

```python
GaussianNB(
    var_smoothing=np.float64(1e-10)
)
```

The trained model is evaluated using the test dataset.

---

# 📈 Model Performance

Based on the recorded execution results:

### Accuracy

```text
80.33%
```

### Confusion Matrix

```text
[[25  4]
 [ 8 24]]
```

### Classification Report

|                Class | Precision | Recall | F1-Score |
| -------------------: | --------: | -----: | -------: |
|                    0 |      0.76 |   0.86 |     0.81 |
|                    1 |      0.86 |   0.75 |     0.80 |
| **Overall Accuracy** |           |        | **0.80** |

The evaluation was performed on **61 test samples**.

> **Note:** Model performance can vary depending on the dataset, preprocessing implementation, library versions, train-test split, and model configuration.

---

# 🌐 Flask Web Application

The trained Machine Learning model is integrated into a Flask application.

The application provides:

* Home page
* About page
* Patient input form
* Heart disease prediction
* Prediction confidence when `predict_proba()` is available
* Input validation
* Error handling

The Flask backend loads:

```text
model.pkl
scaled.pkl
```

The model receives the following input features:

```text
age_tr
sex_tr
cp_tr
thalach_tr
oldpeak_tr
slope_tr
thal_tr
```

The prediction result is displayed as either:

```text
High risk of Heart Disease
```

or

```text
Low risk of Heart Disease
```

The Flask application also calculates prediction confidence when the loaded model supports probability prediction.

---

# 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── main.py
├── app.py
├── train_all_models.py
├── feature_selection.py
├── variable_transformation.py
├── log_code.py
│
├── heart.csv
│
├── model.pkl
├── scaled.pkl
│
├── templates/
│   ├── index.html
│   └── about.html
│
├── static/
│   └── ...
│
├── logs/
│   └── ...
│
├── requirements.txt
└── README.md
```

> Update the filenames in this structure if your GitHub repository uses different filenames.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

Move into the project directory:

```bash
cd heart-disease-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, the major dependencies used by the project include:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
seaborn
imbalanced-learn
xgboost
flask
```

---

# ▶️ Running the Project

## Step 1 — Train the Model

Run the main training file:

```bash
python main.py
```

The training pipeline performs:

```text
Data Loading
     ↓
Train-Test Split
     ↓
Variable Transformation
     ↓
Outlier Handling
     ↓
Feature Selection
     ↓
SMOTE
     ↓
Standard Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Saving
```

The trained files are saved as:

```text
model.pkl
scaled.pkl
```

---

## Step 2 — Start the Flask Application

Run:

```bash
python app.py
```

The Flask server will start locally.

Open the URL displayed in the terminal, typically:

```text
http://127.0.0.1:5000/
```

---

# 🔮 Prediction Process

The user provides the required patient parameters through the web interface.

The Flask application:

1. Receives the input.
2. Converts the values into numerical format.
3. Arranges the features in the required order.
4. Applies the saved scaler.
5. Passes the processed input to the trained model.
6. Generates the prediction.
7. Displays the prediction and confidence.

---

# 📊 Model Evaluation

The project evaluates models using:

```text
Accuracy
Confusion Matrix
Precision
Recall
F1-Score
ROC Curve
```

The ROC curve module compares the classification models visually using False Positive Rate (FPR) and True Positive Rate (TPR).

---

# 💾 Saved Model Files

After training, the project saves:

### `model.pkl`

Contains the trained Gaussian Naive Bayes model.

### `scaled.pkl`

Contains the fitted `StandardScaler`.

Both files are required by the Flask prediction application.

---

# ⚠️ Important Notes

* The Flask application expects the model and scaler files to be available.
* Input features must follow the same feature order used during model training.
* The preprocessing pipeline used during training should be consistent with the preprocessing applied to prediction data.
* `model.pkl` and `scaled.pkl` are generated model artifacts and may depend on the Python/scikit-learn environment used during training.
* This project is intended for **educational and demonstration purposes** and should not be used as a substitute for professional medical diagnosis.

---

# 🚀 Future Improvements

Possible improvements include:

* Hyperparameter optimization for all classification algorithms.
* Cross-validation for more robust model evaluation.
* Feature importance visualization.
* Improved frontend design.
* Interactive prediction dashboard.
* Database integration for storing predictions.
* Deployment to a cloud platform.
* Model versioning.
* Automated ML pipeline.
* Additional evaluation metrics such as ROC-AUC and Precision-Recall AUC.
* Integration of explainable AI techniques such as SHAP.

---

# 👨‍💻 Author

**Mohammed Awez**

Computer Science Graduate | Machine Learning & AI Enthusiast

---

# ⭐ Acknowledgement

This project was developed as part of Machine Learning practice and project development, covering the complete workflow from **data preprocessing to model deployment**.

If you found this project useful, consider giving the repository a ⭐ on GitHub.
