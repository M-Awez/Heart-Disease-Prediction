# ❤️ Heart Disease Prediction Using Machine Learning

A Machine Learning project that predicts the risk of heart disease based on patient health parameters. The project implements a complete ML pipeline including data preprocessing, feature transformation, feature selection, class balancing, feature scaling, model training, evaluation, and Flask deployment.

---

## 📌 Overview

Heart disease is one of the major health concerns worldwide. This project uses Machine Learning algorithms to analyze patient-related features and predict whether a person is at risk of heart disease.

The project includes:

- Data preprocessing
- Yeo-Johnson transformation
- Outlier handling using IQR
- Feature selection
- SMOTE-based class balancing
- Standardization using StandardScaler
- Multiple Machine Learning algorithms
- Model evaluation
- Flask web application
- Saved ML model and scaler

---

## 🚀 Features

- ❤️ Heart disease risk prediction
- 📊 Multiple ML algorithm comparison
- 🔄 Automated data preprocessing
- ⚖️ Handling of imbalanced data using SMOTE
- 📏 Feature scaling using StandardScaler
- 📈 Model evaluation using classification metrics
- 🌐 Flask-based web application
- 💾 Saved model using Pickle
- 📝 Logging of ML pipeline steps

---

## 🧠 Machine Learning Workflow

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
SMOTE
   ↓
StandardScaler
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Flask Prediction Application
📊 Dataset

The project uses the heart.csv dataset.

Dataset Split
Dataset	Samples
Training	242
Testing	61
Total	303

The dataset contains 13 input features and one target variable.

🔧 Data Preprocessing
1. Train-Test Split

The dataset is divided into training and testing sets using an 80:20 ratio.

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
2. Yeo-Johnson Transformation

Yeo-Johnson transformation is applied to the input variables to transform their distributions.

3. Outlier Handling

Outliers are handled using the Interquartile Range (IQR) method.

IQR = Q3 - Q1

Upper Limit = Q3 + 1.5 × IQR
Lower Limit = Q1 - 1.5 × IQR
4. Feature Selection

Constant, quasi-constant, and selected unnecessary features are removed.

Final features used by the model:

age_tr
sex_tr
cp_tr
thalach_tr
oldpeak_tr
slope_tr
thal_tr
5. SMOTE

SMOTE is applied to the training data to balance the target classes.

Before SMOTE:
Class 0 → 109
Class 1 → 133

After SMOTE:
Class 0 → 133
Class 1 → 133
6. Standardization

The final features are standardized using StandardScaler.

The scaler is saved as:

scaled.pkl
🤖 Machine Learning Algorithms

The project implements the following algorithms:

K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Logistic Regression
Decision Tree
Random Forest
AdaBoost
Gradient Boosting
XGBoost
🏆 Final Model

Gaussian Naive Bayes was used as the final prediction model.

GaussianNB(
    var_smoothing=1e-10
)

The trained model is saved as:

model.pkl
📈 Model Performance

The final model achieved an accuracy of:

80.33%

on the 61-sample test set.

Confusion Matrix
[[25  4]
 [ 8 24]]
Classification Report
Class	Precision	Recall	F1-Score
0	0.76	0.86	0.81
1	0.86	0.75	0.80
Accuracy			0.80
🌐 Flask Web Application

The trained model is integrated into a Flask web application.

Application Flow
User Input
    ↓
Flask Backend
    ↓
Input Validation
    ↓
StandardScaler
    ↓
Gaussian Naive Bayes
    ↓
Prediction
    ↓
Heart Disease Risk
Input Features

The application accepts the following features:

Feature	Description
Age	Patient age
Sex	Patient sex
CP	Chest pain type
Thalach	Maximum heart rate achieved
Oldpeak	ST depression
Slope	Slope of peak exercise ST segment
Thal	Thalassemia
Prediction

The application displays:

High risk of Heart Disease

or

Low risk of Heart Disease

It also displays prediction confidence when supported by the model.

📁 Project Structure
Heart-Disease-Prediction/
│
├── static/
│   ├── vt_logo.png
│   └── vt_logo1.jpeg
│
├── templates/
│   ├── about.html
│   └── index.html
│
├── logs/
│   ├── feature_selection.log
│   ├── main.log
│   ├── train_all_models.log
│   └── variable_transformation.log
│
├── app.py
├── feature_selection.py
├── heart.csv
├── log_code.py
├── main.py
├── model.pkl
├── scaled.pkl
├── train_all_models.py
├── variable_transformation.py
├── requirements.txt
└── Procfile
🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
XGBoost
Imbalanced-learn
Data Processing
Pandas
NumPy
SciPy
Visualization
Matplotlib
Web Development
Flask
HTML
CSS
JavaScript
Model Persistence
Pickle
Deployment
Gunicorn
Render
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/heart-disease-prediction.git
2. Navigate to the Project
cd heart-disease-prediction
3. Create Virtual Environment
python -m venv .venv
4. Activate Virtual Environment

Windows:

.venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
▶️ Run the Project
Train the Model
python main.py

This will execute the complete ML pipeline and generate:

model.pkl
scaled.pkl
Start Flask Application
python app.py

Open your browser and visit:

http://127.0.0.1:5000/
📌 Important Files
File	Purpose
main.py	Main ML pipeline
variable_transformation.py	Transformation and outlier handling
feature_selection.py	Feature selection
train_all_models.py	Model training and evaluation
app.py	Flask backend
heart.csv	Dataset
model.pkl	Trained ML model
scaled.pkl	Saved scaler
requirements.txt	Project dependencies
🔮 Future Enhancements
Hyperparameter tuning
Cross-validation
ROC-AUC comparison
SHAP-based model explainability
Improved frontend design
Docker support
Cloud deployment
Larger and more diverse dataset
Real-time prediction dashboard
⚠️ Disclaimer

This project is developed for educational purposes only.

The predictions generated by this application should not be considered a medical diagnosis or a substitute for professional medical advice.

👨‍💻 Author

Mohammed Awez

Computer Science Graduate | AI & Machine Learning Enthusiast

Skills

Python Machine Learning Scikit-learn Flask Pandas NumPy SQL

⭐ If you found this project useful, consider giving the repository a star!


**One correction before you push it:** in your `main.py`, change:

```python
self.sc_obj.transform(abc)
logger.info(nb_obj.predict(abc)[0])

to:

abc_scaled = self.sc_obj.transform(abc)
logger.info(nb_obj.predict(abc_scaled)[0])
