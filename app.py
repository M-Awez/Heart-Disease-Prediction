from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ------------------------------------------------------------------
# Load your trained model (and scaler, if you used one during training)
# Place 'model.pkl' (and 'scaler.pkl' if applicable) in this same folder.
# ------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaled.pkl"

model = None
scaler = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"[WARNING] '{MODEL_PATH}' not found. Add your trained model file to enable predictions.")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    # Scaler is optional — only needed if you scaled features during training
    scaler = None

# The exact feature order your model was trained on
FEATURE_ORDER = [
    "age_tr",
    "sex_tr",
    "cp_tr",
    "thalach_tr",
    "oldpeak_tr",
    "slope_tr",
    "thal_tr",
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, error=None, form_data={})
@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    if model is None:
        return render_template(
            "index.html",
            prediction=None,
            error="Model not loaded. Please add 'model.pkl' to the project folder.",
            form_data=form_data,
        )

    try:
        features = [float(form_data[col]) for col in FEATURE_ORDER]
        features_array = np.array(features).reshape(1, -1)

        if scaler is not None:
            features_array = scaler.transform(features_array)

        prediction = model.predict(features_array)[0]

        # If your model supports probability output, show confidence too
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_array)[0]
            confidence = round(max(proba) * 100, 2)

        result = "High risk of Heart Disease" if int(prediction) == 1 else "Low risk of Heart Disease"

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            error=None,
            form_data=form_data,
        )

    except (ValueError, KeyError) as e:
        return render_template(
            "index.html",
            prediction=None,
            error=f"Invalid input: {e}",
            form_data=form_data,
        )


if __name__ == "__main__":
    app.run(debug=True)