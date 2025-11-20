# 🧠 ML Project – Insurance Charges Prediction (MLOps Workshop)

This project is part of an MLOps workshop.  
The goal is to build a clean and maintainable machine learning pipeline using:

- Python
- Pandas
- Scikit-Learn
- Joblib
- A structured project layout
- Basic MLOps practices such as data validation and metrics tracking

The model predicts **insurance charges** based on customer features.

---

## 📁 Project Structure

ml_project/
├── data/
│   └── raw/
│       └── dataAssurance(in).csv
├── models/
│   └── gradient_boost_model.joblib
├── src/
│   ├── __init__.py
│   ├── model_pipeline.py     
│   ├── train.py              
│   └── test_model.py       
├── results/
│   └── metrics.txt           
└── requirements.txt          

## ⚙️ Installation

Create a virtual environment and install dependencies:


pip install -r requirements.txt

🏗️ What the Pipeline Does
✔️ 1. Loads the raw dataset

Located in data/raw/data.csv.

✔️ 2. Runs non-intrusive data validation

Validation checks for:

missing columns

missing values

unexpected empty fields

Nothing is dropped or modified. Warnings are logged only.

✔️ 3. Preprocesses the data

Including:

label encoding

scaling

imputation

train/test split

✔️ 4. Trains a Gradient Boosting Regressor

The model is saved to:
models/gradient_boost_model.joblib

✔️ 5. Evaluates the model

Metrics:

MAE

MSE

R²

✔️ 6. Stores metrics in JSON

Output saved to:
results/metrics.json

▶️ Training the Model

Run from the project root:
python -m src.train

▶️ Testing / Evaluating the Model

python -m src.test_model


Author : Hind Ch
