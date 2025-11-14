import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def prepare_data() :
	df = pd.read_csv('dataAssurance(in).csv')

	# Encode categoricals
	for col in ['sex', 'smoker', 'region']:
    	df[col] = LabelEncoder().fit_transform(df[col].astype(str))

	# Scale numeric columns
	scaler = StandardScaler()
	df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

	# Apply KNN imputation
	imputer = KNNImputer(n_neighbors=5)
	df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)
	numeric_cols = ['age', 'bmi', 'children', 'charges']

	# Apply only on numeric columns
	imputer = KNNImputer(n_neighbors=5)
	df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
	# Separate target and features
	X = df.drop(columns=["charges"])
	y = df["charges"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

