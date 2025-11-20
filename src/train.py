from src.model_pipeline import prepare_data, train_model, save_model

X_train, X_test, y_train, y_test = prepare_data("data/raw/dataAssurance(in).csv")
gbr = train_model(X_train, y_train)
save_model(gbr, "gradient_boost_model.joblib")

