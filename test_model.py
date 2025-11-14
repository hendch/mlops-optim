from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

X_train, X_test, y_train, y_test = prepare_data("dataAssurance(in).csv")

gbr = train_model(X_train, y_train)

save_model(gbr, "gradient_boost_model.pkl")

loaded_model = load_model("gradient_boost_model.pkl")

evaluate_model(gbr, X_test, y_test)
