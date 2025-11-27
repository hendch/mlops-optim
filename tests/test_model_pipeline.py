from src.model_pipeline import prepare_data, train_model


def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test = prepare_data("data/raw/data.csv")

    # Basic sanity checks
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_train_model_runs():
    X_train, X_test, y_train, y_test = prepare_data("data/raw/data.csv")
    model = train_model(X_train, y_train)

    # Model should have a predict method and return correct shape
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
