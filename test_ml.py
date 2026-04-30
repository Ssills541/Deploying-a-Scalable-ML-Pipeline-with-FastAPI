import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 37],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
                "Private",
            ],
            "fnlgt": [77516, 83311, 215646, 234721, 338409, 284582],
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Bachelors",
                "Masters",
            ],
            "education-num": [13, 13, 9, 7, 13, 14],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
                "Exec-managerial",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
                "Wife",
            ],
            "race": ["White", "White", "White", "Black", "Black", "White"],
            "sex": ["Male", "Male", "Male", "Male", "Female", "Female"],
            "capital-gain": [2174, 0, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40, 40],
            "native-country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "Cuba",
                "United-States",
            ],
            "salary": ["<=50K", "<=50K", "<=50K", "<=50K", ">50K", ">50K"],
        }
    )


def test_process_data_returns_expected_shapes():
    df = sample_dataframe()
    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]
    assert len(encoder.categories_) == len(CAT_FEATURES)
    assert list(lb.classes_) == ["<=50K", ">50K"]


def test_train_model_inference_and_metrics_return_valid_values():
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 1],
            [2, 2],
        ]
    )
    y_train = np.array([0, 0, 1, 1, 1, 1])
    model = train_model(X_train, y_train)

    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_save_and_load_model_preserves_predictions(tmp_path):
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 1, 1])
    model = train_model(X_train, y_train)
    model_path = tmp_path / "model.pkl"

    save_model(model, str(model_path))
    loaded_model = load_model(str(model_path))

    assert np.array_equal(inference(model, X_train), inference(loaded_model, X_train))


def test_performance_on_categorical_slice_returns_metrics():
    df = sample_dataframe()
    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y)

    precision, recall, fbeta = performance_on_categorical_slice(
        df,
        "sex",
        "Female",
        CAT_FEATURES,
        "salary",
        encoder,
        lb,
        model,
    )

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
