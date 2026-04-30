import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
SLICE_OUTPUT_PATH = os.path.join(PROJECT_PATH, "slice_output.txt")

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


def main():
    data = pd.read_csv(DATA_PATH)

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    model_path = os.path.join(PROJECT_PATH, "model", "model.pkl")
    encoder_path = os.path.join(PROJECT_PATH, "model", "encoder.pkl")
    lb_path = os.path.join(PROJECT_PATH, "model", "lb.pkl")

    save_model(model, model_path)
    save_model(encoder, encoder_path)
    save_model(lb, lb_path)

    model = load_model(model_path)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {fbeta:.4f}"
    )

    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("Model performance on categorical data slices\n")

    for col in CAT_FEATURES:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            precision, recall, fbeta = performance_on_categorical_slice(
                test,
                col,
                slice_value,
                CAT_FEATURES,
                "salary",
                encoder,
                lb,
                model,
            )
            with open(SLICE_OUTPUT_PATH, "a", encoding="utf-8") as file:
                print(f"{col}: {slice_value}, Count: {count:,}", file=file)
                print(
                    f"Precision: {precision:.4f} | "
                    f"Recall: {recall:.4f} | F1: {fbeta:.4f}",
                    file=file,
                )


if __name__ == "__main__":
    main()
