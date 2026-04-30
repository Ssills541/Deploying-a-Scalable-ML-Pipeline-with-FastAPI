import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(PROJECT_PATH, "model", "encoder.pkl")
MODEL_PATH = os.path.join(PROJECT_PATH, "model", "model.pkl")

encoder = load_model(ENCODER_PATH)
model = load_model(MODEL_PATH)

app = FastAPI(
    title="Census Income Prediction API",
    description="Predict whether income is >50K or <=50K.",
    version="1.0.0",
)


@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Hello from the API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """Run model inference on one Census-style input row."""
    if hasattr(data, "model_dump"):
        data_dict = data.model_dump()
    else:
        data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=None,
    )
    prediction = inference(model, data_processed)
    return {"result": apply_label(prediction)}
