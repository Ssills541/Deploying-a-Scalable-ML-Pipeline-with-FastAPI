import json

import requests

BASE_URL = "http://127.0.0.1:8000"

r = requests.get(BASE_URL, timeout=10)
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['message']}")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post(
    f"{BASE_URL}/data/",
    data=json.dumps(data),
    headers={"Content-Type": "application/json"},
    timeout=10,
)
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['result']}")
