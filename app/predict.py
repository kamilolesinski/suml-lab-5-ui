from .constants import SERVER_HOST
import requests


def predict(x):
    response = requests.get(url=f"{SERVER_HOST}/model/predict/{x}")
    return response.json().get("y")
