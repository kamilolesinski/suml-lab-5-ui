from .constants import SERVER_HOST
import requests


def train(x, y):
    requests.post(
        json={"x": x, "y": y},
        url=f"{SERVER_HOST}/model/train",
    )
