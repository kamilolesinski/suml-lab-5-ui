from app.constants import SERVER_HOST
import requests


def train(x, y):
    r = requests.post(
        json={"x": x, "y": y},
        url=f"{SERVER_HOST}/model/train",
    )
    print(r)
