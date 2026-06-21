from app.constants import SERVER_HOST
import requests


def get_data():
    response = requests.get(url=f"{SERVER_HOST}/data")
    return response.json()
