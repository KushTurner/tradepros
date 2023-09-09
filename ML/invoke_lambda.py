"""
- Used to call the lambda function for model inference using the API gateway
"""

from requests import get as requests_get
from os import getenv as os_getenv

def get_model_response(ticker, date_to_predict):
    api_key = os_getenv("tradepros_api_key")
    url =  os_getenv("tradepros_model_inference_url")
    event = {
            "ticker": ticker,
            "date_to_predict": date_to_predict
            }
    headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
            }

    return requests_get(url, params = event, headers = headers).json()

print(get_model_response(ticker = "meta", date_to_predict = "2023-08-28"))
print(get_model_response(ticker = "goog", date_to_predict = "2023-08-28"))
print(get_model_response(ticker = "amzn", date_to_predict = "2023-08-28"))
print(get_model_response(ticker = "nflx", date_to_predict = "2023-08-28"))
print(get_model_response(ticker = "baba", date_to_predict = "1950-03-02"))
print(get_model_response(ticker = "baba", date_to_predict ="2015-03-02"))
print(get_model_response(ticker = "baba", date_to_predict ="2015-05-02"))