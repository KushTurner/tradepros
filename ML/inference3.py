from time import time as get_time
from tools import get_model_prediction_2

times = []

for i in range(100):
    start_time = get_time()
    prediction = get_model_prediction_2(model_number_load = 20, tickers = ["meta"])
    end_time = get_time()
    times.append(end_time - start_time)

print(times)
print("Average time:", sum(times) / len(times))