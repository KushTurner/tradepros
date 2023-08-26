# INITIATE MODEL

import torch
import torch.nn.functional as F
from data_handler import *
from tools import find_P_A_R
from model_manager import ModelManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE | {DEVICE}")
M_SEED = 2004
torch.manual_seed(M_SEED)
torch.cuda.manual_seed_all(M_SEED)
G = torch.Generator(device = DEVICE)
G.manual_seed(M_SEED)
DH = DataHandler(device = DEVICE, generator = G)
DH.retrieve_dates(
                tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                start_date = "1/01/2015",
                end_date = "31/12/2019", 
                interval = "1d",
                )
TDH = TextDataHandler(dates = DH.dates, device = DEVICE, generator = G)
TDH.retrieve_data()
model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = TDH)

"""
model_number_load = Number of the model to load, leave empty to create a new model
.initiate_model = Returns model, optimiser and hyperparamaters used to train the model
- Will use DH.retrieve_data before instantiating the model if creating a new model
- Will use DH.retrieve_data after instantiating the model if loading an existing model
"""
model_number_load = 7
manual_hyperparams = {
                    "architecture": "RNN", # Will be deleted after instantiation
                    "N_OR_S": "N",
                    "num_context_days": 10,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "num_folds": 5,
                    "multiplicative_trains": 4,
                    "uses_dated_sentiments": False,
                    }
model, optimiser, hyperparameters, stats, checkpoint_directory = model_manager.initiate_model(model_number_load = model_number_load, manual_hyperparams = manual_hyperparams)
metrics = ["loss", "accuracy", "precision", "recall", "f1"]
BATCH_SIZE = hyperparameters["batch_size"]
num_sets = (hyperparameters["num_folds"] - 1)

for company_data in DH.data_n:
    print("ContainsNaN", company_data.isnan().any().item()) # Check if the tensor contains "nan"

# Create training and test sets and data sequences for this model (must be repeated for each model as num_context_days can vary depending on the model used)
DH.create_sets(num_context_days = hyperparameters["num_context_days"], shuffle_data_sequences = False)
# Create k folds
DH.create_folds(num_folds = hyperparameters["num_folds"], N_OR_S = model.N_OR_S)

# Generate folds for this training iteration
TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2, N_OR_S = model.N_OR_S)

print(f"Hyperparameters used: {hyperparameters}")
print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# GET DATA
import requests
import pandas as pd
from os import getenv as os_get_env
api_key = os_get_env("alphavantage_apikey")
tickers = ["jpm"] #, "meta", "wmt", "ma", "005930.KS"]

all_labels = []
data_n = []
data_s = []
info_dict = {}

if os_path_exists("historical_data") == False:
    os_mkdir("historical_data")

for ticker in tickers:
    
    print(ticker) 
    path_to_data = f"historical_data/{ticker}.csv"
    if os_path_exists(path_to_data):
        print("Loading from disk")
        # Read the dataframe from the path
        # - parse_dates to parse the values in the first column (i.e. the dates) as datetime objects
        # - index_col to specify which column should be used as the index of the dataframe
        DATAFRAME = pd.read_csv(path_to_data, parse_dates = True, index_col = 0)
        
    else:
        print("Calling API")
        # Get data on single ticker
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
        r = requests.get(url)
        api_data = r.json()

        # Received an error, move to the next ticker
        if api_data.get("Error Message") != None: 
            continue
        
        print(api_data)
        print(api_data.keys())
        print(api_data["Time Series (Daily)"].keys())

        # Create dataframe
        """ Notes:
        - .T = Transpose to get ["open", "high", "low", "close", "volume"] as the columns
        - .astype(float) to convert the dtypes from "object" to "float64"
        - .sort_values(by = DATAFRAME.index, ascending = True) to sort the rows by their dates (as)
        """
        DATAFRAME = pd.DataFrame(api_data["Time Series (Daily)"]).T.astype(float)

        # Save dataframe with the dates
        DATAFRAME.to_csv(path_to_data, date_format = "%Y-%m-%d", index = True)

    # Sort dataframe based on the dates (Originally from latest dates to earliest dates)
    DATAFRAME.sort_index(ascending = True, inplace = True)

    # Rename columns
    actual_column_names = ["open", "high", "low", "close", "volume"]
    rename_columns = {prev_name: new_name for prev_name, new_name in zip(DATAFRAME.columns, actual_column_names)}
    DATAFRAME.rename(columns = rename_columns, inplace = True)
    print("Renamed", DATAFRAME)

    # Modify data
    DATAFRAME["ticker"] = [ticker for _ in range(DATAFRAME.shape[0])] # Temporarly created ticker column so taht DH.modify_data works (Ticker column is dropped in original method but this dataframe doesn't contain a ticker column yet)
    print(DATAFRAME.dtypes)
    # Note: Only take the last hyperparameters["num_context_days"] days to create a single data sequence to predict the closing price today
    DATAFRAME = DH.modify_data(DATAFRAME, interval = "1d", dated_sentiments = None, include_date_before_prediction_date = True)[-hyperparameters["num_context_days"]:]
    print("Modified", DATAFRAME)

    # Labels
    labels = DATAFRAME["Target"]
    print("Labels", labels)
    all_labels.append(DH.dataframe_to_ptt(pandas_dataframe = labels, desired_dtype = torch_int_64))
    DATAFRAME.drop("Target", axis = 1, inplace = True)
    
    # Standardise / Normalise
    S_DATAFRAME = DATAFRAME.copy()
    cols_to_alter =  ["open", "close", "high", "low", "volume"] 
    S_DATAFRAME[cols_to_alter] = DH.standardise_columns(dataframe = S_DATAFRAME, cols_to_standard = cols_to_alter)
    DATAFRAME[cols_to_alter] = DH.normalise_columns(dataframe = DATAFRAME, cols_to_norm = cols_to_alter)
    print("Standardised / Normalised", DATAFRAME)
    data_n.append(DH.dataframe_to_ptt(pandas_dataframe = DATAFRAME, desired_dtype = torch_float_32))
    data_s.append(DH.dataframe_to_ptt(pandas_dataframe = S_DATAFRAME, desired_dtype = torch_float_32))

    # Create single data sequence
    days_to_create_sequence = data_n[0][:hyperparameters["num_context_days"]]
    print(days_to_create_sequence.shape)
    data_sequence = days_to_create_sequence.view(1, hyperparameters["num_context_days"], -1) # Convert from [num_context_days, num_features] ---> [batch_size, num_context_days, num_features]
    print(data_sequence.shape)
    
    # Add to data sequences dict
    print(DATAFRAME.index)
    info_dict[ticker] = {"data_sequence": data_sequence, "date_before_prediction_date": DATAFRAME.index[-1]}


for ticker in tickers:
    print(info_dict[ticker])