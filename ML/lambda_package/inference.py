def get_model_prediction(event, context):
    ticker = event["queryStringParameters"]["ticker"]
    date_to_predict = event["queryStringParameters"]["date_to_predict"]
    
    # Importing dependencies + model set-up
    import torch
    from data_handler import DataHandler
    from model_manager import ModelManager
    from datetime import datetime, timedelta
    from yahoo_fin.stock_info import get_data, get_dividends
    import json

    DEVICE = "cpu" # Faster inference times on CPU
    M_SEED = 2004
    torch.manual_seed(M_SEED)
    torch.cuda.manual_seed_all(M_SEED)
    G = torch.Generator(device = DEVICE)
    G.manual_seed(M_SEED)

    DH = DataHandler(device = DEVICE, generator = G)
    model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = None)
    model_number_load = 43
    model, _, hyperparameters, _, _ = model_manager.initiate_model(
                                                                    model_number_load = model_number_load, 
                                                                    manual_hyperparams = None, 
                                                                    inference = True
                                                                    )

    # print(f"Hyperparameters used: {hyperparameters}")
    # print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Creating data sequence

    if hyperparameters["uses_single_sentiments"]:
        # Retrieve news articles information through Polygon API
        from requests import get as requests_get
        from os import getenv as os_getenv
        api_key = os_getenv("polygon_apikey")
        limit = 1000
        url = f"https://api.polygon.io/v2/reference/news?limit={limit}&sort=published_utc&apiKey={api_key}"
        r = requests_get(url)
        api_data = r.json()

        # Find the average sentiment for the ticker using VADER model
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        fields = ["title", "description", "keywords"]
        sentiment = sum([vader.polarity_scores(info_dict[field])["compound"] if info_dict.get(field) else 0 for info_dict in api_data["results"] for field in fields]) / limit

        # Add padding before passing to the model
        padded_sentiment = torch.zeros((hyperparameters["batch_size"], 1)) # Converted to torch.float32 automatically (Sentiment is in float64 accuracy)
        padded_sentiment[0, 0] = sentiment
        # print(padded_sentiment, padded_sentiment.shape)
        # print(sentiment)
        # print(padded_sentiment[0, 0].item())
    
    # Find starting date
    num_days_prev = 365 * 10 #(hyperparameters["num_context_days"] + hyperparameters["rolling_periods"][-1]) * 4
    end_date_as_dt = datetime.strptime(date_to_predict, "%Y-%m-%d")
    start_date_as_dt = end_date_as_dt - timedelta(days = num_days_prev)

    # Convert to MM/DD/YYYY
    start_date = start_date_as_dt.strftime("%Y-%m-%d")
    end_date = end_date_as_dt.strftime("%Y-%m-%d")

    # print(hyperparameters["rolling_periods"][-1])
    # print(hyperparameters["num_context_days"])
    # print(num_days_prev)
    # print(start_date, end_date)

    # Retrieve historical data
    DATA = get_data(
                    ticker = ticker, 
                    start_date = start_date, 
                    end_date = end_date, 
                    index_as_date = True, 
                    interval = "1d"
                    )

    # Adding dividends
    if "dividends" not in hyperparameters["features_to_remove"]:
        try:
            DIVIDENDS = get_dividends(ticker = ticker, start_date = start_date, end_date = end_date, index_as_date = True)
            # Re-index using the dates in the the historical data
            DIVIDENDS = DIVIDENDS.reindex(DATA.index) # Fill value is automatically NaN

            # Use linear interpolation to calculate + fill in the missing rows
            """
            Notes: 
            - Days before the first dividend will always have N/A as its values, so will be removed after DataHandler.modify_data()
            - Use method = "time" to consider the time intervals between data points when estimating missing values (Good for time-series data)
            """
            DIVIDENDS["dividend"] = DIVIDENDS["dividend"].interpolate(method = "time")

            # Add dividends column to historical dataset
            DATA["dividends"] = DIVIDENDS["dividend"]

            # Removes rows which contain "NaN" inside of any columns
            DATA.dropna(inplace = True)

        # No dividends found, set as 0s
        except:
            DATA["dividends"] = [0 for _ in range(DATA.shape[0])]
    
    # Modify data
    DATA = DH.modify_data(
                        D = DATA, 
                        dated_sentiments = None, 
                        include_date_before_prediction_date = True, 
                        hyperparameters = hyperparameters
                        )
    # Remove labels
    DATA.drop("Target", axis = 1, inplace = True)

    # Transform in the context of itself (the company)
    if hyperparameters["transform_after"] == False:
        # Standardise
        if hyperparameters["N_OR_S"] == "S":
            DATA = (DATA[hyperparameters["cols_to_alter"]] - DATA[hyperparameters["cols_to_alter"]].mean()) / DATA[hyperparameters["cols_to_alter"]].std()
        # Normalise
        else:
            DATA[hyperparameters["cols_to_alter"]] = (DATA[hyperparameters["cols_to_alter"]] - DATA[hyperparameters["cols_to_alter"]].min()) / (DATA[hyperparameters["cols_to_alter"]].max() - DATA[hyperparameters["cols_to_alter"]].min())
        DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch.float32)

    # Transform in the context of all companies (i.e. the companies used in training)
    else:
        col_indexes = [DATA.columns.get_loc(column_name) for column_name in hyperparameters["cols_to_alter"]]
        DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch.float32)

        # Standardise
        if hyperparameters["N_OR_S"] == "S":
            DH.data[:,col_indexes] -= hyperparameters["train_data_params"]["S"]["mean"]
            DH.data[:,col_indexes] /= hyperparameters["train_data_params"]["S"]["std"]
        # Normalise
        else:
            DH.data[:, col_indexes] -= hyperparameters["train_data_params"]["N"]["minimums"]
            DH.data[:, col_indexes] /= hyperparameters["train_data_params"]["N"]["differences"]
    
    # Create data sequence (Taking only "num_context_days" data points)
    DATA_SEQUENCE = DH.data[-hyperparameters["num_context_days"]:]

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create batch
    # - batch.shape = [num_context_days, batch_size, num_features]


    if model.__class__.__name__ == "RNN" or model.__class__.__name__ == "LSTM":
        # Sequence shape should be [num_context_days, 1, num_features]
        right_padding = torch.zeros(hyperparameters["num_context_days"], hyperparameters["batch_size"] - 1, hyperparameters["n_features"])
        batch = torch.concat([DATA_SEQUENCE.view(hyperparameters["num_context_days"], 1, hyperparameters["n_features"]), right_padding], dim = 1)

    elif model.__class__.__name__ == "MLP":
        # Sequence shape should be [1, num_features]
        right_padding = torch.zeros(hyperparameters["batch_size"] - 1, hyperparameters["n_features"])
        batch = torch.concat([DATA_SEQUENCE.view(1, hyperparameters["n_features"]), right_padding], dim = 0)

    # Generate prediction
    prediction = torch.nn.functional.softmax(model(inputs = batch.to(device = DEVICE), single_sentiment_values = padded_sentiment if hyperparameters["uses_single_sentiments"] else None), dim = 1)[0]
    answer = torch.argmax(prediction, dim = 0).item()

    # Return response
    response = {
                "confidence": prediction[answer].item(),
                "model_answer": "down" if answer == 0 else "up",
                }
    return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response)
            }

print(get_model_prediction(event = {"queryStringParameters": {"ticker": "meta", "date_to_predict":"2022-03-02"}}, context = None))