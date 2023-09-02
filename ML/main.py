import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from data_handler import *
from tools import find_P_A_R
from model_manager import ModelManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE | {DEVICE}")

M_SEED = 2004
# Seed for the model
torch.manual_seed(M_SEED)
torch.cuda.manual_seed_all(M_SEED)

# Seed for data handler
G = torch.Generator(device = DEVICE)
G.manual_seed(M_SEED)

# Initialising data handler
DH = DataHandler(device = DEVICE, generator = G)

# Retrieve dates for the text data handler (To generate sentiments for tweets on dates in the historical dataset)
DH.retrieve_dates(
                tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                start_date = "01/01/2015", # Date in MM/DD/YYYY format
                end_date = "12/31/2019", 
                interval = "1d",
                )

# Initialising text data handler
TDH = TextDataHandler(dates = DH.dates, device = DEVICE, generator = G)
TDH.retrieve_data()

# ---------------------------------------------------------------------------------------
# Model loading

# Initialise model manager
model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = TDH)

"""
model_number_load = Number of the model to load, leave empty to create a new model
.initiate_model = Returns model, optimiser and hyperparamaters used to train the model

If using for training / testing on the testing set:
    - Will use DH.retrieve_data before instantiating the model if creating a new model
    - Will use DH.retrieve_data after instantiating the model if loading an existing model
"""
model_number_load = None
manual_hyperparams = {
                    "architecture": "RNN", # Will be deleted after instantiation
                    "N_OR_S": "N",
                    "num_context_days": 10,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "num_folds": 5,
                    "multiplicative_trains": 1,
                    "uses_dated_sentiments": False, #True, #False,
                    "uses_single_sentiments": False, #True,
                    "features_to_remove": ["adjclose"],
                    "cols_to_alter": ["open", "close", "high", "adjclose", "low", "volume"],
                    "rolling_periods": [2, 5, 10, 15, 20],
                    "rolling_features": set(
                                            [
                                            "avg_open", 
                                            "open_ratio", 
                                            "avg_close", 
                                            "close_ratio", 
                                            "avg_volume", 
                                            "volume_ratio", 
                                            "trend_sum", 
                                            "trend_mean", 
                                            "close_diff", 
                                            "close_diff_percentage", 
                                            "rsi", 
                                            "macd",
                                            "bollinger_bands",
                                            "average_true_range",
                                            "stochastic_oscillator",
                                            "on_balance_volume"
                                            ]
                                            ),
                    "transform_after": True,
                    "train_split_decimal": 0.8,
                    "train_data_params": None
                    }
# manual_hyperparams = {
#                     "architecture": "MLP", # Will be deleted after instantiation
#                     "N_OR_S": "S",
#                     "num_context_days": 1,
#                     "batch_size": 32,
#                     "learning_rate": 1e-4,
#                     "num_folds": 5,
#                     "multiplicative_trains": 1,
#                     "uses_dated_sentiments": True,
#                     "uses_single_sentiments": True,
#                     "features_to_remove": ["adjclose"],
#                     "cols_to_alter": ["open", "close", "high", "adjclose", "low", "volume"],
#                     "rolling_periods": [2, 5, 10, 15, 20],
#                     "rolling_features": ["avg_open", "open_ratio", "avg_close", "close_ratio", "avg_volume", "volume_ratio", "trend_sum", "trend_mean"],
#                     "transform_after": True,
#                     "train_split_decimal": 0.8,
#                     "train_data_params": None
#                     }
# manual_hyperparams = None
model, optimiser, hyperparameters, stats, checkpoint_directory = model_manager.initiate_model(model_number_load = model_number_load, manual_hyperparams = manual_hyperparams)
metrics = ["loss", "accuracy", "precision", "recall", "f1"]
BATCH_SIZE = hyperparameters["batch_size"]
num_sets = (hyperparameters["num_folds"] - 1) # Number of sets i.e. the number of (TRAIN_FOLDS, VAL_FOLDS) generated, e.g. if num_folds = 5, there will be 4 sets

for company_data in DH.data:
    print("ContainsNaN", company_data.isnan().any().item()) # Check if the tensor contains "nan"

# Create training and test sets and data sequences for this model (must be repeated for each model as num_context_days can vary depending on the model used)
DH.create_sets(num_context_days = hyperparameters["num_context_days"], shuffle_data_sequences = False, train_split_decimal = hyperparameters["train_split_decimal"])
# Create k folds
DH.create_folds(num_folds = hyperparameters["num_folds"])

# Generate folds for this training iteration
TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2, use_single_sentiments = hyperparameters["uses_single_sentiments"])

print(f"Hyperparameters used: {hyperparameters}")
print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")
# ---------------------------------------------------------------------------------------

# Testing generate_batch
X1, Y1, S1 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = hyperparameters["num_context_days"], start_idx = 0, uses_single_sentiments = hyperparameters["uses_single_sentiments"])
print(X1.shape, Y1.shape, S1.shape if S1 != None else None)

X2, Y2, S2 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = hyperparameters["num_context_days"], start_idx = 0, uses_single_sentiments = hyperparameters["uses_single_sentiments"])
print(X2.shape, Y2.shape, S2.shape if S2 != None else None)

X3, Y3, S3 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = hyperparameters["num_context_days"], start_idx = 0, uses_single_sentiments = hyperparameters["uses_single_sentiments"])
print(X3.shape, Y3.shape, S3.shape if S3 != None else None)


# Training:
# Note: Only entered if creating a new model or continuing training on a model that was interrupted
if hyperparameters["fold_number"] != hyperparameters["num_folds"] - 1:
    print("---------------------------------------------------------------------------------------")
    print(f"Starting training from: Fold {hyperparameters['fold_number'] + 1}/{hyperparameters['num_folds']}") # fold_number is the index

    for k in range(hyperparameters["fold_number"], num_sets):

        # Generate folds for this training iteration    
        # Notes:
        # - Window size starts at 2: Window sizes = (2 + 0), (2 + 1) (2 + 2) (2 + 3)
        # - Number of total sets = (num_folds - 1)
        TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2 + k, use_single_sentiments = hyperparameters["uses_single_sentiments"])
        

        # Rolling window variables: (Starting indexes, number of batches in each of the training / validation folds, interval for evaluating on the validation fold)
        train_fold_sidx = 0
        val_fold_sidx = 0
        num_validations = VAL_FOLDS[1].shape[0] - BATCH_SIZE + 1
        num_trains = TRAIN_FOLDS[1].shape[0] - BATCH_SIZE + 1
        validation_interval = int(num_trains / num_validations)
        multiplicative_trains = hyperparameters["multiplicative_trains"] # Number of times each batch will be used to train the model

        print(f"num_trains * multiplicative_trains: {num_trains * multiplicative_trains} | num_trains: {num_trains} | num_validations: {num_validations} | validation_interval: {validation_interval}")
        print(f"Training examples: {TRAIN_FOLDS[1].shape} | Validation examples: {VAL_FOLDS[1].shape}")

        for i in range(num_trains):

            # Generate inputs and labels
            Xtr, Ytr, Str = DH.generate_batch(
                                            batch_size = BATCH_SIZE,
                                            dataset = TRAIN_FOLDS, 
                                            num_context_days = hyperparameters["num_context_days"], 
                                            start_idx = train_fold_sidx,
                                            uses_single_sentiments = hyperparameters["uses_single_sentiments"]
                                            )
            train_fold_sidx += 1

            for j in range(multiplicative_trains): # Train on a single batch j times

                # Forward pass
                logits = model(Xtr, single_sentiment_values = Str)

                # Find training loss
                # Note: Did not use F.softmax and F.nll_loss because of floating point accuracy
                loss = F.cross_entropy(logits, Ytr)

                # Backward pass
                optimiser.zero_grad()
                loss.backward()

                # Update model parameters
                optimiser.step()
            
            # Evaluate the model
            """
            Notes:
            - Trains on all of the training examples in the current training fold, evaluating on the validation set on a set interval, based on the ratio between the sizes of the training folds and validation fold
            - If all of the validation examples have been used up and there are still training examples left, evaluation is not performed and the model will be trained with the remainder of the training examples until the next fold
            """
            if (i == 0 or (num_trains % i) == 1  or (i + 1) % validation_interval == 0) and val_fold_sidx < num_validations:
                with torch.no_grad():
                    # Note: Must set to evaluation mode as BatchNorm layers and Dropout layers behave differently during training and evaluation
                    # BatchNorm layers - stops updating the moving averages in BatchNorm layers and uses running statistics instead of per-batch statistics
                    # Dropout layers - de-activated during evaluation
                    model.eval()

                    # Find the accuracy, precision, recall and f1 score on the training batch
                    preds = F.softmax(logits, dim = 1) # Softmax to find probability distribution
                    train_accuracy, train_precision, train_recall, train_f1 = find_P_A_R(predictions = preds, targets = Ytr)
                    
                    # Find the loss, accuracy, precision, recall and f1 score on a validation batch
                    Xva, Yva, Sva = DH.generate_batch(
                                                        batch_size = BATCH_SIZE, 
                                                        dataset = VAL_FOLDS, 
                                                        num_context_days = hyperparameters["num_context_days"], 
                                                        start_idx = val_fold_sidx,
                                                        uses_single_sentiments = hyperparameters["uses_single_sentiments"]
                                                        )
                    val_fold_sidx += 1

                    v_logits = model(Xva, single_sentiment_values = Sva)
                    v_loss = F.cross_entropy(v_logits, Yva)
                    v_preds = F.softmax(v_logits, dim = 1)
                    val_accuracy, val_precision, val_recall, val_f1 = find_P_A_R(predictions = v_preds, targets = Yva)

                    model.train()
            
            # ----------------------------------------------
            # Tracking stats

            stats["train_loss_i"].append(loss.item())
            stats["train_accuracy_i"].append(train_accuracy)
            stats["train_precision_i"].append(train_precision)
            stats["train_recall_i"].append(train_recall)
            stats["train_f1_i"].append(train_f1)

            stats["val_loss_i"].append(v_loss.item())
            stats["val_accuracy_i"].append(val_accuracy)
            stats["val_precision_i"].append(val_precision)
            stats["val_recall_i"].append(val_recall)
            stats["val_f1_i"].append(val_f1)

            if i == 0 or (num_trains % i) == 1 or (i + 1) % validation_interval == 0: # First, last, validation interval
                print(f"K: {k + 1}/{num_sets} | Epoch: T: {(i + 1) * multiplicative_trains}/{num_trains * multiplicative_trains} V: {val_fold_sidx}/{num_validations} | TLoss: {loss.item()} | VLoss: {v_loss.item()} | TAccuracy: {train_accuracy} | VAccuracy: {val_accuracy} | TPrecision: {train_precision} | VPrecision: {val_precision} | TRecall: {train_recall} | VRecall: {val_recall} | TF1 {train_f1} | VF1: {val_f1}")

        # Record metrics for this fold:
        # -num_trains/validations: = Last num_trains items (i.e. all the statistics from this fold)
        # /num_trains/validations = Average metric in this fold

        for metric in metrics:
            fold_t_key = f"fold_t_{metric}"
            fold_v_key = f"fold_v_{metric}"
            
            stats[fold_t_key].append((sum(stats[f"train_{metric}_i"][-num_trains:]) / num_trains))
            stats[fold_v_key].append((sum(stats[f"val_{metric}_i"][-num_validations:]) / num_validations))
        
        # ----------------------------------------------
        # Saving checkpoint

        hyperparameters["fold_number"] = k + 1 # Saves the index of the next fold to continue training from
        checkpoint = {
                    "model":{
                            "architecture": model.__class__.__name__,
                            "model_state_dict": model.state_dict(),
                            "optimiser_state_dict": optimiser.state_dict(),
                            },
                    "hyperparameters": hyperparameters,
                    "stats": stats
                    }
        
        torch.save(obj = checkpoint, f = f"{checkpoint_directory}/fold_{k}.pth")

# Set model to evaluation mode (For dropout + batch norm layers)
model.eval()

print("-----------------------------------------------------------------")
print("Metrics per fold")

for metric in metrics:
    print(f'T{metric.capitalize()}: {stats[f"fold_t_{metric}"]}')
    print(f'V{metric.capitalize()}: {stats[f"fold_v_{metric}"]}')

print("-----------------------------------------------------------------")
print("Metrics across folds")
for metric in metrics:
    print(f'T{metric.capitalize()}: {sum(stats[f"fold_t_{metric}"]) / num_sets} | V{metric.capitalize()}: {sum(stats[f"fold_v_{metric}"]) / num_sets}')

# Plotting train / validation loss
total_epochs = len(stats["train_loss_i"])
print(total_epochs)

A = 14 # Replace with a factor of the total number of epochs
A = 62 
# A = 42

# for metric in metrics:
#     print("-----------------------------------------------------------------")
#     print(f"{metric.capitalize()} during training")

#     train_metric_i = torch.tensor(stats[f"train_{metric}_i"]).view(-1, A).mean(1)
#     val_metric_i = torch.tensor(stats[f"val_{metric}_i"]).view(-1, A).mean(1)

#     fig, ax = plt.subplots()
#     ax.plot([i for i in range(int(total_epochs / A))], train_metric_i, label = "Train")
#     ax.plot([i for i in range(int(total_epochs / A))], val_metric_i, label = "Validation")
#     ax.legend()
#     plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing on existing data:

if hyperparameters["uses_single_sentiments"] == False:
    print(DH.data.shape)
    # selected_tickers = ["msft", "aapl", "nvda", "amd", "baba", "uber"]
    selected_tickers = ["jpm", "meta", "wmt", "ma", "005930.KS", "nesn.sw"]

    DH.retrieve_data(
                tickers = selected_tickers,
                start_date = "01/01/2023", # MM/DD/YYYY
                end_date = "08/28/2023", # This will be the final date used to predict e.g. 25/08/2023, include_date_before_prediction_date = True is used to include 24/08/2023
                interval = "1d",
                dated_sentiments = None, # Not needed at inference time 
                include_date_before_prediction_date = True,
                hyperparameters = hyperparameters,
                )

    for company in DH.data:
        print("num_days", len(company))

    print(len(DH.dates))
    for dates in DH.dates:
        print("num_dates_for_each_sequence_for_each_company", len(dates))

    # Create data sequences
    DH.create_data_sequences(num_context_days = hyperparameters["num_context_days"], shuffle_data_sequences = False)
    print(len(DH.data))
    print(DH.labels.shape)
    print(len(DH.dates))
    input_data = DH.data
    print(len(input_data))

    # Create batches from all the data sequences
    if model.__class__.__name__ == "RNN":
        # Convert inputs from [batch_size, num_context_days, num_features] to [batch_size, num_features, num_context_days]
        batches = [input_data[i:i + hyperparameters["batch_size"]].transpose(dim0 = 0, dim1 = 1) for i in range(0, len(input_data), hyperparameters["batch_size"])]

        # Padding final batch for batch prompting
        if batches[-1].shape[1] != hyperparameters["batch_size"]:
            right_padding = torch.zeros(hyperparameters["num_context_days"], hyperparameters["batch_size"] - batches[-1].shape[1], batches[-1].shape[2])
            batches[-1] = torch.concat(tensors = [batches[-1], right_padding], dim = 1) # Concatenate on the batch dimension

    elif model.__class__.__name__ == "MLP":
        # Single day sequences
        batches = [input_data[i:i + hyperparameters["batch_size"]] for i in range(0, len(input_data), hyperparameters["batch_size"])]

        # Padding final batch for batch prompting
        if batches[-1].shape[0] != hyperparameters["batch_size"]:
            right_padding = torch.zeros(hyperparameters["batch_size"] - batches[-1].shape[0], batches[-1].shape[1])
            batches[-1] = torch.concat(tensors = [batches[-1], right_padding], dim = 0) # Concatenate on the batch dimension


    # Get all the predictions for each batch
    # Note: [:len(input_data)] to get rid of any padding examples
    all_predictions = torch.concat([F.softmax(model(inputs = batch.to(device = DEVICE), single_sentiment_values = None), dim = 1) for batch in batches])[:len(input_data)]
    print(all_predictions.shape)

    # Create a list containing which company ticker each sequence belongs to, sorting it by the same indices used to sort the data, labels and dates
    print(selected_tickers)
    print(DH.sequence_sizes)
    companies_tickers = [selected_tickers[i] for i in range(len(selected_tickers)) for _ in range(DH.sequence_sizes[i])]
    companies_tickers = [companies_tickers[ind] for ind in DH.sort_indices]
    print(len(companies_tickers))

    """ 
    Labels = Sorted by dates
    Data = Sorted by dates
    Company tickers = Not sorted
    """

    correct_count = 0
    for prediction, label in zip(all_predictions, DH.labels):
        pred_i = torch.argmax(prediction, dim = 0)
        # print(prediction)
        # print(pred_i)
        # print(label)
        # print("----------------")
        correct_count += 1 if label.item() == pred_i.item() else 0

    print(f"Number of uptrends: {DH.labels.count_nonzero().item()} | Number of downtrends {DH.labels.shape[0] - DH.labels.count_nonzero().item()}")
    accuracy, precision, recall, f1_score = find_P_A_R(all_predictions, DH.labels.to(DEVICE))
    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1 Score: {f1_score}")
    print(f"Correct: {correct_count}/{all_predictions.shape[0]} | PercentageCorrect: {(correct_count / all_predictions.shape[0]) * 100}")


    # Create a list of dictionaries containing information about a sequence
    """ Notes:
    - Each data sequence is mapped with:
        - Corresponding ticker
        - The model prediction in the form of a probability distribution
        - Target (the actual trend) 
        - The model prediction where 0 = predicted downward trend, 1 = predicted upward trend
        - Whether the model's prediction was correct
        - Corresponding date

    - If it appears that there are repeated predictions for completely different sequences, its just how the tensor is displayed (rounded)

    # To double-check the number of occurrences of each prediction
    from collections import Counter
    import numpy as np
    all_predictions = [sequence_dict["prediction"].to("cpu") for sequence_dict in prediction_info_dicts]

    # Convert the tensors to strings and use Counter to count occurrences
    tensor_tuple_list = [tuple(tensor) for tensor in all_predictions]
    tensor_counts = Counter(tensor_tuple_list)

    # Print the counts of unique tensors
    for tensor_tuple, count in tensor_counts.items():
        tensor = np.array(tensor_tuple)
        if count > 1:
            print(f"{tensor}: {count}")

    """
    prediction_info_dicts = [{"ticker": ticker, "prediction": prediction, "Target": label.item(), "ModelAnswer": torch.argmax(prediction, dim = 0).item(), "Correct": torch.argmax(prediction, dim = 0).item() == label.item(), "timestamp": timestamp} for prediction, label, ticker, timestamp in zip(all_predictions, DH.labels, companies_tickers, DH.dates)]
    for d in prediction_info_dicts[:5]:
        print(d)
    prediction_info_dicts.sort(key = lambda x: x["timestamp"]) # Sort by date

    for d in prediction_info_dicts[-10:]:
        print("After", d)