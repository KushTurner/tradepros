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

# ---------------------------------------------------------------------------------------
# Model loading

# Initialise model manager
model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = None) # No need for TDH, because there are no calculated sentiments for forward-testing and backtesting dates
model_number_load = 0
model, _, hyperparameters, stats, checkpoint_directory = model_manager.initiate_model(model_number_load = model_number_load, inference = True) # inference = True to not initialise optimiser and set model to eval mode
metrics = ["loss", "accuracy", "precision", "recall", "f1"]
BATCH_SIZE = hyperparameters["batch_size"]
num_sets = (hyperparameters["num_folds"] - 1)
print(f"Hyperparameters used: {hyperparameters}")
print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Printing training results
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

A = 1
for metric in metrics:
    print("-----------------------------------------------------------------")
    print(f"{metric.capitalize()} during training")

    train_metric_i = torch.tensor(stats[f"train_{metric}_i"]).view(-1, A).mean(1)
    val_metric_i = torch.tensor(stats[f"val_{metric}_i"]).view(-1, A).mean(1)

    fig, ax = plt.subplots()
    ax.plot([i for i in range(int(total_epochs / A))], train_metric_i, label = "Train")
    ax.plot([i for i in range(int(total_epochs / A))], val_metric_i, label = "Validation")
    ax.legend()
    plt.show()

# Performing backtesting and forward-testing
if hyperparameters["uses_single_sentiments"] == False:
    print("-----------------------------------------------------------------")
    print(f"Performing backtesting and forward-testing")

    from os import listdir as os_listdir

    # Directory for model
    model_directory = os_listdir(f'{checkpoint_directory}')
    print(model_directory)
    
    # Loading saved test results
    if "external_testing_results.pth" in model_directory:
        # Load test results checkpoint
        checkpoint = torch.load(f"{checkpoint_directory}/external_testing_results.pth")

        for test_name in ["back_testing", "forward_testing"]:
            print("\n")
            print(f"Test name: {test_name}")
            selected_tickers = checkpoint[test_name]["selected_tickers"]
            num_sequences_per_company = checkpoint[test_name]["num_sequences_per_company"]
            num_tickers_used = checkpoint[test_name]["num_tickers_used"]
            total_num_sequences = sum(num_sequences_per_company)
            print(f"Tickers used: {selected_tickers}")
            print(f"Number of sequences per company: {num_sequences_per_company}")
            print(f"Number of tickers used: {num_tickers_used}")
            print(f"Total number of sequences: {total_num_sequences}")

            num_uptrends = checkpoint[test_name]["num_uptrends"]
            num_downtrends = checkpoint[test_name]["num_downtrends"]
            accuracy = checkpoint[test_name]["accuracy"]
            precision = checkpoint[test_name]["precision"]
            recall = checkpoint[test_name]["recall"]
            f1_score = checkpoint[test_name]["f1_score"]
            print(f"Number of uptrends: {num_uptrends} | Number of downtrends {num_downtrends}")
            print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1 Score: {f1_score}")
    
    # Generating test results
    else:
        # Add all of the tickers for testing into a single list
        selected_tickers = []
        for tickers_file in os_listdir("testing_tickers"):
            with open(f"testing_tickers/{tickers_file}", "r") as file:
                selected_tickers.extend(file.read().split())

        testing_dates = [
                        ("2010-01-01", "2015-01-01", "back_testing"), # Back testing
                        ("2020-01-01", "2023-09-01", "forward_testing") # Forward testing
                        ]
        
        test_results_checkpoint = {}

        for t_start_date, t_end_date, test_name in testing_dates:
            print(t_start_date, t_end_date, test_name)

            # Retrieve data
            DH.retrieve_data(
                        tickers = selected_tickers,
                        start_date = t_start_date, # start_date includes the start date in the dataframe, date should be in YYYY-MM-DD format
                        end_date = t_end_date, # end_date, does not include the end date in the dataframe.
                        interval = "1d",
                        dated_sentiments = None, # Not needed at inference time 
                        include_date_before_prediction_date = True, # include_date_before_prediction_date = True is used to include the date before the date to predict. (e.g. includes 2023-09-11 if the date to predict is 2023-09-12)
                        hyperparameters = hyperparameters,
                        )

            # Create data sequences
            DH.create_data_sequences(num_context_days = hyperparameters["num_context_days"], shuffle_data_sequences = False)
            input_data = DH.data
            print(f"Data sequences shape: {input_data.shape}")

            # Create batches from all the data sequences
            if model.__class__.__name__ == "RNN" or model.__class__.__name__ == "LSTM":
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
            print(f"Number of predictions: {all_predictions.shape}")

            # Create a list containing which company ticker each sequence belongs to, sorting it by the same indices used to sort the data, labels and dates
            companies_tickers = [selected_tickers[i] for i in range(len(selected_tickers)) for _ in range(DH.sequence_sizes[i])]
            companies_tickers = [companies_tickers[ind] for ind in DH.sort_indices]

            # Printing out information
            accuracy, precision, recall, f1_score = find_P_A_R(all_predictions, DH.labels.to(DEVICE))
            print(f"Tickers used: {selected_tickers}")
            print(f"Number of sequences per company: {DH.sequence_sizes}")
            print(f"Number of tickers used: {len(selected_tickers)}")
            print(f"Total number of sequences: {len(companies_tickers)}")
            print(f"Number of uptrends: {DH.labels.count_nonzero().item()} | Number of downtrends {DH.labels.shape[0] - DH.labels.count_nonzero().item()}")
            print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1 Score: {f1_score}")

            # Save the results/ inforamtion for this test as a separate checkpoint for this model (Only saving test results)
            test_results = {
                            "selected_tickers" : selected_tickers,
                            "num_sequences_per_company": DH.sequence_sizes,
                            "num_tickers_used": len(selected_tickers),
                            "num_uptrends": DH.labels.count_nonzero().item(),
                            "num_downtrends": DH.labels.shape[0] - DH.labels.count_nonzero().item(),
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1_score
                            }
            test_results_checkpoint[test_name] = test_results
            torch.save(obj = test_results_checkpoint, f = f"{checkpoint_directory}/external_testing_results.pth")

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
            # prediction_info_dicts = [
            #                         {
            #                         "ticker": ticker, 
            #                         "prediction": prediction, 
            #                         "Target": label.item(), 
            #                         "ModelAnswer": torch.argmax(prediction, dim = 0).item(),
            #                         "Correct": torch.argmax(prediction, dim = 0).item() == label.item(), 
            #                         "timestamp": timestamp
            #                         }
            #                         for prediction, label, ticker, timestamp in zip(all_predictions, DH.labels, companies_tickers, DH.dates)
            #                         ]
            # prediction_info_dicts.sort(key = lambda x: x["timestamp"]) # Sort by date