"""
Testing environment specifically for testing features that will be used in models
"""

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
features_to_test = [
                    "default",
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
                    "on_balance_volume",
                    "ichimoku_cloud"
                    ]
rolling_periods = [2, 5, 10, 15, 20]

# Initalise model manager
model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = TDH)

models_info = {}
training = False
model_architectures = ["LSTM", "RNN"]

for selected_model_architecture in model_architectures:
    for test_feature in features_to_test: # For each testing feature
        for r_period in rolling_periods: # For each rolling period
            
            feature_test_model_name = f"{test_feature}_rolling_{r_period}" if test_feature != "default" else "default"
            model_number_load = feature_test_model_name if os_path_exists(f"model_checkpoints/feature_test_models/{selected_model_architecture}/{feature_test_model_name}") else None
            manual_hyperparams = {
                                "architecture": selected_model_architecture,
                                "N_OR_S": "N",
                                "num_context_days": 10,
                                "batch_size": 32,
                                "learning_rate": 1e-3,
                                "num_folds": 5,
                                "multiplicative_trains": 1,
                                "uses_dated_sentiments": False,
                                "uses_single_sentiments": False,
                                "features_to_remove": ["adjclose"],
                                "cols_to_alter": ["open", "close", "high", "adjclose", "low", "volume"],
                                "rolling_periods": [r_period],
                                "rolling_features": [test_feature],
                                "transform_after": True,
                                "train_split_decimal": 0.8,
                                "train_data_params": None
                                }
            print(model_number_load, f"model_checkpoints/feature_test_models/{selected_model_architecture}/{feature_test_model_name}")
            model, optimiser, hyperparameters, stats, checkpoint_directory = model_manager.initiate_model(
                                                                                                        model_number_load = model_number_load, 
                                                                                                        manual_hyperparams = manual_hyperparams, 
                                                                                                        feature_test_model_name = feature_test_model_name,
                                                                                                        inference = not training, # Set to False if training, True if training == False
                                                                                                        )
            


            # Already completely trained
            if hyperparameters["fold_number"] == hyperparameters["num_folds"] - 1:

                # Training model
                if training == True:
                    print(f"Skipping fully trained model: {feature_test_model_name}")

                # Testing model
                else:
                    if selected_model_architecture not in models_info.keys(): 
                        models_info[selected_model_architecture] = {}
                    models_info[selected_model_architecture][feature_test_model_name] = {
                                                                                        "model": model,
                                                                                        "hyperparameters": hyperparameters,
                                                                                        "stats": stats
                                                                                        }

                # Go to next model
                continue

            metrics = ["loss", "accuracy", "precision", "recall", "f1"]
            BATCH_SIZE = hyperparameters["batch_size"]
            num_sets = (hyperparameters["num_folds"] - 1) # Number of sets i.e. the number of (TRAIN_FOLDS, VAL_FOLDS) generated, e.g. if num_folds = 5, there will be 4 sets

            # Create training and test sets + folds
            DH.create_sets(num_context_days = hyperparameters["num_context_days"], shuffle_data_sequences = False, train_split_decimal = hyperparameters["train_split_decimal"])
            DH.create_folds(num_folds = hyperparameters["num_folds"])

            print(f"Hyperparameters used: {hyperparameters}")
            print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"Test feature: {test_feature} | Rolling period: {r_period}")

            # Training:
            # Note: Only entered if creating a new model or continuing training on a model that was interrupted
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
            
            print(f"Completed training for {feature_test_model_name} | Architecture: {selected_model_architecture}")
            if feature_test_model_name == "default":
                print("Skipping to next test feature")
                break

if training == False:
    from math import gcd

    class Tester:
        def __init__(self):
            self.metrics = models_info["LSTM"]["ichimoku_cloud_rolling_2"]["stats"].keys()

        def create_results_dict(self, models_info):

            self.graph_tensors = {}
            """
            tester.graph_tensors.keys() = model architectures
            tester.graph_tensors[model_architecture].keys() = metric names
            tester.graph_tensors[metric_name].keys() = feature names
            tester.graph_tensors[metric_name][feature_name] = Results for that feature for that metric
            """
            for model_architecture in models_info.keys():
                self.graph_tensors[model_architecture] = {}
                for feature_name, model_dict in models_info[model_architecture].items():
                    for metric_name, metric_list in model_dict["stats"].items():
                        if metric_name not in self.graph_tensors[model_architecture]:
                            self.graph_tensors[model_architecture][metric_name] = {}
                        self.graph_tensors[model_architecture][metric_name][feature_name] = metric_list

            print(self.graph_tensors.keys())

        def plot_graphs(self, features_to_test, architectures, show_default):
            
            for model_architecture in architectures:
                for test_feature in features_to_test:
                    for metric_name in self.metrics:
                        stats_for_this_metric = {}

                        # Find the same test feature but for different periods, e.g. train_f1 for average_close_2, average_close_5, average_close_10, etc...
                        for feature_name in self.graph_tensors[model_architecture][metric_name].keys():
                            """Notes:
                            - Is a rolling period
                            - Feature name is "default" and user wants to see the metric results for the default model on the same graph
                            - Test feature is "default", then only display the default metric results on the graph
                            """
                            if feature_name.startswith(f"{test_feature}_rolling") or (feature_name == "default" and show_default) or (test_feature == "default" and feature_name == "default"):
                                stats_for_this_metric[feature_name] = self.graph_tensors[model_architecture][metric_name][feature_name]
                                print(len(stats_for_this_metric[feature_name]))
                                print(feature_name)
                        
                        # After finding all lengths, find the LCF
                        lengths = [len(stat_list) for stat_list in stats_for_this_metric]
                        lcf = self.find_lcf(lengths)

                        print(f"Lengths | {lengths} | LCF {lcf}")

                        fig, ax = plt.subplots()
                        fig.suptitle(metric_name)

                        for feature_name, stat_list in stats_for_this_metric.items():
                            # Fold metrics don't need to be altered
                            if metric_name.startswith("fold"):
                                ax.plot([i for i in range(len(stat_list))], torch.tensor(stat_list), label = feature_name)
                            # Other metrics are too noisy, so apply log10 and 
                            else:
                                ax.plot([i for i in range(len(stat_list))], torch.tensor(stat_list).log10(), label = feature_name)
                                # ax.plot([i for i in range(len(int(stat_list / LCF)))], torch.tensor(stat_list).view(-1, LCF).log10(), label = feature_name)
                        
                        ax.legend()
                        plt.show()

        def find_lcf(self, numbers):
            result = numbers[0]
            for num in numbers[1:]:
                result = gcd(result, num)
            return result
        
        def plot_fold_metrics(self, architectures, show_default):

            # Create a bar chart with each 

            for model_architecture in architectures:
                for metric_name in self.metrics:
                    
                    # Only plot fold metrics
                    if metric_name.startswith("fold"):
                        
                        # For each test feature
                        for test_feature in features_to_test:
                            if test_feature == "default":
                                continue
                            
                            # Create figure and set of subplots
                            fig, ax = plt.subplots()
                            fig.suptitle(f"{model_architecture} | {test_feature} | {metric_name}") # Add title

                            # Create a dictionary for each fold, with the keys being the list of values from each of the periods
                            # E.g. Value = [fold_t_losses[0] for avg_close, then fold_t_losses[1] for avg_close, etc...]
                            features_in_plot = (["default"] if show_default else []) + [f"{test_feature}_rolling_{p}" for p in rolling_periods]
                            print(features_in_plot, test_feature)
                            data = {}
                            for i in range(hyperparameters["num_folds"] - 1): # There should be "num_sets" folds that have stats
                                fold_values = []
                                for feature in features_in_plot:
                                    fold_values.append(self.graph_tensors[model_architecture][metric_name][feature][i])
                                data[f"fold_{i}"] = fold_values

                            # for name, data_list in data.items():
                            #     print(name, len(data_list))

                            # Width of bars
                            total_width = 0.8

                            # Number of bars per group 
                            n_bars = len(rolling_periods)

                            # The width of a single bar
                            bar_width = total_width / n_bars

                            # Cycling colours (Must have)
                            colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] # List of hex strings

                            # List containing handles for the drawn bars (used for the legend)
                            bars = []

                            # Rename each group of bars (to be the periods used in rolling periods)
                            plt.xticks([i for i in range(n_bars + show_default)], (["default"] if show_default else []) + [f"P_{p}" for p in rolling_periods]) 

                            # Iterate over all data
                            for i, (fold_name, values) in enumerate(data.items()):
                                print(fold_name, values)
                                # The offset in x direction of that bar
                                x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

                                # For each of the values
                                for x, value in enumerate(values):
                                    # Draw bar
                                    bar = ax.bar(
                                                x = x + x_offset, 
                                                height = value, 
                                                width = bar_width, 
                                                color = colours[i % len(colours)]
                                                )

                                    # Draw text
                                    ax.text(x = x + x_offset, y = value, s = str(round(value, 3)), ha = "center") # Round values to 2 decimal places
                                
                                # Add a handle to the last drawn bar, which we'll need for the legend
                                bars.append(bar[0])

                            # Draw legend
                            ax.legend(bars, data.keys())
                            plt.show()

        def create_rankings(self, architectures, find_average):
            self.validation_dict = {architecture: {} for architecture in architectures}
            self.train_dict = {architecture: {} for architecture in architectures}
            self.rankings = {architecture: {} for architecture in architectures}

            for architecture in architectures:
                for metric in self.metrics:

                    # print("architectures", self.graph_tensors.keys())
                    # print("metrics", self.graph_tensors[architecture].keys())
                    # print("features", self.graph_tensors[architecture][metric].keys())

                    # Adds to the self.train_dict if the metric type is "train" (e.g. train_loss_i), else self.validation_dict (e.g. val_loss_i)
                    dict_to_add_to = self.train_dict if metric.startswith("train") or metric.startswith("fold_t") else self.validation_dict
                    dict_to_add_to[architecture][metric] = {}

                    for feature in self.graph_tensors[architecture][metric].keys():
                        # Find average performance
                        if find_average:
                            dict_to_add_to[architecture][metric][feature] = sum(self.graph_tensors[architecture][metric][feature]) / len(self.graph_tensors[architecture][metric][feature])
                        # Take last item in the list (final performance)
                        else:
                            dict_to_add_to[architecture][metric][feature] = self.graph_tensors[architecture][metric][feature][-1]

                        print(architecture, metric, feature, len(self.graph_tensors[architecture][metric][feature]), "self.train_dict" if dict_to_add_to == self.train_dict else "val_dict")
                        print(dict_to_add_to[architecture][metric][feature])

                
                    # Sort the features from best to last (For loss, lower is better, for all other metrics, higher is better)
                    sort_by_descending = not(metric.endswith("loss_i") or metric.endswith("loss")) # Ascending for loss, Descending for other metrics
                    print(sort_by_descending)
                    print(dict_to_add_to)
                    dict_to_add_to[architecture][metric] = dict(sorted(dict_to_add_to[architecture][metric].items(), key = lambda items:items[1], reverse = sort_by_descending)) # Sort the entire dictionary for this metric, of feature_names and results, by the results
                    print("AFTER")
                    print(dict_to_add_to)
            
                    # For each feature, add their ranking position for this metric
                    metric_type = "train" if dict_to_add_to == self.train_dict else "validation"
                    for i, feature in enumerate(dict_to_add_to[architecture][metric].keys()):

                        # Create self.rankings for each validation metric type (Keep rankings for train metrics and validation metrics separate)
                        if metric_type not in self.rankings[architecture]:
                            self.rankings[architecture][metric_type] = {}
                        
                        # Add a dictionary containing the overall ranking and the ranking over each metric
                        if feature not in self.rankings[architecture][metric_type]:
                            self.rankings[architecture][metric_type][feature] = {"overall": 0, "per_metric": [i + 1], "per_metric_results": [dict_to_add_to[architecture][metric][feature]]}
                        else:
                            # Append the metric result position for this feature to the end of the per_metric list
                            self.rankings[architecture][metric_type][feature]["per_metric"].append(i + 1)
                            self.rankings[architecture][metric_type][feature]["per_metric_results"].append(dict_to_add_to[architecture][metric][feature])
                    
                    print(dict_to_add_to[architecture][metric].keys())
                    print(self.rankings)
                    print(len(dict_to_add_to[architecture][metric].keys()))

                print(self.rankings[architecture]["train"][feature]["per_metric"])
                
                # For each architecture, add the overall ranking to each feature 
                for metric_type in ["train", "validation"]:
                    selected_dict = self.train_dict if metric_type == "train" else self.validation_dict
                    all_metrics = list(selected_dict[architecture].keys())
                    all_features = selected_dict[architecture][all_metrics[0]].keys()
                    print(all_metrics)
                    print(all_features)
                    print(self.rankings[architecture].keys())
                    sum_ranks = {feature_name: sum(self.rankings[architecture][metric_type][feature_name]["per_metric"]) for feature_name in all_features} # Summing up their ranks across all metrics
                    ordered_overall_ranks = dict(sorted(sum_ranks.items(), key = lambda items:items[1])) # Ordering based on the summed ranks

                    print(sum_ranks)
                    print()
                    print(ordered_overall_ranks)
                    for i, feature_name in enumerate(ordered_overall_ranks.keys()): # Setting the overall ranking
                        self.rankings[architecture][metric_type][feature_name]["overall"] = i + 1

                    print(self.rankings[architecture])

                    # Sort the self.rankings dict based on the overall rank
                    self.rankings[architecture][metric_type] = dict(sorted(self.rankings[architecture][metric_type].items(), key = lambda items:items[1]["overall"]))
                    print(self.rankings[architecture])
            
            print(len(self.rankings["LSTM"]["train"]))
            print(len(self.rankings["LSTM"]["validation"]))
            print(len(self.rankings["RNN"]["train"]))
            print(len(self.rankings["RNN"]["validation"]))
            
        
    print("Displaying statistics")
    tester = Tester()
    tester.create_results_dict(models_info = models_info)
    # tester.plot_graphs(architectures = model_architectures, features_to_test = features_to_test, show_default = True)
    # tester.plot_fold_metrics(architectures = model_architectures, show_default = False)

    tester.create_rankings(architectures = model_architectures, find_average = True)