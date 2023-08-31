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
                    "macd"
                    ]
rolling_periods = [2, 5, 10, 15, 20]

# Initalise model manager
model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = TDH)

models_info = {}
training = False

for test_feature in features_to_test: # For each testing feature
    for r_period in rolling_periods: # For each rolling period
        
        feature_test_model_name = f"{test_feature}_rolling_{r_period}"
        model_number_load = feature_test_model_name if os_path_exists(f"model_checkpoints/feature_test_models/{feature_test_model_name}") else None
        manual_hyperparams = {
                            "architecture": "RNN",
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
                models_info[feature_test_model_name] = {
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
        
        print(f"Completed training for {feature_test_model_name}")


class Tester:
    def create_results_dict(self, models_info):
    
        self.graph_tensors = {}
        self.metrics = []
        """
        tester.graph_tensors.keys() = metric names
        tester.graph_tensors[metric_name].keys() = feature names
        tester.graph_tensors[metric_name][feature_name] = Results for that feature for that metric
        """
        for feature_name, model_dict in models_info.items():
            for metric_name, metric_list in model_dict["stats"].items():
                if metric_name not in self.graph_tensors:
                    self.graph_tensors[metric_name] = {}
                    self.metrics.append(metric_name)
                self.graph_tensors[metric_name][feature_name] = metric_list

    def plot_graphs(self, features_to_test):

        for test_feature in features_to_test:
            for metric_name in self.metrics:
                stats_for_this_metric = {}

                # Find the same test feature but for different periods, e.g. train_f1 for average_close_2, average_close_5, average_close_10, etc...
                for feature_name in self.graph_tensors[metric_name].keys():
                    if feature_name.startswith(f"{test_feature}_rolling"):
                        stats_for_this_metric[feature_name] = self.graph_tensors[metric_name][feature_name]
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
    

    
if training == False:
    from math import gcd
    print("Displaying statistics")
    tester = Tester()
    tester.create_results_dict(models_info = models_info)
    tester.plot_graphs(features_to_test = features_to_test)