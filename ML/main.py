import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from data_handler import *
from models import MLP, RNN
from tools import evaluate_accuracy, find_P_A_R

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
                start_date = "1/01/2015",
                end_date = "31/12/2019", 
                interval = "1d",
                )

# Initialising text data handler
TDH = TextDataHandler(dates = DH.dates, device = DEVICE, generator = G)
TDH.retrieve_data()

# ---------------------------------------------------------------------------------------
# Model loading

model_number_load = 0 # Number of the model to load, set as None to create a new model

# Load existing model
if model_number_load != None:
    checkpoint_directory = f"model_checkpoints/{model_number_load}"

# Creates a new model
else:
    if os_path_exists("model_checkpoints") == False:
        os_mkdir("model_checkpoints")
    
    from os import listdir as os_listdir
    checkpoint_directory = f"model_checkpoints/{len(os_listdir('model_checkpoints'))}"
    os_mkdir(checkpoint_directory) 

if model_number_load != None:
    print("Loading existing model")
    from os import listdir as os_listdir
    existing_checkpoint_path = f"{checkpoint_directory}/fold_{len(os_listdir(f'{checkpoint_directory}')) - 1}.pth"
    checkpoint = torch.load(existing_checkpoint_path) # Load the last checkpoint (Which would be the complete model)
    
    print(existing_checkpoint_path)
    print(checkpoint.keys())

    if checkpoint["architecture"] == "RNN":
        model = RNN(initial_in = checkpoint["n_features"], final_out = 2, N_OR_S = checkpoint["N_OR_S"], device = DEVICE)
        optimiser = torch.optim.Adam(params = model.parameters(), lr = checkpoint["learning_rate"])
    
    elif checkpoint["architecture"] == "MLP":
        model = MLP(initial_in = checkpoint["n_features"], final_out = 2, N_OR_S = checkpoint["N_OR_S"], device = DEVICE)
        optimiser = torch.optim.SGD(params = model.parameters(), lr = checkpoint["learning_rate"])
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"]) 
    num_context_days = checkpoint["num_context_days"]
    num_folds = checkpoint["num_folds"]

else:
    # Note: Use normalised data ("N") for RNN and standardised data ("S") for MLP 
    learning_rate = 1e-3
    model = RNN(initial_in = DH.n_features, final_out = 2, N_OR_S = "N", device = DEVICE)
    optimiser = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    
    # learning_rate = 0.0001
    # model = MLP(initial_in = DH.n_features, final_out = 2, N_OR_S = "S")
    # optimiser = torch.optim.SGD(params = model.parameters(), lr = learning_rate)

    # Prepare data specific to this model:
    num_context_days = 10 if isinstance(model, RNN) else 1 # Number of days used as context (Used for RNN)
    num_folds = 5 # Number of folds used in cross-validation


# Retrieve the data for the historical dataset (combined with the sentiments)
"""
Notes:
- Executed here because dependent on whether an existing model is being loaded and if dated sentiments were used (i.e. sentiments as an input feature)
- DH and TDH still need to be instantiated and prepared so that the testing set will be the same (if evaluating on the test set)
"""
DH.retrieve_data(
                tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                start_date = "1/01/2015",
                end_date = "31/12/2019", 
                interval = "1d",
                transform_after = True,
                dated_sentiments = TDH.dated_sentiments if checkpoint["uses_dated_sentiments"] else None # Dated sentiments for each company (None if not using)
                )

for company_data in DH.data_n:
    print("ContainsNaN", company_data.isnan().any().item()) # Check if the tensor contains "nan"

model.to(device = DEVICE) # Move to selected device

# Create training and test sets and data sequences for this model (must be repeated for each model as num_context_days can vary depending on the model used)
DH.create_sets(num_context_days = num_context_days, shuffle_data_sequences = False)
# Create k folds
DH.create_folds(num_folds = num_folds, N_OR_S = model.N_OR_S)

# Generate folds for this training iteration
TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2, N_OR_S = model.N_OR_S)

# ---------------------------------------------------------------------------------------

# Testing generate_batch
X1, Y1 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days, start_idx = 0)
print(X1.shape, Y1.shape)

X2, Y2 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days, start_idx = 0)
print(X2.shape, Y2.shape)

X3, Y3 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days, start_idx = 0)
print(X3.shape, Y3.shape)

# Training:
if model_number_load == None:
    BATCH_SIZE = 32

    # Over epochs
    train_loss_i = []
    val_loss_i = []
    train_accuracy_i = []
    val_accuracy_i = []
    train_precision_i = []
    val_precision_i = []
    train_recall_i = []
    val_recall_i = []
    train_f1_i = []
    val_f1_i = []

    # Over folds
    fold_t_accuracies = []
    fold_v_accuracies = []
    fold_t_losses = []
    fold_v_losses = []
    fold_t_precisions = []
    fold_v_precisions = []
    fold_t_recalls = []
    fold_v_recalls = []
    fold_t_f1s = []
    fold_v_f1s = []

    num_sets = (num_folds - 1) # Number of sets i.e. the number of (TRAIN_FOLDS, VAL_FOLDS) generated, e.g. if num_folds = 5, there will be 4 sets

    for k in range(num_sets):

        # Generate folds for this training iteration    
        # Notes:
        # - Window size starts at 2: Window sizes = (2 + 0), (2 + 1) (2 + 2) (2 + 3)
        # - Number of total sets = (num_folds - 1)
        TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2 + k, N_OR_S = model.N_OR_S)
        

        # Rolling window variables: (Starting indexes, number of batches in each of the training / validation folds, interval for evaluating on the validation fold)
        train_fold_sidx = 0
        val_fold_sidx = 0
        num_validations = VAL_FOLDS[1].shape[0] - BATCH_SIZE + 1
        num_trains = TRAIN_FOLDS[1].shape[0] - BATCH_SIZE + 1
        validation_interval = int(num_trains / num_validations)
        multiplicative_trains = 1 # Number of times each batch will be used to train the model

        print(f"num_trains * multiplicative_trains: {num_trains * multiplicative_trains} | num_trains: {num_trains} | num_validations: {num_validations} | validation_interval: {validation_interval}")
        print(f"Training examples: {TRAIN_FOLDS[1].shape} | Validation examples: {VAL_FOLDS[1].shape}")

        for i in range(num_trains):

            # Generate inputs and labels
            Xtr, Ytr = DH.generate_batch(batch_size = BATCH_SIZE, dataset = TRAIN_FOLDS, num_context_days = num_context_days, start_idx = train_fold_sidx)
            train_fold_sidx += 1

            for j in range(multiplicative_trains): # Train on a single batch j times

                # Forward pass
                logits = model(Xtr)

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
                    Xva, Yva = DH.generate_batch(batch_size = BATCH_SIZE, dataset = VAL_FOLDS, num_context_days = num_context_days, start_idx = val_fold_sidx)
                    val_fold_sidx += 1

                    v_logits = model(Xva)
                    v_loss = F.cross_entropy(v_logits, Yva)
                    v_preds = F.softmax(v_logits, dim = 1)
                    val_accuracy, val_precision, val_recall, val_f1 = find_P_A_R(predictions = v_preds, targets = Yva)

                    model.train()
            
            # ----------------------------------------------
            # Tracking stats

            train_loss_i.append(loss.item())
            train_accuracy_i.append(train_accuracy)
            train_precision_i.append(train_precision)
            train_recall_i.append(train_recall)
            train_f1_i.append(train_f1)

            val_loss_i.append(v_loss.item())
            val_accuracy_i.append(val_accuracy)
            val_precision_i.append(val_precision)
            val_recall_i.append(val_recall)
            val_f1_i.append(val_f1)

            if i == 0 or (num_trains % i) == 1 or (i + 1) % validation_interval == 0: # First, last, validation interval
                print(f"K: {k + 1}/{num_sets} | Epoch: T: {(i + 1) * multiplicative_trains}/{num_trains * multiplicative_trains} V: {val_fold_sidx}/{num_validations} | TLoss: {loss.item()} | VLoss: {v_loss.item()} | TAccuracy: {train_accuracy} | VAccuracy: {val_accuracy} | TPrecision: {train_precision} | VPrecision: {val_precision} | TRecall: {train_recall} | VRecall: {val_recall} | TF1 {train_f1} | VF1: {val_f1}")

        # Record metrics for this fold:
        # -num_trains: = Last num_trains items (i.e. all the statistics from this fold)
        # /num_trains = Average metric in this fold
        fold_t_losses.append((sum(train_loss_i[-num_trains:]) / num_trains))
        fold_v_losses.append((sum(val_loss_i[-num_trains:]) / num_trains))

        fold_t_accuracies.append((sum(train_accuracy_i[-num_trains:]) / num_trains))
        fold_v_accuracies.append((sum(val_accuracy_i[-num_trains:]) / num_trains))

        fold_t_precisions.append((sum(train_precision_i[-num_trains:]) / num_trains))
        fold_v_precisions.append((sum(val_precision_i[-num_trains:]) / num_trains))

        fold_t_recalls.append((sum(train_recall_i[-num_trains:]) / num_trains))
        fold_v_recalls.append((sum(val_recall_i[-num_trains:]) / num_trains))

        fold_t_f1s.append((sum(train_f1_i[-num_trains:]) / num_trains))
        fold_v_f1s.append((sum(val_f1_i[-num_trains:]) / num_trains))

        # ----------------------------------------------
        # Saving checkpoint
        
        checkpoint = {
                    "architecture": model.__class__.__name__,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "n_features": DH.n_features,

                    "batch_size": BATCH_SIZE,
                    "learning_rate": learning_rate,
                    "multiplicative_trains": multiplicative_trains,
                    "N_OR_S": model.N_OR_S,
                    "uses_dated_sentiments": True,
                    "num_folds": num_folds,
                    "num_context_days": num_context_days,
                    "fold_number": k + 1,

                    "train_loss_i": train_loss_i,
                    "val_loss_i": val_loss_i,
                    
                    "train_accuracy_i": train_accuracy_i,
                    "val_accuracy_i": val_accuracy_i,

                    "train_precision_i": train_precision_i,
                    "val_precision_i": val_precision_i,

                    "train_recall_i": train_recall_i,
                    "val_recall_i": val_recall_i,

                    "train_f1_i": train_f1_i,
                    "val_f1_i": val_f1_i,

                    "fold_t_losses": fold_t_losses,
                    "fold_v_losses": fold_v_losses,

                    "fold_t_accuracies": fold_t_accuracies,
                    "fold_v_accuracies": fold_v_accuracies,

                    "fold_t_precisions": fold_t_precisions,
                    "fold_v_precisions": fold_v_precisions,

                    "fold_t_recalls": fold_t_recalls,
                    "fold_v_recalls": fold_v_recalls,

                    "fold_t_f1s": fold_t_f1s,
                    "fold_v_f1s": fold_v_f1s,
                    }
        torch.save(obj = checkpoint, f = f"{checkpoint_directory}/fold_{k}.pth")

# Set model to evaluation mode (For dropout + batch norm layers)
model.eval()

for checkpoint_val in checkpoint.values():
    if type(checkpoint_val) == list:
        print(checkpoint_val)

print("-----------------------------------------------------------------")
print("Metrics per fold")


print(f"TrainLosses: {fold_t_losses}")
print(f"ValLosses: {fold_v_losses}")
print(f"TrainAccuracies: {fold_t_accuracies}")
print(f"ValAccuracies: {fold_v_accuracies}")
print(f"TrainPrecisions: {fold_t_precisions}")
print(f"ValPrecisions: {fold_v_precisions}")
print(f"TrainRecalls: {fold_t_recalls}")
print(f"ValRecalls: {fold_v_recalls}")
print(f"TrainF1s: {fold_t_f1s}")
print(f"ValF1s: {fold_v_f1s}")

print("-----------------------------------------------------------------")
print("Metrics across folds")

print(f"TAccuracy: {sum(fold_t_accuracies) / num_sets}) | VAccuracy: {sum(fold_v_accuracies) / num_sets} | TLoss: {sum(fold_t_losses) / num_sets} | VLoss: {sum(fold_v_losses) / num_sets} | TPrecision: {sum(fold_t_precisions) / num_sets} | VPrecision: {sum(fold_v_precisions) / num_sets} | TRecall: {sum(fold_t_recalls) / num_sets} | VRecall {sum(fold_v_recalls) / num_sets} | TF1: {sum(fold_t_f1s) / num_sets} | VF1: {sum(fold_v_f1s) / num_sets}")

print("-----------------------------------------------------------------")
print("Loss during training")

# Plotting train / validation loss
total_epochs = len(train_loss_i)
print(total_epochs)
A = 47 # Replace with a factor of the total number of epochs
train_loss_i = torch.tensor(train_loss_i).view(-1, A).mean(1)
val_loss_i = torch.tensor(val_loss_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(total_epochs / A))], train_loss_i, label = "Train")
ax.plot([i for i in range(int(total_epochs / A))], val_loss_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Accuracy during training")

train_accuracy_i = torch.tensor(train_accuracy_i).view(-1, A).mean(1)
val_accuracy_i = torch.tensor(val_accuracy_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(total_epochs / A))], train_accuracy_i, label = "Train")
ax.plot([i for i in range(int(total_epochs / A))], val_accuracy_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Precision during training")

train_precision_i = torch.tensor(train_precision_i).view(-1, A).mean(1)
val_precision_i = torch.tensor(val_precision_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(total_epochs / A))], train_precision_i, label = "Train")
ax.plot([i for i in range(int(total_epochs / A))], val_precision_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Recall during training")

train_recall_i = torch.tensor(train_recall_i).view(-1, A).mean(1)
val_recall_i = torch.tensor(val_recall_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(total_epochs / A))], train_recall_i, label = "Train")
ax.plot([i for i in range(int(total_epochs / A))], val_recall_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("F1 score during training")

train_f1_i = torch.tensor(train_f1_i).view(-1, A).mean(1)
val_f1_i = torch.tensor(val_f1_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(total_epochs / A))], train_f1_i, label = "Train")
ax.plot([i for i in range(int(total_epochs / A))], val_f1_i, label = "Validation")
ax.legend()
plt.show()


# print("-----------------------------------------------------------------")
# print("Accuracy after training")

# # Find final accuracy on train and validation split
# accuracy_steps = 10000
# accuracy_bs = 20
# C = 50
# CHECK_INTERVAL = accuracy_steps // 20

# train_accuracies = evaluate_accuracy(
#                                     steps = accuracy_steps, 
#                                     batch_size = accuracy_bs, 
#                                     generate_batch_f = DH.generate_batch, 
#                                     selected_model = model, 
#                                     check_interval = CHECK_INTERVAL,
#                                     dataset = getattr(DH, f"TRAIN_S{model.N_OR_S}")
#                                     split_name = "Train",
#                                     num_context_days = num_context_days
#                                     )

# val_accuracies = evaluate_accuracy(
#                                 steps = accuracy_steps, 
#                                 batch_size = accuracy_bs, 
#                                 generate_batch_f = DH.generate_batch, 
#                                 selected_model = model, 
#                                 check_interval = CHECK_INTERVAL,
#                                 split_name = "Val",
#                                 num_context_days = num_context_days
#                                 )

# train_accuracies = torch.tensor(train_accuracies).view(-1, C).mean(1)
# val_accuracies = torch.tensor(val_accuracies).view(-1, C).mean(1)

# fig, ax = plt.subplots()
# ax.plot([i for i in range(int(accuracy_steps / C))], train_accuracies, label = "Train")
# ax.plot([i for i in range(int(accuracy_steps / C))], val_accuracies, label = "Validation")
# ax.legend()
# plt.show()