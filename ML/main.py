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

# Seed for data handler
G = torch.Generator(device = DEVICE)
G.manual_seed(M_SEED)

# Initialising text data handler
TDH = TextDataHandler(device = DEVICE, generator = G)
TDH.retrieve_data()

# Initialising data handler
DH = DataHandler(device = DEVICE, generator = G)
DH.retrieve_data(
                tickers = ["amzn", "ebay", "baba", "3690.HK"],
                start_date = "7/07/2003",
                end_date = "7/07/2023", 
                interval = "1d",
                transform_after = True
                )

for company_data in DH.data_n:
    print("ContainsNaN", company_data.isnan().any().item()) # Check if the tensor contains "nan"

# model = MLP(initial_in = DH.n_features, final_out = 2, N_OR_S = "S")
# optimiser = torch.optim.SGD(params = model.parameters(), lr = 0.0001)

model = RNN(initial_in = DH.n_features, final_out = 2, N_OR_S = "N")
optimiser = torch.optim.Adam(params = model.parameters(), lr = 1e-3)

model.to(device = DEVICE) # Move to selected device

# Prepare data specific to this model:
num_context_days = 10 if isinstance(model, RNN) else 1 # Number of days used as context (Used for RNN)
num_folds = 10 # Number of folds used in cross-validation
# Create training and test sets and data sequences for this model (must be repeated for each model as num_context_days can vary depending on the model used)
DH.create_sets(num_context_days = num_context_days, shuffle_data_sequences = False)
# Create k folds
DH.create_folds(num_folds = 10, N_OR_S = model.N_OR_S)

# Generate folds for this training iteration
TRAIN_FOLDS, VAL_FOLDS = DH.retrieve_k_folds(window_size = 2, N_OR_S = model.N_OR_S)

# Testing generate_batch
X1, Y1 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days)
print(X1.shape, Y1.shape)

X2, Y2 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days)
print(X2.shape, Y2.shape)

X3, Y3 = DH.generate_batch(batch_size = 5, dataset = TRAIN_FOLDS, num_context_days = num_context_days)
print(X3.shape, Y3.shape)

# Training:
EPOCHS = 5000 #200000
BATCH_SIZE = 32
STAT_TRACK_INTERVAL = EPOCHS // 20

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

    for i in range(EPOCHS):
        # Generate inputs and labels
        Xtr, Ytr = DH.generate_batch(batch_size = BATCH_SIZE, dataset = TRAIN_FOLDS, num_context_days = num_context_days)

        # Forward pass
        logits = model(Xtr)

        # Find training loss
        # Note: Did not use F.softmax and F.nll_loss because of floating point accuracy
        loss = F.cross_entropy(logits, Ytr)

        with torch.no_grad():

            # Note: Must set to evaluation mode as BatchNorm layers and Dropout layers behave differently during training and evaluation
            # BatchNorm layers - stops updating the moving averages in BatchNorm layers and uses running statistics instead of per-batch statistics
            # Dropout layers - de-activated during evaluation
            model.eval()

            # Find the accuracy, precision, recall and f1 score on the training batch
            preds = F.softmax(logits, dim = 1) # Softmax to find probability distribution
            train_accuracy, train_precision, train_recall, train_f1 = find_P_A_R(predictions = preds, targets = Ytr)
            
            # Find the loss, accuracy, precision, recall and f1 score on a validation batch
            Xva, Yva = DH.generate_batch(batch_size = BATCH_SIZE, dataset = VAL_FOLDS, num_context_days = num_context_days)
            v_logits = model(Xva)
            v_loss = F.cross_entropy(v_logits, Yva)
            v_preds = F.softmax(v_logits, dim = 1)
            val_accuracy, val_precision, val_recall, val_f1 = find_P_A_R(predictions = v_preds, targets = Yva)

            model.train()

        # Backward pass
        optimiser.zero_grad()
        loss.backward()

        # Update model parameters
        optimiser.step()

        # ----------------------------------------------
        # Tracking stats
        
        train_loss_i.append(loss.item())
        val_loss_i.append(v_loss.item())

        train_accuracy_i.append(train_accuracy)
        val_accuracy_i.append(val_accuracy)

        train_precision_i.append(train_precision)
        val_precision_i.append(val_precision)

        train_recall_i.append(train_recall)
        val_recall_i.append(val_recall)

        train_f1_i.append(train_f1)
        val_f1_i.append(val_f1)

        if i == 0 or (i + 1) % STAT_TRACK_INTERVAL == 0:
            print(f"K: {k + 1}/{num_sets} | Epoch: {i + 1} | TLoss: {loss.item()} | VLoss: {v_loss.item()} | TAccuracy: {train_accuracy} | VAccuracy: {val_accuracy} | TPrecision: {train_precision} | VPrecision: {val_precision} | TRecall: {train_recall} | VRecall: {val_recall} | TF1 {train_f1} | VF1: {val_f1}")


    # Record metrics for this fold:
    # -EPOCHS: = Last EPOCHS items (i.e. all the statistics from this fold)
    fold_t_losses.append((sum(train_loss_i[-EPOCHS:]) / EPOCHS))
    fold_v_losses.append((sum(val_loss_i[-EPOCHS:]) / EPOCHS))

    fold_t_accuracies.append((sum(train_accuracy_i[-EPOCHS:]) / EPOCHS))
    fold_v_accuracies.append((sum(val_accuracy_i[-EPOCHS:]) / EPOCHS))

    fold_t_precisions.append((sum(train_precision_i[-EPOCHS:]) / EPOCHS))
    fold_v_precisions.append((sum(val_precision_i[-EPOCHS:]) / EPOCHS))

    fold_t_recalls.append((sum(train_recall_i[-EPOCHS:]) / EPOCHS))
    fold_v_recalls.append((sum(val_recall_i[-EPOCHS:]) / EPOCHS))

    fold_t_f1s.append((sum(train_f1_i[-EPOCHS:]) / EPOCHS))
    fold_v_f1s.append((sum(val_f1_i[-EPOCHS:]) / EPOCHS))

# Set model to evaluation mode (For dropout + batch norm layers)
model.eval()

print("-----------------------------------------------------------------")
print("Metrics per fold")

print(f"TrainAccuracies: {fold_t_accuracies}")
print(f"ValAccuracies: {fold_v_accuracies}")
print(f"TrainLosses: {fold_t_losses}")
print(f"ValLosses: {fold_v_losses}")
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
A = 50
train_loss_i = torch.tensor(train_loss_i).view(-1, A).mean(1)
val_loss_i = torch.tensor(val_loss_i).view(-1, A).mean(1)

# train_loss_i = torch.tensor(torch.log10(train_loss_i)).view(-1, A).mean(1)
# val_loss_i = torch.tensor(torch.log10(val_loss_i)).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / A))], train_loss_i, label = "Train")
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / A))], val_loss_i, label = "Validation")
ax.legend()#
plt.show()

print("-----------------------------------------------------------------")
print("Accuracy during training")

B = 100
train_accuracy_i = torch.tensor(train_accuracy_i).view(-1, B).mean(1)
val_accuracy_i = torch.tensor(val_accuracy_i).view(-1, B).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / B))], train_accuracy_i, label = "Train")
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / B))], val_accuracy_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Precision during training")

C = 100
train_precision_i = torch.tensor(train_precision_i).view(-1, C).mean(1)
val_precision_i = torch.tensor(val_precision_i).view(-1, C).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / C))], train_precision_i, label = "Train")
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / C))], val_precision_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Recall during training")

D = 100
train_recall_i = torch.tensor(train_recall_i).view(-1, D).mean(1)
val_recall_i = torch.tensor(val_recall_i).view(-1, D).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / D))], train_recall_i, label = "Train")
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / D))], val_recall_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("F1 score during training")

E = 100
train_f1_i = torch.tensor(train_f1_i).view(-1, E).mean(1)
val_f1_i = torch.tensor(val_f1_i).view(-1, E).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / E))], train_f1_i, label = "Train")
ax.plot([i for i in range(int((EPOCHS * (num_sets)) / E))], val_f1_i, label = "Validation")
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