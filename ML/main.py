import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from data_handler import DataHandler
from models import MLP, RNN
from tools import evaluate_accuracy, find_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE | {DEVICE}")

M_SEED = 2004
# Seed for the model
torch.manual_seed(M_SEED)

# Seed for data handler
G = torch.Generator(device = DEVICE)
G.manual_seed(M_SEED)

# Initialising data handler
DH = DataHandler(device = DEVICE, generator = G)
num_context_days = None # 10 # Number of days used as context (Used for RNN)
DH.retrieve_data(
                ticker = "amzn", 
                start_date = "7/07/2003",
                end_date = "7/07/2023", 
                interval = "1d",
                normalise = True,
                standardise = False
                )
print(DH.data.shape)
print("ContainsNaN",DH.data.isnan().any().item()) # Check if the tensor contains "nan"

# Create train/val/test splits
DH.create_splits(num_context_days = num_context_days)

# Testing generate_batch
X1, Y1 = DH.generate_batch(batch_size = 5, split_selected = "train", num_context_days = num_context_days)
print(X1.shape, Y1.shape)

X2, Y2 = DH.generate_batch(batch_size = 5, split_selected = "val", num_context_days = num_context_days)
print(X2.shape, Y2.shape)

X3, Y3 = DH.generate_batch(batch_size = 5, split_selected = "test", num_context_days = num_context_days)
print(X3.shape, Y3.shape)


model = MLP(initial_in = DH.n_features, final_out = 2)
optimiser = torch.optim.SGD(params = model.parameters(), lr = 0.0001)

# model = RNN(initial_in = DH.n_features, final_out = 2)
# optimiser = torch.optim.Adam(params = model.parameters(), lr = 1e-3)

model.to(device = DEVICE) # Move to selected device

EPOCHS = 200000
BATCH_SIZE = 32
STAT_TRACK_INTERVAL = EPOCHS // 20

train_loss_i = []
val_loss_i = []
train_accuracy_i = []
val_accuracy_i = []

for i in range(EPOCHS):
    # Generate inputs and labels
    Xtr, Ytr = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "train", num_context_days = num_context_days)

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

        # Find train accuracy on current batch
        preds = F.softmax(logits, dim = 1) # Softmax to find probability distribution
        train_accuracy = find_accuracy(predictions = preds, targets = Ytr, batch_size = BATCH_SIZE)
        train_accuracy_i.append(find_accuracy(predictions = preds, targets = Ytr, batch_size = BATCH_SIZE))

        # Find validation loss
        Xva, Yva = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "val", num_context_days = num_context_days)
        v_logits = model(Xva)
        v_loss = F.cross_entropy(v_logits, Yva)

        # Find validation accuracy on current batch
        v_preds = F.softmax(v_logits, dim = 1) # Softmax to find probability distribution
        val_accuracy = find_accuracy(predictions = v_preds, targets = Yva, batch_size = BATCH_SIZE)
        val_accuracy_i.append(val_accuracy)

        model.train()

    # Backward pass
    optimiser.zero_grad()
    loss.backward()

    # Update model parameters
    optimiser.step()

    # ----------------------------------------------
    # Tracking stats

    train_loss_i.append(loss.log10().item())
    val_loss_i.append(v_loss.log10().item())

    if i == 0 or (i + 1) % STAT_TRACK_INTERVAL == 0:
        print(f"Epoch: {i + 1} | TrainLoss: {loss.item()} | ValLoss: {v_loss.item()} | CurrentTrainAccuracy: {train_accuracy} | CurrentValAccuracy: {val_accuracy}")

# Set model to evaluation mode (For dropout + batch norm layers)
model.eval()

print("-----------------------------------------------------------------")
print("Loss during training")

# Plotting train / validation loss
A = 50
train_loss_i = torch.tensor(train_loss_i).view(-1, A).mean(1)
val_loss_i = torch.tensor(val_loss_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(EPOCHS / A))], train_loss_i, label = "Train")
ax.plot([i for i in range(int(EPOCHS / A))], val_loss_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Accuracy during training")

B = 500
train_accuracy_i = torch.tensor(train_accuracy_i).view(-1, B).mean(1)
val_accuracy_i = torch.tensor(val_accuracy_i).view(-1, B).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(EPOCHS / B))], train_accuracy_i, label = "Train")
ax.plot([i for i in range(int(EPOCHS / B))], val_accuracy_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Accuracy after training")

# Find final accuracy on train and validation split
accuracy_steps = 10000
accuracy_bs = 20
C = 50
CHECK_INTERVAL = accuracy_steps // 20

train_accuracies = evaluate_accuracy(
                                    steps = accuracy_steps, 
                                    batch_size = accuracy_bs, 
                                    generate_batch_f = DH.generate_batch, 
                                    selected_model = model, 
                                    check_interval = CHECK_INTERVAL,
                                    split_name = "Train",
                                    num_context_days = num_context_days
                                    )

val_accuracies = evaluate_accuracy(
                                steps = accuracy_steps, 
                                batch_size = accuracy_bs, 
                                generate_batch_f = DH.generate_batch, 
                                selected_model = model, 
                                check_interval = CHECK_INTERVAL,
                                split_name = "Val",
                                num_context_days = num_context_days
                                )

train_accuracies = torch.tensor(train_accuracies).view(-1, C).mean(1)
val_accuracies = torch.tensor(val_accuracies).view(-1, C).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(accuracy_steps / C))], train_accuracies, label = "Train")
ax.plot([i for i in range(int(accuracy_steps / C))], val_accuracies, label = "Validation")
ax.legend()
plt.show()