import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from data_handler import DataHandler
from models import MLP
from tools import evaluate_accuracy, find_accuracy

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
print(f"DEVICE | {DEVICE}")

M_SEED = 2004
# Seed for the model
torch.manual_seed(M_SEED)

# Seed for data handler
G = torch.Generator(device = DEVICE)
G.manual_seed(M_SEED)

# Initialising data handler
DH = DataHandler(device = DEVICE, generator = G)
DH.retrieve_data(
                ticker = "amzn", 
                start_date = "7/07/2003", 
                end_date = "7/07/2023", 
                interval = "1d" 
                )
print(DH.data.shape)
print(DH.data.isnan().any().item()) # Check if the tensor contains "nan"

# Create train/val/test splits
DH.create_splits()

# Testing generate_batch
X1, Y1 = DH.generate_batch(batch_size = 5, split_selected = "train")
print(X1.shape, Y1.shape)

X2, Y2 = DH.generate_batch(batch_size = 5, split_selected = "val")
print(X2.shape, Y2.shape)

X3, Y3 = DH.generate_batch(batch_size = 5, split_selected = "test")
print(X3.shape, Y3.shape)


model = MLP(initial_in = DH.n_features, final_out = 2)
model.to(device = DEVICE) # Move to selected device
optimiser = torch.optim.SGD(params = model.parameters(), lr = 0.0001)

EPOCHS = 200000
BATCH_SIZE = 32
STAT_TRACK_INTERVAL = EPOCHS // 20

train_loss_i = []
val_loss_i = []
train_accuracy_i = []
val_accuracy_i = []

for i in range(EPOCHS):
    # Generate inputs and labels
    Xtr, Ytr = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "train")

    # Forward pass
    logits = model(Xtr)

    # Find training loss
    # Note: Did not use F.softmax and F.nll_loss because of floating point accuracy
    loss = F.cross_entropy(logits, Ytr)

    with torch.no_grad():
        
        # Find train accuracy on current batch
        preds = F.softmax(logits, dim = 1) # Softmax to find probability distribution
        train_accuracy = find_accuracy(predictions = preds, targets = Ytr, batch_size = BATCH_SIZE)
        train_accuracy_i.append(find_accuracy(predictions = preds, targets = Ytr, batch_size = BATCH_SIZE))

        # Find validation loss
        Xva, Yva = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "val")
        v_logits = model(Xva)
        v_loss = F.cross_entropy(v_logits, Yva)

        # Find validation accuracy on current batch
        v_preds = F.softmax(v_logits, dim = 1) # Softmax to find probability distribution
        val_accuracy = find_accuracy(predictions = v_preds, targets = Yva, batch_size = BATCH_SIZE)
        val_accuracy_i.append(val_accuracy)
    

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


print("-----------------------------------------------------------------")
print("Loss during training")

# Plotting train / validation loss
A = 80
train_loss_i = torch.tensor(train_loss_i).view(-1, A).mean(1)
val_loss_i = torch.tensor(val_loss_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(EPOCHS / A))], train_loss_i, label = "Train")
ax.plot([i for i in range(int(EPOCHS / A))], val_loss_i, label = "Validation")
ax.legend()
plt.show()

print("-----------------------------------------------------------------")
print("Accuracy during training")

B = 800
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
                                    split_name = "Train"
                                    )

val_accuracies = evaluate_accuracy(
                                steps = accuracy_steps, 
                                batch_size = accuracy_bs, 
                                generate_batch_f = DH.generate_batch, 
                                selected_model = model, 
                                check_interval = CHECK_INTERVAL,
                                split_name = "Val"
                                )

train_accuracies = torch.tensor(train_accuracies).view(-1, C).mean(1)
val_accuracies = torch.tensor(val_accuracies).view(-1, C).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(accuracy_steps / C))], train_accuracies, label = "Train")
ax.plot([i for i in range(int(accuracy_steps / C))], val_accuracies, label = "Validation")
ax.legend()
plt.show()