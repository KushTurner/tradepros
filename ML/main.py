import torch
import torch.nn.functional as F
from data_handler import DataHandler
from matplotlib import pyplot as plt
from models import MLP

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

EPOCHS = 20000
BATCH_SIZE = 32
STAT_TRACK_INTERVAL = EPOCHS // 20

train_loss_i = []
val_loss_i = []

for i in range(EPOCHS):
    # Generate inputs and labels
    Xtr, Ytr = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "train")

    # Forward pass
    logits = model(Xtr)

    # Find training loss
    loss = F.cross_entropy(logits, Ytr)

    # Find validation loss
    with torch.no_grad():
        Xva, Yva = DH.generate_batch(batch_size = BATCH_SIZE, split_selected = "val")
        v_logits = model(Xva)
        v_loss = F.cross_entropy(v_logits, Yva)
    
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
        print(f"Epoch: {i + 1} | TrainLoss: {loss.item()} | ValLoss: {v_loss.item()}")


# Plotting train / validation loss
A = 80
train_loss_i = torch.tensor(train_loss_i).view(-1, A).mean(1)
val_loss_i = torch.tensor(val_loss_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(EPOCHS / A))], train_loss_i, label = "Train")
ax.plot([i for i in range(int(EPOCHS / A))], val_loss_i, label = "Validation")
ax.legend()
plt.show()