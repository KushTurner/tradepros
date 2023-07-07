import torch
from data_handler import DataHandler

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
                interval = "1wk" 
                )
print(DH.data.shape)
print(DH.data.isnan().any().item()) # Check if the tensor contains "nan"

# Test batch
X, Y = DH.generate_batch(batch_size = 5)
print(X.shape, Y.shape)
print(X)
print(Y)