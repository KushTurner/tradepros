from torch import no_grad as torch_no_grad
from torch import max as torch_max
from torch import sum as torch_sum
from torch.nn import functional as F

def evaluate_accuracy(steps, batch_size, generate_batch_f, selected_model, check_interval, split_name, num_context_days):
    
    accuracies = []
    num_correct = 0
    num_tested = 0

    with torch_no_grad():
        
        for i in range(steps):
            # Generate batch from a split
            Xa, Ya = generate_batch_f(batch_size, split_name, num_context_days)

            # Forward pass
            logits = selected_model(Xa)

            # Find probability distribution (the predictions)
            preds = F.softmax(logits, dim = 1)

            # Track stats
            num_correct += count_correct_preds(predictions = preds, targets = Ya)
            num_tested += batch_size
            accuracies.append((num_correct / num_tested) * 100)
            
            if (i + 1) % check_interval == 0:
                print(f"Correct predictions: {num_correct} / {num_tested} | {split_name}Accuracy(%): {(num_correct / num_tested) * 100}")
    
    return accuracies

def find_accuracy(predictions, targets, batch_size):
    return (count_correct_preds(predictions, targets) / batch_size) * 100
    
def count_correct_preds(predictions, targets):

    # Selects the highest probability assigned between the two outputs (i.e. the highest probability of the stock price going up or down)
    _, output = torch_max(predictions, dim = 1) 
    
    # Return the number of correct predictions
    return torch_sum(output == targets).item()