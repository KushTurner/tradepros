from torch import no_grad as torch_no_grad
from torch import max as torch_max
from torch import sum as torch_sum
from torch.nn import functional as F
from torch import logical_and as torch_logical_and
    
def count_correct_preds(predictions, targets):

    # Selects the highest probability assigned between the two outputs (i.e. the highest probability of the stock price going up or down)
    _, output = torch_max(predictions, dim = 1) 
    
    # Return the number of correct predictions
    return torch_sum(output == targets).item()

def find_P_A_R(predictions, targets):
    # Finds the precision, accuracy and recall for a given batch

    # Selects the highest probability assigned between the two outputs (i.e. the highest probability of the stock price going up or down)
    _, outputs = torch_max(predictions, dim = 1) 

    # Number of true positives (prediction == 1 when target == 1)
    true_positives = torch_logical_and(outputs == 1, targets == 1).sum().item()

    # Number of false positives (prediction == 1 when target == 0) 
    false_positives = torch_logical_and(outputs == 1, targets == 0).sum().item()
    
    # Number of false negatives (prediction == 0 when target == 0)
    true_negatives = torch_logical_and(outputs == 0, targets == 0).sum().item()

    # Number of false negatives (prediction == 0 when target == 1)
    false_negatives = torch_logical_and(outputs == 0, targets == 1).sum().item()

    # Accuracy = Number of correct predictions / Total number of predictions
    # Precision = Number of true positives / (Number of true positives + Number of false positives) [Proportion of true positive predictions that the stock went up out of all predictions out of all the predictions that the model made that the stock went up]
    # Recall = Number of true positives / (Number of true positives + Number of false negatives) [Proportion of true positive predictions that the stock went up out of all the times it actually went up in the dataset]
    # F1 score = 2 * ((precision * recall) / (precision + recall))

    # Notes: 
    # - Higher F1 score is better (Ranges from 0 to 1)
    # - Recall measures the ability of the model to identify positive instances (stock trend going up) out of all the positive instances that occurred
    # - Precision measures the accuracy of actual positive predictions amongst all the instances predicted as positive by the model
    # - If precision or recall is 0, F1 score will also be 0
    accuracy = ((true_positives + true_negatives) / predictions.shape[0]) * 100
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if precision != 0 or recall != 0 else 0

    # print("O",outputs)
    # print("T",targets)
    # print(true_positives, false_positives, true_negatives, false_negatives)
    # print(accuracy, precision, recall, f1_score)

    return accuracy, precision, recall, f1_score
