import torch

def make_trainable(model, val):
    for p in model.parameters():
        p.requires_grad = val



### TO DO: write metrics
### from sklearn.metrics import precision_recall_curve