import torch
import torch.nn as nn

class Precision(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        correct = torch.sum(torch.round(torch.clip(inputs * targets,0,1)))
        predicted = torch.sum(torch.round(torch.clip(inputs,0,1)))
        precision = correct/predicted
        return precision

class Recall(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        true_positive = torch.sum(torch.round(torch.clip(inputs * targets,0,1)))
        false_negative = torch.sum(torch.round(torch.clip(targets,0,1)))
        recall = true_positive/false_negative
        return recall

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return dice
    
class IoULoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)     
        return 1 - IoU