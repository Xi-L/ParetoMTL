# resnet18 base model for Pareto MTL
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import models

class RegressionTrainResNet(torch.nn.Module):
   
    def __init__(self, model,init_weight):
        super(RegressionTrainResNet, self).__init__()
        
        self.model = model
        self.weights = torch.nn.Parameter(torch.from_numpy(init_weight).float())
        self.ce_loss = CrossEntropyLoss()
    
    def forward(self, x, ts):
        n_tasks = 2
        ys = self.model(x)
        
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.ce_loss(ys[:,i], ts[:,i]) )
        task_loss = torch.stack(task_loss)

        return task_loss

    
class MnistResNet(torch.nn.Module):
    def __init__(self, n_tasks):
        super(MnistResNet, self).__init__()
        self.n_tasks = n_tasks
        self.feature_extractor = models.resnet18(pretrained = False)
        self.feature_extractor.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = torch.nn.Linear(fc_in_features, 100)
        
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(100, 10))
        
    def forward(self, x):
        x = F.relu(self.feature_extractor(x))
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(x))

        return torch.stack(outs, dim=1)


    
    
    
    