import torch
import torch.nn as  nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(15, 5)
        
    def forward(self, x1, x2, x3):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x3 = self.modelC(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(F.relu(x))
        return x

class AvgEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(AvgEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        #self.classifier = nn.Linear(15, 5)
        
    def forward(self, x1, x2, x3):
        x1 = self.modelA(x1).detach().cpu().numpy()
        x2 = self.modelB(x2).detach().cpu().numpy()
        x3 = self.modelC(x3).detach().cpu().numpy()
        #x = torch.cat((x1, x2, x3), dim=1)

        avg = [np.mean([x1[:,i], x2[:,i], x3[:,i]], axis=0) for i in range(x1.shape[1])]
        x = torch.tensor(np.column_stack(avg), requires_grad=True).float()

        return x

class DyadEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC,modelD,modelE,modelF,modelG,modelH):
        super(DyadEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        self.modelF = modelF
        self.modelG = modelG
        self.modelH = modelH
        self.classifier = nn.Linear(40, 5)
        
    def forward(self, x1, x2,x3,x4,x5,x6,x7,x8):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x3 = self.modelC(x3)
        x4 = self.modelD(x4)
        x5 = self.modelE(x5)
        x6 = self.modelF(x6)
        x7 = self.modelG(x7)
        x8 = self.modelH(x8)

        x = torch.cat((x1, x2, x3,x4,x5,x6,x7,x8), dim=1)
        x = self.classifier(F.relu(x))
        return x

class DyadAvgEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC,modelD,modelE,modelF,modelG,modelH):
        super(DyadAvgEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE
        self.modelF = modelF
        self.modelG = modelG
        self.modelH = modelH
        
    def forward(self, x1, x2,x3,x4,x5,x6,x7,x8):
        x1 = self.modelA(x1).detach().cpu().numpy()
        x2 = self.modelB(x2).detach().cpu().numpy()
        x3 = self.modelC(x3).detach().cpu().numpy()
        x4 = self.modelD(x4).detach().cpu().numpy()
        x5 = self.modelE(x5).detach().cpu().numpy()
        x6 = self.modelF(x6).detach().cpu().numpy()
        x7 = self.modelG(x7).detach().cpu().numpy()
        x8 = self.modelH(x8).detach().cpu().numpy()

        avg = [np.mean([x1[:,i], x2[:,i], x3[:,i], x4[:,i], x5[:,i], x6[:,i], x7[:,i], x8[:,i]], axis=0) for i in range(x1.shape[1])]
        x = torch.tensor(np.column_stack(avg), requires_grad=True).float()

        return x
