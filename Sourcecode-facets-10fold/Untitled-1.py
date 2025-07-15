import torch
import argparse
#from eval import test_model
from load_data import *
from models.multmodel import DyadMULTModel, EnsembleDyadMULTModel, MULTModel
from models.resnet50 import ResNet50
from models.ensemble import AvgEnsemble, DyadAvgEnsemble, DyadEnsemble, Ensemble
from train import train_model
from eval import test_model
import torch.nn as  nn
import random
import numpy as np
import pandas as pd
from train import sweep_config  
import wandb
from functools import partial

df3 = pd.read_csv("partition/Labels_val_7.csv")
print(df3[["BFI_Extraversion", "BFI_Agreeableness", "BFI_Conscientiousness", 
            "BFI_NegativeEmotionality", "BFI_OpenMindedness"]].head())
