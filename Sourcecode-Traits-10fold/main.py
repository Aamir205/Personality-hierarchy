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
from train import sweep_config  
import wandb
from functools import partial

sweep_id = wandb.sweep(sweep_config, project="udiva-cyber_2024", entity="amiransari-private")


parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, default='feature',
                    help='name of fusion strategy. One of: ["avg_decision", "decision", "feature", "attention", "idv_decision"]')
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--section', type=str, default="animals",
                    help= 'part of the dataset')
parser.add_argument('--dyad', type=bool, default=True,
                    help= 'dayd or individual')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--num_epochs', type=int, default=7,
                    help='number of epochs (default: 40)')
parser.add_argument('--optimizer', type=str, default="Adam",
                    help= 'part of the dataset')
parser.add_argument('--val', type=str, default="train",
                    help= 'train or test')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--audonly', action='store_true',
                    help='use the crossmodal fusion into aud (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--attn_dropout_aud', type=float, default=0.0,
                    help='attention dropout (for aud)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')
                    

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

torch.manual_seed(1111)
random.seed(1)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args = parser.parse_args()

valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

modality = args.type
BATCH_SIZE = args.batch_size
section = args.section

hyp_params = args
hyp_params.lr = args.lr
hyp_params.num_epochs = args.num_epochs
hyp_params.criterion = nn.MSELoss()
hyp_params.mae = nn.L1Loss()
hyp_params.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
#print("Running on" + section)
if args.val == "train":
    folds = range(10)  # 0 to 9
else:
    folds = [-1]  # Original split

for fold in folds:
    print(f"\n{'='*50}")
    print(f"Starting fold {fold}")
    print(f"{'='*50}")
    if modality in ["avg_decision", "decision", "feature","attention", "idv_decision"]:
        
        trainloader, testloader,valloader, dims = fused_data(modality, BATCH_SIZE, section, args.dyad, args.val, fold)
    
        if modality in ["decision", "avg_decision"]:
            if args.dyad:
                au1=g1=p1=aud1=au2=g2=p2=aud2=ResNet50(5, channels=1)
    
                model = DyadEnsemble(au1,g1,p1,aud1,au2,g2,p2,aud2).to(device)
                if modality == "avg_decision":
                    model = DyadAvgEnsemble(au1,g1,p1,aud1,au2,g2,p2,aud2).to(device)  
            else:
                gaze_model = ResNet50(5, channels=1)
                pose_model = ResNet50(5, channels=1)
                au_model = ResNet50(5, channels=1)
    
                model = Ensemble(gaze_model, pose_model, au_model).to(device)
                if modality == "avg_decision":
                    model = AvgEnsemble(gaze_model, pose_model, au_model).to(device) 
        elif modality == "attention":
            if args.dyad:
                print(dims)
                hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v, hyp_params.orig_d_aud = [x for (x,y) in dims][:4]
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len, hyp_params.aud_len = [y for (x,y) in dims][:4]
                hyp_params.layers = args.nlevels
                model1 = DyadMULTModel(hyp_params).to(device)
                model2 =  DyadMULTModel(hyp_params).to(device)
                hyp_params.orig_d_a, hyp_params.orig_d_v = [360,360]
                hyp_params.a_len, hyp_params.v_len = [args.batch_size,args.batch_size]
    
                model = EnsembleDyadMULTModel(hyp_params, model1, model2).to(device)
            else:
                dims = [(14,80),(18,80),(72,80)]
                hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = [x for (x,y) in dims]
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = [y for (x,y) in dims]
                hyp_params.layers = args.nlevels
                model = MULTModel(hyp_params).to(device)
        ## Feature-level fusion
        else:
            model = ResNet50(5, channels=1).to(device)
    else:
        raise ValueError('Please try another fusion method. Valid options are ["avg_decision", "decision", "feature","attention", "idv_decision"]')
    
    rv = "test"
    ## Specify your own path here
    model_path = "saved_models/"+ section + modality +"_dyadic_+audio_1person" + rv +"_resnet50_fold{fold}"
    
    if args.val == "train":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        def sweep_train():
        
            wandb.init()
            train_model(model, wandb.config, trainloader, testloader,valloader,hyp_params,section,modality,fold)
    
        wandb.agent(sweep_id, function=sweep_train, count=10)
        #train_model(model, hyp_params, trainloader, testloader)
        print("ELO")
        torch.save(model.state_dict(), model_path)
    
    elif args.val == "test":
        model.load_state_dict(torch.load(model_path))
        # Pass fold to test_model if needed
        test_model(model, hyp_params, trainloader, testloader, fold)

