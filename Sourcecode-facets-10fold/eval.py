import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import wandb
import matplotlib.pyplot as plt

tb = SummaryWriter()

def test_model(model, hyp_params, tr_loader, te_loader):

    criterion = hyp_params.criterion
    mae = hyp_params.mae
    device= hyp_params.device

    type = hyp_params.type
    

    for epoch in range(1):
        test_loss = 0
        
        test_predictions = []
        test_labels = []
        test_p = []

        
        model.eval()
        with torch.no_grad(): 
          for i,(images, labels, p) in enumerate(te_loader):

              labels = labels.to(device).float()
              
              if type in ["avg_decision", "decision", "attention"]:
                images = [i.to(device).float() for i in images]
                
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images).to(device)
              else:
                images = images.to(device).float()
                outputs = model(images)

              test_predictions.append(outputs)
              test_labels.append(labels)
              test_p.append(p)
    

          test_predictions = torch.cat(test_predictions)
          test_labels = torch.cat(test_labels)
          test_p = np.concatenate(test_p)

          test_loss = criterion(test_predictions, test_labels)
          test_mae_loss = mae(test_predictions, test_labels)
          test_pcc = stats.pearsonr(test_predictions.detach().cpu().flatten(), test_labels.detach().cpu().flatten())[0]

          print(test_p)
          df = pd.DataFrame(list(zip(test_p, test_predictions, test_labels)),
               columns =['ID', 'Pred', 'Label'])
          grouped = df.groupby(['ID'], as_index=False)
          grouped_losses = []
          grouped_mae = []
          grouped_pcc = []
          best = {}
          worst = {}
          pp_mse = []
          pp_mae = []
          pp_pcc = []


          for g in grouped:
            #print(g[0])
            data =  g[1]
            #print(torch.cat(list(data['Pred'])))
            preds = torch.cat(list(data['Pred'])).view(-1,15)
            labels = torch.cat(list(data['Label'])).view(-1,15)
            g_loss = criterion(preds, labels)
            g_mae = mae(preds, labels)
            g_pcc =  stats.pearsonr(preds.detach().cpu().flatten(), labels.detach().cpu().flatten())[0]

            if grouped_losses and g_loss > max(grouped_losses):
              id = g[0]
              worst = {id:g_loss}
            if grouped_losses and g_loss < min(grouped_losses):
              id = g[0]
              best = {id:g_loss}

            
            p_indv_loss = [criterion(preds[:,i].detach().cpu(), labels[:,i].detach().cpu()) for i in range(15)]
            #print(p_indv_losses)
            p_indv_mae = [mae(preds[:,i].detach().cpu(), labels[:,i].detach().cpu()) for i in range(15)]
            ## indv_mae = [mae(test_predictions[:,i].detach().cpu(), test_labels[:,i].detach().cpu()) for i in range(outputs.shape[1])]
            # p_indv_pcc = [stats.pearsonr(preds.unsqueeze(0)[:,i].detach().cpu(), labels.unsqueeze(0)[:,i].detach().cpu())[0]
            #   for i in range(5)]

            pp_mse.append(p_indv_loss)
            pp_mae.append(p_indv_mae)
            #pp_pcc.append(p_indv_pcc)

            grouped_losses.append(g_loss.cpu())
            grouped_mae.append(g_mae.cpu())
            grouped_pcc.append(g_pcc)

        mse_part = np.mean(grouped_losses)
        mae_part = np.mean(grouped_mae)
        pcc_part = np.mean(grouped_pcc)

        print("Traits_mse_part:", np.round_(np.mean(np.vstack(pp_mse), axis=0), decimals=4))
        print("Traits_mae_part:", np.round_(np.mean(np.vstack(pp_mae), axis=0), decimals=4))
        
        print('MSE_part: %.4f | MAE_part: %.4f | PCC_part: %.4f' \
             %(mse_part, mae_part, pcc_part))
        print("Best:", best, "Worst:", worst)

        indv_losses = [criterion(test_predictions[:,i].detach().cpu(), test_labels[:,i].detach().cpu()) for i in range(outputs.shape[1])]
        indv_mae = [mae(test_predictions[:,i].detach().cpu(), test_labels[:,i].detach().cpu()) for i in range(outputs.shape[1])]
        indv_pcc = [stats.pearsonr(test_predictions[:,i].detach().cpu(), test_labels[:,i].detach().cpu())[0]
          for i in range(outputs.shape[1])]

        mets = np.round_(np.vstack([indv_losses, indv_mae, indv_pcc]), decimals=4)
        print("MSE, MAE, PCC by trait:", mets)

        # print(test_predictions, test_labels)
        print('Epoch: %d | Test Loss: %.4f | Test MAE Loss: %.4f | Test PCC: %.4f' \
            %(epoch, test_loss, test_mae_loss, test_pcc))
        plot_all("all_test",test_predictions.detach().cpu(), test_labels.detach().cpu())


def plot_all(split, preds, labels):

    plt.scatter(labels, preds, c='crimson')
    plt.plot(labels, labels)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    
    plt.savefig(split + ".png")
    plt.show()
