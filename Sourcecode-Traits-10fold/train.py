from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import wandb
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

tb = SummaryWriter()

val_loss = 0.0
eval_loss = 0.0
val_mae_loss = 0.0
eval_mae_loss = 0.0
eval_pcc = 0.0
val_pcc = 0.0

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0001, 0.00001, 0.01]
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 32 
        },
        'type': {
            'values': ["feature"]
        }
        # Other parameters...
    }
}


def train_model(model, config, tr_loader, te_loader,vl_loader,hyp_params,section,modality,fold):
    wandb.login(key = "90f1cfe3d92250664317f8f939985f8cdb79e5f1")
    wandb.init(project="udiva-cyber_2024", entity="amiransari-private",config=config,name=f"{section}_{modality}_fold{fold}",tags=[f"fold{fold}", section, modality])


    # Track best model and metrics
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    best_train_results = None
    best_val_results = None
    best_test_results = None
    patience = 3
    no_improve = 0

    criterion = hyp_params.criterion
    mae = hyp_params.mae
    device= hyp_params.device
    learning_rate = config.get('learning_rate', hyp_params.lr)  # Use default from hyp_params if not in config
    batch_size = config.get('batch_size', hyp_params.batch_size)  # Similarly, for batch_size
    optimizer_type = config.get('optimizer', hyp_params.optimizer)  # Ensure 'optimizer' is also in sweep config
    model_type = config.get('type', hyp_params.type)
    weight_decay = config.get('wd', hyp_params.wd)
    num_epochs = config.get('epochs', hyp_params.num_epochs)

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=hyp_params.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.05)
    type = config.type
    #type = hyp_params.type

    
    for epoch in range(0, hyp_params.num_epochs):
        train_mae_loss = 0
        test_loss = 0
        
        model.train()
        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []


        test_ids = []
        val_ids = []
        train_ids = []

        val_predictions = []
        val_labels = []



        for i,(images, labels,p) in enumerate(tr_loader):
          print(i, len(tr_loader))
          labels = labels.to(device).float()
          
          optimizer.zero_grad()

          if type in ["avg_decision", "decision", "attention", "idv_decision"]:
            images = [i.to(device).float() for i in images]

            if type == "attention":
                yhat, hidden = model(*images)
                
            else:
                yhat = model(*images)
                #yhat = yhat[0]
                yhat = yhat.to(device)

          else:
            images = images.to(device).float() 
            yhat = model(images)

            
          train_predictions.append(yhat)
          train_labels.append(labels)
          train_ids.extend(p.cpu().numpy().tolist())
          loss = criterion(yhat, labels)
          loss.backward()
          optimizer.step()
          

 
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_loss = criterion(train_predictions, train_labels)
        train_mae_loss = mae(train_predictions, train_labels)
        train_pcc = stats.pearsonr(train_predictions.detach().cpu().flatten(), train_labels.detach().cpu().flatten())[0]
        print('Epoch: %d | Loss: %.4f | MAE: %.2f | PCC: %.2f'\
            %(epoch, train_loss, train_mae_loss, train_pcc))
        scheduler.step()





        #VALIDATION LOOP STARTS HERE EHHEHEHEHEHEHHE

        model.eval()
        val_predictions = []
        val_labels = []
        val_ids = []
    
        with torch.no_grad(): 

          for i, (images, labels, p) in enumerate(vl_loader):

              
              labels = labels.to(device).float()

              if type in ["avg_decision", "decision", "attention", "idv_decision"]:
                images = [i.to(device).float() for i in images]
                
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images)
                    outputs = outputs.to(device)

              else:

                images = images.to(device).float()
                outputs = model(images)

              val_predictions.append(outputs)
              val_labels.append(labels)
              val_ids.extend(p.cpu().numpy().tolist())

        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)
        val_loss = criterion(val_predictions, val_labels)
        val_mae_loss = mae(val_predictions, val_labels)
        val_pcc = stats.pearsonr(val_predictions.detach().cpu().flatten(), val_labels.detach().cpu().flatten())[0]
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBbFirst batch of IDs:", p[:50])
          
        print(val_predictions, val_labels)
        print('Epoch: %d | VAL Loss: %.4f | VAL MAE Loss: %.2f | VAL PCC: %.2f' \
              %(epoch, val_loss, val_mae_loss, val_pcc))
          
        # Log metrics to WandB (once per epoch)
        wandb.log({
            "Train_loss": train_loss,
            "Val_loss": val_loss,
            "Train_mae": train_mae_loss,
            "Val_mae": val_mae_loss,
            "Train_pcc": train_pcc,
            "Val_pcc": val_pcc,
            "Epoch": epoch
        })

        # Check if this is the best validation performance
        if val_loss < best_val_loss:
                print(f"New best validation loss: {val_loss:.4f} (previous best: {best_val_loss:.4f})")
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                no_improve = 0
                
                # Save training results for this best epoch
                best_train_results = {
                    'predictions': train_predictions,
                    'labels': train_labels,
                    'ids': train_ids
                }
                
                best_val_results = {
                    'predictions': val_predictions,
                    'labels': val_labels,
                    'ids': val_ids,
                    'loss': val_loss,
                    'mae': val_mae_loss,
                    'pcc': val_pcc
                }
                
                # === TESTING at best epoch ===
                model.eval()
                test_predictions = []
                test_labels = []
                test_ids = []
                
                with torch.no_grad():
                    for i, (images, labels, p) in enumerate(te_loader):
                        labels = labels.to(device).float()

                        if type in ["avg_decision", "decision", "attention", "idv_decision"]:
                            images = [i.to(device).float() for i in images]
                
                            if type == "attention":
                                outputs, hidden = model(*images)
                            else:
                                outputs = model(*images)
                                outputs = outputs.to(device)

                        else:
                            images = images.to(device).float()
                            outputs = model(images)

                        test_predictions.append(outputs)
                        test_labels.append(labels)
                        test_ids.extend(p.cpu().numpy().tolist())
                    
                test_predictions = torch.cat(test_predictions)
                test_labels = torch.cat(test_labels)
                    
                best_test_results = {
                        'predictions': test_predictions,
                        'labels': test_labels,
                        'ids': test_ids
                }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs. Early stopping at epoch {epoch}.")
                break

# === AFTER TRAINING LOOP ===
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        run_id = wandb.run.id
        model_path = f"saved_models/{section}_{modality}_fold{fold}_best_epoch_{best_epoch}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model from epoch {best_epoch} to {model_path}")
    
    # Save predictions for all sets at best epoch
    if best_train_results and best_val_results and best_test_results:
        run_id = wandb.run.id
        f_name = f"{section}_{modality}_{run_id}_fold{fold}_best_epoch_{best_epoch}"
        
        # Save training results
        save_results(
            best_train_results['predictions'],
            best_train_results['labels'],
            best_train_results['ids'],
            f_name,
            "train"
        )
        
        # Save validation results
        save_results(
            best_val_results['predictions'],
            best_val_results['labels'],
            best_val_results['ids'],
            f_name,
            "val"
        )
        
        # Save test results
        save_results(
            best_test_results['predictions'],
            best_test_results['labels'],
            best_test_results['ids'],
            f_name,
            "test"
        )
        
        # Calculate and log test metrics at best epoch
        test_loss = criterion(
            best_test_results['predictions'], 
            best_test_results['labels']
        )
        test_mae = mae(
            best_test_results['predictions'], 
            best_test_results['labels']
        )
        test_pcc = stats.pearsonr(
            best_test_results['predictions'].detach().cpu().flatten(), 
            best_test_results['labels'].detach().cpu().flatten()
        )[0]
        
        print('========== BEST EPOCH TEST ==========')
        print(f'Test Loss: {test_loss:.4f} | MAE: {test_mae:.2f} | PCC: {test_pcc:.2f}')
        
        wandb.log({
            "Best_Test_loss": test_loss,
            "Best_Test_mae": test_mae,
            "Best_Test_pcc": test_pcc,
            "Best_Epoch": best_epoch
        })
    
    # Final WandB logging
    wandb.log({
        "Best_Val_loss": best_val_loss,
        "Best_Epoch": best_epoch
    })
    
    # Return best metrics if needed
    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'test_loss': test_loss.item() if best_test_results else None,
        'test_mae': test_mae.item() if best_test_results else None,
        'test_pcc': test_pcc
    }


def save_results(predictions, labels, ids, f_name_base, set_name):
    """Helper function to save results to CSV"""
    preds_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Flatten if single output, else format per trait
    if preds_np.ndim == 1 or preds_np.shape[1] == 1:
        df = pd.DataFrame({
            'ID': ids,
            'Prediction': preds_np.flatten(),
            'Label': labels_np.flatten()
        })
    else:
        df = pd.DataFrame({
            'ID': np.repeat(ids, preds_np.shape[1]),
            'Trait_Index': np.tile(np.arange(preds_np.shape[1]), len(ids)),
            'Prediction': preds_np.flatten(),
            'Label': labels_np.flatten()
        })
    
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    df.to_csv(f"./results/{f_name_base}_{set_name}_results.csv", index=False)
    print(f"Saved {set_name} results to ./results/{f_name_base}_{set_name}_results.csv")


def plot_metrics(split, preds, labels):
   for i in range(preds.shape[1]):
      plt.scatter(labels[:,i], preds[:,i], c='crimson')
      plt.plot(labels[:,i], labels[:,i])
      plt.xlabel('True Values', fontsize=15)
      plt.ylabel('Predictions', fontsize=15)
      
      plt.savefig(split + "_" + str(i) + ".png")
      plt.show()
      time.sleep(2)
      plt.close()


def plot_all(split, preds, labels):
    plt.scatter(labels, preds, c='crimson')
    plt.plot(labels, labels)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    
    plt.savefig(split + ".png")
    plt.show()
    time.sleep(2)
    plt.close()