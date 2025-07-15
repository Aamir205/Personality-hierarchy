import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from PIL import Image
from glob import glob
import re


class FeatureFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames, p_list):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.p_list = p_list
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False)
    labels = self.label_filenames[idx].astype(float)
    p = self.p_list[idx]

    data = pd.concat([pose_data,gaze_data, au_data], axis = 0).dropna().to_numpy().astype(float)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    
    if idx == self.__len__():  
            raise IndexError  

    return self.transform(scaled_data), labels,p

class DecisionFusionDataset(Dataset):
  def __init__(self, data_filenames, label_filenames, p_list):
    self.data_filenames = data_filenames
    self.label_filenames = label_filenames
    self.p_list = p_list
    self.transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data_filenames)

  def __getitem__(self, idx):

    p = self.p_list[idx]
    gaze_data = pd.read_csv(self.data_filenames[idx][0], header=None, 
                 index_col=False, skiprows=1)
    au_data = pd.read_csv(self.data_filenames[idx][1], header=None, 
                 index_col=False, skiprows=1)
    pose_data = pd.read_csv(self.data_filenames[idx][2], header=None, 
                 index_col=False, skiprows=1)
    labels = self.label_filenames[idx].astype(float)

    data = [pose_data,gaze_data, au_data]
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = [self.transform(scaler.fit_transform(x.to_numpy().astype(float))) for x in data]

    if idx == self.__len__():  
            raise IndexError  

    return data, labels, p


class DyadicFeatureFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = np.concatenate([pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]])
      p2 = np.concatenate([pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]])

      labels = self.label_filenames[idx].astype(float)

      data = np.concatenate([p1,p2]).astype(float)
      scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
      data = scaler.fit_transform(data)
      
      if idx == self.__len__():  
              raise IndexError  
  
      return self.transform(data), labels, p

class DyadicDecisionFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]]
      p2 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]]

      labels = self.label_filenames[idx].astype(float)

      au1,gaze1,pose1,aud1 =  p1
      au2,gaze2,pose2,aud2 =  p2
      
      data = [au1,gaze1,pose1,aud1,au2,gaze2,pose2,aud2]
      scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
      data = [self.transform(scaler.fit_transform(x).astype(float)) for x in data]
      
      if idx == self.__len__():  
              raise IndexError  
      #print(d.shape,l.shape)
      return data, labels, p

class IdvDyadicDecisionFusionDataset(Dataset):
    def __init__(self, data_filenames, label_filenames, p_list):
      self.data_filenames = data_filenames
      self.label_filenames = label_filenames
      self.p_list = p_list
      self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
      return len(self.data_filenames)

    def __getitem__(self, idx):
      p = self.p_list[idx]
      p1 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][0]]
      p2 = [pd.read_csv(i, header=None, 
                 index_col=False) for i in self.data_filenames[idx][1]]

      p1_vis = np.concatenate(p1[1:])
      p1_aud = p1[0].to_numpy()
      p2_vis = np.concatenate(p2[1:])
      p2_aud = p2[0].to_numpy()

      labels = np.concatenate(self.label_filenames[idx]).astype(float)

      data = [p1_vis,p1_aud,p2_vis,p2_aud]
      data = [self.transform(x.astype(float)) for x in data]
      
      if idx == self.__len__():  
              raise IndexError  

      return data, labels, p

# Add this function to handle fold-specific metadata
def get_fold_labels(part, dir, fold):
    if part == "train":
        prefix = f"partition/parts_train.{fold}.csv"
        sessions = f"partition/sessions_train.{fold}.csv"
        labels_file = f"partition/Labels_train_{fold}.csv"
    elif part == "val":
        prefix = f"partition/parts_val.{fold}.csv"
        sessions = f"partition/sessions_val.{fold}.csv"
        labels_file = f"partition/Labels_val_{fold}.csv"
    else:  # test
        prefix = f"partition/parts_test.{fold}.csv"
        sessions = f"partition/sessions_test.{fold}.csv"
        labels_file = f"partition/Labels_test_{fold}.csv"
    
    df = pd.read_csv(prefix)
    df2 = pd.read_csv(sessions)
    df3 = pd.read_csv(labels_file)
    label_idx = ["BFI_Sociability", "BFI_Assertiveness",  "BFI_EnergyLevel", "BFI_Compassion","BFI_Respectfulness", "BFI_Trust", "BFI_Organization",  "BFI_Productiveness", "BFI_Responsibility", "BFI_Anxiety","BFI_Depression",  "BFI_EmotionalVolatility",  "BFI_IntellectualCuriosity",  "BFI_AestheticSensitivity", "BFI_CreativeImagination"]
    #label_idx = ["BFI_Extraversion", "BFI_Agreeableness", "BFI_Conscientiousness", "BFI_NegativeEmotionality", "BFI_OpenMindedness"]

    participants = df2.loc[df2["ID"] == dir]
    p1 = participants["PART.1"].values[0]
    p2 = participants["PART.2"].values[0]
    l1 = np.asarray(df3.loc[df3["ID"].isin([p1,p2])][label_idx].values)
    return p1, p2, l1

## Fetch labels for each session's participants
def get_labels(part, dir):
  if part == "train":
    df = pd.read_csv("metadata_train/parts_train.csv")
    df2 = pd.read_csv("metadata_train/sessions_train.csv")
    df3 = pd.read_csv("metadata_train/Labels_train.csv") 
    
  elif part == "val":
    df = pd.read_csv("metadata_val/parts_val_unmasked.csv")
    df2 = pd.read_csv("metadata_val/sessions_val.csv")
    df3 = pd.read_csv("metadata_val/Labels_val.csv") 
  else:
    df = pd.read_csv("metadata_test/parts_test_unmasked.csv")
    df2 = pd.read_csv("metadata_test/sessions_test.csv")
    df3 = pd.read_csv("metadata_test/Labels_test.csv") 
        
  #label_idx = ["BFI_Extraversion", "BFI_Agreeableness",	"BFI_Conscientiousness", "BFI_NegativeEmotionality", "BFI_OpenMindedness"]
  label_idx = ["BFI_Sociability", "BFI_Assertiveness",  "BFI_EnergyLevel", "BFI_Compassion","BFI_Respectfulness", "BFI_Trust", "BFI_Organization",  "BFI_Productiveness", "BFI_Responsibility", "BFI_Anxiety","BFI_Depression",  "BFI_EmotionalVolatility",  "BFI_IntellectualCuriosity",  "BFI_AestheticSensitivity", "BFI_CreativeImagination"]
  #label_idx = ["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"]
  participants = df2.loc[df2["ID"] == dir]
  p1 = participants["PART.1"].values[0]
  p2 = participants["PART.2"].values[0]
  l1 = np.asarray(df3.loc[df3["ID"].isin([p1,p2])][label_idx].values) # are they predicting it for both???
  return p1,p2, l1

def fused_data(fusion_strategy, batch_size, section, dyad=False, split="val", fold=-1):

    def get_fold_sessions(set_type):
        if fold == -1:  # Original split
            sessions_file = f"metadata_{set_type}/sessions_{set_type}.csv"
        else:  # 10-fold CV
            sessions_file = f"partition/sessions_{set_type}.{fold}.csv"
        return pd.read_csv(sessions_file)['ID'].tolist()

    train_sessions = get_fold_sessions("train")
    val_sessions = get_fold_sessions("val")
    test_sessions = get_fold_sessions("test")

    '''def filter_dirs(dir_list, valid_sessions):
        return [d for d in dir_list if int(os.path.basename(d)) in valid_sessions]'''
    
    def filter_dirs(dir_list, valid_sessions):
      """
      Filters directory paths based on a list of valid session IDs.
      Handles session IDs with/without leading zeros by normalizing to integers.
      """
      valid_set = set(valid_sessions)
      found_sessions = set()
      filtered_dirs = []
  
      for d in dir_list:
          # Extract basename from path
          basename = re.split(r"[\\/]", d)[-1]
          
          # Skip non-digit directory names
          if not basename.isdigit():
              continue
              
          try:
              # Convert to integer to normalize session ID format
              session_id = int(basename)
          except ValueError:
              continue
              
          # Check if normalized session ID is in valid set
          if session_id in valid_set:
              filtered_dirs.append(d)
              found_sessions.add(session_id)
  
      # Report missing sessions
      missing = sorted(valid_set - found_sessions)
      if missing:
          print(f"Warning: The following session IDs were not found: {missing}")
  
      return filtered_dirs
    
    
    # Define base paths for different data types
    base_paths = {
        "spectral": {
            "train": f"{section}_spectral",
            "test": f"{section}_spectral_test",
            "val": f"{section}_spectral_val"
        },
        "audio": {
            "train": f"{section}_audio_spec_train",
            "test": f"{section}_audio_spec_test",
            "val": f"{section}_audio_spec_val"
        }
    }

    # For 10-fold CV, we use the same base directory for all splits
    if fold != -1:
        base_paths["spectral"]["test"] = base_paths["spectral"]["train"]
        base_paths["spectral"]["val"] = base_paths["spectral"]["train"]
        base_paths["audio"]["test"] = base_paths["audio"]["train"]
        base_paths["audio"]["val"] = base_paths["audio"]["train"]

    # Get directories for training data
    pose_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["train"], "pose/*")), train_sessions)
    gaze_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["train"], "gaze/*")), train_sessions)
    au_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["train"], "au/*")), train_sessions)
    aud_dir_list = filter_dirs(glob(os.path.join(base_paths["audio"]["train"], "*")), train_sessions)
    train_dir_list = np.column_stack([pose_dir_list, gaze_dir_list, au_dir_list, aud_dir_list])

    # Get directories for test data
    ts_pose_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["test"], "pose/*")), test_sessions)
    ts_gaze_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["test"], "gaze/*")), test_sessions)
    ts_au_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["test"], "au/*")), test_sessions)
    ts_aud_dir_list = filter_dirs(glob(os.path.join(base_paths["audio"]["test"], "*")), test_sessions)
    test_dir_list = np.column_stack([ts_pose_dir_list, ts_gaze_dir_list, ts_au_dir_list, ts_aud_dir_list])

    # Get directories for validation data
    val_pose_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["val"], "pose/*")), val_sessions)
    val_gaze_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["val"], "gaze/*")), val_sessions)
    val_au_dir_list = filter_dirs(glob(os.path.join(base_paths["spectral"]["val"], "au/*")), val_sessions)
    val_aud_dir_list = filter_dirs(glob(os.path.join(base_paths["audio"]["val"], "*")), val_sessions)
    val_dir_list = np.column_stack([val_pose_dir_list, val_gaze_dir_list, val_au_dir_list, val_aud_dir_list])

    # Rest of your code remains the same...
    train_dirs = train_dir_list
    test_dirs = test_dir_list
    val_dirs = val_dir_list  


    train_file_list = np.zeros((1,4))
    train_labels = np.zeros((1,15))
    
    test_file_list = np.zeros((1,4))
    test_labels =  np.zeros((1,15))

    val_file_list = np.zeros((1,4))
    val_labels =  np.zeros((1,15))

    train_p = []
    test_p = []
    val_p = []



    for i in train_dirs:
        dir = int(os.path.split(i[0])[1])
        #dir = os.path.split(i[0])[1]
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        aud = np.sort([x for x in glob(i[3]+"/*.csv")])
        #print("TRAIN : This is dir\n", dir)

        if fold == -1:  # Original split
            p1,p2,labels = get_labels("train", dir)
        else:
            p1,p2,labels = get_fold_labels("train", dir, fold)

        ls = np.column_stack([pose,gaze,au,aud])
        #p1,p2,labels = get_labels("train",dir)
        #print("TRAIN : This is ls p1 p2 and labels- \n",ls, p1,p2,labels)

        train_file_list = np.concatenate([train_file_list, ls])
        train_labels = np.vstack([train_labels, labels[0], labels[1]])
        train_p += [p1,p2]
        
    for i in test_dirs:
        #dir = int(int(os.path.split(i[0])[1]))
        #dir = os.path.split(i[0])[1]
        dir = int(os.path.split(i[0])[1])
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        aud = np.sort([x for x in glob(i[3]+"/*.csv")])
        #print("TEST : This is dir-", dir)
        #print("TEST : This is ls p1 p2 and labels",ls, p1,p2,labels)

        if fold == -1:  # Original split
            p1,p2,labels = get_labels("test", dir)
        else:
            p1,p2,labels = get_fold_labels("test", dir, fold)
        
        ls = np.column_stack([pose,gaze,au,aud])
        #p1,p2, labels = get_labels(rv, dir)
  
        test_file_list = np.concatenate([test_file_list, ls])
        test_labels = np.vstack([test_labels, labels[0], labels[1]])
        test_p += [p1]
        test_p += [p2]

    for i in val_dirs:
        #dir = int(int(os.path.split(i[0])[1]))
        dir = int(os.path.split(i[0])[1])
        #dir = os.path.split(i[0])[1]
        gaze = np.sort([x for x in glob(i[1]+"/*.csv")])
        pose = np.sort([x for x in glob(i[0]+"/*.csv")])
        au = np.sort([x for x in glob(i[2]+"/*.csv")])
        aud = np.sort([x for x in glob(i[3]+"/*.csv")])
        #print("TEST : This is dir-", dir)
        #print("TEST : This is ls p1 p2 and labels",ls, p1,p2,labels)

        if fold == -1:  # Original split
            p1, p2, labels = get_labels("val", dir)
        else:
            p1,p2,labels = get_fold_labels("val", dir, fold)
        
        ls = np.column_stack([pose,gaze,au,aud])
        #p1,p2, labels = get_labels(val, dir)
  
        val_file_list = np.concatenate([val_file_list, ls])
        val_labels = np.vstack([val_labels, labels[0], labels[1]])
        val_p += [p1]
        val_p += [p2]

    train_file_list = np.delete(train_file_list, 0, axis=0)
    test_file_list = np.delete(test_file_list, 0, axis=0)
    val_file_list = np.delete(val_file_list, 0, axis=0)
    train_labels = np.delete(train_labels, 0, axis=0)
    test_labels = np.delete(test_labels, 0,  axis=0)
    val_labels = np.delete(val_labels, 0,  axis=0)



    dims = [pd.read_csv(glob(i+"/*.csv")[0], header=None, 
                 index_col=False).shape for i in train_dirs[0]]

    train_file_list.sort()
    test_file_list.sort()
    val_file_list.sort()
    train_labels.sort()
    test_labels.sort()
    val_labels.sort()

    if dyad:
      print("1")
      print(fusion_strategy)
      print("Dyadic")
      #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
      #print("Length:", len(val_file_list))
      #print("len(val_file_list)//2:", len(val_file_list)//2)
      ##print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBbBBBBBBBb")
      #print(f"Fold {fold} - val_file_list shape: {val_file_list.shape}")
      #print(f"Fold {fold} - len(val_file_list): {len(val_file_list)}")
      #print(f"Fold {fold} - val_file_list content sample: {val_file_list[:2]}")
      #print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCcc")
      #print(f"Number of validation sessions: {len(val_dirs)}")
      #print(f"Number of validation files: {len(val_file_list)}")
      #print(f"Validation labels shape: {val_labels.shape}")
      print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
      print("Validation session IDs:", val_sessions)
      print("test session IDs:", test_sessions)
      print("Train session IDs:", train_sessions)
      print("Filtered pose dirs in val:", val_pose_dir_list)
      print("Filtered pose dirs in test:", ts_pose_dir_list)
      print("Filtered pose dirs in train:", pose_dir_list)
      print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
      print("Filtered pose dirs in val:", val_gaze_dir_list)
      print("Filtered pose dirs in test:", ts_gaze_dir_list)
      print("Filtered pose dirs in train:", gaze_dir_list)
      print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
      print("Filtered pose dirs in val:", val_au_dir_list)
      print("Filtered pose dirs in test:", ts_au_dir_list)
      print("Filtered pose dirs in train:", au_dir_list)
      print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
      print("Filtered pose dirs in val:", val_aud_dir_list)
      print("Filtered pose dirs in test:", ts_aud_dir_list)
      print("Filtered pose dirs in train:", aud_dir_list)
      

      train_file_list = np.repeat(np.split(train_file_list, len(train_file_list)//2), 2, axis=0)
      test_file_list = np.repeat(np.split(test_file_list, len(test_file_list)//2), 2, axis=0)
      val_file_list = np.repeat(np.split(val_file_list, len(val_file_list)//2), 2, axis=0)

      
      if fusion_strategy in ["avg_decision","decision","attention"]:
        print("2: DyadicDecisionFusionDataset")
        dims =  [(4,80),(18,80),(72,80),(14,80),(4,80),(72,80),(18,80),(14,80)]
        train_dataset = DyadicDecisionFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
        test_dataset = DyadicDecisionFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)
        val_dataset = DyadicDecisionFusionDataset(data_filenames = val_file_list, label_filenames= val_labels, p_list = val_p)

      else: 
          print("3 : DyadicFeatureFusionDataset")
          train_dataset = DyadicFeatureFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = DyadicFeatureFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)
          val_dataset = DyadicFeatureFusionDataset(data_filenames = val_file_list, label_filenames= val_labels, p_list = val_p)

    else:
      if fusion_strategy in ["avg_decision","decision","attention"]:

          train_dataset = DecisionFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = DecisionFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)
          val_dataset = DecisionFusionDataset(data_filenames = val_file_list, label_filenames= val_labels, p_list = val_p)
          print("4 : DecisionFusionDataset")

      else: 
          train_dataset = FeatureFusionDataset(data_filenames = train_file_list,label_filenames= train_labels, p_list = train_p)
          test_dataset = FeatureFusionDataset(data_filenames = test_file_list, label_filenames= test_labels, p_list = test_p)
          val_dataset = FeatureFusionDataset(data_filenames = val_file_list, label_filenames= val_labels, p_list = val_p)
          print("5 : FeatureFusionDataset")


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)
    
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)


    return trainloader, testloader,valloader, dims

