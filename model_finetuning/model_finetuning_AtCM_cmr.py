import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd
from scipy.io import loadmat
from fairseq_signals.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel
from omegaconf import OmegaConf

from sklearn.preprocessing import StandardScaler
import yaml

import sys
from pathlib import Path
parent_dir = Path.cwd().parent
sys.path.insert(0, str(parent_dir))

from model_finetuning.ModelClassAtCM_CMR import *


#Define paths

path_to_config = "path/to/config" #Path to configuration file

path_to_tabular_data = "path/to/tabular" #Path to tabluar data. Should be a csv file containing four columns "LA_max", "LA_min", "LAEF" and "LA_LV_ratio" which contain the CMR imaging indices. Further 
#it should contain a column "patient ids" with patient ids and a column "ecg_files" containing the names of the ecg files to be included in train, validation and testing.

path_to_ECGs = "path/to/ecgs" #path to a folder containing all the ECGs 

path_to_outputs = "path/to/outputs" #path to folder where outputs should be saved

path_to_foundation_model = "path/to/fm" #path where the ECG-FM mimic_iv_ecg_physionet_pretrained.pt model is saved. 


#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)


with open(path_to_config, 'r') as file:
    config = yaml.safe_load(file)

#Load Data
print("Loading Data...", flush=True)    
data = pd.read_csv(path_to_tabular_data, low_memory=False)

#Split data into train/val/test and standardize data
IDs = data["patient_ids"].unique()
train_ids, test_ids = train_test_split(
    IDs, 
    test_size=0.2,      # 20% for test
    random_state=42     # For reproducibility
)
train_ids, val_ids  = train_test_split(
    train_ids, 
    test_size=0.25,      # 20% for test
    random_state=42)

train = data[data["patient_ids"].isin(train_ids)]
y_train = train[["LA_max", "LA_min", "LAEF", "LA_LV_ratio"]]

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
print(scaler_y.mean_, flush = True)
print(scaler_y.scale_, flush = True)

val = data[data["patient_ids"].isin(val_ids)]
y_val = val[["LA_max", "LA_min", "LAEF", "LA_LV_ratio"]]
y_val = scaler_y.transform(y_val)

test = data[data["patient_ids"].isin(test_ids)]
y_test = test[["LA_max", "LA_min", "LAEF", "LA_LV_ratio"]]

train_samples = []
print("Creating train set...", flush=True)
for ecg_file in train["ecg_files"]:
    ecg_samples = []
    ecg_file = path_to_ECGs + ecg_file
    ecg = loadmat(ecg_file)["feats"]
    ecg_tensor = torch.from_numpy(ecg).float()
    train_samples.append(ecg_tensor)
 
print("Creating validation set...", flush=True)
val_samples = []
for ecg_file in val["ecg_files"]:
    ecg_samples = []
    ecg_file = path_to_ECGs + ecg_file
    ecg = loadmat(ecg_file)["feats"]
    ecg_tensor = torch.from_numpy(ecg).float()
    val_samples.append(ecg_tensor)


print("Creating test set...", flush=True)
test_samples = []
for ecg_file in test["ecg_files"]:
    ecg_file = path_to_ECGs + ecg_file
    ecg_samples = []
    ecg = loadmat(ecg_file)["feats"]
    ecg_tensor = torch.from_numpy(ecg).float()
    test_samples.append(ecg_tensor) 

X_train = torch.stack(train_samples, dim=0)

X_val = torch.stack(val_samples, dim=0)

X_test = torch.stack(test_samples, dim=0)


train_ids_df = pd.DataFrame({"train_ids": train_ids})
test_ids_df = pd.DataFrame({"test_ids": test_ids})
val_ids_df = pd.DataFrame({"val_ids": val_ids})

train_ids_df.to_csv(path_to_outputs + "train_ids.csv")
val_ids_df.to_csv(path_to_outputs + "val_ids.csv")
test_ids_df.to_csv(path_to_outputs + "test_ids.csv")

train_dataset = ECGDataset(X_train, y_train)
val_dataset = ECGDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)


#Load Foundation model
ckpt_encoder = torch.load(path_to_foundation_model, weights_only = False)
cfg_encoder = ckpt_encoder['cfg']
cfg_encoder = OmegaConf.create(cfg_encoder["model"])
cfg_encoder["saliency"] = False
pretrained_model = Wav2Vec2CMSCModel(cfg_encoder)
pretrained_model.load_state_dict(ckpt_encoder['model'])


# Fine-tune the model
print("Running model finetuning...", flush=True)
finetuned_model, losses = finetune_model(pretrained_model, train_loader, val_loader, device, criterion=nn.MSELoss(), num_out=4, num_epochs=config["training"]["num_epochs_total"], epoch_probe = config["training"]["num_epochs_regression_head"], 
                                         patience=config["training"]["patience"], learning_rate_regression_head = config["training"]["learning_rate_regression_head"], learning_rate_backbone=config["training"]["learning_rate_backbone"],
                                         weight_decay=config["regularization"]["weight_decay"], dropout_rate=config["regularization"]["dropout_rate"])

#Save the loss curves
losses.to_csv(path_to_outputs+"losses.csv")



