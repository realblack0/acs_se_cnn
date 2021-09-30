from acs_se_cnn.model import SEBlock, ACSLayer
from acs_se_cnn.main_tools import make_results_directory, train_acs

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.functional import F

import pickle
import argparse


###################
## Configuration ##
###################
# SYSTEM
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--device', default="cuda")
parser.add_argument('--subject', type=int, required=True)
args = parser.parse_args()

device = torch.device(args.device)
name = args.name
results_dir = make_results_directory(name, copy_file=__file__, copy_dir="acs_se_cnn")

# LEARNING STRATEGY
batch_size = 20
epochs     = 500
criterion    = nn.BCEWithLogitsLoss()
Optimizer    = torch.optim.RMSprop
lr              = 0.001
# HYPER PARAMETER
sparse_lambda = 1 # ?

fit_data = "2a"
data_path = "cwt_data/2a" if fit_data=="2a" else "cwt_data/2b"

###################
#### Modeling #####
###################

class Model(nn.Module):
    def __init__(
        self, 
        # MODEL HYPER PARAMETER ,
        n_channels = 22 if fit_data=="2a" else 3,
        n_kerenls  = 64,
        r          = 2
    ):
        super().__init__()

        # ATUO CHANNEL SELECTION
        self.acs_layer   = ACSLayer(c=n_channels, r=r)
        
        # FEATURE EXTRACTION
        self.conv_layer1 = nn.Conv2d(n_channels, n_kerenls, kernel_size=(4,4), stride=(2, 2), padding=1)
        self.se_block1   = SEBlock(c=n_kerenls, r=r)
        
        self.conv_layer2 = nn.Conv2d(n_kerenls, n_kerenls, kernel_size=(4,4), stride=(4, 4))
        self.se_block2   = SEBlock(c=n_kerenls, r=r)
        
        self.conv_layer3 = nn.Conv2d(n_kerenls, n_kerenls, kernel_size=(4,4), stride=(4, 4))
        self.se_block3   = SEBlock(c=n_kerenls, r=r)
        
        # OUTPUT
        self.fc1 = nn.Linear(4, 1)
        self.fc2 = nn.Linear(64, 1)
#         self.sigmoid = F.sigmoid()
        
    def forward(self, inputs, return_s_acs=False): 
        """ 
        Args
        ----
            inputs (batch, channel, height, width) 
        """
        # ATUO CHANNEL SELECTION
        x, s_acs = self.acs_layer(inputs)
#         B, _, _, _ = inputs.shape
#         s_acs = inputs.new_zeros(B,22,1,1)
        
        # FEATURE EXTRACTION
        x = self.conv_layer1(x)
        x = F.elu(x)
        x = self.se_block1(x)
        
        x = self.conv_layer2(x)
        x = F.elu(x)
        x = self.se_block2(x)
        
        x = self.conv_layer3(x)
        x = F.elu(x)
        x = self.se_block3(x)        
        
        # OUTPUT
        B, _, _, _ = x.shape
        x = x.reshape(B, 64, 4)
        x = self.fc1(x) # (B, 64, 1)
        x = x.squeeze() # (B, 64)
        x = F.elu(x)
        out = self.fc2(x)
        
        if return_s_acs:
            return out, s_acs
        else:
            return out
        

###################
#### Load Data ####
###################

file_path = f"{data_path}/{'A' if fit_data=='2a' else 'B'}0{args.subject}_64x64_scipy2_cv10.pkl"
print("Load data from:", file_path)

with open(file_path, "rb") as f:
    data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    y = y.reshape(-1,1)
    y = y.astype(np.float64)
    folds = data["folds"]
    
class CWTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)


####################
#    Training      #
####################

cv_hist = []
for i, (train_index, test_index) in enumerate(folds):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test  = X[test_index]
    y_test  = y[test_index]
    
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_test  = torch.tensor(X_test).to(device)
    y_test  = torch.tensor(y_test).to(device)
    
    print("fold", i)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test",  X_test.shape)
    print("y_test",  y_test.shape)

    train_dataset = CWTDataset(X_train, y_train)
    test_dataset  = CWTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    model = Model().double().to(device)

    optimizer = Optimizer(model.parameters(), lr=lr)

    hist = train_acs(
                          name         = name, 
                          tag          = i,
                          model        = model, 
                          # 
                          train_loader = train_loader, 
                          val_loader   = test_loader, 
                          epochs       = epochs,
                          device       = device,
                          # 
                          criterion     = criterion, 
                          optimizer     = optimizer,
                          sparse_lambda = sparse_lambda,
                          #
                          results_dir   = results_dir,
                          )
    
    # 중간 결과 저장
    cv_hist.append(hist)    
    with open(f"{results_dir}/histories_{i}.pkl", "wb") as f:
        pickle.dump(hist, f)

# fold cross validation 결과 출력
running_acc = 0
for i, hist in enumerate(cv_hist):
    acc = hist["val_hist"][-1]['val_acc']
    acc = round(acc, 2)
    running_acc += acc
    print(f"fold {i}: {acc:.2f}")
print(f"mean: {running_acc/10:.2f}")

# 최종 결과 저장
with open(f"{results_dir}/cv_hist.pkl", "wb") as f:
    pickle.dump(cv_hist, f)

