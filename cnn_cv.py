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
parser.add_argument('--seed', default="n")
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
    ):
        super().__init__()
        
        # FEATURE EXTRACTION
        self.conv_layer1 = nn.Conv2d(n_channels, n_kerenls, kernel_size=(4,4), stride=(2, 2), padding=1)
        
        self.conv_layer2 = nn.Conv2d(n_kerenls, n_kerenls, kernel_size=(4,4), stride=(4, 4))
        
        self.conv_layer3 = nn.Conv2d(n_kerenls, n_kerenls, kernel_size=(4,4), stride=(4, 4))
        
        # OUTPUT
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
#         self.sigmoid = F.sigmoid()
        
    def forward(self, inputs, return_s_acs=False): 
        """ 
        Args
        ----
            inputs (batch, channel, height, width) 
        """
        B, _, _, _ = inputs.shape
        s_acs = inputs.new_zeros(B,22,1,1)
        
        # FEATURE EXTRACTION
        x = self.conv_layer1(inputs)
        x = F.elu(x)
        
        x = self.conv_layer2(x)
        x = F.elu(x)
        
        x = self.conv_layer3(x)
        x = F.elu(x)   
        
        # OUTPUT
        x = x.reshape(B, 256)
        x = self.fc1(x) # (B, 64)
        x = F.elu(x)
        out = self.fc2(x)
        
        if return_s_acs:
            return out, s_acs
        else:
            return out
        

###################
#### Load Data ####
###################

file_path = f"{data_path}/{'A' if fit_data=='2a' else 'B'}0{args.subject}_64x64_scipy2_ica_cv10.pkl"
print("Load data from:", file_path)

with open(file_path, "rb") as f:
    data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    y = y.reshape(-1,1)
    folds = data["folds"]
    
X = torch.tensor(X.astype(np.float32)).to(device)
y = torch.tensor(y.astype(np.float32)).to(device)
    
class CWTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
    

###################
#### Functions ####
###################
import os
import shutil
import numpy as np
import time
import pickle

import torch
from torch import nn
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

def current_time():
    return "UTC %d-%02d-%02d %02d:%02d:%02d"%(time.gmtime(time.time())[0:6])


def current_date_hour():
    return "%d%02d%02d%02d"%(time.gmtime(time.time())[0:4])


class MyLogger:
    def __init__(self, text_writer=None, tb_writer=None):
        self.text_writer = text_writer
        self.tb_writer = tb_writer
        
    def write_step(self, mode, epoch, step, accuracy, loss, bce_loss, sparse_loss, time_step):
        if not self.text_writer:
            return 
        else :
            self.text_writer.write(f"{current_time()} :: {epoch:3d}epoch {mode} {step:3d}step: accuracy {accuracy.item():3.20f}% || " \
                                   + f"loss {loss.item():3.20f} || bce_loss {bce_loss.item():3.20f} || sparse_loss {sparse_loss.item():3.20f} ||" \
                                   + f"{time_step//60}min {time_step%60:.2f}sec\n")
            self.text_writer.flush()
                   
    def write_epoch(self, epoch, 
                    train_acc, train_time_epoch,
                    train_loss, train_bce_loss, train_sparse_loss,
                    val_acc, val_time_epoch,
                    val_loss, val_bce_loss, val_sparse_loss):
        
        print(f"Train : acc {train_acc:3.2f}  " \
              + f"loss {train_loss:3.2f}  bce_loss {train_bce_loss:3.2f}  sparse_loss {train_sparse_loss:3.2f}  " \
              + f"{train_time_epoch//60:.2f}min {train_time_epoch%60:.2f}sec")
        print(f"Val   : acc {val_acc:3.2f}  " \
              + f"loss {val_loss:3.2f}  bce_loss {val_bce_loss:3.2f}  sparse_loss {val_sparse_loss:3.2f}  " \
              + f"{val_time_epoch//60}min {val_time_epoch%60:.2f}sec")
        
        if self.text_writer:
            self.text_writer.write(f"Train : acc {train_acc:3.20f}  " \
                                  + f"loss {train_loss:3.20f}  bce_loss {train_bce_loss:3.20f}  sparse_loss {train_sparse_loss:3.20f}  " \
                                  + f"{train_time_epoch//60}min {train_time_epoch%60:.2f}sec\n")
            self.text_writer.flush()
            self.text_writer.write(f"Val : acc {val_acc:3.20f}  " \
                                  + f"loss {val_loss:3.20f}  bce_loss{val_bce_loss:3.20f}  sparse_loss {val_sparse_loss:3.20f}  " \
                                  + f"{val_time_epoch//60}min {val_time_epoch%60:.2f}sec\n")
            self.text_writer.flush()
        
        if self.tb_writer:
            self.tb_writer.add_scalars("accuracy",   {"train":train_acc,         "val":val_acc},  epoch)
            self.tb_writer.add_scalars("loss",       {"train":train_loss,        "val":val_loss}, epoch)
            self.tb_writer.add_scalars("bce_loss",   {"train":train_bce_loss,    "val":val_bce_loss},  epoch)
            self.tb_writer.add_scalars("sparse_loss",{"train":train_sparse_loss, "val":val_sparse_loss}, epoch)

    def close(self):
        if self.text_writer:
            self.text_writer.close()
        if self.tb_writer:
            self.tb_writer.close()
        
        
def train_acs(
                name, 
                tag, 
                model, 
                #
                train_loader, 
                val_loader, 
                epochs, 
                device,
                #
                criterion, 
                optimizer,  
                sparse_lambda,
                #
                results_dir,):
    
    # Log
    print(name, tag)
    text_writer = open(f"{results_dir}/log/{name}-{tag}-{current_date_hour()}.log", "w")
    tb_writer   = SummaryWriter(f"{results_dir}/tb/{tag}")
    logger      = MyLogger(text_writer, tb_writer)
#     logger = MyLogger()
    
    # Setup
    train_hist = []
    val_hist = []
    best_val_loss = np.inf
    best_val_acc = -1
    len_train = len(train_loader.dataset)
    len_val   = len(val_loader.dataset)

    for epoch in range(0, epochs):
        print(f"{epoch} epoch")
        
        time_epoch_start = time.time()
                
        ### TRAIN ###
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_sparse_loss = 0.0
        running_corrects = 0
        step = 0

        for inputs, labels in train_loader:
            time_step_start = time.time()
            B = inputs.shape[0]
                        
            outputs, s_acs = model(inputs, return_s_acs=True)  # feed forward
            probs = torch.sigmoid(outputs)
            preds = probs.round()

            # Accuracy
            corrects = (preds == labels)        
            accuracy = torch.sum(corrects) / B * 100

            # Loss
            bce_loss     = criterion(outputs, labels) 
            sparse_loss  = sparse_lambda * torch.norm(s_acs.squeeze(), 1) / B # sparse loss using B
            loss         = bce_loss + sparse_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # History
            running_corrects    += torch.sum(corrects)
            running_loss        += loss * B
            running_bce_loss    += bce_loss * B
            running_sparse_loss += sparse_loss * B
            time_step = time.time() - time_step_start
            logger.write_step(mode="train", epoch=epoch, step=step, time_step=time_step,
                              accuracy=accuracy, loss=loss, bce_loss=bce_loss, sparse_loss=sparse_loss)
            step += 1
        
        time_epoch = time.time() - time_epoch_start
        Vtime_epoch_start = time.time()
        
        ### VALIDATION ###
        with torch.no_grad():
            model.eval()
            Vrunning_loss = 0.0
            Vrunning_bce_loss = 0.0
            Vrunning_sparse_loss = 0.0
            Vrunning_corrects = 0
            step = 0
            
            for inputs, labels in val_loader:
                time_step_start = time.time()
                B = inputs.shape[0]
                
                outputs, s_acs = model(inputs, return_s_acs=True)  # feed forward
                probs = torch.sigmoid(outputs)
                preds = probs.round()

                # Accuracy
                corrects = (preds == labels)        
                accuracy = torch.sum(corrects) / B * 100

                # Loss
                bce_loss     = criterion(outputs, labels) 
                sparse_loss  = sparse_lambda * torch.norm(s_acs.squeeze(), 1) / B # sparse loss using B
                loss         = bce_loss + sparse_loss
                
                # History
                Vrunning_corrects    += torch.sum(corrects)
                Vrunning_loss        += loss * B
                Vrunning_bce_loss    += bce_loss * B
                Vrunning_sparse_loss += sparse_loss * B
                time_step = time.time() - time_step_start
                logger.write_step(mode="test", epoch=epoch, step=step, time_step=time_step,
                                  accuracy=accuracy, loss=loss, bce_loss=bce_loss, sparse_loss=sparse_loss)
                step += 1

        Vtime_epoch = time.time() - Vtime_epoch_start
        
        ### Epoch Log ###
        train_acc         = running_corrects.item()    / len_train * 100
        train_loss        = running_loss.item()        / len_train
        train_bce_loss    = running_bce_loss.item()    / len_train
        train_sparse_loss = running_sparse_loss.item() / len_train
        
        val_acc           = Vrunning_corrects.item()    / len_val * 100
        val_loss          = Vrunning_loss.item()        / len_val
        val_bce_loss      = Vrunning_bce_loss.item()    / len_val
        val_sparse_loss   = Vrunning_sparse_loss.item() / len_val
        
        logger.write_epoch(epoch=epoch, 
                           train_acc=train_acc, train_time_epoch=time_epoch,
                           train_loss=train_loss, train_bce_loss=train_bce_loss, train_sparse_loss=train_sparse_loss,  
                           val_acc=val_acc, val_time_epoch=Vtime_epoch,
                           val_loss=val_loss, val_bce_loss=val_bce_loss, val_sparse_loss=val_sparse_loss)
        
        train_hist.append({"train_acc":train_acc, "train_loss":train_loss, "train_bce_loss":train_bce_loss, "train_sparse_loss":train_sparse_loss})
        val_hist.append({"val_acc":val_acc,       "val_loss":val_loss,     "val_bce_loss":val_bce_loss,     "val_sparse_loss":val_sparse_loss})
        
        ### Best Check ###
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{results_dir}/models/{name}_{tag}_best_model.h5") # overwrite
            
    # Save Last Model
    torch.save(model.state_dict(), f"{results_dir}/models/{name}_{tag}_{epoch}epoch_model.h5")
    
    logger.close()
    
    return {"train_hist":train_hist, "val_hist":val_hist, "best_val_acc":best_val_acc}
    

####################
#    Training      #
####################

cv_hist = []
for i, (train_index, test_index) in enumerate(folds):
    
    train_dataset = CWTDataset(X[train_index], y[train_index])
    test_dataset  = CWTDataset(X[test_index], y[test_index])

    print("fold", i)
    print("X_train", train_dataset.X.shape)
    print("y_train", train_dataset.y.shape)
    print("X_test",  test_dataset.X.shape)
    print("y_test",  test_dataset.y.shape)

    print("Deterministic:", args.seed)
    if args.seed == "y":
        seed = 2021010556
        print(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    model = Model().to(device)

    # print("weight initialization with glorot:")
    # for n, m in model.named_modules():
    #     if isinstance(m, nn.Conv2d):
    #         print(n)
    #         nn.init.xavier_uniform_(m.weight)
    #         nn.init.xavier_uniform_(m.bias)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

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

