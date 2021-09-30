import os
import shutil
import numpy as np
import telegram
import time
import pickle

import torch
from torch import nn
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

def make_results_directory(name, base=".", copy_file="main.py", copy_dir=None):
    results_dir = f"{base}/results/{name}"
    
    # Valid Check
    if os.path.exists(results_dir):
        print(f"'{results_dir}' already exists!")
        raise
    
    # Create 
    os.mkdir(results_dir)
    os.mkdir(results_dir+"/models")
    os.mkdir(results_dir+"/log")
    os.mkdir(results_dir+"/tb")
    
    # Copy
    shutil.copy(copy_file, results_dir)
    if copy_dir:
        shutil.copytree(copy_dir, results_dir+"/"+copy_dir)
    
    print(f"'{results_dir}' is created!")
    
    return results_dir
    

def current_time():
    return "UTC %d-%02d-%02d %02d:%02d:%02d"%(time.gmtime(time.time())[0:6])


def current_date_hour():
    return "%d%02d%02d%02d"%(time.gmtime(time.time())[0:4])


class MyLogger:
    def __init__(self, text_writer=None, tb_writer=None):
        self.text_writer = text_writer
        self.tb_writer = tb_writer
        
    def write_step(self, mode, epoch, step, accuracy, loss, bce_loss, sparse_loss, time_step):
        if self.text_writer:
            self.text_writer.write(f"{current_time()} :: {epoch}epoch {mode} {step}step: accuracy {accuracy:.2f}% || " \
                                   + f"loss {loss:.6f} || bce_loss {bce_loss:.6f} || sparse_loss {sparse_loss:.6f} ||" \
                                   + f"{time_step//60}min {time_step%60:.2f}sec\n")
            self.text_writer.flush()
                   
    def write_epoch(self, epoch, 
                    train_acc, train_time_epoch,
                    train_loss, train_bce_loss, train_sparse_loss,
                    val_acc, val_time_epoch,
                    val_loss, val_bce_loss, val_sparse_loss):
        print(f"Train : acc {train_acc:.2f}  " \
              + f"loss {train_loss:.6f}  bce_loss {train_bce_loss:.6f}  sparse_loss {train_sparse_loss:.6f}  " \
              + f"{train_time_epoch//60}min {train_time_epoch%60:.2f}sec")
        print(f"Val : acc {val_acc:.2f}  " \
              + f"loss {val_loss:.6f}  bce_loss{val_bce_loss:.6f}  sparse_loss {val_sparse_loss:.6f}  " \
              + f"{val_time_epoch//60}min {val_time_epoch%60:.2f}sec")
        
        if self.text_writer:
            self.text_writer.write(f"Train : acc {train_acc:.2f}  " \
                                  + f"loss {train_loss:.6f}  bce_loss {train_bce_loss:.6f}  sparse_loss {train_sparse_loss:.6f}  " \
                                  + f"{train_time_epoch//60}min {train_time_epoch%60:.2f}sec")
            self.text_writer.flush()
            self.text_writer.write(f"Val : acc {val_acc:.2f}  " \
                                  + f"loss {val_loss:.6f}  bce_loss{val_bce_loss:.6f}  sparse_loss {val_sparse_loss:.6f}  " \
                                  + f"{val_time_epoch//60}min {val_time_epoch%60:.2f}sec")
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
                        
            outputs, s_acs = model(inputs, return_s_acs=True)  # feed forward
            probs = F.sigmoid(outputs)
            preds = probs.round()

            # Accuracy
            corrects = (preds == labels)        
            accuracy = torch.sum(corrects) / inputs.shape[0] * 100

            # Loss
            bce_loss     = criterion(outputs, labels) 
            sparse_loss  = sparse_lambda * torch.norm(s_acs.squeeze(), 1, dim=1).mean()
            loss         = bce_loss + sparse_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # History
            B = inputs.shape[0]
            running_corrects    += torch.sum(corrects).item()
            running_loss        += loss.item() * B
            running_bce_loss    += bce_loss.item() * B
            running_sparse_loss += sparse_loss.item() * B
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
                
                outputs, s_acs = model(inputs, return_s_acs=True)  # feed forward
                probs = F.sigmoid(outputs)
                preds = probs.round()

                # Accuracy
                corrects = (preds == labels)        
                accuracy = torch.sum(corrects) / inputs.shape[0] * 100

                # Loss
                bce_loss     = criterion(outputs, labels) 
                sparse_loss  = sparse_lambda * torch.norm(s_acs.squeeze(), 1, dim=1).mean()
                loss         = bce_loss + sparse_loss
                
                # History
                B = inputs.shape[0]
                Vrunning_corrects    += torch.sum(corrects).item()
                Vrunning_loss        += loss.item() * B
                Vrunning_bce_loss    += bce_loss.item() * B
                Vrunning_sparse_loss += sparse_loss.item() * B
                time_step = time.time() - time_step_start
                logger.write_step(mode="test", epoch=epoch, step=step, time_step=time_step,
                                  accuracy=accuracy, loss=loss, bce_loss=bce_loss, sparse_loss=sparse_loss)
                step += 1

        Vtime_epoch = time.time() - Vtime_epoch_start
        
        ### Epoch Log ###
        train_acc  = running_corrects / len_train * 100
        train_loss = running_loss     / len_train
        train_bce_loss = running_bce_loss / len_train
        train_sparse_loss = running_sparse_loss / len_train
        
        val_acc  = Vrunning_corrects / len_val * 100
        val_loss = Vrunning_loss     / len_val
        val_bce_loss = Vrunning_bce_loss / len_val
        val_sparse_loss = Vrunning_sparse_loss / len_val
        
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
    


