import copy
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from os import mkdir, makedirs
from visualize import *
from torch.utils.tensorboard import SummaryWriter

#from datetime import datetime

def make_fold_structure(exp_path):
    makedirs(exp_path,exist_ok=True)
    for fold_name in  ["Plots", "States"]:
        makedirs(f'{exp_path}/{fold_name}', exist_ok=True) 

        
def save_states(model, optimizer, exp_path):
    torch.save(model.state_dict(), f"{exp_path}/States/model")
    torch.save(optimizer.state_dict(), f"{exp_path}/States/opt")

    
def cross_val_CNN(model, device, train_loader, #test_loader,
                  learn_rate = 3e-3, epochs_num = 60,
                  folds_partition = 5, folds_to_train = 3,
                  criterion=torch.nn.L1Loss(), exp_path = None):
    records = 0
    makedirs(exp_path, exist_ok=True)
    batch_size = train_loader.batch_size
    data, target = train_loader.dataset.tensors
    fold_size = target.shape[0] // folds_partition
    
    plt.figure(figsize = (20,8))
    for i in range(folds_to_train):
        start, end = fold_size * i, fold_size * (i + 1)
        
        train_data, train_target = torch.cat([data[:start],data[end:]]), torch.cat([target[:start],target[end:]])
        val_data, val_target =  data[start:end], target[start:end]        
        train_dataset, val_dataset = TensorDataset(train_data, train_target), TensorDataset(val_data, val_target)      
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)    
        val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size) 
        
        model_CV = copy.deepcopy(model)
        optimizer_CV = torch.optim.Adam(model_CV.parameters(), lr = learn_rate)
        scheduler_CV = torch.optim.lr_scheduler.ExponentialLR(optimizer_CV, gamma = 0.95)
                
        record = train_CNN(model_CV, scheduler_CV, optimizer_CV, device, 
                                                          train_loader, val_loader,
                                                          epochs_num = epochs_num, criterion = criterion,
                                                          exp_path = f'{exp_path}/CV{i}')
        
        if records == 0:
            records = {k: [v] for k,v in record.items()} # make the same, but value is a list
        else:
            for k,v in record.items():
                records[k].append(v)
                
        plt.subplot(1,2,1)
        plt.plot(train_loss, label = f'train CV{i}')
        plt.subplot(1,2,2)
        plt.plot(test_loss, label = f'test CV{i}')
        
        

    plt.legend(fontsize = 20); plt.title('CV train losses',fontsize = 25)
    plt.subplot(1,2,1)
    plt.legend(fontsize = 20); plt.title('CV test losses',fontsize = 25)  
    plt.savefig(f'{exp_path}/all_losses.png')

    df = make_record_df(records, folds_to_train, exp_path)    
    return df


def train_CNN(model, # scheduler
            optimizer, device,
            train_loader, test_loader,
            epochs_num = 40, criterion=torch.nn.L1Loss(),
            pretrained_folder = None, exp_path = None,
            show = False):

    if pretrained_folder is not None:
        model.load_state_dict(torch.load(f'{pretrained_folder}/States/model'))
        optimizer.load_state_dict(torch.load(f'{pretrained_folder}/States/opt'))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)        
    
    def train():
        model.train()
        loss_all, count = 0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outp = model(x_batch.to(device).float())
            loss = criterion(outp, y_batch.to(device).float())
            loss.backward()
            optimizer.step()
            loss_all += loss.item() 
            count += 1  
        train_error = loss_all / count
        return train_error
    
    def test(): # loader, loss_list
        model.eval()        
        loss_all, count = 0, 0        
        with torch.no_grad():
            for x_test_batch, y_test_batch in test_loader:
                outp = model(x_test_batch.to(device).float())
                loss_all +=  criterion(outp, y_test_batch.to(device).float()).item()
                count += 1
        test_error = loss_all / count
        return test_error

    logdir = exp_path + "/logs"
    writer = SummaryWriter(logdir)    
    
    for epoch in tqdm(range(1, epochs_num+1)):  
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_error = train()
        test_error = test()    
        record = Model_Info(model, test_loader,
                           regime = "test",  mode = 'CNN',
                           path_record = None, path_m_by_angle = None,
                           show_distr_by_polar_angle = False, show = False)           
        
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.add_scalar('Loss/train', train_error, epoch)
        writer.add_scalar('Loss/test', test_error, epoch)
        for metric_name, metric in record.items():
            writer.add_scalar(f'Metrics/test/{metric_name}', metric, epoch)        
        
        scheduler.step(test_error)
        
    model.eval()
    
    if exp_path is not None:
        make_fold_structure(exp_path = exp_path)
        save_states(model = model, optimizer = optimizer, exp_path = exp_path)
    
    ############################################## visualize ##############################################
    record = Model_Info(model, test_loader, regime = "test", mode = 'CNN', 
                       show_distr_by_polar_angle = True, show = show,
                       path_record = f'{exp_path}/Plots/record.png',
                       path_m_by_angle = f'{exp_path}/Plots/metrics_by_angle.png')
    
    record_img = Image.open(f'{exp_path}/Plots/record.png')
    metrics_by_angle = Image.open(f'{exp_path}/Plots/metrics_by_angle.png')
    record_img, metrics_by_angle = ToTensor()(record_img), ToTensor()(metrics_by_angle)
    writer.add_image('Record_Image', record_img, global_step=0)
    writer.add_image('Metrics_by_Angle', metrics_by_angle, global_step=0) 
    
    return record

def train_GNN(model: torch.nn.Module, optimizer, device, 
                         train_loader, test_loader, val_loader,
                         criterion = F.mse_loss,                        
                         exp_path = None, #target_name = 'y_polar',
                         pretrained_folder = None,
                         show = True, epochs = 20):
    
    if pretrained_folder is not None:
        model.load_state_dict(torch.load(f'{pretrained_folder}/States/model'))
        optimizer.load_state_dict(torch.load(f'{pretrained_folder}/States/opt'))
          
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)
        
    def train():
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            outp = model(data)          
            y = data.y_polar if outp.shape[-1] == 2 else data.direction
            optimizer.zero_grad()
            loss = criterion(outp, y) 
            loss.backward()
            loss_all += loss.item() # * data.num_graphs
            optimizer.step()
            
        loss_all /= len(train_loader)
        return loss_all 

    def test(loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                outp = model(data)    
                y = data.y_polar if outp.shape[-1] == 2 else data.direction
                error += criterion(outp, y).item() # * data.num_graphs          
        error /= len(loader)
        return error

    logdir = exp_path + "/logs"  
    writer = SummaryWriter(logdir)
    
    for epoch in tqdm(range(1, epochs + 1)):
        lr = scheduler.optimizer.param_groups[0]['lr']       
        train_error = train()       
        test_error = test(test_loader)
        record = Model_Info(model, test_loader,
                           regime = "test",  mode = 'GNN',
                           path_record = None, path_m_by_angle = None,
                           show_distr_by_polar_angle = False, show = False)
        
        
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.add_scalar('Loss/train', train_error, epoch)
        writer.add_scalar('Loss/test', test_error, epoch)
        for metric_name, metric in record.items():
            writer.add_scalar(f'Metrics/test/{metric_name}', metric, epoch)
              
        scheduler.step(test_error)       
               
    if exp_path is not None:
        make_fold_structure(exp_path)
        save_states(model, optimizer, exp_path)
        
    ############################################## visualize ##############################################
    record = Model_Info(model, test_loader, regime = "test", mode = 'GNN', 
                       show_distr_by_polar_angle = True, show = show,
                       path_record = f'{exp_path}/Plots/record.png',
                       path_m_by_angle = f'{exp_path}/Plots/metrics_by_angle.png')
    
    record_img = Image.open(f'{exp_path}/Plots/record.png')
    metrics_by_angle = Image.open(f'{exp_path}/Plots/metrics_by_angle.png')
    record_img, metrics_by_angle = ToTensor()(record_img), ToTensor()(metrics_by_angle)
    writer.add_image('Record_Image', record_img, global_step=0)
    writer.add_image('Metrics_by_Angle', metrics_by_angle, global_step=0) 
    
    writer.close()  
    return record
##############################################################################################################
'''
def fit_E(model, scheduler_Exp, optimizer, device,
        epochs_num = 25, batch_size = 64, tr_set_len = 2 * 512 * 100,
        criterion = torch.nn.L1Loss(),
        exp_path = None ):
   
 
    seq = [j for j in range(int( 1365465/tr_set_len))]
    len_seq = len(seq)
    loss_train, loss_test = [], []
    
    # lnE pred hists  
    train_lnE, val_lnE = [{round(k,1):0 for k in np.arange(0, 20, 0.1)} for i in range(2)] 
    error_amount = 0
    
    num = 0
    for n in tqdm(range(1, epochs_num+1)):
        
        #training
        model.train()
        print('Indeed Epoch = ', n, end = "     ")
        for i in seq:
            train_loss, count = 0, 0
            train_Loader = make_set_E(i, 1, tr_set_len, Batch_size = batch_size, regime = "train")
            for x_batch,y_batch in train_Loader:
                optimizer.zero_grad()
                outp = model(x_batch.to(device).float())
                loss = criterion(outp, y_batch.to(device).float())
                loss.backward()
                optimizer.step()
                
                if num%(len_seq//5) == 0:
                    train_loss += loss.item()
                    count += 1
                # TO DO: optimize it    
                if n == epochs_num:
                    for lnE_value in outp:
                        try:
                            train_lnE[round(lnE_value.item(), 1)] += 1
                        except KeyError:
                            print(' out of train lnE hist, value is: ', round(lnE_value.item(), 1))
                            error_amount += 1
             
            if num%(len_seq//5) == 0:
                loss_train.append(train_loss / count)
                model.eval()
                testLoader = make_set_E(0, -1, 1, Batch_size = batch_size, regime = "val")
                test_loss, count = 0, 0
                for x_test_batch, y_test_batch in testLoader:
                    outp = model(x_test_batch.to(device).float())
                    test_loss +=  criterion(outp, y_test_batch.to(device).float()).item()
                    count += 1
                loss_test.append(test_loss/count)
                model.train()

            num+=1
        scheduler_Exp.step()
        
    model.eval()
    FinalLoader = make_set_E(0, -1, 1, Batch_size = batch_size, regime = "val") 
    for x_test_batch, y_test_batch in FinalLoader:
        outp = model(x_test_batch.to(device).float())
        for lnE_value in outp:
            try:
                val_lnE[round(lnE_value.item(), 1)] += 1
            except KeyError:
                print(' Out of test lnE hist, value is: ', round(lnE_value.item(),1) ) 

    if exp_path is not None:
        make_fold_structure(exp_path = exp_path)
        save_states(model = model, optimizer = optimizer, exp_path = exp_path)
             
    loss_plot(list_test = loss_test, list_train = loss_train , path = exp_path + "/Plots/LOSS.png")
    
    lnE_list = [train_lnE, val_lnE] 
    lnE_hists(lnE_list= lnE_list, path = exp_path + '/Hists/lnE_hists.png')
    
    big_list = [[loss_train,loss_test], lnE_list]    
    print('big_list = [[loss_train,loss_test], [train_lnE, val_lnE] ]')
    
    return  big_list
'''
