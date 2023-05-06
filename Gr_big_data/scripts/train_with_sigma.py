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
from torch.utils.tensorboard import SummaryWriter
#import torch.nn.functional.mse_loss as MSE
from train import make_fold_structure, save_states
from visualize_with_sigma import Model_with_sigma_Info


def MSE_uncert(pred_and_sigma, target):
    pred = pred_and_sigma[:,:-1]
    sigma2 = pred_and_sigma[:,-1]**2
    return torch.mean( torch.log(sigma2) + torch.sum((pred - target)**2, 1) / (sigma2))

def L1_uncert(pred_and_sigma, target):
    pred = pred_and_sigma[:,:-1]
    sigma2 = pred_and_sigma[:,-1]**2
    return torch.mean( torch.log(sigma2) + torch.sum(torch.abs(pred - target), 1) / (sigma2))
    

def train_GNN_with_sigma(model: torch.nn.Module, optimizer, device, 
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
            outp_and_sigma = model(data)          
            y = data.y_polar if outp.shape[-1] == 2 else data.direction
            optimizer.zero_grad()
            loss = criterion(outp_and_sigma, y) 
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
                outp_and_sigma = model(data)    
                y = data.y_polar if outp.shape[-1] == 2 else data.direction
                error += criterion(outp_and_sigma, y).item() # * data.num_graphs          
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

