import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from os import mkdir

#from vizualize import *
#from loaders import make_set_E, make_set_polar, make_set_vec
#from inference import xy_to_angles, xyz_to_angles

def make_fold_structure(exp_path):
    mkdir(exp_path)
    for fold_name in  ["Hists", "Plots", "States"]:
        mkdir(f'{exp_path}/{fold_name}') 
        
def save_states(model, optimizer, exp_path):
    torch.save(model.state_dict(), f"{exp_path}/States/model")
    torch.save(optimizer.state_dict(), f"{exp_path}/States/opt")

#######################################################################################################################
#"/home/leonov/Baikal/Gr_big_data/exps/Polar"
def train_CNN(model, scheduler, optimizer, device,
            train_loader, test_loader,
            train_loss = None, test_loss = None,
            epochs_num = 25, criterion=torch.nn.L1Loss(),
            pretrained_folder = None, exp_path = None):
        
    if train_loss is None: train_loss = []
    if test_loss is None: test_loss = []

    if pretrained_folder is not None:
        model.load_state_dict(torch.load(f'{pretrained_folder}/model.pth'))
        optimizer.load_state_dict(torch.load(f'{pretrained_folder}/opt.pth'))

    for n in tqdm(range(1, epochs_num+1)):   
        model.train()
        loss_all, count = 0, 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outp = model(x_batch.to(device).float())
            loss = criterion(outp,y_batch.to(device).float())
            loss.backward()
            optimizer.step()
            loss_all += loss.item() 
            count += 1
            
        train_loss.append(loss_all / count)

        model.eval()        
        loss_all, count = 0, 0
        
        for x_test_batch, y_test_batch in test_loader:
            outp = model(x_test_batch.to(device).float())
            loss_all +=  criterion(outp, y_test_batch.to(device).float()).item()
            count += 1
            
        test_loss.append(loss_all / count)
        model.train()

        scheduler.step()
        
    model.eval()
    
    if exp_path is not None:
        make_fold_structure(exp_path = exp_path)
        save_states(model = model, optimizer = optimizer, exp_path = exp_path)
    
    np.save(f'{exp_path}/Plots/train_loss.json', train_loss)
    np.save(f'{exp_path}/Plots/test_loss.json', test_loss)
    return train_loss, test_loss    

#path_begin = "/home/leonov/Baikal/Gr_big_data/exps/Graph",
def train_graph_model(model: torch.nn.Module, optimizer, device, 
                         train_loader, test_loader, val_loader,
                         criterion = F.mse_loss,                         
                         exp_path = None,          
                         train_loss = None, test_loss = None, val_loss = None,
                         pretrained_folder = None,
                         show = True, 
                         epochs = 20):
    
    if train_loss is None: train_loss = []
    if test_loss is None: test_loss = []
    if val_loss is None: val_loss = []

    if pretrained_folder is not None:
        model.load_state_dict(torch.load(f'{pretrained_folder}/model.pth'))
        optimizer.load_state_dict(torch.load(f'{pretrained_folder}/opt.pth'))
          
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)
        
    def train():
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.y_polar) #l1_loss
            loss.backward()
            loss_all += loss.item() #* data.num_graphs
            optimizer.step()
            
        loss_all /= len(train_loader)
        train_loss.append(loss_all)
        return loss_all 

    def test(loader,loss_list):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += criterion(model(data), data.y_polar).item() #* data.num_graphs
            
        error /= len(loader)
        loss_list.append(error)
        return error


    best_val_error = None
    for epoch in tqdm(range(1, epochs + 1)):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_error = train()
        test_error = test(test_loader, test_loss)
        scheduler.step(test_error)
        
        if best_val_error is None or val_error <= best_val_error:
            val_error = test(val_loader,val_loss)
            best_val_error = val_error
        
        
        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Train Loss: {train_error:.7f}, '
              f'Val Loss: {val_error:.7f}, Test Loss: {test_error:.7f}')
        
        '''
        if epoch == epochs - 1:
            print("Printing resolution hists and angle distribution for train...  ")
            polar_vec_res_hist(model, device, train_loader, regime = "train")
            print("Printing resolution hists and angle distribution for test...  ")
            polar_vec_res_hist(model, device, test_loader, regime = "test")
        '''
    if exp_path is not None:
        make_fold_structure(exp_path)
        save_states(model, optimizer, exp_path)
    
    
    plt.figure(figsize=(9,6))
    plt.plot(train_loss, label='train', linewidth=2)
    plt.plot(test_loss, label='test', linewidth=2)
    plt.plot(val_loss, label='val', linewidth=2)
    
    plt.title('Loss_plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'{exp_path}/Plots/loss.png' )
    plt.show() 
    return train_loss, test_loss, val_loss



##############################################################################################################
#"/home/leonov/Baikal/Gr_big_data/exps/Energy"
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

