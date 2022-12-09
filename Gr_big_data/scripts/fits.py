#from scripts.loaders import make_set_E, make_set_polar, make_set_vec
#from scripts.v_to_angle import xy_to_angles, xyz_to_angles
#from scripts.plots_hists_angle import loss_plot, lnE_hists
#from scripts.infos import paz_show_info , p_show_info, lnE_show_info
import torch
import torch.nn.functional as F
from tqdm import tqdm
#from tqdm.auto import trange
import numpy as np

from os import mkdir

from resolution import *
from loaders import make_set_E, make_set_polar, make_set_vec
from v_to_angle import xy_to_angles, xyz_to_angles
from plots_hists_angle import loss_plot, lnE_hists
from infos import paz_show_info , p_show_info, lnE_show_info

def make_fold_structure(exp_path):
    mkdir(exp_path)
    for fold_name in  ["Hists", "Plots", "States"]:
        mkdir(f'{exp_path}/{fold_name}') 
        
def save_states(model, optimizer, exp_path, exp_name):
    torch.save(model.state_dict(), f"{exp_path}/States/{exp_name}_model")
    torch.save(optimizer.state_dict(), f"{exp_path}/States/{exp_name}_opt")

############################################################################################## 
# TO DO: make saving
def fit_graph_model(model: torch.nn.Module, optimizer, device, 
                         train_loader, test_loader, val_loader,
                         criterion = F.mse_loss,
                         path_begin = "/home/leonov/Baikal/Gr_big_data/exps/Graph",
                         exp_path = None, exp_name = None,          
                         train_loss = None, test_loss = None, val_loss = None,
                         pretrained_folder = None, 
                         epochs = 20):
    
    if train_loss is None: train_loss = []
    if test_loss is None: test_loss = []
    if val_loss is None: val_loss = []

    if pretrained_folder is not None:
        model.load_state_dict(torch.load(f'{pretrained_folder}/model.pth'))
        optimizer.load_state_dict(torch.load(f'{pretrained_folder}/opt.pth'))
        
    
    model = model.to(device) 
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
        save_states(model, optimizer, exp_path, exp_name)
    
    
    plt.figure(figsize=(9,6))
    plt.plot(train_loss, label='train', linewidth=2)
    plt.plot(test_loss, label='test', linewidth=2)
    plt.plot(val_loss, label='val', linewidth=2)
    
    plt.title('Loss_plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    #plt.savefig(path)
    plt.show() 
    return train_loss, test_loss, val_loss
    
############################################################################################## 

def fit_vec(model, scheduler_Exp, optimizer, device, exp_name, 
        min_angle = 10. ,max_angle = 60.0,
        epochs_num = 25, batch_size = 64,
        criterion=torch.nn.L1Loss(), 
        tr_set_len = 2*512*100, path_begin = "/home/leonov/Baikal/Gr_big_data/exps/Vector" ):
    
    #device = torch.device(f'cuda:{cuda_index}') #if torch.cuda.is_available else torch.device('cpu')
    ones_torch = torch.ones(batch_size).to(device) # for regulation of cosinus
    exp_path = path_begin + exp_name
    
    seq = [j for j in range(int( 1365465/tr_set_len))]
    len_seq, loss_train, loss_test = len(seq), [], []
    
    # здесь будут polar & azimut error всех событий
    train_p_error, val_p_error = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)]
    train_az_error, val_az_error = [{round(k,1):0 for k in np.arange(0.0, 181., 0.1)} for i in range(2)]   
    
    # здесь будут polar error углов от min_angle до max_angle
    train_p_error_cut, val_p_error_cut = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)] 
    
    # здесь будут гистограммы  разрешения всех событий
    train_res, val_res = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)] 
    
    # здесь будут гистограммы  разрешения от min_angle до max_angle
    train_res_cut, val_res_cut = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)] 
    
    # гистограммы  предсказанных углов для сравнения с реальным распределением     
    train_p, val_p = [{k:0 for k in np.arange(0, 181, 1)} for i in range(2)] 
    train_az, val_az = [{k:0 for k in np.arange(0, 361, 1)} for i in range(2)] 
    
    num = 0
    for n in tqdm(range(1, epochs_num+1)): #training
        print('Indeed Epoch = ', n, end = "  ")
        for i in seq:
            train_loss, count = 0, 0
            train_Loader = make_set_vec(i, 1, tr_set_len, Batch_size = batch_size, regime = "train")
            for x_batch, y_batch in train_Loader:
                optimizer.zero_grad()
                outp = model(x_batch.to(device).float())
                loss =  criterion(outp, y_batch.to(device).float())
                loss.backward()
                optimizer.step()
                if num%(len_seq//3) == 0:
                    train_loss += loss.item()
                    count += 1
                # hist filling
                if n == epochs_num:
                    xyz_to_angles(Predicted = outp, Real = y_batch.to(device), ones_torch = ones_torch,
                                p_error_hist_az = train_az_error,
                                p_error_hist = train_p_error,
                                p_error_angle_cut_hist = train_p_error_cut,
                                p_hist = train_p, az_hist = train_az,
                                min_angle = min_angle, max_angle = max_angle)
                    
                    resolution_calculation(outp,y_batch.to(device), ones_torch = ones_torch,
                                           min_angle = min_angle, max_angle = max_angle,
                                           hist = train_res, hist_angle_cut = train_res_cut)
            if num%(len_seq//3) == 0:
                loss_train.append(train_loss / count)
                model.eval()
                testLoader = make_set_vec(0, -1, 1, Batch_size = batch_size, regime = "val")
                test_loss, count = 0, 0 
                for x_test_batch, y_test_batch in testLoader:
                    outp = model(x_test_batch.to(device).float())
                    test_loss +=  criterion(outp, y_test_batch.to(device).float()).item()
                    count += 1
                loss_test.append(test_loss/count)
                model.train()

            num+=1
        scheduler_Exp.step()

    # hist making 
    model.eval()
    FinalLoader = make_set_vec(0, -1, 1, Batch_size = batch_size, regime = "val") 
    plt.figure(figsize=(8,16))
    for x_test_batch,y_test_batch in FinalLoader:
        outp = model(x_test_batch.to(device).float())
        xyz_to_angles(Predicted = outp, Real = y_test_batch.to(device), I_want_scatter_plot = True,
                    ones_torch = ones_torch, 
                    p_error_hist_az = val_az_error,
                    p_error_hist = val_p_error,
                    p_error_angle_cut_hist = val_p_error_cut,
                    p_hist = val_p, az_hist = val_az,
                    min_angle = min_angle, max_angle = max_angle)
        
        resolution_calculation(outp, y_test_batch.to(device), ones_torch = ones_torch,
                               min_angle = min_angle ,max_angle = max_angle,
                               hist = val_res,hist_angle_cut = val_res_cut )

    try:
        make_fold_structure(exp_path = exp_path)
        save_states(model = model,optimizer = optimizer, exp_path = exp_path, exp_name = exp_name)
    except  Exception as e:
        print('smth wrong with make_fold_structure or save_states, do it with output by yourself', '\t', e)
            
    loss_lists = [loss_train , loss_test]
    polar_hists = [train_p , val_p]
    azimut_hists = [train_az, val_az]
    azimut_error = [train_az_error, val_az_error]
    res_hists = [train_res, val_res, train_res_cut, val_res_cut]
    polar_error = [train_p_error, val_p_error, train_p_error_cut, val_p_error_cut]
    
    big_list = [loss_lists, polar_error, azimut_error, res_hists, polar_hists, azimut_hists]
    try:
        paz_show_info(path_start = exp_path, big_list = big_list)
    except Exception as e:
        print('smth wrong with paz_show_info, do it with output by yourself', '\t', e)
    model.train()
    #print('big_list = [loss_lists, polar_error, azimut_error, res_hists, polar_hists, azimut_hists]')
    return big_list

###################################################################################################################################
def fit_p(model, scheduler_Exp, optimizer, device, exp_name,
        min_angle = 10. ,max_angle = 60.0,
        epochs_num = 25, batch_size = 64, tr_set_len = 2 * 512 * 100, 
        criterion=torch.nn.L1Loss(), path_begin = "/home/leonov/Baikal/Gr_big_data/exps/Polar"):

    #device = torch.device(f'cuda:{cuda_index}') #if torch.cuda.is_available else torch.device('cpu')
    exp_path = f'{path_begin}/{exp_name}'
    
    seq = [j for j in range(int( 1365465/tr_set_len))]
    len_seq, loss_train, loss_test = len(seq), [], []
       
    # здесь будут polar & azimut error всех событий
    train_p_error, val_p_error = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)] 
    
    # здесь будут polar error углов от min_angle до max_angle
    train_p_error_cut, val_p_error_cut = [{round(k,1):0 for k in np.arange(0.0, 180., 0.1)} for i in range(2)] 
    
    # гистограммы  предсказанных углов для сравнения с реальным распределением     
    train_p, val_p = [{k:0 for k in np.arange(0, 181, 1)} for i in range(2)] 

    num = 0
    for n in tqdm(range(1, epochs_num+1)):
        model.train()
        #training
        print('Indeed Epoch = ', n, end = "     ")
        for i in seq:
            train_loss, count = 0, 0
            train_Loader = make_set_polar(i, 1, tr_set_len, Batch_size = batch_size, regime = "train")
            for x_batch,y_batch in train_Loader:
                optimizer.zero_grad()
                outp = model(x_batch.to(device).float())
                loss = criterion(outp,y_batch.to(device).float())
                loss.backward()
                optimizer.step()
                
                if num%(len_seq//3) == 0:
                    train_loss += loss.item()
                    count += 1
                    
                if n == epochs_num:
                    xy_to_angles(Predicted = outp, Real = y_batch.to(device),
                                p_hist = train_p, p_error_hist = train_p_error,
                                p_error_angle_cut_hist = train_p_error_cut, 
                                min_angle = min_angle, max_angle = max_angle)
                    
            if num%(len_seq//3) == 0:
                loss_train.append(train_loss / count)
                model.eval()
                testLoader = make_set_polar(0, -1, 1, Batch_size = batch_size, regime = "val")
                test_loss, count = 0, 0
                for x_test_batch, y_test_batch in testLoader:
                    outp = model(x_test_batch.to(device).float())
                    test_loss +=  criterion(outp, y_test_batch.to(device).float()).item()
                    count += 1
                loss_test.append(test_loss/count)
                model.train()

            num += 1
        scheduler_Exp.step()
        
    model.eval()
    
    FinalLoader = make_set_polar(0, -1, 1, Batch_size = batch_size, regime = "val") 
    for x_test_batch, y_test_batch in FinalLoader:
        outp = model(x_test_batch.to(device).float())
        xy_to_angles(Predicted = outp, Real = y_test_batch.to(device), I_want_scatter_plot = True,
                    p_hist = val_p, p_error_hist = val_p_error,
                    p_error_angle_cut_hist = val_p_error_cut,
                    min_angle = min_angle, max_angle = max_angle)

    make_fold_structure(exp_path = exp_path)
    save_states(model = model, optimizer = optimizer, exp_path = exp_path, exp_name = exp_name)
        
    loss_lists = [loss_train, loss_test]
    polar_hists = [train_p, val_p]
    polar_error = [train_p_error, val_p_error, train_p_error_cut, val_p_error_cut]
    big_list = [loss_lists, polar_error, polar_hists]
    
    p_show_info(exp_path, suffix, big_list)  
        
    print('big_list = [loss_lists, polar_error, polar_hists]')
    return big_list

##############################################################################################################

def fit_E(model, scheduler_Exp, optimizer, device, exp_name,
        epochs_num = 25, batch_size = 64, tr_set_len = 2 * 512 * 100,
        criterion = torch.nn.L1Loss(), path_begin = "/home/leonov/Baikal/Gr_big_data/exps/Energy"):
   
    exp_path = f'{path_begin}/{exp_name}'
 
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

    try:
        make_fold_structure(exp_path = exp_path)
        save_states(model = model, optimizer = optimizer, exp_path = exp_path, exp_name = exp_name)
    except:
        print('smth wrong with make_fold_structure or save_states, do it with output by yourself')              
    loss_plot(list_test = loss_test, list_train = loss_train , path = exp_path + "/Plots/LOSS.png")
    
    lnE_list = [train_lnE, val_lnE] 
    lnE_hists(lnE_list= lnE_list, path = exp_path + '/Hists/lnE_hists.png')
    
    big_list = [[loss_train,loss_test], lnE_list]    
    print('big_list = [[loss_train,loss_test], [train_lnE, val_lnE] ]')
    
    return  big_list

