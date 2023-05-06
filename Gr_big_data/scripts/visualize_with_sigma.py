import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.nn.functional import cosine_similarity, normalize
from torch_geometric.nn import global_mean_pool
from visualize import outp_to_res_and_angles, resolution_hist, scatter_plot, angle_hist, metrics_by_angles
LEGEND_SIZE = 20
TITLE_SIZE = 25
LABEL_SIZE = 20
TICKS_SIZE = 20

plt.rcParams.update({'font.size': 20})
METRICS = ['polar_res', 'polar_r2', 'azimut_res', 'direction_res']

###############################   Model_Info     #####################################
## TODO: make smth in case of prediction of azimut
def Model_with_sigma_Info(model, loader, regime = "test",
               bin_size = 10, mode = 'CNN',
               path_record = None, path_m_by_angle = None,
               show_distr_by_polar_angle = True,show = True
               ):
    record = {}
    _device = next(model.parameters()).device
    pol_pred_list, pol_true_list = torch.tensor([]), torch.tensor([]), 
    az_pred_list, az_true_list = torch.tensor([]), torch.tensor([])
    pol_res_list, az_res_list = torch.tensor([]), torch.tensor([])
    resolution_list = torch.tensor([])
    
    with torch.no_grad():       
        if mode == 'CNN': ############# CNN ###############################
            raise NotImplementedError
            for x_batch, y_batch in loader:
                predict = model(x_batch.to(_device).float()).detach().cpu()
                
                resolution, angles = outp_to_res_and_angles(predict, y_batch)  

                p_predict, p_true = angles[0].squeeze(), angles[1].squeeze()
                p_res = (p_predict - p_true).abs()

                pol_res_list = torch.cat((pol_res_list, p_res), axis = 0)         
                pol_true_list = torch.cat((pol_true_list, p_true), axis = 0)
                pol_pred_list = torch.cat((pol_pred_list, p_predict), axis = 0)
                
                if len(angles) == 4:           
                    az_predict, az_true = angles[2].squeeze(), angles[3].squeeze()
                    az_res = (az_predict - az_true).abs()
                    az_res = torch.where(az_res < 180, az_res, 360 - az_res)

                    az_res_list = torch.cat((az_res_list, az_res), axis = 0)  
                    az_true_list = torch.cat((az_true_list, az_true), axis = 0)
                    az_pred_list = torch.cat((az_pred_list, az_predict), axis = 0)
                    resolution_list = torch.cat((resolution_list, resolution), axis = 0)           
        elif mode == 'GNN': ############# GNN ###############################
            for data in loader:
                data = data.to(_device)
                outp_and_sigma = model(data).detach().cpu()  
                outp = outp_and_sigma[:,:-1] #, outp_and_sigma[:,-1]
                
                if outp.shape[1] == 2:
                    y_batch = data.y_polar.squeeze().cpu()                    
                elif outp.shape[1] == 3:
                    y_batch = data.direction.squeeze().cpu()
                    
                resolution, angles = outp_to_res_and_angles(outp, y_batch) 

                p_predict, p_true = angles[0].squeeze(), angles[1].squeeze()
                p_res = (p_predict - p_true).abs()

                pol_res_list = torch.cat((pol_res_list, p_res), axis = 0)         
                pol_true_list = torch.cat((pol_true_list, p_true), axis = 0)
                pol_pred_list = torch.cat((pol_pred_list, p_predict), axis = 0)
                
                if len(angles) == 4:           
                    az_predict, az_true = angles[2].squeeze(), angles[3].squeeze()
                    az_res = (az_predict - az_true).abs()
                    az_res = torch.where(az_res < 180, az_res, 360 - az_res)

                    az_res_list = torch.cat((az_res_list, az_res), axis = 0)  
                    az_true_list = torch.cat((az_true_list, az_true), axis = 0)
                    az_pred_list = torch.cat((az_pred_list, az_predict), axis = 0)
                    resolution_list = torch.cat((resolution_list, resolution), axis = 0)                    
        else:
            print("Emmmm.... What?")
        
    resolution_list = resolution_list.numpy()
    pol_res_list, az_res_list = pol_res_list.numpy(), az_res_list.numpy()   
    az_true_list, az_pred_list = az_true_list.numpy(), az_pred_list.numpy()      
    pol_true_list, pol_pred_list = pol_true_list.numpy(), pol_pred_list.numpy()
    # drawing
    rows = 1 if az_pred_list.shape[0] == 0 else 3   
    plt.figure(figsize = (24, 8 * rows))   
    # polar  resolution
    plt.subplot(rows, 3, 1)
    polar_med = resolution_hist(resolution = pol_res_list, title = f'{regime} polar', x_max = 15)
    record['polar_res'] = polar_med
    # polar scatter
    plt.subplot(rows, 3, 2)
    polar_r2 =scatter_plot(true = pol_true_list, predict = pol_pred_list, title = f'{regime} polar',
                 r2 = r2_score(pol_true_list, pol_pred_list))
    record['polar_r2'] = polar_r2
    # polar distribution
    plt.subplot(rows, 3, 3)
    angle_hist(true = pol_true_list, predict = pol_pred_list, title = f'{regime} polar')
    
    if rows == 3: 
        # azimut  resolution
        plt.subplot(rows, 3, 4)
        azimut_res = resolution_hist(resolution = az_res_list, title = f'{regime} azimut', x_max = 45)
        record['azimut_res'] = azimut_res
        # azimut scatter
        plt.subplot(rows, 3, 5)
        scatter_plot(true = az_true_list, predict = az_pred_list, title = f'{regime} azimut')
        # azimut distribution
        plt.subplot(rows, 3, 6)
        angle_hist(true = az_true_list, predict = az_pred_list, title = f'{regime} azimut')
        # direction  resolution
        plt.subplot(rows, 3, 8)
        direction_res = resolution_hist(resolution = resolution_list, title = f'{regime} direction', x_max = 15)
        record['direction_res'] = direction_res
        
    if path_record != None:
        plt.savefig(path_record)
    ############         Metrics for  certain polar angle values   ###################
    if show_distr_by_polar_angle:
        metrics_by_angles(pol_true_list, pol_res_list, az_res_list, resolution_list,
                          bin_size = bin_size, angle = rows + 1, path = path_m_by_angle, show = show)        
    ##################################################################################    
    if not show:
        plt.close()
    else:
        plt.show()  
    return record
 
    
###########################################                  test_resolution           ##############    
def test_err_and_res_with_sigma(model, loader, mode = 'GNN'):
    error = 0
    _device = next(model.parameters()).device
    resolution_list = torch.tensor([])
    
    with torch.no_grad():       
        if mode == 'CNN': ############# CNN ###############################
            raise NotImplementedError
            for x_batch, y_batch in loader:
                predict = model(x_batch.to(_device).float()).detach().cpu()     
                resolution, angles = outp_to_res_and_angles(predict, y_batch)          
                error += criterion(outp, y_batch).item() 
                if len(angles) == 4:           
                    resolution_list = torch.cat((resolution_list, resolution), axis = 0)
                    
                else:
                    p_predict, p_true = angles[0].squeeze(), angles[1].squeeze()
                    p_res = (p_predict - p_true).abs()
                    resolution_list = torch.cat((resolution_list, p_res), axis = 0)  
                    
        elif mode == 'GNN': ############# GNN ###############################
            for data in loader:
                data = data.to(_device)
                outp_and_sigma = model(data).detach().cpu()  
                outp = outp_and_sigma[:,:-1] #, outp_and_sigma[:,-1]                
                if outp.shape[1] == 2:
                    y_batch = data.y_polar.squeeze().cpu()                    
                elif outp.shape[1] == 3:
                    y_batch = data.direction.squeeze().cpu()
                    
                resolution, angles = outp_to_res_and_angles(outp, y_batch) 
                error += criterion(outp, y_batch).item()  
                if len(angles) == 4:           
                    resolution_list = torch.cat((resolution_list, resolution), axis = 0)  
                    
                else:
                    p_predict, p_true = angles[0].squeeze(), angles[1].squeeze()
                    p_res = (p_predict - p_true).abs()
                    resolution_list = torch.cat((resolution_list, p_res), axis = 0)
        else:
            print("Emmmm.... What?")
    res = np.quantile(resolution_list.numpy(), q = 0.5)
    return res, error / len(loader)

