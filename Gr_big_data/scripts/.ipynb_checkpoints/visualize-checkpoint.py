import torch
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.nn.functional import cosine_similarity, normalize

LEGEND_SIZE =  15

def outp_to_res_and_angles(predict, true): 
    v_pred = normalize(predict.detach()) # нормализую    
    p_true = torch.acos(true[:,-1]) * 180 / np.pi  
    p_predict = torch.acos(v_pred[:,-1]) * 180 / np.pi
    
    outp_size = true.shape[1]
    if outp_size == 2: 
        return None, [p_predict, p_true]
    
    elif outp_size == 3:                 
        az_true = torch.acos(true[:,0] / true[:,:2].norm(dim=1)) * 180 / np.pi
        sin_sign = torch.sign(true[:,1])
        az_true = torch.where(sin_sign > 0, az_true, 360 - az_true)

        az_predict = torch.acos(v_pred[:,0] / v_pred[:,:2].norm(dim=1)) * 180 / np.pi
        sin_sign = torch.sign(v_pred[:,1])
        az_predict = torch.where(sin_sign > 0, az_predict, 360 - az_predict)
        
        resolution = (v_pred * true).sum(axis = 1).clamp(max = 1.).acos() * 180 / np.pi
        return resolution, [p_predict, p_true, az_predict, az_true]
    
    else:
        raise Exception("Strange output, it should be 2 or 3 dim vector")


def loss_plot(train_loss, test_loss, path ):
    plt.figure(figsize=(9,6))
    plt.plot(np.arange(len(test_loss)), test_loss, label='test', linewidth=2)
    plt.plot(np.arange(len(train_loss)), train_loss, label='train', linewidth=2)
    
    plt.title('Loss_plot', fontsize = 25)
    plt.xlabel('iterations', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.legend(fontsize = LEGEND_SIZE)
    
    plt.savefig(path)
    plt.show()    

    
def resolution_hist(resolution, x_min = 0.0, x_max = 15, title = ''):
    title = title.title()
    med, sigm2 = np.quantile(resolution, q = 0.5), np.quantile(resolution, q = 0.68)
 
    hist = plt.hist(resolution, bins = 300, histtype= 'step', density = True)   
    height = hist[0][0]    
    plt.bar(med, height, width=0.5, align='center', color = "red",
            label = f'{title} angle median Resolution is {round(med, 2)}')   
    plt.bar(sigm2, height, width=0.5, align='center', color = "orange",
            label = f'{title} angle 68% Resolution is {round(sigm2, 2)}')
    
    plt.xlabel(f"Resolution", fontsize= 25)
    plt.xlim(left = x_min, right=x_max) 
    plt.title(f"{title} Resolution Distribution", fontsize= 30)
    plt.legend(fontsize = LEGEND_SIZE)   
    return med
    
def angle_hist(true, predict, title = ''):
    title = title.title()
    plt.hist(true, bins = 100, alpha = 0.2, density = True,
             label = f"True {title} Angle Distribution")
    plt.hist(predict, bins = 100, histtype= 'step', density = True,
             label = f"Predicted {title} Angle Distribution")
    plt.xlabel(f"{title} Angle", fontsize = 25)
    plt.title(f"{title} Angle Distribution", fontsize = 25)
    plt.legend(fontsize = LEGEND_SIZE)

def scatter_plot(true, predict, title = '', r2 = None):
    title = title.title()
    plt.scatter(true, predict, s = 0.1)
    plt.xlabel(f'True {title}', fontsize = 25)
    plt.ylabel(f'Predicted {title}', fontsize = 25)
    if r2 != None:
        plt.title(f'{title} Scatter Plot \n R2 = {round(r2,3)}', fontsize = 30) 
        return r2
    else:
        plt.title(f'{title} Scatter Plot', fontsize = 30)     
        
    
def CNN_Info(model, loader, regime = "test", path = None, show = True):
    record = {}
    _device = next(model.parameters()).device
    pol_pred_list, pol_true_list = torch.tensor([]), torch.tensor([]), 
    az_pred_list, az_true_list = torch.tensor([]), torch.tensor([])
    pol_res_list, az_res_list = torch.tensor([]), torch.tensor([])
    resolution_list = torch.tensor([])
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader):
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
        
    resolution_list = resolution_list.numpy()
    pol_res_list, az_res_list = pol_res_list.numpy(), az_res_list.numpy()   
    az_true_list, az_pred_list = az_true_list.numpy(), az_pred_list.numpy()      
    pol_true_list, pol_pred_list = pol_true_list.numpy(), pol_pred_list.numpy()

    # drawing
    rows = 1 if az_pred_list.shape[0] == 0 else 3   
    plt.figure(figsize = (24, 8*rows))
    
    # polar  resolution
    plt.subplot(rows, 3, 1)
    polar_med = resolution_hist(resolution = pol_res_list, title = f'{regime} polar', x_max = 15)
    record['polar_med'] = polar_med
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
    
    if path != None:
        plt.savefig(path)
    if not show:
        plt.close()
    else:
        plt.show()  
    return record
    

def GNN_Info(model, loader, regime = "train"):
    _device = next(model.parameters()).device
    pol_pred_list, pol_true_list, res_list = torch.tensor([]), torch.tensor([]), torch.tensor([])
    #angle_list = torch.tensor([])   
    for data in loader:
        #polar = torch.FloatTensor([hf[regime + '/ev_chars'][ind,0] * (np.pi) / 180])[None,:]
        #y_polar = torch.cat((torch.sin(polar), torch.cos(polar)), axis=1)
        v_pred = normalize(model( data.to(_device) ) ).cpu()
        p_pred = torch.acos(v_pred[:,-1]) 
        p_true = data.polar.squeeze().cpu()   
        pol_res = torch.abs(p_true - p_pred)
        res_list = torch.cat((res_list, pol_res), axis = 0)
        pol_true_list = torch.cat((pol_true_list, p_pred), axis = 0)
        pol_pred_list = torch.cat((pol_pred_list, p_true), axis = 0)
        
    res_list = (res_list / np.pi * 180).detach().numpy()
    #angle_list = (angle_list / np.pi * 180).detach().numpy()
    pol_pred_list = (pol_pred_list / np.pi * 180).detach().numpy()
    pol_true_list = (pol_true_list / np.pi * 180).detach().numpy()
        
    # angle distribution   
    # scatter plot
