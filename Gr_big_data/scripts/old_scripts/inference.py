import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
import torch
from resolution import *

'''

for x_test_batch, y_test_batch in FinalLoader:
    outp = model(x_test_batch.to(device).float())
    xy_to_angles(Predicted = outp, Real = y_test_batch.to(device), I_want_scatter_plot = True,
                p_hist = val_p, p_error_hist = val_p_error,
                p_error_angle_cut_hist = val_p_error_cut,
                min_angle = min_angle, max_angle = max_angle)
'''

def xy_to_polar(predict, true): 
    v_pred = normalize(predict.detach()) # нормализую
    p_predict = torch.acos(v_pred[:,-1]) / (np.pi) * 180 
    p_true = torch.acos(true[:,-1]) / (np.pi) * 180 
    '''
    if I_want_scatter_plot:
        plt.scatter(polar_real.cpu().detach().numpy(), polar_predicted.cpu().detach().numpy(), s = 1, color = "blue", alpha =0.3)
        plt.xlabel("Real polar angle", fontsize = 25)
        plt.ylabel("Predicted polar angle", fontsize = 25)
        plt.title("Scatter plot for polar angle", fontsize = 30)
    '''
    p_predict = p_predict.cpu().detach().numpy()
    p_true = p_true.cpu().detach().numpy()
    return p_predict, p_true

               
def xyz_to_angles(predict, true, ones_torch): 
    v_pred = normalize(predict.detach()) 
    p_real = torch.acos(torch.minimum(true[:,-1], ones_torch)) / (np.pi) * 180 
    p_predict = torch.acos(torch.minimum(v_pred[:,-1], ones_torch)) / (np.pi) * 180 
    # azimut pred
    az_predict = torch.acos( v_pred[:,0] / ((v_pred[:,0])**2 + (v_pred[:,1])**2 + 1e-8)**0.5 ) 
    az_predict = az_predict + (torch.sign(v_pred[:,1])**2) * (1 - torch.sign(v_pred[:,1])) * (np.pi - az_predict) 
    # torch.sign(v[:,1])**2 if there're zero angles 
    
    # azimut_true
    az_true = torch.acos( true[:,0] / ((true[:,0])**2 + (true[:,1])**2 + 1e-8)**0.5 ) 
    az_true = az_true + (torch.sign(true[:,1])**2) * (1 - torch.sign(true[:,1])) * (np.pi - az_true)  

    az_true = az_true / (np.pi) * 180 # to grads
    az_predict = az_predict / (np.pi) * 180 #  to grads

    '''
    if I_want_scatter_plot:
        plt.subplot(2, 1, 1)
        plt.scatter(polar_real.cpu().detach().numpy(), polar_predicted.cpu().detach().numpy(), s=1 ,color = "blue",alpha = 0.3 )
        plt.xlabel("Real polar angle", fontsize = 17)
        plt.ylabel("Predicted polar angle", fontsize = 17)
        plt.title("Scatter plot for polar angle", fontsize = 26)
        
        plt.subplot(2, 1, 2)
        plt.scatter(azimut_real.cpu().detach().numpy(), azimut.cpu().detach().numpy(), s=1 ,color = "blue", alpha = 0.2 )
        plt.xlabel("Real azimut angle", fontsize = 17)
        plt.ylabel("Predicted azimut angle", fontsize = 17)
        plt.title("Scatter plot for azimut angle", fontsize = 26)
    '''
    az_real = az_real.cpu().detach().numpy()
    az_predict = az_predict.cpu().detach().numpy()
    p_real =  p_real.cpu().detach().numpy()
    p_predict = p_predict.cpu().detach().numpy()
    
    return az_real, az_predict, p_real, p_predict 