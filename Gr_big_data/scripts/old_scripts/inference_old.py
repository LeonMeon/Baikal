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

def xy_to_angles(Predicted, Real,
                p_hist, p_error_hist,
                p_error_angle_cut_hist, I_want_scatter_plot = False,
                min_angle = 10., max_angle = 60.): 
    v_pred = normalize(Predicted.detach()) # нормализую
    polar_predicted = torch.acos(v_pred[:,-1]) / (np.pi) * 180 
    polar_real = torch.acos(Real[:,-1]) / (np.pi) * 180 
    if I_want_scatter_plot:
        plt.scatter(polar_real.cpu().detach().numpy(), polar_predicted.cpu().detach().numpy(), s = 1, color = "blue", alpha =0.3)
        plt.xlabel("Real polar angle", fontsize = 25)
        plt.ylabel("Predicted polar angle", fontsize = 25)
        plt.title("Scatter plot for polar angle", fontsize = 30)
        
    for pol_pred, pol_real in zip(polar_predicted, polar_real):
        p_error_hist[round(abs((pol_pred-pol_real).item()),1)] += 1
        # for certain angles !!!!!!!!!!!!!!!!!!!!!!!
        if (pol_real.item() >= min_angle) and (pol_real.item() <= max_angle): 
            p_error_angle_cut_hist[round(abs((pol_pred-pol_real).item()),1)] += 1
        p_hist[pol_pred.short().item()] += 1

               
def xyz_to_angles(Predicted, Real, ones_torch,
                p_hist, az_hist, 
                p_error_hist_az,
                p_error_hist, p_error_angle_cut_hist, I_want_scatter_plot = False,
                min_angle = 10., max_angle = 60., ): 
    v_pred = normalize(Predicted.detach()) 
    polar_real = torch.acos(torch.minimum(Real[:,-1], ones_torch)) / (np.pi) * 180 
    polar_predicted = torch.acos(torch.minimum(v_pred[:,-1], ones_torch)) / (np.pi) * 180 
    # azimut pred
    azimut = torch.acos( v_pred[:,0] / ((v_pred[:,0])**2 + (v_pred[:,1])**2 + 1e-8)**0.5 ) 
    azimut = azimut + (torch.sign(v_pred[:,1])**2) * (1 - torch.sign(v_pred[:,1])) * (np.pi - azimut) 
    # torch.sign(v[:,1])**2 if there're zero angles 
    
    # azimut_real
    azimut_real =torch.acos( Real[:,0] / ((Real[:,0])**2 + (Real[:,1])**2 + 1e-8)**0.5 ) 
    azimut_real = azimut_real + (torch.sign(Real[:,1])**2)*(1 - torch.sign(Real[:,1]))*(np.pi - azimut_real)  

    azimut_real = azimut_real/(np.pi)*180 # to grads
    azimut = azimut/(np.pi)*180 #  to grads

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
        
    azimut = azimut.short()
    for pol_pred, pol_real in zip(polar_predicted, polar_real):
        p_error_hist[round(abs((pol_pred - pol_real).item()),1)] += 1
        # for certain angles !!!!!!!!!!!!!!!!!!!!!!!
        if (pol_real.item() >= min_angle) and (pol_real.item() <= max_angle): 
            p_error_angle_cut_hist[round(abs((pol_pred - pol_real).item()), 1)] += 1        
        p_hist[pol_pred.short().item()] += 1

    for az_pred, az_real in zip(azimut, azimut_real): 
        er = abs((az_pred - az_real).item())
        er= min(360 - er, er)
        p_error_hist_az[round(er, 1)] += 1
        az_hist[az_pred.item()] += 1