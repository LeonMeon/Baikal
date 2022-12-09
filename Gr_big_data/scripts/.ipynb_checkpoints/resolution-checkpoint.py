import torch
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from torch import minimum, acos
from sklearn.metrics import r2_score
from torch.nn.functional import cosine_similarity, normalize


def lnE_res(lnE_pred, lnE_real, hist):
    res = np.abs(lnE_pred - lnE_real)
    for i,res_value in enumerate(res):
        try:
            hist[round(res_value.item(),1)] += 1
        except KeyError:
            print(res_value.item(),"LnE Error is out of range") 
        
def E_res(lnE_pred, lnE_real, hist):
    res = np.abs( np.exp(lnE_pred) - np.exp(lnE_real) )
    for i,res_value in enumerate(res):
        try:
            hist[round(res_value.item(),1)] += 1
        except KeyError:
            print(res_value.item(),"E Error is out of range")  
            
            
def resolution_calculation(Predicted, Real,
                           hist, hist_angle_cut,
                           min_angle = 10., max_angle = 60., ones_torch =  None):
    res = cosine_similarity(Predicted,Real,dim = 1)
    res = acos(minimum(res, ones_torch))/np.pi*180
    polar_real = acos(minimum(Real[:,-1], ones_torch))/(np.pi)*180  # для сравнения
    for i,res_value in enumerate(res):
        try:
            hist[round(res_value.item(),1)] += 1
            if (polar_real[i].item() >= min_angle) and (polar_real[i].item() <= max_angle) :
                hist_angle_cut[round(res_value.item(),1)] += 1
        except KeyError:
            hist[0.0] += 1
            print(res_value.item(),"torch is bad at acos calculation")  

def res_plot(train_dict, val_dict, res_name = "Energy_res",
             path = None, perc_2 = 0.68):
    assert perc_2 > 0.5
    plt.figure(figsize=(13,5))
    colours=["red","red"]
    names = ["train" , "val"]
    for i, d in enumerate([train_dict, val_dict]):
        
        plt.subplot(1,2,i+1)
        s = sum(d.values())
        prep_inter_s ,inter_s = 0, 0
        res_50, res_2 =0,0
        
        for key in list(d.keys()):
            
            if inter_s/s >= perc_2 :
                alpha = (inter_s-perc_2 * s)/(inter_s-prep_inter_s) # alpha*inter_s+(1-alpha)*prep_inter_s == 0.68*s
                res_2 =round(key + 0.1*alpha,2)                     #alpha*(key-0.1)+(1-alpha)*(key)
                break
                
            if inter_s/s >= 0.5 and res_50 == 0:
                alpha = (inter_s-0.5 * s)/(inter_s-prep_inter_s)
                res_50 =round(key + 0.1*alpha,2)
                
            prep_inter_s = inter_s 
            inter_s  += d[key]  
        a=plt.step(list(d.keys())[:300], list(d.values())[:300], color=colours[i], alpha = 0.6)
        plt.bar(res_50, max(d.values()), width=0.5,label=names[i] + "50%" + res_name +"= "+str(res_50),
                color="yellow" , alpha = 0.7 )
        plt.bar(res_2, max(d.values()), width=0.5, label=names[i] + f"{int(perc_2*100)}%" + res_name +" = "+str(res_2),
                color="orange", alpha = 0.7)
        
        plt.legend(fontsize= 12)
        plt.xlabel(res_name + "in_grad",fontsize= 16)
        plt.title(res_name + names[i], fontsize= 22)
        

    plt.savefig(path) 
    plt.show()
    
##################################### GRAPH CASE  ########################
def polar_vec_res_hist(model, device, loader, regime = "train"):
    pol_pred_list, pol_true_list = torch.tensor([]), torch.tensor([])
    #angle_list = torch.tensor([])
    res_list = torch.tensor([])
    print(f'{regime} collect data...')
    for data in loader:
        # from graph dataset making
        #polar = torch.FloatTensor([hf[regime + '/ev_chars'][ind,0] * (np.pi) / 180])[None,:]
        #y_polar = torch.cat((torch.sin(polar), torch.cos(polar)), axis=1)
        v_pred = normalize(model( data.to(device) ) ).cpu()
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
    
    med, sigm2 = np.quantile(res_list, q = 0.5), np.quantile(res_list, q = 0.68)
    
    print(f'{regime} resolution calculation') 
    plt.figure(figsize = (24 , 8))    
    plt.subplot(1,3,1)
    _hist = plt.hist(res_list,bins = 200, histtype= 'step', label = "resolution distribution")
    height = _hist[0][0]
    plt.bar(med, height, width=0.5, align='center',
            label = f'{regime} median is {round(med,2)}', color = "red")
    plt.bar(sigm2, height, width=0.5, align='center',
            label = f'{regime} 68% percentile is {round(sigm2,2)}', color = "orange")
    
    plt.xlabel(f"{regime} polar angle resolution",fontsize= 25)
    plt.title(f"{regime} Resolution",fontsize= 30)
    plt.legend()
    
    # angle distribution
    plt.subplot(1,3,2)
    '''
    path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"
    hf[f'{regime}/ev_chars'][:,0]
    with h5.File(path_to_h5, 'r') as hf:
    '''
    plt.hist(pol_true_list, bins = 100, alpha = 0.2,
                 density = True, label = f"{regime} true polar angle  distribution")
    plt.hist(pol_pred_list, bins = 200, histtype= 'step',
             density = True, label = f"{regime} prediction polar angle  distribution")
    plt.xlabel(f"{regime} polar angle",fontsize= 25)
    plt.title(f"{regime} Polar",fontsize= 30)
    plt.legend()
    
    # scatter plot
    r2 = r2_score(pol_true_list, pol_pred_list)
    plt.subplot(1,3,3)
    plt.scatter(pol_true_list, pol_pred_list, s = 0.1, label = f"R2 = {round(r2,3)}")
    plt.legend(fontsize = 25)
    plt.xlabel(f'True polar', fontsize = 25)
    plt.ylabel(f'Pred polar', fontsize = 25)
    plt.title(f'{regime} Scatter polar plot', fontsize = 30)
    
    plt.show()        
    
