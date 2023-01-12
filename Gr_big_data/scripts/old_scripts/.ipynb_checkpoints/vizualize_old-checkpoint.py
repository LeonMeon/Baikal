import torch
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from resolution import res_plot, lnE_res ,E_res

from torch import minimum, acos
from sklearn.metrics import r2_score
from torch.nn.functional import cosine_similarity, normalize


##############################################################    polar angle prediction 
def p_angle_hist(hist_polar, path, size= 13, name = "train"): 
    real_label = "Real polar angle in " + name + " data"
    pred_label = "Predicted polar angle in " + name + " data"
    
    with h5.File(path_to_h5, 'r') as hf:
        plt.figure(figsize= (13,6))
        sum_value = sum(hist_polar.values())
        Polar=hf["/" + name + "/ev_chars"][:,0] 

        plt.hist(Polar, bins = 180, label = real_label, density = True, histtype = "step",color = "blue")
        plt.bar(list(hist_polar.keys())[:100], np.array(list(hist_polar.values())[:100]) / sum_value, 
                color = "red", label = pred_label)
        plt.legend(fontsize = 18)
        plt.xlabel("Polar angle", fontsize= 25)
        plt.title(name +" polar angle",fontsize = 30)

    plt.savefig(path)     
    plt.show()
    

def p_show_info(path_start, big_list):
    loss_lists, polar_error, polar_hists = big_list

    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0],  path = path_start + "/Plots/LOSS.png")
        
    #histogram of polar angle errors
    res_plot(polar_error[0], polar_error[1],  #hist_train_polar_error, hist_val_polar_error
                 path = path_start + "/Hists/Polar_Error.png", res_name = "Polar_Error")  

    #histogram of polar angle errors for specific angles
    res_plot(polar_error[2], polar_error[3], #hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut,  
             path = path_start + "/Hists/Polar_Error_Angle_Cut.png", res_name = "Polar_Error_Angle_Cut ") 

    #angle histograms
    p_angle_hist(polar_hists[0], name='train', path = path_start + "/Hists/Angles_train.png")
    p_angle_hist(polar_hists[1], name='val', path = path_start + "/Hists/Angles_val.png")  
    
    loss_lists = [loss_train , loss_test]
    polar_hists = [hist_train_polar , hist_val_polar]
    polar_error = [hist_train_polar_error, hist_val_polar_error,
                       hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut] 
    

##############################################################    both angles prediction        
    
def paz_angle_hist(hist_polar, hist_azimut, path, size= 20, name = "train"):
    
    real_p_label = "Real polar angle in " + name + " data"
    pred_p_label = "Predicted polar angle in " + name +" data"
    real_az_label = "Real azimut angle in " + name +" data"
    pred_az_label = "Predicted azimut angle in " + name +" data"

    plt.figure(figsize= (13,18))
    sum_value = sum(hist_polar.values())
    
    with h5.File(path_to_h5, 'r') as hf:

        Polar=hf["/" + name + "/ev_chars"][:,0] 
        Azimut=hf["/" + name + "/ev_chars"][:,1]
        
        plt.subplot(2,1,1)
        plt.hist(Polar,bins=180, label= real_p_label, density=True, histtype="step", color="blue")
        plt.bar(list(hist_polar.keys())[:100], np.array(list(hist_polar.values())[:100]) / sum_value, 
                color="red",label = pred_p_label)
        plt.legend(fontsize = 18)
        plt.xlabel("Polar Angle",fontsize= 25)
        plt.title(name+"_Polar",fontsize= 30)
        
        plt.subplot(2,1,2) 
        plt.hist(Azimut, bins=360, label = real_az_label, density = True, histtype = "step", color = "blue")
        plt.bar(list(hist_azimut.keys()), np.array(list(hist_azimut.values())) / sum_value,
                    align = 'center', color = "red", label = pred_az_label)
        plt.legend(fontsize = 18)
        plt.xlabel("Azimut Angle",fontsize= 25)
        plt.title(name+" Azimut angle",fontsize= 30)
    
    plt.savefig(path) 
    plt.show()
    
########################################################################################################

def paz_show_info(path_start, big_list):
    loss_lists, polar_error, azimut_error, res_hists, polar_hists, azimut_hists = big_list

    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0], 
                  path = path_start + "/Plots/Loss.png" )
              
    res_plot(polar_error[0], polar_error[1],  #hist_train_polar_error, hist_val_polar_error,
                 path = path_start + "/Hists/Polar_Error.png", res_name = "Polar_Error ")
        
    #histogram of azimut angle errors"
    res_plot(azimut_error[0], azimut_error[1],  #hist_train_azimut_error, hist_val_azimut_error",
                 path = path_start + "/Hists/Azimut_Error.png", res_name = "Azimut_Error ") 
        
    #histogram of polar angle errors for specific angles",
    res_plot(polar_error[2], polar_error[3], #hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut,"
                 path = path_start + "/Hists/Polar_Error_Angle_Cut.png", res_name = "Polar_Error_Angle_Cut ") 

    #resolution histograms",
    res_plot(res_hists[0],res_hists[1],
                 path = path_start + "/Hists/RESOLUTIONS.png", res_name = "Resolution ")
        
    #resolution histograms for specific angles",
    res_plot(res_hists[2], res_hists[3],
                 path = path_start + "/Hists/Resolution_Angle_Cut.png", res_name = "Resolution_Angle_Cut ")
        
    #angle histograms",
    paz_angle_hist(polar_hists[0], azimut_hists[0], name='train', path = path_start + "/Hists/Angles_distr_train.png")
    paz_angle_hist(polar_hists[1], azimut_hists[1], name='val', path = path_start + "/Hists/Angles_distr_val.png")
    
###################################################        lnE prediction                 ###################################################    

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


def lnE_hists(lnE_list,path):
    plt.figure(figsize= (13,6))
    with h5.File(path_to_h5, 'r') as hf:
        for i, hist, name in zip([1,2], lnE_list, ["train", "val"]):
            plt.subplot(1, 2, i)
            lnE_real = np.log(hf[f"{name}/ev_chars"][:,2])
            
            inds = np.nonzero(np.array(list(hist.values())))
            summ = sum(hist.values()) * 0.1
            length = len(inds[0])
            
            plt.hist(lnE_real , bins = length, density = True, color = "blue", histtype = "step", label = f"{name} lnE real")
            sum_value = sum(hist.values()) * 0.1
            plt.bar(list(hist.keys()), np.array(list(hist.values())) / sum_value, color = "red", label = f"{name} lnE pred")
            plt.legend(fontsize = 18)
            plt.xlabel(f"{name} lnE", fontsize= 25)
            plt.title(f"{name} lnE ",fontsize = 30)
    
    plt.savefig(path)
    plt.show()        
  
    
def lnE_show_info(path_start, loss_lists, lnE_error):
    #loss plot
    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0],  path = path_start + "/Plots/LOSS.png" )
        
    #histogram of lnE errors
    res_plot(lnE_error[0], lnE_error[1],  #hist_train_lnE_error, hist_val_lnE_error,
                 path = path_start + "/Hists/lnE_Error.png", res_name = "lnE_Error ")   
    
        



            
            
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