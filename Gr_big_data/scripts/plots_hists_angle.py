import numpy as np
import h5py
import matplotlib.pyplot as plt

path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"

def loss_plot(list_test, list_train , path ):
    plt.figure(figsize=(9,6))
    plt.plot(np.arange(len(list_test)), list_test, label='val', linewidth=2)
    plt.plot(np.arange(len(list_train)), list_train, label='train', linewidth=2)
    plt.title('Loss_plot')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(path)
    plt.show()    
    
        
def lnE_hists(lnE_list,path):
    plt.figure(figsize= (13,6))
    with h5py.File(path_to_h5, 'r') as hf:
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
    
  
    
def p_angle_hist(hist_polar, path, size= 13, name = "train"): 
    real_label = "Real polar angle in " + name + " data"
    pred_label = "Predicted polar angle in " + name + " data"
    
    with h5py.File(path_to_h5, 'r') as hf:
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
    


def paz_angle_hist(hist_polar, hist_azimut, path, size= 20, name = "train"):
    
    real_p_label = "Real polar angle in " + name + " data"
    pred_p_label = "Predicted polar angle in " + name +" data"
    real_az_label = "Real azimut angle in " + name +" data"
    pred_az_label = "Predicted azimut angle in " + name +" data"

    plt.figure(figsize= (13,18))
    sum_value = sum(hist_polar.values())
    
    with h5py.File(path_to_h5, 'r') as hf:

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
    
        