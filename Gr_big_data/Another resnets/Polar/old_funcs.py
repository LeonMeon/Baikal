#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
#from sklearn.metrics import r2_score
device = torch.device('cuda:1') if torch.cuda.is_available else torch.device('cpu')
print(device)
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
plt.style.use("seaborn-talk") #"classic" "seaborn-talk" "seaborn"
path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"

# можно отделить хвост и отдельно прогнать его по другой сетке
def make_set(i, di = 1, tr_set_len = 128,Batch_size = 32,regime = "train"):
    with h5py.File(path_to_h5, 'r') as hf:
            Data = hf[regime+ '/data/'][i*int(tr_set_len) : (i+di)*int(tr_set_len),:32]
            Polar=hf[regime+ '/ev_chars'][i*int(tr_set_len) : (i+di)*int(tr_set_len),0]*(np.pi)/180
            #Azimut=hf["/train/ev_chars"][i*int(tr_set_len) : (i+di)*int(tr_set_len),1]*(np.pi)/180
            x=np.expand_dims(np.sin(Polar),axis=1)
            y=np.expand_dims(np.cos(Polar),axis=1)
            target=torch.FloatTensor(np.concatenate((x,y) ,axis=1))
            Data = torch.FloatTensor(Data.swapaxes(1, -1)) # надо,т.к. второй индекс должен быть количеством   последовательностей
            Dataset = torch.utils.data.TensorDataset(Data, target)
            Loader = torch.utils.data.DataLoader(dataset = Dataset, batch_size=Batch_size) #,sampler = sampler    
    return  Loader
# вектора в углы
def v_to_angles( Predicted, Real,
                p_hist, p_error_hist,
                p_error_angle_cut_hist, I_want_scatter_plot = False,
                min_angle = 10., max_angle = 60.): 
    v_pred = torch.nn.functional.normalize(Predicted.detach()) # нормализую
    polar_predicted = torch.acos(v_pred[:,-1])/(np.pi)*180 
    polar_real = torch.acos(Real[:,-1])/(np.pi)*180 
    if I_want_scatter_plot:
        plt.scatter(polar_real.cpu().detach().numpy(),polar_predicted.cpu().detach().numpy(), s=1 ,color = "blue",alpha =0.3 )
        plt.xlabel("Real polar angle", fontsize = 25);plt.ylabel("Predicted polar angle", fontsize = 25)
        plt.title("Scatter plot for polar angle", fontsize = 30)
    for pol_pred, pol_real in zip(polar_predicted, polar_real):
        p_error_hist[round(abs((pol_pred-pol_real).item()),1)] += 1
        # for certain angles !!!!!!!!!!!!!!!!!!!!!!!
        if (pol_real.item() >= min_angle) and (pol_real.item() <= max_angle): 
            p_error_angle_cut_hist[round(abs((pol_pred-pol_real).item()),1)] += 1
        p_hist[pol_pred.short().item()] += 1
    

def loss_plot(list_test, list_train , path , save = True):
    plt.figure(figsize=(9,6))
    plt.plot(np.arange(len(list_test)), list_test, label='val', linewidth=2)
    plt.plot(np.arange(len(list_train)), list_train, label='train', linewidth=2)
    plt.title('Loss_plot')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    if save == True:
        plt.savefig(path)
    plt.show()
def res_plot(train_dict,val_dict, path = None,save = True, res_or_polar = "Resolution " ,size= 13):
    plt.figure(figsize=(13,5))
    colours=["red","red"]
    names = ["train" , "val"]
    for i, d in enumerate([train_dict,val_dict]):
        plt.subplot(1,2,i+1)
        s = sum(d.values())
        prep_inter_s ,inter_s = 0, 0
        res_50, res_68 =0,0
        for key in list(d.keys()):
            if inter_s/s >= 0.68:
                alpha = (inter_s-0.68*s)/(inter_s-prep_inter_s) # alpha*inter_s+(1-alpha)*prep_inter_s == 0.68*s
                res_68 =round(key + 0.1*alpha,2)  #alpha*(key-0.1)+(1-alpha)*(key)
                break
            if inter_s/s >= 0.5 and res_50 == 0:
                alpha = (inter_s-0.5*s)/(inter_s-prep_inter_s)
                res_50 =round(key + 0.1*alpha,2)
            prep_inter_s = inter_s 
            inter_s  += d[key]  
        a=plt.step(list(d.keys())[:50], list(d.values())[:50], color=colours[i],alpha = 0.6)
        plt.bar(res_50, max(d.values()), width=0.5,label=names[i] + "50%" + res_or_polar +"= "+str(res_50),
                color="yellow" , alpha =0.7 )
        plt.bar(res_68, max(d.values()), width=0.5,label=names[i] + "68%" + res_or_polar +" = "+str(res_68) ,
                color="orange",alpha = 0.7)
        plt.legend(fontsize= 12); #plt.suptitle(res_or_polar,fontsize =size+3)
        plt.xlabel(res_or_polar + "in_grad",fontsize= 16); plt.title(res_or_polar + names[i],fontsize= 22)
    if save == True:
        plt.savefig(path) 
    plt.show()

def angle_hist(hist_polar,path, save =True,size= 13,name = "train"): #hist_azimut
    with h5py.File(path_to_h5, 'r') as hf:
        plt.figure(figsize= (13,6))
        sum_value = sum(hist_polar.values())
        Polar=hf["/"+name+"/ev_chars"][ : ,0] 
        #Azimut=hf["/"+name+"/ev_chars"][ : ,1]
        #plt.subplot(2,1,1)
        plt.hist(Polar,bins=180,label="Real polar angle in " + name +" data",density=True,histtype="step",color="blue")
        plt.bar(list(hist_polar.keys())[:100], np.array(list(hist_polar.values())[:100])/sum_value, 
                color="red",label= "Predicted polar angle in " + name +" data")
        plt.legend(fontsize = 18)
        plt.xlabel("Polar angle",fontsize= 25); plt.title(name+" polar angle",fontsize= 30)
        if save == True:
            plt.savefig(path) 
        plt.show()

def fitting(model, scheduler_Exp, scheduler_MultiStep , optimizer,
        min_angle = 10. ,max_angle = 60.0,
        epochs_num = 25, batch_size = 64,
        criterion=torch.nn.L1Loss(),
        save_weights = True, save_plot = True, save_resolution = True,  save_angles = True, save_polar_error = True,   
        suffix = "Nu_MAE_Only_Polar_MC_DATA_",
        path_begin = "/home/leonov/Baikal/Gr_big_data"):
    #path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"
    tr_set_len = 512*100
    seq = [j for j in range(int( 1365465/tr_set_len))]
    print('Num of sub-epochs in Epoch = ', len(seq), '\n')
    len_seq = len(seq)
    loss_train = []
    loss_test = []
    
    # здесь будут polar error всех событий
    hist_train_polar_error = {round(k,1):0 for k in np.arange(0.0, 180, 0.1)} 
    hist_val_polar_error = {round(k,1):0 for k in np.arange(0.0, 180, 0.1)}
    
    # здесь будут polar error углов от min_angle до max_angle
    hist_train_polar_error_angle_cut = {round(k,1):0 for k in np.arange(0.0, 180., 0.1)} 
    hist_val_polar_error_angle_cut = {round(k,1):0 for k in np.arange(0.0, 180., 0.1)} 
    
    # гистограммы  предсказанных углов для сравнения с реальным распределением     
    hist_train_polar = {k:0 for k in np.arange(-5, 181, 1)}
    hist_train_azimut = {k:0 for k in np.arange(-5, 361, 1)}
    hist_val_polar = {k:0 for k in np.arange(-5, 181, 1)} 
    hist_val_azimut = {k:0 for k in np.arange(-5, 361, 1)}  
    num = 0
    for n in range(1, epochs_num+1):
        #training
        print('Indeed Epoch = ', n, end = "     ")
        for i in seq:
            train_Loader = make_set(i,1,tr_set_len,Batch_size = batch_size, regime = "train")
            for x_batch,y_batch in train_Loader:
                optimizer.zero_grad()
                outp = model(x_batch.to(device).float())
                loss =   criterion(outp,y_batch.to(device).float())
                loss.backward()
                optimizer.step()
                # полученный вектор направления превращаю в углы и добавляю в гистограммы
                if n == epochs_num:
                    v_to_angles(Predicted = outp, Real = y_batch.to(device),
                                p_error_hist = hist_train_polar_error,
                                p_error_angle_cut_hist = hist_train_polar_error_angle_cut,
                                p_hist = hist_train_polar,
                                min_angle = min_angle, max_angle = max_angle) # az_hist = hist_train_azimut, to_hists = True
                    #resolution_calculation(outp,y_batch.to(device),hist = hist_train_res,to_hist = True)
            if (num%(len_seq//3) == 0):
                rand_ind=np.random.randint(0,15)
                #print('Sub-epoch number = ', num)
                loss_train.append(loss.item())
                model.eval()
                testLoader = make_set(rand_ind,1,500, Batch_size = 256, regime = "val")
                test_loss=0
                count=0
                for x_test_batch,y_test_batch in testLoader:
                    outp = model(x_test_batch.to(device).float())
                    test_loss +=  criterion(outp,y_test_batch.to(device).float()).item()
                    count+=1
                test_loss /=count
                loss_test.append(test_loss)
                model.train()
                #print("train_loss = ",loss.item(),"  val_loss = ",test_loss)
            num+=1
        scheduler_Exp.step()
        scheduler_MultiStep.step()
        
    model.eval()
    FinalLoader = make_set(0,-1,1, Batch_size = 256, regime = "val") # делаю  loader из всего датасета
    for x_test_batch,y_test_batch in FinalLoader:
        outp = model(x_test_batch.to(device).float())
        v_to_angles(Predicted = outp, Real = y_test_batch.to(device), I_want_scatter_plot = True,
                    p_error_hist = hist_val_polar_error,
                    p_error_angle_cut_hist = hist_val_polar_error_angle_cut,
                    min_angle = min_angle, max_angle = max_angle,
                    p_hist = hist_val_polar) #, az_hist = hist_val_azimut to_hists = True,

    if save_weights == True:
        torch.save(model.state_dict(), path_begin + "/states/" + suffix + "model")
        torch.save(optimizer.state_dict(), path_begin + "/states/" + suffix + "opt")        
    loss_lists = [loss_train , loss_test]
    polar_hists = [hist_train_polar , hist_val_polar]
    polar_error = [hist_train_polar_error, hist_val_polar_error,
                   hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut
                  ]    
    # график лосса
    loss_plot(loss_test, loss_train , path_begin + "/Images/Loss/" + suffix + "LOSS.png", 
              save_plot )
    
    #  гистограмма ошибок  полярного угла
    res_plot(hist_train_polar_error, hist_val_polar_error, 
             path = path_begin + "/Images/Polar_Error/" + suffix+ "Polar_Error.png",
             save = save_polar_error, res_or_polar = "Polar_Error ")
     
    #  гистограмма ошибок  полярного угла для определенных углов
    res_plot(hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut, 
             path = path_begin + "/Images/Polar_Error_Angle_Cut/" + suffix+ "Polar_Error_Angle_Cut.png",
             save = save_polar_error, res_or_polar = "Polar_Error_Angle_Cut ") 
    #гистограммы углов
    angle_hist(hist_train_polar, name='train',
               path = path_begin + "/Images/Angles/" + suffix + "Angles_train.png", 
               save = save_angles)
    angle_hist(hist_val_polar, name='val',
               path = path_begin + "/Images/Angles/" + suffix + "Angles_val.png", 
               save = save_angles)
    
    model.train()
    return  loss_lists, polar_hists ,polar_error 

