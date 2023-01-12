import numpy as np
import torch
import h5py
path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"

def make_set_E(i, di = 1, tr_set_len = 128, Batch_size = 64, regime = "train"):
    assert regime in ["train", "val", "test"]
    with h5py.File(path_to_h5, 'r') as hf:
            Data = hf[regime + '/data/'][i * int(tr_set_len) : (i + di) * int(tr_set_len), :32]
            #lnE = np.log(hf[regime + "ev_chars"][i * int(tr_set_len) : (i + di) * int(tr_set_len),2] + 1)
            lnE = np.log(hf[regime + "/ev_chars"][i * int(tr_set_len) : (i + di) * int(tr_set_len),2])
            target =  torch.FloatTensor( np.expand_dims(lnE , axis = 1 ) )
            Data = torch.FloatTensor(Data.swapaxes(1, -1)) # the second index must be the number of sequences
            Dataset = torch.utils.data.TensorDataset(Data, target)
            Loader = torch.utils.data.DataLoader(dataset = Dataset, batch_size = Batch_size, drop_last = True) #,sampler = sampler    
    return  Loader

def make_set_polar(i, di = 1, tr_set_len = 128, Batch_size = 64, regime = "train"):
    with h5py.File(path_to_h5, 'r') as hf:
        Data = hf[regime + '/data/'][i * int(tr_set_len) : (i + di) * int(tr_set_len), :32]
        Polar=hf[regime + '/ev_chars'][i * int(tr_set_len) : (i + di) * int(tr_set_len),0] * (np.pi) / 180
        x=np.expand_dims(np.sin(Polar),axis=1)
        y=np.expand_dims(np.cos(Polar),axis=1)        
        target=torch.FloatTensor(np.concatenate((x,y) ,axis=1))
        Data = torch.FloatTensor(Data.swapaxes(1, -1)) 
        Dataset = torch.utils.data.TensorDataset(Data, target)
        Loader = torch.utils.data.DataLoader(dataset = Dataset, batch_size = Batch_size)   
    return  Loader

def make_set_vec(i, di = 1, tr_set_len = 128, Batch_size = 64, regime = "train"):
    assert regime in ["train", "val", "test"]
    with h5py.File(path_to_h5, 'r') as hf:
        Data = hf[regime + '/data/'][i * int(tr_set_len) : (i + di) * int(tr_set_len), :32]
        Polar = hf[regime + '/ev_chars'][i * int(tr_set_len) : (i + di) * int(tr_set_len), 0] * (np.pi) / 180
        Azimut = hf[regime + '/ev_chars'][i * int(tr_set_len) : (i + di) * int(tr_set_len), 1] * (np.pi) / 180
        x = np.expand_dims(np.sin(Polar) * np.cos(Azimut), axis=1)
        y = np.expand_dims(np.sin(Polar) * np.sin(Azimut), axis=1)
        z = np.expand_dims(np.cos(Polar), axis=1)
        target = torch.FloatTensor(np.concatenate((x,y,z) ,axis=1))
        Data = torch.FloatTensor(Data.swapaxes(1, -1)) 
        Dataset = torch.utils.data.TensorDataset(Data, target)
        Loader = torch.utils.data.DataLoader(dataset = Dataset, batch_size=Batch_size, drop_last = True)    
    return  Loader
