#import numpy as np
#import uproot as ur
#import h5py as h5
#import matplotlib.pyplot as plt
#import scipy.sparse as sparse

#from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import sys
sys.path.append("/hom/scripts")
from datasets import Dataset_Polar_Azimut
from GNN_models import GCN_easy , GCN_model
from fits import fit_graph_model
from resolution import polar_vec_res_hist

root_path = '/home/leonov/Baikal/Gr_big_data/graphs/all_data/'
path_to_h5 =  "/home/leonov/Baikal/Gr_big_data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"
dataset = Dataset_Polar_Azimut(root = root_path)

regime_inds = dataset.regime_inds
train_dataset = dataset[:regime_inds[0]]
test_dataset = dataset[regime_inds[0] : regime_inds[1]]
val_dataset = dataset[regime_inds[1] : regime_inds[2]]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


