import torch
from torch_geometric.data import InMemoryDataset #, download_url
from tqdm import tqdm
import numpy as np
import h5py as h5


# here are some notes for you to understand. 
class Dataset_Polar_Azimut(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
                
    # returns the dataset source file name 
    @property
    def regime_lengths(self):
        return [1365465, 133807, 134335]
    
    @property
    def regime_inds(self):
        return np.cumsum(self.regime_lengths)
    
    #  names in case of downloading from internet
    @property
    def raw_file_names(self):
        #start = '/home/leonov/Baikal/Gr_big_data/graphs/all_data/raw'
        #return [f'{start}/all_raw_data.pt']
        return None
              
    # returns the save file name required by the process method 。 the name of the dataset you saved later is the same as that in the list 
    @property
    def processed_file_names(self):
        start = '/home/leonov/Baikal/Gr_big_data/datasets/all_data/processed/'
        return [f'{start}/all_data.pt']
        #return [f'{start}/{regime}_data.pt' for regime in ['train','test','val']]
    
    '''
    # used to download datasets from the internet 
    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...
    '''
    
    @property
    def path_to_h5(self):
        return "/home/leonov/Baikal/Gr_big_data/data/mc_baikal_norm_cut-8_ordered_with_MCarlo.h5"
    
    # make big COO format
    @property
    def edges(self):    
        k, l = 2, 35 # only for k_neighbours = 2 , i think no more is not needed
        rows, cols = [], []       
        for i in range(l):
            if i ==0: 
                rows += [i] * k
                cols += [ind for ind in range(k)]
            elif i == l - 1:
                rows += [i] * k
                cols += [ind for ind in range(l - k, l)]
            else: 
                rows += [i] * (k + 1)
                cols += [i-1,i,i+1] 
        return torch.cat((torch.tensor(rows)[None,:], torch.tensor(cols)[None,:]),0)

    def make_graph(self, ind, regime = 'train'):
        with h5.File(self.path_to_h5, 'r') as hf:
            n_nodes = int(np.sum(hf[f"{regime}/mask"][ind], axis =-1))
            n_edges = 3* n_nodes - 2

            edge_indexes = self.edges[:,:n_edges]
            # t, Q, x,y,z ( not sure about first two order)
            # data as [cases_length, 35,5]
            x = torch.FloatTensor(hf[regime + '/data/'][ind,:n_nodes])
            polar = torch.FloatTensor([hf[regime + '/ev_chars'][ind,0] * (np.pi) / 180])[None,:]
            azimut = torch.FloatTensor([hf[regime + '/ev_chars'][ind,1] * (np.pi) / 180])[None,:]
            # only polar
            y_polar = torch.cat((torch.sin(polar), torch.cos(polar)), axis=1)
            # only azimut
            y_azimut = torch.cat((torch.sin(azimut), torch.cos(azimut)), axis=1)
            # direction vector
            v_x = np.expand_dims(np.sin(polar) * np.cos(azimut), axis=1)
            v_y = np.expand_dims(np.sin(polar) * np.sin(azimut), axis=1)
            v_z = np.expand_dims(np.cos(polar), axis=1)
            direction = torch.FloatTensor(np.concatenate((v_x, v_y, v_z), axis=1))

        return Data(x = x, edge_index = edge_indexes,
                    polar = polar, azimut = azimut,
                    y_polar = y_polar, y_azimut = y_azimut,
                    direction = direction)  
    
    # the method used to generate the dataset 
    def process(self):
        data_list = []
        # Read data into huge `Data` list.
        for i, regime in tqdm(enumerate(['train', 'test', 'val'])):
            data_list += [self.make_graph(_ind, regime) for _ind in range(self.regime_lengths[i]) ] #self.regime_lengths[0]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])