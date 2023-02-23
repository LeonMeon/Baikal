import torch
import torch.nn as nn

def RNN(device,  hidden_size = 32, input_size = 5, output_size  = 3):
    model=  nn.Sequential(    
    nn.Conv1d(5, 40,  kernel_size= 3 , stride = 1 ,padding= 1,bias = False ),# 32*8
    nn.BatchNorm1d(40),
    nn.PReLU(),
    nn.Conv1d(40, 1,  kernel_size= 3 , stride = 1 ,padding= 1,bias = False ),# 32*8
    nn.BatchNorm1d(1),
    nn.PReLU(),
    nn.Flatten(),
    nn.RNNCell(input_size  =32,hidden_size = hidden_size , bias=True),
    nn.RNNCell(input_size  =hidden_size,hidden_size = hidden_size , bias=True),   
    nn.Linear(hidden_size, output_size)
    ) 
    print("Parameters amount is ",sum(p.numel() for p in model.parameters()))
    return model.to(device)


def GRU(device, hidden_size = 32, output_size = 2):
    model = nn.Sequential(    
    nn.Conv1d(5, 10,  kernel_size= 3 , stride = 1 ,padding= 1 ,bias = False),
    nn.BatchNorm1d(10),
    nn.PReLU(),
    nn.Conv1d(10, 1,  kernel_size= 3 , stride = 1 ,padding= 1,bias = False ),# 32*8
    nn.BatchNorm1d(1),
    nn.PReLU(),
    nn.Flatten(),
    nn.GRUCell(input_size  =32,hidden_size = hidden_size , bias=True),
    nn.GRUCell(input_size  =hidden_size, hidden_size = hidden_size , bias=True),   
    nn.GRUCell(input_size  =hidden_size, hidden_size = hidden_size , bias=True), 
    nn.Linear(hidden_size, output_size)
    )
        
    print("Parameters amount is ",sum(p.numel() for p in model.parameters()))
    return model.to(device) 
###############################################################################################

H_out = H_cell
L = 32

Bi_dir = False
D = 2 if Bi_dir else 1


h0 = torch.randn(D * Num_Layers, Batch_size, H_cell).to(device)
c0 = torch.randn(D * Num_Layers, Batch_size, H_out).to(device)
class LSTM_net(nn.Module):
    def __init__(self, num_layers = 5, H_in = 5, H_cell = 5):
        super().__init__()
        #self.Lstm_list = [nn.LSTM(H_in, H_cell, Num_Layers, batch_first = True) for i in range(3)]
        self.Lstm_1 = nn.LSTM(5, H_cell, Num_Layers, batch_first = True)
        self.Lstm_2 = nn.LSTM(H_cell, 5, Num_Layers, batch_first = True)
        self.conv_and_dense = nn.Sequential(   
            #nn.Conv1d(5, 5,  kernel_size= 3 , stride = 1 ,padding= 1,bias = False ),# 32*8
            #nn.BatchNorm1d(5),
            #nn.PReLU(),
            nn.Flatten(),
            nn.Linear(160,160),
            nn.BatchNorm1d(160),
            nn.Dropout(p = 0.3),
            nn.PReLU(),
            nn.Linear(160,160),
            nn.BatchNorm1d(160),
            nn.Dropout(p = 0.3),
            nn.PReLU(),
            nn.Linear(160 ,2))
    def forward(self, inputs):
        output = inputs
        cn, hn = c0, h0
        output ,(cn,hn) = self.Lstm_1(output, (cn,hn))
        #print(output.shape ,(cn.shape,hn.shape))
        output ,(cn,hn) = self.Lstm_2(output, (cn,hn))
        output = self.conv_and_dense(output)
        return output
    
################################################################################################

H_out = H_cell
L = 32
Num_Layers = 5
Bi_dir = False




class Lstm_net(nn.Module):
    def __init__(self, H_in = 5, H_cell = 5, Batch_size = 128, Bi_dir = False):
        super().__init__()
        #
        D = 2 if Bi_dir else 1
        h0 = torch.randn(D * Num_Layers, Batch_size, H_cell).to(device)
        c0 = torch.randn(D * Num_Layers, Batch_size, H_out).to(device)
        #
        #self.Lstm_list = [nn.LSTM(H_in, H_cell, Num_Layers, batch_first = True) for i in range(3)]
        self.Lstm_1 = nn.LSTM(5, H_cell, Num_Layers, batch_first = True)
        self.Lstm_2 = nn.LSTM(H_cell, 5, Num_Layers, batch_first = True)
        self.conv_and_dense = nn.Sequential(   
            #nn.Conv1d(5, 5,  kernel_size= 3 , stride = 1 ,padding= 1,bias = False ),# 32*8
            #nn.BatchNorm1d(5),
            #nn.PReLU(),
            nn.Flatten(),
            nn.Linear(160,160),
            nn.BatchNorm1d(160),
            nn.Dropout(p = 0.3),
            nn.PReLU(),
            nn.Linear(160,160),
            nn.BatchNorm1d(160),
            nn.Dropout(p = 0.3),
            nn.PReLU(),
            nn.Linear(160 ,2))
    def forward(self, inputs):
        output = inputs
        cn, hn = c0, h0
        output ,(cn,hn) = self.Lstm_1(output, (cn,hn))
        #print(output.shape ,(cn.shape,hn.shape))
        output ,(cn,hn) = self.Lstm_2(output, (cn,hn))
        output = self.conv_and_dense(output)
        return output