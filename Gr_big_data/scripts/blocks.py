import torch
import torch.nn as nn
POWER = 4
# nn.InstanceNorm1d nn.GroupNorm nn.LayerNorm

class Start_Block(nn.Module):
    def __init__(self,input_size, bias = True ):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv1d(input_size, 5,  kernel_size= 3 , stride = 1 ,padding= 1 ,bias = bias),
            nn.BatchNorm1d(5),
            nn.PReLU()
        )
    def forward(self, inputs):
        return  self.start_block(inputs) 


class Conv_Block(nn.Module):
    def __init__(self,input_size, output_size, ker = 3, bias = True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size= ker, stride=1, padding=1, bias = bias),
            nn.BatchNorm1d(num_features = output_size),
            nn.PReLU()
        )
    def forward(self, inputs):
        return  self.conv_block(inputs)     
    
        
class Triple_Res_block(nn.Module):
    def __init__(self,input_size, ker = 3, bias = True):
        super().__init__()
        self.module = nn.Sequential(    
        Conv_Block(input_size, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size, ker = ker),
          )
        self.conv = nn.Sequential( 
            nn.Conv1d(input_size, input_size, kernel_size =1,bias =  bias), 
            nn.PReLU()
        )
    def forward(self, inputs):
          return  (self.module(inputs) +self.conv(inputs))   

            
class Transition_Block(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.trans_module = nn.Sequential(
            nn.Conv1d(input_size, input_size*2, kernel_size=3, stride=2,padding=1 ), # ,bias = False
            nn.BatchNorm1d(num_features=input_size*2),
            nn.PReLU()
        )
    def forward(self, inputs):
        return  self.trans_module(inputs)         

    
class Full_Block(nn.Module):
    def __init__(self, input_size, ker = 3, amount = 2):
        super().__init__()
        self.blocks = nn.ModuleList([Triple_Res_block(input_size = input_size, ker = ker) for i in range(amount)])
        self.transition = Transition_Block(input_size)
    def forward(self, inputs):
        # ModuleList can act as an iterable, or be indexed using ints
        for block in self.blocks:
            inputs =  block(inputs)
        inputs = self.transition(inputs)
        return inputs 

    
class Dense_Block(nn.Module):
    def __init__(self, input_size, output_size ,dropblock = "no"):
        super().__init__()
        if dropblock == "no":
            self.dense_block = nn.Sequential(
                nn.Linear(input_size,input_size),
                nn.BatchNorm1d(input_size),
                nn.PReLU(),
                nn.Linear(input_size,input_size),
                nn.BatchNorm1d(input_size),
                nn.PReLU(),
                nn.Linear(input_size ,output_size)
            )
        elif dropblock == "before":
            self.dense_block = nn.Sequential(
                nn.Linear(input_size,input_size),
                nn.BatchNorm1d(input_size),
                nn.PReLU(),
                nn.Linear(input_size,input_size),
                nn.BatchNorm1d(input_size),
                nn.PReLU(),
                nn.Linear(input_size ,output_size)
            ) 
            
    def forward(self, inputs):
        return  self.dense_block(inputs)     

    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc1(y).view(b, c // self.reduction_ratio, 1)
        y = self.relu(y)
        y = self.fc2(y).view(b, c)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y
    
    
class Attention1D(nn.Module):
    def __init__(self, in_channels):
        super(Attention1D, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, l = x.size()
        query = self.query_conv(x).view(b, -1, l)  # B x C' x L
        key = self.key_conv(x).view(b, -1, l)  # B x C' x L
        energy = torch.bmm(query.transpose(1, 2), key)  # B x L x L
        attention = torch.softmax(energy, dim=-1)  # B x L x L
        value = self.value_conv(x).view(b, -1, l)  # B x C' x L
        out = torch.bmm(value, attention.transpose(1, 2))  # B x C' x L
        out = out.view(b, c, l)  # B x C x L
        out = self.gamma * out + x  # B x C x L
        return out