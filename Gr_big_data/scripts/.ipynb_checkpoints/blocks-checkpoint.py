import torch
POWER = 4

class Start_Block(torch.nn.Module):
    def __init__(self,input_size ):
        super().__init__()
        self.start_block = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 5,  kernel_size= 3 , stride = 1 ,padding= 1 ,bias = False),
            torch.nn.BatchNorm1d(5),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        return  self.start_block(inputs) 
    
class Transition_Block(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.trans_module = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, input_size*2, kernel_size=3, stride=2,padding=1 ),
            torch.nn.BatchNorm1d(num_features=input_size*2),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        return  self.trans_module(inputs) 

class Conv_Block(torch.nn.Module):
    def __init__(self,input_size, output_size, ker = 3):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, output_size, kernel_size= ker, stride=1, padding=1, bias = False),
            torch.nn.BatchNorm1d(num_features= output_size),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        return  self.conv_block(inputs)     
    
class Double_Res_block(torch.nn.Module):
    def __init__(self,input_size, ker = 3):
        super().__init__()
        self.module = torch.nn.Sequential(    
        Conv_Block(input_size, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size, ker = ker) 
        )
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size, input_size, kernel_size =1,bias =  True),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
          return  (self.module(inputs) +self.conv(inputs))    
    
    
class Triple_Res_block(torch.nn.Module):
    def __init__(self,input_size, ker = 3):
        super().__init__()
        self.module = torch.nn.Sequential(    
        Conv_Block(input_size, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size, ker = ker),
          )
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size, input_size, kernel_size =1,bias =  False),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
          return  (self.module(inputs) +self.conv(inputs))   
        
class Full_Block(torch.nn.Module):
    def __init__(self, input_size, ker = 3, amount = 2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Triple_Res_block(input_size = input_size, ker = ker) for i in range(amount)])
        self.transition = Transition_Block(input_size)
    def forward(self, inputs):
        # ModuleList can act as an iterable, or be indexed using ints
        for block in self.blocks:
            inputs =  block(inputs)
        inputs = self.transition(inputs)
        return inputs        
######################################################            EXPERIMENTS       ######################################################
class ResNet_Block_power_inception(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.module = Triple_Res_block(input_size)
        self.module5x5 =Triple_Res_block(input_size , ker =5) 
        self.unite = Conv_Block(input_size*2, input_size, ker = 1)
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size, input_size, kernel_size =1,bias =  True),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        x3 = self.module(inputs)
        x5 = self.module5x5(inputs)
        x_unite = self.unite(torch.cat([x3,x5], dim=1) )
        return  (x_unite +self.conv(inputs))
    
class Res_conv_block(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size*power, input_size*power, kernel_size =1,bias =  bias_mask),
            torch.nn.PReLU())        
    def forward(self, inputs):
          return  self.conv(inputs) 
class ResNet_Block_power_v3(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.start = ResNet_Block_IN(input_size)
        self.end = ResNet_Block_OUT(input_size)
        self.inter_module_1 = Inter_block(input_size)
        self.inter_module_2 = Inter_block(input_size)
        self.inter_module_3 = Inter_block(input_size)
        self.res_1_2 , self.res_2_3, self.res_1_3 = Res_conv_block(input_size), Res_conv_block(input_size), Res_conv_block(input_size)
    def forward(self, inputs):
        x_1 = self.start(inputs)
        x_inter_1 = self.inter_module_1(x_1)
        x_inter_2 = self.inter_module_2(x_inter_1)+ self.res_1_2(x_1)
        x_inter_3 = self.inter_module_3(x_inter_2) + self.res_2_3(x_inter_1) + self.res_1_3(x_1) 
        #x_out = self.end(x_inter_3)
        return  x_inter_3 
    
class Dense_Block(torch.nn.Module):
    def __init__(self, input_size, output_size ,dropblock = "no"):
        super().__init__()
        if dropblock == "no":
            self.dense_block = torch.nn.Sequential(
                torch.nn.Linear(input_size,input_size),
                torch.nn.BatchNorm1d(input_size),
                torch.nn.PReLU(),
                torch.nn.Linear(input_size,input_size),
                torch.nn.BatchNorm1d(input_size),
                torch.nn.PReLU(),
                torch.nn.Linear(input_size ,output_size)
            )
        elif dropblock == "before":
            self.dense_block = torch.nn.Sequential(
                torch.nn.Linear(input_size,input_size),
                torch.nn.BatchNorm1d(input_size),
                torch.nn.PReLU(),
                torch.nn.Linear(input_size,input_size),
                torch.nn.BatchNorm1d(input_size),
                torch.nn.PReLU(),
                torch.nn.Linear(input_size ,output_size)
            ) 
            
    def forward(self, inputs):
        return  self.dense_block(inputs) 