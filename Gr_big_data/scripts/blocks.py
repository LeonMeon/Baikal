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


class Conv_Block(torch.nn.Module):
    def __init__(self,input_size, output_size, ker = 3):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, output_size, kernel_size= ker, stride=1, padding=1, bias = False),
            torch.nn.BatchNorm1d(num_features = output_size),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        return  self.conv_block(inputs)     
    
        
class Triple_Res_block(torch.nn.Module):
    def __init__(self,input_size, ker = 3):
        super().__init__()
        self.module = torch.nn.Sequential(    
        Conv_Block(input_size, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size, ker = ker),
          )
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size, input_size, kernel_size =1,bias =  False), # TODO: bias =  True
            torch.nn.PReLU()
        )
    def forward(self, inputs):
          return  (self.module(inputs) +self.conv(inputs))   

            
class Transition_Block(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.trans_module = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, input_size*2, kernel_size=3, stride=2,padding=1 ), # ,bias = False
            torch.nn.BatchNorm1d(num_features=input_size*2),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
        return  self.trans_module(inputs)         

    
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
    
