import torch
from blocks import POWER, Start_Block, Dense_Block, Conv_Block, Triple_Res_block, Transition_Block, Full_Block

def ResNet(device, amount = 2, input_size = 5, output_size  = 3):
    model= torch.nn.Sequential(    
    Start_Block(input_size),
        
    Full_Block(input_size = 5, amount = amount), # 32- 16
    Full_Block(input_size = 10, amount = amount), # 16 - 8 
    Full_Block(input_size = 20, amount = amount), # 8 - 4 
      
    Triple_Res_block(40),
    Triple_Res_block(40), # 4
        
    torch.nn.Flatten(),
    Dense_Block(160 ,output_size)
    ) 
    print("Parameters amount is ",sum(p.numel() for p in model.parameters()))
    return model.to(device)

def ResNet_drop_before(device, amount = 2, input_size = 5, output_size  = 3):
    model= torch.nn.Sequential(    
    Start_Block(input_size),
        
    Full_Block(input_size = 5, amount = amount), # 32- 16
    Full_Block(input_size = 10, amount = amount), # 16 - 8 
    Full_Block(input_size = 20, amount = amount), # 8 - 4 
      
    Triple_Res_block(40),
    Triple_Res_block(40), # 4
        
    torch.nn.Flatten(),
    Dense_Block_before(160 ,output_size)
    ) 
    print("Parameters amount is ",sum(p.numel() for p in model.parameters()))
    return model.to(device)

def ResNet_drop_after(device, amount = 2, input_size = 5, output_size  = 3):
    model= torch.nn.Sequential(    
    Start_Block(input_size),
        
    Full_Block(input_size = 5, amount = amount), # 32- 16
    Full_Block(input_size = 10, amount = amount), # 16 - 8 
    Full_Block(input_size = 20, amount = amount), # 8 - 4 
      
    Triple_Res_block(40),
    Triple_Res_block(40), # 4
        
    torch.nn.Flatten(),
    Dense_Block_after(160 ,output_size)
    ) 
    print("Parameters amount is ",sum(p.numel() for p in model.parameters()))
    return model.to(device)