import torch
POWER = 4

######################################################            EXPERIMENTS       ######################################################

class ResBlock_Inception(torch.nn.Module):
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
    
'''       
class DenseConvBlock(torch.nn.Module):
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
'''    
    
#############################################################################3
class SEBlock(torch.nn.Module):
    '''
    https://arxiv.org/pdf/1709.01507.pdf
    
    In short, it's attention in CNN
    
    Use global pooling and NN to get weight of each channel
    
    Understand the importance of each feature map in the stack of all the feature maps  after a convolution 
    and recalibrates that output to reflect that importance before passing the volume to the next layer.
    '''
    def __init__(self, out_channels, reduction_ratio):
        super().__init__()
        
        if out_channels < reduction_ratio :
            reduction_ratio = out_channels
            
        # Global pooling
        # mb use meanpool
        def _squeeze(x) :
            return max_pool3d(x, kernel_size=x.shape[2:])
        
        def _scaling(x, scale) :
            reps  = list(x.shape)
            reps[0] = reps[1] = 1
            scale.repeat(reps)
            x = x * scale
            return x

        self.squeeze = _squeeze
        self.scaling = _scaling
        #use an attention mechanism using a gating network
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels // reduction_ratio, out_channels),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale.squeeze())
        scale = torch.reshape(scale, scale.shape)
        x = self.scaling(x, scale)        
        return x    
################################################################################################

    
class Double_Res_block(torch.nn.Module):
    def __init__(self,input_size, ker = 3):
        super().__init__()
        self.module = torch.nn.Sequential(    
        Conv_Block(input_size, input_size * POWER, ker = ker),
        Conv_Block(input_size * POWER, input_size, ker = ker) 
        )
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv1d(input_size, input_size, kernel_size =1, bias =  True),
            torch.nn.PReLU()
        )
    def forward(self, inputs):
          return  (self.module(inputs) +self.conv(inputs))