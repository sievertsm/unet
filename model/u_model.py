'''
Definition of the UModel architecture allowing for different 
number of channels at each layer.

https://arxiv.org/pdf/1505.04597.pdf
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    Basic building block for the U-net architecture consisting of:
    [batch normalization -> conv -> relu] x 2
    '''
    def __init__(self, Cin, Cout, kernel_size=3, padding=1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=Cin),
            nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=Cout),
            nn.Conv2d(in_channels=Cout, out_channels=Cout, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        
        return self.block(x)


class DownBlock(nn.Module):
    '''
    Builds upon the the basic block by creating a skip connection,
    and downsampling with max pooling
    '''
    def __init__(self, Cin, Cout, block=BasicBlock):
        super().__init__()
        
        self.block = block(Cin, Cout)
        
    def forward(self, x):
        
        x = self.block(x)
        skip = x.clone()
        x = F.max_pool2d(x, 2, 2)
        
        return (x, skip)
    
class UpBlock(nn.Module):
    '''
    Builds upon the basic block by performing a conv transpose operation to upsample, 
    and includes a skip connection in the forward pass.
    '''
    def __init__(self, Cin, Cout, block=BasicBlock, kernel_size=4, stride=2, padding=1, cat=True):
        super().__init__()
        
        Cmid = 2*Cout if cat else Cout
        
        self.up = nn.ConvTranspose2d(Cin, Cout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.block = block(Cmid, Cout)
        self.cat = cat
        
    def forward(self, x, skip):
        
        x = self.up(x)
        
        if self.cat:
            x = torch.cat((x, skip), dim=1)
        else:
            x += skip
            
        x = self.block(x)
        
        return x


class UModel_Denoise(nn.Module):
    '''
    Combines a basic block, downsample block, and an upsample block to create the U-net architecture

    Input:  channels a tuple length 6 containing the channels to be used at each layer
                beginning with the channels in the original image

    Output: x a tensor after completeing the forward pass through the network

    '''
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024), basic_block=BasicBlock, down_block=DownBlock, up_block=UpBlock):
        super().__init__()
        
        # get channel information
        c0, c1, c2, c3, c4, c5 = channels
    
        # define blocks for going down
        self.block1_down = down_block(c0, c1)
        self.block2_down = down_block(c1, c2)
        self.block3_down = down_block(c2, c3)
        self.block4_down = down_block(c3, c4)
        
        # define bottleneck
        self.bottle = basic_block(c4, c5)
        
        # define up blocks
        self.block5_up = up_block(c5, c4)
        self.block6_up = up_block(c4, c3)
        self.block7_up = up_block(c3, c2)
        self.block8_up = up_block(c2, c1)
        
        # final batch normalization and 1x1 convolution to match input
        self.bn_finish = nn.BatchNorm2d(c1)
        self.finish = nn.Conv2d(c1, 1, kernel_size=1)
        
    def forward(self, x):

        # downsample
        x, skip1 = self.block1_down(x)
        x, skip2 = self.block2_down(x)
        x, skip3 = self.block3_down(x)
        x, skip4 = self.block4_down(x)
        
        # middle layers
        x = self.bottle(x)
        
        # upsample with skip connections
        x = self.block5_up(x, skip4)
        x = self.block6_up(x, skip3)
        x = self.block7_up(x, skip2)
        x = self.block8_up(x, skip1)
        
        # reshape with 1x1
        x = self.bn_finish(x)
        x = self.finish(x)
        
        return x
    

class UModel_Segment(nn.Module):
    '''
    Combines a basic block, downsample block, and an upsample block to create the U-net architecture

    Input:  channels a tuple length 7 containing the channels to be used at each layer
                beginning with the channels in the original image, and ending with the number of classes

    Output: x a tensor after completeing the forward pass through the network

    '''
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024, 1), basic_block=BasicBlock, down_block=DownBlock, up_block=UpBlock):
        super().__init__()
        
        # get channel information
        c0, c1, c2, c3, c4, c5, cf = channels
    
        # define blocks for going down
        self.block1_down = down_block(c0, c1)
        self.block2_down = down_block(c1, c2)
        self.block3_down = down_block(c2, c3)
        self.block4_down = down_block(c3, c4)
        
        # define bottleneck
        self.bottle = basic_block(c4, c5)
        
        # define up blocks
        self.block5_up = up_block(c5, c4)
        self.block6_up = up_block(c4, c3)
        self.block7_up = up_block(c3, c2)
        self.block8_up = up_block(c2, c1)
        
        # final batch normalization and 1x1 convolution to match input
        self.bn_finish = nn.BatchNorm2d(c1)
        self.finish = nn.Conv2d(c1, cf, kernel_size=1)
        
    def forward(self, x):

        # downsample
        x, skip1 = self.block1_down(x)
        x, skip2 = self.block2_down(x)
        x, skip3 = self.block3_down(x)
        x, skip4 = self.block4_down(x)
        
        # middle layers
        x = self.bottle(x)
        
        # upsample with skip connections
        x = self.block5_up(x, skip4)
        x = self.block6_up(x, skip3)
        x = self.block7_up(x, skip2)
        x = self.block8_up(x, skip1)
        
        # reshape with 1x1
        x = self.bn_finish(x)
        x = self.finish(x)
        
        return x
    