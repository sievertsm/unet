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
    
class UModel(nn.Module):
    '''
    Combines a basic block, downsample block, and an upsample block to create the U-net architecture
    Input:  
      input_channels -- number of input channels
      num_classes    -- number of output classes
      depth          -- how many layers deep to create the network
      first_layer    -- number of channels in the first layer

    Output: 
      x -- a tensor after completeing the forward pass through the network
    '''
    def __init__(self, input_channels=1, num_classes=2, depth=5, first_layer=64, basic_block=BasicBlock, down_block=DownBlock, up_block=UpBlock):
        super().__init__()

        self._input_channels = input_channels
        self._num_classes = num_classes
        self._depth = depth
        self._first_layer = first_layer
    
        # going down
        # define down channels
        pow, _ = self._next_pow2(first_layer)
        channels_down = [(input_channels, first_layer)]
        channels_down += [(2**(pow+i), 2**(pow+i+1)) for i in range((depth-2))]

        # define down blocks
        self.down_blocks=[]
        for c0, c1 in channels_down:
            self.down_blocks.append(down_block(c0, c1))
        self.down_blocks = torch.nn.ModuleList(self.down_blocks)
        
        # define bottleneck
        c0 = channels_down[-1][1]
        _, c1 = self._next_pow2(c0+1)
        self.bottle = basic_block(c0, c1)

        # going up
        # define up channels
        channels_up = [(c1, c0)]
        channels_up += [(c0, c1) for c1, c0 in channels_down[1:][::-1]]

        # define up blocks
        self.up_blocks=[]
        for c0, c1 in channels_up:
            self.up_blocks.append(up_block(c0, c1))
        self.up_blocks = torch.nn.ModuleList(self.up_blocks)
        
        # final batch normalization and 1x1 convolution to match input
        c0 = channels_up[-1][1]
        c1 = num_classes

        self.bn_finish = nn.BatchNorm2d(c0)
        self.finish = nn.Conv2d(c0, c1, kernel_size=1)
        
    def name(self):
        return f"{self.__class__.__name__}-{self._input_channels}-{self._num_classes}-{self._depth}-{self._first_layer}"

    def _next_pow2(self, x):
        pow=0
        while 2**pow < x:
            pow+=1
        return pow, 2**pow
        
    def forward(self, x):

        # downsample
        skips=[]
        for block in self.down_blocks:
            x, s = block(x)
            skips.append(s)
        
        # middle layers
        x = self.bottle(x)
        
        # upsample with skip connections
        skips = skips[::-1]
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[i])
        
        # reshape with 1x1
        x = self.bn_finish(x)
        x = self.finish(x)
        
#         x = self._activation(x)
        
        return x