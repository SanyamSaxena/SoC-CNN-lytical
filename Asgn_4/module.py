import torch
import torch.nn as nn
import torch.nn.functional as F 

# define a child class of nn.Module for your model
# specify the architecture here itself
# if necessary, make submodules in different cells for structured code

def double_conv(in_c,out_c):
  conv=nn.Sequential(
      nn.Conv2d(in_c, out_c, kernel_size=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_c, out_c, kernel_size=3),
      nn.ReLU(inplace=True),   
  )
  return conv

def crop_img(tensor,target_tensor):
  target_size_x=target_tensor.size()[2]
  tensor_size_x=tensor.size()[2]

  target_size_y=target_tensor.size()[3]
  tensor_size_y=tensor.size()[3]
  
  delta_x=tensor_size_x - target_size_x
  delta_x= delta_x//2
  
  delta_y=tensor_size_y - target_size_y
  delta_y= delta_y//2
  if ((tensor_size_x - target_size_x)%2==1) and ((tensor_size_y - target_size_y)%2==1):
    return tensor[:,:,delta_x:tensor_size_x-delta_x-1,delta_y:tensor_size_y-delta_y-1]
  elif ((tensor_size_x - target_size_x)%2==0) and ((tensor_size_y - target_size_y)%2==1):
    return tensor[:,:,delta_x:tensor_size_x-delta_x,delta_y:tensor_size_y-delta_y-1]
  elif ((tensor_size_x - target_size_x)%2==1) and ((tensor_size_y - target_size_y)%2==0):
    return tensor[:,:,delta_x:tensor_size_x-delta_x-1,delta_y:tensor_size_y-delta_y-1]
  else:
    return tensor[:,:,delta_x:tensor_size_x-delta_x,delta_y:tensor_size_y-delta_y]

def padding(x2,x1):
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]

  x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
  return x1

class UNet(nn.Module):

    def __init__(self):
         super(UNet, self).__init__()

         self.max_pool=   nn.MaxPool2d(2, 2)
         self.down_conv_1=  double_conv(3,16)
         self.down_conv_2=  double_conv(16,32)
         self.down_conv_3=  double_conv(32,64)
         self.up_trans_3 = nn.ConvTranspose2d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=2,
                                              stride=2)
         self.up_conv_3= double_conv(64,32)
         
         self.up_trans_4 = nn.ConvTranspose2d(in_channels=32,
                                              out_channels=16,
                                              kernel_size=2,
                                              stride=2)
         self.up_conv_4= double_conv(32,16)
         
         self.out= nn.Conv2d(
             in_channels=16,
             out_channels=1,
             kernel_size=1
         )
                
    def forward(self, img):
         #encoder
         x1= self.down_conv_1(img)#
         x2=self.max_pool(x1)
         x3= self.down_conv_2(x2)#
         x4=self.max_pool(x3)
         x5= self.down_conv_3(x4)#
                  
         x = self.up_trans_3(x5)
         x = padding(x3,x)
         x=self.up_conv_3(torch.cat([x,x3],1))
         
         x = self.up_trans_4(x)
         x=padding(x1,x)
         x=self.up_conv_4(torch.cat([x,x1],1))
         x=padding(img,x)
         
         x=self.out(x)
         return x  
