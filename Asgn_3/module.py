import torch.nn as nn 

import torch.nn as nn 

# define a child class of nn.Module for your model
# specify the architecture here itself
class NeuralNet(nn.Module):

    def __init__(self,num_classes):
         super(NeuralNet, self).__init__()
         self.l1=   nn.Conv2d(3, 32, kernel_size=3, padding=1)#32 filters dimension remains 32 since stride is 1 by default
         self.l2=   nn.ReLU()
         self.l3=   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#64 filters dimension remains 32 since stride is 1 by default
         self.l4=   nn.ReLU()
         self.l5=   nn.MaxPool2d(2, 2) # output: 64 x 8 x 8
         self.l6=   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
         self.l7=   nn.ReLU()
         self.l8=   nn.BatchNorm2d(128)
         self.l9=   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
         self.l10=   nn.ReLU()
         self.l11=   nn.MaxPool2d(2, 2) # output: 128 x 8 x 8
         self.l19=   nn.Flatten() 
         self.l20=   nn.Linear(128*8*8, 1024)
         self.l21=   nn.ReLU()
         self.l22=   nn.Linear(1024, 256)
         self.l23=   nn.ReLU()
         self.l24=   nn.Linear(256, num_classes)
                
    def forward(self, x):
         out = self.l1(x)
         out = self.l2(out)
         out = self.l3(out)
         out = self.l4(out)
         out = self.l5(out)
         out = self.l6(out)
         out = self.l7(out)
         out = self.l8(out)
         out = self.l9(out)
         out = self.l10(out)
         out = self.l11(out)
         out = self.l19(out)
         out = self.l20(out)
         out = self.l21(out)
         out = self.l22(out)
         out = self.l23(out)
         out = self.l24(out)
         
         # no softmax at the end since already included in cross entropy loss function
         return out

