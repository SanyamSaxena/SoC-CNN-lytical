import torch.nn as nn 

# define a child class of nn.Module for your model
# specify the architecture here itself

# 3 hidden layers  
# 200,100,50 sizes
# activation between them is relu
 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
         super(NeuralNet, self).__init__()
         self.input_size = input_size
         self.l1 = nn.Linear(input_size, hidden_size1) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidden_size1, hidden_size2)
         self.relu = nn.ReLU()
         self.l3 = nn.Linear(hidden_size2, hidden_size3)
         self.relu = nn.ReLU()
         self.l4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
         out = self.l1(x)
         out = self.relu(out)
         out = self.l2(out)
         out = self.relu(out)
         out = self.l3(out)
         out = self.relu(out)
         out = self.l4(out)
         
         # no softmax at the end since already included in cross entropy loss function
         return out

