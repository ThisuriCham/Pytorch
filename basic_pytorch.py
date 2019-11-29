# pytorch is a replace ment of numpy 
# DL research platform which provides maximum flexibility and speed

#tensors are similar to numpy ndarrays
#advantage than numpy is thesde can be used on a GPU to accelerate computing

from __future__ import print_function
import torch
import numpy

x = torch.empty(5,3)
print(x)

y = torch.rand(5,3)
print(y)

z = torch.zeros(5,3, dtype=torch.long)
print(z)

n = torch.tensor([5.5,3])
print(n)

x= x.new_ones(4,3,dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype =torch.float64)
print(x)
print(x.size())
print(y+z)
print(torch.add(y,z))

result = torch.empty(5,3)
torch.add(y,z, out=result)
print(result)

y.add_(z)
print(y)

print(x[:,1]) #indexing like in numpy

x = torch.randn(1)
y =x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())

x =torch.randn(1)
print(x)
print(x.item())

#converting tensor to numpy
# we called this numpy bridging

a = torch.ones(5)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

#converting numpy to tensor
a = np.ones(5)
b= torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#CUDA tensors
#tensors can be moved onto any device



