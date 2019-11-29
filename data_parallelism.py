#multiple GPU usage in pytorch
device = torch.device("cuda:0")
model.to(device)

#now create a new copy of tensor instead of rewriting existing; and we can assign it to a new tensor and use that new tensor on GPU
mytensor = my_tensor.to(device)

#pytorch normally run on one GPU
#we can make it run on multiple gpus parallelly using DataParallel
model = nn.Dataparallel(model)

#**********************************************************************************************************
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size =2

batch_size =30
data_size =100

device = torch.device("cuda:0 if torch.cuda.is_available() else "cpu")#pytorch has given us this possibility to create custom datasets and dsataloaders

#Dataset class
class RandomDataset(Dataset):
	def __init__(self,size,length):
		self.len = length
		self.data = torch.randn(length,size)

	def __getitem__(self,index):
		return self.data[index]

	def __len__(self):
		return self.len

#load data set
rand_loader = DataLoader(dataset = RandomDataset(input_size,data_size),batch_size=batch_size,shuffle-True)

#simple model
class Model(nn.Module):
	def __init__(self,input_size,output_size):
		super(Model,self).__init__()
		self.fc = nn.Linear(input_size,output_size)

	def forward(self,input):
		output = self.fc(input)
		print("Model:input size",input.size(),"output size",output.size())
		return output

model = Model(input_size, output_size)

#This is the core part of the tutorial. First, we need to make a model instance and check if we have multiple GPUs. 
#If we have multiple GPUs, we can wrap our model using nn.DataParallel. Then we can put our model on GPUs by model.to(device)
if torch.cuda.device_count()>1:
	print("let's use",torch.cuda.device_count(),"GPUs!")
	model = nn.DataParallel(model)

model.to(device)

'''
#multiple GPU usage in pytorch
device = torch.device("cuda:0")
model.to(device)
'''

#run the model with data set
for data in rand_loader:
	input = data.to(device)
	output = model(input)
	print("input size",input.size(),"output size",output.size())

#DataParallel splits your data automatically and sends job orders to multiple models on several GPUs. 
#After each model finishes their job, DataParallel collects and merges the results before returning it to you.

