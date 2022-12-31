from torchvision import datasets, transforms
import torch

torch.manual_seed(0)


image = torch.randn((3,3,3))
for i in range(9):
    image[0,int(i/3),i%3] = i
print(image[0])
first_row = torch.clone(image[:,0,:])
image[:,0:-1,:] = image[:,1:,:]
image[:,-1,:] = first_row
first_col = torch.clone(image[:,:,0])
image[:,:,0:-1] = image[:,:,1:]
image[:,:,-1] = first_col
print(image[0])
