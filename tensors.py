import torch
import numpy as np
#innit tensors

#From data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

#From NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#From other tensors
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

#Tensor atributes:
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")