import torch

print("Initializing Tensor a")
a = torch.Tensor([1, 2, 3]) # here we're initializing 'a' as an object of Tensor class.
# we've sent a python list of numbers as the argument in this class.
# Tensor class is a wrapper over C++ codebase.
# when i checked Tensor class there was NO def __init__ there - quite suprising for me!
print(a)
print(a.device) # these are properties on the c++ tensor object and NOT constructor 
# arguments of the python Tensor class.
print(a.dtype)
print(type(a))

print("Initializing tensor b")
b = torch.tensor([1, 2, 3])
print(b)
print(b.device)
print(b.dtype)
print(type(b))
# from intuitive sense the difference b/w torch.Tensor and torch.tensor is that 
# in Tensor we're directly creating the object, while in the case of tensor this function 
# creates the object for us. 
# pytorch's tensor is not a list with methods, it is a pointer to memory and rules to interpret memory.

print(torch.tensor)
print(torch.tensor.__module__)