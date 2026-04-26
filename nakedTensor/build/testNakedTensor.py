import nakedTensor
x = nakedTensor.Tensor1D([1.0, 2.0, 3.0])
print(x)
print(x.get(1))
print(x.numel())
print(hex(x.data_ptr()))