import sys
print(sys.executable)

###############

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

###############

# Create 1000x1000 matrix on GPU
x = torch.rand(1000, 1000).cuda()
y = torch.rand(1000, 1000).cuda()

z = torch.matmul(x, y)

print("Computation finished on:", "GPU" if z.is_cuda else "CPU")

###############
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Simple model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

model = Net().to(device)

# Example input batch: 32 images of 784 features
inputs = torch.randn(32, 784).to(device)
outputs = model(inputs)

print("Output on:", "GPU" if outputs.is_cuda else "CPU")