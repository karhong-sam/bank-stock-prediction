import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the model and move it to the GPU
model = SimpleNN().to('cuda')

# Create a random input tensor and move it to the GPU
input_tensor = torch.randn(64, 784).to('cuda')

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
output = model(input_tensor)

# Compute loss
target = torch.randint(0, 10, (64,)).to('cuda')  # Random target
loss = criterion(output, target)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Finished training step on GPU")
