import torch
from torch import nn

#device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
    torch.manual_seed(42) # it can change for new possibility
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float), requires_grad=True)
    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)