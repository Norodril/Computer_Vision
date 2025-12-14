import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 5

# Bayesian component/ bayesian layer
class BayesianConv2d(nn.Module):
    """
    Weights are represented as distributions (mu, rho).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Rho ensure standard deviation is always positive: sigma = log(1 + exp(rho))
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3.0) # Start with low uncertainty
        
        if self.use_bias:
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        if self.training:
            weight_epsilon = torch.randn_like(weight_sigma)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            if self.use_bias:
                bias_epsilon = torch.randn_like(bias_sigma)
                bias = self.bias_mu + bias_sigma * bias_epsilon
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl = 0.5 * torch.sum(weight_sigma**2 + self.weight_mu**2 - 1 - torch.log(weight_sigma**2))
        
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += 0.5 * torch.sum(bias_sigma**2 + self.bias_mu**2 - 1 - torch.log(bias_sigma**2))
            
        return kl

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = BayesianConv2d(32, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_kl_loss(self):
        return self.conv1.kl_loss() + self.conv2.kl_loss()

# Training function implementing Bayes by Backpropogation
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        # 1. Negative Log Likelihood 
        nll_loss = F.cross_entropy(output, target, reduction='sum')
        
        # 2. KL Divergence (Complexity cost)
        kl_loss = model.get_kl_loss()
        
        # 3. ELBO Loss = NLL + Beta * KL
        # Beta serves as a weighting factor (often 1/num_batches)
        beta = 1.0 / len(train_loader)
        loss = nll_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}\tKL: {kl_loss.item():.6f}')

# Uncertainty estimation
def predict_with_uncertainty(model, device, data, num_samples=10):
    model.train() # Keep dropout/stochasticity active
    preds = []
    
    # Run the same image through the network multiple times
    # Because weights are random samples, output changes slightly every time
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(data.unsqueeze(0).to(device))
            preds.append(F.softmax(output, dim=1).cpu().numpy())
            
    preds = np.array(preds) 
    
    # Mean prediction is the result
    mean_pred = np.mean(preds, axis=0)
    # Variance represents Model uncertainty
    variance = np.var(preds, axis=0)
    
    return mean_pred, variance

if __name__ == "__main__":
    # Load MNIST Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

    model = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training with Bayes by Backprop...")
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)

    test_data = datasets.MNIST('./data', train=False, transform=transform)[0][0]
    mean, var = predict_with_uncertainty(model, DEVICE, test_data)
    
    print("\n--- Prediction Analysis ---")
    print(f"Predicted Class: {np.argmax(mean)}")
    print(f"Uncertainty (Variance sum): {np.sum(var):.4f}") 