import numpy as np
import torch
# logistic regression by pytorch, add a l1-regularization term
# it also has a weight, each feature is penalized by a different weight
class WeightedLasso(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha = 0.1, weight = None, lr = 0.001, epochs = 3000):
        super(WeightedLasso, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.weight = weight
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        if self.weight is not None:
            self.weight = torch.tensor(self.weight)

    def fit(self, X, y):
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        y = y.reshape(-1,1)
        for epoch in range(self.epochs):
            inputs = torch.from_numpy(X).float()
            labels = torch.from_numpy(y).float()
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            if self.weight is None:
                loss += self.alpha * torch.sum(torch.abs(self.linear.weight))
            else:
                loss += self.alpha * torch.sum(self.weight * torch.abs(self.linear.weight))
            loss.backward()
            optimizer.step()

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def coef_(self):
        return self.linear.weight.detach().numpy().flatten()
    
    def predict(self, X):
        inputs = torch.from_numpy(X).float()
        outputs = self(inputs)
        # return 0 1 labels
        return (outputs > 0).float().detach().numpy().flatten()
    
