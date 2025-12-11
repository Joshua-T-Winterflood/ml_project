import torch
import numpy as np

class Heart_Disease_NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 10
        self.num_feats = 10

        self.embedding = torch.nn.Linear(1, self.embed_dim)

        # Self-attention
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=1,
            batch_first=True
        )

        # Classifier
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_feats * self.embed_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # x: (batch, 10)
        x = x.unsqueeze(-1)               # (batch, 10, 1)
        x = self.embedding(x)             # (batch, 10, 10)

        attn_out, _ = self.attn(x, x, x)  # (batch, 10, 10)

        flat = attn_out.reshape(attn_out.size(0), -1)
        return self.mlp(flat)

    def fit(self, X, y, iterations=200):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

        for _ in range(iterations):
            y_pred = self(X)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self(X).squeeze().numpy()
        return (out >= 0.5).astype(int)

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self(X).squeeze().numpy()
        return np.vstack([1 - out, out]).T


