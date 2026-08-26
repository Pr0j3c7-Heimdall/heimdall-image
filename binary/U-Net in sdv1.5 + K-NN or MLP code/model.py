import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class TorchKNN:
    def __init__(self, n_neighbors=20, device='cuda'):
        self.k = n_neighbors
        self.device = device
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # 입력된 Tensor를 GPU로 이동
        self.X_train = X.clone().detach().float().to(self.device)
        self.y_train = y.clone().detach().long().to(self.device)
        
        mean = self.X_train.mean(dim=1, keepdim=True)
        std = self.X_train.std(dim=1, keepdim=True)
        self.X_train = (self.X_train - mean) / (std + 1e-8)

    def predict_proba(self, X):
        X_test = X.clone().detach().float().to(self.device)
        mean = X_test.mean(dim=1, keepdim=True)
        std = X_test.std(dim=1, keepdim=True)
        X_test = (X_test - mean) / (std + 1e-8)
        
        prob_list = []
        batch_size = 1000 
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                X_batch = X_test[i:i+batch_size]
                similarity = torch.mm(X_batch, self.X_train.t())
                _, indices = torch.topk(similarity, self.k, dim=1)
                
                k_labels = self.y_train[indices]
                prob_batch = k_labels.float().mean(dim=1)
                prob_list.extend(prob_batch.cpu().numpy())
                
        return np.array(prob_list)

    def save(self, filepath):
        torch.save({'X_train': self.X_train.cpu(), 'y_train': self.y_train.cpu(), 'k': self.k}, filepath)
        
    def load(self, filepath):
        data = torch.load(filepath, map_location=self.device)
        self.X_train = data['X_train'].to(self.device)
        self.y_train = data['y_train'].to(self.device)
        self.k = data['k']

class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=640, output_dim=2):
        super(TorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def fit(self, X, y, epochs=1000, lr=3e-4, batch_size=32, patience=15):
        # scikit-learn 분할을 위해 잠시 numpy 변환 후 다시 텐서화
        X_np, y_np = X.numpy(), y.numpy()
        X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        
        best_acc = 0.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            self.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                val_outputs = self.net(X_v)
                _, predicted = torch.max(val_outputs, 1)
                val_acc = (predicted == y_v).float().mean().item()
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_acc*100:.2f}%")
                break
                
        if best_state is not None:
            self.load_state_dict(best_state)

    def predict_proba(self, X):
        self.eval()
        X_tensor = X.clone().detach().float().to(self.device)
        with torch.no_grad():
            outputs = self.net(X_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        return probs

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))

def get_model(model_name, input_dim=None, **kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'knn':
        return TorchKNN(n_neighbors=kwargs.get('n_neighbors', 101), device=device)
    elif model_name.startswith('mlp'):
        if input_dim is None: raise ValueError("MLP requires input_dim")
        return TorchMLP(input_dim, hidden_dim=640).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")