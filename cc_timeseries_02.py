import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# Set random seed for reproducibility
random_state = 42

# Load and preprocess dataset
cc = pd.read_excel('cc_default.xls', header=1).dropna()
y = cc['default payment next month']
X_static = cc.drop(columns=['default payment next month'])

bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
seq_cols = bill_cols + pay_cols

scaler_seq = StandardScaler()
X_seq = scaler_seq.fit_transform(X_static[seq_cols])
X_seq = X_seq.reshape(-1, 2, 6)
X_seq = np.transpose(X_seq, (0, 2, 1))

X_static = X_static.drop(columns=seq_cols)
scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(X_static)

# PyTorch Dataset
class CreditDataset(Dataset):
    def __init__(self, X_static, X_seq, y):
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_static[idx], self.X_seq[idx], self.y[idx]

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, static_size, seq_input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_static = nn.Linear(static_size, 32)
        self.fc_combined = nn.Linear(32 + hidden_size, 1)

    def forward(self, x_static, x_seq):
        _, (h_n, _) = self.lstm(x_seq)
        x_static = torch.relu(self.fc_static(x_static))
        x = torch.cat((x_static, h_n[-1]), dim=1)
        return torch.sigmoid(self.fc_combined(x)).squeeze()

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, static_size, seq_input_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(seq_input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_static = nn.Linear(static_size, 32)
        self.fc_combined = nn.Linear(32 + d_model, 1)

    def forward(self, x_static, x_seq):
        x_seq = self.embedding(x_seq)
        x_seq = self.transformer(x_seq).mean(dim=1)
        x_static = torch.relu(self.fc_static(x_static))
        x = torch.cat((x_static, x_seq), dim=1)
        return torch.sigmoid(self.fc_combined(x)).squeeze()

# Training and evaluation
def train_model_cv(model_class, X_static, X_seq, y, n_splits=5, max_epochs=50, patience=5, model_name="Model"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_overall_auc = 0
    best_overall_metrics = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_static, y)):
        ros = RandomOverSampler(random_state=42)
        Xs_train, ys_train = ros.fit_resample(X_static[train_idx], y.iloc[train_idx])
        Xq_train = X_seq[train_idx][ros.sample_indices_]

        train_ds = CreditDataset(Xs_train, Xq_train, ys_train)
        val_ds = CreditDataset(X_static[val_idx], X_seq[val_idx], y.iloc[val_idx])

        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64)

        model = model_class(static_size=X_static.shape[1], seq_input_size=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        best_auc = 0
        best_cm = None
        best_recall = 0
        counter = 0

        for epoch in range(max_epochs):
            model.train()
            for xb_static, xb_seq, yb in train_dl:
                xb_static, xb_seq, yb = xb_static.to(device), xb_seq.to(device), yb.to(device)
                pred = model(xb_static, xb_seq)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for xb_static, xb_seq, yb in val_dl:
                    xb_static, xb_seq = xb_static.to(device), xb_seq.to(device)
                    pred = model(xb_static, xb_seq).cpu().numpy()
                    preds.extend(pred)
                    targets.extend(yb.numpy())

            preds_bin = np.array(preds) >= 0.5
            recall = recall_score(targets, preds_bin)
            auc = roc_auc_score(targets, preds)
            cm = confusion_matrix(targets, preds_bin)

            if auc > best_auc:
                best_auc = auc
                best_cm = cm
                best_recall = recall
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        if best_auc > best_overall_auc:
            best_overall_auc = best_auc
            best_overall_metrics = {
                "model": model_name,
                "fold": fold + 1,
                "auc": best_auc,
                "recall": best_recall,
                "cm": best_cm
            }



# Train LSTM and Transformer
train_model_cv(LSTMModel, X_static, X_seq, y, model_name="LSTM")
train_model_cv(TransformerModel, X_static, X_seq, y, model_name="Transformer")

