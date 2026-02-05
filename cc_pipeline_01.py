import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV

from cc_timeseries import LSTMModel, TransformerModel

# Load and preprocess
df = pd.read_excel('cc_default.xls', header=1).dropna()
y = df['default payment next month']
X = df.drop(columns='default payment next month')

bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
seq_cols = bill_cols + pay_cols
static_cols = [col for col in X.columns if col not in seq_cols]

scaler_static = StandardScaler()
scaler_seq = StandardScaler()

X_static = scaler_static.fit_transform(X[static_cols])
X_seq = scaler_seq.fit_transform(X[seq_cols])
X_seq = X_seq.reshape(-1, 2, 6).transpose(0, 2, 1)

# Split data
X_static_train, X_static_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_static, X_seq, y, test_size=0.2, stratify=y, random_state=42
)

# Oversample
ros = RandomOverSampler(random_state=42)
X_static_train_ros, y_train_ros = ros.fit_resample(X_static_train, y_train)
X_seq_train_ros = X_seq_train[ros.sample_indices_]

# Torch Dataset and training utils
class CreditDataset(Dataset):
    def __init__(self, X_static, X_seq, y):
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_static[idx], self.X_seq[idx], self.y[idx]

def train(model, train_dl, val_dl, max_epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    best_auc = 0
    best_model = None
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

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb_static, xb_seq, yb in val_dl:
                xb_static, xb_seq = xb_static.to(device), xb_seq.to(device)
                pred = model(xb_static, xb_seq).cpu().numpy()
                preds.extend(pred)
                targets.extend(yb.numpy())

        auc = roc_auc_score(targets, preds)
        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_model)
    return model

def evaluate_model(model, X_static_test, X_seq_test, y_test, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_static_tensor = torch.tensor(X_static_test, dtype=torch.float32).to(device)
        X_seq_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
        y_pred_proba = model(X_static_tensor, X_seq_tensor).cpu().numpy()
        y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n{label} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"AUC Score ({label}): {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Most common values where the positive class is predicted ({label}):")
    print(pd.Series(y_pred_proba).value_counts().head())

    return y_pred_proba

# Train & Evaluate Deep Models
train_ds = CreditDataset(X_static_train_ros, X_seq_train_ros, y_train_ros)
val_ds = CreditDataset(X_static_test, X_seq_test, y_test)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)

lstm_model = LSTMModel(static_size=X_static.shape[1], seq_input_size=2)
lstm_model = train(lstm_model, train_dl, val_dl)
y_proba_lstm = evaluate_model(lstm_model, X_static_test, X_seq_test, y_test, "LSTM")

transformer_model = TransformerModel(static_size=X_static.shape[1], seq_input_size=2)
transformer_model = train(transformer_model, train_dl, val_dl)
y_proba_transformer = evaluate_model(transformer_model, X_static_test, X_seq_test, y_test, "Transformer")

# Classical Models
X_classical = pd.DataFrame(X[static_cols], columns=static_cols)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_classical, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = X_train_raw.copy()
X_test_scaled = X_test_raw.copy()
X_train_scaled[['LIMIT_BAL', 'AGE']] = scaler.fit_transform(X_train_scaled[['LIMIT_BAL', 'AGE']])
X_test_scaled[['LIMIT_BAL', 'AGE']] = scaler.transform(X_test_scaled[['LIMIT_BAL', 'AGE']])

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

probas = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train_raw)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\n{name} Confusion Matrix:")
    print(confusion_matrix(y_test_raw, y_pred))
    print(classification_report(y_test_raw, y_pred))
    print(f"AUC Score ({name}): {roc_auc_score(y_test_raw, y_proba):.4f}")
    recall = recall_score(y_test_raw, y_pred, pos_label=1)
    print(f"Recall ({name}): {recall:.4f}")
    probas[name] = y_proba

# Plot ROC Curve
plt.figure(figsize=(10, 6))
for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test_raw, y_proba)
    plt.plot(fpr, tpr, label=name)

fpr, tpr, _ = roc_curve(y_test, y_proba_lstm)
plt.plot(fpr, tpr, label='LSTM')
fpr, tpr, _ = roc_curve(y_test, y_proba_transformer)
plt.plot(fpr, tpr, label='Transformer')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend()
plt.show()

