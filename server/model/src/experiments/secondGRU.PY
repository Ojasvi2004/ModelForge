print("hello")

import torch
from torch.nn import Linear
import sklearn
from datasets import Dataset
import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf

df=pd.read_csv('../datasets/sp500/sp500_stocks.csv')

df=df.sort_values(["symbol","date"])

#Feature Creation
df["return"] = df.groupby("symbol")["close"].pct_change()
df["ma20"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
df["volatility"] = (
    df.groupby("symbol")["return"]
      .transform(lambda x: x.rolling(20).std())
)

df = df.dropna()

stock_lengths = df.groupby("symbol").size()

valid_symbols = stock_lengths[stock_lengths >= 100].index

df = df[df["symbol"].isin(valid_symbols)]

df.isnull().sum()

train = df[df["date"] < "2022-01-01"].copy()

val = df[
    (df["date"] >= "2022-01-01") &
    (df["date"] < "2023-01-01")
].copy()

test = df[df["date"] >= "2023-01-01"].copy()
features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return",
    "ma20",
    "volatility"
]
seq_targets = [
    "open",
    "high",
    "low",
    "close"
]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train[features]=scaler.fit_transform(train[features])
val[features]=scaler.transform(val[features])
test[features]=scaler.transform(test[features])

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class StockDataset(Dataset):

    def __init__(self, df, features, targets, seq_len=60):

        self.seq_len = seq_len
        self.features = features
        self.targets = targets

        self.stock_data = {}
        self.indices = []

        for symbol, stock_df in df.groupby("symbol"):

            stock_df = stock_df.sort_values("date")

            X = stock_df[features].to_numpy(dtype=np.float32)
            Y = stock_df[targets].to_numpy(dtype=np.float32)

            self.stock_data[symbol] = (X, Y)

            for start in range(len(stock_df) - seq_len):
                self.indices.append((symbol, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        symbol, start = self.indices[idx]

        X, Y = self.stock_data[symbol]

        x = X[start:start + self.seq_len]
        y = Y[start + self.seq_len]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )
    

train_dataset = StockDataset(train, features, seq_targets, seq_len=60)
val_dataset = StockDataset(val, features, seq_targets, seq_len=60)
test_dataset = StockDataset(test, features, seq_targets, seq_len=60)


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class GRUModel(nn.Module):
    
    def __init__(self,input_size=8,hidden_size=128,output_size=4, num_layers=2):
        super().__init__()
        
        self.gru=nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc=nn.Linear(in_features=hidden_size,out_features=output_size)
        
    def forward(self,x):
        output,hn=self.gru(x)
        x=output[:,-1,:]
        
        return self.fc(x)
    
model = GRUModel().to(device)


criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

import os


os.makedirs("checkpoints", exist_ok=True)

epochs = 50

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(epochs):

    # ==========================
    # Training
    # ==========================
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True
    )

    for X_batch, y_batch in progress_bar:

        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(X_batch)

        loss = criterion(pred, y_batch)

        loss.backward()


        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ==========================
    # Validation
    # ==========================
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            pred = model(X_batch)

            loss = criterion(pred, y_batch)

            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.5f} | "
        f"Val Loss: {avg_val_loss:.5f}"
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }


    torch.save(
        checkpoint,
        "checkpoints/latest_checkpoint.pth"
    )


    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            checkpoint,
            "checkpoints/best_model.pth"
        )

        print(f"✓ New best model saved! Val Loss = {best_val_loss:.5f}")


history = pd.DataFrame({
    "epoch": range(1, epochs + 1),
    "train_loss": train_losses,
    "val_loss": val_losses
})

history.to_csv(
    "checkpoints/training_history.csv",
    index=False
)

print("Training Complete.")