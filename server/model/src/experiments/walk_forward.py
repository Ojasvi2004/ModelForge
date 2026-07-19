"""
Advanced Quantitative Multi-Asset Pipeline (v3) -- Walk-Forward Validated

Changes from original v3:
  - Switched from standard Adam to AdamW optimizer.
  - Reduced default sequence model learning rate to 1e-4 (slower, more stable gradient descent).
  - Raised weight decay penalty to 1e-2 to strictly regularize parameter growth and solve the Epoch 1 overfitting loop.
"""

print("Starting Advanced Quantitative Multi-Asset Pipeline (v3) -- Walk-Forward Validated")

import os
import json
import time
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

BASE_ARTIFACT_DIR = "./artifacts"
FOLD_ARTIFACT_ROOT = "./artifacts_walkforward"
os.makedirs(BASE_ARTIFACT_DIR, exist_ok=True)
os.makedirs(FOLD_ARTIFACT_ROOT, exist_ok=True)

SEQ_LEN = 60
EPOCHS = 25
PATIENCE = 4
MAX_GRAD_NORM = 1.0
TRANSACTION_COST_BPS = 10
TOP_N = 20
MIN_TRAIN_YEARS = 2      # minimum calendar years of history required before the first fold
MAX_FOLDS = None         # set an int to cap the number of folds while iterating quickly; None = all available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. DATA LOADING & FEATURE ENGINEERING (done ONCE, split-independent)
# ==========================================
def engineer_features(csv_path):
    print("Loading raw CSV data...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(["symbol", "date"])
    df["date"] = pd.to_datetime(df["date"])

    print("Engineering technical features and multi-horizon targets...")
    g = df.groupby("symbol")

    df["daily_return"] = g["close"].pct_change()
    df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
    df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["ma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
    df["close_to_ma5"] = df["close"] / (df["ma5"] + 1e-8)
    df["close_to_ma20"] = df["close"] / (df["ma20"] + 1e-8)
    df["close_to_ma50"] = df["close"] / (df["ma50"] + 1e-8)
    df["ma5_to_ma20"] = df["ma5"] / (df["ma20"] + 1e-8)

    df["high_to_low"] = df["high"] / (df["low"] + 1e-8)
    df["close_to_open"] = df["close"] / (df["open"] + 1e-8)

    df["volatility"] = g["daily_return"].transform(lambda x: x.rolling(20).std())
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["true_range"] = tr
    df["atr14"] = df.groupby("symbol")["true_range"].transform(lambda x: x.rolling(14).mean())
    df["atr_pct"] = df["atr14"] / (df["close"] + 1e-8)

    bb_std20 = g["close"].transform(lambda x: x.rolling(20).std())
    df["bollinger_pctb"] = (df["close"] - df["ma20"]) / (2 * bb_std20 + 1e-8)

    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = (ema12 - ema26) / (df["close"] + 1e-8)
    df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["roc5"] = g["close"].pct_change(5)
    df["roc10"] = g["close"].pct_change(10)
    df["roc20"] = g["close"].pct_change(20)

    df["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-8)
    df["volume_z"] = df.groupby("symbol")["volume"].transform(
        lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
    )

    df["prev_close"] = g["close"].shift(1)
    df["gap"] = df["open"] / (df["prev_close"] + 1e-8) - 1.0

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs + 1e-8))

    df["rsi"] = df.groupby("symbol")["close"].transform(lambda x: calculate_rsi(x))

    market_return = df.groupby("date")["daily_return"].transform("mean")
    df["market_return"] = market_return
    df["relative_return"] = df["daily_return"] - df["market_return"]

    for col in ["daily_return", "rsi", "volume_ratio", "roc10"]:
        df[f"{col}_xrank"] = df.groupby("date")[col].rank(pct=True)

    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5.0)

    horizon_days = {"target_1d": 1, "target_5d": 5, "target_10d": 10, "target_20d": 20}
    for name, h in horizon_days.items():
        if h == 1:
            df[name] = g["daily_return"].shift(-1)
        else:
            df[name] = g["close"].shift(-h) / (df["close"] + 1e-8) - 1.0

    df = df.dropna()
    stock_lengths = df.groupby("symbol").size()
    valid_symbols = stock_lengths[stock_lengths >= 150].index
    df = df[df["symbol"].isin(valid_symbols)]

    return df, list(horizon_days.keys()), max(horizon_days.values())


SEQ_FEATURES = [
    "daily_return", "close_to_ma5", "close_to_ma20", "close_to_ma50", "ma5_to_ma20",
    "high_to_low", "close_to_open", "volatility", "atr_pct", "bollinger_pctb",
    "macd", "macd_signal", "macd_hist", "roc5", "roc10", "roc20",
    "volume_ratio", "volume_z", "gap", "rsi", "relative_return",
    "daily_return_xrank", "rsi_xrank", "volume_ratio_xrank", "roc10_xrank",
    "dow_sin", "dow_cos"
]
CURR_FEATURES = ["daily_return", "volatility", "volume_ratio", "rsi", "relative_return", "roc10_xrank"]

# ==========================================
# 2. DATASET
# ==========================================
class MultiHorizonStockDataset(Dataset):
    def __init__(self, df, seq_features, curr_features, targets, seq_len=60):
        self.seq_len = seq_len
        self.symbols = df["symbol"].to_numpy()
        self.dates = df["date"].astype(str).to_numpy()
        self.X_seq = df[seq_features].to_numpy(dtype=np.float32)
        self.X_curr = df[curr_features].to_numpy(dtype=np.float32)
        self.Y = df[targets].to_numpy(dtype=np.float32)

        self.indices = []
        symbol_indices = {}
        for idx, sym in enumerate(self.symbols):
            symbol_indices.setdefault(sym, []).append(idx)
        for sym, idxs in symbol_indices.items():
            if len(idxs) > seq_len:
                for start_pos in range(len(idxs) - seq_len):
                    self.indices.append(idxs[start_pos])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x_seq = self.X_seq[start:start + self.seq_len].copy()
        x_curr = self.X_curr[start + self.seq_len - 1].copy()
        y = self.Y[start + self.seq_len - 1].copy()
        symbol = self.symbols[start + self.seq_len - 1]
        date = self.dates[start + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.from_numpy(x_curr), torch.from_numpy(y), symbol, date


# ==========================================
# 3. MODEL
# ==========================================
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=32, num_heads=4, num_layers=1, dropout=0.3):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        out = out.transpose(1, 2)
        return self.pooling(out).squeeze(-1)


class UnifiedDeepEncoder(nn.Module):
    def __init__(self, seq_features_dim, static_features_dim, embed_dim=48, hidden_dim=32, horizons=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_features_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gru = nn.GRU(input_size=seq_features_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        transformer_model_dim = 32
        self.transformer = TemporalTransformerEncoder(
            input_dim=seq_features_dim, model_dim=transformer_model_dim, num_heads=4, dropout=dropout
        )
        self.trans_proj = nn.Linear(transformer_model_dim, hidden_dim)
        self.project_embedding = nn.Linear(hidden_dim * 3, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + static_features_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, horizons)
        )

    def forward(self, x_seq, x_curr, return_embedding_only=False):
        lstm_out, _ = self.lstm(x_seq)
        h_lstm = lstm_out[:, -1, :]
        gru_out, _ = self.gru(x_seq)
        h_gru = gru_out[:, -1, :]
        h_trans = self.trans_proj(self.transformer(x_seq))
        h_concat = torch.cat([h_lstm, h_gru, h_trans], dim=-1)
        embedding = self.embed_dropout(self.project_embedding(h_concat))
        if return_embedding_only:
            return embedding
        predictions = self.mlp(torch.cat([embedding, x_curr], dim=-1))
        return predictions, embedding


class HybridHorizonLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        ic_loss = 0.0
        horizons = preds.shape[1]
        for h in range(horizons):
            p, t = preds[:, h], targets[:, h]
            p_cov, t_cov = p - p.mean(), t - t.mean()
            num = torch.sum(p_cov * t_cov)
            denom = torch.sqrt(torch.sum(p_cov ** 2) * torch.sum(t_cov ** 2) + 1e-8)
            ic_loss += (1.0 - num / denom)
        ic_loss = ic_loss / horizons
        return self.alpha * mse_loss + (1.0 - self.alpha) * ic_loss


def compute_ic_per_horizon(preds, targets):
    ics = []
    for h in range(preds.shape[1]):
        ic, _ = stats.spearmanr(preds[:, h], targets[:, h])
        ics.append(float(ic) if not np.isnan(ic) else 0.0)
    return ics


def max_drawdown(returns):
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def summarize_returns(returns, label):
    arr = np.array(returns, dtype=np.float64)
    cum = float(np.prod(1.0 + arr) - 1.0)
    mean = float(np.mean(arr))
    vol = float(np.std(arr) + 1e-8)
    sharpe = float((mean / vol) * np.sqrt(252))
    dd = max_drawdown(arr)
    print(f"  {label}: CumRet {cum*100:.2f}% | Sharpe {sharpe:.3f} | MaxDD {dd*100:.2f}% | Days {len(arr)}")
    return {"cumulative_return": cum, "sharpe": sharpe, "max_drawdown": dd, "days": len(arr)}


# ==========================================
# 4. FOLD BOUNDARY CONSTRUCTION
# ==========================================
def build_walkforward_folds(df, min_train_years=2, max_folds=None):
    """
    Expanding-window walk-forward folds:
      train : everything before val_year (embargoed at the boundary)
      val   : calendar year `val_year`      (used for early stopping + meta-weights)
      test  : calendar year `val_year + 1`  (never touched until final evaluation)
    Then val_year advances by 1 and the window expands. Each fold's train set
    is a strict superset of the previous fold's, mimicking how you'd actually
    retrain a live model over time as more history accumulates.
    """
    years = sorted(df["date"].dt.year.unique())
    min_year, max_year = years[0], years[-1]
    first_val_year = min_year + min_train_years

    folds = []
    val_year = first_val_year
    while val_year + 1 <= max_year:
        folds.append({
            "name": f"train_lt_{val_year}_val_{val_year}_test_{val_year+1}",
            "val_year": val_year,
            "test_year": val_year + 1,
            "train_end": pd.Timestamp(f"{val_year}-01-01"),
            "val_end": pd.Timestamp(f"{val_year+1}-01-01"),
            "test_end": pd.Timestamp(f"{val_year+2}-01-01"),
        })
        val_year += 1

    if max_folds is not None:
        folds = folds[-max_folds:]  # keep the most recent N folds if capped
    return folds


# ==========================================
# 5. SINGLE-FOLD TRAIN/EVAL
# ==========================================
def run_fold(df, fold, targets, max_horizon, fold_idx, n_folds):
    print(f"\n{'='*80}\nFOLD {fold_idx+1}/{n_folds}: {fold['name']}\n{'='*80}")

    embargo = pd.Timedelta(days=max_horizon * 2)
    train_df = df[df["date"] < (fold["train_end"] - embargo)].copy()
    val_df = df[(df["date"] >= fold["train_end"]) & (df["date"] < (fold["val_end"] - embargo))].copy()
    test_df = df[(df["date"] >= fold["val_end"]) & (df["date"] < fold["test_end"])].copy()

    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    if len(train_df) < 10000 or len(val_df) < 500 or len(test_df) < 500:
        print("Skipping fold: insufficient rows after embargo/filtering.")
        return None

    all_feature_cols = list(dict.fromkeys(SEQ_FEATURES + CURR_FEATURES))
    scaler = StandardScaler()
    train_df[all_feature_cols] = scaler.fit_transform(train_df[all_feature_cols])
    val_df[all_feature_cols] = scaler.transform(val_df[all_feature_cols])
    test_df[all_feature_cols] = scaler.transform(test_df[all_feature_cols])

    train_dataset = MultiHorizonStockDataset(train_df, SEQ_FEATURES, CURR_FEATURES, targets, seq_len=SEQ_LEN)
    val_dataset = MultiHorizonStockDataset(val_df, SEQ_FEATURES, CURR_FEATURES, targets, seq_len=SEQ_LEN)
    test_dataset = MultiHorizonStockDataset(test_df, SEQ_FEATURES, CURR_FEATURES, targets, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    fold_dir = os.path.join(FOLD_ARTIFACT_ROOT, fold["name"])
    os.makedirs(fold_dir, exist_ok=True)

    model = UnifiedDeepEncoder(
        seq_features_dim=len(SEQ_FEATURES), static_features_dim=len(CURR_FEATURES),
        embed_dim=48, hidden_dim=32, horizons=len(targets), dropout=0.4
    ).to(device)
    criterion = HybridHorizonLoss(alpha=0.5)
    
    # ADJUSTED: Switched to AdamW with reduced LR (1e-4) and elevated weight decay (1e-2) to prevent instant overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    ckpt_path = os.path.join(fold_dir, "best_hybrid_deep.pth")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"[{fold['name']}] Epoch {epoch+1}/{EPOCHS}")
        for x_seq, x_curr, y, _, _ in progress:
            x_seq, x_curr, y = x_seq.to(device), x_curr.to(device), y.to(device)
            optimizer.zero_grad()
            preds, _ = model(x_seq, x_curr)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for x_seq, x_curr, y, _, _ in val_loader:
                x_seq, x_curr, y = x_seq.to(device), x_curr.to(device), y.to(device)
                preds, _ = model(x_seq, x_curr)
                running_val_loss += criterion(preds, y).item()
                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(y.cpu().numpy())
        avg_val_loss = running_val_loss / len(val_loader)
        val_preds_np = np.concatenate(all_val_preds, axis=0)
        val_targets_np = np.concatenate(all_val_targets, axis=0)
        ics = compute_ic_per_horizon(val_preds_np, val_targets_np)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        print(f"  Val IC -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, ics)))
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(ckpt_path))

    def extract(loader):
        model.eval()
        embeddings, curr_feats, labels, symbols, dates = [], [], [], [], []
        with torch.no_grad():
            for x_seq, x_curr, y, syms, dts in tqdm(loader, desc="Extracting"):
                embeds = model(x_seq.to(device), x_curr.to(device), return_embedding_only=True)
                embeddings.append(embeds.cpu().numpy())
                curr_feats.append(x_curr.numpy())
                labels.append(y.numpy())
                symbols.extend(syms)
                dates.extend(dts)
        return (np.concatenate(embeddings), np.concatenate(curr_feats), np.concatenate(labels),
                np.array(symbols), np.array(dates))

    E_train, C_train, Y_train, sym_train, date_train = extract(train_loader)
    E_val, C_val, Y_val, sym_val, date_val = extract(val_loader)
    E_test, C_test, Y_test, sym_test, date_test = extract(test_loader)

    X_tree_train = np.ascontiguousarray(np.hstack([E_train, C_train]))
    X_tree_val = np.ascontiguousarray(np.hstack([E_val, C_val]))
    X_tree_test = np.ascontiguousarray(np.hstack([E_test, C_test]))

    horizon_tree_models = {h: {} for h in range(len(targets))}
    for h_idx, h_name in enumerate(targets):
        y_tr, y_vl = Y_train[:, h_idx], Y_val[:, h_idx]
        if HAS_XGB:
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                  n_jobs=-1, random_state=42, early_stopping_rounds=20)
            m.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)], verbose=False)
            horizon_tree_models[h_idx]['xgb'] = m
        if HAS_LGB:
            m = lgb.LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                   n_jobs=-1, random_state=42, verbose=-1)
            m.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
            horizon_tree_models[h_idx]['lgb'] = m
        if HAS_CAT:
            m = cb.CatBoostRegressor(iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=5.0,
                                      random_seed=42, verbose=0, early_stopping_rounds=20)
            m.fit(X_tree_train, y_tr, eval_set=(X_tree_val, y_vl))
            horizon_tree_models[h_idx]['cat'] = m

    model.eval()
    mlp_val_preds = []
    with torch.no_grad():
        for x_seq, x_curr, _, _, _ in val_loader:
            preds, _ = model(x_seq.to(device), x_curr.to(device))
            mlp_val_preds.append(preds.cpu().numpy())
    Y_pred_val_mlp = np.concatenate(mlp_val_preds, axis=0)

    meta_weights = {}
    for h_idx in range(len(targets)):
        cols = [Y_pred_val_mlp[:, h_idx]]
        order = ["mlp"]
        for key in ["xgb", "lgb", "cat"]:
            if key in horizon_tree_models[h_idx]:
                cols.append(horizon_tree_models[h_idx][key].predict(X_tree_val))
                order.append(key)
        val_pred_matrix = np.column_stack(cols)
        meta_model = Ridge(alpha=5.0, fit_intercept=False)
        meta_model.fit(val_pred_matrix, Y_val[:, h_idx])
        w = meta_model.coef_
        if w.sum() > 0:
            w = w / w.sum()
        meta_weights[h_idx] = {"weights": w.tolist(), "order": order}

    mlp_test_preds = []
    with torch.no_grad():
        for x_seq, x_curr, _, _, _ in test_loader:
            preds, _ = model(x_seq.to(device), x_curr.to(device))
            mlp_test_preds.append(preds.cpu().numpy())
    Y_pred_test_mlp = np.concatenate(mlp_test_preds, axis=0)

    Y_test_final_preds = np.zeros_like(Y_test)
    for h_idx in range(len(targets)):
        cols = [Y_pred_test_mlp[:, h_idx]]
        for key in ["xgb", "lgb", "cat"]:
            if key in horizon_tree_models[h_idx]:
                cols.append(horizon_tree_models[h_idx][key].predict(X_tree_test))
        w = np.array(meta_weights[h_idx]["weights"])
        Y_test_final_preds[:, h_idx] = np.dot(np.column_stack(cols), w[:len(cols)])

    test_ics = compute_ic_per_horizon(Y_test_final_preds, Y_test)
    print("Test IC -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, test_ics)))

    backtest_df = pd.DataFrame({
        "date": date_test, "symbol": sym_test,
        "pred_signal": Y_test_final_preds[:, 1],
        "actual_return_1d": Y_test[:, 0]
    })
    unique_days = sorted(backtest_df["date"].unique())
    gross_rets, net_rets = [], []
    prev_longs, prev_shorts = set(), set()
    for day in unique_days:
        day_df = backtest_df[backtest_df["date"] == day]
        if len(day_df) < 50:
            continue
        sorted_df = day_df.sort_values("pred_signal", ascending=False)
        longs = set(sorted_df.head(TOP_N)["symbol"])
        shorts = set(sorted_df.tail(TOP_N)["symbol"])
        long_ret = sorted_df[sorted_df["symbol"].isin(longs)]["actual_return_1d"].mean()
        short_ret = sorted_df[sorted_df["symbol"].isin(shorts)]["actual_return_1d"].mean()
        gross_ret = (long_ret - short_ret) / 2.0
        long_turnover = len(longs.symmetric_difference(prev_longs)) / (2 * TOP_N)
        short_turnover = len(shorts.symmetric_difference(prev_shorts)) / (2 * TOP_N)
        turnover = (long_turnover + short_turnover) / (2 * TOP_N) if TOP_N > 0 else 0.0 # safety division fix
        cost = turnover * (TRANSACTION_COST_BPS / 10000.0)
        gross_rets.append(gross_ret)
        net_rets.append(gross_ret - cost)
        prev_longs, prev_shorts = longs, shorts

    if len(gross_rets) == 0:
        print("No tradeable days in this fold's test period.")
        gross_stats, net_stats = None, None
    else:
        gross_stats = summarize_returns(gross_rets, "Gross")
        net_stats = summarize_returns(net_rets, f"Net of {TRANSACTION_COST_BPS}bps")

    # Persist this fold's artifacts (useful for audit; not all folds are "deployed")
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
    for h_idx in range(len(targets)):
        for key, m in horizon_tree_models[h_idx].items():
            joblib.dump(m, os.path.join(fold_dir, f"{key}_h{h_idx}.joblib"))
    with open(os.path.join(fold_dir, "meta_weights.json"), "w") as f:
        json.dump(meta_weights, f, indent=2)
    with open(os.path.join(fold_dir, "feature_config.json"), "w") as f:
        json.dump({"seq_features": SEQ_FEATURES, "curr_features": CURR_FEATURES,
                    "targets": targets, "seq_len": SEQ_LEN}, f, indent=2)

    return {
        "fold_name": fold["name"],
        "val_year": fold["val_year"],
        "test_year": fold["test_year"],
        "test_ic_by_horizon": dict(zip(targets, test_ics)),
        "gross": gross_stats,
        "net": net_stats,
        "fold_dir": fold_dir,
    }


# ==========================================
# 6. MAIN: RUN ALL FOLDS SEQUENTIALLY, THEN AGGREGATE
# ==========================================
def main():
    start_time = time.time()
    df, targets, max_horizon = engineer_features('../datasets/sp500/sp500_stocks.csv')
    folds = build_walkforward_folds(df, min_train_years=MIN_TRAIN_YEARS, max_folds=MAX_FOLDS)

    if not folds:
        raise RuntimeError(
            "No valid walk-forward folds could be built -- your data may span too few years. "
            "Need at least MIN_TRAIN_YEARS + 2 years of history."
        )

    print(f"\nBuilt {len(folds)} walk-forward folds:")
    for f in folds:
        print(f"  {f['name']}")

    all_results = []
    for i, fold in enumerate(folds):
        result = run_fold(df, fold, targets, max_horizon, i, len(folds))
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("Every fold was skipped (insufficient data). Nothing to report.")

    # ---- Aggregate across folds ----
    print(f"\n{'='*80}\nWALK-FORWARD SUMMARY ACROSS {len(all_results)} FOLDS\n{'='*80}")

    ic_table = {t: [] for t in targets}
    sharpe_gross, sharpe_net, cumret_net = [], [], []
    for r in all_results:
        print(f"\n{r['fold_name']} (test year {r['test_year']}):")
        for t in targets:
            ic = r["test_ic_by_horizon"][t]
            ic_table[t].append(ic)
            print(f"    {t}: IC {ic:+.4f}", end="")
        print()
        if r["gross"]:
            print(f"    Gross Sharpe: {r['gross']['sharpe']:.3f} | Net Sharpe: {r['net']['sharpe']:.3f} "
                  f"| Net CumRet: {r['net']['cumulative_return']*100:.2f}%")
            sharpe_gross.append(r["gross"]["sharpe"])
            sharpe_net.append(r["net"]["sharpe"])
            cumret_net.append(r["net"]["cumulative_return"])

    print("\n--- Aggregated Test IC by Horizon (mean +/- std across folds) ---")
    agg_ic = {}
    for t in targets:
        vals = np.array(ic_table[t])
        agg_ic[t] = {"mean": float(vals.mean()), "std": float(vals.std()), "per_fold": vals.tolist()}
        print(f"  {t}: {vals.mean():+.4f} +/- {vals.std():.4f}  (folds: {[f'{v:+.3f}' for v in vals]})")

    agg_summary = {"test_ic_by_horizon": agg_ic}
    if sharpe_net:
        sg, sn, cr = np.array(sharpe_gross), np.array(sharpe_net), np.array(cumret_net)
        print(f"\n--- Aggregated Backtest (mean +/- std across folds) ---")
        print(f"  Gross Sharpe: {sg.mean():.3f} +/- {sg.std():.3f}  (folds: {[f'{v:.2f}' for v in sg]})")
        print(f"  Net Sharpe:   {sn.mean():.3f} +/- {sn.std():.3f}  (folds: {[f'{v:.2f}' for v in sn]})")
        print(f"  Net CumRet:   {cr.mean()*100:.2f}% +/- {cr.std()*100:.2f}%  per fold-year")
        agg_summary["gross_sharpe"] = {"mean": float(sg.mean()), "std": float(sg.std()), "per_fold": sg.tolist()}
        agg_summary["net_sharpe"] = {"mean": float(sn.mean()), "std": float(sn.std()), "per_fold": sn.tolist()}
        agg_summary["net_cumulative_return"] = {"mean": float(cr.mean()), "std": float(cr.std()), "per_fold": cr.tolist()}

    with open(os.path.join(FOLD_ARTIFACT_ROOT, "walkforward_summary.json"), "w") as f:
        json.dump({"folds": [
            {k: v for k, v in r.items() if k != "fold_dir"} for r in all_results
        ], "aggregate": agg_summary}, f, indent=2)

    # ---- Promote the FINAL (most recent) fold's artifacts as the deployable model ----
    final_fold_dir = all_results[-1]["fold_dir"]
    print(f"\nPromoting final fold's artifacts ({final_fold_dir}) to {BASE_ARTIFACT_DIR}/ for deployment...")
    for fname in os.listdir(final_fold_dir):
        src = os.path.join(final_fold_dir, fname)
        dst = os.path.join(BASE_ARTIFACT_DIR, fname)
        if os.path.isfile(src):
            with open(src, "rb") as fs, open(dst, "wb") as fd:
                fd.write(fs.read())

    elapsed = time.time() - start_time
    print(f"\n--- Execution Complete in {elapsed/60:.1f} minutes ---")
    print("\nHOW TO READ THIS:")
    print(" - If test IC per horizon is consistently positive and doesn't swing wildly")
    print("   fold-to-fold, that is meaningfully more trustworthy than a single split.")
    print(" - If any fold shows near-zero or negative IC, that year's regime broke the")
    print("   model -- look at what happened in that year before trusting this in production.")
    print(" - Net Sharpe std across folds tells you how much performance varies by regime;")
    print("   a high mean with high std means it's regime-dependent, not a stable edge.")
    print(f" - Deployable artifacts (from the most recent fold) are in {BASE_ARTIFACT_DIR}/")
    print(f" - Full per-fold artifacts/audit trail are in {FOLD_ARTIFACT_ROOT}/")


if __name__ == "__main__":
    main()