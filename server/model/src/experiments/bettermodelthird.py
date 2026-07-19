# """
# Advanced Quantitative Multi-Asset Pipeline (v2)

# Changes from v1:
#   - Fixed train/val/test boundary leakage via embargo purging
#   - Added ~20 additional engineered features (MACD, Bollinger %B, ATR, ROC,
#     multi-window moving averages, cross-sectional daily rank/z-score,
#     market-relative return, calendar encoding)
#   - Per-horizon Information Coefficient (IC) logged separately from loss
#     at every epoch, so you can see real predictive signal, not just a
#     blended loss number
#   - Cost-aware backtest (configurable bps transaction costs + turnover)
#   - Saves all artifacts (scaler, model weights, tree models, meta-weights,
#     feature list) to ./artifacts/ so they can be loaded by a serving API

# IMPORTANT: this is a research pipeline. It does not, by itself, prove a
# strategy is profitable. See the printed backtest section and the caveats
# at the bottom of this file before treating any output as tradeable.
# """

# print("Starting Advanced Quantitative Multi-Asset Pipeline (v2)")

# import os
# import json
# import joblib
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from tqdm import tqdm

# try:
#     import xgboost as xgb
#     HAS_XGB = True
# except ImportError:
#     HAS_XGB = False

# try:
#     import lightgbm as lgb
#     HAS_LGB = True
# except ImportError:
#     HAS_LGB = False

# try:
#     import catboost as cb
#     HAS_CAT = True
# except ImportError:
#     HAS_CAT = False

# ARTIFACT_DIR = "./artifacts"
# os.makedirs(ARTIFACT_DIR, exist_ok=True)

# # ==========================================
# # 1. DATA LOADING & FEATURE ENGINEERING
# # ==========================================
# print("Loading raw CSV data...")
# df = pd.read_csv('../datasets/sp500/sp500_stocks.csv')
# df = df.sort_values(["symbol", "date"])
# df["date"] = pd.to_datetime(df["date"])

# print("Engineering technical features and multi-horizon targets...")

# g = df.groupby("symbol")

# # --- Base return / momentum features ---
# df["daily_return"] = g["close"].pct_change()
# df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
# df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
# df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
# df["ma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
# df["close_to_ma5"] = df["close"] / (df["ma5"] + 1e-8)
# df["close_to_ma20"] = df["close"] / (df["ma20"] + 1e-8)
# df["close_to_ma50"] = df["close"] / (df["ma50"] + 1e-8)
# df["ma5_to_ma20"] = df["ma5"] / (df["ma20"] + 1e-8)

# df["high_to_low"] = df["high"] / (df["low"] + 1e-8)
# df["close_to_open"] = df["close"] / (df["open"] + 1e-8)

# # --- Volatility / range features ---
# df["volatility"] = g["daily_return"].transform(lambda x: x.rolling(20).std())
# prev_close = g["close"].shift(1)
# tr = pd.concat([
#     df["high"] - df["low"],
#     (df["high"] - prev_close).abs(),
#     (df["low"] - prev_close).abs()
# ], axis=1).max(axis=1)
# df["true_range"] = tr
# df["atr14"] = df.groupby("symbol")["true_range"].transform(lambda x: x.rolling(14).mean())
# df["atr_pct"] = df["atr14"] / (df["close"] + 1e-8)

# # --- Bollinger %B ---
# bb_std20 = g["close"].transform(lambda x: x.rolling(20).std())
# df["bollinger_pctb"] = (df["close"] - df["ma20"]) / (2 * bb_std20 + 1e-8)

# # --- MACD ---
# ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
# ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
# df["macd"] = (ema12 - ema26) / (df["close"] + 1e-8)
# df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
# df["macd_hist"] = df["macd"] - df["macd_signal"]

# # --- Rate of change over multiple windows ---
# df["roc5"] = g["close"].pct_change(5)
# df["roc10"] = g["close"].pct_change(10)
# df["roc20"] = g["close"].pct_change(20)

# # --- Volume features ---
# df["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
# df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-8)
# df["volume_z"] = df.groupby("symbol")["volume"].transform(
#     lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
# )

# # --- Gap ---
# df["prev_close"] = g["close"].shift(1)
# df["gap"] = df["open"] / (df["prev_close"] + 1e-8) - 1.0

# # --- RSI ---
# def calculate_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / (loss + 1e-8)
#     return 100.0 - (100.0 / (1.0 + rs + 1e-8))

# df["rsi"] = df.groupby("symbol")["close"].transform(lambda x: calculate_rsi(x))

# # --- Market-relative feature (equal-weighted "index" proxy) ---
# market_return = df.groupby("date")["daily_return"].transform("mean")
# df["market_return"] = market_return
# df["relative_return"] = df["daily_return"] - df["market_return"]

# # --- Cross-sectional rank features (key for a long/short ranking model) ---
# for col in ["daily_return", "rsi", "volume_ratio", "roc10"]:
#     df[f"{col}_xrank"] = df.groupby("date")[col].rank(pct=True)

# # --- Calendar encoding ---
# df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5.0)
# df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5.0)

# # --- Multi-horizon targets ---
# horizon_days = {"target_1d": 1, "target_5d": 5, "target_10d": 10, "target_20d": 20}
# max_horizon = max(horizon_days.values())
# for name, h in horizon_days.items():
#     if h == 1:
#         df[name] = g["daily_return"].shift(-1)
#     else:
#         df[name] = g["close"].shift(-h) / (df["close"] + 1e-8) - 1.0

# targets = list(horizon_days.keys())

# df = df.dropna()

# stock_lengths = df.groupby("symbol").size()
# valid_symbols = stock_lengths[stock_lengths >= 150].index
# df = df[df["symbol"].isin(valid_symbols)]

# # ==========================================
# # 1b. CHRONOLOGICAL SPLIT WITH EMBARGO (LEAKAGE FIX)
# # ==========================================
# # Targets look up to `max_horizon` days into the future. Rows within
# # `max_horizon` days of a split boundary have labels computed from prices
# # that fall on the other side of that boundary -> leakage. We purge them.
# train_end = pd.Timestamp("2022-01-01")
# val_end = pd.Timestamp("2023-01-01")
# embargo = pd.Timedelta(days=max_horizon * 2)  # calendar days to cover weekends/holidays safely

# train = df[df["date"] < (train_end - embargo)].copy()
# val = df[(df["date"] >= train_end) & (df["date"] < (val_end - embargo))].copy()
# test = df[df["date"] >= val_end].copy()

# print(f"Embargo applied: {embargo.days} calendar days on each split boundary.")
# print(f"Train rows: {len(train)} | Val rows: {len(val)} | Test rows: {len(test)}")

# seq_features = [
#     "daily_return", "close_to_ma5", "close_to_ma20", "close_to_ma50", "ma5_to_ma20",
#     "high_to_low", "close_to_open", "volatility", "atr_pct", "bollinger_pctb",
#     "macd", "macd_signal", "macd_hist", "roc5", "roc10", "roc20",
#     "volume_ratio", "volume_z", "gap", "rsi", "relative_return",
#     "daily_return_xrank", "rsi_xrank", "volume_ratio_xrank", "roc10_xrank",
#     "dow_sin", "dow_cos"
# ]
# curr_features = ["daily_return", "volatility", "volume_ratio", "rsi", "relative_return", "roc10_xrank"]

# scaler = StandardScaler()
# all_feature_cols = list(dict.fromkeys(seq_features + curr_features))  # de-dupe, preserve order
# train[all_feature_cols] = scaler.fit_transform(train[all_feature_cols])
# val[all_feature_cols] = scaler.transform(val[all_feature_cols])
# test[all_feature_cols] = scaler.transform(test[all_feature_cols])

# joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
# with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "w") as f:
#     json.dump({
#         "seq_features": seq_features,
#         "curr_features": curr_features,
#         "targets": targets,
#         "seq_len": 60
#     }, f, indent=2)

# # ==========================================
# # 2. DATASET
# # ==========================================
# class MultiHorizonStockDataset(Dataset):
#     def __init__(self, df, seq_features, curr_features, targets, seq_len=60):
#         self.seq_len = seq_len
#         self.symbols = df["symbol"].to_numpy()
#         self.dates = df["date"].astype(str).to_numpy()
#         self.X_seq = df[seq_features].to_numpy(dtype=np.float32)
#         self.X_curr = df[curr_features].to_numpy(dtype=np.float32)
#         self.Y = df[targets].to_numpy(dtype=np.float32)

#         self.indices = []
#         symbol_indices = {}
#         for idx, sym in enumerate(self.symbols):
#             symbol_indices.setdefault(sym, []).append(idx)
#         for sym, idxs in symbol_indices.items():
#             if len(idxs) > seq_len:
#                 for start_pos in range(len(idxs) - seq_len):
#                     self.indices.append(idxs[start_pos])

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start = self.indices[idx]
#         x_seq = self.X_seq[start:start + self.seq_len].copy()
#         x_curr = self.X_curr[start + self.seq_len - 1].copy()
#         y = self.Y[start + self.seq_len - 1].copy()
#         symbol = self.symbols[start + self.seq_len - 1]
#         date = self.dates[start + self.seq_len - 1]
#         return torch.from_numpy(x_seq), torch.from_numpy(x_curr), torch.from_numpy(y), symbol, date

# SEQ_LEN = 60
# train_dataset = MultiHorizonStockDataset(train, seq_features, curr_features, targets, seq_len=SEQ_LEN)
# val_dataset = MultiHorizonStockDataset(val, seq_features, curr_features, targets, seq_len=SEQ_LEN)
# test_dataset = MultiHorizonStockDataset(test, seq_features, curr_features, targets, seq_len=SEQ_LEN)

# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# # ==========================================
# # 3. MODEL (kept slim + regularized, per prior fix)
# # ==========================================
# class TemporalTransformerEncoder(nn.Module):
#     def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.3):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * 4,
#             batch_first=True, dropout=dropout
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x):
#         out = self.transformer(x)
#         out = out.transpose(1, 2)
#         return self.pooling(out).squeeze(-1)

# class UnifiedDeepEncoder(nn.Module):
#     def __init__(self, seq_features_dim, static_features_dim, embed_dim=48, hidden_dim=32, horizons=4, dropout=0.4):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=seq_features_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
#         self.gru = nn.GRU(input_size=seq_features_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
#         self.transformer = TemporalTransformerEncoder(input_dim=seq_features_dim, dropout=dropout)
#         self.trans_proj = nn.Linear(seq_features_dim, hidden_dim)
#         self.project_embedding = nn.Linear(hidden_dim * 3, embed_dim)
#         self.embed_dropout = nn.Dropout(dropout)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim + static_features_dim, 64), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(32, horizons)
#         )

#     def forward(self, x_seq, x_curr, return_embedding_only=False):
#         lstm_out, _ = self.lstm(x_seq)
#         h_lstm = lstm_out[:, -1, :]
#         gru_out, _ = self.gru(x_seq)
#         h_gru = gru_out[:, -1, :]
#         h_trans = self.trans_proj(self.transformer(x_seq))
#         h_concat = torch.cat([h_lstm, h_gru, h_trans], dim=-1)
#         embedding = self.embed_dropout(self.project_embedding(h_concat))
#         if return_embedding_only:
#             return embedding
#         predictions = self.mlp(torch.cat([embedding, x_curr], dim=-1))
#         return predictions, embedding

# # ==========================================
# # 4. LOSS
# # ==========================================
# class HybridHorizonLoss(nn.Module):
#     def __init__(self, alpha=0.5):
#         super().__init__()
#         self.alpha = alpha
#         self.mse = nn.MSELoss()

#     def forward(self, preds, targets):
#         mse_loss = self.mse(preds, targets)
#         ic_loss = 0.0
#         horizons = preds.shape[1]
#         for h in range(horizons):
#             p, t = preds[:, h], targets[:, h]
#             p_cov, t_cov = p - p.mean(), t - t.mean()
#             num = torch.sum(p_cov * t_cov)
#             denom = torch.sqrt(torch.sum(p_cov ** 2) * torch.sum(t_cov ** 2) + 1e-8)
#             ic_loss += (1.0 - num / denom)
#         ic_loss = ic_loss / horizons
#         return self.alpha * mse_loss + (1.0 - self.alpha) * ic_loss

# def compute_ic_per_horizon(preds, targets):
#     """Spearman rank IC per horizon -- the real 'does this rank stocks well' metric."""
#     ics = []
#     for h in range(preds.shape[1]):
#         ic, _ = stats.spearmanr(preds[:, h], targets[:, h])
#         ics.append(ic if not np.isnan(ic) else 0.0)
#     return ics

# # ==========================================
# # STAGE 1: TRAINING WITH EARLY STOPPING + PER-HORIZON IC LOGGING
# # ==========================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Executing Deep Learning model training on {device}...")

# model = UnifiedDeepEncoder(
#     seq_features_dim=len(seq_features), static_features_dim=len(curr_features),
#     embed_dim=48, hidden_dim=32, horizons=len(targets), dropout=0.4
# ).to(device)

# criterion = HybridHorizonLoss(alpha=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# epochs = 25
# best_val_loss = float("inf")
# patience = 4
# epochs_no_improve = 0
# max_grad_norm = 1.0

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
#     for x_seq, x_curr, y, _, _ in progress:
#         x_seq, x_curr, y = x_seq.to(device), x_curr.to(device), y.to(device)
#         optimizer.zero_grad()
#         preds, _ = model(x_seq, x_curr)
#         loss = criterion(preds, y)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()
#         running_loss += loss.item()
#         progress.set_postfix(loss=f"{loss.item():.4f}")
#     avg_train_loss = running_loss / len(train_loader)

#     model.eval()
#     running_val_loss = 0.0
#     all_val_preds, all_val_targets = [], []
#     with torch.no_grad():
#         for x_seq, x_curr, y, _, _ in val_loader:
#             x_seq, x_curr, y = x_seq.to(device), x_curr.to(device), y.to(device)
#             preds, _ = model(x_seq, x_curr)
#             running_val_loss += criterion(preds, y).item()
#             all_val_preds.append(preds.cpu().numpy())
#             all_val_targets.append(y.cpu().numpy())
#     avg_val_loss = running_val_loss / len(val_loader)

#     val_preds_np = np.concatenate(all_val_preds, axis=0)
#     val_targets_np = np.concatenate(all_val_targets, axis=0)
#     ics = compute_ic_per_horizon(val_preds_np, val_targets_np)

#     print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
#     print(f"  Val Spearman IC by horizon -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, ics)))

#     scheduler.step(avg_val_loss)

#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         epochs_no_improve = 0
#         torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "best_hybrid_deep.pth"))
#         print("  ✓ Saved Best Neural Model Checkpoint")
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping triggered at epoch {epoch+1}.")
#             break

# model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "best_hybrid_deep.pth")))

# # ==========================================
# # STAGE 2: EMBEDDING EXTRACTION
# # ==========================================
# print("\nExtracting deep embeddings...")

# def extract_embeddings_and_labels(loader, model_instance):
#     model_instance.eval()
#     embeddings, curr_feats, labels, symbols, dates = [], [], [], [], []
#     with torch.no_grad():
#         for x_seq, x_curr, y, syms, dts in tqdm(loader, desc="Extracting"):
#             embeds = model_instance(x_seq.to(device), x_curr.to(device), return_embedding_only=True)
#             embeddings.append(embeds.cpu().numpy())
#             curr_feats.append(x_curr.numpy())
#             labels.append(y.numpy())
#             symbols.extend(syms)
#             dates.extend(dts)
#     return (np.concatenate(embeddings), np.concatenate(curr_feats), np.concatenate(labels),
#             np.array(symbols), np.array(dates))

# E_train, C_train, Y_train, sym_train, date_train = extract_embeddings_and_labels(train_loader, model)
# E_val, C_val, Y_val, sym_val, date_val = extract_embeddings_and_labels(val_loader, model)
# E_test, C_test, Y_test, sym_test, date_test = extract_embeddings_and_labels(test_loader, model)

# X_tree_train = np.ascontiguousarray(np.hstack([E_train, C_train]))
# X_tree_val = np.ascontiguousarray(np.hstack([E_val, C_val]))
# X_tree_test = np.ascontiguousarray(np.hstack([E_test, C_test]))

# # ==========================================
# # STAGE 3: TREE ENSEMBLES
# # ==========================================
# print("\nTraining Downstream Tree Ensembles...")
# horizon_tree_models = {h: {} for h in range(len(targets))}

# for h_idx, h_name in enumerate(targets):
#     y_tr, y_vl = Y_train[:, h_idx], Y_val[:, h_idx]
#     print(f"--- Training Horizon {h_name} ---")

#     if HAS_XGB:
#         xgb_reg = xgb.XGBRegressor(
#             n_estimators=300, max_depth=3, learning_rate=0.03,
#             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
#             n_jobs=-1, random_state=42, early_stopping_rounds=20,
#         )
#         xgb_reg.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)], verbose=False)
#         horizon_tree_models[h_idx]['xgb'] = xgb_reg
#         joblib.dump(xgb_reg, os.path.join(ARTIFACT_DIR, f"xgb_h{h_idx}.joblib"))

#     if HAS_LGB:
#         lgb_reg = lgb.LGBMRegressor(
#             n_estimators=300, max_depth=3, learning_rate=0.03,
#             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
#             n_jobs=-1, random_state=42, verbose=-1,
#         )
#         lgb_reg.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)],
#                     callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
#         horizon_tree_models[h_idx]['lgb'] = lgb_reg
#         joblib.dump(lgb_reg, os.path.join(ARTIFACT_DIR, f"lgb_h{h_idx}.joblib"))

#     if HAS_CAT:
#         cat_reg = cb.CatBoostRegressor(
#             iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=5.0,
#             random_seed=42, verbose=0, early_stopping_rounds=20,
#         )
#         cat_reg.fit(X_tree_train, y_tr, eval_set=(X_tree_val, y_vl))
#         horizon_tree_models[h_idx]['cat'] = cat_reg
#         joblib.dump(cat_reg, os.path.join(ARTIFACT_DIR, f"cat_h{h_idx}.joblib"))

# # ==========================================
# # STAGE 4: STACKING WEIGHTS
# # ==========================================
# print("\nOptimizing Stacking/Meta weights on validation predictions...")
# meta_weights = {}
# model.eval()
# mlp_val_preds = []
# with torch.no_grad():
#     for x_seq, x_curr, _, _, _ in val_loader:
#         preds, _ = model(x_seq.to(device), x_curr.to(device))
#         mlp_val_preds.append(preds.cpu().numpy())
# Y_pred_val_mlp = np.concatenate(mlp_val_preds, axis=0)

# for h_idx in range(len(targets)):
#     pred_columns = [Y_pred_val_mlp[:, h_idx]]
#     model_order = ["mlp"]
#     for key in ["xgb", "lgb", "cat"]:
#         if key in horizon_tree_models[h_idx]:
#             pred_columns.append(horizon_tree_models[h_idx][key].predict(X_tree_val))
#             model_order.append(key)
#     val_pred_matrix = np.column_stack(pred_columns)
#     meta_model = Ridge(alpha=5.0, fit_intercept=False)
#     meta_model.fit(val_pred_matrix, Y_val[:, h_idx])
#     w = meta_model.coef_
#     if w.sum() > 0:
#         w = w / w.sum()
#     meta_weights[h_idx] = {"weights": w.tolist(), "order": model_order}
#     print(f"Horizon {targets[h_idx]} Meta Weights {model_order}: {w}")

# with open(os.path.join(ARTIFACT_DIR, "meta_weights.json"), "w") as f:
#     json.dump(meta_weights, f, indent=2)

# # ==========================================
# # STAGE 5: COST-AWARE BACKTEST
# # ==========================================
# print("\nSimulating Trading Strategy on Out-of-Sample Test Set...")

# mlp_test_preds = []
# with torch.no_grad():
#     for x_seq, x_curr, _, _, _ in test_loader:
#         preds, _ = model(x_seq.to(device), x_curr.to(device))
#         mlp_test_preds.append(preds.cpu().numpy())
# Y_pred_test_mlp = np.concatenate(mlp_test_preds, axis=0)

# Y_test_final_preds = np.zeros_like(Y_test)
# for h_idx in range(len(targets)):
#     cols = [Y_pred_test_mlp[:, h_idx]]
#     for key in ["xgb", "lgb", "cat"]:
#         if key in horizon_tree_models[h_idx]:
#             cols.append(horizon_tree_models[h_idx][key].predict(X_tree_test))
#     test_pred_matrix = np.column_stack(cols)
#     w = np.array(meta_weights[h_idx]["weights"])
#     Y_test_final_preds[:, h_idx] = np.dot(test_pred_matrix, w)

# # Report test IC too -- the number that matters most for "does this generalize"
# test_ics = compute_ic_per_horizon(Y_test_final_preds, Y_test)
# print("Test Spearman IC by horizon -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, test_ics)))

# backtest_df = pd.DataFrame({
#     "date": date_test,
#     "symbol": sym_test,
#     "pred_signal": Y_test_final_preds[:, 1],   # 5D prediction as ranking signal
#     "actual_return_1d": Y_test[:, 0]
# })

# TRANSACTION_COST_BPS = 10  # 0.10% per one-way trade; adjust to your realistic costs
# TOP_N = 20

# unique_days = sorted(backtest_df["date"].unique())
# daily_strategy_returns_gross = []
# daily_strategy_returns_net = []
# prev_longs, prev_shorts = set(), set()

# print(f"Running daily portfolio rebalancing (cost: {TRANSACTION_COST_BPS} bps/trade)...")
# for day in unique_days:
#     day_df = backtest_df[backtest_df["date"] == day]
#     if len(day_df) < 50:
#         continue

#     sorted_df = day_df.sort_values("pred_signal", ascending=False)
#     longs = set(sorted_df.head(TOP_N)["symbol"])
#     shorts = set(sorted_df.tail(TOP_N)["symbol"])

#     long_ret = sorted_df[sorted_df["symbol"].isin(longs)]["actual_return_1d"].mean()
#     short_ret = sorted_df[sorted_df["symbol"].isin(shorts)]["actual_return_1d"].mean()
#     gross_ret = (long_ret - short_ret) / 2.0

#     # Turnover-based cost: fraction of the book that changed since yesterday
#     long_turnover = len(longs.symmetric_difference(prev_longs)) / (2 * TOP_N)
#     short_turnover = len(shorts.symmetric_difference(prev_shorts)) / (2 * TOP_N)
#     turnover = (long_turnover + short_turnover) / 2.0
#     cost = turnover * (TRANSACTION_COST_BPS / 10000.0)

#     daily_strategy_returns_gross.append(gross_ret)
#     daily_strategy_returns_net.append(gross_ret - cost)
#     prev_longs, prev_shorts = longs, shorts

# def summarize(returns, label):
#     arr = np.array(returns)
#     cum = np.prod(1.0 + arr) - 1.0
#     mean = np.mean(arr)
#     vol = np.std(arr) + 1e-8
#     sharpe = (mean / vol) * np.sqrt(252)
#     max_dd = _max_drawdown(arr)
#     print(f"\n--- {label} ---")
#     print(f"Trading Days: {len(arr)}")
#     print(f"Cumulative Return: {cum * 100:.2f}%")
#     print(f"Daily Mean Return: {mean * 100:.4f}%")
#     print(f"Annualized Volatility: {vol * np.sqrt(252) * 100:.2f}%")
#     print(f"Sharpe Ratio: {sharpe:.3f}")
#     print(f"Max Drawdown: {max_dd * 100:.2f}%")
#     return {"cumulative_return": cum, "sharpe": sharpe, "max_drawdown": max_dd}

# def _max_drawdown(returns):
#     cum = np.cumprod(1.0 + returns)
#     peak = np.maximum.accumulate(cum)
#     dd = (cum - peak) / peak
#     return dd.min()

# gross_stats = summarize(daily_strategy_returns_gross, "Gross (no costs) Backtest Summary")
# net_stats = summarize(daily_strategy_returns_net, f"Net of {TRANSACTION_COST_BPS}bps Costs Backtest Summary")

# with open(os.path.join(ARTIFACT_DIR, "backtest_summary.json"), "w") as f:
#     json.dump({"gross": gross_stats, "net": net_stats, "test_ic_by_horizon": dict(zip(targets, test_ics))}, f, indent=2)

# print("\n--- Execution Complete ---")
# print(f"All artifacts saved to {ARTIFACT_DIR}/ for serving.")
# print("\nCAVEATS before treating this as deployable:")
# print(" - Single train/val/test split on one historical window; not walk-forward validated across regimes.")
# print(" - Costs modeled as flat bps; real slippage/impact for less liquid names will differ.")
# print(" - No short-borrow availability/cost modeled.")
# print(" - Universe survivorship in the CSV is not verified here (delisted/failed companies may be missing).")


"""
Advanced Quantitative Multi-Asset Pipeline (v2)

Changes from v1:
  - Fixed train/val/test boundary leakage via embargo purging
  - Added ~20 additional engineered features (MACD, Bollinger %B, ATR, ROC,
    multi-window moving averages, cross-sectional daily rank/z-score,
    market-relative return, calendar encoding)
  - Per-horizon Information Coefficient (IC) logged separately from loss
    at every epoch, so you can see real predictive signal, not just a
    blended loss number
  - Cost-aware backtest (configurable bps transaction costs + turnover)
  - Saves all artifacts (scaler, model weights, tree models, meta-weights,
    feature list) to ./artifacts/ so they can be loaded by a serving API

IMPORTANT: this is a research pipeline. It does not, by itself, prove a
strategy is profitable. See the printed backtest section and the caveats
at the bottom of this file before treating any output as tradeable.
"""

print("Starting Advanced Quantitative Multi-Asset Pipeline (v2)")

import os
import json
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

ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ==========================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ==========================================
print("Loading raw CSV data...")
df = pd.read_csv('../datasets/sp500/sp500_stocks.csv')
df = df.sort_values(["symbol", "date"])
df["date"] = pd.to_datetime(df["date"])

print("Engineering technical features and multi-horizon targets...")

g = df.groupby("symbol")

# --- Base return / momentum features ---
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

# --- Volatility / range features ---
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

# --- Bollinger %B ---
bb_std20 = g["close"].transform(lambda x: x.rolling(20).std())
df["bollinger_pctb"] = (df["close"] - df["ma20"]) / (2 * bb_std20 + 1e-8)

# --- MACD ---
ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df["macd"] = (ema12 - ema26) / (df["close"] + 1e-8)
df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
df["macd_hist"] = df["macd"] - df["macd_signal"]

# --- Rate of change over multiple windows ---
df["roc5"] = g["close"].pct_change(5)
df["roc10"] = g["close"].pct_change(10)
df["roc20"] = g["close"].pct_change(20)

# --- Volume features ---
df["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-8)
df["volume_z"] = df.groupby("symbol")["volume"].transform(
    lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
)

# --- Gap ---
df["prev_close"] = g["close"].shift(1)
df["gap"] = df["open"] / (df["prev_close"] + 1e-8) - 1.0

# --- RSI ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs + 1e-8))

df["rsi"] = df.groupby("symbol")["close"].transform(lambda x: calculate_rsi(x))

# --- Market-relative feature (equal-weighted "index" proxy) ---
market_return = df.groupby("date")["daily_return"].transform("mean")
df["market_return"] = market_return
df["relative_return"] = df["daily_return"] - df["market_return"]

# --- Cross-sectional rank features (key for a long/short ranking model) ---
for col in ["daily_return", "rsi", "volume_ratio", "roc10"]:
    df[f"{col}_xrank"] = df.groupby("date")[col].rank(pct=True)

# --- Calendar encoding ---
df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5.0)
df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5.0)

# --- Multi-horizon targets ---
horizon_days = {"target_1d": 1, "target_5d": 5, "target_10d": 10, "target_20d": 20}
max_horizon = max(horizon_days.values())
for name, h in horizon_days.items():
    if h == 1:
        df[name] = g["daily_return"].shift(-1)
    else:
        df[name] = g["close"].shift(-h) / (df["close"] + 1e-8) - 1.0

targets = list(horizon_days.keys())

df = df.dropna()

stock_lengths = df.groupby("symbol").size()
valid_symbols = stock_lengths[stock_lengths >= 150].index
df = df[df["symbol"].isin(valid_symbols)]

# ==========================================
# 1b. CHRONOLOGICAL SPLIT WITH EMBARGO (LEAKAGE FIX)
# ==========================================
# Targets look up to `max_horizon` days into the future. Rows within
# `max_horizon` days of a split boundary have labels computed from prices
# that fall on the other side of that boundary -> leakage. We purge them.
train_end = pd.Timestamp("2022-01-01")
val_end = pd.Timestamp("2023-01-01")
embargo = pd.Timedelta(days=max_horizon * 2)  # calendar days to cover weekends/holidays safely

train = df[df["date"] < (train_end - embargo)].copy()
val = df[(df["date"] >= train_end) & (df["date"] < (val_end - embargo))].copy()
test = df[df["date"] >= val_end].copy()

print(f"Embargo applied: {embargo.days} calendar days on each split boundary.")
print(f"Train rows: {len(train)} | Val rows: {len(val)} | Test rows: {len(test)}")

seq_features = [
    "daily_return", "close_to_ma5", "close_to_ma20", "close_to_ma50", "ma5_to_ma20",
    "high_to_low", "close_to_open", "volatility", "atr_pct", "bollinger_pctb",
    "macd", "macd_signal", "macd_hist", "roc5", "roc10", "roc20",
    "volume_ratio", "volume_z", "gap", "rsi", "relative_return",
    "daily_return_xrank", "rsi_xrank", "volume_ratio_xrank", "roc10_xrank",
    "dow_sin", "dow_cos"
]
curr_features = ["daily_return", "volatility", "volume_ratio", "rsi", "relative_return", "roc10_xrank"]

scaler = StandardScaler()
all_feature_cols = list(dict.fromkeys(seq_features + curr_features))  # de-dupe, preserve order
train[all_feature_cols] = scaler.fit_transform(train[all_feature_cols])
val[all_feature_cols] = scaler.transform(val[all_feature_cols])
test[all_feature_cols] = scaler.transform(test[all_feature_cols])

joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "w") as f:
    json.dump({
        "seq_features": seq_features,
        "curr_features": curr_features,
        "targets": targets,
        "seq_len": 60
    }, f, indent=2)

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

SEQ_LEN = 60
train_dataset = MultiHorizonStockDataset(train, seq_features, curr_features, targets, seq_len=SEQ_LEN)
val_dataset = MultiHorizonStockDataset(val, seq_features, curr_features, targets, seq_len=SEQ_LEN)
test_dataset = MultiHorizonStockDataset(test, seq_features, curr_features, targets, seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 3. MODEL (kept slim + regularized, per prior fix)
# ==========================================
class TemporalTransformerEncoder(nn.Module):
    """
    Projects raw input features into a fixed model_dim before the transformer,
    instead of using the raw feature count as d_model. This decouples the
    encoder from however many engineered features happen to exist -- d_model
    must be divisible by num_heads, and raw feature counts (e.g. 27) often
    aren't divisible by a small head count, which throws an assertion error.
    """
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

# ==========================================
# 4. LOSS
# ==========================================
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
    """Spearman rank IC per horizon -- the real 'does this rank stocks well' metric."""
    ics = []
    for h in range(preds.shape[1]):
        ic, _ = stats.spearmanr(preds[:, h], targets[:, h])
        ics.append(ic if not np.isnan(ic) else 0.0)
    return ics

# ==========================================
# STAGE 1: TRAINING WITH EARLY STOPPING + PER-HORIZON IC LOGGING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Deep Learning model training on {device}...")

model = UnifiedDeepEncoder(
    seq_features_dim=len(seq_features), static_features_dim=len(curr_features),
    embed_dim=48, hidden_dim=32, horizons=len(targets), dropout=0.4
).to(device)

criterion = HybridHorizonLoss(alpha=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

epochs = 25
best_val_loss = float("inf")
patience = 4
epochs_no_improve = 0
max_grad_norm = 1.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for x_seq, x_curr, y, _, _ in progress:
        x_seq, x_curr, y = x_seq.to(device), x_curr.to(device), y.to(device)
        optimizer.zero_grad()
        preds, _ = model(x_seq, x_curr)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    print(f"  Val Spearman IC by horizon -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, ics)))

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "best_hybrid_deep.pth"))
        print("  ✓ Saved Best Neural Model Checkpoint")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "best_hybrid_deep.pth")))

# ==========================================
# STAGE 2: EMBEDDING EXTRACTION
# ==========================================
print("\nExtracting deep embeddings...")

def extract_embeddings_and_labels(loader, model_instance):
    model_instance.eval()
    embeddings, curr_feats, labels, symbols, dates = [], [], [], [], []
    with torch.no_grad():
        for x_seq, x_curr, y, syms, dts in tqdm(loader, desc="Extracting"):
            embeds = model_instance(x_seq.to(device), x_curr.to(device), return_embedding_only=True)
            embeddings.append(embeds.cpu().numpy())
            curr_feats.append(x_curr.numpy())
            labels.append(y.numpy())
            symbols.extend(syms)
            dates.extend(dts)
    return (np.concatenate(embeddings), np.concatenate(curr_feats), np.concatenate(labels),
            np.array(symbols), np.array(dates))

E_train, C_train, Y_train, sym_train, date_train = extract_embeddings_and_labels(train_loader, model)
E_val, C_val, Y_val, sym_val, date_val = extract_embeddings_and_labels(val_loader, model)
E_test, C_test, Y_test, sym_test, date_test = extract_embeddings_and_labels(test_loader, model)

X_tree_train = np.ascontiguousarray(np.hstack([E_train, C_train]))
X_tree_val = np.ascontiguousarray(np.hstack([E_val, C_val]))
X_tree_test = np.ascontiguousarray(np.hstack([E_test, C_test]))

# ==========================================
# STAGE 3: TREE ENSEMBLES
# ==========================================
print("\nTraining Downstream Tree Ensembles...")
horizon_tree_models = {h: {} for h in range(len(targets))}

for h_idx, h_name in enumerate(targets):
    y_tr, y_vl = Y_train[:, h_idx], Y_val[:, h_idx]
    print(f"--- Training Horizon {h_name} ---")

    if HAS_XGB:
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, early_stopping_rounds=20,
        )
        xgb_reg.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)], verbose=False)
        horizon_tree_models[h_idx]['xgb'] = xgb_reg
        joblib.dump(xgb_reg, os.path.join(ARTIFACT_DIR, f"xgb_h{h_idx}.joblib"))

    if HAS_LGB:
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        lgb_reg.fit(X_tree_train, y_tr, eval_set=[(X_tree_val, y_vl)],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        horizon_tree_models[h_idx]['lgb'] = lgb_reg
        joblib.dump(lgb_reg, os.path.join(ARTIFACT_DIR, f"lgb_h{h_idx}.joblib"))

    if HAS_CAT:
        cat_reg = cb.CatBoostRegressor(
            iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=5.0,
            random_seed=42, verbose=0, early_stopping_rounds=20,
        )
        cat_reg.fit(X_tree_train, y_tr, eval_set=(X_tree_val, y_vl))
        horizon_tree_models[h_idx]['cat'] = cat_reg
        joblib.dump(cat_reg, os.path.join(ARTIFACT_DIR, f"cat_h{h_idx}.joblib"))

# ==========================================
# STAGE 4: STACKING WEIGHTS
# ==========================================
print("\nOptimizing Stacking/Meta weights on validation predictions...")
meta_weights = {}
model.eval()
mlp_val_preds = []
with torch.no_grad():
    for x_seq, x_curr, _, _, _ in val_loader:
        preds, _ = model(x_seq.to(device), x_curr.to(device))
        mlp_val_preds.append(preds.cpu().numpy())
Y_pred_val_mlp = np.concatenate(mlp_val_preds, axis=0)

for h_idx in range(len(targets)):
    pred_columns = [Y_pred_val_mlp[:, h_idx]]
    model_order = ["mlp"]
    for key in ["xgb", "lgb", "cat"]:
        if key in horizon_tree_models[h_idx]:
            pred_columns.append(horizon_tree_models[h_idx][key].predict(X_tree_val))
            model_order.append(key)
    val_pred_matrix = np.column_stack(pred_columns)
    meta_model = Ridge(alpha=5.0, fit_intercept=False)
    meta_model.fit(val_pred_matrix, Y_val[:, h_idx])
    w = meta_model.coef_
    if w.sum() > 0:
        w = w / w.sum()
    meta_weights[h_idx] = {"weights": w.tolist(), "order": model_order}
    print(f"Horizon {targets[h_idx]} Meta Weights {model_order}: {w}")

with open(os.path.join(ARTIFACT_DIR, "meta_weights.json"), "w") as f:
    json.dump(meta_weights, f, indent=2)

# ==========================================
# STAGE 5: COST-AWARE BACKTEST
# ==========================================
print("\nSimulating Trading Strategy on Out-of-Sample Test Set...")

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
    test_pred_matrix = np.column_stack(cols)
    w = np.array(meta_weights[h_idx]["weights"])
    Y_test_final_preds[:, h_idx] = np.dot(test_pred_matrix, w)

# Report test IC too -- the number that matters most for "does this generalize"
test_ics = compute_ic_per_horizon(Y_test_final_preds, Y_test)
print("Test Spearman IC by horizon -> " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(targets, test_ics)))

backtest_df = pd.DataFrame({
    "date": date_test,
    "symbol": sym_test,
    "pred_signal": Y_test_final_preds[:, 1],   # 5D prediction as ranking signal
    "actual_return_1d": Y_test[:, 0]
})

TRANSACTION_COST_BPS = 10  # 0.10% per one-way trade; adjust to your realistic costs
TOP_N = 20

unique_days = sorted(backtest_df["date"].unique())
daily_strategy_returns_gross = []
daily_strategy_returns_net = []
prev_longs, prev_shorts = set(), set()

print(f"Running daily portfolio rebalancing (cost: {TRANSACTION_COST_BPS} bps/trade)...")
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

    # Turnover-based cost: fraction of the book that changed since yesterday
    long_turnover = len(longs.symmetric_difference(prev_longs)) / (2 * TOP_N)
    short_turnover = len(shorts.symmetric_difference(prev_shorts)) / (2 * TOP_N)
    turnover = (long_turnover + short_turnover) / 2.0
    cost = turnover * (TRANSACTION_COST_BPS / 10000.0)

    daily_strategy_returns_gross.append(gross_ret)
    daily_strategy_returns_net.append(gross_ret - cost)
    prev_longs, prev_shorts = longs, shorts

def summarize(returns, label):
    arr = np.array(returns)
    cum = np.prod(1.0 + arr) - 1.0
    mean = np.mean(arr)
    vol = np.std(arr) + 1e-8
    sharpe = (mean / vol) * np.sqrt(252)
    max_dd = _max_drawdown(arr)
    print(f"\n--- {label} ---")
    print(f"Trading Days: {len(arr)}")
    print(f"Cumulative Return: {cum * 100:.2f}%")
    print(f"Daily Mean Return: {mean * 100:.4f}%")
    print(f"Annualized Volatility: {vol * np.sqrt(252) * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    return {"cumulative_return": cum, "sharpe": sharpe, "max_drawdown": max_dd}

def _max_drawdown(returns):
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()

gross_stats = summarize(daily_strategy_returns_gross, "Gross (no costs) Backtest Summary")
net_stats = summarize(daily_strategy_returns_net, f"Net of {TRANSACTION_COST_BPS}bps Costs Backtest Summary")

with open(os.path.join(ARTIFACT_DIR, "backtest_summary.json"), "w") as f:
    json.dump({"gross": gross_stats, "net": net_stats, "test_ic_by_horizon": dict(zip(targets, test_ics))}, f, indent=2)

print("\n--- Execution Complete ---")
print(f"All artifacts saved to {ARTIFACT_DIR}/ for serving.")
print("\nCAVEATS before treating this as deployable:")
print(" - Single train/val/test split on one historical window; not walk-forward validated across regimes.")
print(" - Costs modeled as flat bps; real slippage/impact for less liquid names will differ.")
print(" - No short-borrow availability/cost modeled.")
print(" - Universe survivorship in the CSV is not verified here (delisted/failed companies may be missing).")