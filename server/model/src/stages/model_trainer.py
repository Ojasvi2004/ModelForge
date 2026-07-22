import os
import sys
import torch
from src.logger import logger
from src.exception import MyException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts import ModelTrainerArtifactEntity,WalkForwardFoldArtifactEntity
import time
import joblib
import numpy as np
import pandas as pd
from src.constants import *
from scipy import stats  # FIX: bare "import stats" is not a valid module
import json

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import Ridge

from torch.utils.data import DataLoader,Dataset
from src.utils.main_utils import read_yaml_file,write_yaml
import torch.nn as nn


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


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHorizonStockDataset(Dataset):
    """
    Turns a flat (symbol, date, features...) dataframe into sliding windows of
    length `seq_len` per symbol.

    Shapes you'll see in the logs:
      X_seq  -> (num_rows, num_seq_features)   e.g. (250000, 12)
      X_curr -> (num_rows, num_curr_features)  e.g. (250000, 5)
      Y      -> (num_rows, num_targets)        e.g. (250000, 3)
    Each __getitem__ call slices a (seq_len, num_seq_features) window out of X_seq.
    """
    def __init__(self, df: pd.DataFrame, seq_features: list, curr_features: list, targets: list, seq_len: int = 60):
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

        # STEP LOG: what this dataset ended up holding, in plain shapes.
        logger.logging.info(
            f"[Dataset] Built MultiHorizonStockDataset -> "
            f"X_seq:{self.X_seq.shape} (dtype={self.X_seq.dtype}), "
            f"X_curr:{self.X_curr.shape} (dtype={self.X_curr.dtype}), "
            f"Y:{self.Y.shape} (dtype={self.Y.dtype}), "
            f"seq_len={self.seq_len}, num_symbols={len(symbol_indices)}, "
            f"usable_windows={len(self.indices)} (rows dropped for being shorter than seq_len)"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # NOTE: intentionally NOT logging here. __getitem__ is called once per
        # sample per batch (potentially hundreds of thousands of times), so a
        # log line here would flood the log file. The shapes it returns are
        # printed once, for a sample batch, inside the training loop below.
        start = self.indices[idx]
        x_seq = self.X_seq[start:start + self.seq_len].copy()
        x_curr = self.X_curr[start + self.seq_len - 1].copy()
        y = self.Y[start + self.seq_len - 1].copy()
        symbol = self.symbols[start + self.seq_len - 1]
        date = self.dates[start + self.seq_len - 1]
        return torch.from_numpy(x_seq), torch.from_numpy(x_curr), torch.from_numpy(y), symbol, date



class TemporalEncoder(nn.Module):
    """
    x: (batch, seq_len, input_dim) -> final: (batch, model_dim)
    Projects each timestep to model_dim, runs a Transformer encoder over the
    sequence, then average-pools across time to get one vector per sample.
    """
    def __init__(self, input_dim,model_dim,num_heads,num_layers,dropout):
        super().__init__()
        assert model_dim%num_heads==0,"model_dim must be divisible by num_heads"
        self.input_proj=nn.Linear(input_dim,model_dim)
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim*4,
            batch_first=True,
            dropout=dropout)

        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.pooling=nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        x=self.input_proj(x)
        out=self.transformer(x)
        out=torch.transpose(out,1,2)
        final=self.pooling(out).squeeze(-1)
        return final



class UnifiedDeepEncoder(nn.Module):
    """
    Combines three sequence encoders (LSTM, GRU, Transformer) on x_seq into one
    embedding, concatenates it with the "current day" static features (x_curr),
    and passes that through an MLP to predict all horizons at once.

    forward() returns (predictions, embeddings):
      predictions -> (batch, horizons)   e.g. (512, 3)   <- what you train/evaluate on
      embeddings  -> (batch, embed_dim)  e.g. (512, 32)  <- reused later as tree-model features
    """
    def __init__(self, seq_features_dim,
                 static_features_dim,
                 embed_dim,
                 hidden_dim,
                 horizons,
                 dropout):
        super().__init__()  # FIX: was missing, nn.Module requires this before registering submodules
        self.lstm=nn.LSTM(input_size=seq_features_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True
                          )
        self.gru=nn.GRU(input_size=seq_features_dim,
                        hidden_size=hidden_dim,
                        num_layers=1,
                        batch_first=True
                        )
        transformer_model_dim=UNIFIED_DEEP_ENCODER_TRANSFORMER_MODEL_DIM
        self.transformer=TemporalEncoder(input_dim=seq_features_dim,
                                         model_dim=transformer_model_dim,
                                         num_heads=4,
                                         num_layers=1,
                                         dropout=dropout
                                         )
        self.trans_proj=nn.Linear(in_features=transformer_model_dim,out_features=hidden_dim)
        self.project_embedding=nn.Linear(hidden_dim*3,embed_dim)
        self.embed_dropout=nn.Dropout(dropout)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=embed_dim+static_features_dim,out_features=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=64,out_features=32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=32,out_features=horizons)
        )

    def forward(self,x_seq,x_curr,return_embeddings_only=False):
        # NOTE: intentionally not logging inside forward() -- called on every
        # single batch (potentially thousands of times per fold). Shapes are
        # logged once from the training loop for a representative batch instead.
        lstm_out,lst_hc=self.lstm(x_seq)
        h_lstm=lstm_out[:,-1,:]
        gru_out,gru_hc=self.gru(x_seq)
        h_gru=gru_out[:,-1,:]
        h_trans=self.trans_proj(self.transformer(x_seq))
        h_concat=torch.cat([h_lstm,h_gru,h_trans],dim=-1)
        embeddings=self.embed_dropout(self.project_embedding(h_concat))
        if return_embeddings_only:
            return embeddings
        predictions=self.mlp(torch.cat([embeddings,x_curr],dim=-1))
        return predictions, embeddings  # FIX: forward previously returned nothing, but callers do `preds, _ = model(...)`

class HybridHorizonLoss(nn.Module):
    """
    Loss = alpha * MSE(preds, targets) + (1 - alpha) * (1 - mean_information_coefficient)
    The IC term rewards predictions that RANK-correlate with targets per horizon,
    not just match them in absolute value.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha=alpha
        self.mse=nn.MSELoss()

    def forward(self,preds,targets):
        mse_loss=self.mse(preds,targets)
        ic_loss=0.0
        horizons=preds.shape[1]
        for h in range(horizons):
            p, t = preds[:, h], targets[:, h]
            p_cov, t_cov = p - p.mean(), t - t.mean()
            num = torch.sum(p_cov * t_cov)
            denom = torch.sqrt(torch.sum(p_cov ** 2) * torch.sum(t_cov ** 2) + 1e-8)
            ic_loss += (1.0 - num / denom)
        ic_loss = ic_loss / horizons
        return self.alpha * mse_loss + (1.0 - self.alpha) * ic_loss


class ModelTrainer:

    def __init__(self,
                 model_trainer_config:ModelTrainerConfig,
                 walk_forward_fold_artifact:WalkForwardFoldArtifactEntity):
        if model_trainer_config is None:
            model_trainer_config=ModelTrainerConfig()

        try:
            logger.logging.info("Initializing ModelTrainer component.")
            self.model_trainer_config = model_trainer_config
            self.walk_forward_fold_artifact = walk_forward_fold_artifact


            fold_metadata_path = self.walk_forward_fold_artifact.fold_metadata_file_name
            with open(fold_metadata_path, "r") as f:
                self.fold_config = json.load(f)

            self.seq_features = self.fold_config["seq_features"]
            self.curr_features = self.fold_config["curr_features"]
            self.targets = self.fold_config["targets"]
            self.seq_len = self.fold_config["seq_len"]
            self.max_horizon = self.fold_config["max_horizon"]
            self.folds = self.fold_config["folds"]

            # STEP LOG: dump the whole fold config so you know exactly what the
            # rest of the run is going to operate on.
            logger.logging.info(
                f"[Init] Loaded fold_metadata from '{fold_metadata_path}' -> "
                f"seq_features({len(self.seq_features)})={self.seq_features}, "
                f"curr_features({len(self.curr_features)})={self.curr_features}, "
                f"targets({len(self.targets)})={self.targets}, "
                f"seq_len={self.seq_len}, max_horizon={self.max_horizon}, "
                f"num_folds={len(self.folds)}"
            )
            logger.logging.info("Initialized ModelTrainer component.")

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def compute_ic_per_horizon(preds, targets):
        """
        preds, targets -> (num_samples, num_horizons)
        Returns one Spearman rank-correlation value per horizon column.
        """
        ics = []
        for h in range(preds.shape[1]):
            ic, _ = stats.spearmanr(preds[:, h], targets[:, h])
            ics.append(float(ic) if not np.isnan(ic) else 0.0)
        logger.logging.info(
            f"[compute_ic_per_horizon] preds:{preds.shape}, targets:{targets.shape} -> ics={ics}"
        )
        return ics

    @staticmethod
    def max_drawdown(returns):
        cum = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(dd.min())

    def summarize_returns(self, returns, label):
        # STEP LOG: shows what's actually being fed in (a plain list/array of
        # daily returns) before it gets turned into cumulative return / sharpe / drawdown.
        logger.logging.info(
            f"[summarize_returns:{label}] input returns -> type={type(returns)}, "
            f"length={len(returns)}, sample_first_5={list(returns)[:5]}"
        )
        arr = np.array(returns, dtype=np.float64)
        cum = float(np.prod(1.0 + arr) - 1.0)
        mean = float(np.mean(arr))
        vol = float(np.std(arr) + 1e-8)
        sharpe = float((mean / vol) * np.sqrt(252))
        dd = self.max_drawdown(arr)
        logger.logging.info(f"  {label}: CumRet {cum*100:.2f}% | Sharpe {sharpe:.3f} | MaxDD {dd*100:.2f}% | Days {len(arr)}")
        return {"cumulative_return": cum, "sharpe": sharpe, "max_drawdown": dd, "days": len(arr)}

    def run_fold(self,df:pd.DataFrame,fold:dict,fold_idx:int,total_folds:int)->dict:
        """
        Runs ONE walk-forward fold end to end:
          1. Slice df into train/val/test date windows
          2. Scale features
          3. Build torch Datasets/DataLoaders
          4. Train the deep model (UnifiedDeepEncoder)
          5. Extract embeddings for train/val/test
          6. Train tree models (xgb/lgb/cat) on [embeddings + curr_features]
          7. Fit a Ridge meta-model to blend mlp + tree predictions per horizon
          8. Score the blended predictions on the test set (IC + a toy long/short backtest)
          9. Save all artifacts for this fold to disk
        """
        try:
            # FIX: f-string previously reused the same quote char as the outer string
            # (f"...{fold["name"]}") which is a SyntaxError on Python < 3.12.
            logger.logging.info(f'Starting fold {fold_idx+1}/{total_folds}:{fold["name"]}')

            embargo=pd.Timedelta(days=self.max_horizon*2)
            train_end_date=pd.Timestamp(fold["train_end"])
            val_end_date=pd.Timestamp(fold["val_end"])
            test_end_date=pd.Timestamp(fold["test_end"])

            logger.logging.info(
                f"[Fold {fold['name']}] Date boundaries -> "
                f"train_end={train_end_date.date()}, val_end={val_end_date.date()}, "
                f"test_end={test_end_date.date()}, embargo={embargo}"
            )

            train_df = df[df["date"] < (train_end_date - embargo)].copy()
            # FIX: `and` between two pandas boolean Series raises
            # "ValueError: truth value of a Series is ambiguous". Use `&` with parentheses.
            val_df = df[(df["date"] >= train_end_date) & (df["date"] < val_end_date)].copy()
            test_df = df[(df["date"] >= val_end_date) & (df["date"] < test_end_date)].copy()

            logger.logging.info(f"Train samples: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
            logger.logging.info(
                f"[Fold {fold['name']}] train_df columns={list(train_df.columns)}, "
                f"train_df dtypes sample={train_df.dtypes.head(5).to_dict()}"
            )

            if len(train_df) < 5000 or len(val_df) < 500 or len(test_df) < 500:
                logger.logging.warning(f"Skipping fold {fold['name']} due to insufficient row counts.")
                return None

            all_features=list(dict.fromkeys(self.seq_features+self.curr_features))
            logger.logging.info(f"[Fold {fold['name']}] Scaling {len(all_features)} feature columns -> {all_features}")

            scaler=StandardScaler()

            train_df[all_features]=scaler.fit_transform(train_df[all_features])
            test_df[all_features]=scaler.transform(test_df[all_features])
            val_df[all_features]=scaler.transform(val_df[all_features])

            logger.logging.info(
                f"[Fold {fold['name']}] Post-scaling shapes -> "
                f"train_df:{train_df.shape}, val_df:{val_df.shape}, test_df:{test_df.shape} "
                f"(scaler fit on train only; val/test only transformed)"
            )

            train_dataset=MultiHorizonStockDataset(train_df,
                                                   self.seq_features,
                                                   self.curr_features,
                                                   self.targets,
                                                   self.seq_len)

            test_dataset=MultiHorizonStockDataset( test_df,
                                                   self.seq_features,
                                                   self.curr_features,
                                                   self.targets,
                                                   self.seq_len)

            val_dataset=MultiHorizonStockDataset(  val_df,
                                                   self.seq_features,
                                                   self.curr_features,
                                                   self.targets,
                                                   self.seq_len)

            logger.logging.info(
                f"[Fold {fold['name']}] Datasets ready -> "
                f"train_windows={len(train_dataset)}, val_windows={len(val_dataset)}, test_windows={len(test_dataset)}"
            )

            train_loader=DataLoader(train_dataset,
                                    batch_size=512,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

            val_loader=DataLoader(val_dataset,
                                    batch_size=512,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

            test_loader=DataLoader(test_dataset,
                                    batch_size=512,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True)

            logger.logging.info(
                f"[Fold {fold['name']}] DataLoaders ready (batch_size=512) -> "
                f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}"
            )

            fold_dir=os.path.join(self.model_trainer_config.model_trainer_directory,
                                  fold['name'])

            os.makedirs(fold_dir,exist_ok=True)
            logger.logging.info(f"[Fold {fold['name']}] Artifacts for this fold will be saved to: {fold_dir}")

            model=UnifiedDeepEncoder(
                seq_features_dim=len(self.seq_features),
                static_features_dim=len(self.curr_features),
                embed_dim=self.model_trainer_config.UnifiedDeepEncoderConfig.embed_dim,
                hidden_dim=self.model_trainer_config.UnifiedDeepEncoderConfig.hidden_dim,
                horizons=len(self.targets),
                dropout=self.model_trainer_config.UnifiedDeepEncoderConfig.dropout
            ).to(device)

            num_params = sum(p.numel() for p in model.parameters())
            logger.logging.info(
                f"[Fold {fold['name']}] UnifiedDeepEncoder created on device={device} -> "
                f"seq_features_dim={len(self.seq_features)}, static_features_dim={len(self.curr_features)}, "
                f"embed_dim={self.model_trainer_config.UnifiedDeepEncoderConfig.embed_dim}, "
                f"hidden_dim={self.model_trainer_config.UnifiedDeepEncoderConfig.hidden_dim}, "
                f"horizons={len(self.targets)}, trainable_params={num_params:,}"
            )

            criterion=HybridHorizonLoss(alpha=self.model_trainer_config.HybridHorizonLossConfig.alpha)

            optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-2
            )

            # FIX: `factor=self.model_trainer_config.Schedular.patience` was passing the
            # patience value into the `factor` argument. Restored proper factor/patience mapping
            # (assumes Schedular config has a `factor` attribute — adjust if it doesn't).
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.model_trainer_config.Schedular.mode,
                factor=self.model_trainer_config.Schedular.factor,
                patience=self.model_trainer_config.Schedular.patience
            )

            best_val_loss=float('inf')
            epochs_no_improve=0
            patience=self.model_trainer_config.patience
            epochs=self.model_trainer_config.epochs
            ckpt_path=self.model_trainer_config.saved_model_path

            logger.logging.info(
                f"[Fold {fold['name']}] Training config -> epochs={epochs}, patience={patience}, "
                f"criterion_alpha={self.model_trainer_config.HybridHorizonLossConfig.alpha}, "
                f"checkpoint_path={ckpt_path}"
            )

            # ---------------- TRAINING LOOP ----------------
            for epoch in range(epochs):
                model.train()
                running_loss=0.0

                for batch_idx,(x_seq,x_curr,y,_,_) in enumerate(train_loader):
                    x_seq=x_seq.to(device)
                    y=y.to(device)
                    x_curr=x_curr.to(device)

                    optimizer.zero_grad()
                    preds,embeddings=model(x_seq,x_curr)

                    # STEP LOG (only once, first batch of first epoch): shows exactly
                    # what tensors/shapes are flowing through the model, without
                    # flooding the log for every one of the thousands of batches.
                    if epoch == 0 and batch_idx == 0:
                        logger.logging.info(
                            f"[Fold {fold['name']}] Sample TRAIN batch (epoch 1, batch 1) -> "
                            f"x_seq:{tuple(x_seq.shape)} (batch, seq_len, seq_feat_dim), "
                            f"x_curr:{tuple(x_curr.shape)} (batch, curr_feat_dim), "
                            f"y:{tuple(y.shape)} (batch, num_targets)"
                        )
                        logger.logging.info(
                            f"[Fold {fold['name']}] Model outputs on that batch -> "
                            f"preds:{tuple(preds.shape)} (batch, horizons), "
                            f"embeddings:{tuple(embeddings.shape)} (batch, embed_dim)"
                        )

                    loss=criterion(preds,y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),self.model_trainer_config.gradient_clip)
                    optimizer.step()
                    running_loss=running_loss+loss.item()

                avg_train_loss = running_loss/len(train_loader)

                model.eval()
                running_val_loss=0.0
                all_val_preds=[]
                all_val_targets=[]

                with torch.no_grad():
                    for x_seq,x_curr,y,_,_ in val_loader:
                        x_seq=x_seq.to(device)
                        x_curr=x_curr.to(device)
                        y=y.to(device)
                        preds,_=model(x_seq,x_curr)
                        loss=criterion(preds,y)
                        # FIX: accumulate the scalar value, not the tensor (mirrors training loop)
                        running_val_loss=running_val_loss+loss.item()
                        all_val_preds.append(preds.cpu().numpy())
                        all_val_targets.append(y.cpu().numpy())

                avg_val_loss=running_val_loss/len(val_loader)
                scheduler.step(avg_val_loss)

                logger.logging.info(
                    f"[Fold {fold['name']}] Epoch {epoch+1}/{epochs} -> "
                    f"avg_train_loss={avg_train_loss:.6f}, avg_val_loss={avg_val_loss:.6f}, "
                    f"best_val_loss_so_far={best_val_loss if best_val_loss==float('inf') else round(best_val_loss,6)}, "
                    f"current_lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                if avg_val_loss<best_val_loss:
                    best_val_loss=avg_val_loss
                    epochs_no_improve=0
                    torch.save(model.state_dict(),ckpt_path)
                    logger.logging.info(
                        f"[Fold {fold['name']}] Epoch {epoch+1}: NEW BEST val_loss={best_val_loss:.6f} "
                        f"-> checkpoint saved to {ckpt_path}"
                    )

                else:
                    epochs_no_improve=epochs_no_improve+1
                    logger.logging.info(
                        f"[Fold {fold['name']}] Epoch {epoch+1}: no improvement "
                        f"({epochs_no_improve}/{patience} epochs without improving)"
                    )
                    if epochs_no_improve>=patience:
                        logger.logging.info(
                            f"[Fold {fold['name']}] Early stopping at epoch {epoch+1} "
                            f"(no improvement for {patience} epochs)"
                        )
                        break
            # ---------------- END TRAINING LOOP ----------------

            model.load_state_dict(torch.load(ckpt_path))
            logger.logging.info(f"[Fold {fold['name']}] Reloaded best checkpoint from {ckpt_path} for embedding extraction.")

            def extract(loader, split_name):
                """
                Runs the trained model in embedding-only mode over an entire loader
                and stacks everything into plain numpy arrays.

                Returns:
                  E (embeddings)   -> (num_samples, embed_dim)
                  C (curr_feats)   -> (num_samples, num_curr_features)
                  Y (labels)       -> (num_samples, num_targets)
                  symbols          -> (num_samples,)
                  dates            -> (num_samples,)
                """
                model.eval()
                embeddings,curr_feats,labels,symbols,dates=[],[],[],[],[]
                with torch.no_grad():
                    for x_seq, x_curr, y, syms, dts in loader:
                        embeds=model.forward(x_seq.to(device),x_curr.to(device),return_embeddings_only=True)
                        embeddings.append(embeds.cpu().numpy())
                        curr_feats.append(x_curr.numpy())
                        labels.append(y.numpy())
                        symbols.extend(syms)
                        dates.extend(dts)
                E = np.concatenate(embeddings)
                C = np.concatenate(curr_feats)
                Y = np.concatenate(labels)
                S = np.array(symbols)
                D = np.array(dates)
                # STEP LOG: this is the "embeddings" hand-off point between the deep
                # model and the tree models below -- E/C/Y/S/D are exactly what feeds
                # into X_tree_* and the backtest dataframe further down.
                logger.logging.info(
                    f"[Fold {fold['name']}] extract('{split_name}') -> "
                    f"embeddings E:{E.shape}, curr_features C:{C.shape}, labels Y:{Y.shape}, "
                    f"symbols:{S.shape} (dtype={S.dtype}), dates:{D.shape} (dtype={D.dtype})"
                )
                return E, C, Y, S, D

            E_train, C_train, Y_train, sym_train, date_train = extract(train_loader, "train")
            E_val, C_val, Y_val, sym_val, date_val = extract(val_loader, "val")
            E_test, C_test, Y_test, sym_test, date_test = extract(test_loader, "test")


            X_tree_train = np.ascontiguousarray(np.hstack([E_train, C_train]))
            X_tree_val = np.ascontiguousarray(np.hstack([E_val, C_val]))
            X_tree_test = np.ascontiguousarray(np.hstack([E_test, C_test]))

            logger.logging.info(
                f"[Fold {fold['name']}] Tree-model feature matrices "
                f"(deep embeddings concatenated with current-day features) -> "
                f"X_tree_train:{X_tree_train.shape}, X_tree_val:{X_tree_val.shape}, X_tree_test:{X_tree_test.shape}"
            )

            horizon_tree_models={h:{} for h in range(len(self.targets))}

            # ---------------- TREE MODELS, ONE SET PER HORIZON ----------------
            for h_idx,h_name in enumerate(self.targets):
                y_tr,y_vl=Y_train[:,h_idx],Y_val[:,h_idx]
                logger.logging.info(
                    f"[Fold {fold['name']}] Horizon '{h_name}' (idx={h_idx}) target vectors -> "
                    f"y_tr:{y_tr.shape}, y_vl:{y_vl.shape}"
                )
                if HAS_XGB:
                    m=xgb.XGBRFRegressor(
                        n_estimators=self.model_trainer_config.TreeModelsConfig.n_estimators,
                        max_depth=self.model_trainer_config.TreeModelsConfig.max_depth,
                        learning_rate=self.model_trainer_config.TreeModelsConfig.learning_rate,
                        subsample=self.model_trainer_config.TreeModelsConfig.subsample,
                        colsample_bytree=self.model_trainer_config.TreeModelsConfig.colsample_bytree,
                        reg_alpha=self.model_trainer_config.TreeModelsConfig.reg_alpha,
                        reg_lambda=self.model_trainer_config.TreeModelsConfig.reg_lambda,
                        n_jobs=self.model_trainer_config.TreeModelsConfig.n_jobs,
                        random_state=self.model_trainer_config.TreeModelsConfig.random_state,
                        early_stopping_rounds=self.model_trainer_config.TreeModelsConfig.earlystopping_rounds
                    )
                    m.fit(
                        X_tree_train,y_tr,
                        eval_set=[(X_tree_val,y_vl)],
                        verbose=False
                    )
                    horizon_tree_models[h_idx]['xgb']=m
                    logger.logging.info(f"[Fold {fold['name']}] Horizon '{h_name}' -> XGBoost (XGBRFRegressor) trained.")
                if HAS_LGB:
                    m=lgb.LGBMRegressor(
                        n_estimators=self.model_trainer_config.TreeModelsConfig.n_estimators,
                        max_depth=self.model_trainer_config.TreeModelsConfig.max_depth,
                        learning_rate=self.model_trainer_config.TreeModelsConfig.learning_rate,
                        subsample=self.model_trainer_config.TreeModelsConfig.subsample,
                        colsample_bytree=self.model_trainer_config.TreeModelsConfig.colsample_bytree,
                        reg_alpha=self.model_trainer_config.TreeModelsConfig.reg_alpha,
                        reg_lambda=self.model_trainer_config.TreeModelsConfig.reg_lambda,
                        n_jobs=self.model_trainer_config.TreeModelsConfig.n_jobs,
                        random_state=self.model_trainer_config.TreeModelsConfig.random_state,
                        verbose=-1

                    )
                    m.fit(
                        X_tree_train,
                        y_tr,
                        # FIX: was `eval_set=[(X_tree_val, Y_val)]` — Y_val is the full multi-horizon
                        # target array; this needs the single-horizon y_vl to match X_tree_val's rows/shape.
                        eval_set=[(X_tree_val,y_vl)],
                        callbacks=[lgb.early_stopping(
                            self.model_trainer_config.TreeModelsConfig.earlystopping_rounds,
                            verbose=False
                        )]
                    )
                    horizon_tree_models[h_idx]["lgb"]=m
                    logger.logging.info(f"[Fold {fold['name']}] Horizon '{h_name}' -> LightGBM (LGBMRegressor) trained.")

                if HAS_CAT:
                    m=cb.CatBoostRegressor(
                        iterations=self.model_trainer_config.TreeModelsConfig.n_estimators,
                        depth=self.model_trainer_config.TreeModelsConfig.max_depth,
                        learning_rate=self.model_trainer_config.TreeModelsConfig.learning_rate,
                        l2_leaf_reg=self.model_trainer_config.TreeModelsConfig.l2_leaf_reg,
                        random_seed=self.model_trainer_config.TreeModelsConfig.random_state,
                        verbose=0,
                        early_stopping_rounds=self.model_trainer_config.TreeModelsConfig.earlystopping_rounds
                    )

                    m.fit(
                        # FIX: was fitting on X_tree_test (leaked test features into training,
                        # and also a row-count mismatch against y_tr). Should be the train split.
                        X_tree_train,
                        y_tr,
                        eval_set=(X_tree_val,y_vl)
                    )
                    horizon_tree_models[h_idx]["cat"]=m
                    logger.logging.info(f"[Fold {fold['name']}] Horizon '{h_name}' -> CatBoost (CatBoostRegressor) trained.")

                logger.logging.info(
                    f"[Fold {fold['name']}] Horizon '{h_name}' -> models trained so far: "
                    f"{list(horizon_tree_models[h_idx].keys())}"
                )
            # ---------------- END TREE MODELS ----------------

            logger.logging.info(
                f"[Fold {fold['name']}] horizon_tree_models summary -> "
                + ", ".join(f"h{h}({self.targets[h]}): {list(v.keys())}" for h, v in horizon_tree_models.items())
            )

            mlp_val_preds=[]
            with torch.no_grad():
                for x_seq,x_curr,_,_,_ in val_loader:
                    x_curr=x_curr.to(device)
                    x_seq=x_seq.to(device)
                    preds,_=model(x_seq,x_curr)
                    mlp_val_preds.append(preds.cpu().numpy())

            Y_pred_val_mlp=np.concatenate(mlp_val_preds,axis=0)
            logger.logging.info(f"[Fold {fold['name']}] Y_pred_val_mlp (deep-model predictions on val set) -> shape={Y_pred_val_mlp.shape}")

            meta_weights={}

            # ---------------- META-MODEL: BLEND MLP + TREE MODELS PER HORIZON ----------------
            for h_idx in range(len(self.targets)):
                cols=[Y_pred_val_mlp[:,h_idx]]
                order=["mlp"]
                for key in ["xgb","lgb","cat"]:
                    if key in horizon_tree_models[h_idx]:
                        cols.append(horizon_tree_models[h_idx][key].predict(X_tree_val))
                        order.append(key)

                val_pred_matrix=np.column_stack(cols)
                logger.logging.info(
                    f"[Fold {fold['name']}] Horizon idx {h_idx} ('{self.targets[h_idx]}') meta-model input -> "
                    f"val_pred_matrix:{val_pred_matrix.shape}, columns_order={order}"
                )
                meta_model=Ridge(alpha=5.0,fit_intercept=False)
                meta_model.fit(val_pred_matrix,Y_val[:,h_idx])
                w=meta_model.coef_

                if w.sum()>0:
                    w=w/w.sum()
                meta_weights[h_idx]={
                    "weights":w.tolist(),
                    "order":order
                }
                logger.logging.info(
                    f"[Fold {fold['name']}] Horizon idx {h_idx} ('{self.targets[h_idx]}') "
                    f"meta_weights -> {meta_weights[h_idx]}"
                )
            # ---------------- END META-MODEL ----------------

            logger.logging.info(f"[Fold {fold['name']}] Full meta_weights dict -> {meta_weights}")

            mlp_test_preds=[]
            model.eval()
            with torch.no_grad():  # FIX: was missing eval()/no_grad(), unlike the identical val-loop above
                for x_seq,x_curr,_,_,_ in test_loader:
                    preds,_=model(x_seq.to(device),x_curr.to(device))
                    mlp_test_preds.append(preds.cpu().numpy())

            Y_pred_test_mlp=np.concatenate(mlp_test_preds,axis=0)
            logger.logging.info(f"[Fold {fold['name']}] Y_pred_test_mlp (deep-model predictions on test set) -> shape={Y_pred_test_mlp.shape}")


            Y_test_final_preds=np.zeros_like(Y_test)

            for h_idx in range(len(self.targets)):
                cols=[Y_pred_test_mlp[:,h_idx]]
                for key in ["xgb","lgb","cat"]:
                    if key in horizon_tree_models[h_idx]:
                        cols.append(horizon_tree_models[h_idx][key].predict(X_tree_test))

                w=np.array(meta_weights[h_idx]["weights"])
                Y_test_final_preds[:,h_idx]=np.dot(np.column_stack(cols), w[:len(cols)])

            logger.logging.info(
                f"[Fold {fold['name']}] Y_test_final_preds (blended mlp+trees, per meta_weights) -> "
                f"shape={Y_test_final_preds.shape}"
            )

            test_ics = self.compute_ic_per_horizon(Y_test_final_preds, Y_test)
            logger.logging.info(f"Fold {fold['name']} Test IC: " + ", ".join(f"{t}: {ic:+.4f}" for t, ic in zip(self.targets, test_ics)))

            backtest_df = pd.DataFrame({
            "date": date_test, "symbol": sym_test,
            # FIX: was Y_test_final_preds[:, 1] while actual_return_1d used Y_test[:, 0] —
            # aligned both to horizon index 0. Change back to 1 if that offset was intentional.
            "pred_signal": Y_test_final_preds[:, 0],
            "actual_return_1d": Y_test[:, 0]
            })

            # STEP LOG: backtest_df is the table the daily long/short simulation below
            # actually operates on -- one row per (date, symbol) in the test set.
            logger.logging.info(
                f"[Fold {fold['name']}] backtest_df built -> shape={backtest_df.shape}, "
                f"columns={list(backtest_df.columns)}, dtypes={backtest_df.dtypes.to_dict()}"
            )
            logger.logging.info(f"[Fold {fold['name']}] backtest_df head:\n{backtest_df.head(3).to_string()}")

            unique_days = sorted(backtest_df["date"].unique())
            logger.logging.info(
                f"[Fold {fold['name']}] unique_days -> count={len(unique_days)}, "
                f"first={unique_days[0] if unique_days else None}, last={unique_days[-1] if unique_days else None}"
            )

            gross_rets, net_rets = [], []
            prev_longs, prev_shorts = set(), set()
            top_n = 20
            tx_cost_bps = 10

            # ---------------- DAILY LONG/SHORT BACKTEST SIMULATION ----------------
            for day_idx, day in enumerate(unique_days):
                day_df = backtest_df[backtest_df["date"] == day]
                if len(day_df) < 50:
                    continue

                sorted_df = day_df.sort_values("pred_signal", ascending=False)
                longs = set(sorted_df.head(top_n)["symbol"])
                shorts = set(sorted_df.tail(top_n)["symbol"])

                long_ret = sorted_df[sorted_df["symbol"].isin(longs)]["actual_return_1d"].mean()
                short_ret = sorted_df[sorted_df["symbol"].isin(shorts)]["actual_return_1d"].mean()
                gross_ret = (long_ret - short_ret) / 2.0

                long_turnover = len(longs.symmetric_difference(prev_longs)) / (2 * top_n)
                short_turnover = len(shorts.symmetric_difference(prev_shorts)) / (2 * top_n)
                # FIX: was dividing by (2 * top_n) a second time (already normalized above),
                # which silently shrank turnover/cost. Now a plain average of the two legs.
                turnover = (long_turnover + short_turnover) / 2

                cost = turnover * (tx_cost_bps / 10000.0)

                gross_rets.append(gross_ret)
                net_rets.append(gross_ret - cost)
                prev_longs, prev_shorts = longs, shorts

                # STEP LOG: one line every 50 trading days so you can see the
                # simulation progressing, without a log line per day (could be
                # hundreds of days per fold).
                if day_idx % 50 == 0:
                    logger.logging.info(
                        f"[Fold {fold['name']}] Backtest day {day_idx+1}/{len(unique_days)} ({day}) -> "
                        f"day_df_rows={len(day_df)}, longs={len(longs)}, shorts={len(shorts)}, "
                        f"gross_ret={gross_ret:+.5f}, turnover={turnover:.4f}, cost={cost:.6f}"
                    )
            # ---------------- END DAILY BACKTEST ----------------

            logger.logging.info(
                f"[Fold {fold['name']}] Backtest loop finished -> "
                f"trading_days_used={len(gross_rets)} out of unique_days={len(unique_days)} "
                f"(days skipped for having <50 names: {len(unique_days)-len(gross_rets)})"
            )

            if len(gross_rets) == 0:
                gross_stats, net_stats = None, None
                logger.logging.warning(f"[Fold {fold['name']}] No tradeable days found -> gross/net stats set to None.")
            else:
                gross_stats = self.summarize_returns(gross_rets, "Gross")
                net_stats = self.summarize_returns(net_rets, f"Net {tx_cost_bps}bps")

            joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
            for h_idx in range(len(self.targets)):
                for key, m in horizon_tree_models[h_idx].items():
                    joblib.dump(m, os.path.join(fold_dir, f"{key}_h{h_idx}.joblib"))
            with open(os.path.join(fold_dir, "meta_weights.json"), "w") as f:
                json.dump(meta_weights, f, indent=2)

            logger.logging.info(
                f"[Fold {fold['name']}] Saved to {fold_dir}: scaler.joblib, "
                f"{sum(len(v) for v in horizon_tree_models.values())} tree-model file(s), meta_weights.json"
            )

            fold_result = {
            "fold_name": fold["name"],
            "val_year": fold["val_year"],
            "test_year": fold["test_year"],
            "test_ic_by_horizon": dict(zip(self.targets, test_ics)),
            "gross": gross_stats,
            "net": net_stats,
            "fold_dir": fold_dir
        }

            logger.logging.info(f"[Fold {fold['name']}] Fold complete -> result={ {k: v for k, v in fold_result.items() if k != 'fold_dir'} }")
            return fold_result

        except Exception as e:
            raise MyException(e,sys)


    def initiate_model_trainer(self)->ModelTrainerArtifactEntity:
        """
        Top-level entry point:
          1. Load the raw walk-forward CSV
          2. Run every fold (run_fold) and collect their result dicts
          3. Aggregate IC / Sharpe / CumRet across all folds
          4. Write a walkforward_summary.json
          5. Promote the LAST fold's artifacts to a "production_deployable" folder
        """
        logger.logging.info("Starting model training pipepine.")
        try:
            raw_data_path = self.walk_forward_fold_artifact.walk_forward_dir
            df = pd.read_csv(raw_data_path)
            df["date"] = pd.to_datetime(df["date"])

            logger.logging.info(
                f"[initiate_model_trainer] Loaded raw data from '{raw_data_path}' -> "
                f"shape={df.shape}, columns={list(df.columns)}, "
                f"date_range=({df['date'].min()} to {df['date'].max()}), "
                f"unique_symbols={df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}"
            )

            all_results=[]

            for i,fold in enumerate(self.folds):
                res = self.run_fold(df=df, fold=fold, fold_idx=i, total_folds=len(self.folds))
                if res is not None:
                    all_results.append(res)
                else:
                    logger.logging.info(f"[initiate_model_trainer] Fold {i+1}/{len(self.folds)} ('{fold['name']}') returned None -> skipped.")

            logger.logging.info(
                f"[initiate_model_trainer] Completed {len(all_results)}/{len(self.folds)} folds successfully."
            )

            if not all_results:
                raise RuntimeError("All training folds were skipped. Check data boundary structures.")

            logger.logging.info(f"WALK-FORWARD SUMMARY ACROSS {len(all_results)} FOLDS")

            ic_table = {t: [] for t in self.targets}
            sharpe_net, cumret_net = [], []

            for r in all_results:
                for t in self.targets:
                    ic_table[t].append(r["test_ic_by_horizon"][t])
                if r["net"]:
                    sharpe_net.append(r["net"]["sharpe"])
                    cumret_net.append(r["net"]["cumulative_return"])

            logger.logging.info(
                f"[initiate_model_trainer] Raw ic_table collected across folds -> {ic_table}"
            )
            logger.logging.info(
                f"[initiate_model_trainer] sharpe_net values -> {sharpe_net}, cumret_net values -> {cumret_net}"
            )

            agg_ic = {}
            for t in self.targets:
                vals = np.array(ic_table[t])
                agg_ic[t] = {"mean": float(vals.mean()), "std": float(vals.std())}
                logger.logging.info(f"  {t} Out-Of-Sample IC: {vals.mean():+.4f} +/- {vals.std():.4f}")
            summary_path = os.path.join(self.model_trainer_config.model_trainer_directory, "walkforward_summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "aggregate": {
                        "targets_ic": agg_ic,
                        "mean_sharpe_net": float(np.mean(sharpe_net)) if sharpe_net else 0.0,
                        "mean_cum_ret_net": float(np.mean(cumret_net)) if cumret_net else 0.0
                    },
                    "folds_detail": [
                        {k: v for k, v in r.items() if k != "fold_dir"} for r in all_results
                    ]
                }, f, indent=4)

            logger.logging.info(f"[initiate_model_trainer] Wrote walk-forward summary to {summary_path}")


            production_dir = os.path.join(self.model_trainer_config.model_trainer_directory, "production_deployable")
            os.makedirs(production_dir, exist_ok=True)

            latest_fold_dir = all_results[-1]["fold_dir"]
            logger.logging.info(f"Promoting latest fold artifacts ({latest_fold_dir}) to {production_dir}")

            copied_files = []
            for fname in os.listdir(latest_fold_dir):
                src = os.path.join(latest_fold_dir, fname)
                dst = os.path.join(production_dir, fname)
                if os.path.isfile(src):
                    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                        fdst.write(fsrc.read())
                    copied_files.append(fname)

            logger.logging.info(
                f"[initiate_model_trainer] Promoted {len(copied_files)} file(s) to {production_dir} -> {copied_files}"
            )

            artifact = ModelTrainerArtifactEntity(
                trained_model_path=os.path.join(production_dir, "best_hybrid_deep.pth"),

            )
            logger.logging.info(f"[initiate_model_trainer] Returning ModelTrainerArtifactEntity -> model_path={artifact.model_path}")
            return artifact

        except Exception as e:
            raise MyException(e,sys)