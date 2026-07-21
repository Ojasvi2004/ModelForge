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
import stats
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
    

        



class TemporalEncoder(nn.Module):
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

    def __init__(self, seq_features_dim,
                 static_features_dim,
                 embed_dim,
                 hidden_dim,
                 horizons,
                 dropout):
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
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,horizons)
        )

    def forward(self,x_seq,x_curr,return_embeddings_only=False):
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

class HybridHorizonLoss(nn.Module):
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
            logger.logging.info("Initialized ModelTrainer component.")
            
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def compute_ic_per_horizon(preds, targets):
        ics = []
        for h in range(preds.shape[1]):
            ic, _ = stats.spearmanr(preds[:, h], targets[:, h])
            ics.append(float(ic) if not np.isnan(ic) else 0.0)
        return ics

    @staticmethod
    def max_drawdown(returns):
        cum = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(dd.min())

    def summarize_returns(self, returns, label):
        arr = np.array(returns, dtype=np.float64)
        cum = float(np.prod(1.0 + arr) - 1.0)
        mean = float(np.mean(arr))
        vol = float(np.std(arr) + 1e-8)
        sharpe = float((mean / vol) * np.sqrt(252))
        dd = self.max_drawdown(arr)
        logger.logging.info(f"  {label}: CumRet {cum*100:.2f}% | Sharpe {sharpe:.3f} | MaxDD {dd*100:.2f}% | Days {len(arr)}")
        return {"cumulative_return": cum, "sharpe": sharpe, "max_drawdown": dd, "days": len(arr)}
    
    def run_fold(self,df:pd.DataFrame,fold:dict,fold_idx:int,total_folds:int)->dict:
        
        try:
            logger.logging.info(f"Starting fold {fold_idx+1}/{total_folds}:{fold["name"]}")

            embargo=pd.Timedelta(days=self.max_horizon*2)
            train_end_date=pd.Timestamp(fold["train_end"])
            val_end_date=pd.Timestamp(fold["val_end"])
            test_end_date=pd.Timestamp(fold["test_end"])

            train_df = df[df["date"] < (train_end_date - embargo)].copy()
            val_df = df[df["date"]>=train_end_date and df['date']<val_end_date].copy()
            test_df=df[df["date"]>=val_end_date and df["date"]<test_end_date].copy()
            
            logger.logging.info(f"Train samples: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

            if len(train_df) < 5000 or len(val_df) < 500 or len(test_df) < 500:
                logger.logging.warning(f"Skipping fold {fold['name']} due to insufficient row counts.")
                return None
            
            all_features=list(dict.fromkeys(self.seq_features+self.curr_features))

            scaler=StandardScaler()

            train_df[all_features]=scaler.fit_transform(train_df[all_features])
            test_df[all_features]=scaler.transform(test_df[all_features])
            val_df[all_features]=scaler.transform(val_df[all_features])

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
            
            fold_dir=os.path.join(self.model_trainer_config.model_trainer_directory,
                                  fold['name'])
            
            os.makedirs(fold_dir,exist_ok=True)

            model=UnifiedDeepEncoder(
                seq_features_dim=len(self.seq_features),
                static_features_dim=len(self.curr_features),
                embed_dim=self.model_trainer_config.UnifiedDeepEncoderConfig.embed_dim,
                hidden_dim=self.model_trainer_config.UnifiedDeepEncoderConfig.hidden_dim,
                horizons=len(self.targets),
                dropout=self.model_trainer_config.UnifiedDeepEncoderConfig.dropout
            ).to(device)

            criterion=HybridHorizonLoss(alpha=self.model_trainer_config.HybridHorizonLossConfig.alpha)
            
            optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-2
            )

            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.model_trainer_config.Schedular.mode,
                factor=self.model_trainer_config.Schedular.patience
            )

            best_val_loss=float('inf')
            epochs_no_improve=0
            patience=self.model_trainer_config.patience
            epochs=self.model_trainer_config.epochs
            ckpt_path=self.model_trainer_config.saved_model_path

            for epoch in range(epochs):
                model.train()
                running_loss=0.0

                for x_seq,x_curr,y,_,_ in train_loader:
                    x_seq=x_seq.to(device)
                    y=y.to(device)
                    x_curr=x_curr.to(device)
                    optimizer.zero_grad()
                    preds,_=model(x_seq,x_curr)
                    loss=criterion(preds,y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),self.model_trainer_config.gradient_clip)
                    optimizer.step()
                    running_loss=running_loss+loss.item()

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
                        running_val_loss=running_val_loss+loss
                        all_val_preds.append(preds.cpu().numpy())
                        all_val_targets.append(y.cpu().numpy())
                
                avg_val_loss=running_val_loss/len(val_loader)
                scheduler.step(avg_val_loss)

                if avg_val_loss<best_val_loss:
                    best_val_loss=avg_val_loss
                    epochs_no_improve=0
                    torch.save(model.state_dict(),ckpt_path)
                
                else:
                    epochs_no_improve=epochs_no_improve+1
                    if epochs_no_improve>=patience:
                        break

            model.load_state_dict(torch.load(ckpt_path))
            def extract(loader):
                model.eval()
                embeddings,curr_feats,labels,symbols,dates=[],[],[],[]
                with torch.no_grad():
                    for x_seq, x_curr, y, syms, dts in loader:
                        embeds=model.forward(x_seq.to(device),x_curr.to(device),return_embeddings_only=True)
                        embeddings.append(embeds)
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

            horizon_tree_models={h:{} for h in range(len(self.targets))}

            for h_idx,h_name in enumerate(self.targets):
                return
        except Exception as e:
            raise MyException(e,sys)
        
    
