import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .dataset import TextEEDataset, EmbeddingDataset
from .model import EmbeddingEntityRegressor

def extract_cls_embeddings(df, model_name="distilbert-base-uncased", batch_size=64, device="cpu", max_length=256):
    """
    Extract transformer CLS embeddings for a pandas DataFrame.
    """
    print(f"Loading transformer model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # We will temporarily use a dummy category mapping to construct TextEEDataset
    dummy_id_to_ind = {}
    dummy_default_ind = 0
    
    dataset = TextEEDataset(
        df, 
        id_to_ind=dummy_id_to_ind, 
        default_ind=dummy_default_ind, 
        transform=False, 
        test=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    hidden_size = model.config.hidden_size
    embeddings = np.zeros((len(df), hidden_size), dtype=np.float32)
    
    print("Extracting CLS embeddings...")
    total = 0
    with torch.no_grad():
        for strings, _ in tqdm(dataloader):
            B = len(strings)
            # Tokenize strings
            inputs = tokenizer(
                strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # CLS token representation is outputs[0][:, 0, :]
            cls_outputs = outputs[0][:, 0, :].cpu().numpy()
            embeddings[total : total + B] = cls_outputs
            total += B
            
    return embeddings

def train_mlp_regressor(
    train_emb, train_df, 
    val_emb, val_df, 
    id_to_ind, default_ind, 
    epochs=10, lr=1e-3, batch_size=128, device="cpu",
    mean=6.5502, std=0.9601
):
    """
    Train the hybrid regressor on pre-computed text embeddings + category entity embeddings.
    """
    print("Preparing Datasets...")
    train_dataset = EmbeddingDataset(train_emb, train_df, id_to_ind, default_ind, mean=mean, std=std)
    val_dataset = EmbeddingDataset(val_emb, val_df, id_to_ind, default_ind, mean=mean, std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    text_feat_dim = train_emb.shape[1]
    num_embeddings = len(id_to_ind) + 1
    
    model = EmbeddingEntityRegressor(
        text_feature_dim=text_feat_dim,
        embedding_dim=32,
        num_embeddings=num_embeddings
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    loss_fn = torch.nn.MSELoss()
    
    best_mape = float('inf')
    best_weights = None
    
    print("Training hybrid regressor...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for embs, type_idxs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            embs = embs.to(device)
            type_idxs = type_idxs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(embs, type_idxs).squeeze()
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        mape_sum = 0
        with torch.no_grad():
            for embs, type_idxs, targets in val_loader:
                embs = embs.to(device)
                type_idxs = type_idxs.to(device)
                targets = targets.to(device)
                
                outputs = model(embs, type_idxs).squeeze()
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * len(targets)
                
                # Calculate MAPE in raw target domain (convert back from log target scale)
                pred_raw = torch.exp(outputs * std + mean)
                target_raw = torch.exp(targets * std + mean)
                mape_sum += torch.sum(torch.abs(pred_raw - target_raw) / (target_raw + 1e-8)).item()
                
        val_loss /= len(val_dataset)
        val_mape = mape_sum / len(val_dataset)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAPE: {val_mape * 100:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_mape < best_mape:
            best_mape = val_mape
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"Best Validation MAPE: {best_mape * 100:.2f}%")
    model.load_state_dict(best_weights)
    return model

def predict_mlp_regressor(model, embeddings, df, id_to_ind, default_ind, batch_size=128, device="cpu", mean=6.5502, std=0.9601):
    """
    Generate raw predictions (lengths) using the trained hybrid regressor.
    """
    model.eval()
    dataset = EmbeddingDataset(embeddings, df, id_to_ind, default_ind, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = np.zeros(len(df), dtype=np.float32)
    total = 0
    with torch.no_grad():
        for embs, type_idxs in loader:
            B = len(embs)
            embs = embs.to(device)
            type_idxs = type_idxs.to(device)
            
            outputs = model(embs, type_idxs).squeeze()
            if B == 1:
                outputs = outputs.unsqueeze(0)
            # Inverse log transform to raw length
            pred_raw = torch.exp(outputs * std + mean).cpu().numpy()
            preds[total : total + B] = pred_raw
            total += B
            
    return preds
