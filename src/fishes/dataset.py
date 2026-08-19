import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import math
import re

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower().strip()
    text = re.sub(r'[\u00d7\u2715]', ' x ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextEEDataset(Dataset):
    """
    Dataset that returns combined clean text, product type ID mapped index, and scaled length target.
    Used for extracting embeddings or end-to-end training.
    """
    def __init__(self, data_or_path, id_to_ind, default_ind, transform=True, test=False, mean=6.5502, std=0.9601):
        self.test = test
        self.transform = transform
        self.id_to_ind = id_to_ind
        self.default_ind = default_ind
        self.mean = mean
        self.std = std
        
        if isinstance(data_or_path, str):
            self.data = pd.read_csv(data_or_path)
        else:
            self.data = data_or_path.copy()
            
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title = str(row.get('TITLE', ''))
        bullet = str(row.get('BULLET_POINTS', ''))
        desc = str(row.get('DESCRIPTION', ''))
        type_id = row.get('PRODUCT_TYPE_ID', -1)
        
        # Clean and combine text
        title_c = clean_text(title)
        bullet_c = clean_text(bullet)
        desc_c = clean_text(desc)
        string = f"Title: {title_c}, Bullet Points: {bullet_c}, Description: {desc_c}"
        
        # Map product type ID to index
        type_idx = self.id_to_ind.get(type_id, self.default_ind)
        
        if not self.test:
            length = float(row.get('PRODUCT_LENGTH', 1.0))
            if self.transform:
                # Log target scaling with clipping to prevent extremely large values
                length = (min(math.log(max(length, 1.0)), 12) - self.mean) / self.std
            return string, type_idx, np.float32(length)
            
        return string, type_idx

class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed text embeddings (numpy array) and target lengths.
    Used for extremely fast training of the MLP hybrid regressor.
    """
    def __init__(self, embeddings_path_or_array, targets_df, id_to_ind, default_ind, mean=6.5502, std=0.9601):
        self.mean = mean
        self.std = std
        self.id_to_ind = id_to_ind
        self.default_ind = default_ind
        
        if isinstance(embeddings_path_or_array, str):
            self.embeddings = np.load(embeddings_path_or_array)
        else:
            self.embeddings = embeddings_path_or_array
            
        self.targets = targets_df.reset_index(drop=True)
        assert len(self.embeddings) == len(self.targets), "Embeddings and targets must have same length!"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        row = self.targets.iloc[idx]
        type_id = row.get('PRODUCT_TYPE_ID', -1)
        type_idx = self.id_to_ind.get(type_id, self.default_ind)
        
        embedding = self.embeddings[idx]
        
        # Handle training vs testing
        if 'PRODUCT_LENGTH' in row:
            length = float(row['PRODUCT_LENGTH'])
            length_scaled = (min(math.log(max(length, 1.0)), 12) - self.mean) / self.std
            return embedding, type_idx, np.float32(length_scaled)
            
        return embedding, type_idx
