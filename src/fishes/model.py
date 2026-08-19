import torch
from torch import nn

class Regressor(nn.Module):
    """
    Shallow 2-layer MLP regressor for prediction from features.
    """
    def __init__(self, num_feature):
        super(Regressor, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(num_feature, 256))
        self.reg = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.reg(x)

class Regressor2(nn.Module):
    """
    Deeper 3-layer MLP regressor for prediction from concatenated features.
    """
    def __init__(self, num_feature):
        super(Regressor2, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(num_feature, 512))
        self.layer_2 = nn.Sequential(nn.Linear(512, 256))
        self.reg = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.reg(x)

class EmbeddingEntityRegressor(nn.Module):
    """
    Efficient regressor that combines pre-computed text embeddings
    with learned entity embeddings for PRODUCT_TYPE_ID.
    """
    def __init__(self, text_feature_dim, embedding_dim, num_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.regressor = Regressor2(text_feature_dim + embedding_dim)

    def forward(self, text_emb, type_id):
        cat_emb = self.embedding(type_id)
        x = torch.cat([text_emb, cat_emb], dim=1)
        return self.regressor(x)

class TransformerEntityRegressor(nn.Module):
    """
    End-to-end transformer + entity embedding regressor.
    Finetunes the transformer model directly.
    """
    def __init__(self, transformer_name, embedding_dim, num_embeddings):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.num_feature = self.transformer.config.hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.regressor = Regressor2(self.num_feature + embedding_dim)

    def forward(self, string, type_id, device):
        inp = self.tokenizer(
            string,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inp = {k: v.to(device) for k, v in inp.items()}
        output = self.transformer(**inp)
        cls = output[0][:, 0, :] # Extract CLS embedding
        
        cat_emb = self.embedding(type_id.to(device))
        x = self.regressor(torch.cat([cls, cat_emb], dim=1))
        return x
