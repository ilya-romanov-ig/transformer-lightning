import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = self.model_dim // self.num_heads

        #создаём веса для K V Q
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim) # output

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, q, v, k, mask=None):
        batch_size = q.size(0)

        # линейное преобразование весов
        Q = self.w_q(q)
        V = self.w_v(v)
        K = self.w_k(k)

        # приводим тензоры для многоголовой обработки
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # вычисляем коэфы внимания
        attentions = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # применяем маску
        if mask:
            attentions = attentions.masked_fill(mask == 0, -1e9)

        # вычисляем веса
        atten_w = torch.softmax(attentions, dim=-1)
        atten_w = self.dropout(atten_w)

        output = torch.matmul(atten_w, V).transpose(1, 2).contiguous().view(
            batch_size, -1, self.model_dim
        )

        return self.w_o(output)
    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, model_dim)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, model_dim, 2).float() *
                             (-math.log(10000) / model_dim))
        
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # поэлементное сложение последовательности с позиционными кодировками
        return x + self.pe[:x.size(1), :].transpose(0, 1)
    
class FFN(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalEncoding(model_dim)

        self.attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.self_attention(
            self.attention_norm, # Q
            self.attention_norm, # V
            self.attention_norm, # K
            mask
        )

        x = x + self.dropout(self.attention_norm(x)) # Residual Connection