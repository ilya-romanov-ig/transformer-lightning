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
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim)
        self.linear2 = nn.Linear(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformetBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FFN(model_dim, ff_dim, dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.self_attention(
            self.attention_norm(x), # Q
            self.attention_norm(x), # V
            self.attention_norm(x), # K
            mask
        )
        x = x + self.dropout(attention_output) # Residual Connection

        ffn_output = self.feed_forward(self.ffn_norm(x))
        x = x + self.dropout(ffn_output)

        return x
    
class Transformer(pl.LightningModule):
    def __init__(
            self, 
            src_vocab_size,
            target_vocab_size,
            model_dim=512,
            num_heads=8,
            num_layers=6,
            ff_dim=2048,
            dropout=0.1,
            max_seq_len=5000,
            learning_rate=1e-4
        ):
        super().__init__()
        self.save_hyperparameters()

        # ENCODER
        self.src_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.src_pos_encoding = PositionalEncoding(model_dim, max_seq_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformetBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # DECODER
        self.target_embedding = nn.Embedding(target_vocab_size, model_dim)
        self.target_pos_encoding = PositionalEncoding(model_dim, max_seq_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformetBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.self_attention_norm = nn.LayerNorm(model_dim)
        self.cross_attention_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate

        self.output_layer = nn.Linear(model_dim, target_vocab_size)

    def encode(self, src, src_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.model_dim)
        src_emb = self.src_pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        for l in self.encoder_layers:
            src_emb = l(src_emb, src_mask)

        return src_emb
    
    def decode(self, target, encoder_output, target_mask=None, encoder_output_mask=None):
        target_emb = self.src_embedding(target) * math.sqrt(self.model_dim)
        target_emb = self.src_pos_encoding(target_emb)
        target_emb = self.dropout(target_emb)

        self_attention = self.self_attention(
            self.self_attention_norm(target_emb),
            self.self_attention_norm(target_emb),
            self.self_attention_norm(target_emb),
            target_mask
        )
        target_emb = target_emb + self.dropout(self_attention)

        cross_attention = self.cross_attention(
            self.cross_attention_norm(target_emb), # Q
            self.cross_attention_norm(encoder_output), # V (encoder)
            self.cross_attention_norm(encoder_output), # K (encoder)
            encoder_output_mask
        )
        target_emb = target_emb + self.dropout(cross_attention)

        for l in self.decoder_layers:
            target_emb = l(target_emb, target_mask)

        return self.output_layer(target_emb)
    
    def forward(self, src, target, src_mask=None, target_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(target, encoder_output, target_mask, src_mask)
        return decoder_output
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch  
        logits = self(src, tgt[:, :-1])
        
        loss = nn.CrossEntropyLoss(ignore_index=0)(
            logits.view(-1, logits.size(-1)), 
            tgt[:, 1:].contiguous().view(-1)
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        return optimizer