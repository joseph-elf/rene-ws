import torch
import torch.nn as nn
import torch.nn.functional as F

from package.AttentionLayer import *


class FeedForwardLayer(nn.Module):
    def __init__(self, d_emb_in,d_k, d_emb_out):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(d_emb_in,d_k),
            nn.GELU(),
            nn.Linear(d_k,d_emb_out)
        )

    def forward(self,X):
        return self.NN(X)
        





class TransformerLayer(nn.Module):
    def __init__(self, d_emb_tok,nb_heads,context_window=0,mlp_multiplication = 4):
        assert d_emb_tok%nb_heads == 0
        
        super().__init__()

        self.norm1 = nn.LayerNorm(d_emb_tok)
        self.norm2 = nn.LayerNorm(d_emb_tok)

        if nb_heads == 1:
            self.attention = SelfAttention(d_emb_in=d_emb_tok,d_k=d_emb_tok, d_emb_out=d_emb_tok,context_window=context_window)
        else :
            d_per_head = d_emb_tok // nb_heads
            self.attention = MultiHeadAttention(d_emb_in=d_emb_tok, d_k=d_per_head, d_emb_out=d_emb_tok, nb_heads=nb_heads,context_window=context_window)

        self.feed_forward = FeedForwardLayer(d_emb_in=d_emb_tok,d_k=mlp_multiplication*d_emb_tok, d_emb_out=d_emb_tok)

    def forward(self,X):
        # Attention layer
        X = X + self.attention(self.norm1(X))

        # Feed forward layer
        X = X + self.feed_forward(self.norm2(X))

        return X