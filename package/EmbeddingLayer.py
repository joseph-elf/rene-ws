import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, context_window, d_emb):
        super().__init__()

        self.E = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_emb)
        self.P = nn.Embedding(num_embeddings=context_window,embedding_dim=d_emb)
        self.register_buffer("position", torch.arange(0,context_window))


    def forward(self, tokens):
        batch_size, context_window = tokens.shape

        #position = torch.arange(0,context_window,device=tokens.device)

        emb_vocab = self.E(tokens)
        emb_pos = self.P(self.position)[None,:,:] # non necessary but give more readable tensor

        return emb_vocab + emb_pos

