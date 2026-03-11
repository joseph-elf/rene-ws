import torch
import torch.nn as nn
import torch.nn.functional as F

from package.AttentionLayer import *
from package.TransformerLayer import *
from package.EmbeddingLayer import *



class GPT(nn.Module):
    def __init__(self,vocab_size=256, context_window=128, d_emb=32,nb_layers=6,nb_heads=4,mlp_multiplication=3):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_window = context_window
        self.d_emb = d_emb
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.mlp_multiplication = mlp_multiplication

        self.embedding = EmbeddingLayer(vocab_size=vocab_size, context_window=context_window, d_emb=d_emb)

        self.transformer_blocks = nn.ModuleList([TransformerLayer(d_emb_tok=d_emb,nb_heads=nb_heads,context_window=context_window,mlp_multiplication=mlp_multiplication) for i in range(nb_layers)])

        self.final_norm = nn.LayerNorm(normalized_shape=d_emb)

        self.linear_head = nn.Linear(in_features=d_emb,out_features=vocab_size,bias=False)




    def forward(self,tokens):
        #batch_size, context_window = tokens.shape
        # output : batch_size x context_windows x vocab_size

        x = self.embedding(tokens)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.final_norm(x)

        logits = self.linear_head(x)

        return logits
    

    def loss(self,Xs,Ys): # [batch_size,context_windows]
        device = next(self.parameters()).device
    
        batch_size = len(Xs)

        x = Xs.to(device)
        y = Ys.to(device)

        logits = self.forward(x)
        batch_size, context_windows,vocab_size = logits.shape

        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) / batch_size

        return loss





    def save(self,filename,weights_only = False):
        if weights_only:
            torch.save(self.state_dict(), filename)
        else :
            torch.save(self, filename)

    def load_weights(self,filename):
        self.load_state_dict(torch.load(filename, weights_only=True))


    def architecture(self):
        nb_param = 0 
        print("PARAMETERS :")
        for name, param in self.named_parameters():
            c= 1
            for i in param.size():
                c *= i
            nb_param += c

            print(f"{name}: {param.device} {param.size()} {c}")
        print("")
        print("BUFFERS :")
        for name, buf in self.named_buffers():
            print(name, buf.device)

        print("")
        print(f"Total number of parameters : {nb_param}")



    

    @classmethod
    def load(cls, filename):
        return torch.load(filename,weights_only=False)







        

