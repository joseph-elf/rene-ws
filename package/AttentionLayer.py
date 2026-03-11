import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_emb_in,d_k, d_emb_out,context_window = 0):
        super().__init__()

        self.d_emb_in = d_emb_in
        self.d_k = d_k
        self.d_emb_out = d_emb_out

        bias = False
        self.Wq = nn.Linear(in_features = d_emb_in,out_features = d_k, bias = bias)
        self.Wk = nn.Linear(in_features = d_emb_in,out_features = d_k, bias = bias)
        self.Wv = nn.Linear(in_features = d_emb_in,out_features = d_emb_out, bias = bias)

        if context_window>0:
            self.register_buffer("M", torch.tril(torch.ones(context_window,context_window),diagonal=0)==0)
        else :
            context_window = 1024
            self.register_buffer("M", torch.tril(torch.ones(context_window,context_window),diagonal=0)==0)

    def forward(self,X):
        batch_size, context_window, d_emb_in = X.shape

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        S = (Q @ K.transpose(-2,-1))    /   (self.d_k**0.5)   

        # Causal masking
        S = S.masked_fill(self.M[:context_window,:context_window],value=float("-inf"))

        


        # Attention
        A = F.softmax(S,dim=-1) @ V

        return A
        











class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb_in, d_k, d_emb_out, nb_heads,context_window=0):
        # Better if d_k * nb_heads = d_emb_in = d_emb_out
        super().__init__()
        
        self.d_emb_in = d_emb_in
        self.d_k = d_k
        self.d_emb_out = d_emb_out
        self.nb_heads = nb_heads


        bias = False

        self.Wq = nn.Linear(in_features = d_emb_in,out_features = nb_heads*d_k, bias = bias)
        self.Wk = nn.Linear(in_features = d_emb_in,out_features = nb_heads*d_k, bias = bias)
        self.Wv = nn.Linear(in_features = d_emb_in,out_features = nb_heads*d_k, bias = bias)

        self.Wo = nn.Linear(in_features = nb_heads*d_k,out_features = d_emb_out, bias = bias)


        if context_window>0:
            self.register_buffer("M", torch.tril(torch.ones(context_window,context_window),diagonal=0)==0)
        else :
            context_window = 1024
            self.register_buffer("M", torch.tril(torch.ones(context_window,context_window),diagonal=0)==0)
        

    def forward(self, X):
        batch_size, context_window, d_emb_in = X.shape
        # [ batch_size , context_window , d_emb_in ]



        Q = self.Wq(X).view(batch_size,context_window,self.nb_heads,self.d_k).transpose(1,2)
        K = self.Wk(X).view(batch_size,context_window,self.nb_heads,self.d_k).transpose(1,2)
        V = self.Wv(X).view(batch_size,context_window,self.nb_heads,self.d_k).transpose(1,2)
        # [ batch_size , nb_heads , context_window , d_k ]

        S = (Q @ K.transpose(-2,-1))    /   (self.d_k**0.5)


        # Causal masking
        S = S.masked_fill(self.M[:context_window,:context_window],value=float("-inf"))

        # Attention
        A = F.softmax(S,dim=-1) @ V
        # [ batch_size , nb_heads , context_window , d_k ]


        return self.Wo(   A.transpose(1,2).contiguous().view(batch_size,context_window,self.nb_heads*self.d_k)   )
        





