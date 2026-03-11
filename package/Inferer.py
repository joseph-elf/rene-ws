import sys

import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from package.BPETokenizer import * 
from package.TextDataset import * 
from package.GPT import * 



class Inferer:
    def __init__(self,model,tokenizer,device):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        self.word_delimiter = []

        self.vocab_size = tokenizer.vocab_size
        self.block_size = self.model.context_window




        self.init_prompt = "bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que "


        self.buffer_size = 10 * self.block_size
        self.buffer_tok = self.tokenizer.encode(self.init_prompt)




        


    @torch.no_grad()
    def infer_next_token(self,input_toks, temperature=1.0,k=50):
        # If context too long, crop to block_size
        tokens_cond = input_toks[:, -self.block_size:]

        # Forward pass
        logits = self.model(tokens_cond)

        # Take last tokens (n+1)
        logits = logits[:, -1, :] / temperature

        #Restrict to the k most probable tokens
        values, indices = torch.topk(logits, k)

        #Evaluate probabilities of these k tokens
        probs = F.softmax(values, dim=-1)

        #Sample from this probability distribution
        return indices.gather(-1, torch.multinomial(probs, 1))


    @torch.no_grad()
    def infer_N_next_tokens(self,N_tokens,input_toks, temperature=1.0,k=50):
        word = torch.empty((N_tokens, 1), dtype=torch.long)

        for i in range(N_tokens):
            next_token = self.infer_next_token(input_toks, temperature=temperature,k=k)
            word[i] = next_token
            input_toks = torch.cat((input_toks, next_token), dim=1) 

        return word
    

    @torch.no_grad()
    def update_buffer(self,N_tokens, temperature=1.0,k=50):
        word = torch.empty((N_tokens, 1), dtype=torch.long)

        for i in range(N_tokens):
            next_token = self.infer_next_token(self.buffer_tok, temperature=temperature,k=k)
            word[i] = next_token
            self.buffer_tok = torch.cat((self.buffer_tok, next_token), dim=1) 

        return word

    


    

    @torch.no_grad()
    def infer_next_word(self,max_N_tokens,input_toks, temperature=1.0,k=50):
        word = torch.empty((max_N_tokens, 1), dtype=torch.long)

        for i in range(max_N_tokens):
            next_token = self.infer_next_token(input_toks, temperature=temperature,k=k)
            word[i] = next_token
            input_toks = torch.cat((input_toks, next_token), dim=1) 
            
        return word





    # @torch.no_grad()
    # def generate(self,input_toks, max_new_tokens=100, temperature=1.0,k=50):
    #     # receive input_toks as torch tensor on the proper device

    #     # Encode prompt
    #     # tokens = self.tokenizer.encode(prompt)


    #     # tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    #     for _ in range(max_new_tokens):

    #         # If context too long, crop to block_size
    #         tokens_cond = input_toks[:, -self.block_size:]

    #         # Forward pass
    #         logits = self.model(tokens_cond)

    #         # Take last tokens (n+1)
    #         logits = logits[:, -1, :] / temperature

    #         #Restrict to the k most probable tokens
    #         values, indices = torch.topk(logits, k)
    #         #Evaluate probabilities of these k tokens
    #         probs = F.softmax(values, dim=-1)
    #         #Sample from this probability distribution
    #         next_token = indices.gather(-1, torch.multinomial(probs, 1))





    #         # Append
    #         tokens = torch.cat((tokens, next_token), dim=1)






    #     # Decode full sequence
    #     output = self.tokenizer.decode(tokens[0].tolist())
    #     return output



    
    # @torch.no_grad()
    # def generate(self,prompt, max_new_tokens=100, temperature=1.0,k=50):
    #     self.model.eval()

    #     # Encode prompt
    #     tokens = self.tokenizer.encode(prompt)
    #     tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    #     for _ in range(max_new_tokens):

    #         # If context too long, crop to block_size
    #         block_size = 128
    #         block_size = self.model.context_window
    #         tokens_cond = tokens[:, -block_size:]

    #         # Forward pass
    #         logits = self.model(tokens_cond)

    #         # Take last time step
    #         logits = logits[:, -1, :] / temperature

    #         # # Convert to probabilities
    #         # probs = F.softmax(logits, dim=-1)

    #         # # Sample next token
    #         # next_token = torch.multinomial(probs, num_samples=1)
        
    #         values, indices = torch.topk(logits, k)
    #         probs = F.softmax(values, dim=-1)
    #         next_token = indices.gather(-1, torch.multinomial(probs, 1))

    #         # Append
    #         tokens = torch.cat((tokens, next_token), dim=1)

    #     # Decode full sequence
    #     output = self.tokenizer.decode(tokens[0].tolist())
    #     return output





    # def save(self,filename):
    #     torch.save(self, filename)

    # def save_model(self,filename,weights_only = False):
    #     if weights_only:
    #         torch.save(self.model.state_dict(), filename)
    #     else :
    #         torch.save(self.model, filename)

    # def load_model(self,filename):
    #     self.model = GPT.load(filename)

    # @classmethod
    # def load(cls, filename):
    #     return torch.load(filename,weights_only=False)
