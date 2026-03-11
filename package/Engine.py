import sys

import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from package.BPETokenizer import * 
from package.TextDataset import * 
from package.GPT import * 

import matplotlib.pyplot as plt

from IPython.display import clear_output, display










class Engine:
    def __init__(self,model,tokenizer,device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        #self.loader = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4) #torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

        self.vocab_size = tokenizer.vocab_size

        self.losses = []
        self.batch_counter = 0
        self.saving_counter = 0
        self.init_prompt = "bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que "


    
    def train(self,loader,nb_epochs = 1,sampling_frequency = 1000,nb_batch_max = 10000, print_frequency = 1.):
        nb_batches = min(len(loader),nb_batch_max)

        fig, ax = plt.subplots(figsize=(6,4))
        start_time_epochs = time.time()

        self.model.train()
        for epoch in range(nb_epochs):

            start_time_batches = time.time()

            loader_iter = iter(loader)
            for batch in range(nb_batches):

                x, y = next(loader_iter)
                x, y = x.to(self.device), y.to(self.device)

                
                if (self.batch_counter)%sampling_frequency == 0 :
                    filename = f"training_historic/{self.saving_counter:04d}.w"
                    self.model.save(filename)
                    self.saving_counter += 1
                self.batch_counter += 1

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1)) # view to reshape
                loss.backward()
                self.optimizer.step()

                self.losses.append(loss.item())

                if print_frequency is not None :
                    if batch%print_frequency == 0 :
                        clear_output(wait=True)
                        #self.plot_losses(self.losses,fig,ax)
                        self.print_progress(start_time_epochs,epoch,nb_epochs,start_time_batches,batch,nb_batches)

                        sys.stdout.write(f"\n")
                        sys.stdout.write(f"Loss : {loss.item()}\n")
                        sys.stdout.flush()

                        out = self.generate(self.init_prompt,max_new_tokens=30,temperature=0.9,k=10)
                        sys.stdout.write(f"\n")
                        sys.stdout.write(f"{out[678:]}\n")
                        sys.stdout.flush()

                        


    def print_progress(self,start_time_epochs,epoch,nb_epochs,start_time_batches,batch,nb_batches,width=40):

        elapsed_batches = time.time() - start_time_batches
        progress_batches = batch / nb_batches
        filled_batches = int(width * progress_batches)
        bar_batches = "█" * filled_batches + "-" * (width - filled_batches)
        speed_batches = batch / elapsed_batches if elapsed_batches > 0 else 0
        eta_batches = (nb_batches - batch) / speed_batches if speed_batches > 0 else 0


        elapsed_epochs = time.time() - start_time_epochs
        progress_epochs = epoch / nb_epochs
        filled_epochs = int(width * progress_epochs)
        bar_epochs = "█" * filled_epochs + "-" * (width - filled_epochs)
        speed_epochs = epoch / elapsed_epochs if elapsed_epochs > 0 else 0
        eta_epochs = (nb_epochs - epoch) / speed_epochs if speed_epochs > 0 else 0


        sys.stdout.write(
            f"\r|{bar_epochs}| {epoch}/{nb_epochs} "
            f"[{elapsed_epochs:0.1f}s<{eta_epochs:0.1f}s, {speed_epochs:0.1f} it/s]"
        )
        sys.stdout.write(f"\n")
        sys.stdout.write(
            f"\r|{bar_batches}| {batch}/{nb_batches} "
            f"[{elapsed_batches:0.1f}s<{eta_batches:0.1f}s, {speed_batches:0.1f} it/s]"
        )
        sys.stdout.flush()

    def plot_losses(self,losses,fig,ax):

        ax.clear()
        ax.plot(losses)
        ax.grid(True)
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_ylim(0, max(losses)+0.1)
        
        display(fig)

    
    def loss(self,Xs,Ys): # [batch_size,context_windows]
        self.model.eval()
        loss = self.model.loss(Xs,Ys)
        return loss
    
    @torch.no_grad()
    def generate(self,prompt, max_new_tokens=100, temperature=1.0,k=50):
        self.model.eval()

        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        for _ in range(max_new_tokens):

            # If context too long, crop to block_size
            block_size = 128
            block_size = self.model.context_window
            tokens_cond = tokens[:, -block_size:]

            # Forward pass
            logits = self.model(tokens_cond)

            # Take last time step
            logits = logits[:, -1, :] / temperature

            # # Convert to probabilities
            # probs = F.softmax(logits, dim=-1)

            # # Sample next token
            # next_token = torch.multinomial(probs, num_samples=1)
        
            values, indices = torch.topk(logits, k)
            probs = F.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, 1))

            # Append
            tokens = torch.cat((tokens, next_token), dim=1)

        # Decode full sequence
        output = self.tokenizer.decode(tokens[0].tolist())
        return output





    def save(self,filename):
        torch.save(self, filename)

    def save_model(self,filename,weights_only = False):
        if weights_only:
            torch.save(self.model.state_dict(), filename)
        else :
            torch.save(self.model, filename)

    def load_model(self,filename):
        self.model = GPT.load(filename)

    @classmethod
    def load(cls, filename):
        return torch.load(filename,weights_only=False)
