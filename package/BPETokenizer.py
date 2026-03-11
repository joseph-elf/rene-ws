import numpy as np
import string
import pickle
from collections import Counter

import sentencepiece as spm
import os


class BPETokenizer:
    def __init__(self,text = [],vocab_size = 4000):
        if len(text) != 0:
            sentences = text.split("\n")

            spm.SentencePieceTrainer.train(
                sentence_iterator = iter(sentences),
                model_prefix="BPEtemporary",
                vocab_size=vocab_size,
                model_type='bpe',
                character_coverage=1.0,
                input_sentence_size=1000000,
                shuffle_input_sentence=True
            )

            self.sp = spm.SentencePieceProcessor()
            self.sp.load("BPEtemporary.model")
            os.remove("BPEtemporary.model")
            os.remove("BPEtemporary.vocab")


            self.vocabulary = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
            self.vocab_size = self.sp.get_piece_size()
        else :
            print("Don't forget to load your tokenizer")
            self.vocabulary = 0
            self.vocab_size = 0


    def copy(self, other):
        self.sp = other.sp
        #self.sp.load("BPEtokenizer.model")
        self.vocab_size = self.sp.vocab_size()
        self.vocabulary = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]


    def encode(self,text):
        return self.sp.encode(text, out_type=int)
    
    def decode(self,tokens):
        return self.sp.decode(tokens)
    
    def save_tokens(self, tokens,filename = "tokens.tok"):
        all = {
            "self" : self,
             "tokens" : tokens
        }
        with open(filename, "wb") as f:
            pickle.dump(all, f)
    
    def load_tokens(self,filename = "tokens.tok"):
        with open(filename, "rb") as f:
            all = pickle.load(f)
        self.copy(all["self"])
        return all["tokens"]



