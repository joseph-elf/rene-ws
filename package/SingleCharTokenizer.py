import numpy as np
import string
import pickle
from collections import Counter


#(string.printable)


class SingleCharTokenizer:
    def __init__(self,text = []):
        if len(text)==0:
            self.vocabulary = sorted(set(string.printable))
        else :
            self.vocabulary = sorted(set(text))
        self.vocab_to_tok = {v:t for t,v in enumerate(self.vocabulary)}
        self.tok_to_vocab = {t:v for t,v in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)

    def encode(self,text):
        return [tok for tok in (self.vocab_to_tok.get(word,None) for word in text) if tok is not None]
    
    def decode(self,tokens):
        return ''.join([self.tok_to_vocab[tok] for tok in tokens])
    

    def save_tokens(self,tokens,filename = "tokens.tok"):
        all = {
            "vocabulary" : self.vocabulary,
            "tokens" : tokens
        }
        with open(filename, "wb") as f:
            pickle.dump(all, f)
    
    def load_tokens(self,filename = "tokens.tok"):
        with open(filename, "rb") as f:
            all = pickle.load(f)
        self.__init__(all["vocabulary"])
        return all["tokens"]



    
    def inspect_tokens(self, text, top_k=20):

        tokens = self.encode(text)

        print("=== BASIC INFO ===")
        print("Number of characters:", len(text))
        print("Number of tokens:", len(tokens))
        print("Vocabulary size:", self.vocab_size)
        print()

        print("=== VOCABULARY ===")
        print(self.vocabulary)
        print()

        counter = Counter(tokens)
        print("=== MOST COMMON TOKENS ===")
        for tok, count in counter.most_common(top_k):
            word = repr(self.tok_to_vocab.get(tok))
            print(f"{word:6} -> {count}")
        print()

        print("=== RAREST TOKENS ===")
        for tok, count in sorted(counter.items(), key=lambda x: x[1])[:top_k]:
            char = repr(self.tok_to_vocab.get(tok))
            print(f"{char:6} -> {count}")
            
        return tokens
    