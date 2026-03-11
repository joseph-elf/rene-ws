from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self,tokens,context_window,sliding_windows = 0):
        super().__init__()
        self.tokens = tokens
        self.context_window = context_window
        c = int ( sliding_windows * context_window ) 
        if c<1:
            c=1
        elif c>context_window : 
            c = context_window
        self.windows_step = c


    def __len__(self):
        return (len(self.tokens)-self.context_window)// self.windows_step + 1


    def __getitem__(self, index):
        start = index * self.windows_step
        x = self.tokens[start:start+self.context_window]
        y = self.tokens[start+1:start+self.context_window+1]
        return x, y
    







