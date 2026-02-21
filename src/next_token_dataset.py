import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def build_vocab(texts, min_freq=3):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

class NextTokenDataset(Dataset):
    def __init__(self, texts, vocab, max_len=32):
        self.samples = []
        for text in texts:
            ids = [vocab.get(w, 1) for w in text.split()] + [2]
            if len(ids) < 2:
                continue
            ids = ids[:max_len]
            self.samples.append((ids[:-1], ids[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys
