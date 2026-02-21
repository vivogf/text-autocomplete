import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)

    def generate(self, prefix_ids, max_new_tokens=10, eos_id=2):
        self.eval()
        ids = list(prefix_ids)
        device = next(self.parameters()).device
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([ids], dtype=torch.long)
                logits = self.forward(x)
                next_id = logits[0, -1].argmax().item()
                if next_id == eos_id:
                    break
                ids.append(next_id)
        return ids
