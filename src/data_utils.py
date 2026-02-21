import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

def clean_string(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_split(path, seq_len=7):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    cleaned_texts = [clean_string(line) for line in raw_lines]
    cleaned_texts = [t for t in cleaned_texts if len(t.split()) >= seq_len]

    train_val, test_texts = train_test_split(cleaned_texts, test_size=0.1, random_state=42)
    train_texts, val_texts = train_test_split(train_val, test_size=0.1/0.9, random_state=42)

    return train_texts, val_texts, test_texts
