# src/eval_transformer_pipeline.py
import torch
from transformers import pipeline
from rouge_score import rouge_scorer


def evaluate_transformer_rouge(dataset, vocab, num_samples=1000):
    idx2word = {v: k for k, v in vocab.items()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=False)
    generator = pipeline("text-generation", model="distilgpt2", device=0)

    r1_scores, r2_scores = [], []

    for i in range(min(num_samples, len(dataset))):
        x, y = dataset[i]
        seq_len = len(x)
        prefix_len = max(1, seq_len * 3 // 4)

        prefix_ids = x[:prefix_len].tolist()
        target_ids = y[prefix_len:].tolist()

        prefix_text = " ".join(idx2word.get(i, "") for i in prefix_ids if i not in (0, 1, 2))
        target_text = " ".join(idx2word.get(i, "") for i in target_ids if i not in (0, 1, 2))

        if not prefix_text.strip() or not target_text.strip():
            continue

        new_tokens = len(target_text.split()) + 5
        result = generator(prefix_text, max_new_tokens=new_tokens, do_sample=False,
                           truncation=True, pad_token_id=50256)
        generated_text = result[0]["generated_text"]
        hypothesis = generated_text[len(prefix_text):].strip()

        if not hypothesis:
            continue

        scores = scorer.score(target_text, hypothesis)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)

    print(f"ROUGE-1: {sum(r1_scores)/len(r1_scores):.4f}")
    print(f"ROUGE-2: {sum(r2_scores)/len(r2_scores):.4f}")
    return r1_scores, r2_scores
