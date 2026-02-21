# src/eval_lstm.py
import torch
from rouge_score import rouge_scorer


def ids_to_text(ids, idx2word):
    return " ".join(idx2word.get(i, "<unk>") for i in ids if i not in (0, 2))


def evaluate_rouge(model, dataset, vocab, device, num_samples=10000):
    idx2word = {v: k for k, v in vocab.items()}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=False)

    r1_scores, r2_scores = [], []
    model.eval()

    for i in range(min(num_samples, len(dataset))):
        x, y = dataset[i]
        seq_len = len(x)
        prefix_len = max(1, seq_len * 3 // 4)

        prefix_ids = x[:prefix_len].tolist()
        target_ids = y[prefix_len:].tolist()

        generated_ids = model.generate(prefix_ids, max_new_tokens=len(target_ids), eos_id=2)
        generated_ids = generated_ids[prefix_len:]

        reference = ids_to_text(target_ids, idx2word)
        hypothesis = ids_to_text(generated_ids, idx2word)

        if not reference.strip() or not hypothesis.strip():
            continue

        scores = scorer.score(reference, hypothesis)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)

    print(f"ROUGE-1: {sum(r1_scores)/len(r1_scores):.4f}")
    print(f"ROUGE-2: {sum(r2_scores)/len(r2_scores):.4f}")
    return r1_scores, r2_scores
