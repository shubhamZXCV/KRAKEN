from collections import Counter
from statistics import mean, median
from datasets import load_dataset
import spacy
import re  # NEW (for splitting predicate labels into words)

data_files = {
    "train":      "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/validation/0000.parquet",
    "test":       "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/test/0000.parquet",
}

# Load parquet directly (works with datasets>=3.x)
ds = load_dataset("parquet", data_files=data_files)

# Text column is "target" in GEM/web_nlg
def get_texts(split_ds):
    if "target" in split_ds.column_names:
        return split_ds["target"]
    raise KeyError(f"No 'target' column found. Columns: {split_ds.column_names}")

texts = []
for split in ds.keys():
    texts.extend(get_texts(ds[split]))
print(f"Total texts: {len(texts):,}")

# NEW: extract predicate labels from the "input" field
def extract_predicates_from_input(entry):
    """Return a list of predicate labels from one example's 'input' field.

    Expects triples serialized like: "Subject | predicate_label | Object".
    Supports a single string or a list of such strings.
    """
    preds = []
    items = entry if isinstance(entry, list) else [entry]
    for t in items:
        if isinstance(t, str):
            parts = [p.strip() for p in t.split("|")]
            if len(parts) >= 3:
                preds.append(parts[1])
    return preds

# NEW: tokenize a predicate label into "edge words"
def edge_words_from_pred(pred):
    text = pred.replace("_", " ")
    return re.findall(r"[A-Za-z]+", text.lower())

# NEW: edge / predicate statistics from ds[*]["input"]
from collections import Counter

edge_label_freq = Counter()
edge_word_freq = Counter()
total_edges = 0
num_samples = 0

for split in ds.keys():
    dsplit = ds[split]
    if "input" not in dsplit.column_names:
        raise KeyError(f"No 'input' column in split '{split}'. Columns: {dsplit.column_names}")

    for entry in dsplit["input"]:
        num_samples += 1
        preds = extract_predicates_from_input(entry)
        total_edges += len(preds)
        edge_label_freq.update(preds)
        for p in preds:
            edge_word_freq.update(edge_words_from_pred(p))

unique_edge_labels = len(edge_label_freq)
edge_word_vocab_size = len(edge_word_freq)
avg_edges_per_sample = (total_edges / num_samples) if num_samples else 0.0
top20_edge_labels = edge_label_freq.most_common(20)
top20_edge_words = edge_word_freq.most_common(20)

# Fast tokenization + POS with spaCy (disable heavy pipes)
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
token_freq = Counter()
pos_counts = Counter()
total_tokens = 0

for doc in nlp.pipe(texts, batch_size=512):
    for tok in doc:
        if tok.is_alpha:  # keep alphabetic tokens only
            token = tok.text.lower()
            token_freq[token] += 1
            pos_counts[tok.pos_] += 1
            total_tokens += 1

vocab_size = len(token_freq)
common_nouns = pos_counts.get("NOUN", 0)   # common nouns
proper_nouns = pos_counts.get("PROPN", 0)  # proper nouns

# Sentence-length stats (per sample, alpha tokens only)
lengths = []
for doc in nlp.pipe(texts, batch_size=512):
    lengths.append(sum(1 for t in doc if t.is_alpha))

avg_len = mean(lengths) if lengths else 0
med_len = median(lengths) if lengths else 0
p95_len = sorted(lengths)[int(0.95 * len(lengths)) - 1] if lengths else 0

rare_words = sum(1 for _, c in token_freq.items() if c == 1)
top20 = token_freq.most_common(20)

# Print summary
print("\n=== WEBNLG (en) STATISTICS ===")
print(f"Tokens (alpha-only): {total_tokens:,}")
print(f"Vocabulary size:     {vocab_size:,}")
print(f"Type-Token Ratio:    {vocab_size/total_tokens:.4f}")

print("\nPOS counts:")
for k in sorted(pos_counts):
    print(f"{k:>6}: {pos_counts[k]:,}")
print(f"\nCommon nouns (NOUN):  {common_nouns:,}")
print(f"Proper nouns (PROPN): {proper_nouns:,}")

print("\nSentence length (words): "
      f"avg={avg_len:.2f}, median={med_len}, p95={p95_len}")
print(f"Rare words (freq=1): {rare_words:,}")

print("\nTop 20 tokens:")
for w, c in top20:
    print(f"  {w:15s} {c}")

# Save CSV
with open("webnlg_stats.csv", "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"tokens,{total_tokens}\n")
    f.write(f"vocab_size,{vocab_size}\n")
    f.write(f"ttr,{vocab_size/total_tokens:.6f}\n")
    f.write(f"common_nouns,{common_nouns}\n")
    f.write(f"proper_nouns,{proper_nouns}\n")
    f.write(f"avg_len,{avg_len:.4f}\n")
    f.write(f"median_len,{med_len}\n")
    f.write(f"p95_len,{p95_len}\n")
    f.write(f"rare_words,{rare_words}\n")
    f.write(f"total_samples,{num_samples}\n")
    f.write(f"total_edges,{total_edges}\n")
    f.write(f"avg_edges_per_sample,{avg_edges_per_sample:.6f}\n")
    f.write(f"unique_edge_labels,{unique_edge_labels}\n")
    f.write(f"edge_word_vocab_size,{edge_word_vocab_size}\n")


print("\n=== EDGE / PREDICATE STATISTICS ===")
print(f"Total samples:              {num_samples:,}")
print(f"Total edges (triples):      {total_edges:,}")
print(f"Avg edges per sample:       {avg_edges_per_sample:.3f}")
print(f"Unique edge labels:         {unique_edge_labels:,}")
print(f"Edge-word vocabulary size:  {edge_word_vocab_size:,}")

print("\nTop 20 edge labels:")
for w, c in top20_edge_labels:
    print(f"  {w:30s} {c}")

print("\nTop 20 edge words:")
for w, c in top20_edge_words:
    print(f"  {w:15s} {c}")

print("\nSaved: webnlg_stats.csv")
