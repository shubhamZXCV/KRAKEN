import pickle
from collections import Counter
from datasets import load_dataset

# -------------------------------
# Utility to tokenize reference texts
# -------------------------------
def tokenize_text(text: str):
    """Normal tokenization for free-form sentences."""
    return text.lower().split()

# -------------------------------
# Extract triples from GEM/web_nlg entry
# -------------------------------
def extract_from_entry(entry):
    """
    Each entry has:
      entry['input'] -> list of strings like "Germany | capital | Berlin"
      entry['target'] -> list of reference texts
    """
    triples = []
    for triple_str in entry["input"]:
        parts = triple_str.split(" | ")
        if len(parts) == 3:
            subj, pred, obj = parts
            # Strip quotes and keep as single token
            subj = subj.strip()
            pred = pred.strip()
            obj = obj.strip().strip('"')
            triples.append((subj, pred, obj))
    return triples

# -------------------------------
# Extract vocab from triples + lexicalisations
# -------------------------------
def extract_from_triples_and_lex(triples, targets, vocab_counter):
    # Add subject, predicate, object as single tokens
    for s, p, o in triples:
        vocab_counter.update([s])  # keep whole string
        vocab_counter.update([p])
        vocab_counter.update([o])

    # Add tokens from reference texts (normal tokenization)
    for ref in targets:
        vocab_counter.update(tokenize_text(ref))

# -------------------------------
# Main function: build vocab from HF dataset
# -------------------------------
def build_vocab_webnlg(output_path="webnlg_vocab.pkl"):
    vocab_counter = Counter()

    dataset = load_dataset("GEM/web_nlg", "en")  # has {train, validation, test}

    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue
        print(f"📂 Processing {split} split...")
        for entry in dataset[split]:
            triples = extract_from_entry(entry)
            targets = entry["target"]  # list of reference texts
            extract_from_triples_and_lex(triples, targets, vocab_counter)

    vocab = {
        "word2idx": {w: i for i, (w, _) in enumerate(vocab_counter.most_common(), start=2)},
        "idx2word": {i: w for i, (w, _) in enumerate(vocab_counter.most_common(), start=2)},
        "freqs": dict(vocab_counter),
    }
    vocab["word2idx"]["<pad>"] = 0
    vocab["word2idx"]["<unk>"] = 1
    vocab["idx2word"][0] = "<pad>"
    vocab["idx2word"][1] = "<unk>"

    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"✅ Vocabulary saved to {output_path}")
    print(f"📊 Vocab size: {len(vocab['word2idx'])}")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    build_vocab_webnlg("webnlg_vocab.pkl")
