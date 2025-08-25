# -*- coding: utf-8 -*-
"""
Clean, Windows-friendly WebNLG stats (no spaCy, no multiprocessing)
Buckets (computed independently):
1) NODES  : subjects & objects from triples (input)
2) EDGES  : predicates from triples (input)
3) NATLANG: target reference texts (target)
"""

from datasets import load_dataset
from collections import Counter
from statistics import mean, median
import re
import nltk

# ---- Ensure NLTK models are present (quietly) ----
def _safe_download(pkg):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[-1], quiet=True)

# Tokenizer + POS tagger (English)
#_safe_download("tokenizers/punkt")
# NLTK >=3.8 uses averaged_perceptron_tagger_eng; older uses averaged_perceptron_tagger
try:
    _safe_download("taggers/averaged_perceptron_tagger_eng")
    TAGGER_PACKAGE = "averaged_perceptron_tagger_eng"
except Exception:
    _safe_download("taggers/averaged_perceptron_tagger")
    TAGGER_PACKAGE = "averaged_perceptron_tagger"

from nltk import pos_tag_sents  # keep only what we use

# -----------------------------
# 0) Load dataset via Parquet (no loader script)
# -----------------------------
DATA_FILES = {
    "train":      "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/validation/0000.parquet",
    "test":       "https://huggingface.co/datasets/GEM/web_nlg/resolve/refs/convert/parquet/en/test/0000.parquet",
}
ds = load_dataset("parquet", data_files=DATA_FILES)

# -----------------------------
# 1) Triple parsing & tokenization helpers
# -----------------------------
ALPHA = re.compile(r"[A-Za-z]+")  # keep alpha words only

def parse_triple_line(line):
    """Split 'Subject | predicate | Object' -> (S,P,O). Return ('','','') on failure."""
    parts = [p.strip() for p in str(line).split("|")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""

def iter_triples(entry):
    """entry may be list[str] or str. Yield (S,P,O) triples."""
    if entry is None:
        return
    items = entry if isinstance(entry, list) else [entry]
    for t in items:
        s, p, o = parse_triple_line(t)
        if s or p or o:
            yield s, p, o

def label_tokens(label):
    """Tokenize node/edge labels like 'New_York_City', 'birth_place' -> ['new','york','city']"""
    return [m.group(0).lower() for m in ALPHA.finditer(str(label).replace("_", " "))]

def flatten(list_of_lists):
    for x in list_of_lists:
        for y in x:
            yield y

# Penn -> coarse POS (enough for noun stats)
def coarse_pos(tag):
    if tag in ("NN", "NNS"):
        return "NOUN"
    if tag in ("NNP", "NNPS"):
        return "PROPN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    if tag in ("PRP", "PRP$"):
        return "PRON"
    if tag == "IN":
        return "ADP"
    if tag in ("CC",):
        return "CCONJ"
    if tag in ("DT", "PDT", "WDT"):
        return "DET"
    return tag  # fallback to original

def pos_count_tokens(token_lists, batch_size=5000):
    """POS-tag a sequence of token lists (sentences). Returns (token_freq, pos_counts, total_tokens)."""
    token_freq = Counter()
    pos_counts = Counter()
    total = 0
    # tag in batches to keep memory in check
    for i in range(0, len(token_lists), batch_size):
        batch = token_lists[i : i + batch_size]
        tagged_sents = pos_tag_sents(batch, tagset=None)  # Penn tags
        for sent in tagged_sents:
            for tok, tag in sent:
                if ALPHA.fullmatch(tok):  # alpha-only
                    t = tok.lower()
                    token_freq[t] += 1
                    pos_counts[coarse_pos(tag)] += 1
                    total += 1
    return token_freq, pos_counts, total

def basic_counts(token_freq):
    vocab = len(token_freq)
    tokens = sum(token_freq.values())
    ttr = (vocab / tokens) if tokens else 0.0
    rare = sum(1 for _, c in token_freq.items() if c == 1)
    return vocab, tokens, ttr, rare

# -----------------------------
# 2) Collect raw data into three buckets
# -----------------------------
node_labels = []   # subjects + objects (raw strings)
edge_labels = []   # predicates        (raw strings)
nat_texts   = []   # target texts      (raw strings)

for split in ds.keys():
    dsplit = ds[split]
    # natural language
    if "target" not in dsplit.column_names:
        raise KeyError(f"No 'target' column in '{split}'. Columns: {dsplit.column_names}")
    nat_texts.extend(dsplit["target"])

    # nodes/edges from triples
    if "input" not in dsplit.column_names:
        raise KeyError(f"No 'input' column in '{split}'. Columns: {dsplit.column_names}")
    for entry in dsplit["input"]:
        for s, p, o in iter_triples(entry):
            if s: node_labels.append(s)
            if o: node_labels.append(o)
            if p: edge_labels.append(p)

# -----------------------------
# 3) NODES ONLY
# -----------------------------
num_node_labels      = len(node_labels)
unique_node_labels   = len(set(node_labels))
node_tokenized       = [label_tokens(lbl) for lbl in node_labels]
node_token_freq, node_pos_counts, node_total_tokens = pos_count_tokens(node_tokenized)
node_vocab, _, node_ttr, node_rare = basic_counts(node_token_freq)

# -----------------------------
# 4) EDGES ONLY
# -----------------------------
num_edges_total      = len(edge_labels)             # total predicate instances (triples)
unique_edge_labels   = len(set(edge_labels))
edge_tokenized       = [label_tokens(lbl) for lbl in edge_labels]
edge_token_freq, edge_pos_counts, edge_total_tokens = pos_count_tokens(edge_tokenized)
edge_vocab, _, edge_ttr, edge_rare = basic_counts(edge_token_freq)

# -----------------------------
# 5) NATURAL LANGUAGE ONLY
# -----------------------------
# Tokenize each text; keep alpha-only tokens

nat_tokenized = []
nat_lengths   = []
for t in nat_texts:
    toks = [m.group(0).lower() for m in ALPHA.finditer(t)]
    nat_tokenized.append(toks)
    nat_lengths.append(len(toks))

nat_token_freq, nat_pos_counts, nat_total_tokens = pos_count_tokens(nat_tokenized)
nat_vocab, _, nat_ttr, nat_rare = basic_counts(nat_token_freq)
nat_avg_len = mean(nat_lengths) if nat_lengths else 0.0
nat_med_len = median(nat_lengths) if nat_lengths else 0
nat_p95_len = sorted(nat_lengths)[max(0, int(0.95 * len(nat_lengths)) - 1)] if nat_lengths else 0

# -----------------------------
# 6) PRINT SUMMARIES
# -----------------------------
print("\n=== NODES (subjects + objects) ===")
print(f"Node labels (count):        {num_node_labels:,}")
print(f"Unique node labels:         {unique_node_labels:,}")
print(f"Node tokens (alpha-only):   {node_total_tokens:,}")
print(f"Node vocab size:            {node_vocab:,}")
print(f"Node TTR:                   {node_ttr:.4f}")
print(f"Node rare words (freq=1):   {node_rare:,}")
print("Node POS (coarse):")
for k in sorted(node_pos_counts):
    print(f"  {k:>6}: {node_pos_counts[k]:,}")

print("\n=== EDGES (predicates) ===")
print(f"Edges (triples count):      {num_edges_total:,}")
print(f"Unique edge labels:         {unique_edge_labels:,}")
print(f"Edge tokens (alpha-only):   {edge_total_tokens:,}")
print(f"Edge vocab size:            {edge_vocab:,}")
print(f"Edge TTR:                   {edge_ttr:.4f}")
print(f"Edge rare words (freq=1):   {edge_rare:,}")
print("Edge POS (coarse):")
for k in sorted(edge_pos_counts):
    print(f"  {k:>6}: {edge_pos_counts[k]:,}")

print("\n=== NATURAL LANGUAGE (target) ===")
print(f"Texts (count):              {len(nat_texts):,}")
print(f"Tokens (alpha-only):        {nat_total_tokens:,}")
print(f"Vocab size:                 {nat_vocab:,}")
print(f"TTR:                        {nat_ttr:.4f}")
print(f"Rare words (freq=1):        {nat_rare:,}")
print("POS (coarse):")
for k in sorted(nat_pos_counts):
    print(f"  {k:>6}: {nat_pos_counts[k]:,}")
print(f"Sentence length (words):    avg={nat_avg_len:.2f}, median={nat_med_len}, p95={nat_p95_len}")

# -----------------------------
# 7) SAVE CSVs (one per bucket)
# -----------------------------
with open("nodes_stats.csv", "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"node_labels,{num_node_labels}\n")
    f.write(f"unique_node_labels,{unique_node_labels}\n")
    f.write(f"node_tokens,{node_total_tokens}\n")
    f.write(f"node_vocab_size,{node_vocab}\n")
    f.write(f"node_ttr,{node_ttr:.6f}\n")
    for k in sorted(node_pos_counts):
        f.write(f"pos_{k},{node_pos_counts[k]}\n")
    f.write(f"node_rare_words,{node_rare}\n")

with open("edges_stats.csv", "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"edges_total,{num_edges_total}\n")
    f.write(f"unique_edge_labels,{unique_edge_labels}\n")
    f.write(f"edge_tokens,{edge_total_tokens}\n")
    f.write(f"edge_vocab_size,{edge_vocab}\n")
    f.write(f"edge_ttr,{edge_ttr:.6f}\n")
    for k in sorted(edge_pos_counts):
        f.write(f"pos_{k},{edge_pos_counts[k]}\n")
    f.write(f"edge_rare_words,{edge_rare}\n")

with open("natural_stats.csv", "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"texts_count,{len(nat_texts)}\n")
    f.write(f"tokens,{nat_total_tokens}\n")
    f.write(f"vocab_size,{nat_vocab}\n")
    f.write(f"ttr,{nat_ttr:.6f}\n")
    for k in sorted(nat_pos_counts):
        f.write(f"pos_{k},{nat_pos_counts[k]}\n")
    f.write(f"rare_words,{nat_rare}\n")
    f.write(f"avg_len,{nat_avg_len:.4f}\n")
    f.write(f"median_len,{nat_med_len}\n")
    f.write(f"p95_len,{nat_p95_len}\n")

print("\nSaved: nodes_stats.csv, edges_stats.csv, natural_stats.csv")
