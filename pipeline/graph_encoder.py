import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation
from datasets import load_dataset
import pickle

# -------------------------
# 1. Load vocabulary (build_webngl_vocab.py)
# -------------------------
with open("webnlg_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]

vocab_size = len(word2idx)
embedding_dim = 128

def nodes_to_indices(nodes, word2idx):
    # if not nodes in word2idx:
    #     print("Nodes not found in vocabulary")
        # raise ValueError("Nodes not found in vocabulary")
    return [word2idx[n] if n in word2idx else word2idx["<unk>"] for n in nodes]

# -------------------------
# 2. Load WebNLG dataset
# -------------------------
dataset = load_dataset("GEM/web_nlg", "en", split="train")
if dataset:
    print("Dataset loaded successfully.")
else:
    print("Failed to load dataset.")



# -------------------------
# 3. Utility: Parse triples into graph (nodes + edges)
# -------------------------
def parse_triples(example):
    """
    Parse a dataset entry into nodes and edges for graph construction.
    Example["input"] is a list of strings: ["A | r | B", "B | r2 | C", ...]
    """
    triples = []
    if isinstance(example, dict):
        triple_list = example["input"]
    elif isinstance(example, list):
        triple_list = example
    else:
        raise ValueError(f"Unexpected format: {type(example)}")

    nodes = set()
    edges = []
    src_list, dst_list = [], []

    for t in triple_list:
        parts = t.split(" | ")
        if len(parts) != 3:
            continue
        src, rel, dst = parts
        src, rel, dst = src.strip(), rel.strip(), dst.strip().strip('"')
    

        nodes.add(src)
        nodes.add(dst)
        edges.append(rel)

        src_list.append(src)
        dst_list.append(dst)

    return list(nodes), edges, src_list, dst_list




# -------------------------
# 4. Graph Encoder Model
# -------------------------
class GraphEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, gat_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # GAT for message passing
        self.gat1 = GATConv(emb_dim, hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True)

        # Attention pooling (learn weights over nodes)
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

    def forward(self, node_idxs, edge_index):
        x = self.embedding(node_idxs)

        # run GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # global attention pooling → graph embedding
        graph_emb = self.att_pool(x, index=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return graph_emb


# -------------------------
# 5. Example Run
# -------------------------
sample = dataset[19]  # take one graph
print("Sample entry:", sample["input"])  # e.g., "Amsterdam_Airport_Schiphol | runwayName | \"18R/36L 'Polderbaan'\""
# After parsing triples
nodes, edges, src, dst = parse_triples(sample)

# Map node names to local indices (0 ... len(nodes)-1)
node_list = list(nodes)
node_name_to_local_idx = {n: i for i, n in enumerate(node_list)}

src_local_idx = [node_name_to_local_idx[s] for s in src]
dst_local_idx = [node_name_to_local_idx[d] for d in dst]
edge_index = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long)

# Node features: vocab indices for each node in node_list
x = torch.tensor([word2idx.get(n, word2idx["<unk>"]) for n in node_list], dtype=torch.long)

# model
model = GraphEncoder(vocab_size, embedding_dim, hidden_dim=64)

graph_emb = model(x, edge_index)
print("Graph embedding shape:", graph_emb.shape)
print("Graph embedding:", graph_emb)
