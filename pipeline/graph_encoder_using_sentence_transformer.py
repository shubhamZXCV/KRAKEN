import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# -------------------------
# 1. Load dataset (WebNLG)
# -------------------------
dataset = load_dataset("GEM/web_nlg", "en", split="train")
if dataset:
    print("✅ Dataset loaded successfully.")
else:
    print("❌ Failed to load dataset.")


# -------------------------
# 2. Sentence Transformer model
# -------------------------
# Pretrained model that converts text (node/edge labels) into dense embeddings
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = st_model.get_sentence_embedding_dimension()
print(f"✅ Loaded SentenceTransformer with embedding dim = {embedding_dim}")


# -------------------------
# 3. Parse triples into graph (nodes + edges)
# -------------------------
def parse_triples(example):
    """
    Parse a dataset entry into nodes and edges for graph construction.
    Example["input"] is a list of strings: ["A | r | B", "B | r2 | C", ...]
    Returns:
        nodes      - list of unique node names
        edges      - list of relation strings
        src_list   - source node names
        dst_list   - destination node names
    """
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
# 4. Graph Encoder with GAT
# -------------------------
class GraphEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, gat_heads=4):
        super().__init__()

        # Graph Attention Layers
        self.gat1 = GATConv(emb_dim, hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True)

        # Attention pooling (aggregates node embeddings → graph embedding)
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

    def forward(self, node_feats, edge_index):
        """
        node_feats : (num_nodes, emb_dim) tensor of node embeddings
        edge_index : (2, num_edges) tensor with [src_indices, dst_indices]
        """
        x = F.elu(self.gat1(node_feats, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Attention pooling (pool across all nodes → graph embedding)
        graph_emb = self.att_pool(
            x,
            index=torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        return graph_emb


# -------------------------
# 5. Example Run
# -------------------------
# Pick one example graph
sample = dataset[35406]
print("Sample entry:", sample["input"])

# Parse triples → get nodes, edges, src/dst
nodes, edges, src, dst = parse_triples(sample)

# Map node names to local indices (0 ... len(nodes)-1)
node_list = list(nodes)
node_name_to_local_idx = {n: i for i, n in enumerate(node_list)}

# Build edge_index tensor (shape: [2, num_edges])
src_local_idx = [node_name_to_local_idx[s] for s in src]
dst_local_idx = [node_name_to_local_idx[d] for d in dst]
edge_index = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long)

# -------------------------
# 6. Node embeddings from SentenceTransformer
# -------------------------
# Each node is represented by its semantic embedding
node_embeddings = st_model.encode(node_list, convert_to_tensor=True)  # shape: (num_nodes, emb_dim)

# If you want edge embeddings, you can also do:
# edge_embeddings = st_model.encode(edges, convert_to_tensor=True)
# (but vanilla GATConv ignores edge features, you'd need RGCNConv or GATv2Conv for that)


# -------------------------
# 7. Run Graph Encoder
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = GraphEncoder(embedding_dim, hidden_dim=64).to(device)

# Ensure node embeddings + edge_index are on same device
node_embeddings = node_embeddings.to(device)
edge_index = edge_index.to(device)

# Forward pass
graph_emb = model(node_embeddings, edge_index)

print("Graph embedding shape:", graph_emb.shape)
print("Graph embedding vector:", graph_emb)

