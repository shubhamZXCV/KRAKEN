import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel, AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image

# --- Configuration (Must match training configuration) ---
CHECKPOINT_DIR = "checkpoints_multimodal_g2t" 
DATA_DIR = "webnlg"  
# Assuming test/val images are under 'graphs_val' for this inference example
GRAPH_IMG_DIR_INF = os.path.join(DATA_DIR, "graphs_test") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM_GAT = 64
FINAL_EMB_DIM = 768 
S_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
SEQ2SEQ_MODEL_NAME = "facebook/bart-base"

print(f"Using device: {DEVICE}")

# Initialize fixed models
ST_MODEL = SentenceTransformer(S_TRANSFORMER_MODEL).to(DEVICE).eval()
VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
VIT_MODEL = ViTModel.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
VIT_MODEL.requires_grad_(False) 
EMBEDDING_DIM_ST = ST_MODEL.get_sentence_embedding_dimension() # 384


# --- Utility Function: Graph Parsing ---
def parse_triples(triple_list):
    """Parses WebNLG triples into graph components."""
    nodes = set()
    edges = []
    src_list, dst_list = [], []
    for t in triple_list:
        parts = t.split(" | ")
        if len(parts) != 3: continue
        src, rel, dst = parts
        src, rel, dst = src.strip(), rel.strip(), dst.strip().strip('"')
        nodes.add(src)
        nodes.add(dst)
        edges.append(rel)
        src_list.append(src)
        dst_list.append(dst)
    return list(nodes), edges, src_list, dst_list


# --- Model Classes (Necessary to instantiate the model for loading weights) ---
class GraphEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, gat_heads=4):
        super().__init__()
        self.gat1 = GATConv(emb_dim, hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True)
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        )
    def forward(self, node_feats, edge_index):
        cloned_node_feats = node_feats.clone()
        x = F.elu(self.gat1(cloned_node_feats, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        graph_emb = self.att_pool(x, index=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return graph_emb 


class MultimodalGraphToText(nn.Module):
    def __init__(self, graph_encoder, vit_model, hidden_dim_graph, final_dim):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.vit_model = vit_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
        
        self.graph_projection = nn.Linear(hidden_dim_graph, final_dim)
        self.fusion_layer = nn.Linear(final_dim * 2, final_dim)
        
    def forward(self, node_embeddings, edge_index, image_tensor):
        # 1. Graph Embedding
        graph_emb_64 = self.graph_encoder(node_embeddings, edge_index)
        graph_emb_768 = self.graph_projection(graph_emb_64)

        # 2. Vision Embedding (frozen)
        with torch.no_grad(): 
            vit_outputs = self.vit_model(pixel_values=image_tensor)
            vision_emb_768 = vit_outputs.last_hidden_state[:, 0, :]
        
        # 3. Fusion
        combined_features = torch.cat((graph_emb_768, vision_emb_768), dim=-1)
        fused_emb = torch.tanh(self.fusion_layer(combined_features)) 
        
        # Output as the single-token sequence for the encoder
        encoder_input = fused_emb.unsqueeze(1) 
        
        return encoder_input

class MultimodalInference:
    def __init__(self, checkpoint_path, graph_encoder, vit_model, device):
        # Initialize the full model structure
        self.model = MultimodalGraphToText(
            graph_encoder=graph_encoder, 
            vit_model=vit_model, 
            hidden_dim_graph=HIDDEN_DIM_GAT, 
            final_dim=FINAL_EMB_DIM
        ).to(device)
        
        # Load trained weights
        try:
            # Load only the model state dictionary for the best model
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Successfully loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Ensure the checkpoint file exists and the model definition matches the training code.")
            
        self.model.eval()
        self.transformer = self.model.transformer
        self.tokenizer = self.model.tokenizer
        self.device = device
        
    @torch.no_grad()
    def generate(self, node_embeddings, edge_index, image_path, max_length=128, num_beams=5):
        
        # --- 1. Prepare Vision Input ---
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")

        vision_inputs = VIT_PROCESSOR(images=image, return_tensors="pt")
        image_tensor = vision_inputs['pixel_values'].to(self.device)
            
        # --- 2. Get Fused Embedding (Encoder Input) ---
        # Call the model's forward pass to get the custom encoder input sequence
        encoder_input = self.model(
            node_embeddings=node_embeddings.to(self.device), 
            edge_index=edge_index.to(self.device), 
            image_tensor=image_tensor
        )
        # encoder_input shape: (1, 1, 768)
        
        # --- 3. Run BART Generation ---
        # Pass the fused embedding as the encoder's input sequence embeddings
        generated_ids = self.transformer.generate(
            inputs_embeds=encoder_input,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        
        # --- 4. Decode and Return ---
        generated_text = self.tokenizer.decode(
            generated_ids.squeeze(), 
            skip_special_tokens=True
        )
        return generated_text

# --- Example Inference Execution ---

# 1. Instantiate the GAT Encoder (must be separate for the Inference class init)
inference_graph_encoder = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT)

# 2. Initialize the Inference Handler
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

if not os.path.exists(BEST_CHECKPOINT_PATH):
    print(f"\nFATAL ERROR: Checkpoint not found at {BEST_CHECKPOINT_PATH}. Run training first.")
else:
    inference_handler = MultimodalInference(
        checkpoint_path=BEST_CHECKPOINT_PATH,
        graph_encoder=inference_graph_encoder,
        vit_model=VIT_MODEL, # Already initialized and frozen
        device=DEVICE
    )

    # 3. Load a Sample (e.g., the first sample from the validation set)
    val_raw_data = load_dataset("GEM/web_nlg", "en", split="test")
    
    INFERENCE_INDEX_IN_SPLIT = 12 # Example: Take the very first validation sample
    sample_inference = val_raw_data[INFERENCE_INDEX_IN_SPLIT]
    
    # Assuming val images are named sequentially starting from 0 (index 0 for val split)
    image_path_inference = os.path.join(GRAPH_IMG_DIR_INF, f"webnlg_pydot_{INFERENCE_INDEX_IN_SPLIT}.png") 

    # 4. Prepare Graph Inputs for the Sample
    nodes, _, src, dst = parse_triples(sample_inference["input"])
    
    node_list = list(nodes)
    node_name_to_local_idx = {n: i for i, n in enumerate(node_list)}

    src_local_idx = [node_name_to_local_idx[s] for s in src]
    dst_local_idx = [node_name_to_local_idx[d] for d in dst]
    edge_index_inf = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long).to(DEVICE)

    # Node embeddings (using the frozen SentenceTransformer)
    node_embeddings_inf = ST_MODEL.encode(node_list, convert_to_tensor=True).float().to(DEVICE)

    # 5. Run Generation
    print(f"\n--- Starting Inference for Validation Sample {INFERENCE_INDEX_IN_SPLIT} ---")
    print("Input Triples:", sample_inference["input"])

    generated_text = inference_handler.generate(
        node_embeddings=node_embeddings_inf,
        edge_index=edge_index_inf,
        image_path=image_path_inference,
        num_beams=4 # Use a moderate beam size for better quality
    )

    print("\n✅ Ground Truth Text:")
    print(f"   {sample_inference['target']}")
    print("\n📝 Generated Text:")
    print(f"   {generated_text}")
