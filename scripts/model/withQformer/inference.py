import torch
# from train import (
#     MultimodalGraphToText, GraphEncoder, load_qa_data, parse_triples,
#     DEVICE, HIDDEN_DIM_GAT, FINAL_EMB_DIM, NUM_Q_TOKENS,
#     EMBEDDING_DIM_ST, VIT_MODEL, ST_MODEL, VIT_PROCESSOR,
#     S_TRANSFORMER_MODEL, VIT_MODEL_NAME, SEQ2SEQ_MODEL_NAME
# )
import os
import logging
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
# from train import WebNLGQADataset, collate_fn_qa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation 
from torch_geometric.data import Data, Batch 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel, AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm 
import logging 
import re 
import os
import json # New: For loading JSON QA data
from torch.utils.data import DataLoader

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints_multimodal_g2t_shubham" 
DATA_DIR = "webnlg"  
# GRAPH_IMG_DIR = os.path.join(DATA_DIR, "graphs_train") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE = 32
LEARNING_RATE = 5e-5
# EPOCHS = 20
HIDDEN_DIM_GAT = 64
FINAL_EMB_DIM = 768 
S_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
SEQ2SEQ_MODEL_NAME = "facebook/bart-base"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ... (Configuration block from your original code) ...

# Initialize fixed models (Ensure they are on DEVICE)
# ... (ST_MODEL, VIT_PROCESSOR, VIT_MODEL initialization from your original code) ...
# Initialize fixed models
ST_MODEL = SentenceTransformer(S_TRANSFORMER_MODEL).to(DEVICE).eval()
VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
VIT_MODEL = ViTModel.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
VIT_MODEL.requires_grad_(False) 
EMBEDDING_DIM_ST = ST_MODEL.get_sentence_embedding_dimension()
# Final embedding dimension must match LLM embedding dimension (768 for BART-base)
FINAL_EMB_DIM = 768 
NUM_Q_TOKENS = 32 # Number of soft prompt tokens

# --- Utility Function: Graph Parsing ---
def parse_triples(triple_list):
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
    
class QFormerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        # Self-attention on queries
        q_res = q
        q = self.norm1(q)
        sa_out, _ = self.self_attn(q, q, q)
        q = q_res + self.dropout(sa_out)

        # Cross-attention: queries attend to kv (vision+graph)
        q_res = q
        q = self.norm2(q)
        ca_out, _ = self.cross_attn(q, kv, kv)
        q = q_res + self.dropout(ca_out)

        # FFN
        q_res = q
        q = self.norm3(q)
        q = q_res + self.dropout(self.mlp(q))
        return q

class QFormerBridge(nn.Module):
    """
    Q-Former closer to BLIP-2: stacked layers where each layer
    contains self-attn, cross-attn (to image/graph features), and FFN.
    """
    def __init__(self, num_queries=32, feature_dim=768, num_heads=8, num_layers=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, feature_dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(feature_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(feature_dim)
        self.llm_projection = nn.Linear(feature_dim, feature_dim)

    def forward(self, combined_input_features):
        # combined_input_features: (B, S, D)  -> keys/values for cross-attention
        B = combined_input_features.size(0)
        q = self.queries.expand(B, -1, -1)  # (B, num_queries, D)
        kv = combined_input_features  # (B, S, D)
        for layer in self.layers:
            q = layer(q, kv)
        q = self.final_ln(q)
        q = self.llm_projection(q)
        return q


# --- Utility Function: Load and Process QA JSON ---
# def load_qa_data(json_path, split_name):
#     """Loads QA data and flattens it into a list of QA samples."""
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     qa_samples = []
#     for entry in data:
#         # Assuming the 'triples' list is available in the JSON for graph processing
#         triples = entry['triples']
#         # The 'webnlg_id' is used to get the image index
#         index_str = entry['webnlg_id'].split('/')[-1]
        
#         for qa in entry['qa_pairs']:
#             qa_samples.append({
#                 'webnlg_id': entry['webnlg_id'],
#                 'index_str': index_str,
#                 'triples': triples,
#                 'question': qa['question'],
#                 'answer': qa['answer'],
#                 'split_name': split_name
#             })
#     return qa_samples

def load_qa_data(json_path, split_name):
    """Loads WebNLG data with target text."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for entry in data:
        triples = entry.get('triples', [])
        webnlg_id = entry.get('webnlg_id', '')
        target = entry.get('sentence', '')  # Get the target text
        
        if not webnlg_id or not target:
            logger.warning(f"Missing webnlg_id or target in entry, skipping: {entry}")
            continue
            
        index_str = webnlg_id.split('/')[-1]
        
        # Store each QA pair with the target text
        for qa in entry.get('qa_pairs', []):
            question = qa.get('question') or qa.get('q') or qa.get('prompt')
            answer = qa.get('answer')
            
            if question is None or answer is None:
                logger.warning(f"Skipping QA pair with missing question/answer in {webnlg_id}: {qa}")
                continue
            
            samples.append({
                'webnlg_id': webnlg_id,
                'index_str': index_str,
                'triples': triples,
                'question': question,
                'answer': answer,
                'target': target,  # Include the target text
                'split_name': split_name
            })
    return samples

# --- Custom Dataset (Updated) ---
class WebNLGQADataset(torch.utils.data.Dataset):
    def __init__(self, qa_data, st_model, vit_processor, graph_img_root):
        self.data = qa_data
        self.graph_img_root = graph_img_root
        self.st_model = st_model
        self.vit_processor = vit_processor
        self.cache = {} 
        # Embeddings for relations might be beneficial, but keeping node embedding for now

    def __len__(self):
        return len(self.data)

    def get_graph_features(self, sample):
        # Using the simplified parse_triples from your original code
        nodes, _, src, dst = parse_triples(sample["triples"])
        
        if not nodes:
            # Handle empty graph case if necessary, e.g., return dummy data
            return None 

        # SentenceTransformer.encode may return numpy array or torch.Tensor depending on args/version.
        node_embeddings = self.st_model.encode(nodes, convert_to_tensor=True)
        if isinstance(node_embeddings, np.ndarray):
            node_embeddings = torch.from_numpy(node_embeddings)
        if isinstance(node_embeddings, torch.Tensor):
            node_embeddings = node_embeddings.float().to(DEVICE)
        else:
            # fallback
            node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32, device=DEVICE)

        node_name_to_local_idx = {n: i for i, n in enumerate(nodes)}
        src_local_idx = [node_name_to_local_idx[s] for s in src]
        dst_local_idx = [node_name_to_local_idx[d] for d in dst]
        edge_index = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long).to(DEVICE)
        
        graph_data = Data(x=node_embeddings, edge_index=edge_index)
        return graph_data

    def __getitem__(self, index):
        sample = self.data[index]
        graph_data = self.get_graph_features(sample)

        # Determine image path (try multiple filename variants)
        split_folder = "graphs_train" if sample['split_name'] == 'train' else "graphs_test"
        raw_idx = str(sample['index_str'])
        candidates = []
        # exact as-is (current behavior)
        candidates.append(os.path.join(self.graph_img_root, split_folder, f"webnlg_pydot_{raw_idx}.png"))
        # strip common 'Id'/'id' prefix and use digits only
        m = re.search(r'(\d+)', raw_idx)
        if m:
            candidates.append(os.path.join(self.graph_img_root, split_folder, f"webnlg_pydot_{m.group(1)}.png"))
        # also try removing 'Id' if present
        candidates.append(os.path.join(self.graph_img_root, split_folder, f"webnlg_pydot_{raw_idx.replace('Id','').replace('id','')}.png"))
        # fallback: maybe file is just the index.png
        candidates.append(os.path.join(self.graph_img_root, split_folder, f"{raw_idx}.png"))

        image_path = None
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break

        if image_path is None:
            logger.warning(f"Image not found at {candidates[0]}. Tried alternatives: {candidates}. Returning zeros.")
            image_tensor = torch.zeros(3, 224, 224)
            if graph_data is None:
                graph_data = Data(x=torch.zeros(1, EMBEDDING_DIM_ST), edge_index=torch.empty((2, 0), dtype=torch.long)).to(DEVICE)
            else:
                graph_data = graph_data.to(DEVICE)
            return graph_data, image_tensor.to(DEVICE), sample["question"], sample["answer"], sample["target"]

        # Load Image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            logger.warning(f"Failed to open image at {image_path}. Returning zeros.")
            image_tensor = torch.zeros(3, 224, 224)
            if graph_data is None:
                graph_data = Data(x=torch.zeros(1, EMBEDDING_DIM_ST), edge_index=torch.empty((2, 0), dtype=torch.long)).to(DEVICE)
            else:
                graph_data = graph_data.to(DEVICE)
            return graph_data, image_tensor.to(DEVICE), sample["question"], sample["answer"], sample["target"]

        # Process Image
        vision_inputs = self.vit_processor(images=image, return_tensors="pt")
        image_tensor = vision_inputs['pixel_values'].squeeze(0).to(DEVICE)
 
        # Ensure graph_data is a valid Data on DEVICE
        if graph_data is None:
            graph_data = Data(x=torch.zeros(1, EMBEDDING_DIM_ST), edge_index=torch.empty((2, 0), dtype=torch.long)).to(DEVICE)
        else:
            graph_data = graph_data.to(DEVICE)
 
        return graph_data, image_tensor, sample["question"], sample["answer"], sample["target"]


# --- Custom Collate Function (PyG Batching) ---
def collate_fn_qa(batch):
    graph_data_list = [item[0] for item in batch]
    image_tensors = [item[1] for item in batch]
    questions = [item[2] for item in batch]
    answers = [item[3] for item in batch]
    targets = [item[4] for item in batch]  # Add targets

    batched_graph = Batch.from_data_list(graph_data_list).to(DEVICE)
    batched_images = torch.stack(image_tensors, dim=0)

    return batched_graph, batched_images, questions, answers, targets


# --- Graph Encoder (Keeping yours) ---
class GraphEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, gat_heads=4):
        super().__init__()
        self.gat1 = GATConv(emb_dim, hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * gat_heads, hidden_dim, heads=1, concat=True)
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

    def forward(self, batched_graph):
        node_feats = batched_graph.x
        edge_index = batched_graph.edge_index
        batch_map = batched_graph.batch 

        cloned_node_feats = node_feats.clone()
        x = F.elu(self.gat1(cloned_node_feats, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        graph_emb = self.att_pool(
            x,
            index=batch_map 
        )
        return graph_emb 


# --- Multimodal Graph-to-Text Model (UPDATED with Q-Former and FUSION) ---
class MultimodalGraphToText(nn.Module):
    def __init__(self, graph_encoder, vit_model, hidden_dim_graph, final_dim, num_q_tokens=32):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.vit_model = vit_model
        self.tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
        # Get the token embedding layer from the encoder for prepending text
        self.encoder_embedding = self.transformer.get_encoder().embed_tokens
        
        # Projection for GAT output (64 -> 768)
        self.graph_projection = nn.Linear(hidden_dim_graph, final_dim) 
        
        # Q-Former Bridge
        self.qformer_bridge = QFormerBridge(
            num_queries=num_q_tokens, 
            feature_dim=final_dim
        ).to(DEVICE)
        
        # Project target text embedding (ST_MODEL output dim -> 768)
        self.target_text_projection = nn.Linear(EMBEDDING_DIM_ST, final_dim)

    def forward(self, batched_graph, image_tensor, questions, answers=None, targets=None, st_model=None):
        B = image_tensor.size(0)
        
        # 1. Graph Features (GAT)
        graph_emb_64 = self.graph_encoder(batched_graph)
        graph_emb_768_vec = self.graph_projection(graph_emb_64) # (B, 768)
        
         # Use targets for text fusion instead of answers
        target_text_embeddings = None
        if targets:
            # Encode the target text using the frozen S-Transformer
            with torch.no_grad():
                target_embeddings = st_model.encode(targets, convert_to_tensor=True)
            if isinstance(target_embeddings, np.ndarray):
                target_embeddings = torch.from_numpy(target_embeddings)
            if isinstance(target_embeddings, torch.Tensor):
                target_embeddings = target_embeddings.float().to(DEVICE)
            else:
                target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32, device=DEVICE)
            target_text_embeddings = self.target_text_projection(target_embeddings)
             
            # Fusion with graph embeddings
            fused_gt_emb = (graph_emb_768_vec + target_text_embeddings) / 2
        else:
            fused_gt_emb = graph_emb_768_vec

        fused_gt_sequence = fused_gt_emb.unsqueeze(1) # (B, 1, 768)
        
        # 3. Vision Features (ViT - Frozen)
        with torch.no_grad(): 
            vit_outputs = self.vit_model(pixel_values=image_tensor)
            vision_emb_full = vit_outputs.last_hidden_state # (B, 197, 768)
        
        # 4. Prepare Input for Q-Former (Graph/Text + Vision)
        combined_features = torch.cat((fused_gt_sequence, vision_emb_full), dim=1) 
        
        # 5. Q-Former Processing
        q_tokens = self.qformer_bridge(combined_features) # (B, num_q_tokens, 768)
        
        # 6. LLM (BART) Integration (Q-Tokens + Question Embedding)
        
        # Tokenize the question
        tokenized_question = self.tokenizer(
            questions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(DEVICE)

        # Get the question token embeddings
        question_embeddings = self.encoder_embedding(tokenized_question.input_ids)
        
        # Combine Q-tokens (visual prompt) and question embeddings (prefix)
        encoder_input = torch.cat((q_tokens, question_embeddings), dim=1)
        encoder_attention_mask = torch.cat(
            (torch.ones((B, self.qformer_bridge.num_queries), dtype=torch.long, device=DEVICE), 
             tokenized_question.attention_mask), 
            dim=1
        )
        
        # 7. Training/Inference
        if answers is not None:
            # Training: Prepare Target for LLM Decoder (Suffix)
            tokenized_output = self.tokenizer(
                answers, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=128
            )
            labels = tokenized_output.input_ids.to(DEVICE)
            
            # Pass the combined soft-prompt + prefix to the LLM
            outputs = self.transformer(
                inputs_embeds=encoder_input,
                attention_mask=encoder_attention_mask,
                labels=labels,
            )
            return outputs.loss, outputs
        else:
             # Inference: Return the prepared encoder input for generation
             return encoder_input, encoder_attention_mask

def run_inference():
    # Set paths
    DATA_DIR = "webnlg"
    TEST_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_train.json")
    CHECKPOINT_PATH = os.path.join("checkpoints_multimodal_g2t_shubham", "epoch_8_val_loss_6.9627.pt")
    # CHECKPOINT_PATH = os.path.join("checkpoints_multimodal_g2t_shubham", "best_model.pt")
    BATCH_SIZE = 32

    # Load test data
    logger.info("Loading test data...")
    test_qa_data = load_qa_data(TEST_JSON, 'test')
    test_dataset = WebNLGQADataset(test_qa_data, ST_MODEL, VIT_PROCESSOR, DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_qa)
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Initialize model
    logger.info("Initializing model...")
    graph_encoder = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT).to(DEVICE)
    model = MultimodalGraphToText(
        graph_encoder=graph_encoder,
        vit_model=VIT_MODEL,
        hidden_dim_graph=HIDDEN_DIM_GAT,
        final_dim=FINAL_EMB_DIM,
        num_q_tokens=NUM_Q_TOKENS
    ).to(DEVICE)

    # Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # Support both raw state_dict or checkpoint dicts with 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                logger.warning("Failed to load checkpoint directly (%s). Trying non-strict load.", e)
                model.load_state_dict(checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    # Run inference
    logger.info("Starting inference...")
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (batched_graph, img_tensor, questions, answers, targets) in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            # Get model predictions
            encoder_input, encoder_attention_mask = model(
                batched_graph=batched_graph,
                image_tensor=img_tensor,
                questions=questions,
                targets=targets,
                st_model=ST_MODEL
            )

            # # Generate answers
            # generated_ids = model.transformer.generate(
            #     inputs_embeds=encoder_input,
            #     attention_mask=encoder_attention_mask,
            #     max_length=128,
            #     num_beams=4,
            #     length_penalty=1.0,
            #     early_stopping=True
            # )
            # Modify generation parameters for single-word answers
            generated_ids = model.transformer.generate(
            inputs_embeds=encoder_input,
            attention_mask=encoder_attention_mask,
            max_length=8,  # Reduced from 128
            min_length=1,  # Add this
            num_beams=1,   # Reduced from 4
            do_sample=False,
            temperature=1.0,
            top_p=0.9,
            early_stopping=True
            )

            # Decode generated answers
            generated_answers = model.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )

            # Store results
            for q, a, gen_a in zip(questions, answers, generated_answers):
                print(f"Q: {q}\nGT: {a}\nGen: {gen_a}\n---")
                results.append({
                    'question': q,
                    'ground_truth': a,
                    'generated': gen_a
                })

    # Save results
    import json
    output_file = os.path.join(DATA_DIR, "test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Inference complete. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    results = run_inference()
    
    # Print some examples
    print("\nExample predictions:")
    for i, res in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {res['question']}")
        print(f"Ground Truth: {res['ground_truth']}")
        print(f"Generated: {res['generated']}")
