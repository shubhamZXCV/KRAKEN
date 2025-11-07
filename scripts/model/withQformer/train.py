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
CHECKPOINT_DIR = "checkpoints_multimodal_g2t_shubham2" 
DATA_DIR = "webnlg"  
GRAPH_IMG_DIR = os.path.join(DATA_DIR, "graphs_train") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LEARNING_RATE = 3e-3
EPOCHS = 20
HIDDEN_DIM_GAT = 512
FINAL_EMB_DIM = 768 
S_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
SEQ2SEQ_MODEL_NAME = "facebook/bart-base"

# Logging Setup
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'training.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT, filemode='w')
logger = logging.getLogger(__name__)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logger.info(f"Using device: {DEVICE}")
logger.info(f"Batch Size set to: {BATCH_SIZE}")
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

        node_embeddings = self.st_model.encode(nodes, convert_to_tensor=True).float().to(DEVICE)
        
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
        split_folder = "graphs_train" if sample['split_name'] == 'train' else "graphs_val"
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

    batched_graph = Batch.from_data_list(graph_data_list)
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
                target_embeddings_inf = st_model.encode(targets, convert_to_tensor=True)
            target_embeddings = torch.from_numpy(target_embeddings_inf.cpu().numpy()).float().to(DEVICE)
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


# --- Data Loading ---
logger.info("Loading WebNLG QA train and validation splits...")
# Update to load the JSON files from the DATA_DIR
TRAIN_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_train.json")
VAL_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_val.json")

train_qa_data = load_qa_data(TRAIN_JSON, 'train')
val_qa_data = load_qa_data(VAL_JSON, 'val')

# Use the new QA Dataset class
train_dataset = WebNLGQADataset(train_qa_data, ST_MODEL, VIT_PROCESSOR, DATA_DIR)
val_dataset = WebNLGQADataset(val_qa_data, ST_MODEL, VIT_PROCESSOR, DATA_DIR)

# Use the new QA Collate function
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_qa)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_qa)

logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


# --- Initialization ---
graph_encoder_model = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT).to(DEVICE)

model = MultimodalGraphToText(
    graph_encoder=graph_encoder_model,
    vit_model=VIT_MODEL,
    hidden_dim_graph=HIDDEN_DIM_GAT,
    final_dim=FINAL_EMB_DIM,
    num_q_tokens=NUM_Q_TOKENS
).to(DEVICE)

# Only optimize trainable parts (Q-Former, GAT Encoder, Projection Layers)
trainable_params = list(model.graph_encoder.parameters()) + list(model.qformer_bridge.parameters()) + \
                   list(model.graph_projection.parameters()) + list(model.target_text_projection.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

# --- Validation Function (Adapted) ---
def evaluate(model, data_loader, device, desc="Validation"):
    model.eval()
    total_val_loss = 0
    val_iterator = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batched_graph, img_tensor, questions, answers , targets in val_iterator:
            loss, _ = model(
                batched_graph=batched_graph.to(device),
                image_tensor=img_tensor,
                questions=questions,
                answers=answers, # Training mode
                targets=targets, # Pass targets for fusion
                st_model=ST_MODEL # Pass the Sentence Transformer model
            )
            total_val_loss += loss.item()
            val_iterator.set_postfix({'Loss': total_val_loss / (val_iterator.n + 1)})

    avg_val_loss = total_val_loss / len(data_loader)
    return avg_val_loss

# --- Training Loop (Adapted) ---
logger.info("Starting Training...")
print("started training")

# 🌟 INTEGRATION POINT: Load checkpoint before starting the loop
# --- Checkpoint Loading Function ---
def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """
    Loads the latest or specified checkpoint to resume training.
    Returns: start_epoch, best_val_loss
    """
    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint_path = None

    # Find the latest checkpoint file by parsing the epoch number from filenames
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pt')]
    
    if checkpoint_files:
        # Regex to extract the epoch number
        epoch_numbers = []
        for f in checkpoint_files:
            match = re.search(r'epoch_(\d+)', f)
            if match:
                epoch_numbers.append((int(match.group(1)), f))
        
        if epoch_numbers:
            # Sort by epoch number and select the last one
            latest_file = max(epoch_numbers, key=lambda x: x[0])[1]
            latest_checkpoint_path = os.path.join(checkpoint_dir, latest_file)
        
    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update training parameters
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf')) # Use .get for robustness
        
        logger.info(f"✅ Resuming training from checkpoint: {latest_checkpoint_path}")
        logger.info(f"Starting Epoch: {start_epoch + 1}, Previous Best Val Loss: {best_val_loss:.4f}")
        print(f"\n✅ Resuming training from Epoch {start_epoch + 1}. Previous best validation loss: {best_val_loss:.4f}")
    else:
        logger.info("❌ No existing checkpoint found. Starting training from Epoch 1.")
        print("\n❌ No existing checkpoint found. Starting training from Epoch 1.")

    return start_epoch, best_val_loss
# ... (load_checkpoint function from your original code) ...
start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT_DIR, DEVICE)


for epoch in range(start_epoch, EPOCHS): 
    current_epoch = epoch + 1
    model.train() 
    total_train_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Epoch {current_epoch}/{EPOCHS} (Train)", leave=True)
    
    # 1. Training Phase
    for batched_graph, img_tensor, questions, answers, targets in train_iterator:
        optimizer.zero_grad()

        # Log which samples are in this batch (shorten list for readability)
        # try:
        #     logger.info(f"Training batch - epoch {current_epoch}, batch_idx {train_iterator.n}, sample_ids: {sample_ids[:8]}")
        # except Exception:
        #     logger.info(f"Training batch - epoch {current_epoch}, batch_idx {train_iterator.n}")
        
        # Pass the Sentence Transformer model for target text embedding
        loss, _ = model(
            batched_graph=batched_graph.to(DEVICE), 
            image_tensor=img_tensor,
            questions=questions,
            answers=answers, # Training mode
            targets=targets, # Pass targets for fusion
            st_model=ST_MODEL
        )
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_iterator.set_postfix({'Avg Loss': total_train_loss / (train_iterator.n + 1)})

    avg_train_loss = total_train_loss / len(train_loader)
    
    # 2. Validation Phase
    avg_val_loss = evaluate(model, val_loader, DEVICE, desc=f"Epoch {current_epoch}/{EPOCHS} (Val)")
    
    # 3. Logging and Checkpointing
    log_message = (
        f"--- Epoch {current_epoch} Complete ---\n"
        f"Average Training Loss: {avg_train_loss:.4f}\n"
        f"Average Validation Loss: {avg_val_loss:.4f}"
    )
    logger.info(log_message)
    print(f"\n{log_message}") 
    
    # Save checkpoint with the current epoch number
    checkpoint_name = f"epoch_{current_epoch}_val_loss_{avg_val_loss:.4f}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    torch.save({
        'epoch': current_epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Update and save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(model.state_dict(), best_path) 
        logger.info(f"New best model saved to {best_path}")

logger.info("Training complete.")
print("\nTraining complete. Check 'training.log' for detailed output.")