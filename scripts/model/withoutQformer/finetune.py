import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, AttentionalAggregation 
from torch_geometric.data import Data, Batch 
from transformers import ViTImageProcessor, ViTModel, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer 
from PIL import Image
from tqdm import tqdm 
import logging 
import re 
import json 
import sys
from typing import List, Dict, Any, Tuple

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints_multimodal_qa" 
PRETRAINED_DIR = "checkpoints_multimodal_g2t" 
DATA_DIR = "webnlg"  
GRAPH_IMG_DIR = os.path.join(DATA_DIR, "graphs") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32 
LEARNING_RATE = 5e-5
EPOCHS = 20
HIDDEN_DIM_GAT = 64
FINAL_EMB_DIM = 768 
S_TRANSFORMER_MODEL = "all-MiniLM-L6-v2" 
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
SEQ2SEQ_MODEL_NAME = "facebook/bart-base"
MAX_QUESTION_LENGTH = 64 
MAX_ANSWER_LENGTH = 128

# Dummy Data Paths (REPLACE THESE WITH ACTUAL PATHS)
TRAIN_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_train.json")
VAL_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_val.json")

# Logging Setup
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'training_qa.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=LOG_FORMAT, filemode='w')
logger = logging.getLogger(__name__)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) 

logger.info(f"Using device: {DEVICE}")

# Initialize fixed models
ST_MODEL = SentenceTransformer(S_TRANSFORMER_MODEL).to(DEVICE).eval()
VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
VIT_MODEL = ViTModel.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
VIT_MODEL.requires_grad_(False) 
EMBEDDING_DIM_ST = ST_MODEL.get_sentence_embedding_dimension()

# Initialize BART Tokenizer globally
BART_TOKENIZER = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)


# --- Utility Function: Graph Parsing ---
def parse_triples(triple_list: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
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


# --- Utility Function: Data Loading ---
def load_qa_data(json_path: str, split_name: str) -> List[Dict[str, Any]]:
    """Loads QA data from a custom JSON structure."""
    if not os.path.exists(json_path):
        logger.error(f"FATAL: QA JSON file not found at {json_path}. Exiting.")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for entry in data:
        triples = entry.get('triples', [])
        webnlg_id = entry.get('webnlg_id', '')
        target = entry.get('sentence', '')
        
        if not webnlg_id: continue
            
        index_str = webnlg_id.split('/')[-1]
        
        for qa in entry.get('qa_pairs', []):
            question = qa.get('question') or qa.get('q') or qa.get('prompt')
            answer = qa.get('answer')
            
            if question is None or answer is None: continue
            
            samples.append({
                'webnlg_id': webnlg_id,
                'index_str': index_str,
                'triples': triples,
                'question': question,
                'answer': answer,
                'target': target,
                'split_name': split_name
            })
    return samples

# --- Custom Dataset ---
class WebNLGQADataset(torch.utils.data.Dataset):
    def __init__(self, qa_data, st_model, vit_processor, graph_img_root, tokenizer):
        self.data = qa_data
        self.graph_img_root = graph_img_root
        self.st_model = st_model
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer 
        self.cache = {} 

    def __len__(self):
        return len(self.data)

    def get_graph_features(self, sample: Dict[str, Any]) -> Data:
        nodes, _, src, dst = parse_triples(sample["triples"])
        
        if not nodes:
            return None 

        with torch.no_grad():
            node_embeddings = self.st_model.encode(nodes, convert_to_tensor=True).float().to(DEVICE)
        
        node_name_to_local_idx = {n: i for i, n in enumerate(nodes)}
        src_local_idx = [node_name_to_local_idx[s] for s in src]
        dst_local_idx = [node_name_to_local_idx[d] for d in dst]
        edge_index = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long).to(DEVICE)
        
        graph_data = Data(x=node_embeddings, edge_index=edge_index)
        return graph_data

    def __getitem__(self, index: int) -> Tuple[Data, torch.Tensor, torch.Tensor, str, str]:
        sample = self.data[index]
        graph_data = self.get_graph_features(sample)
        
        split_folder = "graphs_train" if sample['split_name'] == 'train' else "graphs_val"
        raw_idx = str(sample['index_str'])
        image_path = os.path.join(self.graph_img_root, split_folder, f"webnlg_pydot_{raw_idx}.png")
        
        image_tensor = torch.zeros(3, 224, 224)
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                vision_inputs = self.vit_processor(images=image, return_tensors="pt")
                image_tensor = vision_inputs['pixel_values'].squeeze(0)
            except Exception:
                pass # Use zeros if load fails

        if graph_data is None:
            graph_data = Data(x=torch.zeros(1, EMBEDDING_DIM_ST), edge_index=torch.empty((2, 0), dtype=torch.long))

        graph_data = graph_data.to(DEVICE)
        image_tensor = image_tensor.to(DEVICE)
        
        question_tokens = self.tokenizer(
            sample["question"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_QUESTION_LENGTH
        ).input_ids.squeeze(0).to(DEVICE)
        
        return graph_data, image_tensor, question_tokens, sample["answer"], sample["target"]


# --- Custom Collate Function ---
def collate_fn_qa(batch: List[Tuple[Data, torch.Tensor, torch.Tensor, str, str]]
                  ) -> Tuple[Batch, torch.Tensor, torch.Tensor, List[str], List[str]]:
    graph_data_list = [item[0] for item in batch]
    image_tensors = [item[1] for item in batch]
    question_token_ids = [item[2] for item in batch]
    answers = [item[3] for item in batch]
    targets = [item[4] for item in batch]

    batched_graph = Batch.from_data_list(graph_data_list)
    batched_images = torch.stack(image_tensors, dim=0)
    batched_question_tokens = torch.stack(question_token_ids, dim=0) 

    return batched_graph, batched_images, batched_question_tokens, answers, targets

# --- Graph Encoder ---
class GraphEncoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, gat_heads: int = 4):
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

    def forward(self, batched_graph: Batch) -> torch.Tensor:
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


# --- Multimodal Graph-to-Text Model (UPDATED: Two-Stage Fusion) ---
class MultimodalGraphToTextQA(nn.Module):
    def __init__(self, graph_encoder: GraphEncoder, vit_model: ViTModel, hidden_dim_graph: int, final_dim: int):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.vit_model = vit_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.bart_embedding = self.transformer.get_input_embeddings() 
        
        self.graph_projection = nn.Linear(hidden_dim_graph, final_dim)
        
        # Stage 1: Original Graph + Vision Fusion (Input size: 768 * 2 = 1536)
        self.G_V_fusion = nn.Linear(final_dim * 2, final_dim) 
        
        # Stage 2: Question Integration (Input size: G_V_fused + Question = 768 + 768 = 1536)
        self.final_fusion_layer = nn.Linear(final_dim * 2, final_dim) 

    def forward(self, batched_graph: Batch, image_tensor: torch.Tensor, 
                question_tokens: torch.Tensor, answer: List[str] = None) -> Tuple[torch.Tensor, Any]:
        
        # 1. Feature Extraction (Frozen components)
        with torch.no_grad(): 
            graph_emb_64 = self.graph_encoder(batched_graph)
            graph_emb_768 = self.graph_projection(graph_emb_64)

            vit_outputs = self.vit_model(pixel_values=image_tensor)
            vision_emb_768 = vit_outputs.last_hidden_state[:, 0, :]
            
            question_token_embeddings = self.bart_embedding(question_tokens)
            question_embedding = torch.mean(question_token_embeddings, dim=1) 
            
        # 2. Stage 1 Fusion (G + V) - Uses pre-trained/frozen weights
        GV_combined = torch.cat((graph_emb_768, vision_emb_768), dim=-1)
        GV_fused = torch.tanh(self.G_V_fusion(GV_combined)) 

        # 3. Stage 2 Fusion (G_V_fused + Question) - Uses new/frozen weights
        final_combined = torch.cat((GV_fused, question_embedding), dim=-1)
        fused_emb = torch.tanh(self.final_fusion_layer(final_combined)) 
        
        encoder_input = fused_emb.unsqueeze(1) 
        
        # 4. Text Generation 
        labels = None
        if answer is not None:
            tokenized_output = self.tokenizer(
                answer, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_ANSWER_LENGTH
            )
            labels = tokenized_output.input_ids.to(DEVICE)

        if labels is not None:
            outputs = self.transformer(
                inputs_embeds=encoder_input,
                labels=labels,
            )
            return outputs.loss, outputs
        else:
             return encoder_input, None

# --- Pretrained Weight Loading for Transfer Learning (MODIFIED for layer remapping) ---
def load_pretrained_weights(model: MultimodalGraphToTextQA, pretrained_checkpoint_path: str, device: torch.device, logger: logging.Logger):
    """
    Loads pretrained weights from a checkpoint, handling layer remapping ('fusion_layer' -> 'G_V_fusion').
    """
    if not os.path.exists(pretrained_checkpoint_path):
        logger.warning(f"❌ Pretrained checkpoint not found at: {pretrained_checkpoint_path}. Starting from scratch.")
        return

    logger.info(f"Loading pretrained weights from: {pretrained_checkpoint_path}")
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    else:
        pretrained_state_dict = checkpoint

    # --- KEY MAPPING: Remap old 'fusion_layer' to new 'G_V_fusion' ---
    remapped_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k == 'fusion_layer.weight':
            remapped_state_dict['G_V_fusion.weight'] = v
        elif k == 'fusion_layer.bias':
            remapped_state_dict['G_V_fusion.bias'] = v
        else:
            remapped_state_dict[k] = v
    
    model_state_dict = model.state_dict()
    
    # Filter keys based on shape/existence
    keys_to_load = {}
    keys_to_skip = []
    
    for name, param in remapped_state_dict.items():
        if name in model_state_dict and model_state_dict[name].shape == param.shape:
            keys_to_load[name] = param
        else:
            keys_to_skip.append(name)

    # Load compatible weights (strict=False ignores skipped keys, like the new final_fusion_layer)
    model.load_state_dict(keys_to_load, strict=False)
    
    logger.info("✅ Pretrained weights loaded successfully (compatible layers only).")
    if keys_to_skip:
        logger.info(f"⚠️ Skipped loading incompatible layers: {keys_to_skip}")
        print(f"⚠️ Skipped incompatible layers: {keys_to_skip}")


# --- Validation Function ---
def evaluate(model: MultimodalGraphToTextQA, data_loader: DataLoader, device: torch.device, desc: str = "Validation"):
    model.eval()
    total_val_loss = 0
    val_iterator = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batched_graph, img_tensor, question_tokens, answer, _ in val_iterator: 
            loss, _ = model(
                batched_graph=batched_graph.to(device),
                image_tensor=img_tensor,
                question_tokens=question_tokens,
                answer=answer 
            )
            total_val_loss += loss.item()
            val_iterator.set_postfix({'Loss': total_val_loss / (val_iterator.n + 1)})

    avg_val_loss = total_val_loss / len(data_loader)
    return avg_val_loss

# --- Main Execution ---

# 1. Data Loading 
logger.info("Loading QA data...")
train_qa_data = load_qa_data(TRAIN_JSON, split_name='train')
val_qa_data = load_qa_data(VAL_JSON, split_name='validation')

train_dataset = WebNLGQADataset(train_qa_data, ST_MODEL, VIT_PROCESSOR, GRAPH_IMG_DIR, BART_TOKENIZER)
val_dataset = WebNLGQADataset(val_qa_data, ST_MODEL, VIT_PROCESSOR, GRAPH_IMG_DIR, BART_TOKENIZER)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_qa)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_qa)

logger.info(f"Train QA samples: {len(train_dataset)}, Validation QA samples: {len(val_dataset)}")


# 2. Initialization and Freezing
graph_encoder_model = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT).to(DEVICE)

model = MultimodalGraphToTextQA(
    graph_encoder=graph_encoder_model,
    vit_model=VIT_MODEL,
    hidden_dim_graph=HIDDEN_DIM_GAT,
    final_dim=FINAL_EMB_DIM
).to(DEVICE)

# --- ❄️ FREEZING ---
logger.info("Freezing GraphEncoder parameters.")
for param in graph_encoder_model.parameters():
    param.requires_grad = False

logger.info("Freezing Graph Projection, G_V_fusion, and final_fusion_layer.")
for param in model.graph_projection.parameters():
    param.requires_grad = False
for param in model.G_V_fusion.parameters():
    param.requires_grad = False
for param in model.final_fusion_layer.parameters():
    param.requires_grad = False


# 3. Optimizer (Only BART transformer is trainable)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
logger.info(f"Optimizer initialized with only trainable parameters (BART: Encoder, Decoder, Embeddings).")


# 4. Transfer Weights and Start Training
PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, "best_model.pt") 

load_pretrained_weights(model, PRETRAINED_PATH, DEVICE, logger) 

# Start from epoch 0 for the new QA task
start_epoch, best_val_loss = 0, float('inf') 

logger.info("Starting Training for QA Fine-tuning...")
print("started training for QA fine-tuning")

# Training Loop
for epoch in range(start_epoch, EPOCHS): 
    current_epoch = epoch + 1
    model.train() 
    total_train_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Epoch {current_epoch}/{EPOCHS} (Train QA)", leave=True)
    
    # 1. Training Phase
    for batched_graph, img_tensor, question_tokens, answer, _ in train_iterator:
        optimizer.zero_grad()
        
        loss, _ = model(
            batched_graph=batched_graph.to(DEVICE), 
            image_tensor=img_tensor,
            question_tokens=question_tokens,
            answer=answer
        )
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        train_iterator.set_postfix({'Avg Loss': total_train_loss / (train_iterator.n + 1)})

    avg_train_loss = total_train_loss / len(train_loader)
    
    # 2. Validation Phase
    avg_val_loss = evaluate(model, val_loader, DEVICE, desc=f"Epoch {current_epoch}/{EPOCHS} (Val QA)")
    
    # 3. Logging and Checkpointing
    log_message = (
        f"--- Epoch {current_epoch} Complete ---\n"
        f"Average Training Loss: {avg_train_loss:.4f}\n"
        f"Average Validation Loss: {avg_val_loss:.4f}"
    )
    logger.info(log_message)
    print(f"\n{log_message}") 
    
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
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(model.state_dict(), best_path) 
        logger.info(f"New best model saved to {best_path}")


logger.info("QA Fine-tuning complete.")
print("\nQA Fine-tuning complete. Check 'training_qa.log' for detailed output.")