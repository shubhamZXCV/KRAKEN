import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, AttentionalAggregation 
from torch_geometric.data import Data, Batch 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel, AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm 
import logging 
import re # Used for parsing checkpoint file names

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints_multimodal_g2t" 
DATA_DIR = "webnlg"  
GRAPH_IMG_DIR = os.path.join(DATA_DIR, "graphs_train") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32 
LEARNING_RATE = 5e-5
EPOCHS = 20
HIDDEN_DIM_GAT = 64
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

# Initialize fixed models
ST_MODEL = SentenceTransformer(S_TRANSFORMER_MODEL).to(DEVICE).eval()
VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
VIT_MODEL = ViTModel.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
VIT_MODEL.requires_grad_(False) 
EMBEDDING_DIM_ST = ST_MODEL.get_sentence_embedding_dimension()


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


# --- Custom Dataset ---
class WebNLGDataset(Dataset):
    def __init__(self, raw_data, graph_img_dir, st_model, vit_processor, split_name):
        self.data = raw_data
        self.graph_img_dir = graph_img_dir
        self.st_model = st_model
        self.vit_processor = vit_processor
        self.cache = {} 
        self.split_name = split_name

    def __len__(self):
        return len(self.data)

    def get_graph_features(self, index):
        if index in self.cache:
            return self.cache[index]

        sample = self.data[index]
        nodes, _, src, dst = parse_triples(sample["input"])
        
        node_embeddings = self.st_model.encode(nodes, convert_to_tensor=True).float().to(DEVICE)
        
        node_name_to_local_idx = {n: i for i, n in enumerate(nodes)}
        src_local_idx = [node_name_to_local_idx[s] for s in src]
        dst_local_idx = [node_name_to_local_idx[d] for d in dst]
        edge_index = torch.tensor([src_local_idx, dst_local_idx], dtype=torch.long).to(DEVICE)
        
        target = sample["target"][0] if isinstance(sample["target"], list) else sample["target"]
        
        graph_data = Data(x=node_embeddings, edge_index=edge_index)

        features = (graph_data, target)
        self.cache[index] = features
        return features

    def __getitem__(self, index):
        graph_data, target = self.get_graph_features(index)

        original_index = self.data[index]['index']
        
        if self.split_name == 'validation':
             current_img_dir = os.path.join(DATA_DIR, "graphs_val")
        else:
             current_img_dir = os.path.join(DATA_DIR, "graphs_train")
             
        image_path = os.path.join(current_img_dir, f"webnlg_pydot_{original_index}.png") 
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            logger.warning(f"Image not found at {image_path}. Returning zeros.")
            image_tensor = torch.zeros(1, 3, 224, 224) 
            return graph_data, image_tensor.squeeze(0), target
            
        vision_inputs = self.vit_processor(images=image, return_tensors="pt")
        image_tensor = vision_inputs['pixel_values'].squeeze(0)
        
        return graph_data, image_tensor.to(DEVICE), target


# --- Custom Collate Function (PyG Batching) ---
def collate_fn(batch):
    graph_data_list = [item[0] for item in batch]
    image_tensors = [item[1] for item in batch]
    targets = [item[2] for item in batch] 

    batched_graph = Batch.from_data_list(graph_data_list)
    batched_images = torch.stack(image_tensors, dim=0)

    return batched_graph, batched_images, targets


# --- Graph Encoder ---
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


# --- Multimodal Graph-to-Text Model ---
class MultimodalGraphToText(nn.Module):
    def __init__(self, graph_encoder, vit_model, hidden_dim_graph, final_dim):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.vit_model = vit_model
        self.tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.graph_projection = nn.Linear(hidden_dim_graph, final_dim)
        self.fusion_layer = nn.Linear(final_dim * 2, final_dim)
        
    def forward(self, batched_graph, image_tensor, target=None):
        graph_emb_64 = self.graph_encoder(batched_graph)
        graph_emb_768 = self.graph_projection(graph_emb_64)

        with torch.no_grad(): 
            vit_outputs = self.vit_model(pixel_values=image_tensor)
            vision_emb_768 = vit_outputs.last_hidden_state[:, 0, :]
        
        combined_features = torch.cat((graph_emb_768, vision_emb_768), dim=-1)
        fused_emb = torch.tanh(self.fusion_layer(combined_features)) 
        
        encoder_input = fused_emb.unsqueeze(1) 
        
        labels = None
        if target:
            tokenized_output = self.tokenizer(
                target, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=128
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


# --- Data Loading ---
logger.info("Loading official WebNLG train and validation splits...")
train_raw_data = load_dataset("GEM/web_nlg", "en", split="train")
val_raw_data = load_dataset("GEM/web_nlg", "en", split="validation") 

SUBSET_TRAIN_SIZE = 35425 
SUBSET_VAL_SIZE = 1666

train_subset_data = train_raw_data.select(range(SUBSET_TRAIN_SIZE)).add_column("index", list(range(SUBSET_TRAIN_SIZE)))
val_subset_data = val_raw_data.select(range(SUBSET_VAL_SIZE)).add_column("index", list(range(SUBSET_VAL_SIZE)))

train_dataset = WebNLGDataset(train_subset_data, GRAPH_IMG_DIR, ST_MODEL, VIT_PROCESSOR, split_name='train')
val_dataset = WebNLGDataset(val_subset_data, GRAPH_IMG_DIR, ST_MODEL, VIT_PROCESSOR, split_name='validation')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


# --- Initialization ---
graph_encoder_model = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT).to(DEVICE)

model = MultimodalGraphToText(
    graph_encoder=graph_encoder_model,
    vit_model=VIT_MODEL,
    hidden_dim_graph=HIDDEN_DIM_GAT,
    final_dim=FINAL_EMB_DIM
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Validation Function ---
def evaluate(model, data_loader, device, desc="Validation"):
    model.eval()
    total_val_loss = 0
    val_iterator = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batched_graph, img_tensor, target in val_iterator:
            loss, _ = model(
                batched_graph=batched_graph.to(device),
                image_tensor=img_tensor,
                target=target
            )
            total_val_loss += loss.item()
            val_iterator.set_postfix({'Loss': total_val_loss / (val_iterator.n + 1)})

    avg_val_loss = total_val_loss / len(data_loader)
    return avg_val_loss

# --- Training Loop ---
logger.info("Starting Training...")
print("started training")

# 🌟 INTEGRATION POINT: Load checkpoint before starting the loop
start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT_DIR, DEVICE)

# Loop starts from the epoch *after* the last completed epoch (start_epoch + 1)
for epoch in range(start_epoch, EPOCHS): 
    current_epoch = epoch + 1
    model.train() 
    total_train_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Epoch {current_epoch}/{EPOCHS} (Train)", leave=True)
    
    # 1. Training Phase
    for batched_graph, img_tensor, target in train_iterator:
        optimizer.zero_grad()
        
        loss, _ = model(
            batched_graph=batched_graph.to(DEVICE), 
            image_tensor=img_tensor,
            target=target
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