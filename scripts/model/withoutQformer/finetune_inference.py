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
import json 
import sys
from typing import List, Dict, Any, Tuple

# --- Configuration (Must match training configuration) ---
CHECKPOINT_DIR = "checkpoints_multimodal_qa" 
DATA_DIR = "webnlg"  
GRAPH_IMG_DIR = os.path.join(DATA_DIR, "graphs") 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32 
HIDDEN_DIM_GAT = 64
FINAL_EMB_DIM = 768 
S_TRANSFORMER_MODEL = "all-MiniLM-L6-v2" 
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
SEQ2SEQ_MODEL_NAME = "facebook/bart-base"
MAX_QUESTION_LENGTH = 64 
MAX_ANSWER_LENGTH = 128

# --- Paths for Inference ---
TEST_JSON = os.path.join(DATA_DIR, "webnlg_graph_qa_test.json")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
RESULTS_FILE = "qa_inference_results.json" 

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
        print(f"FATAL: QA JSON file not found at {json_path}. Please check the path.")
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

# --- Custom Dataset for Inference ---
class WebNLGQADataset(torch.utils.data.Dataset):
    def __init__(self, qa_data, st_model, vit_processor, graph_img_root, tokenizer):
        self.data = qa_data
        self.graph_img_root = graph_img_root
        self.st_model = st_model
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer 
        self.cache = {} 
        self.EMBEDDING_DIM_ST = st_model.get_sentence_embedding_dimension()

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

    def __getitem__(self, index: int) -> Tuple[Data, torch.Tensor, torch.Tensor, str, str, Dict[str, Any]]:
        sample = self.data[index]
        graph_data = self.get_graph_features(sample)
        
        # Adjust split folder logic for inference (assuming 'test' folder)
        split_folder = "graphs_test"
        raw_idx = str(sample['index_str'])
        image_path = os.path.join(self.graph_img_root, split_folder, f"webnlg_pydot_{raw_idx}.png")
        
        image_tensor = torch.zeros(3, 224, 224)
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                vision_inputs = self.vit_processor(images=image, return_tensors="pt")
                image_tensor = vision_inputs['pixel_values'].squeeze(0)
            except Exception:
                pass 

        if graph_data is None:
            # Create dummy graph for empty input
            graph_data = Data(x=torch.zeros(1, self.EMBEDDING_DIM_ST), edge_index=torch.empty((2, 0), dtype=torch.long))

        graph_data = graph_data.to(DEVICE)
        image_tensor = image_tensor.to(DEVICE)
        
        question_tokens = self.tokenizer(
            sample["question"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_QUESTION_LENGTH
        ).input_ids.squeeze(0).to(DEVICE)
        
        return graph_data, image_tensor, question_tokens, sample["answer"], sample["target"], sample


# --- Custom Collate Function for Inference ---
def collate_fn_qa_inference(batch: List[Tuple[Data, torch.Tensor, torch.Tensor, str, str, Dict[str, Any]]]
                  ) -> Tuple[Batch, torch.Tensor, torch.Tensor, List[str], List[str], List[Dict[str, Any]]]:
    graph_data_list = [item[0] for item in batch]
    image_tensors = [item[1] for item in batch]
    question_token_ids = [item[2] for item in batch]
    answers = [item[3] for item in batch]
    targets = [item[4] for item in batch]
    samples_metadata = [item[5] for item in batch]

    batched_graph = Batch.from_data_list(graph_data_list)
    batched_images = torch.stack(image_tensors, dim=0)
    batched_question_tokens = torch.stack(question_token_ids, dim=0) 

    return batched_graph, batched_images, batched_question_tokens, answers, targets, samples_metadata

# --- Graph Encoder ---
class GraphEncoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, gat_heads: int = 4):
        super().__init__()
        # PyG modules are dynamically imported/instantiated
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


# --- Multimodal Graph-to-Text Model (QA Version) ---
class MultimodalGraphToTextQA(nn.Module):
    def __init__(self, graph_encoder: GraphEncoder, vit_model: ViTModel, hidden_dim_graph: int, final_dim: int):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.vit_model = vit_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_NAME)
        self.bart_embedding = self.transformer.get_input_embeddings() 
        
        self.graph_projection = nn.Linear(hidden_dim_graph, final_dim)
        
        # Stage 1: Original Graph + Vision Fusion (Input size: 1536)
        self.G_V_fusion = nn.Linear(final_dim * 2, final_dim) 
        
        # Stage 2: Question Integration (Input size: 1536)
        self.final_fusion_layer = nn.Linear(final_dim * 2, final_dim) 

    def forward(self, batched_graph: Batch, image_tensor: torch.Tensor, 
                question_tokens: torch.Tensor, answer: List[str] = None) -> Tuple[torch.Tensor, Any]:
        
        # 1. Feature Extraction (Frozen components in no_grad)
        with torch.no_grad(): 
            graph_emb_64 = self.graph_encoder(batched_graph)
            graph_emb_768 = self.graph_projection(graph_emb_64)

            vit_outputs = self.vit_model(pixel_values=image_tensor)
            vision_emb_768 = vit_outputs.last_hidden_state[:, 0, :]
            
            question_token_embeddings = self.bart_embedding(question_tokens)
            question_embedding = torch.mean(question_token_embeddings, dim=1) 
            
        # 2. Stage 1 Fusion (G + V)
        GV_combined = torch.cat((graph_emb_768, vision_emb_768), dim=-1)
        GV_fused = torch.tanh(self.G_V_fusion(GV_combined)) 

        # 3. Stage 2 Fusion (G_V_fused + Question)
        final_combined = torch.cat((GV_fused, question_embedding), dim=-1)
        fused_emb = torch.tanh(self.final_fusion_layer(final_combined)) 
        
        encoder_input = fused_emb.unsqueeze(1) 
        
        # 4. Text Generation (Skip loss calculation for inference)
        if answer is None:
             return encoder_input, None
        else:
            # This path is for loss calculation only (not used in pure inference script)
            tokenized_output = self.tokenizer(answer, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_ANSWER_LENGTH)
            labels = tokenized_output.input_ids.to(DEVICE)
            outputs = self.transformer(inputs_embeds=encoder_input, labels=labels)
            return outputs.loss, outputs


# --- Inference Function ---
def generate_answers(model: MultimodalGraphToTextQA, test_loader: DataLoader, device: torch.device):
    model.eval()
    all_results = []
    
    print(f"\n--- Starting Inference on {len(test_loader.dataset)} samples ---")
    
    inference_iterator = tqdm(test_loader, desc="Generating Answers", leave=False)
    
    with torch.no_grad():
        for batched_graph, img_tensor, question_tokens, ground_truth_answer, ground_truth_target, samples_metadata in inference_iterator:
            
            # 1. Get the single fused encoder input vector
            encoder_input_vector, _ = model(
                batched_graph=batched_graph.to(device), 
                image_tensor=img_tensor,
                question_tokens=question_tokens,
                answer=None 
            )

            # 2. BART Generation
            generated_ids = model.transformer.generate(
                inputs_embeds=encoder_input_vector,
                max_length=MAX_ANSWER_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

            # 3. Decode the generated IDs
            generated_answers = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 4. Store Results
            for i in range(len(generated_answers)):
                result = samples_metadata[i].copy()
                result['predicted_answer'] = generated_answers[i].strip()
                result['ground_truth_answer'] = ground_truth_answer[i]
                result['ground_truth_sentence'] = ground_truth_target[i]
                all_results.append(result)

    print("--- Inference Complete ---")
    return all_results

# --- Main Execution for Inference ---
if __name__ == "__main__":
    
    # Initialize fixed models
    ST_MODEL = SentenceTransformer(S_TRANSFORMER_MODEL).to(DEVICE).eval()
    VIT_PROCESSOR = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    VIT_MODEL = ViTModel.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
    BART_TOKENIZER = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME)
    EMBEDDING_DIM_ST = ST_MODEL.get_sentence_embedding_dimension()
    
    print(f"Using device: {DEVICE}")

    # 1. Initialize Model Structure
    graph_encoder_model = GraphEncoder(EMBEDDING_DIM_ST, HIDDEN_DIM_GAT).to(DEVICE)
    model = MultimodalGraphToTextQA(
        graph_encoder=graph_encoder_model,
        vit_model=VIT_MODEL,
        hidden_dim_graph=HIDDEN_DIM_GAT,
        final_dim=FINAL_EMB_DIM
    ).to(DEVICE)
    
    # 2. Load Best Weights
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"FATAL: Best model checkpoint not found at {BEST_MODEL_PATH}. Exiting.")
        sys.exit(1)

    print(f"Loading best model weights from: {BEST_MODEL_PATH}")
    model_state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    
    # Load state dict strictly, assuming the final fine-tuned model structure is correct
    model.load_state_dict(model_state_dict, strict=True) 
    model.eval()
    
    # 3. Load Test Data
    test_qa_data = load_qa_data(TEST_JSON, split_name='test')
    test_dataset = WebNLGQADataset(test_qa_data, ST_MODEL, VIT_PROCESSOR, GRAPH_IMG_DIR, BART_TOKENIZER)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_qa_inference)
    print(f"Loaded {len(test_dataset)} test samples.")

    # 4. Run Inference
    results = generate_answers(model, test_loader, DEVICE)

    # 5. Save Results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Inference completed. Results saved to {RESULTS_FILE}")