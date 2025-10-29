import torch
import torch.nn as nn
from transformers import AutoModel, GemmaForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

# --- DIAGNOSTIC STEP ---
# This will tell us exactly which version of PyTorch is being used.
print(f"Using PyTorch version: {torch.__version__}")

# A helper function to see which parts of the model are trainable
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

# --- MODEL COMPONENTS ---

class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))
    def forward(self, x):
        return self.model(x)

class QFormer(nn.Module):
    def __init__(self, num_queries, encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_hidden_size))
        qformer_config = AutoConfig.from_pretrained("bert-base-uncased")
        qformer_config.is_decoder, qformer_config.add_cross_attention = True, True
        qformer_config.hidden_size, qformer_config.encoder_hidden_size = llm_hidden_size, encoder_hidden_size
        self.cross_attention_layer = nn.TransformerDecoderLayer(d_model=llm_hidden_size, nhead=8, dim_feedforward=llm_hidden_size * 4, batch_first=True)
    def forward(self, vision_features):
        queries = self.queries.expand(vision_features.shape[0], -1, -1)
        return self.cross_attention_layer(tgt=queries, memory=vision_features)
        
class GraphTextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.dummy_layer = nn.Linear(10, output_dim)
        print("Graph/Text Encoder Initialized (Placeholder).")
    def forward(self, graph_data, text_tokens):
        dummy_input = torch.randn(graph_data.size(0), 10).to(self.dummy_layer.weight.device)
        return self.dummy_layer(dummy_input).unsqueeze(1)

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model_name = "facebook/deit-base-distilled-patch16-224"
        self.llm_model_name = "google/gemma-2b"
        
        vision_config = AutoConfig.from_pretrained(self.vision_model_name)
        llm_config = AutoConfig.from_pretrained(self.llm_model_name)
        
        # DEFINITIVE FIX: Add use_safetensors=True to both model loading calls.
        # This forces the script to use the safe file format and bypass the version check.
        self.vision_encoder = AutoModel.from_pretrained(self.vision_model_name, use_safetensors=True)
        self.llm = GemmaForCausalLM.from_pretrained(self.llm_model_name, use_safetensors=True)
        
        self.projector = MLPProjector(input_dim=vision_config.hidden_size, output_dim=llm_config.hidden_size)
        self.qformer = QFormer(num_queries=32, encoder_hidden_size=vision_config.hidden_size, llm_hidden_size=llm_config.hidden_size)
        self.graph_text_encoder = GraphTextEncoder(output_dim=llm_config.hidden_size)
        
    def forward(self, graph_data, text_instruction_tokens, image_pixels, llm_target_tokens=None):
        graph_text_embeds = self.graph_text_encoder(graph_data, text_instruction_tokens)
        vision_outputs = self.vision_encoder(pixel_values=image_pixels)
        image_patch_embeds = vision_outputs.last_hidden_state
        qformer_embeds = self.qformer(image_patch_embeds)
        projected_embeds = self.projector(image_patch_embeds)
        target_embeds = self.llm.model.embed_tokens(llm_target_tokens)
        final_input_embeds = torch.cat([graph_text_embeds, qformer_embeds, projected_embeds, target_embeds], dim=1)
        attention_mask = torch.ones(final_input_embeds.shape[:2], dtype=torch.long, device=final_input_embeds.device)
        outputs = self.llm(inputs_embeds=final_input_embeds, attention_mask=attention_mask, labels=llm_target_tokens)
        return outputs

def train_phase_2(full_task_dataset):
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("\n--- Starting Training Phase 2: End-to-End Fine-Tuning ---")
    
    model = MultimodalModel()

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
        
    lora_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model.llm = get_peft_model(model.llm, lora_config)
    
    if accelerator.is_main_process:
        print("Trainable parameters in Phase 2 (with LoRA):")
        print_trainable_parameters(model)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    
    model, optimizer = accelerator.prepare(model, optimizer)

    # ... Training loop placeholder ...
    if accelerator.is_main_process:
        print("Model setup complete. Placeholder for training loop.")
        print("--- Phase 2 Complete ---")

    accelerator.wait_for_everyone()
    unwrapped_llm = accelerator.unwrap_model(model.llm)
    unwrapped_llm.save_pretrained(
        "final_llm_lora_adapters",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

if __name__ == "__main__":
    phase2_dummy_dataset = [{
        'graph': torch.randn(1, 10), 
        'text_instr': torch.randint(0, 1000, (1, 20)),
        'image': torch.randn(1, 3, 224, 224),
        'llm_response': torch.randint(0, 1000, (1, 100))
    }]

    train_phase_2(phase2_dummy_dataset)
    
    print("\nDistributed skeleton code execution finished.")

# import torch
# import torch.nn as nn
# from transformers import AutoProcessor, AutoModel, GemmaForCausalLM, AutoConfig
# from peft import get_peft_model, LoraConfig, TaskType
# from accelerate import Accelerator # ACCELERATE CHANGE: Import Accelerator

# # A helper function to see which parts of the model are trainable
# def print_trainable_parameters(model):
#     """Prints the number of trainable parameters in the model."""
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || "
#         f"trainable%: {100 * trainable_params / all_param:.2f}"
#     )

# # --- MODEL COMPONENTS (No changes needed here, same as before) ---

# class MLPProjector(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.model = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))
#     def forward(self, x):
#         return self.model(x)

# class QFormer(nn.Module):
#     def __init__(self, num_queries, encoder_hidden_size, llm_hidden_size):
#         super().__init__()
#         self.num_queries = num_queries
#         self.queries = nn.Parameter(torch.randn(1, num_queries, llm_hidden_size))
#         qformer_config = AutoConfig.from_pretrained("bert-base-uncased")
#         qformer_config.is_decoder = True
#         qformer_config.add_cross_attention = True
#         qformer_config.hidden_size = llm_hidden_size
#         qformer_config.encoder_hidden_size = encoder_hidden_size
#         self.cross_attention_layer = nn.TransformerDecoderLayer(d_model=llm_hidden_size, nhead=8, dim_feedforward=llm_hidden_size * 4, batch_first=True)
#     def forward(self, vision_features):
#         batch_size = vision_features.shape[0]
#         queries = self.queries.expand(batch_size, -1, -1)
#         return self.cross_attention_layer(tgt=queries, memory=vision_features)
        
# class GraphTextEncoder(nn.Module):
#     def __init__(self, output_dim):
#         super().__init__()
#         self.dummy_layer = nn.Linear(10, output_dim)
#         print("Graph/Text Encoder Initialized (Placeholder).")
#     def forward(self, graph_data, text_tokens):
#         # In a real scenario, you'd move graph_data to the correct device.
#         # Accelerate handles tensors in data loaders, but manual tensors need care.
#         dummy_input = torch.randn(graph_data.size(0), 10).to(self.dummy_layer.weight.device)
#         return self.dummy_layer(dummy_input).unsqueeze(1)

# class MultimodalModel(nn.Module):
#     """
#     The main model that ties all components together as per your diagram.
#     """
#     def __init__(self):
#         super().__init__()
        
#         # --- Configs ---
#         # FINAL FIX: Use the correct, verified model ID for a distilled Vision Transformer
#         self.vision_model_name = "facebook/deit-base-distilled-patch16-224"
#         self.llm_model_name = "google/gemma-2b"
        
#         # We need to use from_pretrained with use_auth_token=True if you logged in
#         # or it will be handled automatically by the environment if you logged in via CLI
#         vision_config = AutoConfig.from_pretrained(self.vision_model_name)
#         llm_config = AutoConfig.from_pretrained(self.llm_model_name)
        
#         # --- Models ---
#         self.vision_encoder = AutoModel.from_pretrained(self.vision_model_name)
#         self.llm = GemmaForCausalLM.from_pretrained(self.llm_model_name)
        
#         # --- Bridges ---
#         # The hidden_size for DeiT is also 768, so no other changes are needed here.
#         self.projector = MLPProjector(
#             input_dim=vision_config.hidden_size, # 768 for DeiT
#             output_dim=llm_config.hidden_size   # 2048 for Gemma
#         )
#         self.qformer = QFormer(
#             num_queries=32,
#             encoder_hidden_size=vision_config.hidden_size,
#             llm_hidden_size=llm_config.hidden_size
#         )
        
#         # --- Graph/Text Path ---
#         self.graph_text_encoder = GraphTextEncoder(output_dim=llm_config.hidden_size)

#     # The forward method remains unchanged
#     def forward(self, graph_data, text_instruction_tokens, image_pixels, llm_target_tokens=None):
#         graph_text_embeds = self.graph_text_encoder(graph_data, text_instruction_tokens)
#         vision_outputs = self.vision_encoder(pixel_values=image_pixels)
#         image_patch_embeds = vision_outputs.last_hidden_state
#         qformer_embeds = self.qformer(image_patch_embeds)
#         projected_embeds = self.projector(image_patch_embeds)
#         target_embeds = self.llm.model.embed_tokens(llm_target_tokens)
#         final_input_embeds = torch.cat([graph_text_embeds, qformer_embeds, projected_embeds, target_embeds], dim=1)
#         attention_mask = torch.ones(final_input_embeds.shape[:2], dtype=torch.long, device=final_input_embeds.device)
#         outputs = self.llm(inputs_embeds=final_input_embeds, attention_mask=attention_mask, labels=llm_target_tokens)
#         return outputs

# # --- TRAINING PHASES (MODIFIED FOR ACCELERATE) ---

# def train_phase_2(full_task_dataset):
#     # ACCELERATE CHANGE: Initialize Accelerator
#     accelerator = Accelerator()
    
#     print("\n--- Starting Training Phase 2: End-to-End Fine-Tuning ---")
    
#     # --- Model Initialization ---
#     model = MultimodalModel() # No .to(device) here!

#     # --- Setup ---
#     for param in model.vision_encoder.parameters():
#         param.requires_grad = False
        
#     lora_config = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
#     model.llm = get_peft_model(model.llm, lora_config)
    
#     # We only print parameters on the main process to avoid spam
#     if accelerator.is_main_process:
#         print("Trainable parameters in Phase 2 (with LoRA):")
#         print_trainable_parameters(model)
    
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=2e-5
#     )
    
#     # ACCELERATE CHANGE: Prepare models, optimizers, and data loaders
#     # The dataloader should be defined here
#     # TODO: Create your full_task_dataloader
#     # full_task_dataloader = DataLoader(full_task_dataset, batch_size=YOUR_BATCH_SIZE)
#     # model, optimizer, full_task_dataloader = accelerator.prepare(
#     #     model, optimizer, full_task_dataloader
#     # )
#     model, optimizer = accelerator.prepare(model, optimizer)

#     # --- Training Loop ---
#     # TODO: This loop should use the prepared dataloader
#     # for epoch in range(num_epochs):
#     #     for batch in full_task_dataloader:
#     #         optimizer.zero_grad()
#     #         outputs = model(
#     #             graph_data=batch['graph'],
#     #             text_instruction_tokens=batch['text_instr'],
#     #             image_pixels=batch['image'],
#     #             llm_target_tokens=batch['llm_response']
#     #         )
#     #         loss = outputs.loss
#     #         
#     #         # ACCELERATE CHANGE: Use accelerator.backward()
#     #         accelerator.backward(loss)
#     #         
#     #         optimizer.step()
#     #         
#     #         # ACCELERATE CHANGE: Print only on the main process
#     #         if accelerator.is_main_process:
#     #             print(f"Loss: {loss.item()}")

#     print("--- Phase 2 Complete ---")
    
#     # ACCELERATE CHANGE: Save the model correctly
#     # Unwrap the model to get the original PeftModel
#     # Wait for all processes to finish before saving
#     accelerator.wait_for_everyone()
#     unwrapped_llm = accelerator.unwrap_model(model.llm)
#     unwrapped_llm.save_pretrained(
#         "final_llm_lora_adapters",
#         is_main_process=accelerator.is_main_process,
#         save_function=accelerator.save,
#     )


# # --- MAIN EXECUTION ---
# if __name__ == "__main__":
#     # Note: Phase 1 is omitted for brevity but would follow the same Accelerate pattern
#     # You would call accelerator.prepare() and accelerator.backward() in the same way.
    
#     phase2_dummy_dataset = [{
#         'graph': torch.randn(1, 10), 
#         'text_instr': torch.randint(0, 1000, (1, 20)),
#         'image': torch.randn(1, 3, 224, 224),
#         'llm_response': torch.randint(0, 1000, (1, 100))
#     }]

#     train_phase_2(phase2_dummy_dataset)
    
#     print("\nDistributed skeleton code execution finished.")