import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Projections(nn.Module):
    def __init__(self, clip_embed, phi_embed, num_projection_layers=6):
        super().__init__()

        self.output = nn.Linear(clip_embed, phi_embed)
        self.norm = nn.LayerNorm(phi_embed)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phi_embed, phi_embed),
                    nn.GELU(),  
                    nn.Linear(phi_embed, phi_embed),
                )
                for _ in range(num_projection_layers)
            ]
        )

    def forward(self, x):
        x = self.output(x)
        x = self.norm(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual 
        
        return x

class ClipPhi3Model(nn.Module):
    def __init__(self, model_name, clip_embed, phi_embed):
        super().__init__()
        self.phi = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True)
        self.projections = Projections(clip_embed, phi_embed)
        
        # Load image embeddings and convert to tensors if necessary
        self.image_embeddings = torch.load('clip_embeddings.pt')
        for key, value in self.image_embeddings.items():
            if isinstance(value, np.ndarray):
                self.image_embeddings[key] = torch.from_numpy(value).to(torch.bfloat16)
            else:
                self.image_embeddings[key] = value.to(torch.bfloat16)
        
        # Convert projections to bfloat16
        self.projections.to(torch.bfloat16)
        
        # Freeze phi model weights
        for param in self.phi.parameters():
            param.requires_grad = False
    
    def forward(self, image_ids, input_ids):
        # Apply embeddings to input_ids
        text_embeds = self.phi.get_input_embeddings()(input_ids)
        
        # Load image embeddings
        image_embeds = torch.stack([self.image_embeddings[id] for id in image_ids]).to(device)
        
        # Apply projection to image embeddings
        projected_image_embeds = self.projections(image_embeds)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)
        
        # Pass through phi-3 model
        outputs = self.phi(inputs_embeds=combined_embeds)
        
        return outputs  # Return logits instead of the full output

# Usage example:
# model = ClipPhi3Model("microsoft/phi-2", clip_embed=512, phi_embed=2560)
# outputs = model(image_ids=['image1', 'image2'], input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]))
