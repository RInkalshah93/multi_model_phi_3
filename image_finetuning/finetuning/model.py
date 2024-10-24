import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel

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

class ClipPhi3Model(PreTrainedModel):
    def __init__(self, phi_model, clip_embed, phi_embed, projection_path=None):
        super().__init__(phi_model.config)
        self.phi = phi_model
        self.projections = Projections(clip_embed, phi_embed)
        if projection_path:
            self.projections.load_state_dict(torch.load(projection_path, map_location=device), strict=False)
        
        # Convert projections to bfloat16
        self.projections.to(torch.bfloat16)

    def forward(self, conversations_ids, image_embeds, conversations_mask, labels):
        # Apply embeddings to input_ids
        text_embeds = self.phi.get_input_embeddings()(conversations_ids)
        
        # Apply projection to image embeddings
        projected_image_embeds = self.projections(image_embeds)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)
        
        # Pass through phi-3 model
        outputs = self.phi(inputs_embeds=combined_embeds, attention_mask=conversations_mask, labels=labels, return_dict=True)
        
        return outputs
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.phi.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory):
        # Load the Phi-3.5 model
        self.phi.save_pretrained(save_directory)

        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.projections.state_dict(), projector_path)

        # Save the config
        self.config.save_pretrained(save_directory)
        
