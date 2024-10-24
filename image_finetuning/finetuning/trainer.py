import os
import torch
from transformers import Trainer

class MultimodalTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir = None, state_dict = None):
        # Save the projection layer separately
        projection_layer_path = os.path.join(output_dir, "projection_layer")
        os.makedirs(projection_layer_path, exist_ok=True)
        torch.save(self.model.projections.state_dict(), os.path.join(projection_layer_path, "pytorch_model.bin"))

        # Save the Phi-3.5 QLoRA weights separately
        phi_model_path = os.path.join(output_dir, "phi_model")
        os.makedirs(phi_model_path, exist_ok=True)
        self.model.phi.save_pretrained(phi_model_path)