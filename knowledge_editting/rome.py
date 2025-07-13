import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np

class ROMEEditor:
    """
    ROME (Rank-One Model Editing) implementation
    Edits factual knowledge by updating specific layers in transformer models
    """
    
    def __init__(self, model_name: str):
        """
        Initialize ROME editor
        
        Args:
            model_name: HuggingFace model name
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def locate_knowledge_layer(self, subject: str, relation: str) -> int:
        """
        Locate which layer stores specific factual knowledge
        
        Args:
            subject: Subject of the fact (e.g., "Paris")
            relation: Relation type (e.g., "capital of")
            
        Returns:
            Layer index where knowledge is stored
        """
        # Simplified implementation - in practice, this uses causal tracing
        prompt = f"The {relation} {subject} is"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        layer_activations = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_activations.append((layer_idx, output.detach()))
            return hook
        
        # Register hooks for each layer
        hooks = []
        for i, layer in enumerate(self.model.encoder.layer):
            hook = layer.register_forward_hook(hook_fn(i))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations to find knowledge layer (simplified)
        # In practice, this involves more sophisticated analysis
        knowledge_layer = len(layer_activations) // 2  # Middle layer heuristic
        
        return knowledge_layer
    
    def edit_knowledge(self, 
                      subject: str, 
                      relation: str, 
                      old_object: str, 
                      new_object: str) -> None:
        """
        Edit factual knowledge in the model
        
        Args:
            subject: Subject of the fact
            relation: Relation type
            old_object: Current (incorrect) object
            new_object: New (correct) object
        """
        # Locate the layer storing this knowledge
        target_layer = self.locate_knowledge_layer(subject, relation)
        
        # Create prompts for old and new facts
        old_prompt = f"The {relation} {subject} is {old_object}"
        new_prompt = f"The {relation} {subject} is {new_object}"
        
        # Tokenize prompts
        old_inputs = self.tokenizer(old_prompt, return_tensors="pt").to(self.device)
        new_inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.device)
        
        # Get activations for both prompts
        old_activations = self._get_layer_activations(old_inputs, target_layer)
        new_activations = self._get_layer_activations(new_inputs, target_layer)
        
        # Compute the difference (edit vector)
        edit_vector = new_activations - old_activations
        
        # Apply rank-one update to the target layer
        self._apply_rank_one_update(target_layer, edit_vector)
        
        print(f"Knowledge edited: {subject} {relation} {old_object} -> {new_object}")
    
    def _get_layer_activations(self, inputs: Dict, layer_idx: int) -> torch.Tensor:
        """Get activations from a specific layer"""
        activations = None
        
        def hook_fn(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        # Register hook
        hook = self.model.encoder.layer[layer_idx].register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            self.model(**inputs)
        
        # Remove hook
        hook.remove()
        
        return activations
    
    def _apply_rank_one_update(self, layer_idx: int, edit_vector: torch.Tensor) -> None:
        """Apply rank-one update to modify layer parameters"""
        target_layer = self.model.encoder.layer[layer_idx]
        
        # Get the feed-forward network weights
        if hasattr(target_layer, 'intermediate'):
            # BERT-style model
            weight_matrix = target_layer.intermediate.dense.weight
        else:
            # Other architectures
            weight_matrix = target_layer.feed_forward.dense.weight
        
        # Compute rank-one update
        # This is a simplified version - actual ROME uses more sophisticated methods
        edit_matrix = torch.outer(edit_vector.squeeze(), edit_vector.squeeze())
        
        # Apply the update
        with torch.no_grad():
            weight_matrix.data += 0.001 * edit_matrix[:weight_matrix.size(0), :weight_matrix.size(1)]

# Usage example
editor = ROMEEditor("bert-base-uncased")
editor.edit_knowledge("Paris", "capital of", "Germany", "France")
