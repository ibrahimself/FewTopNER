import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel
from typing import Dict, Optional, Tuple, List
import logging
import math

logger = logging.getLogger(__name__)

class CrossLingualAttention(nn.Module):
    """Enhanced cross-lingual attention with language-aware processing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-head attention for cross-lingual alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Language-specific key/query transformations
        self.language_keys = nn.ModuleDict({
            lang: nn.Linear(config.hidden_size, config.hidden_size)
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        self.language_queries = nn.ModuleDict({
            lang: nn.Linear(config.hidden_size, config.hidden_size)
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        # Language pair alignment matrices
        self.alignment_matrices = nn.ParameterDict({
            f"{l1}_{l2}": nn.Parameter(torch.eye(config.hidden_size))
            for l1 in ['en', 'fr', 'de', 'es', 'it']
            for l2 in ['en', 'fr', 'de', 'es', 'it']
            if l1 != l2
        })
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def get_language_code(self, lang_id: int) -> str:
        """Convert language ID to code"""
        lang_map = {0: 'en', 1: 'fr', 2: 'de', 3: 'es', 4: 'it'}
        return lang_map[lang_id.item()]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        language_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_alignment_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-lingual attention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            language_ids: [batch_size]
            attention_mask: [batch_size, seq_len]
            return_alignment_scores: Whether to return alignment scores
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Prepare attention mask
        attn_mask = attention_mask.float().masked_fill(
            attention_mask == 0,
            float('-inf')
        ).bool()
        
        # Process each language pair in the batch
        aligned_states = torch.zeros_like(hidden_states)
        alignment_scores = torch.zeros(batch_size, seq_len, seq_len).to(hidden_states.device)
        
        for i in range(batch_size):
            src_lang = self.get_language_code(language_ids[i])
            
            # Get language-specific transformations
            query_transform = self.language_queries[src_lang]
            key_transform = self.language_keys[src_lang]
            
            # Transform input for this language
            query = query_transform(hidden_states[i:i+1])
            key = key_transform(hidden_states[i:i+1])
            
            # Apply cross-lingual attention
            aligned, attn_weights = self.cross_attention(
                query=query.transpose(0, 1),
                key=key.transpose(0, 1),
                value=hidden_states[i:i+1].transpose(0, 1),
                key_padding_mask=~attn_mask[i:i+1],
                need_weights=True
            )
            
            # Apply language pair alignments
            for j in range(batch_size):
                if i != j:
                    tgt_lang = self.get_language_code(language_ids[j])
                    align_key = f"{src_lang}_{tgt_lang}"
                    if align_key in self.alignment_matrices:
                        aligned = torch.matmul(
                            aligned.transpose(0, 1),
                            self.alignment_matrices[align_key]
                        ).transpose(0, 1)
            
            aligned_states[i:i+1] = aligned.transpose(0, 1)
            alignment_scores[i] = attn_weights.squeeze(0)
        
        # Project and normalize output
        output = self.output_projection(aligned_states)
        output = self.layer_norm(output + hidden_states)  # Residual connection
        
        if return_alignment_scores:
            return output, alignment_scores
        return output, None

class ContrastiveCrossLingualLearning(nn.Module):
    """Contrastive learning for cross-lingual alignment"""
    def __init__(self, config):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor([config.temperature]))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.contrastive_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        language_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive loss for cross-lingual alignment
        
        Args:
            features: [batch_size, seq_len, hidden_size]
            language_ids: [batch_size]
            attention_mask: [batch_size, seq_len]
        """
        # Get masked average pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_features = features * mask_expanded
        pooled_features = masked_features.sum(1) / mask_expanded.sum(1)
        
        # Project features
        projected = self.projection(pooled_features)
        projected = F.normalize(projected, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(projected, projected.t()) / self.temperature
        
        # Create labels for positive pairs (same content, different languages)
        labels = torch.zeros_like(sim_matrix)
        for i in range(len(language_ids)):
            for j in range(len(language_ids)):
                if i != j and language_ids[i] != language_ids[j]:
                    labels[i, j] = 1
                    
        # Compute contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return projected, loss

class LanguageCalibration(nn.Module):
    """Enhanced language calibration with cross-lingual alignment"""
    def __init__(self, config):
        super().__init__()
        
        # Language-specific adapters
        self.language_adapters = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.LayerNorm(config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size)
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        # Cross-lingual attention
        self.cross_lingual_attention = CrossLingualAttention(config)
        
        # Contrastive learning
        self.contrastive_learning = ContrastiveCrossLingualLearning(config)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        language_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Apply language calibration with cross-lingual alignment"""
        batch_size = hidden_states.size(0)
        
        # Apply language-specific adapters
        adapted_states = torch.zeros_like(hidden_states)
        for lang_id, adapter in enumerate(self.language_adapters.values()):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            adapted_states += lang_mask * adapter(hidden_states)
        
        # Apply cross-lingual attention
        aligned_states, alignment_scores = self.cross_lingual_attention(
            adapted_states,
            language_ids,
            attention_mask,
            return_alignment_scores=True
        )
        
        # Apply contrastive learning
        projected_features, contrastive_loss = self.contrastive_learning(
            aligned_states,
            language_ids,
            attention_mask
        )
        
        return {
            'hidden_states': aligned_states,
            'alignment_scores': alignment_scores,
            'projected_features': projected_features,
            'contrastive_loss': contrastive_loss
        }
    
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# Update SharedEncoder class to use enhanced alignment
class SharedEncoder(nn.Module):
    """Enhanced shared encoder with sophisticated cross-lingual alignment"""
    def __init__(self, config):
        super().__init__()
        
        # Previous initialization code remains the same
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.positional_encoding = PositionalEncoding(
            config.hidden_size,
            max_len=config.max_length
        )
        
        # Enhanced language calibration
        self.language_calibration = LanguageCalibration(config)
        
        # Rest of the initialization remains the same
        self.task_projector = TaskProjector(config)
        self.apply(self._init_weights)
        self.gradient_scale = getattr(config, 'gradient_scale', 1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced cross-lingual alignment"""
        # XLM-RoBERTa encoding
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        hidden_states = self.positional_encoding(hidden_states)
        
        # Enhanced language calibration
        calibration_outputs = self.language_calibration(
            hidden_states,
            language_ids,
            attention_mask
        )
        
        # Project to task-specific features
        entity_features, topic_features = self.task_projector(
            calibration_outputs['hidden_states'],
            language_ids
        )
        
        if self.training and self.gradient_scale != 1.0:
            entity_features = entity_features * self.gradient_scale
            topic_features = topic_features * self.gradient_scale
        
        if return_dict:
            return {
                'hidden_states': hidden_states,
                'calibrated_states': calibration_outputs['hidden_states'],
                'entity_features': entity_features,
                'topic_features': topic_features,
                'attention_mask': attention_mask,
                'alignment_scores': calibration_outputs['alignment_scores'],
                'contrastive_loss': calibration_outputs['contrastive_loss']
            }
        
        return (
            hidden_states,
            calibration_outputs['hidden_states'],
            entity_features,
            topic_features,
            attention_mask
        )