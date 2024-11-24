import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

class CrossTaskAttention(nn.Module):
    """Enhanced multi-head attention for NER and Topic feature integration"""
    def __init__(self, config):
        super().__init__()
        
        # Multi-head attention parameters
        self.num_heads = config.bridge_num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Task-specific projections
        self.ner_q_proj = nn.Linear(config.entity_feature_size, config.hidden_size)
        self.ner_k_proj = nn.Linear(config.entity_feature_size, config.hidden_size)
        self.ner_v_proj = nn.Linear(config.entity_feature_size, config.hidden_size)
        
        self.topic_q_proj = nn.Linear(config.topic_feature_size, config.hidden_size)
        self.topic_k_proj = nn.Linear(config.topic_feature_size, config.hidden_size)
        self.topic_v_proj = nn.Linear(config.topic_feature_size, config.hidden_size)
        
        # Language-specific biases
        self.language_biases = nn.ParameterDict({
            lang: nn.Parameter(torch.zeros(config.hidden_size))
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_type: str,  # 'ner' or 'topic'
        language_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-task attention with language awareness"""
        batch_size = query.size(0)
        
        # Select task-specific projections
        if query_type == 'ner':
            q_proj = self.ner_q_proj
            k_proj = self.topic_k_proj
            v_proj = self.topic_v_proj
        else:
            q_proj = self.topic_q_proj
            k_proj = self.ner_k_proj
            v_proj = self.ner_v_proj
        
        # Project and reshape
        q = q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Add language-specific biases
        for lang_id, bias in enumerate(self.language_biases.values()):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1, 1)
            q = q + (lang_mask * bias.view(1, 1, 1, -1))
        
        # Compute attention with relative position bias
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.head_dim * self.num_heads
        )
        
        return context, attn_weights

class TaskGating(nn.Module):
    """Enhanced gating mechanism with task and language awareness"""
    def __init__(self, config):
        super().__init__()
        
        # Task-specific transformations
        self.ner_transform = nn.Sequential(
            nn.Linear(config.entity_feature_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
        )
        
        self.topic_transform = nn.Sequential(
            nn.Linear(config.topic_feature_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
        )
        
        # Language-aware gate networks
        self.gate_networks = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 2),  # 2 gates: one for each task
                nn.Sigmoid()
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        ner_features: torch.Tensor,
        topic_features: torch.Tensor,
        language_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Language-aware gating"""
        # Transform features
        ner_transformed = self.ner_transform(ner_features)
        topic_transformed = self.topic_transform(topic_features)
        
        # Compute gates for each language
        combined = torch.cat([ner_transformed, topic_transformed], dim=-1)
        gates = torch.zeros(combined.size(0), combined.size(1), 2).to(combined.device)
        
        for lang_id, gate_network in enumerate(self.gate_networks.values()):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            gates += lang_mask * gate_network(combined)
        
        # Apply gates
        gated_ner = ner_transformed * gates[..., 0].unsqueeze(-1)
        gated_topic = topic_transformed * gates[..., 1].unsqueeze(-1)
        
        return self.dropout(gated_ner), self.dropout(gated_topic)

class LightweightBridge(nn.Module):
    """Enhanced bridge module for cross-task and cross-lingual integration"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Cross-task attention
        self.cross_attention = CrossTaskAttention(config)
        
        # Task gating
        self.gating = TaskGating(config)
        
        # Feature fusion with language adaptation
        self.ner_fusion = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.entity_feature_size)
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        self.topic_fusion = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.topic_feature_size)
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        # Task-specific normalization
        self.ner_norm = nn.LayerNorm(config.entity_feature_size)
        self.topic_norm = nn.LayerNorm(config.topic_feature_size)
        
    def forward(
        self,
        ner_features: torch.Tensor,
        topic_features: torch.Tensor,
        language_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Enhanced cross-task integration"""
        # Gate features with language awareness
        gated_ner, gated_topic = self.gating(
            ner_features, 
            topic_features, 
            language_ids
        )
        
        # Cross-task attention
        ner_context, ner_attn = self.cross_attention(
            gated_ner, gated_topic, gated_topic,
            query_type='ner',
            language_ids=language_ids,
            attention_mask=attention_mask
        )
        
        topic_context, topic_attn = self.cross_attention(
            gated_topic, gated_ner, gated_ner,
            query_type='topic',
            language_ids=language_ids,
            attention_mask=attention_mask
        )
        
        # Language-specific feature fusion
        enhanced_ner = torch.zeros_like(ner_features)
        enhanced_topic = torch.zeros_like(topic_features)
        
        for lang_id, (ner_fuser, topic_fuser) in enumerate(zip(
            self.ner_fusion.values(), self.topic_fusion.values())):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            
            ner_combined = torch.cat([gated_ner, ner_context], dim=-1)
            topic_combined = torch.cat([gated_topic, topic_context], dim=-1)
            
            enhanced_ner += lang_mask * ner_fuser(ner_combined)
            enhanced_topic += lang_mask * topic_fuser(topic_combined)
        
        # Residual connections and normalization
        final_ner = self.ner_norm(enhanced_ner + ner_features)
        final_topic = self.topic_norm(enhanced_topic + topic_features)
        
        return {
            'ner_features': final_ner,
            'topic_features': final_topic,
            'ner_attention': ner_attn,
            'topic_attention': topic_attn,
            'gates': {
                'ner': gated_ner,
                'topic': gated_topic
            }
        }
    
    def compute_alignment_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        ner_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss to encourage task alignment"""
        ner_attn = outputs['ner_attention']
        topic_attn = outputs['topic_attention']
        
        # Create label-based attention mask
        label_mask = (ner_labels != -100).float()
        
        # Compute attention consistency loss
        consistency_loss = F.mse_loss(
            torch.bmm(ner_attn, topic_attn.transpose(-2, -1)),
            torch.eye(ner_attn.size(-1), device=ner_attn.device).expand(
                ner_attn.size(0), -1, -1
            ),
            reduction='none'
        )
        
        # Apply masks
        masked_loss = consistency_loss * label_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        
        return masked_loss.sum() / (label_mask.sum() + 1e-6)