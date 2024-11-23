import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional
import logging

class CrossTaskAttention(nn.Module):
    """
    Multi-head attention for cross-task feature integration
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-head attention parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projections for queries, keys, and values
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-task attention
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            key: Key tensor [batch_size, seq_len, hidden_size]
            value: Value tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        # Compute attention weights and apply dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.config.hidden_size
        )
        output = self.out_proj(context)
        
        return output, attn_weights

class GatingMechanism(nn.Module):
    """
    Gating mechanism for controlled feature integration
    """
    def __init__(self, config):
        super().__init__()
        
        # Feature transformation
        self.entity_transform = nn.Linear(config.entity_feature_size, config.hidden_size)
        self.topic_transform = nn.Linear(config.topic_feature_size, config.hidden_size)
        
        # Gate networks
        self.entity_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        self.topic_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        entity_features: torch.Tensor,
        topic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gating mechanism for feature integration
        
        Args:
            entity_features: Entity features
            topic_features: Topic features
        Returns:
            Tuple of (gated entity features, gated topic features)
        """
        # Transform features to common space
        entity_transformed = self.entity_transform(entity_features)
        topic_transformed = self.topic_transform(topic_features)
        
        # Compute gate values
        combined = torch.cat([entity_transformed, topic_transformed], dim=-1)
        entity_gate_values = self.entity_gate(combined)
        topic_gate_values = self.topic_gate(combined)
        
        # Apply gates
        gated_entity = entity_transformed * entity_gate_values
        gated_topic = topic_transformed * topic_gate_values
        
        return self.dropout(gated_entity), self.dropout(gated_topic)

class LightweightBridge(nn.Module):
    """
    Lightweight bridge for cross-task feature integration
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cross-task attention
        self.cross_attention = CrossTaskAttention(config)
        
        # Gating mechanism
        self.gating = GatingMechanism(config)
        
        # Feature fusion
        self.entity_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.entity_feature_size)
        )
        
        self.topic_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.topic_feature_size)
        )
        
        # Task-specific layer normalization
        self.entity_norm = nn.LayerNorm(config.entity_feature_size)
        self.topic_norm = nn.LayerNorm(config.topic_feature_size)
        
    def forward(
        self,
        entity_features: torch.Tensor,
        topic_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate features between NER and Topic Modeling tasks
        
        Args:
            entity_features: Entity features [batch_size, seq_len, entity_feature_size]
            topic_features: Topic features [batch_size, seq_len, topic_feature_size]
            attention_mask: Optional attention mask
        Returns:
            Dictionary containing enhanced features and attention weights
        """
        # Gate features
        gated_entity, gated_topic = self.gating(entity_features, topic_features)
        
        # Cross-task attention from entity to topic
        entity_context, entity_attn = self.cross_attention(
            gated_entity, gated_topic, gated_topic, attention_mask
        )
        
        # Cross-task attention from topic to entity
        topic_context, topic_attn = self.cross_attention(
            gated_topic, gated_entity, gated_entity, attention_mask
        )
        
        # Combine original and context features
        enhanced_entity = torch.cat([gated_entity, entity_context], dim=-1)
        enhanced_topic = torch.cat([gated_topic, topic_context], dim=-1)
        
        # Feature fusion
        final_entity = self.entity_fusion(enhanced_entity)
        final_topic = self.topic_fusion(enhanced_topic)
        
        # Residual connection and normalization
        final_entity = self.entity_norm(final_entity + entity_features)
        final_topic = self.topic_norm(final_topic + topic_features)
        
        return {
            'entity_features': final_entity,
            'topic_features': final_topic,
            'entity_attention': entity_attn,
            'topic_attention': topic_attn
        }
    
    def compute_cross_task_loss(
        self,
        entity_features: torch.Tensor,
        topic_features: torch.Tensor,
        entity_labels: torch.Tensor,
        topic_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute additional loss to encourage meaningful cross-task interactions
        """
        # Get enhanced features
        outputs = self.forward(entity_features, topic_features, attention_mask)
        
        # Compute consistency loss between tasks
        entity_attn = outputs['entity_attention']
        topic_attn = outputs['topic_attention']
        
        # Encourage attention to relevant parts based on labels
        entity_mask = (entity_labels != -100).float()
        topic_mask = (topic_labels != -100).float()
        
        consistency_loss = torch.mean(
            torch.abs(
                torch.bmm(entity_attn, topic_attn.transpose(-2, -1)) - \
                torch.eye(entity_attn.size(-1), device=entity_attn.device)
            )
        )
        
        return self.config.consistency_weight * consistency_loss