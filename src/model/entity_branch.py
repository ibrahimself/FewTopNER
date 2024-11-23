import torch
import torch.nn as nn
from torchcrf import CRF
from typing import Dict, List, Tuple, Optional
import numpy as np

class EntityEncoder(nn.Module):
    """
    Encoder for entity recognition with BiLSTM
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # BiLSTM for sequence encoding
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.projection = nn.Linear(
            2 * config.lstm_hidden_size,  # bidirectional
            config.entity_feature_size
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input sequences
        
        Args:
            sequence_output: Tensor from transformer [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Encoded sequences [batch_size, seq_len, entity_feature_size]
        """
        # Apply dropout to inputs
        sequence_output = self.dropout(sequence_output)
        
        # Pack padded sequence
        lengths = attention_mask.sum(dim=1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            sequence_output,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        
        # Unpack sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            padding_value=0
        )
        
        # Project to feature size
        entity_features = self.projection(lstm_output)
        
        return entity_features

class EntityBranch(nn.Module):
    """
    Named Entity Recognition branch with prototypical networks and CRF
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Entity encoder
        self.encoder = EntityEncoder(config)
        
        # Prototype network
        self.prototype_network = nn.Sequential(
            nn.Linear(config.entity_feature_size, config.prototype_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prototype_dim, config.prototype_dim)
        )
        
        # Entity classifier
        self.entity_classifier = nn.Linear(
            config.prototype_dim,
            config.num_entity_labels
        )
        
        # CRF layer
        self.crf = CRF(
            num_tags=config.num_entity_labels,
            batch_first=True
        )
        
        # Loss weights for different entity types
        self.register_buffer(
            'label_weights',
            torch.tensor(config.label_weights)
            if hasattr(config, 'label_weights')
            else torch.ones(config.num_entity_labels)
        )
        
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entity prototypes from support set
        
        Args:
            support_features: Encoded features from support set
            support_labels: Entity labels for support set
            attention_mask: Attention mask for valid tokens
        Returns:
            Tensor of entity prototypes
        """
        prototypes = []
        
        for label_id in range(self.config.num_entity_labels):
            # Create mask for this entity type
            label_mask = (support_labels == label_id) & attention_mask
            
            if label_mask.sum() > 0:
                # Get features for this entity type
                label_features = support_features[label_mask]
                prototype = label_features.mean(dim=0)
                prototypes.append(prototype)
            else:
                # Create zero prototype if no examples
                prototypes.append(
                    torch.zeros(self.config.prototype_dim, device=support_features.device)
                )
                
        return torch.stack(prototypes)

    def compute_distances(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between features and prototypes
        
        Args:
            features: Entity features [batch_size, seq_len, prototype_dim]
            prototypes: Entity prototypes [num_labels, prototype_dim]
        Returns:
            Distance matrix [batch_size, seq_len, num_labels]
        """
        # Reshape features for distance computation
        batch_size, seq_len, _ = features.shape
        features_flat = features.view(-1, self.config.prototype_dim)
        
        # Compute distances
        distances = torch.cdist(features_flat, prototypes)
        
        # Reshape back to sequence form
        distances = distances.view(batch_size, seq_len, -1)
        
        return distances

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        support_set: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass through entity branch
        
        Args:
            sequence_output: Transformer outputs
            attention_mask: Attention mask
            support_set: Optional support set for few-shot learning
            labels: Optional labels for training
        Returns:
            Dictionary containing outputs and loss
        """
        # Encode sequences
        entity_features = self.encoder(sequence_output, attention_mask)
        
        # Get prototype features
        prototype_features = self.prototype_network(entity_features)
        
        outputs = {
            'entity_features': entity_features,
            'prototype_features': prototype_features
        }
        
        # Few-shot learning with support set
        if support_set is not None:
            prototypes = self.compute_prototypes(
                support_set['prototype_features'],
                support_set['labels'],
                support_set['attention_mask']
            )
            
            # Compute distances to prototypes
            distances = self.compute_distances(prototype_features, prototypes)
            emissions = -distances  # Negative distances as emissions
        else:
            # Regular classification
            emissions = self.entity_classifier(prototype_features)
        
        outputs['emissions'] = emissions
        
        # CRF decoding
        if labels is not None:
            # Compute CRF loss
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            outputs['loss'] = loss
        else:
            # Decode best path
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            outputs['predictions'] = predictions
            
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with label weights
        
        Args:
            outputs: Forward pass outputs
            labels: Entity labels
            attention_mask: Attention mask
        Returns:
            Weighted loss
        """
        emissions = outputs['emissions']
        mask = attention_mask.bool()
        
        # Basic CRF loss
        crf_loss = -self.crf(emissions, labels, mask=mask, reduction='none')
        
        # Apply label weights
        label_weights = self.label_weights[labels]
        weighted_loss = crf_loss * label_weights
        
        # Average over non-padded tokens
        return weighted_loss.sum() / mask.sum()