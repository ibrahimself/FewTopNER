import torch
import torch.nn as nn
from TorchCRF import CRF
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EntityPrototypeNetwork(nn.Module):
    """Prototype network for entity features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projection layers
        self.projector = nn.Sequential(
            nn.Linear(config.model.entity_feature_size, config.model.prototype_dim),
            nn.LayerNorm(config.model.prototype_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.prototype_dim, config.model.prototype_dim)
        )
        
        # Initialize prototype memory
        self.register_buffer(
            'prototypes',
            torch.zeros(config.model.num_entity_labels, config.model.prototype_dim)
        )
        
        # Temperature parameter for distance scaling
        self.temperature = nn.Parameter(torch.tensor([1.0]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to prototype space"""
        return self.projector(features)
    
    def compute_prototype_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for prototype learning"""
        # Compute distances to prototypes
        distances = torch.cdist(features, self.prototypes)
        
        # Scale distances by learned temperature
        scaled_distances = -distances / self.temperature
        
        # Cross entropy loss
        loss = nn.CrossEntropyLoss()(scaled_distances, labels)
        
        return loss

class EntityEncoder(nn.Module):
    """BiLSTM encoder for entity recognition"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=config.model.hidden_size,
            hidden_size=config.model.lstm_hidden_size,
            num_layers=config.model.lstm_layers,
            bidirectional=True,
            dropout=config.model.dropout if config.model.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Language-specific adapters
        self.language_adapters = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size),
                nn.LayerNorm(self.config.model.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.model.hidden_size, self.config.model.hidden_size)
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        # Output projection
        self.projection = nn.Linear(
            2 * self.config.model.lstm_hidden_size,
            self.config.model.entity_feature_size
        )
        
        self.dropout = nn.Dropout(self.config.model.dropout)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode sequences with language-specific processing
        
        Args:
            sequence_output: XLM-R outputs [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            language_ids: Language identifiers [batch_size]
        """
        batch_size = sequence_output.size(0)
        
        # Apply language-specific adapters
        adapted_outputs = torch.zeros_like(sequence_output)
        for lang_id, adapter in enumerate(self.language_adapters.values()):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            adapted_outputs += lang_mask * adapter(sequence_output)
        
        # Pack sequences
        lengths = attention_mask.sum(dim=1).cpu()
        packed_input = pack_padded_sequence(
            adapted_outputs,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # BiLSTM processing
        packed_output, _ = self.bilstm(packed_input)
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            padding_value=0
        )
        
        # Project to final feature space
        entity_features = self.projection(lstm_output)
        
        return entity_features

class EntityBranch(nn.Module):
    """Named Entity Recognition branch with prototype learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Entity encoder
        self.encoder = EntityEncoder(self.config)
        
        # Prototype network
        self.prototype_network = EntityPrototypeNetwork(self.config)
        
        # CRF layer
        self.crf = CRF(
            num_labels=self.config.model.num_entity_labels
        )
        
        # Loss weights based on WikiNEuRal dataset statistics
        label_weights = self._compute_label_weights()
        self.register_buffer('label_weights', label_weights)
    
    def _compute_label_weights(self) -> torch.Tensor:
        """Compute label weights based on WikiNEuRal statistics"""
        # Statistics from WikiNEuRal paper
        label_counts = {
            'O': 2.40e6,      # Other
            'PER': 51000,     # Person
            'ORG': 31000,     # Organization
            'LOC': 67000,     # Location
            'MISC': 45000     # Miscellaneous
        }
        
        # Convert to weights (inverse frequency)
        total = sum(label_counts.values())
        weights = torch.tensor([
            total / count for label, count in label_counts.items()
        ])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
        support_set: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass for entity recognition
        
        Args:
            sequence_output: XLM-R outputs
            attention_mask: Attention mask
            language_ids: Language identifiers
            support_set: Support set for few-shot learning
            labels: Entity labels for training
        """
        # Encode sequences
        entity_features = self.encoder(sequence_output, attention_mask, language_ids)
        
        # Project to prototype space
        prototype_features = self.prototype_network(entity_features)
        
        outputs = {
            'entity_features': entity_features,
            'prototype_features': prototype_features
        }
        
        if support_set is not None:
            # Few-shot learning mode
            support_features = self.prototype_network(support_set['entity_features'])
            prototype_loss = self.prototype_network.compute_prototype_loss(
                support_features,
                support_set['labels']
            )
            outputs['prototype_loss'] = prototype_loss
            
            # Compute distances to support set examples
            distances = torch.cdist(
                prototype_features.view(-1, self.config.model.prototype_dim),
                support_features.view(-1, self.config.model.prototype_dim)
            )
            emissions = -distances.view(prototype_features.shape[0], -1, self.config.model.num_entity_labels)
        else:
            # Regular sequence labeling mode
            emissions = self.crf.get_emission_score(prototype_features)
        
        outputs['emissions'] = emissions
        
        # CRF decoding
        if labels is not None:
            mask = attention_mask.bool()
            crf_loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            outputs['crf_loss'] = crf_loss
            
            # Combine losses
            outputs['loss'] = crf_loss
            if 'prototype_loss' in outputs:
                outputs['loss'] += self.config.model.prototype_loss_weight * outputs['prototype_loss']
        else:
            # Decode best path
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            outputs['predictions'] = predictions
        
        return outputs

    def compute_similarity_matrix(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity matrix between query and support features"""
        query_features = self.prototype_network(query_features)
        support_features = self.prototype_network(support_features)
        
        similarities = torch.matmul(
            query_features, support_features.transpose(-2, -1)
        )
        
        return similarities / np.sqrt(self.config.model.prototype_dim)
