import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Dict, Optional, Tuple
import logging

class MultilingualCalibrationLayer(nn.Module):
    """
    Language-specific calibration layer for cross-lingual alignment
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Language-specific scaling factors
        self.language_scaling = nn.ParameterDict({
            lang: nn.Parameter(torch.ones(config.hidden_size))
            for lang in config.languages
        })
        
        # Cross-lingual alignment
        self.alignment = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        language: str
    ) -> torch.Tensor:
        """
        Apply language-specific calibration
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            language: Language code
        Returns:
            Calibrated tensor [batch_size, seq_len, hidden_size]
        """
        # Apply language-specific scaling
        if language in self.language_scaling:
            scale = self.language_scaling[language]
            hidden_states = hidden_states * scale.unsqueeze(0).unsqueeze(0)
        
        # Apply cross-lingual alignment
        aligned_states = self.alignment(hidden_states)
        
        # Layer normalization and dropout
        calibrated_states = self.layer_norm(aligned_states)
        calibrated_states = self.dropout(calibrated_states)
        
        return calibrated_states

class FeatureProjector(nn.Module):
    """
    Projects shared features to task-specific spaces
    """
    def __init__(self, config):
        super().__init__()
        
        # Feature projections for different tasks
        self.entity_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.projection_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_size, config.entity_feature_size)
        )
        
        self.topic_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.projection_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_size, config.topic_feature_size)
        )
        
        # Layer normalization for each projection
        self.entity_norm = nn.LayerNorm(config.entity_feature_size)
        self.topic_norm = nn.LayerNorm(config.topic_feature_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features for each task
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
        Returns:
            Tuple of entity and topic features
        """
        # Project to entity space
        entity_features = self.entity_projection(hidden_states)
        entity_features = self.entity_norm(entity_features)
        
        # Project to topic space
        topic_features = self.topic_projection(hidden_states)
        topic_features = self.topic_norm(topic_features)
        
        return entity_features, topic_features

class SharedEncoder(nn.Module):
    """
    Shared multilingual encoder with calibration and task-specific projections
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load pre-trained XLM-RoBERTa
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
        
        # Freeze certain layers if specified
        if hasattr(config, 'freeze_layers') and config.freeze_layers:
            self.freeze_layers(config.freeze_layers)
        
        # Multilingual calibration
        self.calibration = MultilingualCalibrationLayer(config)
        
        # Feature projection
        self.projector = FeatureProjector(config)
        
        # Gradient scaling for stable training
        self.gradient_scale = config.gradient_scale
        
    def freeze_layers(self, num_layers: int):
        """
        Freeze bottom layers of the transformer
        
        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.xlm_roberta.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze encoder layers
        for layer in self.xlm_roberta.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
                
        self.logger.info(f"Froze {num_layers} layers of XLM-RoBERTa")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language: str,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoder
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            language: Language code
            token_type_ids: Optional token type IDs
            return_dict: Whether to return dictionary
        Returns:
            Dictionary of outputs or tuple of tensors
        """
        # XLM-RoBERTa forward pass
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply language calibration
        calibrated_output = self.calibration(
            sequence_output,
            language
        )
        
        # Scale gradients
        if self.training and self.gradient_scale != 1.0:
            calibrated_output = calibrated_output * self.gradient_scale
            
        # Project to task-specific features
        entity_features, topic_features = self.projector(calibrated_output)
        
        if return_dict:
            return {
                'sequence_output': sequence_output,
                'calibrated_output': calibrated_output,
                'entity_features': entity_features,
                'topic_features': topic_features,
                'attention_mask': attention_mask
            }
        
        return (
            sequence_output,
            calibrated_output,
            entity_features,
            topic_features,
            attention_mask
        )
    
    def get_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create attention mask from input IDs
        
        Args:
            input_ids: Input token IDs
            pad_token_id: Optional padding token ID
        Returns:
            Attention mask tensor
        """
        if pad_token_id is None:
            pad_token_id = self.xlm_roberta.config.pad_token_id
            
        attention_mask = (input_ids != pad_token_id).long()
        return attention_mask

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]