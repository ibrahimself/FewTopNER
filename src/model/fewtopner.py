import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from .shared_encoder import SharedEncoder
from .entity_branch import EntityBranch
from .topic_branch import TopicBranch
from .bridge import LightweightBridge

class FewTopNER(nn.Module):
    """
    FewTopNER: Few-shot Learning for Joint Named Entity Recognition and Topic Modeling
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.shared_encoder = SharedEncoder(config)
        self.entity_branch = EntityBranch(config)
        self.topic_branch = TopicBranch(config)
        self.bridge = LightweightBridge(config)
        
        # Initialize loss weights
        self.entity_loss_weight = config.entity_loss_weight
        self.topic_loss_weight = config.topic_loss_weight
        self.bridge_loss_weight = config.bridge_loss_weight
        
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language: str
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input text using shared encoder
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            language: Language code
        Returns:
            Dictionary of encoded features
        """
        return self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            language=language
        )
        
    def process_support_set(
        self,
        support_set: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process support set for few-shot learning
        
        Args:
            support_set: Dictionary containing support set features
        Returns:
            Processed support features
        """
        # Encode support set
        support_encoded = self.encode_text(
            input_ids=support_set['input_ids'],
            attention_mask=support_set['attention_mask'],
            language=support_set['language']
        )
        
        # Get entity and topic features
        entity_support = self.entity_branch(
            support_encoded['entity_features'],
            support_set['attention_mask'],
            labels=support_set['entity_labels']
        )
        
        topic_support = self.topic_branch(
            support_encoded['topic_features'],
            support_set['attention_mask'],
            labels=support_set['topic_labels']
        )
        
        return {
            'entity_features': entity_support['prototype_features'],
            'topic_features': topic_support['prototype_features'],
            'entity_labels': support_set['entity_labels'],
            'topic_labels': support_set['topic_labels'],
            'attention_mask': support_set['attention_mask']
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language: str,
        entity_labels: Optional[torch.Tensor] = None,
        topic_labels: Optional[torch.Tensor] = None,
        support_set: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FewTopNER
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            language: Language code
            entity_labels: Optional entity labels for training
            topic_labels: Optional topic labels for training
            support_set: Optional support set for few-shot learning
        Returns:
            Dictionary containing outputs and losses
        """
        outputs = {}
        
        # Process support set if provided
        if support_set is not None:
            support_features = self.process_support_set(support_set)
        else:
            support_features = None
            
        # Encode input text
        encoded_features = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            language=language
        )
        
        # Initial branch processing
        entity_outputs = self.entity_branch(
            encoded_features['entity_features'],
            attention_mask,
            support_set=support_features,
            labels=entity_labels
        )
        
        topic_outputs = self.topic_branch(
            encoded_features['topic_features'],
            attention_mask,
            support_set=support_features,
            labels=topic_labels
        )
        
        # Cross-task integration through bridge
        bridge_outputs = self.bridge(
            entity_features=entity_outputs['entity_features'],
            topic_features=topic_outputs['topic_features'],
            attention_mask=attention_mask
        )
        
        # Update features with enhanced versions
        enhanced_entity_outputs = self.entity_branch(
            bridge_outputs['entity_features'],
            attention_mask,
            support_set=support_features,
            labels=entity_labels
        )
        
        enhanced_topic_outputs = self.topic_branch(
            bridge_outputs['topic_features'],
            attention_mask,
            support_set=support_features,
            labels=topic_labels
        )
        
        # Collect outputs
        outputs.update({
            'entity_logits': enhanced_entity_outputs['logits'],
            'topic_logits': enhanced_topic_outputs['logits'],
            'entity_features': bridge_outputs['entity_features'],
            'topic_features': bridge_outputs['topic_features'],
            'attention_weights': {
                'entity': bridge_outputs['entity_attention'],
                'topic': bridge_outputs['topic_attention']
            }
        })
        
        # Compute losses during training
        if self.training and entity_labels is not None and topic_labels is not None:
            # Entity recognition loss
            entity_loss = enhanced_entity_outputs['loss']
            
            # Topic modeling loss
            topic_loss = enhanced_topic_outputs['loss']
            
            # Bridge consistency loss
            bridge_loss = self.bridge.compute_cross_task_loss(
                entity_features=entity_outputs['entity_features'],
                topic_features=topic_outputs['topic_features'],
                entity_labels=entity_labels,
                topic_labels=topic_labels,
                attention_mask=attention_mask
            )
            
            # Combined loss
            total_loss = (
                self.entity_loss_weight * entity_loss +
                self.topic_loss_weight * topic_loss +
                self.bridge_loss_weight * bridge_loss
            )
            
            outputs.update({
                'loss': total_loss,
                'entity_loss': entity_loss,
                'topic_loss': topic_loss,
                'bridge_loss': bridge_loss
            })
            
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language: str,
        support_set: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            language: Language code
            support_set: Optional support set for few-shot learning
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                language=language,
                support_set=support_set
            )
            
            # Get entity predictions
            entity_predictions = outputs['entity_logits'].argmax(dim=-1)
            
            # Get topic predictions
            topic_predictions = outputs['topic_logits'].argmax(dim=-1)
            
            return {
                'entity_predictions': entity_predictions,
                'topic_predictions': topic_predictions,
                'attention_weights': outputs['attention_weights']
            }
            
    def get_attention_weights(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for visualization
        
        Args:
            outputs: Model outputs
        Returns:
            Dictionary containing attention weights
        """
        return {
            'entity_to_topic': outputs['attention_weights']['entity'],
            'topic_to_entity': outputs['attention_weights']['topic']
        }