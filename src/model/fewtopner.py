import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from transformers import XLMRobertaModel

from .shared_encoder import SharedEncoder
from .entity_branch import EntityBranch
from .topic_branch import TopicBranch
from .bridge import LightweightBridge

logger = logging.getLogger(__name__)

class FewTopNER(nn.Module):
    """
    FewTopNER: Few-shot Learning for Named Entity Recognition with Topic-Enhanced Features
    
    Architecture integrates:
    - Multilingual XLM-R encoder
    - WikiNEuRal NER branch
    - Wikipedia Topic branch
    - Cross-task and cross-lingual bridge
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core components
        self.shared_encoder = SharedEncoder(config)
        self.entity_branch = EntityBranch(config)
        self.topic_branch = TopicBranch(config)
        self.bridge = LightweightBridge(config)
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'entity': getattr(config, 'entity_loss_weight', 1.0),
            'topic': getattr(config, 'topic_loss_weight', 1.0),
            'bridge': getattr(config, 'bridge_loss_weight', 0.1),
            'contrastive': getattr(config, 'contrastive_loss_weight', 0.1)
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language_ids: torch.Tensor,
        texts: List[str],  # Original texts for topic modeling
        entity_labels: Optional[torch.Tensor] = None,
        topic_labels: Optional[torch.Tensor] = None,
        support_set: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FewTopNER
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            language_ids: Language identifiers [batch_size]
            texts: Original texts for topic feature extraction
            entity_labels: NER labels [batch_size, seq_len]
            topic_labels: Topic labels [batch_size]
            support_set: Optional support set for few-shot learning
        """
        # Get initial shared representations
        shared_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            language_ids=language_ids,
            return_dict=True
        )
        
        # Process support set if provided
        if support_set is not None:
            support_outputs = self.shared_encoder(
                input_ids=support_set['input_ids'],
                attention_mask=support_set['attention_mask'],
                language_ids=support_set['language_ids'],
                return_dict=True
            )
            
            processed_support = {
                'entity': {
                    'features': support_outputs['entity_features'],
                    'labels': support_set['entity_labels'],
                    'attention_mask': support_set['attention_mask']
                },
                'topic': {
                    'features': support_outputs['topic_features'],
                    'texts': support_set['texts'],
                    'labels': support_set['topic_labels']
                }
            }
        else:
            processed_support = None
        
        # Initial task-specific processing
        entity_outputs = self.entity_branch(
            sequence_output=shared_outputs['calibrated_states'],
            attention_mask=attention_mask,
            language_ids=language_ids,
            support_set=processed_support['entity'] if processed_support else None,
            labels=entity_labels if self.training else None
        )
        
        topic_outputs = self.topic_branch(
            sequence_output=shared_outputs['calibrated_states'],
            attention_mask=attention_mask,
            texts=texts,
            language_ids=language_ids,
            support_set=processed_support['topic'] if processed_support else None,
            labels=topic_labels if self.training else None
        )
        
        # Cross-task integration through bridge
        bridge_outputs = self.bridge(
            ner_features=entity_outputs['prototype_features'],
            topic_features=topic_outputs['prototype_features'],
            language_ids=language_ids,
            attention_mask=attention_mask
        )
        
        # Enhanced task predictions
        enhanced_entity = self.entity_branch(
            sequence_output=bridge_outputs['ner_features'],
            attention_mask=attention_mask,
            language_ids=language_ids,
            support_set=processed_support['entity'] if processed_support else None,
            labels=entity_labels if self.training else None
        )
        
        enhanced_topic = self.topic_branch(
            sequence_output=bridge_outputs['topic_features'],
            attention_mask=attention_mask,
            texts=texts,
            language_ids=language_ids,
            support_set=processed_support['topic'] if processed_support else None,
            labels=topic_labels if self.training else None
        )
        
        outputs = {
            'entity_outputs': enhanced_entity,
            'topic_outputs': enhanced_topic,
            'attention_weights': bridge_outputs['ner_attention'],
            'cross_attention': {
                'entity_to_topic': bridge_outputs['ner_attention'],
                'topic_to_entity': bridge_outputs['topic_attention']
            }
        }
        
        # Compute losses during training
        if self.training and entity_labels is not None and topic_labels is not None:
            losses = {}
            
            # Task-specific losses
            losses['entity'] = enhanced_entity['loss'] * self.loss_weights['entity']
            losses['topic'] = enhanced_topic['loss'] * self.loss_weights['topic']
            
            # Bridge alignment loss
            losses['bridge'] = self.bridge.compute_alignment_loss(
                bridge_outputs,
                entity_labels,
                attention_mask
            ) * self.loss_weights['bridge']
            
            # Cross-lingual contrastive loss
            losses['contrastive'] = shared_outputs['contrastive_loss'] * self.loss_weights['contrastive']
            
            # Total loss
            losses['total'] = sum(losses.values())
            outputs['losses'] = losses
        
        return outputs
    
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        support_set: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Inference mode prediction"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                language_ids=batch['language_ids'],
                texts=batch['texts'],
                support_set=support_set
            )
            
            predictions = {
                'entities': outputs['entity_outputs']['predictions'],
                'topics': outputs['topic_outputs']['predictions'],
                'attention_maps': outputs['cross_attention']
            }
            
            return predictions
    
    def get_attention_maps(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization"""
        attention_maps = {
            'entity_to_topic': outputs['cross_attention']['entity_to_topic'],
            'topic_to_entity': outputs['cross_attention']['topic_to_entity'],
            'tokens': batch['input_ids'],
            'entity_labels': batch.get('entity_labels'),
            'topic_labels': batch.get('topic_labels')
        }
        return attention_maps

    def set_topic_models(self, topic_models: Dict):
        """Set pre-trained topic models for each language"""
        self.topic_branch.set_lda_models(topic_models)