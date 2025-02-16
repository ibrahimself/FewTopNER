import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import XLMRobertaModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary

class TopicEncoder(nn.Module):
    """Combined LDA and Transformer encoder for topic modeling with proper device handling"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Language-specific LDA models
        self.lda_models = {}
        self.dictionaries = {}
        
        # Language adapters
        self.language_adapters = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(config.model.hidden_size, config.model.hidden_size),
                nn.LayerNorm(config.model.hidden_size),
                nn.ReLU(),
                nn.Linear(config.model.hidden_size, config.model.hidden_size)
            )
            for lang in ['en', 'fr', 'de', 'es', 'it']
        })
        
        # Feature fusion
        self.topic_fusion = nn.Sequential(
            nn.Linear(config.model.hidden_size + config.model.num_topics, config.model.topic_hidden_size),
            nn.LayerNorm(config.model.topic_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )

    def set_lda_model(self, language: str, lda_model: LdaModel, dictionary: Dictionary):
        """Set LDA model for a specific language"""
        self.lda_models[language] = lda_model
        self.dictionaries[language] = dictionary
    
    def get_lda_features(self, texts: List[str], language: str) -> torch.Tensor:
        """Get LDA topic distributions with device handling"""
        device = next(self.parameters()).device
        
        if language not in self.lda_models:
            print(f"Warning: No LDA model found for language: {language}. Using zero vector.")
            return torch.zeros(len(texts), self.config.model.num_topics, 
                             dtype=torch.float, device=device)
            
        lda_model = self.lda_models[language]
        dictionary = self.dictionaries[language]
        
        # Convert texts to BOW
        bows = [dictionary.doc2bow(text.split()) for text in texts]
        
        # Get topic distributions
        topic_dists = []
        for bow in bows:
            dist = np.zeros(self.config.model.num_topics)
            for topic_id, prob in lda_model.get_document_topics(bow):
                dist[topic_id] = prob
            topic_dists.append(dist)
            
        # Create tensor on the same device as the model
        return torch.tensor(topic_dists, dtype=torch.float, device=device)
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        language_ids: torch.Tensor,
        languages: List[str]
    ) -> torch.Tensor:
        """Forward pass with consistent device handling"""
        device = sequence_output.device
        batch_size = sequence_output.size(0)
        
        # Ensure all inputs are on the correct device
        attention_mask = attention_mask.to(device)
        language_ids = language_ids.to(device)
        
        # Apply language-specific adapters
        adapted_outputs = torch.zeros_like(sequence_output)
        for lang_id, lang in enumerate(languages):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            adapted_output = self.language_adapters[lang](sequence_output)
            adapted_outputs += lang_mask * adapted_output
        
        # Pool sequence outputs
        masked_outputs = adapted_outputs * attention_mask.unsqueeze(-1)
        pooled_outputs = masked_outputs.sum(1) / (attention_mask.sum(1).unsqueeze(-1) + 1e-10)
        
        # Get LDA features for each language
        lda_features_list = []
        for i, text in enumerate(texts):
            lang = languages[language_ids[i].item()]
            lda_feat = self.get_lda_features([text], lang)
            lda_features_list.append(lda_feat)
        
        # Concatenate LDA features
        if lda_features_list:
            lda_features = torch.cat(lda_features_list, dim=0)
        else:
            lda_features = torch.zeros(batch_size, self.config.model.num_topics, device=device)
        
        # Ensure both features are on the same device before concatenating
        pooled_outputs = pooled_outputs.to(device)
        lda_features = lda_features.to(device)
        
        # Combine features
        combined_features = torch.cat([pooled_outputs, lda_features], dim=-1)
        topic_features = self.topic_fusion(combined_features)
        
        return topic_features

class TopicBranch(nn.Module):
    """Topic modeling branch with prototype learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Topic encoder with device handling
        self.encoder = TopicEncoder(config)
        
        # Prototype network
        self.prototype_network = TopicPrototypeNetwork(config)
        
        # Topic classifier
        self.topic_classifier = nn.Linear(config.model.prototype_dim, config.model.num_topics)
        
        # Loss weights
        self.prototype_weight = config.model.prototype_weight
        self.classification_weight = config.model.classification_weight
    
    def to(self, device):
        """Ensure proper device movement for all components"""
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.prototype_network = self.prototype_network.to(device)
        self.topic_classifier = self.topic_classifier.to(device)
        return self
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        language_ids: torch.Tensor,
        languages: List[str] = None,
        support_set: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """Forward pass with device consistency"""
        device = sequence_output.device
        
        if languages is None:
            languages = list(self.encoder.language_adapters.keys())
            
        # Move inputs to correct device
        attention_mask = attention_mask.to(device)
        language_ids = language_ids.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Encode topics
        topic_features = self.encoder(
            sequence_output,
            attention_mask,
            texts,
            language_ids,
            languages
        )
        
        # Get prototype features
        prototype_features = self.prototype_network(topic_features)
        
        outputs = {
            'topic_features': topic_features,
            'prototype_features': prototype_features
        }
        
        if support_set is not None:
            support_features = self.prototype_network(support_set['topic_features'].to(device))
            similarities = self.prototype_network.compute_similarities(
                prototype_features,
                support_features
            )
            outputs['similarities'] = similarities
            
            if labels is not None:
                prototype_loss = nn.CrossEntropyLoss()(similarities, labels)
                outputs['prototype_loss'] = prototype_loss
        else:
            logits = self.topic_classifier(prototype_features)
            outputs['logits'] = logits
            
            if labels is not None:
                classification_loss = nn.CrossEntropyLoss()(logits, labels)
                outputs['classification_loss'] = classification_loss
        
        if labels is not None:
            total_loss = 0
            if 'prototype_loss' in outputs:
                total_loss += self.prototype_weight * outputs['prototype_loss']
            if 'classification_loss' in outputs:
                total_loss += self.classification_weight * outputs['classification_loss']
            outputs['loss'] = total_loss
        
        return outputs
    
    def set_lda_models(self, lda_models: Dict[str, Tuple[LdaModel, Dictionary]]):
        """Set LDA models for all languages"""
        for lang, (lda_model, dictionary) in lda_models.items():
            self.encoder.set_lda_model(lang, lda_model, dictionary)