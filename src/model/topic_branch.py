import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import XLMRobertaModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary

class TopicPrototypeNetwork(nn.Module):
    """Prototype network for topic modeling"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projection layers
        self.projector = nn.Sequential(
            nn.Linear(config.model.topic_hidden_size, config.model.prototype_dim),
            nn.LayerNorm(config.model.prototype_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.prototype_dim, config.model.prototype_dim)
        )
        
        # Learned temperature parameter
        self.temperature = nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)
    
    def compute_similarities(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """Compute scaled cosine similarities"""
        query_norm = query_features / query_features.norm(dim=-1, keepdim=True)
        support_norm = support_features / support_features.norm(dim=-1, keepdim=True)
        
        similarities = torch.matmul(query_norm, support_norm.transpose(-2, -1))
        return similarities / self.temperature

class TopicEncoder(nn.Module):
    """Combined LDA and Transformer encoder for topic modeling"""
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
        """Get LDA topic distributions"""
        if language not in self.lda_models:
            raise ValueError(f"No LDA model found for language: {language}")
            
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
            
        return torch.tensor(topic_dists, dtype=torch.float)
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        language_ids: torch.Tensor,
        languages: List[str]
    ) -> torch.Tensor:
        """
        Encode topics with language-specific processing
        
        Args:
            sequence_output: XLM-R outputs [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            texts: Original texts
            language_ids: Language identifiers [batch_size]
            languages: List of language codes
        """
        batch_size = sequence_output.size(0)
        
        # Apply language-specific adapters
        adapted_outputs = torch.zeros_like(sequence_output)
        for lang_id, lang in enumerate(languages):
            lang_mask = (language_ids == lang_id).view(-1, 1, 1)
            adapted_outputs += lang_mask * self.language_adapters[lang](sequence_output)
        
        # Pool sequence outputs
        masked_outputs = adapted_outputs * attention_mask.unsqueeze(-1)
        pooled_outputs = masked_outputs.sum(1) / attention_mask.sum(1).unsqueeze(-1)
        
        # Get LDA features for each language
        lda_features = []
        for i, text in enumerate(texts):
            lang = languages[language_ids[i]]
            lda_feat = self.get_lda_features([text], lang)
            lda_features.append(lda_feat)
        lda_features = torch.cat(lda_features, dim=0)
        
        # Combine features
        combined_features = torch.cat([pooled_outputs, lda_features], dim=-1)
        topic_features = self.topic_fusion(combined_features)
        
        return topic_features

class TopicBranch(nn.Module):
    """Topic modeling branch with prototype learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Topic encoder
        self.encoder = TopicEncoder(config)
        
        # Prototype network
        self.prototype_network = TopicPrototypeNetwork(config)
        
        # Topic classifier
        self.topic_classifier = nn.Linear(config.model.prototype_dim, config.model.num_topics)
        
        # Loss weights
        self.prototype_weight = config.model.prototype_weight
        self.classification_weight = config.model.classification_weight
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        language_ids: torch.Tensor,
        languages: List[str],
        support_set: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """Forward pass for topic branch"""
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
            # Few-shot mode
            support_features = self.prototype_network(support_set['topic_features'])
            
            # Compute similarities to support set
            similarities = self.prototype_network.compute_similarities(
                prototype_features,
                support_features
            )
            outputs['similarities'] = similarities
            
            if labels is not None:
                prototype_loss = nn.CrossEntropyLoss()(similarities, labels)
                outputs['prototype_loss'] = prototype_loss
        else:
            # Regular classification mode
            logits = self.topic_classifier(prototype_features)
            outputs['logits'] = logits
            
            if labels is not None:
                classification_loss = nn.CrossEntropyLoss()(logits, labels)
                outputs['classification_loss'] = classification_loss
        
        # Compute total loss if in training mode
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