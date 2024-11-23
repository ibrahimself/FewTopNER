import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class LDABertEncoder(nn.Module):
    """
    Combined LDA and BERT encoder for topic modeling
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_topics = config.num_topics
        self.bert_dim = config.bert_dim
        self.hidden_dim = config.hidden_dim
        
        # BERT encoder (using Sentence-BERT)
        self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # LDA model (initialized during training)
        self.lda_model = None
        self.dictionary = None
        
        # Projection layers
        self.bert_projection = nn.Linear(self.bert_dim, self.hidden_dim)
        self.lda_projection = nn.Linear(self.num_topics, self.hidden_dim)
        
        # Topic encoder
        self.topic_encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.num_topics)
        )
        
    def train_lda(self, texts: List[List[str]]):
        """
        Train LDA model on preprocessed texts
        
        Args:
            texts: List of tokenized documents
        """
        # Create dictionary
        self.dictionary = Dictionary(texts)
        
        # Create corpus
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=10,
            alpha='auto',
            random_state=42
        )
        
    def get_lda_vectors(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Get LDA topic distributions for texts
        
        Args:
            texts: List of tokenized documents
        Returns:
            Tensor of topic distributions
        """
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda first.")
            
        # Convert texts to corpus
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # Get topic distributions
        lda_vectors = []
        for bow in corpus:
            topic_dist = [0] * self.num_topics
            for topic_id, prob in self.lda_model.get_document_topics(bow):
                topic_dist[topic_id] = prob
            lda_vectors.append(topic_dist)
            
        return torch.tensor(lda_vectors, dtype=torch.float)

    def get_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get BERT embeddings for texts
        
        Args:
            texts: List of original (non-tokenized) texts
        Returns:
            Tensor of BERT embeddings
        """
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=self.config.bert_batch_size,
            show_progress_bar=False
        )
        return torch.tensor(embeddings, dtype=torch.float)

class TopicBranch(nn.Module):
    """
    Topic Modeling branch of FewTopNER using LDA-BERT
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # LDA-BERT encoder
        self.encoder = LDABertEncoder(config)
        
        # Prototype network for few-shot learning
        self.prototype_network = nn.Sequential(
            nn.Linear(config.hidden_dim, config.prototype_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prototype_dim, config.prototype_dim)
        )
        
        # Topic classifier
        self.classifier = nn.Linear(config.prototype_dim, config.num_topics)
        
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute topic prototypes from support set
        
        Args:
            support_features: Encoded features from support set
            support_labels: Topic labels for support set
        Returns:
            Tensor of topic prototypes
        """
        prototypes = []
        for topic in range(self.config.num_topics):
            # Get features for this topic
            mask = (support_labels == topic)
            if mask.sum() > 0:
                topic_features = support_features[mask]
                prototype = topic_features.mean(dim=0)
                prototypes.append(prototype)
            else:
                # Create zero prototype if no examples
                prototypes.append(torch.zeros_like(support_features[0]))
                
        return torch.stack(prototypes)

    def forward(
        self,
        texts: List[str],
        tokenized_texts: List[List[str]],
        support_set: Optional[Dict] = None
    ) -> Dict:
        """
        Forward pass through topic branch
        
        Args:
            texts: Original texts
            tokenized_texts: Tokenized texts for LDA
            support_set: Optional support set for few-shot learning
        Returns:
            Dictionary containing outputs and loss
        """
        # Get LDA vectors
        lda_vectors = self.encoder.get_lda_vectors(tokenized_texts)
        
        # Get BERT embeddings
        bert_embeddings = self.encoder.get_bert_embeddings(texts)
        
        # Project both representations
        lda_projected = self.encoder.lda_projection(lda_vectors)
        bert_projected = self.encoder.bert_projection(bert_embeddings)
        
        # Combine representations
        combined = torch.cat([lda_projected, bert_projected], dim=-1)
        
        # Encode topics
        topic_features = self.encoder.topic_encoder(combined)
        
        # Get prototype features
        prototype_features = self.prototype_network(topic_features)
        
        outputs = {
            'topic_features': topic_features,
            'prototype_features': prototype_features
        }
        
        # Few-shot learning with support set
        if support_set is not None:
            prototypes = self.compute_prototypes(
                support_set['prototype_features'],
                support_set['labels']
            )
            
            # Compute distances to prototypes
            distances = torch.cdist(prototype_features, prototypes)
            logits = -distances
            
            outputs['logits'] = logits
        else:
            # Regular classification
            logits = self.classifier(prototype_features)
            outputs['logits'] = logits
            
        return outputs

    def compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute loss for topic modeling
        
        Args:
            outputs: Forward pass outputs
            labels: Topic labels
            reduction: Loss reduction method
        Returns:
            Loss tensor
        """
        logits = outputs['logits']
        
        if reduction == 'none':
            return nn.functional.cross_entropy(
                logits,
                labels,
                reduction='none'
            )
        
        return nn.functional.cross_entropy(logits, labels)