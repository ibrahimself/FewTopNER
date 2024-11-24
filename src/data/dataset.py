import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FewShotBatch:
    """Container for few-shot batches"""
    support_ner: Dict[str, torch.Tensor]
    support_topic: Dict[str, torch.Tensor]
    query_ner: Dict[str, torch.Tensor]
    query_topic: Dict[str, torch.Tensor]
    languages: List[str]

class FewTopNERDataset(Dataset):
    """
    Dataset class that combines WikiNEuRal NER and Wikipedia Topic data.
    Handles both regular batches and few-shot episodes.
    """
    def __init__(
        self, 
        ner_features: Union[List[Dict], Dict[str, List[Dict]]],
        topic_features: Union[List[Dict], Dict[str, List[Dict]]],
        languages: Union[str, List[str]],
        is_few_shot: bool = False,
        max_length: int = 128
    ):
        self.is_few_shot = is_few_shot
        self.max_length = max_length
        
        # Handle single language vs multi-language case
        self.languages = [languages] if isinstance(languages, str) else languages
        
        # Initialize features based on whether it's multilingual or single language
        if isinstance(ner_features, dict):
            self.ner_features = []
            self.topic_features = []
            for lang in self.languages:
                if lang in ner_features and lang in topic_features:
                    self.ner_features.extend(ner_features[lang])
                    self.topic_features.extend(topic_features[lang])
        else:
            self.ner_features = ner_features
            self.topic_features = topic_features
        
        # Verify data alignment
        if len(self.ner_features) != len(self.topic_features):
            raise ValueError(
                f"Mismatched features: NER={len(self.ner_features)}, "
                f"Topic={len(self.topic_features)}"
            )

    def __len__(self) -> int:
        return len(self.ner_features)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item combining NER and Topic features"""
        ner_feature = self.ner_features[idx]
        topic_feature = self.topic_features[idx]
        
        item = {
            # NER features
            'input_ids': torch.tensor(ner_feature['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(ner_feature['attention_mask'], dtype=torch.long),
            'label_ids': torch.tensor(ner_feature['label_ids'], dtype=torch.long),
            'language_id': torch.tensor(ner_feature['language_id'], dtype=torch.long),
            
            # Topic features
            'topic_features': torch.tensor(topic_feature['features'], dtype=torch.float),
            
            # Original text and metadata (for debugging and analysis)
            'original_tokens': ner_feature.get('original_tokens', []),
            'original_tags': ner_feature.get('original_tags', []),
            'language': topic_feature['language']
        }
        
        return item

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        batch_size = len(batch)
        
        # Get max sequence length in this batch
        max_len = max(len(item['input_ids']) for item in batch)
        max_len = min(max_len, 128)  # Cap at max_length
        
        # Initialize tensors
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        label_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        language_ids = torch.zeros(batch_size, dtype=torch.long)
        topic_features = torch.zeros((batch_size, batch[0]['topic_features'].size(0)), dtype=torch.float)
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = item['input_ids']
            attention_mask[i, :seq_len] = item['attention_mask']
            label_ids[i, :seq_len] = item['label_ids']
            language_ids[i] = item['language_id']
            topic_features[i] = item['topic_features']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids,
            'language_ids': language_ids,
            'topic_features': topic_features
        }

class FewShotEpisode:
    """
    Enhanced few-shot episode handling for FewTopNER.
    Supports multiple languages and balanced sampling.
    """
    def __init__(
        self,
        support_set: Dict[str, Dict[str, List]],
        query_set: Dict[str, Dict[str, List]],
        languages: List[str],
        n_way: int,
        k_shot: int
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.languages = languages
        
        # Create datasets for support and query sets
        self.support = FewTopNERDataset(
            support_set['ner'],
            support_set['topic'],
            languages,
            is_few_shot=True
        )
        
        self.query = FewTopNERDataset(
            query_set['ner'],
            query_set['topic'],
            languages,
            is_few_shot=True
        )
        
        # Store prototypes for each entity type
        self.prototypes = {}
    
    def get_support(self) -> FewTopNERDataset:
        return self.support
    
    def get_query(self) -> FewTopNERDataset:
        return self.query
    
    def get_batch(self, batch_size: int = None) -> FewShotBatch:
        """Get a balanced batch from support and query sets"""
        if batch_size is None:
            batch_size = self.n_way * self.k_shot
        
        # Sample from support set
        support_indices = np.random.choice(
            len(self.support),
            size=batch_size,
            replace=False
        )
        
        # Sample from query set
        query_indices = np.random.choice(
            len(self.query),
            size=batch_size,
            replace=False
        )
        
        # Get batches
        support_batch = [self.support[i] for i in support_indices]
        query_batch = [self.query[i] for i in query_indices]
        
        # Collate batches
        support_data = self.support.collate_fn(support_batch)
        query_data = self.query.collate_fn(query_batch)
        
        return FewShotBatch(
            support_ner=support_data,
            support_topic=support_data,
            query_ner=query_data,
            query_topic=query_data,
            languages=self.languages
        )

def create_episodes(
    ner_data: Dict[str, Dict],
    topic_data: Dict[str, Dict],
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int
) -> List[FewShotEpisode]:
    """Create few-shot episodes from NER and Topic data"""
    episodes = []
    
    for _ in range(n_episodes):
        # Sample languages
        languages = np.random.choice(
            list(ner_data.keys()),
            size=min(3, len(ner_data)),
            replace=False
        )
        
        # Create support and query sets
        support_set = {'ner': {}, 'topic': {}}
        query_set = {'ner': {}, 'topic': {}}
        
        for lang in languages:
            # Sample examples for this language
            n_examples = n_way * k_shot
            ner_indices = np.random.choice(
                len(ner_data[lang]),
                size=n_examples + n_query,
                replace=False
            )
            
            # Split into support and query
            support_set['ner'][lang] = [ner_data[lang][i] for i in ner_indices[:n_examples]]
            query_set['ner'][lang] = [ner_data[lang][i] for i in ner_indices[n_examples:]]
            
            # Match with topic features
            support_set['topic'][lang] = [topic_data[lang][i] for i in ner_indices[:n_examples]]
            query_set['topic'][lang] = [topic_data[lang][i] for i in ner_indices[n_examples:]]
        
        # Create episode
        episode = FewShotEpisode(
            support_set=support_set,
            query_set=query_set,
            languages=languages,
            n_way=n_way,
            k_shot=k_shot
        )
        
        episodes.append(episode)
    
    return episodes