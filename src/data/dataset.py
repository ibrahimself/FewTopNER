import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np

class FewTopNERDataset(Dataset):
    """
    Dataset class that combines NER and Topic data for FewTopNER.
    Handles both regular batches and few-shot episodes.
    """
    def __init__(
        self, 
        ner_features: List[Dict],
        topic_features: List[Dict],
        language: str,
        is_few_shot: bool = False
    ):
        self.ner_features = ner_features
        self.topic_features = topic_features
        self.language = language
        self.is_few_shot = is_few_shot
        
        # Verify data alignment
        assert len(ner_features) == len(topic_features), \
            f"Mismatched features: NER={len(ner_features)}, Topic={len(topic_features)}"

    def __len__(self) -> int:
        return len(self.ner_features)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing both NER and Topic features
        """
        item = {
            # NER features
            'input_ids': torch.tensor(self.ner_features[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.ner_features[idx]['attention_mask'], dtype=torch.long),
            'label_ids': torch.tensor(self.ner_features[idx]['label_ids'], dtype=torch.long),
            
            # Topic features
            'topic_vector': torch.tensor(self.topic_features[idx]['topic_vector'], dtype=torch.float),
            'topic_label': torch.tensor(self.topic_features[idx]['topic_label'], dtype=torch.long),
            
            # Metadata
            'language': self.language
        }
        
        return item

class FewShotEpisode:
    """
    Represents a few-shot episode containing support and query sets.
    """
    def __init__(
        self,
        support_ner: List[Dict],
        support_topic: List[Dict],
        query_ner: List[Dict],
        query_topic: List[Dict],
        language: str
    ):
        self.support = FewTopNERDataset(support_ner, support_topic, language, is_few_shot=True)
        self.query = FewTopNERDataset(query_ner, query_topic, language, is_few_shot=True)
        self.language = language
        
    def get_support(self) -> FewTopNERDataset:
        return self.support
        
    def get_query(self) -> FewTopNERDataset:
        return self.query
