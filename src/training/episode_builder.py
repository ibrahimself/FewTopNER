import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import logging
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class EpisodeBuilder:
    """Episode builder aligned with FewTopNER trainer"""
    
    def __init__(self, config):
        self.config = config
        
        # Episode settings
        self.n_way = config.n_way
        self.k_shot = config.k_shot
        self.n_query = config.n_query
        self.min_examples = config.min_examples_per_class
        self.batch_size = config.batch_size
        
        # Language settings
        self.languages = ['en', 'fr', 'de', 'es', 'it']
        self.num_languages_per_episode = config.num_languages_per_episode
        
        # NER label mapping from WikiNEuRal
        self.ner_labels = {
            'O': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6,
            'B-MISC': 7, 'I-MISC': 8
        }
    
    def create_episode(
        self,
        datasets: Dict[str, Dataset]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create a single episode with support and query loaders
        
        Args:
            datasets: Dictionary of datasets by language
        Returns:
            Tuple of (support_loader, query_loader)
        """
        # Select languages for this episode
        episode_langs = random.sample(
            self.languages,
            min(self.num_languages_per_episode, len(self.languages))
        )
        
        # Collect examples for support and query sets
        support_examples = []
        query_examples = []
        
        for lang_idx, lang in enumerate(episode_langs):
            dataset = datasets[lang]
            lang_support, lang_query = self._sample_language_examples(
                dataset,
                lang_idx
            )
            support_examples.extend(lang_support)
            query_examples.extend(lang_query)
        
        # Create loaders
        support_loader = DataLoader(
            support_examples,
            batch_size=self.config.support_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        query_loader = DataLoader(
            query_examples,
            batch_size=self.config.query_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        return support_loader, query_loader
    
    def _sample_language_examples(
        self,
        dataset: Dataset,
        lang_idx: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Sample examples for one language"""
        # Group by entity types
        entity_examples = defaultdict(list)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            labels = example['entity_labels']
            
            # Get unique entity types
            unique_entities = set(
                label.item() for label in labels
                if label != -100 and label != self.ner_labels['O']
            )
            
            # Add to all relevant entity types
            for entity_type in unique_entities:
                entity_examples[entity_type].append(example)
        
        # Sample for support and query sets
        support_examples = []
        query_examples = []
        
        # Select entity types
        available_types = [
            ent_type for ent_type, examples in entity_examples.items()
            if len(examples) >= self.k_shot + self.n_query
        ]
        
        if len(available_types) >= self.n_way:
            selected_types = random.sample(available_types, self.n_way)
            
            for entity_type in selected_types:
                examples = entity_examples[entity_type]
                selected = random.sample(examples, self.k_shot + self.n_query)
                
                # Add language ID to examples
                for example in selected:
                    example['language_id'] = lang_idx
                
                support_examples.extend(selected[:self.k_shot])
                query_examples.extend(selected[self.k_shot:])
        
        return support_examples, query_examples
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples"""
        # Initialize batch dictionary
        batch_dict = {
            'input_ids': [],
            'attention_mask': [],
            'entity_labels': [],
            'topic_labels': [],
            'language_ids': [],
            'texts': []
        }
        
        # Collect tensors
        for example in batch:
            for key in batch_dict:
                if key == 'texts':
                    batch_dict[key].append(example.get('text', ''))
                elif key == 'language_ids':
                    batch_dict[key].append(example['language_id'])
                else:
                    batch_dict[key].append(example[key])
        
        # Convert to tensors
        for key in batch_dict:
            if key not in ['texts', 'language_ids']:
                batch_dict[key] = torch.stack(batch_dict[key])
            elif key == 'language_ids':
                batch_dict[key] = torch.tensor(batch_dict[key])
        
        return batch_dict
    
    def create_episode_loader(
        self,
        datasets: Dict[str, Dataset],
        num_episodes: int,
        infinite: bool = True
    ):
        """
        Create an episode loader compatible with the trainer
        
        Args:
            datasets: Dictionary of datasets by language
            num_episodes: Number of episodes to create
            infinite: Whether to loop infinitely
        """
        while True:
            for _ in range(num_episodes):
                support_loader, query_loader = self.create_episode(datasets)
                
                # Sample a batch for the bridge
                bridge_batch = next(iter(support_loader))
                
                yield bridge_batch, support_loader, query_loader
            
            if not infinite:
                break

    def get_dataloader(
        self,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor],
        support_batch_size: int,
        query_batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create dataloaders from support and query sets
        
        Args:
            support_set: Support set tensors
            query_set: Query set tensors
            support_batch_size: Batch size for support set
            query_batch_size: Batch size for query set
        Returns:
            Support and query dataloaders
        """
        # Create tensor datasets
        support_dataset = TensorDataset(
            support_set['input_ids'],
            support_set['attention_mask'],
            support_set['entity_labels'],
            support_set['topic_labels'],
            support_set['language_ids']
        )
        
        query_dataset = TensorDataset(
            query_set['input_ids'],
            query_set['attention_mask'],
            query_set['entity_labels'],
            query_set['topic_labels'],
            query_set['language_ids']
        )
        
        # Create dataloaders
        support_loader = DataLoader(
            support_dataset,
            batch_size=support_batch_size,
            shuffle=True
        )
        
        query_loader = DataLoader(
            query_dataset,
            batch_size=query_batch_size,
            shuffle=True
        )
        
        return support_loader, query_loader