import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import logging
from torch.utils.data import Dataset

class EpisodeBuilder:
    """
    Builds few-shot episodes for training FewTopNER
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Episode settings
        self.n_way = config.n_way
        self.k_shot = config.k_shot
        self.n_query = config.n_query
        self.min_examples_per_class = config.min_examples_per_class
        
    def create_entity_episode(
        self,
        dataset: Dataset,
        entity_types: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Create episode for entity recognition
        
        Args:
            dataset: Source dataset
            entity_types: Optional list of entity types to include
        Returns:
            Tuple of (support set, query set)
        """
        # Group examples by entity type
        entity_examples = defaultdict(list)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            labels = example['entity_labels']
            
            # Get unique entity types in this example
            unique_entities = set(
                label.item() for label in labels
                if label != -100 and label != self.config.o_label
            )
            
            # Add example to all its entity types
            for entity_type in unique_entities:
                if entity_types is None or entity_type in entity_types:
                    entity_examples[entity_type].append(idx)
                    
        # Select entity types for this episode
        available_types = [
            ent_type for ent_type, examples in entity_examples.items()
            if len(examples) >= self.min_examples_per_class
        ]
        
        if len(available_types) < self.n_way:
            self.logger.warning(
                f"Only {len(available_types)} entity types available for episode"
            )
            return None, None
            
        selected_types = random.sample(available_types, self.n_way)
        
        # Build support and query sets
        support_indices = []
        query_indices = []
        
        for entity_type in selected_types:
            examples = entity_examples[entity_type]
            selected = random.sample(examples, self.k_shot + self.n_query)
            
            support_indices.extend(selected[:self.k_shot])
            query_indices.extend(selected[self.k_shot:])
            
        return self._create_sets(dataset, support_indices, query_indices)
        
    def create_topic_episode(
        self,
        dataset: Dataset,
        topics: Optional[List[int]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Create episode for topic modeling
        
        Args:
            dataset: Source dataset
            topics: Optional list of topics to include
        Returns:
            Tuple of (support set, query set)
        """
        # Group examples by topic
        topic_examples = defaultdict(list)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            topic = example['topic_label'].item()
            
            if topics is None or topic in topics:
                topic_examples[topic].append(idx)
                
        # Select topics for this episode
        available_topics = [
            topic for topic, examples in topic_examples.items()
            if len(examples) >= self.min_examples_per_class
        ]
        
        if len(available_topics) < self.n_way:
            self.logger.warning(
                f"Only {len(available_topics)} topics available for episode"
            )
            return None, None
            
        selected_topics = random.sample(available_topics, self.n_way)
        
        # Build support and query sets
        support_indices = []
        query_indices = []
        
        for topic in selected_topics:
            examples = topic_examples[topic]
            selected = random.sample(examples, self.k_shot + self.n_query)
            
            support_indices.extend(selected[:self.k_shot])
            query_indices.extend(selected[self.k_shot:])
            
        return self._create_sets(dataset, support_indices, query_indices)
        
    def create_joint_episode(
        self,
        dataset: Dataset,
        balance_tasks: bool = True
    ) -> Tuple[Dict, Dict]:
        """
        Create episode balancing both entity and topic tasks
        
        Args:
            dataset: Source dataset
            balance_tasks: Whether to ensure balance between tasks
        Returns:
            Tuple of (support set, query set)
        """
        # Create separate episodes for each task
        entity_support, entity_query = self.create_entity_episode(dataset)
        topic_support, topic_query = self.create_topic_episode(dataset)
        
        if entity_support is None or topic_support is None:
            return None, None
            
        if balance_tasks:
            # Ensure equal representation of both tasks
            min_support = min(len(entity_support['input_ids']), 
                            len(topic_support['input_ids']))
            min_query = min(len(entity_query['input_ids']), 
                          len(topic_query['input_ids']))
            
            # Randomly select examples to maintain balance
            support_indices = random.sample(range(len(entity_support['input_ids'])), 
                                         min_support)
            query_indices = random.sample(range(len(entity_query['input_ids'])), 
                                       min_query)
            
            entity_support = self._select_indices(entity_support, support_indices)
            entity_query = self._select_indices(entity_query, query_indices)
            
            support_indices = random.sample(range(len(topic_support['input_ids'])), 
                                         min_support)
            query_indices = random.sample(range(len(topic_query['input_ids'])), 
                                       min_query)
            
            topic_support = self._select_indices(topic_support, support_indices)
            topic_query = self._select_indices(topic_query, query_indices)
            
        # Combine episodes
        support_set = self._merge_sets(entity_support, topic_support)
        query_set = self._merge_sets(entity_query, topic_query)
        
        return support_set, query_set
        
    def create_episodes(
        self,
        dataset: Dataset,
        num_episodes: int,
        joint: bool = True
    ) -> List[Tuple[Dict, Dict]]:
        """
        Create multiple episodes for training
        
        Args:
            dataset: Source dataset
            num_episodes: Number of episodes to create
            joint: Whether to create joint episodes
        Returns:
            List of (support set, query set) pairs
        """
        episodes = []
        
        for _ in range(num_episodes):
            if joint:
                support_set, query_set = self.create_joint_episode(dataset)
            else:
                # Alternate between entity and topic episodes
                if len(episodes) % 2 == 0:
                    support_set, query_set = self.create_entity_episode(dataset)
                else:
                    support_set, query_set = self.create_topic_episode(dataset)
                    
            if support_set is not None and query_set is not None:
                episodes.append((support_set, query_set))
                
        return episodes
        
    def _create_sets(
        self,
        dataset: Dataset,
        support_indices: List[int],
        query_indices: List[int]
    ) -> Tuple[Dict, Dict]:
        """
        Create support and query sets from indices
        """
        support_set = {
            'input_ids': [],
            'attention_mask': [],
            'entity_labels': [],
            'topic_labels': [],
            'language': []
        }
        
        query_set = {
            'input_ids': [],
            'attention_mask': [],
            'entity_labels': [],
            'topic_labels': [],
            'language': []
        }
        
        # Collect support set examples
        for idx in support_indices:
            example = dataset[idx]
            for key in support_set:
                support_set[key].append(example[key])
                
        # Collect query set examples
        for idx in query_indices:
            example = dataset[idx]
            for key in query_set:
                query_set[key].append(example[key])
                
        # Convert lists to tensors
        support_set = {
            k: torch.stack(v) if k != 'language' else v
            for k, v in support_set.items()
        }
        
        query_set = {
            k: torch.stack(v) if k != 'language' else v
            for k, v in query_set.items()
        }
        
        return support_set, query_set
        
    def _select_indices(
        self,
        data_dict: Dict,
        indices: List[int]
    ) -> Dict:
        """
        Select specific indices from a dictionary of tensors
        """
        return {
            k: v[indices] if k != 'language' else [v[i] for i in indices]
            for k, v in data_dict.items()
        }
        
    def _merge_sets(
        self,
        set1: Dict,
        set2: Dict
    ) -> Dict:
        """
        Merge two sets of examples
        """
        merged = {}
        for key in set1:
            if key == 'language':
                merged[key] = set1[key] + set2[key]
            else:
                merged[key] = torch.cat([set1[key], set2[key]], dim=0)
        return merged