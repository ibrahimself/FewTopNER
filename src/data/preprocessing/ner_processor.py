import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from transformers import XLMRobertaTokenizer
from dataclasses import dataclass
import logging
import os

@dataclass
class InputExample:
    """Single example for NER"""
    guid: str
    words: List[str]
    labels: List[str]

@dataclass
class InputFeatures:
    """Features created from InputExample"""
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: List[int]

class NERDataset(Dataset):
    """Dataset for NER tasks"""
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            'input_ids': torch.tensor(feature.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(feature.attention_mask, dtype=torch.long),
            'label_ids': torch.tensor(feature.label_ids, dtype=torch.long)
        }

class NERProcessor:
    """Process NER data for multiple languages"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.label_map = self._create_label_map()
        self.logger = logging.getLogger(__name__)
        
    def _create_label_map(self) -> Dict[str, int]:
        """Create mapping from NER labels to IDs"""
        labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 
                 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        return {label: idx for idx, label in enumerate(labels)}

    def read_conll_data(self, file_path: str) -> List[InputExample]:
        """
        Read NER data in CoNLL format
        
        Args:
            file_path: Path to CoNLL file
        Returns:
            List of InputExample objects
        """
        examples = []
        words = []
        labels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                guid = 0
                for line in f:
                    if line.startswith('-DOCSTART-') or line.strip() == '':
                        if words:
                            examples.append(InputExample(
                                guid=f"example-{guid}",
                                words=words,
                                labels=labels
                            ))
                            guid += 1
                            words = []
                            labels = []
                        continue
                    
                    splits = line.strip().split()
                    if len(splits) >= 2:  # Ensure we have both word and label
                        words.append(splits[0])
                        labels.append(splits[-1])
                
                if words:  # Add the last example
                    examples.append(InputExample(
                        guid=f"example-{guid}",
                        words=words,
                        labels=labels
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
            
        return examples

    def convert_examples_to_features(
        self, 
        examples: List[InputExample]
    ) -> List[InputFeatures]:
        """
        Convert InputExamples to InputFeatures
        
        Args:
            examples: List of InputExample objects
        Returns:
            List of InputFeatures objects
        """
        features = []
        
        for example in examples:
            tokens = []
            label_ids = []
            
            for word, label in zip(example.words, example.labels):
                word_tokens = self.tokenizer.tokenize(word)
                
                # Handle empty tokens
                if not word_tokens:
                    word_tokens = [self.tokenizer.unk_token]
                    
                tokens.extend(word_tokens)
                
                # Assign the label to the first token, mark others as -100
                label_ids.extend([self.label_map[label]] + [-100] * (len(word_tokens) - 1))

            # Truncate if necessary
            if len(tokens) > self.config.max_seq_length - 2:
                tokens = tokens[:self.config.max_seq_length - 2]
                label_ids = label_ids[:self.config.max_seq_length - 2]

            # Add special tokens
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            label_ids = [-100] + label_ids + [-100]  # CLS and SEP tokens get -100

            # Convert tokens to ids and create attention mask
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            # Pad to max length
            padding_length = self.config.max_seq_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [-100] * padding_length

            features.append(InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_ids=label_ids
            ))

        return features

    def process_language(self, language: str) -> Dict[str, NERDataset]:
        """
        Process NER data for a specific language
        
        Args:
            language: Language code (e.g., 'en', 'fr')
        Returns:
            Dictionary containing train/val/test datasets
        """
        datasets = {}
        base_path = os.path.join(self.config.data_path, 'ner', language)
        
        for split in ['train', 'dev', 'test']:
            file_path = os.path.join(base_path, f"{split}.conll")
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                continue
                
            examples = self.read_conll_data(file_path)
            features = self.convert_examples_to_features(examples)
            datasets[split] = NERDataset(features)
            
            self.logger.info(f"Processed {len(examples)} examples for {language} {split}")
            
        return datasets

    def create_few_shot_episodes(
        self,
        dataset: NERDataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ) -> List[Dict]:
        """
        Create few-shot episodes from dataset
        
        Args:
            dataset: NERDataset object
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to create
        Returns:
            List of episodes, each containing support and query sets
        """
        episodes = []
        
        # Group features by label
        label_to_features = defaultdict(list)
        for feature in dataset.features:
            labels = [l for l in feature.label_ids if l != -100 and l != self.label_map['O']]
            for label in set(labels):
                label_to_features[label].append(feature)

        # Create episodes
        for _ in range(n_episodes):
            # Sample n_way classes
            available_labels = list(label_to_features.keys())
            if len(available_labels) < n_way:
                self.logger.warning(f"Not enough labels for {n_way}-way episodes")
                continue
                
            episode_labels = random.sample(available_labels, n_way)
            
            support_set = []
            query_set = []
            
            # Sample support and query examples for each label
            for label in episode_labels:
                label_features = label_to_features[label]
                if len(label_features) < k_shot + n_query:
                    continue
                    
                sampled_features = random.sample(label_features, k_shot + n_query)
                support_set.extend(sampled_features[:k_shot])
                query_set.extend(sampled_features[k_shot:])
            
            if support_set and query_set:
                episodes.append({
                    'support': NERDataset(support_set),
                    'query': NERDataset(query_set)
                })

        return episodes