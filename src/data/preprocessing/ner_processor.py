import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from transformers import XLMRobertaTokenizer
from dataclasses import dataclass
import logging
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WikiNeuRalExample:
    """Single example from WikiNEuRal dataset"""
    guid: str
    tokens: List[str]
    ner_tags: List[str]
    language: str
    
@dataclass
class WikiNeuRalFeatures:
    """Features for WikiNEuRal dataset with original tags preserved"""
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: List[int]
    language_id: int
    original_tokens: List[str]  # For debugging and analysis
    original_tags: List[str]    # For debugging and analysis

class WikiNeuRalDataset(Dataset):
    """Dataset class for WikiNEuRal"""
    def __init__(self, features, label_map: Dict[str, int], language_map: Dict[str, int]):
        self.features = features
        self.label_map = label_map
        self.language_map = language_map
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            'input_ids': torch.tensor(feature.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(feature.attention_mask, dtype=torch.long),
            'label_ids': torch.tensor(feature.label_ids, dtype=torch.long),
            'language_id': torch.tensor(feature.language_id, dtype=torch.long)
        }

class WikiNeurxalProcessor:
    """Processor for WikiNEuRal dataset following the official specifications"""
    
    # Official label mapping from the dataset
    LABEL_MAP = {
        'O': 0, 
        'B-PER': 1, 'I-PER': 2, 
        'B-ORG': 3, 'I-ORG': 4, 
        'B-LOC': 5, 'I-LOC': 6, 
        'B-MISC': 7, 'I-MISC': 8
    }
    
    # We'll use only these 5 languages as specified in your project
    SUPPORTED_LANGUAGES = ['en', 'fr', 'de', 'es', 'it']
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.language_map = {lang: idx for idx, lang in enumerate(self.SUPPORTED_LANGUAGES)}
        self.logger = logging.getLogger(__name__)
        
        # Dataset statistics for monitoring
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def read_wikineural_file(self, file_path: str, language: str) -> List[WikiNeuRalExample]:
        """Read WikiNEuRal data file"""
        examples = []
        current_tokens = []
        current_tags = []
        sentence_idx = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line or line.startswith('-DOCSTART-'):
                        if current_tokens:
                            examples.append(WikiNeuRalExample(
                                guid=f"{language}-{sentence_idx}",
                                tokens=current_tokens,
                                ner_tags=current_tags,
                                language=language
                            ))
                            # Update statistics
                            self._update_stats(language, current_tags)
                            sentence_idx += 1
                            current_tokens = []
                            current_tags = []
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:  # Ensure we have both token and tag
                        token, tag = parts[0], parts[-1]
                        current_tokens.append(token)
                        current_tags.append(tag)
                
                # Add the last example if exists
                if current_tokens:
                    examples.append(WikiNeuRalExample(
                        guid=f"{language}-{sentence_idx}",
                        tokens=current_tokens,
                        ner_tags=current_tags,
                        language=language
                    ))
                    self._update_stats(language, current_tags)
                    
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise
            
        return examples
    
    def _update_stats(self, language: str, tags: List[str]):
        """Update dataset statistics"""
        for tag in tags:
            if tag != 'O':
                entity_type = tag[2:]  # Remove B- or I- prefix
                self.stats[language][entity_type] += 1
            else:
                self.stats[language]['OTHER'] += 1
    
    def convert_examples_to_features(
        self,
        examples: List[WikiNeuRalExample],
        max_length: int = 128
    ) -> List[WikiNeuRalFeatures]:
        """Convert examples to features with proper BIO tag handling"""
        
        features = []
        for example in tqdm(examples, desc="Converting examples to features"):
            tokens = []
            label_ids = []
            
            for token, tag in zip(example.tokens, example.ner_tags):
                # Tokenize word into WordPiece tokens
                word_tokens = self.tokenizer.tokenize(token)
                if not word_tokens:
                    word_tokens = [self.tokenizer.unk_token]
                
                tokens.extend(word_tokens)
                
                # Assign the label to the first token, -100 to others
                label_ids.append(self.LABEL_MAP[tag])
                label_ids.extend([-100] * (len(word_tokens) - 1))
            
            # Truncate if needed
            if len(tokens) > max_length - 2:  # Account for [CLS] and [SEP]
                tokens = tokens[:max_length - 2]
                label_ids = label_ids[:max_length - 2]
            
            # Add special tokens
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            label_ids = [-100] + label_ids + [-100]
            
            # Convert tokens to ids
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [-100] * padding_length
            
            features.append(WikiNeuRalFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_ids=label_ids,
                language_id=self.language_map[example.language],
                original_tokens=example.tokens,
                original_tags=example.ner_tags
            ))
        
        return features

    def create_few_shot_episodes(
        self,
        datasets: Dict[str, Dict[str, WikiNeuRalDataset]],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int,
    ) -> List[Dict]:
        """Create few-shot episodes ensuring entity type balance"""
        
        def get_entity_examples(dataset: WikiNeuRalDataset) -> Dict[str, List[WikiNeuRalFeatures]]:
            """Group examples by entity type"""
            entity_examples = defaultdict(list)
            entity_types = ['PER', 'ORG', 'LOC', 'MISC']
            
            for feature in dataset.features:
                # Track which entity types appear in this example
                example_entities = set()
                for label_id in feature.label_ids:
                    if label_id != -100 and label_id != self.LABEL_MAP['O']:
                        tag = list(self.LABEL_MAP.keys())[list(self.LABEL_MAP.values()).index(label_id)]
                        if tag.startswith('B-'):
                            entity_type = tag[2:]
                            example_entities.add(entity_type)
                
                # Add example to each entity type it contains
                for entity_type in example_entities:
                    entity_examples[entity_type].append(feature)
            
            return entity_examples
        
        episodes = []
        for lang in self.SUPPORTED_LANGUAGES:
            logger.info(f"\nDataset statistics for {lang}:")
            for entity_type, count in self.stats[lang].items():
                logger.info(f"{entity_type}: {count}")
        
        # Get entity examples for each language's training set
        all_entity_examples = {
            lang: get_entity_examples(data['train'])
            for lang, data in datasets.items()
        }
        
        # Create episodes
        for episode_idx in tqdm(range(n_episodes), desc="Creating episodes"):
            # Sample languages for this episode
            episode_langs = random.sample(self.SUPPORTED_LANGUAGES, 
                                       min(len(self.SUPPORTED_LANGUAGES), 3))
            
            # Get common entity types across selected languages
            common_entities = set.intersection(*[
                set(all_entity_examples[lang].keys())
                for lang in episode_langs
            ])
            
            if len(common_entities) < n_way:
                continue
            
            # Sample entity types for this episode
            episode_entities = random.sample(list(common_entities), n_way)
            
            support_set = []
            query_set = []
            
            # Sample examples for each entity type from each language
            for entity_type in episode_entities:
                for lang in episode_langs:
                    examples = all_entity_examples[lang][entity_type]
                    if len(examples) < k_shot + n_query:
                        continue
                    
                    # Ensure no overlap between support and query sets
                    sampled_examples = random.sample(examples, k_shot + n_query)
                    support_set.extend(sampled_examples[:k_shot])
                    query_set.extend(sampled_examples[k_shot:])
            
            if len(support_set) >= n_way * k_shot and len(query_set) >= n_way:
                episodes.append({
                    'support': WikiNeuRalDataset(support_set, self.LABEL_MAP, self.language_map),
                    'query': WikiNeuRalDataset(query_set, self.LABEL_MAP, self.language_map)
                })
        
        return episodes

    def save_statistics(self, output_path: str):
        """Save dataset statistics to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
