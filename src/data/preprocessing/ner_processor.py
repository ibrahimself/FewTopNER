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

class WikiNeuralProcessor:
    """Processor for WikiNEuRal NER dataset"""
    
    def __init__(self, config):
        self.config = config
        self.languages = config.model.languages
        self.label_map = {
            'O': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6,
            'B-MISC': 7, 'I-MISC': 8
        }

    def process_all_languages(self) -> Dict[str, Dict[str, List]]:
        """
        Process WikiNEuRal data for all languages
        
        Returns:
            Dictionary of processed data by language and split
        """
        datasets = {}
        
        for lang in self.languages:
            logger.info(f"Processing WikiNEuRal data for {lang}")
            try:
                lang_datasets = self.process_language(lang)
                if lang_datasets:
                    datasets[lang] = lang_datasets
            except Exception as e:
                logger.error(f"Error processing language {lang}: {e}")
                continue
        
        return datasets

    def process_language(self, language: str) -> Dict[str, List]:
        """Process data for a specific language"""
        processed_data = {}
        base_path = Path(self.config.preprocessing.wikineural_path) / language
        
        for split in ['train', 'dev', 'test']:
            file_path = base_path / f"{split}.conllu"
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                examples = self.read_conll_file(file_path, language)
                processed_data[split] = examples
                logger.info(f"Processed {len(examples)} examples for {language} {split}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return processed_data if processed_data else None

    def read_conll_file(self, file_path: Path, language: str) -> List[Dict]:
        """Read and process CoNLL format file"""
        examples = []
        current_words = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('-DOCSTART-') or not line:
                    if current_words:
                        example = self.create_example(
                            current_words,
                            current_labels,
                            language
                        )
                        if example:
                            examples.append(example)
                        current_words = []
                        current_labels = []
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    current_words.append(parts[0])
                    current_labels.append(parts[-1])
            
            # Handle last example
            if current_words:
                example = self.create_example(
                    current_words,
                    current_labels,
                    language
                )
                if example:
                    examples.append(example)
        
        return examples

    def create_example(
        self,
        words: List[str],
        labels: List[str],
        language: str
    ) -> Optional[Dict]:
        """Create a single example from words and labels"""
        try:
            # Convert labels to IDs
            label_ids = [self.label_map.get(label, self.label_map['O']) 
                        for label in labels]
            
            return {
                'words': words,
                'labels': labels,
                'label_ids': label_ids,
                'language': language,
                'length': len(words)
            }
        except Exception as e:
            logger.error(f"Error creating example: {e}")
            return None

    def get_statistics(self, datasets: Dict[str, Dict[str, List]]) -> Dict:
        """Compute dataset statistics"""
        stats = {}
        
        for lang, lang_data in datasets.items():
            lang_stats = {
                'total_examples': 0,
                'total_tokens': 0,
                'entity_counts': defaultdict(int)
            }
            
            for split, examples in lang_data.items():
                split_stats = self.compute_split_statistics(examples)
                lang_stats['total_examples'] += split_stats['num_examples']
                lang_stats['total_tokens'] += split_stats['num_tokens']
                
                for entity_type, count in split_stats['entity_counts'].items():
                    lang_stats['entity_counts'][entity_type] += count
            
            stats[lang] = lang_stats
        
        return stats

    def compute_split_statistics(self, examples: List[Dict]) -> Dict:
        """Compute statistics for a data split"""
        stats = {
            'num_examples': len(examples),
            'num_tokens': sum(ex['length'] for ex in examples),
            'entity_counts': defaultdict(int)
        }
        
        for example in examples:
            for label in example['labels']:
                if label != 'O':
                    entity_type = label[2:]  # Remove B- or I- prefix
                    stats['entity_counts'][entity_type] += 1
        
        return stats

    def verify_data(self, datasets: Dict[str, Dict[str, List]]) -> bool:
        """Verify processed data"""
        try:
            for lang, lang_data in datasets.items():
                for split, examples in lang_data.items():
                    # Check for non-empty examples
                    if not examples:
                        logger.error(f"No examples found for {lang} {split}")
                        return False
                    
                    # Verify example format
                    for ex in examples:
                        if not all(k in ex for k in ['words', 'labels', 'label_ids', 'language']):
                            logger.error(f"Invalid example format in {lang} {split}")
                            return False
                        
                        # Verify label consistency
                        if len(ex['words']) != len(ex['labels']):
                            logger.error(f"Mismatched words and labels in {lang} {split}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying data: {e}")
            return False

class WikiNeuralProcessor1:
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
