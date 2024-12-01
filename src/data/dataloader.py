import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Iterator, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .dataset import FewTopNERDataset, FewShotEpisode, FewShotBatch, create_episodes
from .preprocessing.ner_processor import WikiNeuralProcessor
from .preprocessing.topic_processor import WikiTopicProcessor

logger = logging.getLogger(__name__)

class FewTopNERDataLoader:
    """DataLoader for FewTopNER combining NER and Topic data"""
    
    def __init__(
        self,
        ner_processor: WikiNeuralProcessor,
        wiki_topic_processor: WikiTopicProcessor,
        config
    ):
        self.ner_processor = ner_processor
        self.topic_processor = wiki_topic_processor
        self.config = config
        
        # Cache for processed data
        self.processed_data = {
            'ner': {},
            'topic': {},
            'combined': {}
        }
        
        # Load initial data
        self.datasets = self.load_all_languages(['train', 'val', 'test'])

    def load_all_languages(
        self,
        splits: List[str] = ['train', 'val', 'test']
    ) -> Dict[str, Dict[str, FewTopNERDataset]]:
        """Load and process all language data"""
        datasets = {}
        
        # Process NER data
        logger.info("Processing WikiNEuRal NER data...")
        ner_data = self.ner_processor.process_all_languages()
        
        # Process Topic data
        logger.info("Processing Wikipedia topic data...")
        topic_data = self.topic_processor.process_all_languages()
        
        # Combine for each language
        for lang in self.config.model.languages:
            if lang not in ner_data or lang not in topic_data:
                logger.warning(f"Skipping {lang}: missing data")
                continue
            
            datasets[lang] = {}
            for split in splits:
                if split in ner_data[lang]:
                    try:
                        dataset = self._create_combined_dataset(
                            ner_data[lang][split],
                            topic_data[lang],
                            lang,
                            split
                        )
                        datasets[lang][split] = dataset
                        
                        # Cache data
                        self.processed_data['ner'][(lang, split)] = ner_data[lang][split]
                        self.processed_data['topic'][lang] = topic_data[lang]
                        self.processed_data['combined'][(lang, split)] = dataset
                        
                        logger.info(f"Created dataset for {lang} {split}: {len(dataset)} examples")
                    except Exception as e:
                        logger.error(f"Error creating dataset for {lang} {split}: {e}")
                        continue
        
        return datasets

    def _create_combined_dataset(
        self,
        ner_data: Dict,
        topic_model: Tuple,
        language: str,
        split: str
    ) -> FewTopNERDataset:
        """Create combined dataset from NER and Topic data"""
        lda_model, dictionary = topic_model
        
        # Get texts from NER data
        texts = [
            ' '.join(feature.original_tokens)
            for feature in ner_data.features
            if hasattr(feature, 'original_tokens')
        ]
        
        # Create topic features
        topic_features = []
        for text in tqdm(texts, desc=f"Creating topic features for {language} {split}"):
            topic_vector = self.topic_processor.create_topic_features(
                text,
                language,
                lda_model,
                dictionary
            )
            if topic_vector is not None:
                topic_features.append({
                    'features': topic_vector.numpy(),
                    'language': language
                })
        
        return FewTopNERDataset(
            ner_features=ner_data.features,
            topic_features=topic_features,
            languages=language,
            is_few_shot=(split == 'train'),
            max_length=self.config.model.max_length
        )

    def create_few_shot_episodes(
        self,
        datasets: Dict[str, Dict[str, FewTopNERDataset]],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ) -> List[FewShotEpisode]:
        """Create few-shot episodes"""
        train_data = {
            lang: {
                'ner': datasets[lang]['train'].ner_features,
                'topic': datasets[lang]['train'].topic_features
            }
            for lang in datasets
            if 'train' in datasets[lang]
        }
        
        return create_episodes(
            ner_data=train_data,
            topic_data=train_data,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
            n_episodes=n_episodes
        )

    def get_dataloader(
        self,
        dataset: FewTopNERDataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader for dataset"""
        if batch_size is None:
            batch_size = self.config.data.train_batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
    
class FewShotEpisodeLoader:
    """Loader for few-shot episodes"""
    
    def __init__(
        self,
        episodes: List[FewShotEpisode],
        config,
        infinite: bool = False
    ):
        self.episodes = episodes
        self.config = config
        self.infinite = infinite
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __iter__(self) -> Iterator[Tuple[FewShotBatch, DataLoader, DataLoader]]:
        """Iterate over episodes"""
        while True:
            for episode in self.episodes:
                # Get balanced batch
                batch = episode.get_batch(
                    batch_size=self.config.data.episode_batch_size
                )
                
                # Create support loader
                support_loader = DataLoader(
                    episode.get_support(),
                    batch_size=self.config.data.support_batch_size,
                    shuffle=True,
                    collate_fn=episode.get_support().collate_fn,
                    num_workers=self.config.data.num_workers,
                    pin_memory=True
                )
                
                # Create query loader
                query_loader = DataLoader(
                    episode.get_query(),
                    batch_size=self.config.data.query_batch_size,
                    shuffle=False,
                    collate_fn=episode.get_query().collate_fn,
                    num_workers=self.config.data.num_workers,
                    pin_memory=True
                )
                
                yield batch, support_loader, query_loader
            
            if not self.infinite:
                break

def create_dataloaders(
    config,
    splits: List[str] = ['train', 'val', 'test']
) -> Tuple[Dict[str, Dict[str, DataLoader]], FewShotEpisodeLoader]:
    """Create all necessary dataloaders"""
    try:
        # Initialize processors
        logger.info("Initializing processors...")
        wikineural_processor = WikiNeuralProcessor(config)
        wiki_topic_processor = WikiTopicProcessor(config)
        
        # Create main dataloader
        logger.info("Creating main dataloader...")
        loader = FewTopNERDataLoader(
            ner_processor=wikineural_processor,
            wiki_topic_processor=wiki_topic_processor,
            config=config
        )
        
        # Create regular dataloaders
        logger.info("Creating regular dataloaders...")
        dataloaders = {
            split: {
                lang: loader.get_dataloader(
                    loader.datasets[lang][split],
                    batch_size=config.data.train_batch_size if split == 'train' 
                             else config.data.eval_batch_size,
                    shuffle=(split == 'train')
                )
                for lang in loader.datasets
                if split in loader.datasets[lang]
            }
            for split in splits
        }
        
        # Create few-shot episodes
        logger.info("Creating few-shot episodes...")
        episodes = loader.create_few_shot_episodes(
            loader.datasets,
            n_way=config.data.n_way,
            k_shot=config.data.k_shot,
            n_query=config.data.n_query,
            n_episodes=config.training.episodes_per_epoch
        )
        
        # Create episode loader
        logger.info("Creating episode loader...")
        episode_loader = FewShotEpisodeLoader(
            episodes=episodes,
            config=config,
            infinite=True
        )
        
        logger.info("Successfully created all dataloaders")
        return dataloaders, episode_loader
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        raise

class DataLoaderManager:
    """Manager class for handling all dataloaders"""
    
    def __init__(self, config):
        self.config = config
        self.dataloaders = None
        self.episode_loader = None
        
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Initialize all dataloaders"""
        self.dataloaders, self.episode_loader = create_dataloaders(self.config)
    
    def get_train_loader(self, language: str) -> Optional[DataLoader]:
        """Get training dataloader for specific language"""
        return self.dataloaders.get('train', {}).get(language)
    
    def get_val_loader(self, language: str) -> Optional[DataLoader]:
        """Get validation dataloader for specific language"""
        return self.dataloaders.get('val', {}).get(language)
    
    def get_test_loader(self, language: str) -> Optional[DataLoader]:
        """Get test dataloader for specific language"""
        return self.dataloaders.get('test', {}).get(language)
    
    def get_episode_loader(self) -> FewShotEpisodeLoader:
        """Get few-shot episode loader"""
        return self.episode_loader
    
    def cleanup(self):
        """Cleanup resources"""
        self.dataloaders = None
        self.episode_loader = None
        torch.cuda.empty_cache()